"""Experiment-only transport and sandbox health guards. Frozen RISE files stay intact."""
from __future__ import annotations
import functools
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace


class SandboxPaused(RuntimeError):
    """Infrastructure stop; ordinary Exception so the frozen Agent saves its checkpoint."""


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class StopState:
    def __init__(self, pause_path: Path, events_path: Path):
        self.pause_path, self.events_path = Path(pause_path), Path(events_path)
        self._event = threading.Event(); self._lock = threading.Lock(); self.local = threading.local()

    @property
    def paused(self): return self._event.is_set()

    def raise_if_paused(self):
        if self.paused: raise SandboxPaused('Infrastructure pause; no new query/tool/model request is permitted')

    def pause(self, kind: str, details: dict):
        row = {'unix_time': time.time(), 'kind': kind, 'query_id': getattr(self.local, 'qid', None), **details}
        with self._lock:
            first = not self.paused
            self._event.set()
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
            if first: atomic_json(self.pause_path, {'status': 'infrastructure_paused', 'first_failure': row})
        return row


def is_docker_infrastructure_failure(returncode, stderr):
    text = str(stderr or '').lower()
    if returncode == 125: return True  # Docker itself could not run the container.
    markers = ('cannot connect to the docker daemon', 'error response from daemon',
               'permission denied while trying to connect to the docker daemon',
               'oci runtime create failed', 'failed to create task for container',
               'unable to find image', 'pull access denied', 'no such image',
               'docker: command not found')
    return bool(returncode) and any(marker in text for marker in markers)


def _exception_details(exc):
    cmd = getattr(exc, 'cmd', None)
    return {'error_type': type(exc).__name__, 'command': cmd if isinstance(cmd, (str, list, tuple)) else None,
            'returncode': getattr(exc, 'returncode', None),
            'stdout': str(getattr(exc, 'stdout', '') or '')[-65536:],
            'stderr': str(getattr(exc, 'stderr', '') or '')[-65536:],
            'stderr_retained': getattr(exc, 'stderr', None) is not None}


def instrument_docker(original, state):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        state.raise_if_paused()
        try:
            result = original(*args, **kwargs)
        except subprocess.CalledProcessError as exc:
            state.pause('docker_prepare_or_cleanup_failed', _exception_details(exc)); raise
        except subprocess.TimeoutExpired as exc:
            command = getattr(exc, 'cmd', ())
            # A user command may legitimately time out; preparation/cleanup may not.
            if isinstance(command, (list, tuple)) and len(command) > 1 and command[1] != 'run':
                state.pause('docker_prepare_or_cleanup_timeout', _exception_details(exc))
            raise
        except OSError as exc:
            state.pause('docker_spawn_failed', _exception_details(exc)); raise
        if is_docker_infrastructure_failure(result.returncode, result.stderr):
            state.pause('docker_runtime_start_failed', {'returncode': result.returncode,
                'stdout': str(result.stdout or '')[-65536:], 'stderr': str(result.stderr or '')[-65536:],
                'stderr_retained': True})
        return result
    return wrapped


class GuardedLocalClient:
    def __init__(self, client, model, base_url, log_path, state):
        self._client, self._model, self._base_url = client, model, base_url
        self._log_path, self._state = Path(log_path), state
        if str(client.base_url).rstrip('/') != base_url.rstrip('/'):
            raise RuntimeError('Solver endpoint differs from the immutable manifest')
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __getattr__(self, name): return getattr(self._client, name)

    def with_options(self, **kwargs):
        return type(self)(self._client.with_options(**kwargs), self._model, self._base_url,
                          self._log_path, self._state)

    def _create(self, *args, **kwargs):
        self._state.raise_if_paused()
        if kwargs.get('model') != self._model: raise RuntimeError('Only Qwen3.5-35B-A3B is authorized')
        started = time.monotonic()
        row = {'unix_time': time.time(), 'query_id': getattr(self._state.local, 'qid', None),
               'requested_model': self._model, 'requested_max_tokens': kwargs.get('max_tokens'),
               'transport': 'local-vLLM', 'status': 'error'}
        try:
            response = self._client.chat.completions.create(*args, **kwargs)
            row['reported_model'] = str(response.model)
            if row['reported_model'] != self._model:
                self._state.pause('solver_model_identity_mismatch', {'reported_model': row['reported_model']})
                raise SandboxPaused('Solver model identity differs from the frozen model')
            choice = response.choices[0] if response.choices else None
            row.update(status='ok', response_id=response.id, finish_reason=choice.finish_reason if choice else None,
                       tool_calls=len(choice.message.tool_calls or []) if choice else 0,
                       usage=response.usage.model_dump() if response.usage else None)
            return response
        except Exception as exc:
            row['error_type'] = type(exc).__name__; raise
        finally:
            row['request_seconds'] = time.monotonic() - started
            with self._state._lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def install_runtime_hooks(state, model, base_url, log_path):
    """Install before the frozen CLI imports its named functions."""
    import rise.agent as agent
    import rise.decompose as decompose
    import rise.rise_agent as rise_agent
    import rise.tools as tools
    original_client = decompose.make_client
    original_chat, original_run = agent.chat_with_retry, rise_agent.run_rise_agent
    original_bash = rise_agent.make_bash_tool
    tools._run_docker_bash = instrument_docker(tools._run_docker_bash, state)

    def client_factory(requested_model=None):
        state.raise_if_paused()
        if requested_model != model: raise RuntimeError('Frozen solver model mismatch')
        return GuardedLocalClient(original_client(requested_model), model, base_url, log_path, state)

    @functools.wraps(original_chat)
    def chat(*args, **kwargs):
        state.raise_if_paused()  # Before tokenization as well as generation.
        return original_chat(*args, **kwargs)

    @functools.wraps(original_bash)
    def bash_factory(*args, **kwargs):
        original = original_bash(*args, **kwargs)
        def bash(parameters):
            state.raise_if_paused()
            result = original(parameters)
            if isinstance(result, str):
                low = result.lower()
                if low.startswith(('error: failed to spawn bash:', 'error: bash subprocess exception:')):
                    if not state.paused: state.pause('sandbox_start_failed', {'public_error': result[-65536:]})
                if '[stderr]' in low and ('bwrap:' in low or is_docker_infrastructure_failure(1, result)):
                    if not state.paused: state.pause('sandbox_runtime_start_failed', {'public_error': result[-65536:]})
            return result  # Preserve the original tool return so Agent records this completed turn.
        return bash

    @functools.wraps(original_run)
    def run(*args, **kwargs):
        state.raise_if_paused()  # Blocks queued questions before tool/workspace initialization.
        state.local.qid = kwargs.get('query_id')
        try:
            value = original_run(*args, **kwargs)
            termination = str(getattr(value, 'terminated_by', '') or '')
            if termination.startswith('api_error') and not state.paused:
                # Retry exhaustion is a transport failure, not a scored answer.
                # Retain the Agent's existing checkpoint and stop queued work.
                state.pause('solver_api_failed', {'termination_kind': termination.split(':', 1)[0]})
            # Covers failure on the last allowed turn; keep cp before CLI writes a terminal row/unlinks cp.
            state.raise_if_paused()
            return value
        except OSError as exc:
            if exc.errno in {28, 122} and not state.paused:
                state.pause('filesystem_quota_or_full', {'errno': exc.errno})
            raise
        finally:
            state.local.qid = None

    decompose.make_client, agent.chat_with_retry = client_factory, chat
    rise_agent.make_bash_tool, rise_agent.run_rise_agent = bash_factory, run


def install_reranker_hooks(state):
    """Pause on native backend failure even when rerank returns a JSON error.

    Tool argument errors remain normal model-visible results. Hooks run inside
    each isolated cell worker, before any backend instance can be created.
    """
    from rise.reranker_backend import InputTooLongError, QwenReranker
    from rise.rerank_v13 import RerankService

    for name in ("get_tokenizer", "score"):
        original = getattr(QwenReranker, name)
        def guarded(self, *args, _original=original, _name=name, **kwargs):
            state.raise_if_paused()
            try:
                return _original(self, *args, **kwargs)
            except Exception as exc:
                if _name == "score" and isinstance(exc, (ValueError, InputTooLongError)):
                    raise
                if not state.paused:
                    state.pause("reranker_backend_failed", {"operation": _name,
                        "error_type": type(exc).__name__, "error": str(exc)[-4096:]})
                raise
        setattr(QwenReranker, name, guarded)

    append_diagnostic = RerankService._append_diagnostic
    def guarded_diagnostic(self, record):
        state.raise_if_paused()
        try:
            return append_diagnostic(self, record)
        except OSError as exc:
            if not state.paused:
                state.pause("reranker_diagnostic_storage_failed", {"errno": exc.errno,
                    "error_type": type(exc).__name__})
            raise
    RerankService._append_diagnostic = guarded_diagnostic
