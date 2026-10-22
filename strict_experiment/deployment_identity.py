"""Bind a local model endpoint to its live process and measured serving code.

Only public deployment metadata is collected. No process environment, private
credential, generated text or weight shard is read. The returned receipt must
be embedded in the new service configuration before the experiment freezes it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse
from urllib.request import Request, urlopen

VERSION = "strict-local-serving-identity-v1"
MODEL = "Qwen3.5-35B-A3B"
PROC = Path("/proc")
SECRET_FLAG = re.compile(r"^--(?:api[-_]?key|(?:hf-|access-|auth-)?token|password|secret|authorization)(?:=|$)", re.IGNORECASE)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError("Serving identity requires a regular file")
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _flag(command, name, default=None):
    values = []
    for i, arg in enumerate(command):
        if arg == name:
            if i + 1 == len(command):
                raise RuntimeError("Missing serving flag value: " + name)
            values.append(command[i + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) > 1:
        raise RuntimeError("Duplicate serving flag: " + name)
    return values[0] if values else default


def _configuration(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    command = value.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part or "\x00" in part for part in command):
        raise RuntimeError("Service configuration requires an explicit argv command")
    if any(SECRET_FLAG.search(part) for part in command if part.startswith("--")):
        raise RuntimeError("Credential-bearing serving commands cannot enter the public identity receipt")
    if value.get("model") != MODEL or _flag(command, "--served-model-name") != MODEL:
        raise RuntimeError("Service configuration model identity differs")
    try:
        serve = command.index("serve")
        model = command[serve + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Expected an explicit vllm serve model-directory command") from exc
    if model.startswith("-") or not Path(model).is_dir():
        raise RuntimeError("An existing local model directory is required")
    host = _flag(command, "--host", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        raise RuntimeError("Serving identity only supports the authorized local endpoint")
    port = _flag(command, "--port", "8000")
    if not str(port).isdigit() or not 1 <= int(port) <= 65535:
        raise RuntimeError("Invalid service port")
    parser = _flag(command, "--reasoning-parser-plugin")
    if not parser:
        raise RuntimeError("Explicit versioned reasoning parser plugin is required")
    return value, {"command": command, "model_root": str(Path(model).resolve(strict=True)),
        "base_url": f"http://127.0.0.1:{port}/v1", "parser_plugin": str(Path(parser).resolve(strict=True)),
        "reasoning_parser": _flag(command, "--reasoning-parser"),
        "tool_call_parser": _flag(command, "--tool-call-parser")}


def _process(pid):
    if type(pid) is not int or pid < 1:
        raise RuntimeError("A positive serving PID is required")
    root = PROC / str(pid)
    try:
        stat = (root / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            raise RuntimeError("The recorded serving process is a zombie")
        start = int(fields[19])
        raw = (root / "cmdline").read_bytes()
        command = [part.decode("utf-8", errors="strict") for part in raw.rstrip(b"\0").split(b"\0")]
        executable = str((root / "exe").resolve(strict=True))
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError("Cannot verify the live serving process") from exc
    if not command or not command[0]:
        raise RuntimeError("Live serving argv is empty")
    return {"pid": pid, "process_start_ticks": start, "actual_command": command, "executable": executable}


def _match_command(expected, process):
    actual = process["actual_command"]
    launcher = Path(expected[0]).resolve(strict=True)
    if "python" in launcher.name:
        if str(Path(actual[0]).resolve(strict=True)) != str(launcher) or actual[1:] != expected[1:]:
            raise RuntimeError("Live serving argv differs from configured Python command")
        interpreter = launcher
    else:
        try:
            with launcher.open(encoding="utf-8") as stream:
                first_line = stream.readline().strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError("Cannot read the serving launcher's interpreter") from exc
        if not first_line.startswith("#!"):
            raise RuntimeError("Serving launcher lacks an explicit interpreter")
        shebang = first_line[2:].strip().split()
        if len(shebang) != 1 or "python" not in Path(shebang[0]).name:
            raise RuntimeError("Serving launcher must use one explicit Python interpreter")
        interpreter = Path(shebang[0]).resolve(strict=True)
        if len(actual) < 2 or str(Path(actual[0]).resolve(strict=True)) != str(interpreter) or str(Path(actual[1]).resolve(strict=True)) != str(launcher) or actual[2:] != expected[1:]:
            raise RuntimeError("Live serving argv differs from configured vllm launcher")
    if process["executable"] != str(interpreter):
        raise RuntimeError("Live /proc executable differs from the launcher's interpreter")
    # Keep the venv interpreter spelling: resolving its symlink loses the venv.
    python = expected[0] if "python" in launcher.name else shebang[0]
    return {"python": python, "interpreter": file_identity(interpreter), "launcher": file_identity(launcher)}


def _python_identity(python):
    # Do not import vllm/torch, initialize CUDA, or run any model. Metadata and
    # known Python source paths suffice for version/source fingerprinting.
    source = """import json,hashlib,importlib.metadata as m
from pathlib import Path
packages={name:m.version(name) for name in ['vllm','torch','transformers']}
root=Path(m.distribution('vllm').locate_file('vllm')).resolve()
names=['__init__.py','entrypoints/openai/api_server.py','entrypoints/chat_utils.py','reasoning/qwen3_reasoning_parser.py','tool_parsers/qwen3coder_tool_parser.py']
chat_candidates=['entrypoints/openai/chat_completion/serving.py','entrypoints/openai/serving_chat.py']
chat=next((name for name in chat_candidates if (root/name).is_file()),None)
if chat is None:raise RuntimeError('Required chat serving source is absent')
names.append(chat)
files=[]
for name in names:
 p=root/name
 if not p.is_file():raise RuntimeError('Required serving source is absent: '+name)
 data=p.read_bytes();files.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'packages':packages,'vllm_package_root':str(root),'source_files':files},sort_keys=True))
"""
    result = subprocess.run([python, "-B", "-I", "-c", source], capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise RuntimeError("Cannot fingerprint the serving interpreter's package metadata and source")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Serving package probe did not return its public JSON contract") from exc
    if set(value.get("packages", {})) != {"vllm", "torch", "transformers"}:
        raise RuntimeError("Incomplete serving package identity")
    return value


def _get_models(base_url):
    request = Request(base_url.rstrip("/") + "/models", headers={"Accept": "application/json"})
    with urlopen(request, timeout=15) as response:
        body = json.load(response)
    selected = [row for row in body.get("data", []) if row.get("id") == MODEL]
    if len(selected) != 1:
        raise RuntimeError("Endpoint must expose the exact authorized model once")
    row = selected[0]
    if not isinstance(row.get("root"), str) or not Path(row["root"]).is_dir():
        raise RuntimeError("Endpoint /models must disclose its real local model root")
    return {key: row.get(key) for key in ("id", "root", "parent", "max_model_len")}


def collect_service_identity(service_config_path, pid=132):
    """Return a new receipt; caller embeds it as service_config.actual_identity."""
    _, configured = _configuration(service_config_path)
    before = _process(pid)
    launch = _match_command(configured["command"], before)
    package = _python_identity(launch["python"])
    model = _get_models(configured["base_url"])
    if str(Path(model["root"]).resolve(strict=True)) != configured["model_root"]:
        raise RuntimeError("Endpoint model root differs from the live command's model directory")
    model_files = [file_identity(Path(configured["model_root"]) / name)
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json")]
    parser_files = [file_identity(configured["parser_plugin"])]
    official = Path(configured["parser_plugin"]).with_name("official_qwen3_reasoning_parser.py")
    if not official.is_file():
        raise RuntimeError("The configured parser plugin's versioned official source is absent")
    parser_files.append(file_identity(official))
    if _process(pid) != before:
        raise RuntimeError("Serving process changed while collecting its identity")
    receipt = {"version": VERSION, **before, "configured_command": configured["command"],
        "base_url": configured["base_url"], "model_root": configured["model_root"],
        "reported_model": model, "launch": launch, **package,
        "reasoning_parser": configured["reasoning_parser"], "tool_call_parser": configured["tool_call_parser"],
        "parser_files": parser_files, "model_metadata_files": model_files,
        "weight_shards_rehashed": False}
    receipt["identity_sha256"] = digest(receipt)
    return receipt


def verify_state(config):
    """Fail closed on process/code/model-root drift; return measured identity.

    `config` is the frozen experiment configuration. All mismatches, including
    PID reuse, raise RuntimeError; no process is started/stopped or repaired.
    """
    try:
        path = config["model_identity"]["service_config"]
        source, configured = _configuration(path)
        expected = source.get("actual_identity")
        if not isinstance(expected, dict) or expected.get("version") != VERSION or expected.get("identity_sha256") != digest({key: value for key, value in expected.items() if key != "identity_sha256"}):
            raise RuntimeError("Frozen service config lacks a valid measured actual_identity")
        target = urlparse(config["base_url"])
        expected_url = urlparse(configured["base_url"])
        if target.scheme != expected_url.scheme or target.hostname not in {"127.0.0.1", "localhost"} or target.port != expected_url.port or target.path.rstrip("/") != "/v1":
            raise RuntimeError("Experiment endpoint differs from the measured local service")
        actual = collect_service_identity(path, expected["pid"])
        if actual != expected:
            raise RuntimeError("Live serving process, source, packages or model identity changed")
        return {"verified": True, "version": VERSION, "identity_sha256": actual["identity_sha256"],
            "pid": actual["pid"], "process_start_ticks": actual["process_start_ticks"],
            "model_root": actual["model_root"], "packages": actual["packages"]}
    except RuntimeError:
        raise
    except (OSError, KeyError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot verify the frozen live serving identity") from exc
