#!/usr/bin/env python3
"""RISE 轨迹阅读器 —— 只读的本地 Web 浏览器（仅标准库）。

把一个或多个运行目录中的 `_per_query/qid_*.json` 与
`_traces/qid_*/single.json` 渲染成可交互网页：运行级统计、按判分 /
终止原因筛选题目、逐轮查看思考内容 / 工具调用参数 / 完整工具结果。
只读，不修改任何轨迹文件。

用法::

    python scripts/trajectory_reader.py                # 自动扫描仓库内所有运行目录
    python scripts/trajectory_reader.py --run-dir runs/x
    python scripts/trajectory_reader.py --port 8787 --no-open

默认地址 http://127.0.0.1:8787/ 。扫描时跳过 .venv / .git / __pycache__
/ corpus 等目录；任何包含 `_per_query` 或 `_traces` 子目录的文件夹都算
一个运行目录。聚合统计（正确率、终止原因分布等）全部从 per-query 行
文件现场计算，不信任运行目录里可能过期的 `_summary.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import traceback
import urllib.parse
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
_PRUNE_DIRS = {".venv", ".git", "__pycache__", "node_modules", "corpus",
               ".idea", ".vscode", "structured"}
_QID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RID_RE = re.compile(r"^[0-9a-f]{12}$")


def _sort_key(qid: str) -> tuple[int, int | str, str]:
    return (0, int(qid), qid) if qid.isdecimal() else (1, 0, qid)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_run_dir(d: Path) -> bool:
    return (d / "_per_query").is_dir() or (d / "_traces").is_dir()


def _run_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def discover_run_dirs(roots: list[Path], *, max_depth: int = 6) -> list[Path]:
    """Find every directory that looks like a run (has _per_query or _traces)."""
    found: list[Path] = []
    seen: set[Path] = set()

    def visit(root: Path) -> None:
        root = root.resolve()
        if not root.is_dir() or root in seen:
            return
        if _is_run_dir(root):
            seen.add(root)
            found.append(root)
            return
        base = len(root.parts)
        for dirpath, dirnames, _filenames in os.walk(root):
            d = Path(dirpath)
            if _is_run_dir(d):
                if d not in seen:
                    seen.add(d)
                    found.append(d)
                dirnames[:] = []
                continue
            dirnames[:] = sorted(n for n in dirnames if n not in _PRUNE_DIRS)
            if len(d.parts) - base >= max_depth:
                dirnames[:] = []

    for root in roots:
        visit(root)
    found.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return found


def _trace_file(run_dir: Path, qid: str) -> Path | None:
    """Locate the trace JSON for a qid, preferring single.json."""
    qdir = run_dir / "_traces" / f"qid_{qid}"
    if not qdir.is_dir():
        return None
    preferred = qdir / "single.json"
    if preferred.is_file():
        return preferred
    others = sorted(qdir.glob("*.json"))
    return others[0] if others else None


def _trace_index(run_dir: Path) -> dict[str, Path]:
    traces: dict[str, Path] = {}
    troot = run_dir / "_traces"
    if not troot.is_dir():
        return traces
    for qdir in troot.iterdir():
        if not (qdir.is_dir() and qdir.name.startswith("qid_")):
            continue
        qid = qdir.name[len("qid_"):]
        f = _trace_file(run_dir, qid)
        if f is not None:
            traces[qid] = f
    return traces


def _blocked_map(run_dir: Path) -> dict[str, str]:
    p = run_dir / "_blocked_queries.json"
    if not p.is_file():
        return {}
    try:
        obj = _read_json(p)
    except (OSError, ValueError):
        return {}
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    return {}


def _first_row_field(rows: dict[str, dict], key: str) -> Any:
    for qid in sorted(rows, key=_sort_key):
        row = rows.get(qid)
        if isinstance(row, dict) and key in row:
            return row[key]
    return None


def run_overview(run_dir: Path) -> dict[str, Any]:
    """Cheap run metadata for the runs listing (counts only, no full parse)."""
    n_rows = len(list((run_dir / "_per_query").glob("qid_*.json"))) \
        if (run_dir / "_per_query").is_dir() else 0
    n_traces = len(_trace_index(run_dir))
    n_blocked = len(_blocked_map(run_dir))
    model = None
    run_hash = None
    first = next(iter(sorted((run_dir / "_per_query").glob("qid_*.json"))), None) \
        if (run_dir / "_per_query").is_dir() else None
    if first is not None:
        try:
            row = _read_json(first)
            if isinstance(row, dict):
                model = row.get("model")
                run_hash = row.get("run_config_hash")
        except (OSError, ValueError):
            pass
    try:
        rel = str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(run_dir)
    return {
        "id": _run_id(run_dir),
        "name": run_dir.name,
        "relpath": rel,
        "mtime": run_dir.stat().st_mtime,
        "n_rows": n_rows,
        "n_traces": n_traces,
        "n_blocked": n_blocked,
        "model": model,
        "run_config_hash": run_hash,
    }


def run_index(run_dir: Path) -> dict[str, Any]:
    """Full index: aggregates + one entry per query. Computed from row files."""
    rows: dict[str, dict[str, Any]] = {}
    per_dir = run_dir / "_per_query"
    if per_dir.is_dir():
        for p in sorted(per_dir.glob("qid_*.json")):
            try:
                obj = _read_json(p)
            except (OSError, ValueError):
                continue
            if isinstance(obj, dict):
                rows[p.stem[len("qid_"):]] = obj
    traces = _trace_index(run_dir)
    blocked = _blocked_map(run_dir)

    entries: list[dict[str, Any]] = []
    all_qids = set(rows) | set(traces) | set(blocked)
    for qid in sorted(all_qids, key=_sort_key):
        row = rows.get(qid)
        if row is not None:
            judge = row.get("judge") or {}
            entries.append({
                "qid": qid,
                "has_row": True,
                "has_trace": qid in traces,
                "question": row.get("question") or "",
                "gold_answer": row.get("gold_answer") or "",
                "is_correct": bool(row.get("is_correct")),
                "judge_pending": (judge.get("reasoning") or "") == "",
                "terminated_by": row.get("terminated_by") or "",
                "n_turns": row.get("n_turns"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "total_cost_usd": row.get("total_cost_usd"),
                "coverage_any": row.get("coverage_any"),
            })
        else:
            entry: dict[str, Any] = {
                "qid": qid,
                "has_row": False,
                "has_trace": qid in traces,
                "question": "",
                "gold_answer": "",
                "is_correct": False,
                "judge_pending": True,
                "terminated_by": "",
                "n_turns": None,
                "elapsed_seconds": None,
                "total_cost_usd": None,
                "coverage_any": None,
            }
            if qid in traces:
                try:
                    tr = _read_json(traces[qid])
                    if isinstance(tr, dict):
                        up = tr.get("user_prompt") or ""
                        if up.startswith("QUESTION:\n"):
                            up = up[len("QUESTION:\n"):]
                        entry["question"] = up
                        totals = tr.get("totals") or {}
                        entry["n_turns"] = totals.get("n_turns")
                        entry["terminated_by"] = tr.get("terminated_by") or ""
                except (OSError, ValueError):
                    pass
            entries.append(entry)

    judged_rows = list(rows.values())
    n_correct = sum(1 for r in judged_rows if r.get("is_correct"))
    termination: Counter[str] = Counter()
    tool_totals: Counter[str] = Counter()
    for r in judged_rows:
        termination[str(r.get("terminated_by") or "unknown").split(":", 1)[0]] += 1
        for name, count in (r.get("tool_usage") or {}).items():
            if isinstance(count, int):
                tool_totals[str(name)] += count
    n_rows = len(judged_rows)

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in judged_rows
                if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    try:
        rel = str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(run_dir)
    overview = {
        "id": _run_id(run_dir),
        "name": run_dir.name,
        "relpath": rel,
        "n_rows": n_rows,
        "n_traces": len(traces),
        "n_blocked": len(blocked),
        "n_correct": n_correct,
        "accuracy": (n_correct / n_rows) if n_rows else None,
        "n_judge_pending": sum(1 for e in entries if e["judge_pending"] and e["has_row"]),
        "mean_n_turns": _mean("n_turns"),
        "mean_elapsed_seconds": _mean("elapsed_seconds"),
        "mean_total_cost_usd": _mean("total_cost_usd"),
        "termination": dict(sorted(termination.items())),
        "tool_usage_totals": dict(sorted(tool_totals.items())),
        "models": sorted({str(r.get("model")) for r in judged_rows if r.get("model")}),
        "run_config_hash": _first_row_field(rows, "run_config_hash"),
        "schema_versions": sorted({str(r.get("schema_version")) for r in judged_rows
                                   if r.get("schema_version") is not None}),
    }
    return {"run": overview, "queries": entries, "blocked": blocked}


def query_detail(run_dir: Path, qid: str) -> dict[str, Any] | None:
    row_path = run_dir / "_per_query" / f"qid_{qid}.json"
    trace_path = _trace_file(run_dir, qid)
    if not row_path.is_file() and trace_path is None:
        return None
    row = None
    if row_path.is_file():
        try:
            row = _read_json(row_path)
        except (OSError, ValueError) as exc:
            row = {"_read_error": f"{type(exc).__name__}: {exc}"}
    trace = None
    if trace_path is not None:
        try:
            trace = _read_json(trace_path)
        except (OSError, ValueError) as exc:
            trace = {"_read_error": f"{type(exc).__name__}: {exc}"}
    blocked = _blocked_map(run_dir).get(qid)
    return {"row": row, "trace": trace, "blocked_reason": blocked}


class RunRegistry:
    """Thread-safe map run-id -> run dir, rescanned on demand."""

    def __init__(self, roots: list[Path], pinned: list[Path]) -> None:
        self.roots = roots
        self.pinned = [p.resolve() for p in pinned]
        self._lock = threading.Lock()
        self._runs: dict[str, Path] = {}
        self.scan()

    def scan(self) -> dict[str, Path]:
        dirs = discover_run_dirs(self.roots)
        for p in self.pinned:
            if _is_run_dir(p) and p not in dirs:
                dirs.append(p)
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        with self._lock:
            self._runs = {_run_id(d): d for d in dirs}
        return dict(self._runs)

    def get(self, rid: str) -> Path | None:
        with self._lock:
            return self._runs.get(rid)


_REGISTRY: RunRegistry | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "RiseTrajectoryReader/1.0"
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # keep the console quiet
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    # ---- routing --------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        try:
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                runs = _REGISTRY.scan()
                self._json([run_overview(d) for d in runs.values()])
                return
            m = re.fullmatch(r"/api/run/([0-9a-f]{12})/index", path)
            if m:
                run_dir = _REGISTRY.get(m.group(1))
                if run_dir is None:
                    self._error(404, f"unknown run id {m.group(1)!r}")
                    return
                self._json(run_index(run_dir))
                return
            m = re.fullmatch(r"/api/run/([0-9a-f]{12})/query/([^/]+)", path)
            if m:
                rid, qid = m.group(1), urllib.parse.unquote(m.group(2))
                run_dir = _REGISTRY.get(rid)
                if run_dir is None:
                    self._error(404, f"unknown run id {rid!r}")
                    return
                if not _QID_RE.match(qid):
                    self._error(400, f"invalid query id {qid!r}")
                    return
                detail = query_detail(run_dir, qid)
                if detail is None:
                    self._error(404, f"no row or trace for qid {qid!r}")
                    return
                self._json(detail)
                return
            self._error(404, f"no such path {path!r}")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            try:
                self._error(500, "internal server error")
            except Exception:
                pass


# =============================================================================
# Embedded single-page frontend. Kept as one raw string so the whole reader
# stays a single stdlib-only file with zero static assets.
# =============================================================================
_PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RISE 轨迹阅读器</title>
<style>
  :root{
    --bg:#f6f7f9; --panel:#ffffff; --line:#e3e6ea; --ink:#1c2128; --dim:#6a737d;
    --blue:#0b66c3; --green:#1a7f37; --red:#cf222e; --amber:#9a6700;
    --chip:#eef1f4; --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{font:14px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
       color:var(--ink);background:var(--bg)}
  header{display:flex;align-items:center;gap:14px;padding:10px 16px;background:var(--panel);
         border-bottom:1px solid var(--line);flex-wrap:wrap}
  .brand{font-weight:700;font-size:15px;white-space:nowrap}
  select,input{font:inherit;color:var(--ink);background:#fff;border:1px solid var(--line);
               border-radius:6px;padding:4px 8px}
  input{outline:none}
  input:focus,select:focus{border-color:var(--blue);box-shadow:0 0 0 2px rgba(11,102,195,.15)}
  #layout{display:flex;height:calc(100% - 55px)}
  aside{width:340px;min-width:280px;background:var(--panel);border-right:1px solid var(--line);
        display:flex;flex-direction:column;overflow:hidden}
  main{flex:1;overflow:auto;padding:18px 26px 80px}
  .empty{color:var(--dim);margin-top:80px;text-align:center;font-size:15px}
  /* sidebar */
  #stats{padding:10px 14px;border-bottom:1px solid var(--line);font-size:12.5px;color:var(--dim);
         display:flex;flex-wrap:wrap;gap:6px}
  .chip{background:var(--chip);border-radius:10px;padding:1px 9px;white-space:nowrap}
  #filters{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:10px 14px;border-bottom:1px solid var(--line)}
  #qsearch{grid-column:1 / -1}
  #qlist{overflow:auto;flex:1}
  .q{padding:9px 14px;border-bottom:1px solid var(--line);cursor:pointer}
  .q:hover{background:#f2f6fb}
  .q.active{background:#e8f1fb;box-shadow:inset 3px 0 0 var(--blue)}
  .q-top{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
  .qid{font-family:var(--mono);font-weight:600}
  .badge{font-size:11.5px;border-radius:9px;padding:0 7px;white-space:nowrap}
  .b-ok{background:#dafbe1;color:var(--green)} .b-bad{background:#ffebe9;color:var(--red)}
  .b-dim{background:var(--chip);color:var(--dim)} .b-warn{background:#fff8c5;color:var(--amber)}
  .b-block{background:#ffebe9;color:var(--red)}
  .q-text{color:#3d4650;font-size:12.5px;margin-top:2px;
          overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
  .q-meta{color:var(--dim);font-size:11.5px;margin-top:3px;font-family:var(--mono)}
  /* main */
  .crumbs{color:var(--dim);font-size:12.5px;margin-bottom:6px}
  h1{font-size:18px;margin:0 0 4px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .kv{margin:14px 0 4px;font-weight:600;font-size:13px}
  pre{background:#f6f8fa;border:1px solid var(--line);border-radius:8px;padding:10px 12px;
      overflow:auto;font:12.3px/1.5 var(--mono);white-space:pre-wrap;word-break:break-word;margin:4px 0}
  .metric-grid{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
  .metric{background:var(--panel);border:1px solid var(--line);border-radius:8px;
          padding:6px 12px;min-width:92px}
  .metric .n{font:600 14px var(--mono)}
  .metric .l{font-size:11.5px;color:var(--dim)}
  details{margin:8px 0;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
  details summary{cursor:pointer;padding:7px 12px;font-size:13px;border-radius:8px;user-select:none}
  details[open] summary{border-bottom:1px solid var(--line);border-radius:8px 8px 0 0}
  details>div.pad, details>pre{margin:8px 10px 10px}
  .turn{border-color:#cfd6de}
  .turn>summary{background:#fbfcfd;font-family:var(--mono);font-size:12.8px;
                display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .call{background:#fff;margin:6px 10px}
  .call>summary{font-family:var(--mono);font-size:12.3px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .tool{border-radius:5px;padding:0 7px;font-weight:700;font-size:11.5px}
  .t-search{background:#e1d4f5;color:#5f3aa6} .t-bash{background:#d0e3ff;color:#0847a8}
  .t-read{background:#d4f2dc;color:#11652c} .t-other{background:var(--chip);color:var(--dim)}
  .err{color:var(--red);font-weight:600}
  .banner{border-radius:8px;padding:8px 14px;margin:10px 0;font-size:13px}
  .banner.warn{background:#fff8c5;border:1px solid #e7d24b;color:var(--amber)}
  .banner.bad{background:#ffebe9;border:1px solid #ffb1ad;color:var(--red)}
  .turnbar{display:flex;gap:8px;align-items:center;margin:16px 0 8px;flex-wrap:wrap}
  .turnbar input{width:150px}
  .mut{color:var(--dim)}
  details.turn.hide{display:none}
  .loadmore{display:block;width:calc(100% - 20px);margin:6px 10px;padding:6px;cursor:pointer;
            background:#f0f3f6;border:1px dashed var(--line);border-radius:8px;font:inherit;color:var(--dim)}
  .loadmore:hover{background:#e8edf2}
  .full{max-height:420px}
  a{color:var(--blue)}
</style>
</head>
<body>
<header>
  <span class="brand">RISE 轨迹阅读器</span>
  <select id="runSel"></select>
  <span id="runHint" class="mut" style="font-size:12.5px"></span>
</header>
<div id="layout">
  <aside>
    <div id="stats"></div>
    <div id="filters">
      <input id="qsearch" placeholder="搜索题号 / 问题 / 答案">
      <select id="fStatus">
        <option value="all">全部状态</option>
        <option value="correct">判对</option>
        <option value="wrong">判错</option>
        <option value="done">正常完成</option>
        <option value="undone">未正常完成</option>
        <option value="pending">待判分</option>
        <option value="blocked">被封锁（无行）</option>
      </select>
      <select id="fSort">
        <option value="qid-asc">题号 ↑</option>
        <option value="qid-desc">题号 ↓</option>
        <option value="turns-desc">轮数 多→少</option>
        <option value="cost-desc">费用 高→低</option>
        <option value="time-desc">用时 长→短</option>
      </select>
    </div>
    <div id="qlist"></div>
  </aside>
  <main id="main"><div class="empty">在左侧选择一个题目查看轨迹</div></main>
</div>
<script>
"use strict";
const esc = s => String(s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmtInt = n => (typeof n === "number" && isFinite(n)) ? n.toLocaleString("en-US") : "—";
const fmtMoney = n => (typeof n === "number" && isFinite(n)) ? "$" + n.toFixed(4) : "—";
const fmtPct = n => (typeof n === "number" && isFinite(n)) ? (100 * n).toFixed(1) + "%" : "—";
function fmtSec(s){
  if (typeof s !== "number" || !isFinite(s)) return "—";
  if (s < 90) return s.toFixed(1) + " 秒";
  const m = s / 60;
  return m < 90 ? m.toFixed(1) + " 分" : (m / 60).toFixed(1) + "时";
}
const TERM_ZH = {done:"正常完成", max_turns:"轮数上限", max_iterations:"轮数上限",
                 coerced_max_turns:"强制轮数上限", wall_clock_timeout:"墙钟超时",
                 output_truncated:"输出截断", context_limit:"上下文超限", unknown:"未知"};
const termZh = t => TERM_ZH[String(t||"").split(":")[0]] || t;
const FINISH_ZH = {stop:"正常结束", tool_calls:"工具调用", length:"长度截断", content_filter:"内容过滤"};

let RUNS = [], RID = null, INDEX = null, QID = null, DETAIL = null;

async function api(path){
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.text()) || ("HTTP " + r.status));
  return r.json();
}

/* ---------------- run + index ---------------- */
async function loadRuns(){
  RUNS = await api("/api/runs");
  const sel = document.getElementById("runSel");
  sel.innerHTML = "";
  for (const r of RUNS){
    const o = document.createElement("option");
    o.value = r.id;
    o.textContent = `${r.name}（${r.n_rows} 行 / ${r.n_traces} 轨迹${r.n_blocked ? " / " + r.n_blocked + " 封锁" : ""}）`;
    sel.appendChild(o);
  }
  const hash = parseHash();
  if (hash.rid && RUNS.some(r => r.id === hash.rid)) sel.value = hash.rid;
  else if (RUNS.length) sel.value = RUNS[0].id;
  sel.onchange = () => loadIndex(sel.value, true);
  if (RUNS.length) await loadIndex(sel.value, false, hash.qid);
  else document.getElementById("main").innerHTML =
    '<div class="empty">未发现运行目录（需包含 _per_query 或 _traces）</div>';
}

async function loadIndex(rid, resetMain, wantQid){
  RID = rid; QID = null; DETAIL = null;
  INDEX = await api(`/api/run/${rid}/index`);
  renderStats();
  renderList();
  if (resetMain) document.getElementById("main").innerHTML =
    '<div class="empty">在左侧选择一个题目查看轨迹</div>';
  const target = wantQid && INDEX.queries.some(q => q.qid === wantQid) ? wantQid : null;
  if (target) openQuery(target);
  else setHash(null);
}

function renderStats(){
  const r = INDEX.run;
  const bits = [];
  bits.push(`<span class="chip">行 <b>${r.n_rows}</b></span>`);
  bits.push(`<span class="chip">轨迹 <b>${r.n_traces}</b></span>`);
  if (r.accuracy !== null && r.accuracy !== undefined)
    bits.push(`<span class="chip">判对 <b>${r.n_correct}</b> = <b>${fmtPct(r.accuracy)}</b></span>`);
  if (r.n_judge_pending) bits.push(`<span class="chip b-warn">待判 ${r.n_judge_pending}</span>`);
  if (r.n_blocked) bits.push(`<span class="chip b-bad">封锁 ${r.n_blocked}</span>`);
  if (r.mean_n_turns !== null && r.mean_n_turns !== undefined)
    bits.push(`<span class="chip">平均 <b>${r.mean_n_turns.toFixed(1)}</b> 轮</span>`);
  for (const [k, v] of Object.entries(r.termination || {}))
    bits.push(`<span class="chip">${esc(termZh(k))} ${v}</span>`);
  if ((r.models || []).length) bits.push(`<span class="chip">${esc(r.models.join(", "))}</span>`);
  if (r.run_config_hash) bits.push(`<span class="chip mono">#${esc(r.run_config_hash)}</span>`);
  document.getElementById("stats").innerHTML = bits.join("");
  document.getElementById("runHint").textContent = r.relpath;
}

/* ---------------- query list ---------------- */
function qMatchFilter(e){
  const st = document.getElementById("fStatus").value;
  if (st === "correct") return e.has_row && e.is_correct;
  if (st === "wrong") return e.has_row && !e.is_correct;
  if (st === "done") return e.has_row && String(e.terminated_by).split(":")[0] === "done";
  if (st === "undone") return e.has_row && String(e.terminated_by).split(":")[0] !== "done";
  if (st === "pending") return e.has_row && e.judge_pending;
  if (st === "blocked") return !e.has_row;
  return true;
}
function renderList(){
  const qs = document.getElementById("qsearch").value.trim().toLowerCase();
  const sort = document.getElementById("fSort").value;
  let items = INDEX.queries.filter(qMatchFilter);
  if (qs) items = items.filter(e =>
    (e.qid + "\n" + e.question + "\n" + e.gold_answer).toLowerCase().includes(qs));
  items = items.slice().sort((a, b) => {
    if (sort === "qid-desc") return cmpQid(b.qid, a.qid);
    if (sort === "turns-desc") return num(b.n_turns) - num(a.n_turns);
    if (sort === "cost-desc") return num(b.total_cost_usd) - num(a.total_cost_usd);
    if (sort === "time-desc") return num(b.elapsed_seconds) - num(a.elapsed_seconds);
    return cmpQid(a.qid, b.qid);
  });
  const box = document.getElementById("qlist");
  box.innerHTML = "";
  for (const e of items){
    const d = document.createElement("div");
    d.className = "q" + (e.qid === QID ? " active" : "");
    d.dataset.qid = e.qid;
    let badges = "";
    if (!e.has_row) badges = `<span class="badge b-block">封锁</span>`;
    else if (e.judge_pending) badges = `<span class="badge b-warn">待判</span>`;
    else badges = `<span class="badge ${e.is_correct ? "b-ok" : "b-bad"}">${e.is_correct ? "判对" : "判错"}</span>`;
    if (e.has_row && String(e.terminated_by).split(":")[0] !== "done")
      badges += ` <span class="badge b-dim">${esc(termZh(e.terminated_by))}</span>`;
    d.innerHTML = `<div class="q-top"><span class="qid">qid ${esc(e.qid)}</span> ${badges}</div>
      <div class="q-text">${esc(e.question)}</div>
      <div class="q-meta">${fmtInt(e.n_turns)} 轮 · ${fmtMoney(e.total_cost_usd)} · ${fmtSec(e.elapsed_seconds)}</div>`;
    d.onclick = () => openQuery(e.qid);
    box.appendChild(d);
  }
  const head = document.createElement("div");
  head.className = "q"; head.style.background = "#f6f7f9"; head.style.cursor = "default";
  head.innerHTML = `<div class="mut" style="font-size:12px">显示 ${items.length} / ${INDEX.queries.length} 题</div>`;
  box.prepend(head);
}
function num(v){ return typeof v === "number" && isFinite(v) ? v : -1; }
function cmpQid(a, b){
  const na = parseInt(a, 10), nb = parseInt(b, 10);
  if (!isNaN(na) && !isNaN(nb) && String(na) === a && String(nb) === b) return na - nb;
  return a < b ? -1 : a > b ? 1 : 0;
}

/* ---------------- query detail ---------------- */
async function openQuery(qid){
  QID = qid;
  setHash(qid);
  document.querySelectorAll(".q").forEach(el =>
    el.classList.toggle("active", el.dataset.qid === qid));
  const main = document.getElementById("main");
  main.innerHTML = '<div class="empty">加载中…</div>';
  try { DETAIL = await api(`/api/run/${RID}/query/${encodeURIComponent(qid)}`); }
  catch (err){ main.innerHTML = `<div class="empty">加载失败：${esc(err.message)}</div>`; return; }
  renderDetail();
}

function renderDetail(){
  const main = document.getElementById("main");
  const {row, trace, blocked_reason} = DETAIL;
  const r = row && !row._read_error ? row : null;
  const t = trace && !trace._read_error ? trace : null;
  const parts = [];
  parts.push(`<div class="crumbs">${esc(INDEX.run.relpath)} · qid ${esc(QID)}</div>`);
  parts.push(`<h1>qid ${esc(QID)}`);
  if (blocked_reason) parts.push(`<span class="badge b-block">被封锁：${esc(blocked_reason)}</span>`);
  if (r){
    if (r.is_correct) parts.push('<span class="badge b-ok">判对</span>');
    else if ((r.judge || {}).reasoning === "") parts.push('<span class="badge b-warn">待判分</span>');
    else parts.push('<span class="badge b-bad">判错</span>');
    parts.push(`<span class="badge b-dim">${esc(termZh(r.terminated_by))}</span>`);
  }
  parts.push("</h1>");
  if (row && row._read_error) parts.push(`<div class="banner bad">行文件读取失败：${esc(row._read_error)}</div>`);
  if (trace && trace._read_error) parts.push(`<div class="banner bad">轨迹文件读取失败：${esc(trace._read_error)}</div>`);

  if (r){
    const judge = r.judge || {};
    parts.push(`<div class="kv">问题</div><pre>${esc(r.question)}</pre>`);
    parts.push(`<div class="metric-grid">
      <div class="metric"><div class="n">${esc(r.gold_answer || "—")}</div><div class="l">标准答案</div></div>
      <div class="metric"><div class="n">${esc(judge.extracted_final_answer || "—")}</div><div class="l">抽取答案</div></div>
      <div class="metric"><div class="n">${fmtInt(r.n_turns)}</div><div class="l">轮数</div></div>
      <div class="metric"><div class="n">${fmtInt(r.elapsed_seconds)}</div><div class="l">用时（秒）</div></div>
      <div class="metric"><div class="n">${fmtMoney(r.total_cost_usd)}</div><div class="l">总费用</div></div>
      <div class="metric"><div class="n">${fmtInt(r.n_surfaced)}</div><div class="l">呈现文档数</div></div>
      <div class="metric"><div class="n">${r.coverage_any ?? "—"} / ${r.coverage_mean ?? "—"} / ${r.coverage_all ?? "—"}</div><div class="l">gold 覆盖 any/mean/all</div></div>
      <div class="metric"><div class="n">${r.evidence_coverage_mean ?? "—"}</div><div class="l">证据覆盖 mean</div></div>
    </div>`);
    const u = r.agent_usage || {};
    parts.push(`<div class="metric-grid">
      <div class="metric"><div class="n">${fmtInt(u.input_tokens)}</div><div class="l">输入 token</div></div>
      <div class="metric"><div class="n">${fmtInt(u.cached_input_tokens)}</div><div class="l">缓存输入</div></div>
      <div class="metric"><div class="n">${fmtInt(u.output_tokens)}</div><div class="l">输出 token</div></div>
      <div class="metric"><div class="n">${fmtInt(u.reasoning_tokens)}</div><div class="l">思考 token</div></div>
      <div class="metric"><div class="n">${fmtMoney(r.agent_cost_usd)}</div><div class="l">agent 费用</div></div>
      <div class="metric"><div class="n">${fmtMoney(r.judge_cost_usd)}</div><div class="l">判分费用</div></div>
    </div>`);
    const tools = Object.entries(r.tool_usage || {});
    if (tools.length)
      parts.push(`<div class="kv">工具用量</div><div>${tools.map(([k, v]) =>
        `<span class="chip" style="font-family:var(--mono)">${esc(k)} × ${v}</span>`).join(" ")}</div>`);
    if (r.final_text)
      parts.push(`<details><summary>模型最终回答（${r.final_text.length} 字符）</summary>
        <pre>${esc(r.final_text)}</pre></details>`);
    if (judge.reasoning)
      parts.push(`<details><summary>判分理由（模型 ${esc(judge.model || "—")}，置信度 ${judge.confidence ?? "—"}）</summary>
        <pre>${esc(judge.reasoning)}</pre></details>`);
    if (r.discarded_attempt_audit && (r.discarded_attempt_audit.tokens || r.discarded_attempt_audit.cost_usd))
      parts.push(`<details><summary>废弃尝试审计（tokens ${fmtInt(r.discarded_attempt_audit.tokens)}，${fmtMoney(r.discarded_attempt_audit.cost_usd)}）</summary>
        <pre>${esc(JSON.stringify(r.discarded_attempt_audit, null, 2))}</pre></details>`);
    if (r.surfaced_relpaths && r.surfaced_relpaths.length)
      parts.push(`<details><summary>呈现过的文档路径（${r.surfaced_relpaths.length}）</summary>
        <pre class="full">${esc(r.surfaced_relpaths.join("\n"))}</pre></details>`);
    if (r.all_retrieved && r.all_retrieved.length)
      parts.push(`<details><summary>检索过的 docid（${r.all_retrieved.length}）</summary>
        <pre class="full">${esc(r.all_retrieved.join("\n"))}</pre></details>`);
    if (r.gold_doc_ids && r.gold_doc_ids.length)
      parts.push(`<div class="kv">gold docid</div><pre>${esc(r.gold_doc_ids.join("\n"))}</pre>`);
    if (r.runner_specific && Object.keys(r.runner_specific).length)
      parts.push(`<details><summary>runner_specific</summary>
        <pre>${esc(JSON.stringify(r.runner_specific, null, 2))}</pre></details>`);
    if (r.run_config && Object.keys(r.run_config).length)
      parts.push(`<details><summary>run_config</summary>
        <pre>${esc(JSON.stringify(r.run_config, null, 2))}</pre></details>`);
  } else if (t){
    let q = t.user_prompt || "";
    if (q.startsWith("QUESTION:\n")) q = q.slice(10);
    parts.push(`<div class="kv">问题（来自轨迹）</div><pre>${esc(q)}</pre>`);
  }

  if (t){
    if (t.error) parts.push(`<div class="banner warn">轨迹记录的终止错误：${esc(t.error)}</div>`);
    if (t.system_prompt) parts.push(`<details><summary>系统提示（${t.system_prompt.length} 字符）</summary>
      <pre class="full">${esc(t.system_prompt)}</pre></details>`);
    if (!r && t.user_prompt) parts.push(`<details><summary>用户提示（${t.user_prompt.length} 字符）</summary>
      <pre class="full">${esc(t.user_prompt)}</pre></details>`);
    if (!r && t.final_text) parts.push(`<details><summary>最终回答（${t.final_text.length} 字符）</summary>
      <pre>${esc(t.final_text)}</pre></details>`);
    const turns = Array.isArray(t.turns) ? t.turns : null;
    if (turns){
      parts.push(`<div class="turnbar">
        <input id="tsearch" placeholder="在轨迹中搜索（内容/思考/参数/工具结果）">
        <span id="tmatch" class="mut"></span>
        <span style="flex:1"></span>
        <input id="jump" type="number" min="0" placeholder="轮次 #" style="width:80px">
        <button id="jumpBtn" style="font:inherit;padding:4px 10px;cursor:pointer">跳转</button>
        <button id="expandAll" style="font:inherit;padding:4px 10px;cursor:pointer">全部展开</button>
        <button id="collapseAll" style="font:inherit;padding:4px 10px;cursor:pointer">全部收起</button>
      </div>`);
      parts.push(`<div id="turns"></div>`);
    } else {
      parts.push(`<div class="banner warn">该轨迹没有可识别的 turns 字段，原始 JSON 如下</div>
        <details open><summary>原始轨迹</summary><pre class="full">${esc(JSON.stringify(t, null, 2))}</pre></details>`);
    }
  }
  if (!r && !t) parts.push('<div class="banner bad">既无行文件也无轨迹文件</div>');

  main.innerHTML = parts.join("");
  if (t && Array.isArray(t.turns)) renderTurns(t.turns);
}

/* ---------------- turn timeline (lazy) ---------------- */
const TURN_CHUNK = 40, CALL_CHUNK = 40, EXPAND_CAP = 25;
let _turnEls = [], _turnsData = null, _turnsRendered = 0;

function renderTurns(turns){
  _turnsData = turns; _turnEls = []; _turnsRendered = 0;
  turns.forEach((tn, i) => _turnEls.push(turnCard(tn, i)));
  const box = document.getElementById("turns");
  box.innerHTML = "";
  renderTurnChunk(0);

  const ts = document.getElementById("tsearch");
  let deb = null;
  ts.oninput = () => { clearTimeout(deb); deb = setTimeout(applyTurnSearch, 250); };
  document.getElementById("jumpBtn").onclick = jumpToTurn;
  document.getElementById("jump").onkeydown = ev => { if (ev.key === "Enter") jumpToTurn(); };
  document.getElementById("expandAll").onclick = expandVisible;
  document.getElementById("collapseAll").onclick = () => _turnEls.forEach(el => { el.open = false; });
}

function renderTurnChunk(start){
  const box = document.getElementById("turns");
  const more = box.querySelector("button.loadmore.turnmore");
  if (more) more.remove();
  const end = Math.min(_turnsData.length, start + TURN_CHUNK);
  for (let i = start; i < end; i++) box.appendChild(_turnEls[i]);
  _turnsRendered = Math.max(_turnsRendered, end);
  if (end < _turnsData.length){
    const b = document.createElement("button");
    b.className = "loadmore turnmore";
    b.textContent = `载入更多轮次（剩余 ${_turnsData.length - end} 轮）`;
    b.onclick = () => renderTurnChunk(end);
    box.appendChild(b);
  }
}

function turnCard(tn, idx){
  const d = document.createElement("details");
  d.className = "turn";
  const tok = tn.tokens || {};
  const calls = Array.isArray(tn.tool_calls) ? tn.tool_calls : [];
  const fin = FINISH_ZH[tn.finish_reason] || tn.finish_reason || "—";
  const s = document.createElement("summary");
  s.innerHTML =
    `<b>Turn ${tn.turn ?? idx}</b>` +
    `<span class="badge ${fin === "正常结束" ? "b-ok" : "b-dim"}">${esc(fin)}</span>` +
    `<span>调用 ${calls.length}</span>` +
    `<span title="输入 / 缓存 / 输出 / 思考 token">tok ${fmtInt(tok.prompt)}/${fmtInt(tok.cached_prompt)}/${fmtInt(tok.completion)}/${fmtInt(tok.reasoning)}</span>` +
    `<span>${fmtMoney(tn.cost_usd)}</span>`;
  const badge = document.createElement("span");
  badge.className = "tmatch badge b-dim";
  badge.style.display = "none";
  s.appendChild(badge);
  d.appendChild(s);
  d.ontoggle = () => { if (d.open) buildTurn(d, idx); };
  return d;
}

function buildTurn(el, idx){
  if (el.dataset.built) return;
  el.dataset.built = "1";
  const tn = _turnsData[idx];
  const body = document.createElement("div");
  body.className = "pad";
  if (tn.reasoning_content)
    body.insertAdjacentHTML("beforeend",
      `<details><summary>思考内容（${tn.reasoning_content.length} 字符）</summary>
        <pre class="full">${esc(tn.reasoning_content)}</pre></details>`);
  if (tn.content)
    body.insertAdjacentHTML("beforeend",
      `<div class="kv">回复内容（${tn.content.length} 字符）</div><pre>${esc(tn.content)}</pre>`);
  const calls = Array.isArray(tn.tool_calls) ? tn.tool_calls : [];
  if (calls.length){
    const holder = document.createElement("div");
    holder.dataset.next = "0";
    body.appendChild(holder);
    renderCallChunk(holder, calls);
  }
  el.appendChild(body);
}

function renderCallChunk(holder, calls){
  const start = parseInt(holder.dataset.next, 10) || 0;
  const end = Math.min(calls.length, start + CALL_CHUNK);
  for (let i = start; i < end; i++) holder.appendChild(callCard(calls[i], i));
  holder.dataset.next = String(end);
  const old = holder.querySelector("button.loadmore");
  if (old) old.remove();
  if (end < calls.length){
    const b = document.createElement("button");
    b.className = "loadmore";
    b.textContent = `载入更多工具调用（剩余 ${calls.length - end}）`;
    b.onclick = () => renderCallChunk(holder, calls);
    holder.appendChild(b);
  }
}

function callCard(c, idx){
  const d = document.createElement("details");
  d.className = "call";
  const cls = {search:"t-search", bash:"t-bash", read:"t-read"}[c.name] || "t-other";
  const flags = [];
  if (c.error) flags.push('<span class="err">工具返回错误</span>');
  if (c.result_truncated) flags.push('<span class="badge b-warn">结果截断存储</span>');
  const summaryText = (c.result_summary || "").slice(0, 90).replace(/\s+/g, " ");
  const s = document.createElement("summary");
  s.innerHTML = `<span class="tool ${cls}">${esc(c.name || "?")}</span>` +
    `<span>#${idx + 1}</span>` +
    `<span>${typeof c.duration_seconds === "number" ? c.duration_seconds.toFixed(2) + "s" : "—"}</span>` +
    `<span>${fmtInt(c.result_chars)} 字符</span>` + flags.join("") +
    `<span class="mut">${esc(summaryText)}</span>`;
  d.appendChild(s);
  const body = document.createElement("div");
  body.className = "pad";
  const argsPretty = (c.args && Object.keys(c.args).length)
      ? JSON.stringify(c.args, null, 2)
      : (c.args_raw || "");
  body.insertAdjacentHTML("beforeend", `<div class="kv">参数</div><pre>${esc(argsPretty || "—")}</pre>`);
  if (c.result_summary)
    body.insertAdjacentHTML("beforeend", `<div class="kv">结果摘要</div><pre>${esc(c.result_summary)}</pre>`);
  if (c.error)
    body.insertAdjacentHTML("beforeend", `<div class="kv err">错误</div><pre>${esc(c.error)}</pre>`);
  if (c.result_text)
    body.insertAdjacentHTML("beforeend",
      `<details><summary>完整结果文本（${fmtInt(c.result_chars)} 字符）</summary>
        <pre class="full">${esc(c.result_text)}</pre></details>`);
  else if (c.result_text_preview)
    body.insertAdjacentHTML("beforeend",
      `<details><summary>结果预览（旧轨迹仅存预览）</summary>
        <pre class="full">${esc(c.result_text_preview)}</pre></details>`);
  if (Array.isArray(c.result_paths) && c.result_paths.length)
    body.insertAdjacentHTML("beforeend",
      `<details><summary>返回路径（${c.result_paths.length}）</summary>
        <pre class="full">${esc(c.result_paths.join("\n"))}</pre></details>`);
  if (Array.isArray(c.result_unresolved_paths) && c.result_unresolved_paths.length)
    body.insertAdjacentHTML("beforeend",
      `<details><summary>未解析路径（${c.result_unresolved_paths.length}）</summary>
        <pre class="full">${esc(c.result_unresolved_paths.join("\n"))}</pre></details>`);
  if (Array.isArray(c.result_docids) && c.result_docids.length)
    body.insertAdjacentHTML("beforeend",
      `<details><summary>返回 docid（${c.result_docids.length}）</summary>
        <pre class="full">${esc(c.result_docids.join("\n"))}</pre></details>`);
  d.appendChild(body);
  return d;
}

/* ---------------- turn search + jump ---------------- */
function turnMatch(tn, q){
  if ((tn.content || "").toLowerCase().includes(q)) return true;
  if ((tn.reasoning_content || "").toLowerCase().includes(q)) return true;
  for (const c of (Array.isArray(tn.tool_calls) ? tn.tool_calls : [])){
    if ((c.result_summary || "").toLowerCase().includes(q)) return true;
    if ((c.result_text_preview || "").toLowerCase().includes(q)) return true;
    if ((c.error || "").toLowerCase().includes(q)) return true;
    const a = (c.args && Object.keys(c.args).length) ? JSON.stringify(c.args) : (c.args_raw || "");
    if (a.toLowerCase().includes(q)) return true;
    if ((c.result_text || "").toLowerCase().includes(q)) return true;
  }
  return false;
}
function applyTurnSearch(){
  const raw = document.getElementById("tsearch").value.trim();
  const note = document.getElementById("tmatch");
  if (!raw){
    _turnEls.forEach(el => el.classList.remove("hide"));
    note.textContent = "";
    return;
  }
  const q = raw.toLowerCase();
  let hits = 0, lastHit = -1;
  _turnEls.forEach((el, i) => {
    const m = turnMatch(_turnsData[i], q);
    el.classList.toggle("hide", !m);
    const badge = el.querySelector(".tmatch");
    if (m){
      hits++;
      lastHit = i;
      badge.textContent = "命中";
      badge.style.display = "";
    } else badge.style.display = "none";
  });
  ensureTurnVisible(lastHit);   // render chunks through the last matching turn
  note.textContent = `${hits} 轮匹配`;
}
function ensureTurnVisible(i){
  if (i < 0) return;
  while (!_turnEls[i].parentElement && _turnsRendered <= i) renderTurnChunk(_turnsRendered);
}
function expandVisible(){
  let n = 0;
  for (let i = 0; i < _turnsData.length && n < EXPAND_CAP; i++){
    const el = _turnEls[i];
    if (el.classList.contains("hide")) continue;
    if (!el.parentElement){ if (n >= EXPAND_CAP) break; ensureTurnVisible(i); }
    if (!el.parentElement) continue;   // beyond the rendered frontier; skip safely
    el.open = true;
    buildTurn(el, i);
    n++;
  }
  if (n === EXPAND_CAP) document.getElementById("tmatch").textContent =
    `已展开前 ${EXPAND_CAP} 个可见轮次（大型轨迹请逐轮展开）`;
}
function jumpToTurn(){
  const n = parseInt(document.getElementById("jump").value, 10);
  if (isNaN(n)) return;
  const i = _turnsData.findIndex(tn => tn.turn === n);
  const idx = i === -1 ? (n >= 0 && n < _turnsData.length ? n : -1) : i;
  if (idx === -1) return;
  ensureTurnVisible(idx);
  const el = _turnEls[idx];
  el.open = true;
  buildTurn(el, idx);
  el.scrollIntoView({behavior: "smooth", block: "start"});
}

/* ---------------- hash routing ---------------- */
function parseHash(){
  const m = /^#\/run\/([0-9a-f]{12})(?:\/qid\/([^/]+))?$/.exec(location.hash);
  return m ? {rid: m[1], qid: m[2] ? decodeURIComponent(m[2]) : null} : {rid: null, qid: null};
}
function setHash(qid){
  const h = qid ? `#/run/${RID}/qid/${encodeURIComponent(qid)}` : `#/run/${RID}`;
  if (location.hash !== h) history.replaceState(null, "", h);
}

document.getElementById("qsearch").oninput = renderList;
document.getElementById("fStatus").onchange = renderList;
document.getElementById("fSort").onchange = renderList;
loadRuns().catch(err => {
  document.getElementById("main").innerHTML =
    `<div class="empty">初始化失败：${esc(err.message)}</div>`;
});
</script>
</body>
</html>
"""


def _pick_port(preferred: int) -> int:
    import socket
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"ports {preferred}-{preferred + 19} are all in use")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="RISE 轨迹阅读器：只读的本地 Web 轨迹浏览器（仅标准库）")
    ap.add_argument("--root", type=Path, action="append", default=None,
                    help="扫描根目录（可重复；默认为仓库根目录）")
    ap.add_argument("--run-dir", type=Path, action="append", default=None,
                    help="直接指定运行目录（可重复），不再自动扫描")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args(argv)

    global _REGISTRY
    pinned = [p for p in (args.run_dir or [])]
    if pinned:
        roots: list[Path] = []
        for p in pinned:
            if not _is_run_dir(p.resolve()):
                print(f"error: {p} 不是运行目录（缺少 _per_query 或 _traces）", file=sys.stderr)
                return 2
    else:
        roots = args.root or [REPO_ROOT]

    _REGISTRY = RunRegistry(roots, pinned)
    runs = _REGISTRY.scan()
    if not runs:
        print("error: 未发现任何运行目录（需包含 _per_query 或 _traces 子目录）", file=sys.stderr)
        return 2

    port = _pick_port(args.port) if args.host == "127.0.0.1" else args.port
    server = ThreadingHTTPServer((args.host, port), Handler)
    server.daemon_threads = True
    url = f"http://{args.host}:{port}/"
    print("RISE 轨迹阅读器（只读）")
    print(f"  地址: {url}")
    print(f"  已发现 {len(runs)} 个运行目录:")
    for d in runs.values():
        ov = run_overview(d)
        print(f"    - {ov['name']}  [{ov['relpath']}]  "
              f"{ov['n_rows']} 行 / {ov['n_traces']} 轨迹"
              + (f" / {ov['n_blocked']} 封锁" if ov['n_blocked'] else ""))
    print("  Ctrl+C 停止")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
