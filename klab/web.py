"""`klab web`:本地的「算子 LeetCode」。左边题面与讲解,右边写算子,Run / Submit 出体检单。

布局对标 LeetCode:
    左  Description(题面)/ Editorial(优化路线)/ Solutions(同题的其他语言实现)/ Submissions(提交历史)/ Result(体检单)
    右  语言下拉(工具链)+ 后端下拉 + Monaco 编辑器 + case 选择 + Run / Submit

概念映射:
    一道题 = problems/<problem>/(题面 + 讲解 + 各语言骨架);meta.toml 的 problem 键把多个算子聚成一道题
    Run    = klab check(秒级,只看对不对)
    Submit = klab run  (check → bench → ncu → 体检单,约一分钟),并把当次源码快照存进 runs/<id>/submission/

边界没变:只读 `runs/` 与仓库里的源码、只 fork `klab` 子进程,不引数据库、不引 web 框架。
只监听 127.0.0.1;算子名 / 后端名 / case 名一律先过白名单才进子进程命令行,子进程不走 shell。
"""
from __future__ import annotations

import errno
import html
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from klab.config import default_target, load_targets
from klab.harness.spec import KernelSpec

MODES = {"run": "run", "check": "check", "bench": "bench", "ncu": "ncu", "ptx": "ptx"}

# 工具链 -> (编辑器语法, 主源文件名)。cuda / tk 写 .cu,其余写 .py。
TOOLCHAIN_LANG = {
    "triton": ("python", "kernel.py"),
    "tilelang": ("python", "kernel.py"),
    "cute": ("python", "kernel.py"),
    "cuda": ("cpp", "kernel.cu"),
    "tk": ("cpp", "kernel.cu"),
}

WEBUI = Path(__file__).with_name("webui")
MONACO = WEBUI / "vendor" / "monaco"
MONACO_VERSION = "0.56.0"

# ---------------------------------------------------------------- markdown 子集

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_SEP = re.compile(r"^:?-{2,}:?$")


def _inline(s: str) -> str:
    return _BOLD.sub(r"<strong>\1</strong>", _CODE.sub(r"<code>\1</code>", html.escape(s)))


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(md: str) -> str:
    """kreport 与 problems/ 用的 markdown 子集 -> HTML:标题、段落、GFM 表格、无序/有序列表、围栏代码块、粗体、行内代码。"""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            lang = s[3:].strip()
            i += 1
            body = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            out.append(f'<pre class="code" data-lang="{html.escape(lang)}"><code>{html.escape(chr(10).join(body))}</code></pre>')
        elif s.startswith("#"):
            n = min(len(s) - len(s.lstrip("#")), 6)
            out.append(f"<h{n}>{_inline(s[n:].strip())}</h{n}>")
            i += 1
        elif s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            head, *rest = block
            if rest and all(_SEP.match(c) for c in _cells(rest[0]) if c):
                rest = rest[1:]
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in _cells(head)) + "</tr></thead><tbody>")
            out += ["<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in _cells(r)) + "</tr>" for r in rest]
            out.append("</tbody></table>")
        elif s.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].strip().startswith("- "):
                out.append(f"<li>{_inline(lines[i].strip()[2:])}</li>")
                i += 1
            out.append("</ul>")
        elif re.match(r"^\d+\. ", s):
            out.append("<ol>")
            while i < len(lines) and re.match(r"^\d+\. ", lines[i].strip()):
                out.append(f"<li>{_inline(lines[i].strip().split('. ', 1)[1])}</li>")
                i += 1
            out.append("</ol>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", "|", "- ", "```")) \
                    and not re.match(r"^\d+\. ", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


# ---------------------------------------------------------------- 仓库状态

def _parse_run(run_dir: Path) -> dict | None:
    """runs/<时间>-<算子>-<后端>-<模式>/。算子名与后端名都可能含 '-',靠 result.json 里的 kernel 反推后端。"""
    rj = run_dir / "result.json"
    if not rj.is_file():
        return None
    try:
        d = json.loads(rj.read_text())
    except Exception:
        return None
    kernel, mode = d.get("kernel", ""), d.get("mode", "")
    mid = run_dir.name[len("20000101-000000-"):]
    if not (kernel and mode and mid.startswith(kernel + "-") and mid.endswith("-" + mode)):
        return None
    res = d.get("results", [])
    best = next((r for r in res if "speedup_vs_ref" in r), None)
    return {
        "id": run_dir.name,
        "when": run_dir.name[:15],
        "kernel": kernel,
        "target": mid[len(kernel) + 1: -len(mode) - 1],
        "mode": mode,
        "device": d.get("device", {}).get("device", ""),
        "has_report": (run_dir / "report.md").is_file(),
        "has_code": (run_dir / "submission").is_dir(),
        "ok": all(r.get("ok", True) for r in res),
        "speedup": best.get("speedup_vs_ref") if best else None,
        "tflops": best.get("tflops") if best else None,
        "case": best.get("case") if best else (res[0].get("case") if res else None),
    }


def list_runs(root: Path, limit: int = 80) -> list[dict]:
    runs = []
    for p in sorted((root / "runs").glob("*/"), reverse=True):
        r = _parse_run(p)
        if r:
            runs.append(r)
        if len(runs) >= limit:
            break
    return runs


def primary_source(root: Path, name: str, toolchain: str) -> Path:
    """编辑器编辑的那个文件。按工具链定主文件名;真不存在就退回目录里第一个源码文件。"""
    kdir = root / "kernels" / name
    _, fname = TOOLCHAIN_LANG.get(toolchain, ("python", "kernel.py"))
    p = kdir / fname
    if p.is_file():
        return p
    if kdir.is_dir():
        for q in sorted(kdir.iterdir()):
            if q.is_file() and q.suffix in (".py", ".cu", ".cuh", ".cpp"):
                return q
    return p


def list_kernels(root: Path) -> list[dict]:
    out = []
    spec_names = set()
    for sd in sorted((root / "specs").iterdir() if (root / "specs").is_dir() else []):
        if not (sd / "meta.toml").is_file():
            continue
        spec_names.add(sd.name)
        try:
            spec = KernelSpec.load(sd)
        except Exception as e:
            out.append({"name": sd.name, "problem": sd.name, "ready": False, "note": f"meta.toml 读不了:{e}"})
            continue
        src = primary_source(root, sd.name, spec.toolchain)
        lang, _ = TOOLCHAIN_LANG.get(spec.toolchain, ("python", "kernel.py"))
        out.append({
            "name": sd.name,
            "problem": spec.problem,
            "ready": src.is_file(),
            "toolchain": spec.toolchain,
            "language": lang,
            "source": src.name,
            "cases": [c["name"] for c in spec.cases],
            "features": spec.features,
            "note": "" if src.is_file() else f"kernels/{sd.name}/{src.name} 不存在",
        })
    for kd in sorted((root / "kernels").iterdir() if (root / "kernels").is_dir() else []):
        if kd.is_dir() and kd.name not in spec_names:
            out.append({"name": kd.name, "problem": kd.name, "ready": False,
                        "note": "还没接线:specs/ 下没有它,让 agent 按 playbook 生成"})
    return out


def list_problems(root: Path) -> list[dict]:
    """按 meta.toml 的 problem 键把算子聚成题。一道题 = 一个数学定义 + 若干语言的实现。"""
    by: dict[str, list[dict]] = {}
    for k in list_kernels(root):
        by.setdefault(k["problem"], []).append(k)
    out = []
    for name, impls in sorted(by.items()):
        pdir = root / "problems" / name
        out.append({
            "name": name,
            "title": _problem_title(pdir) or name,
            "impls": sorted(impls, key=lambda k: k["name"]),
            "has_doc": (pdir / "problem.md").is_file(),
            "ready": any(k.get("ready") for k in impls),
        })
    return out


def _problem_title(pdir: Path) -> str:
    p = pdir / "problem.md"
    if not p.is_file():
        return ""
    for line in p.read_text().splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def state(root: Path) -> dict:
    cfgs = load_targets(root)
    return {
        "repo": str(root),
        "problems": list_problems(root),
        "targets": [{"name": n, "kind": c.kind, "gpu": c.gpu, "host": c.host} for n, c in cfgs.items()],
        "default_target": default_target(root),
        "runs": list_runs(root),
    }


def backbone_for(root: Path, problem: str, toolchain: str) -> str:
    """骨架:契约(常量 + 函数签名)给全,函数体留空。面板的「重置」恢复到它。"""
    _, fname = TOOLCHAIN_LANG.get(toolchain, ("python", "kernel.py"))
    t = root / "problems" / problem / "backbone" / f"{toolchain}{Path(fname).suffix}"
    return t.read_text() if t.is_file() else ""


def header_doc(code: str, suffix: str) -> list[str]:
    """参考答案开头那段说明文字。

    `.py` 取模块 docstring;`.cu` 取开头连续的 `//` 注释块 —— C++ 里没有 docstring,
    只认 docstring 的话 cuda 与 tk 的阶梯在面板里就只剩一个文件名。
    """
    if suffix == ".py":
        m = re.match(r'\s*"""(.*?)"""', code, re.S)
        return m.group(1).splitlines() if m else []
    out: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped.startswith("//"):
            break
        out.append(stripped[2:].removeprefix(" "))
    return out


def body_of(code: str, suffix: str) -> str:
    """去掉开头那段讲解之后的代码本体,供「与上一级的差异」比对。

    阶梯每一级都会重写开头的讲解,连着比的话 diff 满屏都是散文,真正改动的那几行反而被埋掉 ——
    而阶梯的卖点恰恰是「每级只改一处」。
    """
    if suffix == ".py":
        m = re.match(r'\s*""".*?"""', code, re.S)
        rest = code[m.end():] if m else code
    else:
        lines = code.splitlines(keepends=True)
        i = 0
        while i < len(lines) and lines[i].strip().startswith("//"):
            i += 1
        rest = "".join(lines[i:])
    return rest.lstrip("\n")


def solutions_for(root: Path, problem: str, toolchain: str) -> list[dict]:
    """参考答案阶梯:problems/<题>/solutions/<工具链>/<序号>-<名字>.<后缀>。

    文件名的序号决定顺序;开头说明的第一行是标题,其余是讲解(见 header_doc)。
    """
    d = root / "problems" / problem / "solutions" / toolchain
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if not f.is_file() or f.suffix not in (".py", ".cu"):
            continue
        code = f.read_text()
        doc = [ln.rstrip() for ln in header_doc(code, f.suffix)]
        while doc and not doc[0].strip():
            doc.pop(0)
        title = doc[0].strip() if doc else f.stem
        note = "\n".join(doc[1:]).strip()
        out.append({"id": f.stem, "title": title, "note": note, "code": code,
                    "body": body_of(code, f.suffix)})
    return out


# ---------------------------------------------------------------- 跑子进程

_lock = threading.Lock()  # 后端就一张卡,并发跑出来的数字没有意义


def _snapshot(root: Path, kernel: str, new_dirs: list[Path]) -> None:
    """把这次提交用的源码原样存进新产生的 run 目录,让「204 TFLOPS 对应哪份代码」可回溯。"""
    kdir = root / "kernels" / kernel
    if not kdir.is_dir():
        return
    for d in new_dirs:
        dst = d / "submission"
        dst.mkdir(exist_ok=True)
        for f in kdir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)


def stream_klab(root: Path, kernel: str, target: str, mode: str, cases: list[str]):
    """yield 子进程输出;结束后打一行 __KLAB_RUN__ <run_id> 让前端知道读哪次结果。"""
    before = {p.name for p in (root / "runs").glob("*/")}
    cmd = [sys.executable, "-m", "klab.cli", MODES[mode], kernel, "--target", target]
    for c in cases:
        cmd += ["--case", c]
    env = {**os.environ, "COLUMNS": "150", "TERM": "dumb", "NO_COLOR": "1"}
    yield f"$ klab {MODES[mode]} {kernel} --target {target}{''.join(' --case ' + c for c in cases)}\n\n"
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    code = proc.wait()
    new = [p for p in (root / "runs").glob("*/") if p.name not in before]
    _snapshot(root, kernel, new)
    latest = next((r["id"] for r in list_runs(root) if r["kernel"] == kernel and r["target"] == target), "-")
    yield f"\n[klab] 退出码 {code}\n__KLAB_RUN__ {latest}\n"


# ---------------------------------------------------------------- 编辑器取用

def ensure_monaco(quiet: bool = False) -> bool:
    """Monaco 是 24 MB 的预编译产物,gitignore 掉,首次用时从 npm registry 取(照 ThunderKittens 的先例)。"""
    if (MONACO / "vs" / "loader.js").is_file():
        return True
    url = f"https://registry.npmjs.org/monaco-editor/-/monaco-editor-{MONACO_VERSION}.tgz"
    if not quiet:
        print(f"首次运行:取编辑器 monaco-editor {MONACO_VERSION}(MIT,约 24 MB)…", flush=True)
    try:
        buf = io.BytesIO(urllib.request.urlopen(url, timeout=180).read())
        with tarfile.open(fileobj=buf, mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.startswith("package/min/vs/"):
                    out = MONACO / m.name[len("package/min/"):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(tf.extractfile(m).read())
            lic = next((m for m in tf.getmembers() if m.name == "package/LICENSE"), None)
            if lic:
                (MONACO / "LICENSE").write_bytes(tf.extractfile(lic).read())
    except Exception as e:
        print(f"取编辑器失败({e});页面会退回纯文本编辑框。手动装:见 README「本地面板」一节", flush=True)
        return False
    return True


# ---------------------------------------------------------------- HTTP

_STATIC_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
                 ".json": "application/json", ".map": "application/json", ".ttf": "font/ttf",
                 ".svg": "image/svg+xml", ".woff": "font/woff", ".woff2": "font/woff2"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root: Path = Path(".")

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", code)

    def _chunked(self, chunks) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for c in chunks:
                b = c.encode()
                self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # 页面关了,子进程仍会跑完

    def _static(self, rel: str) -> None:
        p = (WEBUI / rel).resolve()
        if not str(p).startswith(str(WEBUI.resolve())) or not p.is_file():
            return self._send(b"not found", "text/plain", 404)
        self._send(p.read_bytes(), _STATIC_TYPES.get(p.suffix, "application/octet-stream") + "; charset=utf-8")

    # ---- 白名单 ----
    def _kernel(self, name: str) -> dict | None:
        return next((k for k in list_kernels(self.root) if k["name"] == name and k.get("ready")), None)

    # ---- 路由 ----
    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if path == "/":
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/api/state":
            return self._json(state(self.root))
        if path == "/api/problem":
            name = q.get("name", [""])[0]
            if not name or "/" in name or ".." in name:
                return self._json({"error": "bad name"}, 400)
            pdir = self.root / "problems" / name
            doc = pdir / "problem.md"
            ed = pdir / "editorial.md"
            return self._json({
                "name": name,
                "description": md_to_html(doc.read_text()) if doc.is_file() else
                    f"<p class='empty'>还没有题面。让 agent 写 <code>problems/{html.escape(name)}/problem.md</code>。</p>",
                "editorial": md_to_html(ed.read_text()) if ed.is_file() else
                    f"<p class='empty'>还没有优化路线。让 agent 写 <code>problems/{html.escape(name)}/editorial.md</code>。</p>",
            })
        if path == "/api/source":
            k = self._kernel(q.get("kernel", [""])[0])
            if not k:
                return self._json({"error": "未知或没接线的算子"}, 404)
            src = primary_source(self.root, k["name"], k["toolchain"])
            return self._json({"kernel": k["name"], "path": f"kernels/{k['name']}/{src.name}",
                               "language": k["language"], "code": src.read_text(),
                               "backbone": backbone_for(self.root, k["problem"], k["toolchain"])})
        if path == "/api/solutions":
            k = self._kernel(q.get("kernel", [""])[0])
            if not k:
                return self._json({"error": "未知或没接线的算子"}, 404)
            return self._json({"toolchain": k["toolchain"], "language": k["language"],
                               "solutions": solutions_for(self.root, k["problem"], k["toolchain"])})
        if path == "/api/report":
            rid = q.get("run", [""])[0]
            md = self.root / "runs" / rid / "report.md"
            if not rid or ".." in rid or "/" in rid or not md.is_file():
                return self._json({"error": "这次运行没有体检单(只有 Submit / ncu 会生成)"}, 404)
            return self._json({"run": rid, "html": md_to_html(md.read_text())})
        if path == "/api/submission":
            rid = q.get("run", [""])[0]
            d = self.root / "runs" / rid / "submission"
            if not rid or ".." in rid or "/" in rid or not d.is_dir():
                return self._json({"error": "这次提交没有存代码快照"}, 404)
            return self._json({"run": rid, "files": {f.name: f.read_text() for f in sorted(d.iterdir()) if f.is_file()}})
        if path == "/api/history":
            from klab.compare import collect
            ks = q.get("kernel") or None
            return self._json({"rows": collect(self.root, ks, None, True)})
        return self._json({"error": "no such route"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if path == "/api/save":
            return self._json(self._save(body))
        if path != "/api/run":
            return self._json({"error": "no such route"}, 404)

        k = self._kernel(body.get("kernel", ""))
        target, mode = body.get("target", ""), body.get("mode", "run")
        if not k:
            return self._json({"error": f"未知或没接线的算子 {body.get('kernel')!r}"}, 400)
        if target not in {t["name"] for t in state(self.root)["targets"]}:
            return self._json({"error": f"targets.toml 里没有后端 {target!r}"}, 400)
        if mode not in MODES:
            return self._json({"error": f"未知模式 {mode!r}"}, 400)
        cases = [c for c in body.get("cases", []) if c in k["cases"]]  # 只认 meta.toml 里有的 case
        if "code" in body:
            r = self._save({"kernel": k["name"], "code": body["code"]})
            if "error" in r:
                return self._json(r, 400)
        if not _lock.acquire(blocking=False):
            return self._json({"error": "后端正忙:一次只跑一个,等当前这次跑完"}, 409)
        try:
            self._chunked(stream_klab(self.root, k["name"], target, mode, cases))
        finally:
            _lock.release()

    def _save(self, body: dict) -> dict:
        k = self._kernel(body.get("kernel", ""))
        if not k:
            return {"error": f"未知或没接线的算子 {body.get('kernel')!r}"}
        code = body.get("code")
        if not isinstance(code, str):
            return {"error": "缺 code"}
        src = primary_source(self.root, k["name"], k["toolchain"])
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(code)
        return {"saved": f"kernels/{k['name']}/{src.name}", "bytes": len(code.encode())}


def _listener_pid(port: int) -> int | None:
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True, timeout=5).stdout.splitlines()
        return int(out[1].split()[1]) if len(out) > 1 else None
    except Exception:
        return None


def _is_our_panel(port: int) -> bool:
    """端口上蹲着的是不是我们自己的面板 —— 只有确认是,才敢杀。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3) as r:
            return "repo" in json.loads(r.read().decode())
    except Exception:
        return False


def _take_over(port: int) -> bool:
    """旧面板跑的是它启动时的代码,改完必须重起。默认自动替掉它,省得每次手动 kill。"""
    if not _is_our_panel(port):
        return False
    pid = _listener_pid(port)
    if not pid or pid == os.getpid():
        return False
    print(f"停掉旧面板(PID {pid})…", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    for _ in range(50):                      # 最多等 5 秒让端口释放
        time.sleep(0.1)
        if _listener_pid(port) is None:
            return True
    return False


def _port_taken(port: int) -> str:
    pid = _listener_pid(port)
    who = f"(PID {pid},`kill {pid}` 停掉它)" if pid else ""
    return (
        f"端口 {port} 被别的程序占着{who} —— 不是 klab 面板,所以没敢自动停。\n"
        f"要么停掉它,要么换个端口:uv run klab web --port {port + 1}"
    )


def serve(root: Path, port: int = 8777, open_browser: bool = True, restart: bool = True) -> None:
    Handler.root = root
    srv = None
    for attempt in (1, 2):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)  # 先占端口再取 Monaco,免得白下 24 MB
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            if attempt == 1 and restart and _take_over(port):
                continue
            raise SystemExit(_port_taken(port)) from None
    assert srv is not None
    ensure_monaco()
    url = f"http://127.0.0.1:{port}"
    print(f"kernellab 面板 {url}(Ctrl-C 退出)")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n再见")
    finally:
        srv.server_close()
