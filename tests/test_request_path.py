"""No compile, tune, self-check, build or CUDA graph capture blocks a request (owner rule). Every
call that does one sits under `request_path.forbid`, runs on a background thread, or is one of
`KNOWN`'s named violations (cr-210), so the source cannot grow an unguarded one; and through the
real executor, a guarded self-check on the request thread is the run's `request_path.blocking`
while the same one on a background thread during the request is nothing.
"""

from __future__ import annotations

import ast
import re
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import pytest

import cozy_runtime
from cozy_runtime.author import _request_path as request_path
from cozy_runtime.internal import kernel_cache, kernel_compile, package_interface, verify
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, PrepareRequest, Start
from cozy_runtime.internal.seam import Channel

SRC = Path(cozy_runtime.__file__).parent
#: Calls that compile, tune or capture, by their dotted name's tail.
PRIMITIVES = (
    "triton.compile",
    "cute.compile",
    "torch.compile",
    "cpp_extension.load",
    "cpp_extension.load_inline",
    "do_bench",
    "do_bench_cudagraph",
    "capture_begin",
    "cuda.graph",
    "make_graphed_callables",
    "cozy_runtime.graphs.capture",
)
#: A process the source names as a compiler runs one.
COMPILERS = re.compile(r"^(nvcc|ninja|ptxas|cmake|gcc|g\+\+|clang\+*|c\+\+|cc|setup\.py)$")
#: The Runtime's own blocking kernel work, besides each builder `kernel_compile` runs.
OWN = {
    "cozy_runtime.internal.kernel_compile.wait",
    "cozy_runtime.internal.kernel_compile.run",
    "cozy_runtime.internal.verify.verified",
    "cozy_runtime.internal.lora_bake._ready",
    "cozy_runtime.internal.attention_sol_fa4.warm",
    "cozy_runtime.graphs.capture",
}


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


class _Scan(ast.NodeVisitor):
    """One module's primitive calls, whether each is covered, its guards and its known sites."""

    def __init__(self, module: str, tree: ast.Module) -> None:
        self.module, self.scope, self.covered = module, [module], 0
        self.names = {node.name: module for node in tree.body if isinstance(node, ast.FunctionDef)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.names.update({a.asname or a.name: node.module for a in node.names})
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        handed = {
            _dotted(arg)
            for node in calls
            for arg in (
                [k.value for k in node.keywords if k.arg == "target"]
                + (node.args[:1] if _dotted(node.func).endswith(".submit") else [])
            )
        }
        #: functions only ever handed to another thread: what they do runs beside the request
        self.background = {n.rpartition(".")[2] for n in handed if "" not in n.split(".")} - {
            _dotted(node.func).rpartition(".")[2] for node in calls
        }
        self.calls: list[tuple[str, int, str, bool]] = []
        self.guarded: set[str] = set()
        self.known: list[tuple[str, str, int]] = []

    def resolve(self, node: ast.expr) -> str:
        name = _dotted(node)
        head, _, rest = name.partition(".")
        origin = self.names.get(head)
        return f"{origin}.{name}" if origin else name

    def guard(self, node: ast.expr) -> str:
        name = _dotted(node.func) if isinstance(node, ast.Call) else ""
        return name.removeprefix("request_path.") if name.startswith("request_path.") else ""

    def _def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        decorated = any(self.guard(d) == "forbid" for d in node.decorator_list)
        if decorated:
            self.guarded.add(".".join(self.scope))
        covers = decorated or node.name in self.background
        self.covered += covers
        self.generic_visit(node)
        self.covered -= covers
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _def

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_With(self, node: ast.With) -> None:
        here, covers = ".".join(self.scope), False
        for item in node.items:
            kind, call = self.guard(item.context_expr), item.context_expr
            if kind == "forbid":
                self.guarded.add(here)
                covers = True
            elif kind == "known" and isinstance(call, ast.Call):
                site = ast.literal_eval(call.args[0])
                self.known.append((site, here, node.lineno))
                covers = covers or (site == here and site in request_path.KNOWN)
            self.visit(item)
        self.covered += covers
        for statement in node.body:
            self.visit(statement)
        self.covered -= covers

    def visit_Call(self, node: ast.Call) -> None:
        name = self.resolve(node.func)
        literals = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)]
        compiler = name.startswith("subprocess.") and any(
            isinstance(v, str) and COMPILERS.match(Path(v).name) for v in literals
        )
        if compiler or any(name == p or name.endswith("." + p) for p in PRIMITIVES):
            self.calls.append((".".join(self.scope), node.lineno, name, self.covered > 0))
        self.generic_visit(node)


def _scans() -> list[tuple[Path, _Scan]]:
    scans = []
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        tree = ast.parse(path.read_text())
        scan = _Scan(module, tree)
        scan.visit(tree)
        scans.append((path, scan))
    return scans


def test_every_compile_tune_capture_or_compiler_call_is_guarded_backgrounded_or_known() -> None:
    scans = _scans()
    unguarded = [
        f"{path.relative_to(SRC.parent)}:{line}: {where} calls {name} on whatever thread calls "
        "it: guard it with `request_path.forbid`, or run it on a background thread"
        for path, scan in scans
        for where, line, name, covered in scan.calls
        if not covered
    ]
    assert unguarded == []
    found = {name for _, scan in scans for _, _, name, _ in scan.calls}
    assert {"triton.compile", "cute.compile", "cozy_runtime.graphs.capture"} <= found, found


def test_the_runtimes_own_builds_waits_and_self_checks_each_hold_the_guard() -> None:
    builders = {f"{m}.{f}" for m, f in (v.split(":") for v in kernel_compile.BUILDERS.values())}
    guarded = set().union(*(scan.guarded for _, scan in _scans()))
    assert sorted((OWN | builders) - guarded) == []


def test_the_known_violations_are_exactly_cr_210s_two_captures_each_marked_where_it_runs() -> None:
    assert request_path.KNOWN == {
        "cozy_runtime.graphs._record": "cr-210",
        "cozy_runtime.lm._Engine._replay": "cr-210",
    }
    marked = sorted((site, here) for _, scan in _scans() for site, here, _ in scan.known)
    assert marked == [(site, site) for site in sorted(request_path.KNOWN)]


MODULE = "request_path_fixture"
SOURCE = """
import threading
import time

import msgspec
from cozy_runtime.author import App, Context, Telemetry
from cozy_runtime.internal import verify

app = App()


class Request(msgspec.Struct):
    background: bool = False


class Result(msgspec.Struct):
    verdict: str


def probe() -> verify.Verdict:
    time.sleep(0.05)
    return verify.Verdict(True, "probed")


@app.entrypoint
def generate(ctx: Context, payload: Request, tel: Telemetry) -> Result:
    verdicts = []

    def check() -> None:
        verdicts.append(verify.verified("request-path-fixture", ctx.request_id, 0, probe))

    if payload.background:
        beside = threading.Thread(target=check)
        beside.start()
        tel.log("the request goes on while it checks")
        beside.join()
    else:
        check()
    return Result(verdicts[0].detail)
"""


@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Executor]:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject = "{MODULE}:app"\n')
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(
        package_interface.canonical_bytes(package_interface.build(discover(project)))
    )
    binding = {"application": f"{MODULE}:app", "package_interface": str(interface)}
    left, right = socket.socketpair()
    with left, right:
        host = Executor(Channel(left), tmp_path / "root")
        (tmp_path / "root").mkdir()
        assert host.start(msgspec.convert({**binding, "devices": ""}, Start))["ok"]
        load = {"devices": "", "binding": binding, "budgets": {}, "construction": "a"}
        assert host.load(msgspec.convert(load, Load))["ok"]
        assert host.activate(Activate(construction="a"))["ok"]
        # The machine's store: a self-check keeps its unfinished-probe marker there.
        monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [kernel_cache.Store(tmp_path / "u")])
        yield host
    sys.modules.pop(MODULE, None)


def _run(host: Executor, request_id: str, background: bool) -> dict[str, Any]:
    payload = {"background": background}
    prepared = host.serve_request(
        PrepareRequest(request_id=request_id, entrypoint="generate", payload=payload)
    )
    assert prepared["ok"], prepared
    spool = host.root / request_id
    spool.mkdir()
    reply = host.serve_invoke(
        Invoke(request_id=request_id, entrypoint="generate", spool=str(spool), deadline_s=30)
    )
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    return reply


def _blocking(reply: dict[str, Any]) -> list[tuple[str, str, object]]:
    return [
        (row["kind"], row["value"], row["fields"].get("work"))
        for row in reply["observations"]
        if row["name"] in ("request_path.blocking", "kernel work blocked the request")
    ]


def test_a_self_check_on_the_request_thread_blocks_it_and_one_beside_it_does_not(
    executor: Executor,
) -> None:
    work = "kernel self-check: request-path-fixture"
    on_request = _run(executor, "on-request", background=False)
    assert _blocking(on_request) == [("confession", work, None), ("log", "warn", work)]
    [row] = [r for r in on_request["observations"] if r["name"] == "request_path.blocking"]
    assert row["fields"]["times"] == 1 and row["fields"]["ms"] >= 50, row
    beside = _run(executor, "beside", background=True)
    assert _blocking(beside) == []
    for request_id in ("on-request", "beside"):  # both really probed, once each
        assert [k for k in verify._KNOWN if k[1] == request_id] != []
