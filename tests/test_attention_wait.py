"""A run never waits for a compile: a site whose kernel still compiles serves the best one
ready, and the construction takes the kernel at its next selection once it is built, which
is the run's next step. Real Diffusers attention sites, a real source registry, real builder
processes and the real store; the kernel is a registry wheel a CPU can build, named in a
model's ladder and in the ranked table.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import socket
import sys
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from cozy_runtime.internal import (
    attention,
    attention_upstream,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    liveness,
    package_interface,
)
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import (
    Activate,
    Binding,
    Invoke,
    Load,
    PrepareRequest,
    Start,
)
from cozy_runtime.internal.seam import Channel
from test_kernel_compile import _held, _registry

CPU = DeviceFacts(kind="cpu", name="cpu", sm=0, driver="", configuration="", index=0)
TINY = attention.Candidate(
    name="tiny",
    backend="_cozy_tiny",
    local=attention_upstream.SDPA,
    min_sm=0,
    dtypes=frozenset(),
    max_head_dim=0,
    artifact=kernel_sources.PYTHON,
    import_name="tiny_kernel",
)
#: A live builder that keeps moving bytes for as long as a test holds its key.
MOVING = "import os, time\nfor _ in range(3000): os.write(2, b'.'); time.sleep(0.02)"


def _skip() -> str:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    try:
        installed = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "diffusers is not installed"
    return "" if installed == "0.40.0" else f"diffusers is {installed}, the proof needs 0.40.0"


pytestmark = pytest.mark.skipif(bool(_skip()), reason=_skip())


def _sites() -> dict[str, object]:
    import torch
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    sites = [
        WanAttention(dim=64, heads=2, dim_head=32, processor=WanAttnProcessor()) for _ in range(2)
    ]
    return {"dit": torch.nn.ModuleList(sites)}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[kernel_cache.Store]:
    """A machine run's store and scratch scope, and `tiny` known to selection."""
    scope = tmp_path / "run"
    scope.mkdir()
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(
        kernel_sources, "job", partial(kernel_sources.job, root=tmp_path / "sources")
    )
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", 0.05)
    monkeypatch.setattr(kernel_compile, "POLL_SECONDS", 0.01)
    monkeypatch.setitem(attention.BY_NAME, "tiny", TINY)
    for table in ("_READY", "_BROKEN", "_STATES"):
        monkeypatch.setattr(attention, table, {})
    found = kernel_cache.Store(tmp_path / "u1")
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [found])
    path = list(sys.path)
    yield found
    sys.path[:] = path
    sys.modules.pop("tiny_kernel", None)
    importlib.invalidate_caches()


def _ladder(*names: str) -> Any:
    return lambda context: names


def _built(store: kernel_cache.Store, job: kernel_compile.Job) -> None:
    kernel_compile.submit(store, job)  # nobody builds it now: this starts the build
    while kernel_compile.status(store, job).state != "ready":
        time.sleep(kernel_compile.POLL_SECONDS)


@pytest.mark.parametrize("choose", [_ladder("tiny", "sdpa"), None], ids=["ladder", "ranked"])
def test_a_site_serves_what_is_ready_while_its_kernel_compiles_and_takes_it_once_built(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choose: Any
) -> None:
    _registry(tmp_path / "sources")
    monkeypatch.setattr(attention, "CANDIDATES", (TINY, attention.BY_NAME["sdpa"]))
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    try:
        first = attention.select(CPU, _sites(), choose=choose)
        # Selected while the build still runs: nothing waited for it.
        assert kernel_compile.status(store, job).state == "compiling"
        assert first.hosts == {"dit": {"sdpa": 2}} and first.waiting == ("tiny",), first
        assert not attention.compiled(first.waiting, CPU)
    finally:
        os.close(lock)
        builder.kill()
        builder.wait()
    _built(store, job)
    assert attention.compiled(first.waiting, CPU)
    later = attention.select(CPU, _sites(), choose=choose)
    assert later.hosts == {"dit": {"tiny": 2}} and later.waiting == (), later


def test_a_kernel_whose_compile_failed_is_skipped_with_why_and_the_next_one_serves(
    store: kernel_cache.Store, tmp_path: Path
) -> None:
    _registry(tmp_path / "sources", corrupt=True)
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    kernel_compile.submit(store, job)
    while kernel_compile.status(store, job).state != "failed":
        time.sleep(kernel_compile.POLL_SECONDS)
    applied = attention.select(CPU, _sites(), choose=_ladder("tiny", "sdpa"))
    assert applied.hosts == {"dit": {"sdpa": 2}} and applied.waiting == ()
    assert "compile_source_mismatch" in (applied.skipped or {})["tiny"], applied.skipped
    assert attention.evidence(["sdpa"])[1]["state"] == "failed"


#: A package whose four steps run the construction's attention sites, and whose kernel build
#: ends during step 0 (`FINISH`, the test's).
STEPS = """
import msgspec
import torch
from cozy_runtime.author import App, Context, Telemetry

app = App()
SITES, FINISH, HELD = [], [], []


class Request(msgspec.Struct):
    steps: int = 4


class Result(msgspec.Struct):
    steps: int


@app.entrypoint
def generate(ctx: Context, payload: Request, tel: Telemetry) -> Result:
    on_step = tel.step_callback(payload.steps, stage="denoise")
    for step in range(payload.steps):
        with torch.no_grad():
            for site in SITES:
                site(torch.ones(1, 4, 64))
        HELD.append([site.processor._attention_backend.value for site in SITES])
        if step == 0:
            FINISH.pop()()
        on_step(step)
    return Result(payload.steps)
"""


def test_a_compile_that_finishes_mid_run_serves_the_rest_of_the_run(
    store: kernel_cache.Store, tmp_path: Path
) -> None:
    """Owner ruling 2026-10-09: a run whose preferred kernel finishes compiling partway
    through switches to it at the next step. The real executor runs the package's steps on
    the construction's sites; the evidence names which steps served which kernel."""
    _registry(tmp_path / "sources")
    project = tmp_path / "project"
    project.mkdir()
    (project / "steps_fixture.py").write_text(STEPS)
    (project / "package.toml").write_text('[application]\nobject = "steps_fixture:app"\n')
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(
        package_interface.canonical_bytes(package_interface.build(discover(project)))
    )
    binding = {"application": "steps_fixture:app", "package_interface": str(interface)}
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)

    def finish() -> None:  # the build that was running ends; the real builder publishes it
        os.close(lock)
        builder.kill()
        builder.wait()
        _built(store, job)

    left, right = socket.socketpair()
    with left, right:
        host = Executor(Channel(left), tmp_path / "root")
        (tmp_path / "root").mkdir()
        assert host.start(msgspec.convert({**binding, "devices": ""}, Start))["ok"]
        load = {"devices": "", "binding": binding, "budgets": {}, "construction": "a"}
        assert host.load(msgspec.convert(load, Load))["ok"]
        assert host.activate(Activate(construction="a"))["ok"]
        construction, roots = host.constructions["a"], _sites()
        construction.backend, construction.device_facts = SimpleNamespace(roots=roots), CPU
        construction._reselect = partial(
            construction._prepare_attention,
            Binding(model_parameter_name="model"),
            roots,
            "",
            CPU,
            choose=_ladder("tiny", "sdpa"),
        )
        construction.attention = construction._reselect()
        construction.attention_defaults = attention.snapshot(roots)
        assert construction.attention.waiting == ("tiny",)
        package = sys.modules["steps_fixture"]
        package.SITES[:] = list(roots["dit"])
        held: list[list[str]] = package.HELD

        def run(request_id: str) -> dict[str, Any]:
            held.clear()
            assert host.serve_request(
                PrepareRequest(request_id=request_id, entrypoint="generate", payload={})
            )["ok"]
            spool = host.root / request_id
            spool.mkdir()
            reply = host.serve_invoke(
                Invoke(
                    request_id=request_id, entrypoint="generate", spool=str(spool), deadline_s=60
                )
            )
            assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
            [rank] = reply["execution"]["ranks"]
            return dict(rank["attention"])

        package.FINISH[:] = [finish]
        switched = run("switched")
        tiny, sdpa = ["_cozy_tiny"] * 2, ["_cozy_sdpa"] * 2
        assert held == [sdpa, tiny, tiny, tiny]
        segments = [{k: v for k, v in row.items() if k != "at_us"} for row in switched["segments"]]
        assert segments == [
            {"stage": "", "step": 0, "observed": "sdpa", "dense": ""},
            {"stage": "denoise", "step": 1, "observed": "tiny", "dense": ""},
        ]
        assert switched["observed"] == "dit=sdpa+tiny"
        assert {"sdpa", "tiny"} <= {impl.split(":")[0] for impl in switched["impl"].split(",")}
        served = {row["kernel"]: row for row in switched["kernels"]}
        assert served["tiny"]["served"] and served["tiny"]["state"] == "ready"
        # The construction keeps the kernel: the next run serves it from its first step.
        assert construction.attention.hosts == {"dit": {"tiny": 2}}
        package.FINISH[:] = [lambda: None]
        steady = run("steady")
        assert held == [tiny] * 4
        assert steady["observed"] == "tiny" and "segments" not in steady
    sys.modules.pop("steps_fixture", None)
