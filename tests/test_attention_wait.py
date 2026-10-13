"""A run never waits for a compile: a site whose kernel still compiles serves the best one
ready, and the construction takes the kernel at its next selection once it is built, which
is the run's next step. A request's explicit `--attention-kernel` is the opposite: it waits
for its own kernel's build and holds that kernel for every step, or is refused. Real
Diffusers attention sites, a real source registry, real builder processes and the real
store; the kernel is a registry wheel a CPU can build, named in a model's ladder and in the
ranked table. A cancel ends the override's wait at once, and the wait is a stage of its run.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import os
import socket
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest

from cozy_runtime.internal import (
    attention,
    attention_upstream,
    executor,
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
    Invoke,
    Load,
    PrepareRequest,
    Start,
)
from cozy_runtime.internal.seam import Channel, SeamError
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
    # Tests in one second build the same wheel, so the same key: each has its own builders.
    monkeypatch.setattr(kernel_compile, "_STARTED", {})
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
        if step == 0 and FINISH:
            FINISH.pop()()
        on_step(step)
    return Result(payload.steps)
"""


@contextmanager
def _host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """A real executor serving the STEPS package on two Wan attention sites, its selection
    made while `tiny` built (the construction passed over it). Every frame the executor sends
    its machine is collected, a compile wait's progress among them."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "steps_fixture.py").write_text(STEPS)
    (project / "package.toml").write_text('[application]\nobject = "steps_fixture:app"\n')
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(
        package_interface.canonical_bytes(package_interface.build(discover(project)))
    )
    binding = {"application": "steps_fixture:app", "package_interface": str(interface)}
    left, right = socket.socketpair()
    channel, frames = Channel(left), []
    monkeypatch.setattr(kernel_compile, "waiting", partial(executor._compiling, channel))

    def read() -> None:
        with contextlib.suppress(SeamError, OSError):
            while (frame := Channel(right).recv()) is not None:
                frames.append(frame)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    with left, right:
        host = Executor(channel, tmp_path / "root")
        (tmp_path / "root").mkdir()
        assert host.start(msgspec.convert({**binding, "devices": ""}, Start))["ok"]
        load = {"devices": "", "binding": binding, "budgets": {}, "construction": "a"}
        assert host.load(msgspec.convert(load, Load))["ok"]
        assert host.activate(Activate(construction="a"))["ok"]
        construction, roots = host.constructions["a"], _sites()
        construction.backend, construction.device_facts = SimpleNamespace(roots=roots), CPU
        construction._reselect = partial(
            attention.select, CPU, roots, choose=_ladder("tiny", "sdpa")
        )
        construction.attention = construction._reselect()
        construction.attention_defaults = attention.snapshot(roots)
        package = sys.modules["steps_fixture"]
        package.SITES[:] = list(cast(Iterable[object], roots["dit"]))
        package.FINISH[:] = [lambda: None]
        replies: list[dict[str, Any]] = []

        def run(request_id: str, kernel: str = "") -> dict[str, Any]:
            """One request; its rank's attention record, or the prepare's refusal."""
            package.HELD.clear()
            prepared = host.serve_request(
                PrepareRequest(
                    request_id=request_id,
                    entrypoint="generate",
                    payload={},
                    attention_kernel=kernel,
                )
            )
            if not prepared["ok"]:
                return prepared
            spool = host.root / request_id
            spool.mkdir()
            reply = host.serve_invoke(
                Invoke(
                    request_id=request_id,
                    entrypoint="generate",
                    spool=str(spool),
                    deadline_s=60,
                    attention_kernel=kernel,
                )
            )
            assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
            replies.append(reply)
            [rank] = reply["execution"]["ranks"]
            return dict(rank["attention"])

        yield SimpleNamespace(
            construction=construction,
            package=package,
            held=package.HELD,
            frames=frames,
            run=run,
            replies=replies,
        )
    reader.join()
    sys.modules.pop("steps_fixture", None)


def _finisher(store: kernel_cache.Store, job: kernel_compile.Job, lock: int, builder: Any) -> Any:
    def finish() -> None:  # the build that was running ends; the real builder takes the key
        os.close(lock)
        builder.kill()
        builder.wait()
        _built(store, job)

    return finish


TINY_SITES, SDPA_SITES = ["_cozy_tiny"] * 2, ["_cozy_sdpa"] * 2


def test_a_compile_that_finishes_mid_run_serves_the_rest_of_the_run(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner ruling 2026-10-09: a run whose preferred kernel finishes compiling partway
    through switches to it at the next step. The real executor runs the package's steps on
    the construction's sites; the evidence names which steps served which kernel."""
    _registry(tmp_path / "sources")
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    with _host(tmp_path, monkeypatch) as host:
        assert host.construction.attention.waiting == ("tiny",)
        host.package.FINISH[:] = [_finisher(store, job, lock, builder)]
        switched = host.run("switched")
        assert host.held == [SDPA_SITES, TINY_SITES, TINY_SITES, TINY_SITES]
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
        assert host.construction.attention.hosts == {"dit": {"tiny": 2}}
        host.package.FINISH[:] = [lambda: None]
        steady = host.run("steady")
        assert host.held == [TINY_SITES] * 4
        assert steady["observed"] == "tiny" and "segments" not in steady


def test_an_override_holds_its_kernel_for_every_step_while_another_finishes_compiling(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same build ending during step 0 that switches a default run (above) leaves an
    explicit `--attention-kernel` alone: every step runs the kernel it named."""
    _registry(tmp_path / "sources")
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    with _host(tmp_path, monkeypatch) as host:
        assert host.construction.attention.waiting == ("tiny",)
        host.package.FINISH[:] = [_finisher(store, job, lock, builder)]
        pinned = host.run("pinned", "sdpa")
        assert kernel_compile.status(store, job).state == "ready"
        assert host.held == [SDPA_SITES] * 4
        assert pinned["observed"] == "sdpa" and "segments" not in pinned, pinned
        # Its default selection takes the built kernel at the next request, as before.
        assert host.run("default")["observed"] == "tiny" and host.held == [TINY_SITES] * 4


def test_an_override_naming_a_compiling_kernel_waits_for_it_then_runs_every_step_on_it(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request waits for its kernel's own build, its progress saying so, and never
    starts on the kernel that is ready meanwhile. The wait is the run's first stage."""
    _registry(tmp_path / "sources")
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    with _host(tmp_path, monkeypatch) as host:
        assert host.construction.attention.hosts == {"dit": {"sdpa": 2}}
        reply: dict[str, Any] = {}
        request = threading.Thread(target=lambda: reply.update(host.run("waits", "tiny")))
        request.start()
        while request.is_alive() and not _waiting(host.frames):
            time.sleep(kernel_compile.POLL_SECONDS)
        seen = time.time() * 1000
        assert request.is_alive() and not host.held, reply
        assert _waiting(host.frames)[0]["stage_fraction"] == 0.25
        finishing = time.time() * 1000
        _finisher(store, job, lock, builder)()
        request.join()
        assert host.held == [TINY_SITES] * 4
        assert reply["observed"] == "tiny" and "segments" not in reply, reply
        attribution = host.replies[-1]["attribution"]
        wait = attribution["stages"]["waiting for tiny compile"]
        ended = wait["started_unix_ms"] + wait["total_ms"]
        assert wait["count"] == 1 and wait["started_unix_ms"] <= seen, wait
        assert finishing - 1 <= ended <= attribution["steps"]["denoise"]["series"][0][0], wait


def test_a_cancel_ends_an_override_waiting_for_its_compile_at_once_and_the_build_goes_on(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker's cancel marker landing (TensorD's rename) ends the waiting request canceled
    at once: the wait sleeps on it, not on its poll (an hour here). The build goes on, and a
    later request runs every step on the kernel it built."""
    _registry(tmp_path / "sources")
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    with _host(tmp_path, monkeypatch) as host:
        monkeypatch.setattr(kernel_compile, "POLL_SECONDS", 3600.0)
        reply: dict[str, Any] = {}
        request = threading.Thread(
            target=lambda: reply.update(host.run("canceled", "tiny")), daemon=True
        )
        request.start()
        while request.is_alive() and not _waiting(host.frames):
            time.sleep(0.01)
        pending = tmp_path / "root" / "executor.cancel.pending"
        pending.write_text("canceled")
        canceled = time.monotonic()
        pending.rename(tmp_path / "root" / "executor.cancel")
        request.join(30)
        took = time.monotonic() - canceled
        assert not request.is_alive() and took < 1.0, took
        assert (reply["ok"], reply["code"], reply["terminal"]) == (False, "cancelled", "canceled")
        assert builder.poll() is None and kernel_compile.status(store, job).state == "compiling"
        assert not host.held, reply
        monkeypatch.setattr(kernel_compile, "POLL_SECONDS", 0.01)
        _finisher(store, job, lock, builder)()
        assert host.run("later", "tiny")["observed"] == "tiny"
        assert host.held == [TINY_SITES] * 4


def test_an_override_whose_compile_fails_falls_back_with_the_compiles_reason(
    store: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kernel whose build fails while the request waits cannot run here: the request runs
    on the prepared kernel and its evidence says why. One this hardware cannot serve falls
    back at once. Neither fails the run, and neither is silent."""
    _registry(tmp_path / "sources", corrupt=True)
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock, builder = _held(store, job, MOVING)
    with _host(tmp_path, monkeypatch) as host:
        reply: dict[str, Any] = {}
        request = threading.Thread(target=lambda: reply.update(host.run("fails", "tiny")))
        request.start()
        while request.is_alive() and not _waiting(host.frames):
            time.sleep(kernel_compile.POLL_SECONDS)
        os.close(lock)  # the build ends; this machine's builder takes the key and fails it
        builder.kill()
        builder.wait()
        request.join()
        (fallback,) = reply["fallbacks"]
        assert (fallback["preferred"], fallback["selected"]) == ("tiny", "sdpa"), reply
        assert "compile_source_mismatch" in fallback["reason"], reply
        assert host.held[-1] == SDPA_SITES
        unsupported = host.run("unsupported", "flash-attn3")
        assert unsupported["fallbacks"][0]["preferred"] == "flash-attn3", unsupported
        assert "sm90" in unsupported["fallbacks"][0]["reason"], unsupported
        assert host.held[-1] == SDPA_SITES


def _waiting(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in frames if f.get("stage") == "waiting for tiny compile"]
