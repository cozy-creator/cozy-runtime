"""No request waits for a kernel (owner: no compile wait; nothing blocking on the request path).
A Sol step whose object still builds runs the dense reference, and the object serves from the
step boundary after it is ready; the run's records say which step served which. The real
executor serves a package's steps through Sol's own dispatch, against a real machine store whose
build lock a live builder process holds; the object is a stand-in (this host has no CUDA), and
so is its sparse call. Nothing it does blocks the request thread (`request_path.blocking`).
"""

from __future__ import annotations

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
    attention_sol,
    executor,
    kernel_cache,
    kernel_compile,
    kernel_ready,
    liveness,
    package_interface,
)
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, PrepareRequest, Start
from cozy_runtime.internal.seam import Channel
from test_kernel_compile import _held

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)

#: A live builder that keeps moving bytes for as long as a test holds its key.
MOVING = "import os, time\nfor _ in range(3000): os.write(2, b'.'); time.sleep(0.02)"
MODULE = "sol_steps_fixture"
#: Four steps of two Sol sites on a 130-token document, dense until step 1. A step served dense
#: is q + 1 (the bound reference), sparse v + 2 (the stand-in object). `FINISH` runs in step 1.
SOURCE = """
import msgspec
import torch
from cozy_runtime.author import App, Context, Telemetry
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal import attention_sol

app = App()
FINISH, SERVED, COUNTED = [], [], []


class Request(msgspec.Struct):
    steps: int = 4


class Result(msgspec.Struct):
    steps: int


@app.entrypoint
def generate(ctx: Context, payload: Request, tel: Telemetry) -> Result:
    on_step = tel.step_callback(payload.steps, stage="denoise")
    q = torch.zeros(1, 130, 2, 128, dtype=torch.bfloat16)
    v = q + 4
    token = attention_sol._SITE.set(attention_sol.Site("dit", "blocks.0.attn"))
    try:
        for step in range(payload.steps):
            with attention_scope(AttentionLayout(130, 0, step, dense_until_step=1)):
                outs = [attention_sol.sol_attention(q, q, v) for _ in range(2)]
            SERVED.append(["sparse" if torch.equal(o, v + 2) else "dense" for o in outs])
            if step == 1 and FINISH:
                FINISH.pop()()
            on_step(step)
    finally:
        attention_sol._SITE.reset(token)
    COUNTED[:] = [dict(attention_sol._COUNTS.get() or {})]
    return Result(payload.steps)
"""


def _job(library: str, tokens: int, capability: tuple[int, int]) -> kernel_compile.Job:
    """Sol's job for this length, keyed by a stand-in for CuTe's library inputs."""
    inputs = {"variant": attention_sol.variant(tokens, capability), "library": library}
    spec = {"route": attention_sol.route(capability)}
    return kernel_compile.Job("sol", kernel_cache.Key.of(attention_sol.KERNEL, inputs), spec)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """A real executor serving the package on an SM90 Sol route, its store a machine run's."""
    scope = tmp_path / "run"
    scope.mkdir()
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setitem(executor._SEALED, "TMPDIR", str(scope))  # the run's failures stay here
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", 0.05)
    monkeypatch.setattr(kernel_compile, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(kernel_compile, "_STARTED", {})
    for name in ("_card", "_capability"):
        monkeypatch.setattr(attention_sol, name, lambda device: (9, 0))
    job = partial(_job, str(tmp_path))
    monkeypatch.setattr(attention_sol, "job", job)
    monkeypatch.setattr(attention_sol, "_scope", lambda capability: "sol-attn-stand-in")
    monkeypatch.setattr(attention_sol, "load", lambda data: ("stand-in", data))
    monkeypatch.setattr(attention_sol, "_native", lambda q, k, v, *_: v + 2)
    attention_sol._job.cache_clear()
    held = attention_sol.binding()
    attention_sol.bind_dense(lambda *, query, key, value, scale: query + 1, "test-dense")
    store = kernel_cache.Store(tmp_path / "u1")
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
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
        process = Executor(Channel(left), tmp_path / "root")
        (tmp_path / "root").mkdir()
        assert process.start(msgspec.convert({**binding, "devices": ""}, Start))["ok"]
        load = {"devices": "", "binding": binding, "budgets": {}, "construction": "a"}
        assert process.load(msgspec.convert(load, Load))["ok"]
        assert process.activate(Activate(construction="a"))["ok"]

        def run(request_id: str) -> dict[str, Any]:
            package.SERVED.clear()
            prepared = process.serve_request(
                PrepareRequest(request_id=request_id, entrypoint="generate", payload={})
            )
            assert prepared["ok"], prepared
            spool = process.root / request_id
            spool.mkdir()
            reply = process.serve_invoke(
                Invoke(request_id=request_id, entrypoint="generate", spool=str(spool),
                       deadline_s=60)
            )  # fmt: skip
            assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
            return reply

        package = sys.modules[MODULE]
        yield SimpleNamespace(store=store, job=job(191, (9, 0)), package=package, run=run)
    attention_sol.rebind(held)
    attention_sol._job.cache_clear()
    sys.modules.pop(MODULE, None)


def _rows(reply: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        {"value": row["value"], **row["fields"]}
        for row in reply["observations"]
        if row["name"] == name
    ]


def _landed(name: str) -> None:
    while (slot := kernel_ready._SLOTS[name]).state == "opening":
        time.sleep(kernel_compile.POLL_SECONDS)
    assert slot.state == "landed", (slot.state, kernel_ready.failed(name))


DENSE, SPARSE = ["dense"] * 2, ["sparse"] * 2


def test_a_sparse_step_runs_dense_while_its_object_builds_and_sparse_from_the_next_step(
    host: SimpleNamespace,
) -> None:
    lock, builder = _held(host.store, host.job, MOVING)
    name = f"{attention_sol.KERNEL} {host.job.key.digest[:16]}"

    def finish() -> None:  # the builder publishes the object; its opener links it
        host.store.put(host.job.key, b"object", {})
        os.close(lock)
        builder.kill()
        builder.wait()
        _landed(name)

    host.package.FINISH[:] = [finish]
    epoch = kernel_ready.epoch()
    reply = host.run("builds")
    assert host.package.SERVED == [DENSE, DENSE, SPARSE, SPARSE]
    assert kernel_ready.epoch() == epoch + 1  # what keys on the kernels serving misses once
    assert host.package.COUNTED == [
        {
            "sparse": 4,
            "dense_step": 2,
            "dense_building": 2,
            "dense_path": 0,
            "dense_prefix": 0,
        }
    ]
    kernels = [(r["value"], r["step"]) for r in _rows(reply, "attention.sol.kernel")]
    assert kernels == [("building", 0), ("built", 2)], kernels
    assert _rows(reply, "kernel.ready") == [{"value": name, "stage": "denoise", "step": 2}]
    assert _rows(reply, "request_path.blocking") == []
    # The next run serves the object from its first sparse step.
    host.package.FINISH[:] = []
    steady = host.run("steady")
    assert host.package.SERVED == [DENSE, SPARSE, SPARSE, SPARSE]
    assert [(r["value"], r["step"]) for r in _rows(steady, "attention.sol.kernel")] == [
        ("store", 0)
    ]
    assert _rows(steady, "kernel.ready") == _rows(steady, "request_path.blocking") == []


def test_an_object_whose_build_failed_leaves_every_step_dense_and_says_why(
    host: SimpleNamespace,
) -> None:
    """The builder ends without the object; the real one the wait then starts cannot build a
    stand-in key either, so the machine run records the failure."""
    lock, builder = _held(host.store, host.job, MOVING)
    name = f"{attention_sol.KERNEL} {host.job.key.digest[:16]}"

    def finish() -> None:
        os.close(lock)
        builder.kill()
        builder.wait()
        while kernel_ready._SLOTS[name].state == "opening":
            time.sleep(kernel_compile.POLL_SECONDS)
        assert kernel_ready.failed(name)

    host.package.FINISH[:] = [finish]
    reply = host.run("fails")
    assert host.package.SERVED == [DENSE] * 4
    [counts] = host.package.COUNTED
    assert (counts["dense_step"], counts["dense_building"], counts["sparse"]) == (6, 2, 0)
    rows = _rows(reply, "attention.sol.kernel")
    assert [(r["value"], r["step"]) for r in rows] == [("building", 0), ("failed", 2)]
    assert "CompileFailed" in rows[-1]["state"], rows[-1]
    assert _rows(reply, "request_path.blocking") == []
