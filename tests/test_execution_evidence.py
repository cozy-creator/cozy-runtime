"""An attempt's execution record through the real executor: ranks, setup legs, step series."""

from __future__ import annotations

import logging
import os
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author._observations import MAX_SERIES, Attribution, EventRing
from cozy_runtime.internal import execution_evidence, executor_replies, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import _SEALED, Executor, _send_reply
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, PrepareRequest, Start
from cozy_runtime.internal.executor_replies import AttemptReply
from cozy_runtime.internal.seam import MAX_FRAME, Channel
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions

MODULE = "execution_evidence_fixture"
SOURCE = """
import time
import msgspec
import numpy
from cozy_runtime.author import App, Context, Telemetry

app = App()


class Request(msgspec.Struct):
    steps: int = 4


class Result(msgspec.Struct):
    steps: int


@app.entrypoint
def generate(ctx: Context, payload: Request, tel: Telemetry) -> Result:
    for index in range(3):
        tel.log("evidence row", index=index)
    tel.log("delivery", complete=True, delivered=payload.steps)
    tel.log("unreadable", shape=[1, 2])
    fields = {f"k{i}": i for i in range(10)}
    tel.log("m" * 300, seed=2**60, loss=float("nan"), scale=numpy.float32(0.5), **fields)
    tel.metric("loss", float("inf"))
    with tel.stage("condition"):
        time.sleep(0.01)
    on_step = tel.step_callback(payload.steps, stage="denoise")
    for step in range(payload.steps):
        time.sleep(0.002)
        on_step(step)
    return Result(payload.steps)
"""


def _project(root: Path) -> Path:
    project = root / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject = "{MODULE}:app"\n')
    return project


@pytest.fixture
def executor(tmp_path: Path) -> Iterator[Executor]:
    project = _project(tmp_path)
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(
        package_interface.canonical_bytes(package_interface.build(discover(project)))
    )
    binding = {
        "application": f"{MODULE}:app",
        "package_interface": str(interface),
    }
    devices = _SEALED.get("CUDA_VISIBLE_DEVICES", "")
    left, right = socket.socketpair()
    with left, right:
        host = Executor(Channel(left), tmp_path / "root")
        (tmp_path / "root").mkdir()
        assert host.start(msgspec.convert({**binding, "devices": devices}, Start))["ok"]
        load = {"devices": devices, "binding": binding, "budgets": {}, "construction": "a"}
        assert host.load(msgspec.convert(load, Load))["ok"]
        assert host.activate(Activate(construction="a"))["ok"]
        yield host
    sys.modules.pop(MODULE, None)


def _invoke(host: Executor, request_id: str, **extra: Any) -> dict[str, Any]:
    prepared = host.serve_request(
        PrepareRequest(request_id=request_id, entrypoint="generate", payload={"steps": 4})
    )
    assert prepared["ok"], prepared
    spool = host.root / request_id
    spool.mkdir()
    return host.serve_invoke(
        msgspec.convert(
            {
                "request_id": request_id,
                "entrypoint": "generate",
                "spool": str(spool),
                "deadline_s": 30,
                **extra,
            },
            Invoke,
        )
    )


def test_an_attempt_records_its_rank_setup_and_every_step(executor: Executor) -> None:
    reply = _invoke(executor, "evidence")
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    execution = reply["execution"]
    assert execution["degree"] == 1
    [rank] = execution["ranks"]
    assert set(rank) == {
        "rank",
        "pid",
        "ordinal",
        "uuid",
        "arch",
        "start_us",
        "end_us",
        "attention",
    }
    assert (rank["rank"], rank["pid"]) == (0, os.getpid())
    assert 0 < rank["start_us"] <= rank["end_us"]
    assert rank["attention"] == {"requested": "", "observed": "", "impl": ""}
    boot = execution["executor"]
    assert "package_import" in boot["legs_ms"] and boot["started_unix_ms"] * 1000 < rank["start_us"]
    assert set(execution["construction"]) >= {"prepared_unix_ms", "prepare_ms", "legs_ms"}

    attribution = reply["attribution"]
    condition = attribution["stages"]["condition"]
    assert condition["started_unix_ms"] <= condition["ended_unix_ms"]
    denoise = attribution["steps"]["denoise"]
    assert denoise["count"] == 4 and denoise["series_dropped"] == 0
    ends = [end for end, _ms in denoise["series"]]
    assert len(ends) == 4 and ends == sorted(ends)
    assert condition["ended_unix_ms"] <= ends[0] <= rank["end_us"] // 1000
    assert sum(ms for _end, ms in denoise["series"]) == pytest.approx(denoise["total_ms"], abs=0.01)


def test_telemetry_of_any_shape_reaches_the_worker_and_never_fails_the_attempt(
    executor: Executor, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Run 1566: one `complete=True` field failed a finished attempt `executor_reply_malformed`.
    A 300-character log message, a 2^60 seed, NaN, a numpy scalar and ten fields each raised
    into the handler or broke the seam. All arrive bounded, as scalars JSON holds exactly; a
    row an older executor sends that the worker cannot read costs itself, with one warning."""
    frame = _invoke(executor, "scalars")
    frame["observations"].append({"kind": "log", "name": "older", "fields": {"shape": [1, 2]}})
    reply = executor_replies.decode(frame, AttemptReply)
    assert reply.ok and reply.outcome is not None, reply
    assert reply.outcome.terminal == "succeeded", reply.outcome
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=tmp_path / "store"),
        InMemoryControlHost(),
    )
    attempt = AttemptRecord("scalars", 1, b"digest", {})
    try:
        with caplog.at_level(logging.WARNING):
            worker.engine.absorb(attempt, reply, apply_ledger=False)
    finally:
        worker.shutdown()
    rows = {row["name"]: row for row in attempt.ring.rows()}
    fields: Any = rows["delivery"]["fields"]
    assert fields["complete"] is True and fields["delivered"] == 4
    assert rows["unreadable"]["fields"] == {"shape": "[1, 2]"}
    (long,) = [row for name, row in rows.items() if str(name).startswith("mmm")]
    assert len(str(long["name"])) == 200 and "~" in str(long["name"])
    assert long["fields"] == {
        "seed": str(2**60), "loss": "nan", "scale": 0.5, "k0": 0, "k1": 1, "k2": 2, "k3": 3, "k4": 4
    }
    assert rows["loss"]["value"] == "inf"
    assert "older" not in rows and attempt.caps["unreadable_rows"] == 1
    (warning,) = [row.getMessage() for row in caplog.records if "cannot read" in row.getMessage()]
    assert "'older'" in warning and "got `array`" in warning, warning


def test_a_reply_too_large_for_one_frame_sheds_telemetry_and_keeps_its_result(
    executor: Executor,
) -> None:
    """A reply over one frame looped forever once one observation row was left, and its
    fallback wrote nulls and `minimal_reply: True` into members the worker typed str/bool/int,
    so the worker read `executor_reply_malformed` and lost the outcome. Through the real seam,
    telemetry is shed and the result arrives; an irreducible reply arrives as its own refusal."""
    reply = _invoke(executor, "oversize")
    reply["attribution"] = {"padding": "x" * (MAX_FRAME + 1)}
    irreducible = {**reply, "frames": [{"handle": "x" * (MAX_FRAME + 1)}]}
    left, right = socket.socketpair()
    with left, right:
        _send_reply(Channel(left), reply)
        kept = executor_replies.decode(Channel(right).recv(), AttemptReply)
        _send_reply(Channel(left), irreducible)
        refused = executor_replies.decode(Channel(right).recv(), AttemptReply)
    assert kept.ok and kept.outcome is not None and kept.outcome.terminal == "succeeded", kept
    assert kept.observation_caps["dropped_attribution_for_frame"] == 1 and not kept.attribution
    assert (refused.ok, refused.code) == (False, "reply_too_large"), refused
    assert refused.observation_caps["minimal_reply"] == 1


def test_the_request_pin_is_what_every_rank_reports_as_requested() -> None:
    hosts = {"base_model/ref2va_dit": {"sdpa": 52}, "base_model/video_vae": {"sdpa": 36}}
    assert execution_evidence.observed(hosts) == "sdpa"
    hosts["base_model/ref2va_dit"] = {"flash-attn3": 50, "sdpa": 2}
    assert execution_evidence.observed(hosts) == "ref2va_dit=flash-attn3+sdpa,video_vae=sdpa"
    held: dict[int, execution_evidence.RankEvidence] = {}
    for start, end in ((5, 9), (2, 4), (10, 12)):
        execution_evidence.widen(held, execution_evidence.record(1, {"pid": 7}, start, end))
    assert (held[1]["start_us"], held[1]["end_us"], held[1]["pid"]) == (2, 12, 7)


def test_a_kernel_still_compiling_is_evidence_with_its_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-selection serves the next ready kernel past one still compiling, and the rank's
    evidence says which served and how far along the other is."""
    pytest.importorskip("diffusers")
    import hashlib
    import json
    import zipfile

    from cozy_runtime.internal import attention, attention_upstream, kernel_cache, kernel_sources
    from cozy_runtime.internal.encoding import DeviceFacts

    wheel = tmp_path / "late_kernel-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("late_kernel/__init__.py", "")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    row = {"version": "1.0", "file": wheel.name, "sha256": digest}
    (tmp_path / "index.json").write_text(json.dumps({"sources": {"late-kernel": row}}))
    monkeypatch.setattr(kernel_sources, "ROOT", tmp_path)
    store = kernel_cache.Store(tmp_path / "u0")
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    lock = store.claim(job.key)
    assert lock is not None
    store.set_state(job.key, {"progress": 0.42, "started_unix_ms": 1})
    late = attention.Candidate(
        name="late",
        backend="_cozy_late_test",
        local=attention_upstream.SDPA,
        min_sm=0,
        dtypes=frozenset(),
        max_head_dim=0,
        artifact=kernel_sources.PYTHON,
        import_name="late_kernel",
    )
    cpu = DeviceFacts("cpu", "CPU", 0, "", "", 0)
    monkeypatch.setattr(attention, "CANDIDATES", (late, *attention.CANDIDATES))
    try:
        skipped: dict[str, str] = {}
        assert [ready.name for ready in attention.available(cpu, skipped)] == ["sdpa"]
        assert "compiling (42%)" in skipped["late"]
        rows = {row["kernel"]: row for row in attention.evidence(("sdpa",))}
        assert rows["sdpa"] == {"kernel": "sdpa", "state": "ready", "line": "ready", "served": True}
        assert rows["late"]["state"] == "compiling" and rows["late"]["progress"] == 0.42
        assert not rows["late"]["served"] and rows["late"]["line"] == "compiling (42%)"
    finally:
        os.close(lock)
        attention._STATES.pop("late", None)


def test_a_long_schedule_keeps_a_bounded_series_and_counts_the_rest() -> None:
    attribution = Attribution()
    for _ in range(MAX_SERIES + 3):
        attribution.step("denoise", 1.0)
    track: Any = attribution.document()["steps"]
    track = track["denoise"]
    assert (len(track["series"]), track["series_dropped"], track["count"]) == (
        MAX_SERIES,
        3,
        MAX_SERIES + 3,
    )


def test_a_live_row_carries_the_sequence_the_worker_deduplicates_by() -> None:
    ring = EventRing()
    first, second = ring.emit("log", "a"), ring.emit("log", "b")
    assert (first.seq, second.seq) == (1, 2)
