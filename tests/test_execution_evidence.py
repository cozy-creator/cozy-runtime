"""An attempt's execution record through the real executor: ranks, setup legs, step series."""

from __future__ import annotations

import os
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author._observations import MAX_SERIES, Attribution, EventRing
from cozy_runtime.internal import execution_evidence, executor_replies, package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import _SEALED, Executor, _send_reply
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, PrepareRequest, Start
from cozy_runtime.internal.executor_replies import AttemptReply
from cozy_runtime.internal.seam import Channel

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
    tel.log("clash", message="m", name="a", kind="b", value=1)
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


def test_a_log_field_may_share_a_name_with_the_record_it_lands_in(executor: Executor) -> None:
    reply = _invoke(executor, "clash")
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    [row] = [row for row in reply["observations"] if row["name"] == "clash"]
    assert (row["kind"], row["value"]) == ("log", "info")
    assert row["fields"] == {"message": "m", "name": "a", "kind": "b", "value": 1}


def test_a_reply_of_several_mib_crosses_the_seam_whole(executor: Executor) -> None:
    """A frame has no size rule: 8 MiB of attribution, which the old 64 KiB frame shed (and
    refused the reply past it), crosses the real seam whole beside the result."""
    reply = _invoke(executor, "large")
    reply["attribution"]["padding"] = "x" * (8 << 20)
    left, right = socket.socketpair()
    with left, right:
        sending = threading.Thread(target=_send_reply, args=(Channel(left), reply))
        sending.start()
        received = Channel(right).recv(timeout=60)
        sending.join()
    assert received is not None and received["attribution"]["padding"] == "x" * (8 << 20)
    kept = executor_replies.decode(received, AttemptReply)
    assert kept.ok and kept.outcome is not None and kept.outcome.terminal == "succeeded", kept
    assert reply["result_ref"] and msgspec.to_builtins(kept.result_ref) == reply["result_ref"]


def test_a_telemetry_key_read_as_a_credential_costs_the_reply_only_its_telemetry(
    executor: Executor,
) -> None:
    """`tel.log(..., token=1)`: the seam refuses a frame carrying a credential-named key, so
    the reply goes without its telemetry, counted, and keeps its result."""
    reply = _invoke(executor, "credential")
    reply["observations"][0]["fields"] = {"token": 1}
    left, right = socket.socketpair()
    with left, right:
        _send_reply(Channel(left), reply)
        kept = executor_replies.decode(Channel(right).recv(), AttemptReply)
    assert kept.ok and kept.outcome is not None and kept.outcome.terminal == "succeeded", kept
    assert not kept.observations and not kept.attribution and not kept.execution
    assert kept.observation_caps["dropped_observations_for_credential"] == 1


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
    from functools import partial

    from cozy_runtime.internal import attention, attention_upstream, kernel_cache, kernel_sources
    from cozy_runtime.internal.encoding import DeviceFacts

    wheel = tmp_path / "late_kernel-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("late_kernel/__init__.py", "")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    row = {"version": "1.0", "file": wheel.name, "sha256": digest}
    (tmp_path / "index.json").write_text(json.dumps({"sources": {"late-kernel": row}}))
    monkeypatch.setattr(kernel_sources, "job", partial(kernel_sources.job, root=tmp_path))
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
        assert "late" not in skipped  # still compiling is not a skip: it is evidence
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
