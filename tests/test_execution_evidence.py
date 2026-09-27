"""An attempt's execution record through the real executor: ranks, setup legs, step series."""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import sample_python
from cozy_runtime.author._observations import MAX_SERIES, Attribution, EventRing
from cozy_runtime.internal import child_env, execution_evidence, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import _SEALED, Executor
from cozy_runtime.internal.local import LocalRequest, run_slice
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker.session import WorkerOptions
from test_end_to_end import NO_EXECUTOR

MODULE = "execution_evidence_fixture"
SOURCE = """
import time
import msgspec
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
        "development_project": str(project),
    }
    devices = _SEALED.get("CUDA_VISIBLE_DEVICES", "")
    left, right = socket.socketpair()
    with left, right:
        host = Executor(Channel(left), tmp_path / "root")
        (tmp_path / "root").mkdir()
        assert host.start({**binding, "devices": devices})["ok"]
        load = {"devices": devices, "binding": binding, "budgets": {}, "construction": "a"}
        assert host.load(load)["ok"]
        assert host.activate({"construction": "a"})["ok"]
        yield host
    sys.modules.pop(MODULE, None)


def _invoke(host: Executor, request_id: str, **extra: Any) -> dict[str, Any]:
    prepared = host.serve_request(
        {"request_id": request_id, "entrypoint": "generate", "payload": {"steps": 4}}
    )
    assert prepared["ok"], prepared
    spool = host.root / request_id
    spool.mkdir()
    return host.serve_invoke(
        {
            "request_id": request_id,
            "entrypoint": "generate",
            "spool": str(spool),
            "deadline_s": 30,
            **extra,
        }
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


def test_the_request_pin_is_what_every_rank_reports_as_requested() -> None:
    hosts = {"base_model/ref2va_dit": {"sdpa": 52}, "base_model/video_vae": {"sdpa": 36}}
    assert execution_evidence.observed(hosts) == "sdpa"
    hosts["base_model/ref2va_dit"] = {"flash-attn3": 50, "sdpa": 2}
    assert execution_evidence.observed(hosts) == "ref2va_dit=flash-attn3+sdpa,video_vae=sdpa"
    held: dict[int, dict[str, Any]] = {}
    for start, end in ((5, 9), (2, 4), (10, 12)):
        execution_evidence.widen(held, execution_evidence.record(1, {"pid": 7}, start, end))
    assert (held[1]["start_us"], held[1]["end_us"], held[1]["pid"]) == (2, 12, 7)


def test_a_kernel_the_probe_refuses_is_evidence_with_its_measurement() -> None:
    """A kernel 5% off exact attention runs, and the probe refuses it by name: the rank's
    `skipped` evidence carries the refusal with the numbers behind it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.internal import attention, attention_ulysses
    from cozy_runtime.internal.encoding import DeviceFacts

    def off(query: Any, key: Any, value: Any, **_options: Any) -> Any:
        exact = torch.nn.functional.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (query, key, value))
        )
        return (exact * 1.05).transpose(1, 2)

    candidate = attention.Candidate(
        name="off-by-five",
        backend="_cozy_off_by_five_test",
        local=attention_ulysses.Local(off, lambda *_: "scaled"),
        min_sm=0,
        dtypes=frozenset(),
        max_head_dim=0,
        rel_l2_max=1e-2,
    )
    try:
        with pytest.raises(attention.AttentionRefusal) as refused:
            attention._resolve(candidate, DeviceFacts("cpu", "CPU", 0, "", "", 0))
        assert refused.value.code == "attention_kernel_out_of_budget"
        [row] = [r for r in attention.evidence(())["skipped"] if r["kernel"] == "off-by-five"]
        assert row["status"] == "attention_kernel_out_of_budget"
        assert (row["source"], row["budget"], row["budget_state"]) == (
            "prepare",
            1e-2,
            "calibrated",
        )
        assert row["shape"] == list(attention.PROBE_SHORT)
        assert set(row["rel_l2_by_rows"]) == {"sampled", "first", "last", "tile_edges"}
        assert min(row["rel_l2_by_rows"].values()) == pytest.approx(0.05, abs=5e-3)
        assert "off-by-five" not in str(attention.evidence(("off-by-five",)).get("skipped"))
    finally:
        attention._MEASURED.pop("off-by-five", None)


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


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_the_triage_bundle_keeps_the_record_and_each_row_once() -> None:
    """The real worker and executor child: the kept bundle carries the execution record, and
    rows the live lane delivered are not admitted again from the reply."""
    base = {
        k: v
        for k, v in os.environ.items()
        if not child_env.erased(k) and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
    }
    with tempfile.TemporaryDirectory(prefix="cz-ev.", dir="/tmp") as raw:
        root = Path(raw)
        project = _project(root)
        config = RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(sorted(base.items())),
        )
        with contextlib.chdir(project), contextlib.redirect_stdout(sys.stderr):
            outcome = run_slice(
                config,
                LocalRequest(
                    entrypoint="generate", payload={"steps": 4}, outputs=(), request_id="ev-1"
                ),
                workspace=root / "run",
                package_installation_id="",
                package_interface_document=package_interface.build(discover(project)),
                project=project,
                development_tensorfs_root=config.cozy_home / "tensorfs",
                options=WorkerOptions(
                    root=root / "run" / "worker", devices="0", python=str(sample_python())
                ),
            )
    sys.modules.pop(MODULE, None)
    assert outcome.result == {"steps": 4}, (outcome.error, outcome.faults)
    assert outcome.triage, (outcome.triage_subject, outcome.terminal.get("triage_bundle"))
    measurements = outcome.triage["measurements"]
    [rank] = measurements["execution"]["ranks"]
    assert measurements["execution"]["degree"] == 1 and rank["pid"] not in (0, os.getpid())
    assert len(measurements["attribution"]["steps"]["denoise"]["series"]) == 4
    rows = [row for row in outcome.triage["events"] if row["name"] == "evidence row"]
    assert [row["fields"]["index"] for row in rows] == [0, 1, 2], rows
