"""The models-only lane's real TensorFS `ensure` against a real Hub and object store that fail
on purpose. Each pull ends with the checkpoint verified and no admission temp left, or with
a refusal naming TensorFS's code and its detail; none is bounded by a wall clock."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import machine_materialization
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from fault_hub import (
    ALWAYS,
    ASSET,
    HEADER,
    Checkpoint,
    Corrupt,
    Delay,
    Drip,
    FaultHub,
    Gate,
    Reset,
    Rule,
    Stall,
    Truncate,
    Watched,
    checkpoint,
    enveloped,
    on,
    temps,
    tunnel_404,
    weights,
)
from test_device_lanes import _config


@contextmanager
def machine(
    tmp_path: Path, *rules: Rule, lifetime: int = 3600, point: Checkpoint | None = None
) -> Iterator[tuple[Worker, FaultHub]]:
    point = point or checkpoint(tmp_path / "origin")
    with FaultHub(point, *rules, lifetime=lifetime) as hub:
        tensorfs.Store.init(str(tmp_path / "store"))
        authority = PublicationAuthority(
            hub.origin,
            object_storage_hosts=("127.0.0.1",),
            access_token="execution-access",
            expires_at=int(time.time()) + 3600,
        )
        worker = Worker(
            _config(tmp_path / "home"),
            WorkerOptions(
                root=tmp_path / "worker",
                tensorfs_root=tmp_path / "store",
                hubs=(authority,),
                accelerator_backend="none",
                devices="",
                sole_supervisor=True,
            ),
            InMemoryControlHost(),
        )
        try:
            yield worker, hub
        finally:
            worker.shutdown()


def preparing(worker: Worker, hub: FaultHub) -> Watched[None]:
    point = hub.checkpoint
    delegation = documents.canonical_bytes(
        pb.DownloadDelegation(
            models=[pb.DownloadModelRef(model=point.model, manifest=point.manifest)]
        )
    )
    return Watched(
        hub,
        lambda: machine_materialization.ensure(worker, delegation, hub=hub.origin),
        lambda: worker.activity[-1].step if worker.activity else "",
    )


def landed(worker: Worker, hub: FaultHub) -> None:
    preparing(worker, hub).result()
    assert worker.workspace is not None
    point = hub.checkpoint
    fill.store(worker.workspace.store_root).verify_checkpoint_source(
        point.model, point.manifest, point.length
    )
    assert temps(worker.workspace.store_root) == []


def refused(worker: Worker, hub: FaultHub) -> str:
    with pytest.raises(WorkspaceRefusal) as refusal:
        preparing(worker, hub).result()
    assert worker.workspace is not None
    assert temps(worker.workspace.store_root) == []
    return str(refusal.value)


@pytest.mark.parametrize("cut", [Reset(after=100), Truncate(after=100)])
def test_an_object_cut_mid_body_is_asked_again_and_lands(
    tmp_path: Path, cut: Reset | Truncate
) -> None:
    with machine(tmp_path, on(HEADER, 1, 1, cut)) as (worker, hub):
        landed(worker, hub)
        assert hub.asks(HEADER) == 2, hub.log()


def test_a_url_that_dies_mid_object_is_minted_again_and_lands(tmp_path: Path) -> None:
    dying = on(HEADER, 1, 1, Delay(2, Reset(after=10)))
    with machine(tmp_path, dying, lifetime=1) as (worker, hub):
        landed(worker, hub)
        assert hub.asks("presign") >= 2, hub.log()


def test_a_tunnel_page_in_front_of_the_hub_is_weather_never_not_found(tmp_path: Path) -> None:
    with machine(tmp_path, on("closure", 1, 1, tunnel_404())) as (worker, hub):
        landed(worker, hub)
        assert hub.asks("closure") == 2, hub.log()


def test_the_hubs_own_not_found_names_its_code_and_message(tmp_path: Path) -> None:
    missing = enveloped(404, "release.not_found", "no release 1.0.0 of proof/model")
    with machine(tmp_path, on("closure", 1, ALWAYS, missing)) as (worker, hub):
        reason = refused(worker, hub)
        assert "REF_NOT_FOUND" in reason and "no release 1.0.0 of proof/model" in reason
        assert hub.asks("closure") == 1, hub.log()


def test_wrong_bytes_forever_are_refused_by_identity_and_never_installed(tmp_path: Path) -> None:
    with machine(tmp_path, on(ASSET, 1, ALWAYS, Corrupt())) as (worker, hub):
        reason = refused(worker, hub)
        assert "OBJECT_ID_MISMATCH" in reason and ASSET in reason, reason
        assert worker.workspace is not None
        assert not fill.store(worker.workspace.store_root).contains(ASSET)


@pytest.mark.parametrize("stalled", ["closure", "header"])
def test_a_machine_stop_mid_transfer_ends_the_pull_as_stopped(tmp_path: Path, stalled: str) -> None:
    """A stalled Hub answer sends no progress event, so only a cancellation reaches it."""
    route = HEADER if stalled == "header" else stalled
    with machine(tmp_path, on(route, 1, ALWAYS, Stall(after=10))) as (worker, hub):
        pulling = preparing(worker, hub)
        hub.until(lambda: hub.asks(route) == 1)
        worker.request_stop()
        with pytest.raises(WorkspaceRefusal, match="machine stopped") as refusal:
            pulling.result()
        assert "TRANSFER_FAILED" not in str(refusal.value)
        assert worker.workspace is not None and temps(worker.workspace.store_root) == []


def test_a_hub_that_has_not_answered_shows_as_waiting_for_it(tmp_path: Path) -> None:
    """No timer on a live Hub's answer, but the wait is on the machine's activity lane, so a
    wedge is visible (and a stop cancels it, as above)."""
    answered = threading.Event()
    with machine(tmp_path, on("closure", 1, 1, Gate(answered))) as (worker, hub):
        pulling = preparing(worker, hub)
        point = hub.checkpoint
        hub.until(
            lambda: any(
                row.step.startswith(f"{point.model}: waiting for the Hub: ")
                for row in worker.activity
            ),
            lambda: len(worker.activity),
        )
        answered.set()
        pulling.result()


def test_a_dripping_object_does_not_hold_the_pull_to_its_crawl(tmp_path: Path) -> None:
    """Run 2322: a 33 GB pull landed its body at 400 MB/s, then eight objects crawled at
    6-50 KB/s for 15 minutes. Two objects here crawl at 16 KiB/s, 256 s each alone; the pull
    must finish far inside that, by restarting their slow requests (TensorFS 0.3.85 on)."""
    size, rate = 4 << 20, 16 << 10
    point = weights(tmp_path / "origin", tensors=16, size=size)
    parts = sorted(hex_ for hex_, body in point.bodies.items() if len(body) == size)
    crawling = [on(hex_, 1, ALWAYS, Drip(rate)) for hex_ in parts[:2]]

    def pull(root: Path, *rules: Rule) -> tuple[float, Path]:
        with machine(root, *rules, point=point) as (worker, hub):
            began = time.monotonic()
            landed(worker, hub)
            assert worker.workspace is not None
            return time.monotonic() - began, Path(worker.workspace.store_root)

    body, _ = pull(tmp_path / "body")
    tail, store_root = pull(tmp_path / "tail", *crawling)
    crawl = size / rate
    assert tail < crawl / 4, f"the tail crawled: {tail:.1f} s against a {body:.1f} s body"
    decisions = (store_root / "logs" / "transport.log").read_text()
    assert " restart " in decisions and "restarts=" in decisions
