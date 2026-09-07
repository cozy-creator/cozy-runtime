"""Live monotonic clocks must remain serializable through concurrent acquisition."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

from cozy_runtime.internal.worker.acquire import AcquisitionObservation, LegName
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def test_acquisition_clock_is_relative_and_safe_on_long_running_hosts() -> None:
    before = time.monotonic_ns()
    observation = AcquisitionObservation()
    observation.start("package")
    started = observation.snapshot().package.started_monotonic_ns
    after = time.monotonic_ns()
    assert 0 < started <= after - before + 1
    # This executes with an actual >110-day host clock in the private rental
    # regression; short-lived CI hosts prove the same bounded-origin contract.
    print(f"actual host monotonic nanoseconds: {after}; observed start: {started}")
    observation.finish("package")
    snapshot = observation.snapshot()
    raw, digest = documents.identity(snapshot)
    assert (
        documents.read(raw, pb.PlacementAcquisitionObservation)["package"]["started_monotonic_ns"]
        == started
    )
    assert len(digest) == 32


def test_parallel_acquisition_legs_keep_elapsed_time_and_progress() -> None:
    snapshots: list[pb.PlacementAcquisitionObservation] = []
    durations: dict[LegName, tuple[int, int]] = {}
    lock = Lock()

    def publish(snapshot: pb.PlacementAcquisitionObservation) -> None:
        with lock:
            snapshots.append(snapshot)

    observation = AcquisitionObservation(publish)
    barrier = Barrier(2)

    def execute(leg: LegName) -> None:
        before = time.monotonic_ns()
        observation.start(leg)
        inner_start = time.monotonic_ns()
        barrier.wait()
        for _ in range(8):
            observation.add(leg, downloaded=19, reused=23)
        Event().wait(0.002)
        inner_end = time.monotonic_ns()
        observation.finish(leg)
        after = time.monotonic_ns()
        with lock:
            durations[leg] = (inner_end - inner_start, after - before)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute, ("package", "model")))
    result = observation.snapshot()
    for name in ("package", "model"):
        leg = getattr(result, name)
        elapsed = leg.ended_monotonic_ns - leg.started_monotonic_ns
        assert durations[name][0] <= elapsed <= durations[name][1]
        assert leg.downloaded_bytes == 8 * 19
        assert leg.reused_bytes == 8 * 23
        assert leg.started_monotonic_ns > 0
    assert len(snapshots) == 20
    for snapshot in snapshots:
        # Exercise the same nested canonical serializer used by worker snapshots.
        documents.canonical_bytes(snapshot)
