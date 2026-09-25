"""A real executor owner must be gone before its replacement measures the same pins.

These CPU callables exercise lifecycle, not GPU capacity. The corresponding H100
failure is retained separately: a live old owner left only 9 GB for group admission.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalRecordOwner,
    LocalRequest,
    development_source_digest,
    stage_development_placement,
)
from cozy_runtime.internal.worker.attempts import AttemptRecord, Released
from cozy_runtime.internal.worker.child import Executor, ExecutorSupervision
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config, _workspace, needs_executor


class ObservedWorker(Worker):
    prior: Executor | None = None
    neighbor: Executor | None = None
    before_assignment: list[tuple[bool, bool]]

    def _assign_lane(
        self,
        placement: Placement,
        bindings: Mapping[str, DeclaredBinding],
        supervision: ExecutorSupervision,
    ) -> bool:
        if placement.placement_id == "replacement" and self.prior is not None:
            assert self.neighbor is not None
            self.before_assignment.append((self.prior.alive(), self.neighbor.alive()))
        return super()._assign_lane(placement, bindings, supervision)


def wait_for(predicate: Callable[[], bool]) -> None:
    end = time.monotonic() + 20
    while not predicate():
        assert time.monotonic() < end, "local lifecycle fixture stopped making progress"
        time.sleep(0.005)


def offer(
    worker: Worker, placement: pb.Placement, root: Path, request_id: str, *, wait: bool
) -> AttemptRecord:
    owner = LocalRecordOwner(
        LocalRequest(entrypoint="serve", payload={"wait": wait}, outputs=(), request_id=request_id),
        {"serve": documents.spell(placement.entrypoints[0].entrypoint_binding_digest)},
        root / request_id,
        package_code_digest=documents.spell(placement.development.source_digest),
        placement=placement,
    )
    owner._dispatch()
    while True:
        frame = owner.host.inbound.get_nowait()
        assert frame is not None
        if frame.HasField("attempt_offer"):
            # The one-shot owner routes to plc-local; this fixture owns two
            # named placements and must route to the one it actually selected.
            frame.attempt_offer.placement_id = placement.placement_id
            attempt = worker.engine.offer(frame.attempt_offer)
            worker._bind_attempt_slot(attempt)
            worker.engine.hold(attempt)
            worker.engine.prepare_and_accept(attempt)
            return attempt


def acknowledge(worker: Worker, outcome: pb.AttemptOutcome) -> None:
    worker.on_outcome_ack(
        pb.AttemptOutcomeAck(
            worker_boot_id=worker.fence.worker_boot_id,
            record_owner_epoch=worker.fence.record_owner_epoch,
            control_stream_epoch=worker.fence.control_stream_epoch,
            request_id=outcome.request_id,
            attempt_ordinal=outcome.attempt_ordinal,
            invocation_spec_digest=outcome.invocation_spec_digest,
            outcome_id=outcome.outcome_id,
            outcome_digest=outcome.outcome_digest,
        )
    )


@needs_executor
def test_replacement_waits_for_work_then_retires_before_assignment() -> None:
    with _workspace() as root:
        project = root / "project"
        project.mkdir()
        (project / "package.toml").write_text('[application]\nobject="handoff_fixture:app"\n')
        (project / "handoff_fixture.py").write_text(
            "import time\nfrom pathlib import Path\nimport msgspec\n"
            "from cozy_runtime.author import App, Context\napp=App()\n"
            "class Request(msgspec.Struct):\n    wait: bool = False\n"
            "class Result(msgspec.Struct):\n    value: int\n"
            "@app.entrypoint\ndef serve(ctx: Context, payload: Request) -> Result:\n"
            "    if payload.wait:\n"
            f"        root=Path({str(root)!r})\n"
            "        (root/'entered').touch()\n"
            "        while not (root/'release').exists():\n"
            "            ctx.raise_if_cancelled()\n            time.sleep(0.005)\n"
            "    return Result(7)\n"
        )
        config = _config(root / "home")
        staged = stage_development_placement(
            config.cozy_home,
            package_interface.build(discover(project)),
            package="local/handoff",
            release="0.0.0",
            source_digest=development_source_digest(project),
            artifact_cache=root / "artifacts",
            tensorfs_root=root / "store",
            project=project,
        )
        initial, neighbor, replacement = (pb.Placement() for _ in range(3))
        for placement, name in (
            (initial, "primary"),
            (neighbor, "neighbor"),
            (replacement, "replacement"),
        ):
            placement.CopyFrom(staged.placement)
            placement.placement_id = name

        def desired(revision: int, primary: pb.Placement) -> pb.DesiredWorkerState:
            raw, digest = documents.identity(pb.PlacementSet(placements=[primary, neighbor]))
            return pb.DesiredWorkerState(
                revision=revision,
                wire_minor=49,
                posture=pb.POSTURE_ACCEPTING,
                placement_set=pb.DesiredPlacementSet(
                    placement_set_digest=digest,
                    placement_set_canonical_bytes=raw,
                    device_pins=[
                        pb.PlacementDevicePin(
                            placement_id=primary.placement_id, device_ordinals=[0]
                        ),
                        pb.PlacementDevicePin(placement_id="neighbor", device_ordinals=[1]),
                    ],
                ),
            )

        worker = ObservedWorker(
            config,
            WorkerOptions(
                root=root / "worker",
                devices="0,1",
                tensorfs_root=staged.tensorfs_root,
                artifact_cache=staged.artifact_cache,
                development_project=project,
                development_package=staged.placement.development.package,
                development_release=staged.placement.development.release,
                development_source_digest=documents.spell(
                    staged.placement.development.source_digest
                ),
            ),
            InMemoryControlHost(),
        )
        worker.before_assignment = []
        failures: list[BaseException] = []
        outcomes: list[pb.AttemptOutcome] = []
        threads: list[threading.Thread] = []
        try:
            with contextlib.chdir(project):
                worker._apply_desired_state(desired(1, initial))
                worker.converge_placement(1)
            worker.prior = worker.supervision.current
            worker.neighbor = worker.hosted["neighbor"].supervision.current
            assert worker.prior is not None and worker.prior.alive()
            assert worker.neighbor is not None and worker.neighbor.alive()

            # A new revision with the same placement identity does not discard a
            # usable generation. The later different identity must retire it.
            worker._apply_desired_state(desired(2, initial))
            worker.converge_placement(2)
            assert worker.supervision.current is worker.prior

            attempt = offer(worker, initial, root, "active", wait=True)

            def execute() -> None:
                try:
                    result = worker.engine.execute(attempt)
                    outcomes.append(
                        worker.engine.finish(result) if isinstance(result, Released) else result
                    )
                    acknowledge(worker, outcomes[-1])
                except BaseException as exc:
                    failures.append(exc)

            running = threading.Thread(target=execute)
            threads.append(running)
            running.start()
            wait_for(lambda: (root / "entered").exists() or bool(failures))
            assert not failures, failures
            worker._apply_desired_state(desired(3, replacement))

            def replace() -> None:
                try:
                    with contextlib.chdir(project):
                        worker.converge_placement(3)
                except BaseException as exc:
                    failures.append(exc)

            replacing = threading.Thread(target=replace)
            threads.append(replacing)
            replacing.start()
            wait_for(
                lambda: (
                    bool(worker.placement and worker.placement.serving == pb.SERVING_STATE_DRAINING)
                    or bool(failures)
                )
            )
            assert not failures, failures
            assert worker.prior.alive() and worker.neighbor.alive()
            assert worker.before_assignment == []
            (root / "release").touch()
            for thread in threads:
                thread.join(20)
                assert not thread.is_alive()
            assert not failures, failures
            assert outcomes
            assert (
                documents.read(outcomes[0].outcome_canonical_bytes, pb.AttemptOutcomeBody)["status"]
                == pb.OUTCOME_STATUS_SUCCEEDED
            )
            assert worker.before_assignment == [(False, True)]
            assert not worker.prior.alive()
            assert worker.supervision.current is not worker.prior
            assert worker.supervision.current is not None and worker.supervision.current.alive()
            assert worker.hosted["neighbor"].supervision.current is worker.neighbor
            assert worker.neighbor.alive()
            after = offer(worker, replacement, root, "after", wait=False)
            result = worker.engine.execute(after)
            terminal = worker.engine.finish(result) if isinstance(result, Released) else result
            assert (
                documents.read(terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)["status"]
                == pb.OUTCOME_STATUS_SUCCEEDED
            )
            acknowledge(worker, terminal)
        finally:
            (root / "release").touch()
            for thread in threads:
                thread.join(20)
            worker.shutdown()


@needs_executor
def test_only_the_process_identity_decides_between_in_place_and_replacement() -> None:
    """proto-061 D2: a started process is kept for a new document of the same placement,
    package, environment, device pin and lane shape, and replaced for anything else."""
    with _workspace() as root:
        project = root / "project"
        project.mkdir()
        (project / "package.toml").write_text('[application]\nobject="identity_fixture:app"\n')
        (project / "identity_fixture.py").write_text(
            "import msgspec\nfrom cozy_runtime.author import App, Context\napp=App()\n"
            "class Request(msgspec.Struct):\n    wait: bool = False\n"
            "class Result(msgspec.Struct):\n    value: int\n"
            "@app.entrypoint\ndef serve(ctx: Context, payload: Request) -> Result:\n"
            "    return Result(7)\n"
        )
        config = _config(root / "home")
        staged = stage_development_placement(
            config.cozy_home,
            package_interface.build(discover(project)),
            package="local/identity",
            release="0.0.0",
            source_digest=development_source_digest(project),
            artifact_cache=root / "artifacts",
            tensorfs_root=root / "store",
            project=project,
        )
        placement = pb.Placement()
        placement.CopyFrom(staged.placement)
        placement.placement_id = "primary"
        raw, digest = documents.identity(pb.PlacementSet(placements=[placement]))
        worker = Worker(
            config,
            WorkerOptions(
                root=root / "worker",
                devices="0",
                tensorfs_root=staged.tensorfs_root,
                artifact_cache=staged.artifact_cache,
                development_project=project,
                development_package=staged.placement.development.package,
                development_release=staged.placement.development.release,
                development_source_digest=documents.spell(
                    staged.placement.development.source_digest
                ),
            ),
            InMemoryControlHost(),
        )
        try:
            with contextlib.chdir(project):
                worker._apply_desired_state(
                    pb.DesiredWorkerState(
                        revision=1,
                        wire_minor=49,
                        posture=pb.POSTURE_ACCEPTING,
                        placement_set=pb.DesiredPlacementSet(
                            placement_set_digest=digest, placement_set_canonical_bytes=raw
                        ),
                    )
                )
                worker.converge_placement(1)
            outgoing = worker.placement
            executor = worker.supervision.current
            assert outgoing is not None and executor is not None and executor.started
            assert outgoing.serving == pb.SERVING_STATE_DISPATCHABLE
            # the weightless construction is built with the process, before any attempt
            assert executor.loaded_bindings() == set(worker.bindings)
            entry = dict(outgoing.document)
            same = worker._placement_from_entry(entry, b"next")
            assert worker._grows_in_place(outgoing, same, worker.bindings)
            other_env = worker._placement_from_entry(
                {**entry, "environment_digest": "sha256:" + "e" * 64}, b"next"
            )
            assert not worker._grows_in_place(outgoing, other_env, worker.bindings)
            pinned = worker._placement_from_entry(entry, b"next")
            pinned.device_pin = (1,)
            assert not worker._grows_in_place(outgoing, pinned, worker.bindings)
            renamed = worker._placement_from_entry({**entry, "placement_id": "other"}, b"next")
            assert not worker._grows_in_place(outgoing, renamed, worker.bindings)
        finally:
            worker.shutdown()
