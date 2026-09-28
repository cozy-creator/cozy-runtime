"""A real executor owner must be gone before its replacement measures the same pins.

These CPU callables exercise lifecycle, not GPU capacity. The corresponding H100
failure is retained separately: a live old owner left only 9 GB for group admission.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.development_launch import stage_development_placement
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.child import Executor, ExecutorSupervision
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from local_owner import LocalRecordOwner, LocalRequest, claimable
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
        package_installation_id=placement.development.installation_id,
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
def test_only_the_process_identity_decides_between_in_place_and_replacement() -> None:
    """proto-061 D2: a started process is kept for a new document of the same placement,
    package, installation, device pin and lane shape, and replaced for anything else."""
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
            installation_id="local-" + uuid.uuid4().hex,
            artifact_cache=root / "artifacts",
            tensorfs_root=root / "store",
            project=project,
        )
        placement = pb.Placement()
        placement.CopyFrom(staged.placement)
        placement.placement_id = "primary"
        raw, digest = documents.identity(pb.PlacementSet(placements=[placement]))
        worker = Worker(
            *claimable(
                config,
                WorkerOptions(
                    root=root / "worker",
                    devices="0",
                    tensorfs_root=staged.tensorfs_root,
                    artifact_cache=staged.artifact_cache,
                    development_project=project,
                    development_package=staged.placement.development.package,
                    development_release=staged.placement.development.release,
                    development_installation_id=staged.placement.development.installation_id,
                ),
            ),
            InMemoryControlHost(),
        )
        try:
            with contextlib.chdir(project):
                worker._apply_desired_state(
                    pb.DesiredWorkerState(
                        revision=1,
                        wire_minor=WIRE_MINOR,
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
            other_installation = worker._placement_from_entry(
                {**entry, "installation_id": "local-" + uuid.uuid4().hex}, b"next"
            )
            assert not worker._grows_in_place(outgoing, other_installation, worker.bindings)
            pinned = worker._placement_from_entry(entry, b"next")
            pinned.device_pin = (1,)
            assert not worker._grows_in_place(outgoing, pinned, worker.bindings)
            renamed = worker._placement_from_entry({**entry, "placement_id": "other"}, b"next")
            assert not worker._grows_in_place(outgoing, renamed, worker.bindings)
        finally:
            worker.shutdown()
