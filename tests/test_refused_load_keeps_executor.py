"""A load refused before it touched the process fails its request; the executor keeps serving."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import pytest

from cozy_runtime.internal import child_env, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest, stage_development_placement
from cozy_runtime.internal.worker.attempts import AttemptRefusal, Released
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

SOURCE = """
import msgspec
from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct):
    pass


class Result(msgspec.Struct):
    value: int


@app.entrypoint
def serve(ctx: Context, payload: Request) -> Result:
    return Result(7)
"""


def _serve(
    worker: Worker, placement: pb.Placement, root: Path, request_id: str
) -> dict[str, object]:
    owner = LocalRecordOwner(
        LocalRequest(
            entrypoint="serve",
            payload={},
            outputs=(),
            request_id=request_id,
        ),
        {"serve": documents.spell(placement.entrypoints[0].entrypoint_binding_digest)},
        root / request_id,
        package_installation_id=placement.installation_id,
        placement=placement,
    )
    owner._dispatch()
    while True:
        frame = owner.host.inbound.get_nowait()
        if frame is not None and frame.HasField("attempt_offer"):
            break
    attempt = worker.engine.offer(frame.attempt_offer)
    worker._bind_attempt_slot(attempt)
    worker.engine.hold(attempt)
    worker.engine.prepare_and_accept(attempt)
    result = worker.engine.execute(attempt)
    outcome = worker.engine.finish(result) if isinstance(result, Released) else result
    return documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_refused_load_then_success_on_the_same_executor() -> None:
    with tempfile.TemporaryDirectory(prefix="cz-load.", dir="/tmp") as raw:
        root = Path(raw)
        project = root / "project"
        project.mkdir()
        (project / "package.toml").write_text('[application]\nobject="refusal_fixture:app"\n')
        (project / "refusal_fixture.py").write_text(SOURCE)
        home = root / "home"
        config = RuntimeConfig(
            cozy_home=home,
            credentials=Credentials(),
            child_base_env=tuple(
                sorted(
                    (key, value)
                    for key, value in os.environ.items()
                    if not child_env.erased(key) and key != "PYTHONPATH"
                )
            ),
        )
        staged = stage_development_placement(
            home,
            package_interface.build(discover(project)),
            package="local/refusal",
            release="0.0.0",
            installation_id="install-refusal",
            artifact_cache=root / "artifacts",
            tensorfs_root=root / "store",
            project=project,
        )
        placement = staged.placement
        worker = Worker(
            config,
            WorkerOptions(
                root=root / "worker",
                devices="0",
                tensorfs_root=staged.tensorfs_root,
                artifact_cache=staged.artifact_cache,
                development_project=project,
                development_package=placement.development.package,
                development_release=placement.development.release,
                development_installation_id=placement.installation_id,
            ),
            InMemoryControlHost(),
        )
        raw_set, digest = documents.identity(pb.PlacementSet(placements=[placement]))
        try:
            with contextlib.chdir(project):
                worker._apply_desired_state(
                    pb.DesiredWorkerState(
                        revision=1,
                        wire_minor=WIRE_MINOR,
                        posture=pb.POSTURE_ACCEPTING,
                        placement_set=pb.DesiredPlacementSet(
                            placement_set_digest=digest, placement_set_canonical_bytes=raw_set
                        ),
                    )
                )
                worker.converge_placement(1)
            executor = worker.supervision.current
            assert executor is not None and executor.alive()

            # The worker's own load path with a pin the executor refuses for a binding with
            # no attention site, before it touches the process: that load fails with the
            # request's refusal and the executor keeps serving.
            tenant = worker._tenant(placement.placement_id)
            group = list(tenant.bindings.values())
            with pytest.raises(AttemptRefusal) as refused:
                worker._load_construction(
                    tenant, executor, "sha256:" + "a" * 64, group, attention_pin="sdpa"
                )
            assert worker.supervision.current is executor and executor.alive()
            assert not executor.poisoned and not tenant.failed_bindings
            assert refused.value.code == "attention_kernel_unsupported"
            assert refused.value.cause == pb.CAUSE_CODE_INVALID_REQUEST

            served = _serve(worker, placement, root, "plain")
            assert served["status"] == pb.OUTCOME_STATUS_SUCCEEDED, served
            assert worker.supervision.current is executor and executor.alive()
        finally:
            worker.shutdown()
