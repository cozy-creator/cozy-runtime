"""Cold model work queues before construction; only device entry needs loaded weights."""

import base64
import json
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.attempts import AttemptRefusal
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import Placement, Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config

BINDING = "sha256:" + "b" * 64
ENVIRONMENT = "sha256:" + "e" * 64


def offer(
    request: str, *, binding: str = BINDING, environment: str = ENVIRONMENT, seed: int = 0
) -> pb.AttemptOffer:
    payload = json.dumps({"seed": seed}).encode()
    payload_digest = documents.spell(documents.digest_of(payload))
    raw, digest = documents.identity(
        pb.InvocationSpec(
            environment_digest=environment,
            payload_digest=payload_digest,
            inputs=[
                pb.InputBinding(
                    input_id="payload",
                    digest=payload_digest,
                    length=len(payload),
                    kind_mime="application/json",
                )
            ],
            serving=pb.ServingInvocationSpec(
                entrypoint_binding_digest=binding,
                attempt_binding_id=binding,
                bindings_digest=documents.spell(b"s" * 32),
            ),
        )
    )
    return pb.AttemptOffer(
        request_id=request,
        attempt_ordinal=1,
        placement_id="model",
        invocation_spec_digest=digest,
        invocation_spec_canonical_bytes=raw,
        grant=pb.DeliveryGrant(
            invocation_spec_digest=digest,
            inputs=[
                pb.InputAccess(
                    input_id="payload",
                    url="data:application/json;base64," + base64.b64encode(payload).decode(),
                )
            ],
        ),
    )


def test_cold_model_queue_validates_without_loading_or_losing_faults(tmp_path: Path) -> None:
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(root=tmp_path / "worker"),
        InMemoryControlHost(),
    )
    worker.placement = Placement(
        placement_id="model",
        placement_set_digest=b"p" * 32,
        package={},
        development={},
        environment_digest=bytes.fromhex("e" * 64),
        package_interface=("", 0),
        bindings_digest=b"s" * 32,
        models=(),
        entrypoints=(),
        environment={},
        document={},
    )
    worker.bindings[BINDING] = DeclaredBinding(
        entrypoint_binding_digest=BINDING,
        entrypoint="generate",
        model_class="fixture:Model",
        model_binding_path="model",
        model_parameter_name="model",
        release="sha256:" + "d" * 64,
        environment_digest=ENVIRONMENT,
    )
    try:
        assert not worker.bindings[BINDING].weightless
        assert not worker.prepared_models
        for index in range(5):
            attempt = worker.engine.offer(offer(f"cold-{index}", seed=index))
            worker._bind_attempt_slot(attempt)
            worker.engine.hold(attempt)
            accepted = worker.engine.stage(attempt)
            assert accepted.request_id == attempt.request_id
            assert attempt.state == "staged" and attempt.payload == {"seed": index}
            assert attempt.prepared_model is None and attempt.executor is None
            assert attempt.spool is not None and attempt.spool.is_dir()
        assert worker.supervision.current is None and not worker.prepared_models
        # A queue acceptance is not permission to execute without construction.
        assert worker.engine.claim_staged(attempt)
        with pytest.raises(AttemptRefusal) as unloaded:
            worker.engine.enter(attempt)
        assert unloaded.value.code == "construction_not_loaded"

        for incoming, expected in (
            (offer("unknown", binding="sha256:" + "a" * 64), "unknown_binding"),
            (offer("other-environment", environment="sha256:" + "a" * 64), "environment_mismatch"),
        ):
            refused = worker.engine.offer(incoming)
            worker._bind_attempt_slot(refused)
            with pytest.raises(AttemptRefusal) as failure:
                worker.engine.stage(refused)
            assert failure.value.code == expected

        worker.failed_bindings[BINDING] = "model_asset_missing: tokenizer/vocab.json"
        failed = worker.engine.offer(offer("failed-load"))
        worker._bind_attempt_slot(failed)
        with pytest.raises(AttemptRefusal) as failure:
            worker.engine.stage(failed)
        assert failure.value.code == "binding_faulted"
        assert "model_asset_missing" in failure.value.detail
    finally:
        worker.shutdown()
