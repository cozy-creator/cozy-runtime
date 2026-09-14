from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import tensorfs

from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import grants, machine_checkpoint_inputs
from cozy_runtime.internal.worker import machine_model_defaults as defaults
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import ack, complete
from test_model_runtime_closure import _snapshot

REVISION = "sha256:" + "11" * 32
MANIFEST = {"digest": "sha256:" + "22" * 32, "length": 321}


def captured() -> dict[str, Any]:
    return {
        "bindings": [{"callee_revision_digest": REVISION, "entrypoint": "generate"}],
        "model_defaults": [
            {
                "callee_revision_digest": REVISION,
                "entrypoint": "generate",
                "parameter": "adapter",
                "public_origin": "https://catalog.example",
                "rungs": [
                    {"gpu": "H100", "repository": "proof/adapter", "manifest": MANIFEST},
                    {"gpu": "*", "repository": "proof/adapter", "manifest": MANIFEST},
                ],
            },
            {
                "callee_revision_digest": REVISION,
                "entrypoint": "generate",
                "parameter": "model",
                "unavailable_code": "model_default_unbound",
            },
        ],
    }


def peer(capture: dict[str, Any]) -> Worker:
    return cast(
        Worker,
        SimpleNamespace(
            config=SimpleNamespace(bootstrap_credential="local-bootstrap"),
            options=SimpleNamespace(accelerator_backend="none"),
            executions=SimpleNamespace(capture_root=lambda *_: "root", capture=lambda *_: capture),
        ),
    )


def target() -> Target:
    return Target(
        revision=REVISION,
        entrypoint="generate",
        interface_digest=REVISION,
        declaration={
            "models": [{"path": "generate.models.adapter"}, {"path": "generate.models.model"}]
        },
        prepared_revision={},
        binding=None,
        digest=REVISION,
    )


def test_default_capture_requires_complete_exact_inventory_and_capable_peer() -> None:
    body = captured()
    interfaces = {REVISION: {"entrypoints": [{"name": "generate", **target().declaration}]}}
    request = pb.MachineExecutionSubmit(claim=pb.Claim(wire_minor=56))
    defaults.verify(peer(body), request, body, interfaces)
    mutations: list[Callable[[dict[str, Any]], Any]] = [
        lambda value: value["model_defaults"].pop(),
        lambda value: value["model_defaults"].reverse(),
        lambda value: value["model_defaults"][0].update(parameter="foreign"),
        lambda value: value["model_defaults"][0].update(
            public_origin="https://user:secret@catalog.example"
        ),
        lambda value: value["model_defaults"][0].update(
            public_origin="https://catalog.example/foreign"
        ),
        lambda value: value["model_defaults"][0].update(
            unavailable_code="model_default_unavailable"
        ),
        lambda value: value["model_defaults"][0]["rungs"][0].update(repository="local/forged"),
        lambda value: value["model_defaults"][0]["rungs"][0].update(gpu="*"),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(body)
        mutate(changed)
        with pytest.raises(WorkspaceRefusal):
            defaults.verify(peer(changed), request, changed, interfaces)
    request.claim.wire_minor = 55
    with pytest.raises(WorkspaceRefusal, match="wire56"):
        defaults.verify(peer(body), request, body, interfaces)
    assert defaults.valid_origin("http://127.0.0.1:8819", local=True)
    assert not defaults.valid_origin("http://127.0.0.1:8819", local=False)
    assert not defaults.valid_origin("http://169.254.169.254", local=True)


def test_unused_or_overridden_defaults_do_not_touch_catalog_or_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("unused default reached TensorFS")

    monkeypatch.setattr(fill, "tensorfs_module", forbidden)
    empty_peer = cast(Worker, SimpleNamespace())
    call = cast(Call, SimpleNamespace(parent_request="root"))
    assert defaults.select(empty_peer, "owner", call, target(), {"model": {}, "adapter": {}}) == []
    defaults.materialize(empty_peer, [], ExitStack())
    worker = peer(captured())
    selected = defaults.select(worker, "owner", call, target(), {"model": {}, "adapter": None})
    assert (
        len(selected) == 1 and selected[0]["parameter"] == "adapter" and selected[0]["gpu"] == "*"
    )
    with pytest.raises(WorkspaceRefusal, match="unavailable for model"):
        defaults.select(worker, "owner", call, target(), {"adapter": {}})
    assert defaults.matches("H100", "NVIDIA H100 80GB HBM3")
    assert not defaults.matches("H100", "NVIDIA H200")
    assert not defaults.matches("H100", "")


def test_serving_default_uses_real_catalog_input_pin_without_derived_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, length, store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    operation = store.begin_operation("default-fixture", "proof", "adapter")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1.0.0", "bf16", manifest, length)
    workspace = Workspace(root)
    row = {
        "parameter": "adapter",
        "public_origin": "https://unreachable.example",
        "repository": "proof/adapter",
        "manifest": {"digest": manifest, "length": length},
    }
    worker = peer({})
    worker.workspace = workspace

    def no_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an already verified exact native checkpoint was downloaded")

    monkeypatch.setattr(tensorfs, "pull", no_download)
    with ExitStack() as read_access:
        defaults.materialize(worker, [row], read_access)
        bindings, accesses = defaults.inputs([row])
        spec = pb.InvocationSpec(
            inputs=bindings,
            serving=pb.ServingInvocationSpec(
                entrypoint_binding_digest=REVISION,
                attempt_binding_id=REVISION,
                bindings_digest=REVISION,
            ),
        )
        raw, digest = documents.identity(spec)
        offer = pb.AttemptOffer(
            request_id="serving-child",
            attempt_ordinal=1,
            invocation_spec_canonical_bytes=raw,
            invocation_spec_digest=digest,
            grant=pb.DeliveryGrant(invocation_spec_digest=digest, inputs=accesses),
        )
        entries = grants.model_inputs(
            grants.bind(documents.read(raw, pb.InvocationSpec), offer.grant, digest)
        )
        with pytest.raises(WorkspaceRefusal, match="not retained"), workspace.locked() as db:
            Workspace.accept_in(db, "owner", offer)
        with workspace.locked() as db:
            assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0
        with machine_checkpoint_inputs.admission(workspace, "owner", offer.request_id):
            machine_checkpoint_inputs.preflight(workspace, "owner", offer, entries)
            for entry in entries.values():
                machine_checkpoint_inputs.retain(workspace, "owner", offer, entry)
            executions = Executions(workspace)
            executions.submit("owner", "serving-input", b"x" * 32, offer)
    with workspace.locked() as db:
        pin = db.execute(
            "SELECT * FROM execution_checkpoint_inputs WHERE recipient='serving-child'"
        ).fetchone()
        assert pin["input_id"] == "model:adapter" and pin["repository"] == "proof/adapter"
        assert pin["state"] == "held"
        for table in ("weights", "native_calls", "holds", "execution_model_holds"):
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    tensorfs.gc(store.root)
    restored = Executions(Workspace(root))
    result = complete(restored, "serving-child", 17)
    restored.acknowledge_collection("owner", ack(result))
    assert store.checkpoint_root(pin["native_owner"])["released"]
