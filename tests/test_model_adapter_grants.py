"""Normal serving admission retains every generated adapter source capability."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_lora_composition import compose, fixture
from test_machine_execution import ack, complete

from cozy_runtime.internal.worker import grants, machine_checkpoint_inputs
from cozy_runtime.internal.worker import machine_model_defaults as defaults
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def offer(bindings: list[pb.InputBinding], accesses: list[pb.InputAccess]) -> pb.AttemptOffer:
    spec = pb.InvocationSpec(
        inputs=bindings,
        serving=pb.ServingInvocationSpec(
            entrypoint_binding_digest="sha256:" + "11" * 32,
            attempt_binding_id="sha256:" + "11" * 32,
            bindings_digest="sha256:" + "11" * 32,
        ),
    )
    raw, digest = documents.identity(spec)
    return pb.AttemptOffer(
        request_id="adapted-serving",
        attempt_ordinal=1,
        invocation_spec_canonical_bytes=raw,
        invocation_spec_digest=digest,
        grant=pb.DeliveryGrant(invocation_spec_digest=digest, inputs=accesses),
    )


@pytest.mark.parametrize("parameter", ["model", "base_model"])
def test_generated_adapter_sources_bind_and_retain_without_parameter_collisions(
    tmp_path: Path,
    parameter: str,
) -> None:
    store, base, adapter, _ = fixture(tmp_path)
    composed, _ = compose(store, tmp_path, "composed", base, [(adapter, 0.8)])
    for name, artifact in (("base", base), ("style", adapter), ("composed", composed)):
        store.replace_local(
            None,
            name,
            artifact.tensorfs_receipt_digest,
            artifact.manifest.digest,
            artifact.manifest.length,
        )
    source: dict[str, object] = {
        "parameter": parameter,
        "repository": "local/base",
        "manifest": {"digest": base.manifest.digest, "length": base.manifest.length},
        "adapters": [
            {
                "model": "local/style",
                "manifest": adapter.manifest.digest,
                "manifest_length": adapter.manifest.length,
                "component": "transformer",
                "source_component": "adapter",
                "scale": "0.8",
            }
        ],
        "composed": {
            "model": "local/composed",
            "manifest": composed.manifest.digest,
            "manifest_length": composed.manifest.length,
        },
    }
    bindings, accesses = defaults.inputs(
        [
            source,
            dict(source, parameter=parameter + "_adapter_0", adapters=[], composed=None),
            {
                "parameter": "turbo_lora",
                "repository": "local/style",
                "manifest": {"digest": adapter.manifest.digest, "length": adapter.manifest.length},
            },
        ]
    )
    submitted = offer(bindings, accesses)
    selected = grants.model_inputs(
        grants.bind(
            documents.read(submitted.invocation_spec_canonical_bytes, pb.InvocationSpec),
            submitted.grant,
            submitted.invocation_spec_digest,
        )
    )
    assert set(selected) == {
        parameter,
        parameter + ".adapter.0",
        parameter + ".composed",
        parameter + "_adapter_0",
        "turbo_lora",
    }
    assert selected[parameter + ".adapter.0"].digest == documents.raw(adapter.manifest.digest)
    assert selected[parameter + ".composed"].digest == documents.raw(composed.manifest.digest)
    assert selected[parameter + "_adapter_0"].digest == documents.raw(base.manifest.digest)
    assert selected["turbo_lora"].digest == documents.raw(adapter.manifest.digest)
    workspace = Workspace(Path(store.root))
    with machine_checkpoint_inputs.admission(workspace, "owner", submitted.request_id):
        machine_checkpoint_inputs.preflight(workspace, "owner", submitted, selected)
        for entry in selected.values():
            machine_checkpoint_inputs.retain(workspace, "owner", submitted, entry)
        executions = Executions(workspace)
        executions.submit(
            "owner",
            "adapted-submit",
            b"x" * 32,
            submitted,
            expected_execution_workspace_id=executions.workspace_id,
        )
    with workspace.locked() as db:
        roots = [
            str(row["native_owner"])
            for row in db.execute("SELECT native_owner FROM execution_checkpoint_inputs")
        ]
    assert len(roots) == 5
    for root in roots:
        held = store.checkpoint_root(root)
        assert held is not None and not held["released"]
    outcome = complete(executions, submitted.request_id, 14)
    executions.acknowledge_collection("owner", ack(outcome))
    for root in roots:
        held = store.checkpoint_root(root)
        assert held is not None and held["released"]


@pytest.mark.parametrize(
    "parameter",
    [
        "model.adapter",
        "model.adapter.-1",
        "model.adapter.01",
        "model.adapter.0.extra",
        "model.composed.extra",
        "model..composed",
        "model/../composed",
        "model.other",
        "model.adapter.0.composed",
        "model.adapter.12345678901234567890",
    ],
)
def test_malformed_auxiliary_model_inputs_are_refused(parameter: str) -> None:
    bindings, accesses = defaults.inputs(
        [
            {
                "parameter": parameter,
                "repository": "local/base",
                "manifest": {
                    "digest": "sha256:" + "11" * 32,
                    "length": 100,
                },
            }
        ]
    )
    submitted = offer(bindings, accesses)
    with pytest.raises(grants.GrantRefusal, match="grant_model_identity"):
        grants.bind(
            documents.read(submitted.invocation_spec_canonical_bytes, pb.InvocationSpec),
            submitted.grant,
            submitted.invocation_spec_digest,
        )


@pytest.mark.parametrize("catalog", [False, True])
def test_auxiliary_sources_do_not_expand_authored_job_parameters(catalog: bool) -> None:
    bindings, accesses = defaults.inputs(
        [
            {
                "parameter": "model.adapter.0",
                "repository": "local/base",
                "manifest": {
                    "digest": "sha256:" + "11" * 32,
                    "length": 100,
                },
            }
        ]
    )
    if not catalog:
        accesses[0].ClearField("catalog_model")
    submitted = offer(bindings, accesses)
    spec = documents.from_body(
        documents.read(submitted.invocation_spec_canonical_bytes, pb.InvocationSpec),
        pb.InvocationSpec,
    )
    spec.ClearField("serving")
    spec.job.installation_id = "local-" + "11" * 16
    spec.job.job_descriptor_id = "fixture"
    raw, digest = documents.identity(spec)
    submitted.grant.invocation_spec_digest = digest
    with pytest.raises(grants.GrantRefusal, match="grant_model_identity"):
        grants.bind(documents.read(raw, pb.InvocationSpec), submitted.grant, digest)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("no_catalog", "grant_model_serving_refused"),
        ("repository", "grant_catalog_access"),
        ("mime", "grant_model_identity"),
        ("length", "grant_model_identity"),
        ("order", "grant_model_identity"),
        ("url", "grant_model_identity"),
    ],
)
def test_auxiliary_serving_sources_keep_exact_access_checks(mutation: str, code: str) -> None:
    bindings, accesses = defaults.inputs(
        [
            {
                "parameter": "model.composed",
                "repository": "local/base",
                "manifest": {
                    "digest": "sha256:" + "11" * 32,
                    "length": 100,
                },
            }
        ]
    )
    if mutation == "no_catalog":
        accesses[0].ClearField("catalog_model")
    elif mutation == "repository":
        accesses[0].catalog_model.repository = "local/../base"
    elif mutation == "mime":
        bindings[0].kind_mime = "application/json"
    elif mutation == "length":
        bindings[0].length = 0
    elif mutation == "order":
        bindings[0].order = 1
    elif mutation == "url":
        accesses[0].url = "model://sha256:" + "22" * 32
    submitted = offer(bindings, accesses)
    with pytest.raises(grants.GrantRefusal, match=code):
        grants.bind(
            documents.read(submitted.invocation_spec_canonical_bytes, pb.InvocationSpec),
            submitted.grant,
            submitted.invocation_spec_digest,
        )
