"""Internal driver for the public banked-acceptance CLI proof.

Shell owns the installed command boundaries. This driver assembles a real provisioned Marco/Polo
overlay, drives it once through LocalRecordOwner -> Worker -> disposable executor, then writes the
closed request whose evidence the installed CLI banks. No alternate handler path exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Any, cast

# This driver is executed from `/` to prove the installed Runtime. Its sibling support
# module is test mechanics, not an import path inherited from the source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from proof_data import installed_base_ownership, inventory, pack_installed, wheel

from cozy_runtime.internal import canonical, descriptor, local, overlay
from cozy_runtime.internal.config import read_config
from cozy_runtime.internal.placement_materialization import package_overlay_input
from cozy_runtime.internal.worker.session import WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class ProofFailure(RuntimeError):
    """An observed banked-acceptance fact disagreed with its contract."""


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup")
    setup.add_argument("root", type=Path)
    setup.add_argument("project_wheel", type=Path)
    setup.add_argument("package_descriptor", type=Path)
    run = sub.add_parser("run")
    run.add_argument("root", type=Path)
    failure = sub.add_parser("failure")
    failure.add_argument("root", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "setup":
        _setup(args.root, args.project_wheel, args.package_descriptor)
    elif args.command == "run":
        _run(args.root)
    elif args.command == "failure":
        _failure(args.root)
    else:
        _verify(args.root)
    return 0


def _setup(root: Path, project_wheel: Path, descriptor_path: Path) -> None:
    root = root.resolve()
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    base_wheels = root / "base-wheels"
    base_wheels.mkdir()
    environment_root = root / "environment"
    artifact_store = root / "artifact-store"
    artifact_store.mkdir()

    descriptor_body = descriptor.read_bytes(descriptor_path.read_bytes(), str(descriptor_path))
    descriptor_raw = descriptor.canonical_bytes(descriptor_body)
    project = inventory(project_wheel.resolve())
    base = inventory(pack_installed(base_wheels, "cozy-runtime"))
    distributions, roots = installed_base_ownership()
    distributions = [row for row in distributions if row["distribution"] != project.distribution]
    filtered_roots: list[dict[str, object]] = []
    for row in roots:
        providers = [
            provider
            for provider in cast(list[str], row["providers"])
            if provider != project.distribution
        ]
        if providers:
            filtered_roots.append({"providers": providers, "root": row["root"]})
    roots = filtered_roots
    target = _target()
    profile = {
        "accelerator_build": "cpu",
        "os_cpu": "linux-x86",
        "python_abi": target["python_abi"],
        "torch_release": "2.13.0",
    }
    wheelhouse_raw = canonical.write(
        {
            "base_distributions": distributions,
            "base_import_roots": roots,
            "compatibility_profile": profile,
            "format": "WheelhouseManifest/3",
            "minimum_cuda_version": "none",
            "minimum_driver_version": "none",
            "platform_target": target,
            "wheels": [wheel(base, import_roots=True)],
        }
    )
    resolved_raw = canonical.write(
        {
            "compatibility_profile": profile,
            "format": "ResolvedWheelSet/3",
            "overlay_bytes": 0,
            "platform_target": target,
            "resolution_lock_digest": _digest(b"acceptance-lock"),
            "resolution_lock_length": len(b"acceptance-lock"),
            "wheelhouse_manifest_digest": _digest(wheelhouse_raw),
            "wheels": [],
        }
    )
    bundle_raw = canonical.write(
        {
            "package_descriptor": _ref(descriptor_raw),
            "package_release_id": "local/marco@acceptance-1",
            "format": "tensorhub.package_bundle/3",
            "project_wheel": wheel(project, import_roots=True),
            "resolved_wheel_set": _ref(resolved_raw),
            "source_tree": _ref(b"source tree"),
            "wheelhouse_manifest": _ref(wheelhouse_raw),
        }
    )
    environment_spec_raw = documents.canonical_bytes(
        pb.PackageEnvironmentSpec(
            compatibility_profile=pb.CompatibilityProfile(**profile),
            platform_target=pb.PlatformTarget(**target),
            package_bundle_digest=documents.digest_of(bundle_raw),
            project_wheel_digest=documents.raw(project.digest),
            wheelhouse_manifest_digest=documents.digest_of(wheelhouse_raw),
        )
    )
    source = package_overlay_input(
        environment_spec_bytes=environment_spec_raw,
        package_bundle_bytes=bundle_raw,
        resolved_wheel_set_bytes=resolved_raw,
        wheelhouse_manifest_bytes=wheelhouse_raw,
        wheel_paths={project.digest: project_wheel.resolve()},
    )
    installed = overlay.materialize(
        source,
        environment_root,
        python=Path(sys.executable),
        base_manifest=_write(inputs / "wheelhouse-manifest.json", wheelhouse_raw),
    )
    _write(inputs / "installed-environment-receipt.json", installed.receipt_bytes)
    _write(inputs / "package-environment-spec.json", environment_spec_raw)
    _write(inputs / "package-bundle.json", bundle_raw)
    _write(inputs / "resolved-wheel-set.json", resolved_raw)
    _write(inputs / "package-descriptor.json", descriptor_raw)

    closure = local.weightless_release_documents(descriptor_body)
    plan_rows: list[dict[str, Any]] = []
    for entrypoint, raw in sorted(closure.plans.items()):
        path = _write(inputs / f"plan-{entrypoint}.json", raw)
        plan_rows.append({"kind": "entrypoint_binding_plan", "ref": _local_ref(path)})
    object_set_path = _write(inputs / "artifact-object-set.json", closure.artifact_object_set)
    object_set_ref = _ref(closure.artifact_object_set)
    execution_raw = canonical.write(
        {
            "canonical_execution": {
                "build_record_digest": _digest(b"build-record"),
                "package_descriptor_digest": _digest(descriptor_raw),
                "package_control_manifest_digest": _digest(b"control-manifest"),
                "environment_spec_digest": _digest(environment_spec_raw),
                "installed_environment_receipt_digest": installed.receipt_digest,
            },
            "compatible_accelerator_models": ["CPU"],
            "function": "marco",
            "kind": "tensorhub.package_execution/2",
            "model_roots": [],
        }
    )
    execution_path = _write(inputs / "package-execution.json", execution_raw)
    binding_release_raw = canonical.write(
        {
            "package_descriptor": _ref(descriptor_raw),
            "documents": [
                {"kind": row["kind"], "ref": _public_ref(row["ref"])} for row in plan_rows
            ],
            "package_execution_digest": _digest(execution_raw),
            "environment_spec": _ref(environment_spec_raw),
            "format": "tensorhub.package_binding_release/1",
            "installed_environment_receipt": _ref(installed.receipt_bytes),
            "object_set_digest": str(object_set_ref["digest"]),
        }
    )
    _write(inputs / "package-binding-release.json", binding_release_raw)
    fixture_request = {
        "package_descriptor": _local_ref(inputs / "package-descriptor.json"),
        "entrypoint": "marco",
        "format": "cozy.runtime.AcceptanceFixtureRequest/1",
        "input_root": str(inputs),
    }
    _json(root / "fixture-request.json", fixture_request)
    state = {
        "artifact_store": str(artifact_store),
        "binding_documents": plan_rows,
        "control_runtime_digest": base.digest,
        "package_descriptor_digest": _digest(descriptor_raw),
        "package_release_id": "local/marco@acceptance-1",
        "environment_root": str(environment_root),
        "environment_spec_digest": _digest(environment_spec_raw),
        "installed_environment_receipt_digest": installed.receipt_digest,
        "object_set": _local_ref(object_set_path),
        "execution": _local_ref(execution_path),
    }
    _json(root / "state.json", state)


def _run(root: Path) -> None:
    root = root.resolve()
    inputs = root / "inputs"
    state = _read_json(root / "state.json")
    payload_raw = (root / "fixture-payload.json").read_bytes()
    payload = canonical.parse_canonical(payload_raw)
    if payload != {"message": "marco"}:
        raise ProofFailure(f"fixture changed: {payload!r}")
    descriptor_body = descriptor.read_bytes((inputs / "package-descriptor.json").read_bytes())
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(root / "home"),
        }
    )
    facts = local.LocalPlacementFacts(
        environment_spec_digest=str(state["environment_spec_digest"]),
        installed_environment_receipt_digest=str(state["installed_environment_receipt_digest"]),
        package_descriptor_digest=str(state["package_descriptor_digest"]),
        environment_root=Path(str(state["environment_root"])),
        wheelhouse_manifest=inputs / "wheelhouse-manifest.json",
    )
    outcome = local.run_slice(
        config,
        local.LocalRequest(
            entrypoint="marco",
            payload=payload,
            outputs=(),
            request_id="acceptance-marco",
        ),
        workspace=root / "ordinary-attempt",
        release_id=str(state["package_release_id"]),
        package_descriptor=descriptor_body,
        placement_facts=facts,
        options=WorkerOptions(
            root=root / "ordinary-attempt" / "worker",
            release_id=str(state["package_release_id"]),
            control_runtime_digest=str(state["control_runtime_digest"]),
            no_warm=True,
        ),
        delete_journal=False,
    )
    if outcome.status != "OUTCOME_STATUS_SUCCEEDED" or outcome.result != {"message": "polo"}:
        raise ProofFailure(
            f"ordinary handler failed: {outcome.status} {outcome.result} {outcome.error}"
        )
    placement_path = _write(inputs / "placement-set.json", outcome.placement_set_canonical_bytes)
    invocation_path = _write(inputs / "invocation-spec.json", outcome.invocation_canonical_bytes)
    outcome_path = _write(inputs / "attempt-outcome.json", outcome.terminal_canonical_bytes)
    fixture_path = _write(inputs / "fixture-payload.json", payload_raw)
    accepted = {
        "attempt_ordinal": int(outcome.accepted["attempt_ordinal"]),
        "executor_generation": int(outcome.accepted["executor_generation"]),
        "invocation_spec_digest": str(outcome.accepted["invocation_spec_digest"]),
        "model_construction_digest": str(outcome.accepted["model_construction_digest"]),
        "placement_id": str(outcome.accepted["placement_id"]),
        "plan_digest": str(outcome.accepted["plan_digest"]),
        "request_id": str(outcome.accepted["request_id"]),
    }
    request = {
        "accelerator_qualification": outcome.accelerator_qualification,
        "artifact_object_set": state["object_set"],
        "artifact_store": state["artifact_store"],
        "attempt_accepted": accepted,
        "attempt_outcome": _local_ref(outcome_path),
        "binding_documents": state["binding_documents"],
        "control_runtime_digest": state["control_runtime_digest"],
        "package_binding_release": _local_ref(inputs / "package-binding-release.json"),
        "package_bundle": _local_ref(inputs / "package-bundle.json"),
        "package_descriptor": _local_ref(inputs / "package-descriptor.json"),
        "package_environment_spec": _local_ref(inputs / "package-environment-spec.json"),
        "package_execution": state["execution"],
        "entrypoint": "marco",
        "environment_root": state["environment_root"],
        "fixture_payload": _local_ref(fixture_path),
        "format": "cozy.runtime.ExecutionObservationRequest/1",
        "input_root": str(inputs),
        "installed_environment_receipt": _local_ref(inputs / "installed-environment-receipt.json"),
        "invocation_spec": _local_ref(invocation_path),
        "observed_placement_set_digest": outcome.placement_set_digest,
        "placement_set": _local_ref(placement_path),
        "resolved_wheel_set": _local_ref(inputs / "resolved-wheel-set.json"),
        "wheelhouse_manifest": _local_ref(inputs / "wheelhouse-manifest.json"),
        "worker_resources": outcome.worker_resources,
    }
    _json(root / "observation-request.json", request)


def _verify(root: Path) -> None:
    value = canonical.parse_canonical((root / "execution-observation.json").read_bytes())
    if not isinstance(value, dict):
        raise ProofFailure("observation is not an object")
    expected = {
        "format": "cozy.runtime.ExecutionObservation/1",
        "boot_warmup": False,
        "entrypoint": "marco",
        "execution_started": True,
        "logical_model_digest": "",
        "model_construction_contract_digests": [],
        "result_digest": _digest(b'{"message":"polo"}'),
        "status": int(pb.OutcomeStatus.OUTCOME_STATUS_SUCCEEDED),
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ProofFailure(f"{field}: got {value.get(field)!r}, wanted {wanted!r}")
    manifest = value.get("output_manifest")
    if manifest != {"outputs": []}:
        raise ProofFailure(f"weightless manifest is not explicit empty: {manifest!r}")
    if value.get("payload_digest") != _digest(b'{"message":"marco"}'):
        raise ProofFailure("observation did not bind the exact Marco fixture")
    if "candidate_id" in value or "qualification_id" in value or "outputs_aborted" in value:
        raise ProofFailure("observation claimed a Hub-owned correlation or abort fact")
    failed = canonical.parse_canonical((root / "failed-observation.json").read_bytes())
    if not isinstance(failed, dict) or failed.get("status") != int(
        pb.OutcomeStatus.OUTCOME_STATUS_FAILED
    ):
        raise ProofFailure("failure observation was not banked neutrally")
    if failed.get("execution_started") is not True or failed.get("result_digest") != "":
        raise ProofFailure("failure observation invented success or dropped execution_started")
    print("acceptance observation: ordinary Marco -> Polo attempt banked with exact closure")


def _failure(root: Path) -> None:
    root = root.resolve()
    request = _read_json(root / "observation-request.json")
    accepted = request["attempt_accepted"]
    if not isinstance(accepted, dict):
        raise ProofFailure("attempt_accepted is not an object")
    raw = documents.canonical_bytes(
        pb.AttemptOutcomeBody(
            request_id=str(accepted["request_id"]),
            attempt_ordinal=int(accepted["attempt_ordinal"]),
            invocation_spec_digest=str(accepted["invocation_spec_digest"]),
            status=pb.OutcomeStatus.OUTCOME_STATUS_FAILED,
            safe_message="planted qualification failure",
            cause=pb.OutcomeCause(
                code=pb.CauseCode.CAUSE_CODE_EXECUTOR_FAULT,
                origin=pb.CauseOrigin.CAUSE_ORIGIN_EXECUTOR,
            ),
            execution_started=True,
        )
    )
    path = _write(root / "inputs" / "failed-attempt-outcome.json", raw)
    request["attempt_outcome"] = _local_ref(path)
    _json(root / "failed-request.json", request)


def _target() -> dict[str, str]:
    libc_name, libc_version = platform.libc_ver()
    machine = platform.machine().lower()
    return {
        "accelerator_abi": "none",
        "accelerator_backend": "none",
        "libc": f"{libc_name}{libc_version}",
        "os_arch": f"linux/{'amd64' if machine in {'x86_64', 'amd64'} else machine}",
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def _ref(raw: bytes) -> dict[str, Any]:
    return {"digest": _digest(raw), "length": len(raw)}


def _local_ref(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {**_ref(raw), "path": str(path.resolve())}


def _public_ref(local_ref: object) -> dict[str, Any]:
    if not isinstance(local_ref, dict):
        raise ProofFailure("local ref is not an object")
    return {"digest": local_ref["digest"], "length": local_ref["length"]}


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path.resolve()


def _json(path: Path, value: object) -> None:
    path.write_bytes(canonical.write(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = canonical.parse_canonical(path.read_bytes())
    if not isinstance(value, dict):
        raise ProofFailure(f"{path} is not an object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
