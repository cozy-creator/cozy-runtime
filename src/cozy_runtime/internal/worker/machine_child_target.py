"""Resolve a Python child from the accepted capture, never from a client install database."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import builtin_operations, job_plan, package_interface
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .session import Worker


@dataclass(frozen=True)
class Target:
    revision: str
    entrypoint: str
    interface_digest: str
    declaration: dict[str, Any]
    prepared_revision: dict[str, Any]
    binding: JobBinding | None
    digest: str
    serving_envelope: bool = False


def resolve(worker: Worker, parent: AttemptRecord, call: Call) -> Target:
    if worker.executions is None or parent.job is None:
        raise WorkspaceRefusal("child call needs an accepted captured job parent")
    owner = worker.fence.record_owner_id
    root = worker.executions.capture_root(owner, parent.request_id)
    capture = worker.executions.capture(owner, root)
    intent = canonical_json.decode(call.intent)
    if intent["module"] == builtin_operations.MODULE:
        assert worker.machine_calls is not None
        return worker.machine_calls.builtins.resolve(parent, call, intent)
    caller = parent.job.build_id
    selected = next(
        (
            binding
            for binding in capture["bindings"]
            if (
                binding["caller_revision_digest"],
                binding["interface_digest"],
                binding["module"],
                binding["export"],
            )
            == (caller, intent["interface_digest"], intent["module"], intent["export"])
        ),
        None,
    )
    if selected is None:
        raise WorkspaceRefusal("child callable is absent from the frozen capture")
    revision, entrypoint = selected["callee_revision_digest"], selected["entrypoint"]
    prepared = worker.executions.preparation(owner, root)["revisions"].get(revision)
    if prepared is None:
        raise WorkspaceRefusal("captured child environment is not retained")
    interface_ref = prepared["placement"]["package_interface"]
    if interface_ref["digest"] != intent["interface_digest"]:
        raise WorkspaceRefusal("captured child interface changed")
    interface_path = (
        Path(worker.options.artifact_cache or "") / documents.raw(interface_ref["digest"]).hex()
    )
    with interface_path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    if (
        len(raw) != interface_ref["length"]
        or "sha256:" + hashlib.sha256(raw).hexdigest() != interface_ref["digest"]
    ):
        raise WorkspaceRefusal("captured child interface bytes are unavailable")
    interface = package_interface.read_bytes(raw, "captured child")
    declared = next(
        (
            entry
            for entry in [*interface["jobs"], *interface["entrypoints"]]
            if entry["name"] == entrypoint
        ),
        None,
    )
    if declared is None:
        raise WorkspaceRefusal("captured child has no callable declaration")
    invocable = declared.get("invocable", {})
    if (invocable.get("module"), invocable.get("export")) != (intent["module"], intent["export"]):
        raise WorkspaceRefusal("captured child export changed")
    if invocable.get("memoize") and {"egress", "secrets"}.intersection(
        invocable.get("capabilities", [])
    ):
        raise WorkspaceRefusal("memoized child cannot carry effect or secret capabilities")
    parent_declaration = worker.engine._job_declaration(parent.job, parent.job.job_descriptor_id)
    if parent_declaration.get("invocable", {}).get("memoize") and not invocable.get("memoize"):
        raise WorkspaceRefusal("memoized parent cannot invoke an impure child")
    binding = None
    if declared in interface["jobs"]:
        descriptor = package_interface.job_descriptor_id(interface, entrypoint)
        binding = JobBinding.read(
            json.loads(
                job_plan.path(
                    worker.config.cozy_home / "job-plans", revision, descriptor
                ).read_bytes()
            )
        )
    identity = canonical_json.encode(
        {
            "local_revision_digest": revision,
            "interface_digest": intent["interface_digest"],
            "entrypoint": entrypoint,
            "module": intent["module"],
            "export": intent["export"],
        }
    )
    return Target(
        revision,
        entrypoint,
        intent["interface_digest"],
        declared,
        prepared,
        binding,
        "sha256:" + hashlib.sha256(identity).hexdigest(),
        binding is None and selected["caller_revision_digest"] != revision,
    )


def computation(
    target: Target,
    arguments: dict[str, Any],
    numerical: bytes,
    capture: dict[str, Any] | None = None,
) -> bytes:
    """Bind the callee's frozen implementation and inputs, excluding caller and custody."""
    inputs = dict(arguments)
    for model in target.declaration.get("models", []):
        parameter = model["path"].removeprefix(target.entrypoint + ".models.")
        value = inputs.get(parameter)
        if not isinstance(value, dict) or not isinstance(value.get("manifest"), dict):
            raise WorkspaceRefusal("memoized Model argument requires an exact retained manifest")
        inputs[parameter] = {"manifest": value["manifest"]}
    identity: dict[str, Any] = {
        "target_digest": target.digest,
        "installed_receipt_digest": target.prepared_revision["receipt_digest"],
        "inputs": inputs,
    }
    if capture is not None:
        identity["capture"] = capture
    base = "sha256:" + hashlib.sha256(canonical_json.encode(identity)).hexdigest()
    return hashlib.sha256(
        canonical_json.encode(
            {"computation_digest": base, "numerical_environment_digest": documents.spell(numerical)}
        )
    ).digest()
