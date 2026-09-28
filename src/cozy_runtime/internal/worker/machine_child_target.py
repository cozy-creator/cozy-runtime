"""Resolve a Python child from the accepted capture, never from a client install database."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import builtin_operations, job_plan, package_interface
from cozy_runtime.internal.worker import machine_deferred
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .session import Worker


@dataclass(frozen=True, slots=True)
class Target:
    installation_id: str
    entrypoint: str
    declaration: dict[str, Any]
    prepared_installation: dict[str, Any]
    binding: JobBinding | None
    operation_identity: str
    serving_envelope: bool = False
    callee: str = ""  # the capture's name for it: an installation id or a deferred key


def resolve(worker: Worker, parent: AttemptRecord, call: Call) -> Target:
    if worker.executions is None or parent.job is None:
        raise WorkspaceRefusal("child call needs an accepted captured job parent")
    intent = canonical_json.decode(call.intent)
    if intent["module"] == builtin_operations.MODULE:
        assert worker.machine_calls is not None
        return worker.machine_calls.builtins.resolve(parent, call, intent)
    return resolve_export(worker, parent, intent["module"], intent["export"])


def resolve_export(
    worker: Worker, parent: AttemptRecord, module: str, export: str, *, wait: bool = True
) -> Target:
    """Resolve preparation and invocation through the same captured callable binding."""
    if worker.executions is None or parent.job is None:
        raise WorkspaceRefusal("child call needs an accepted captured job parent")
    owner = worker.fence.record_owner_id
    root = worker.executions.capture_root(owner, parent.request_id)
    capture = worker.executions.capture(owner, root)
    intent = {"module": module, "export": export}
    caller = parent.job.installation_id
    callers = {caller, _deferred_key(worker, caller)}
    selected = next(
        (
            binding
            for binding in capture["bindings"]
            if (
                binding.get("caller_installation_id") or binding.get("caller_deferred_key"),
                binding["module"],
                binding["export"],
            )
            in {(name, intent["module"], intent["export"]) for name in callers}
        ),
        None,
    )
    if selected is None:
        raise WorkspaceRefusal("child callable is absent from the accepted installation bindings")
    callee, entrypoint = (
        selected.get("callee_installation_id") or selected.get("callee_deferred_key", ""),
        selected["entrypoint"],
    )
    if selected.get("callee_deferred_key"):
        row = machine_deferred.rows(capture).get(callee)
        if row is None or worker.machine_calls is None:
            raise WorkspaceRefusal("child callable names an undeclared deferred installation")
        prepared = worker.machine_calls.deferred.prepared(row, wait=wait)
    else:
        prepared = worker.executions.preparation(owner, root)["installations"].get(callee)
    if prepared is None:
        raise WorkspaceRefusal("child installation is not retained")
    identifier = prepared["installation_id"]
    import base64

    raw = base64.b64decode(prepared["placement"]["package_interface"], validate=True)
    interface = package_interface.read_bytes(raw, "installed child")
    declared = next(
        (
            entry
            for entry in [*interface["jobs"], *interface["entrypoints"]]
            if entry["name"] == entrypoint
        ),
        None,
    )
    if declared is None:
        raise WorkspaceRefusal("installed child has no callable declaration")
    if declared.get("internal", False) and caller != identifier:
        raise WorkspaceRefusal("internal_callable: child requires the same installation")
    invocable = declared.get("invocable", {})
    if (invocable.get("module"), invocable.get("export")) != (intent["module"], intent["export"]):
        raise WorkspaceRefusal("child does not name the installed export")
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
                    worker.config.cozy_home / "job-plans",
                    identifier,
                    descriptor,
                ).read_bytes()
            )
        )
    operation_identity = str(invocable.get("operation_identity", ""))
    if invocable.get("memoize") and not operation_identity:
        worker.note(
            "memoization",
            f"{intent['module']}.{intent['export']} runs uncached: "
            "callable implementation identity is unavailable",
        )
    return Target(
        identifier,
        entrypoint,
        declared,
        prepared,
        binding,
        operation_identity,
        binding is None and caller != identifier,
        callee,
    )


def _deferred_key(worker: Worker, installation: str) -> str:
    """A running deferred callee's key, from its installed package and release."""
    for placement in worker.prepared_installations.values():
        if placement.installation_id == installation:
            return f"{placement.document.package.package}@{placement.document.package.release}"
    return ""


def computation(
    target: Target,
    arguments: dict[str, Any],
    numerical: bytes,
    capture: dict[str, Any] | None = None,
) -> bytes:
    """Bind the callee's frozen implementation and inputs, excluding caller and custody."""
    if not target.operation_identity:
        return b""
    inputs = dict(arguments)
    for model in target.declaration.get("models", []):
        parameter = model["path"].removeprefix(target.entrypoint + ".models.")
        value = inputs.get(parameter)
        if not isinstance(value, dict) or not isinstance(value.get("manifest"), dict):
            raise WorkspaceRefusal("memoized Model argument requires an exact retained manifest")
        inputs[parameter] = {"manifest": value["manifest"]}
    identity: dict[str, Any] = {
        "operation_identity": target.operation_identity,
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
