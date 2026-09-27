"""Verify captured callable inventory against Runtime's existing immutable preparations."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime.internal import (
    canonical,
    package_installation,
    package_interface,
)
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_model_defaults

if TYPE_CHECKING:
    from .session import Worker


def installation_identity(placement: dict[str, Any]) -> str:
    """Opaque installed-generation lookup, never a code fingerprint."""
    return str(placement.get("installation_id", ""))


def verify(worker: Worker, request: pb.MachineExecutionSubmit) -> bytes:
    raw = request.capture_canonical_bytes
    if (
        not 0 < len(raw) <= canonical.DOC_MAX_BYTES
        or documents.digest_of(raw) != request.capture_digest
    ):
        raise WorkspaceRefusal("execution capture does not bind its bounded canonical bytes")
    capture = documents.read(raw, pb.MachineExecutionCapture)
    inventory = capture.get("installed_packages", [])
    if not isinstance(inventory, list) or not 1 <= len(inventory) <= 128:
        raise WorkspaceRefusal("execution capture needs bounded installed packages")
    root = capture.get("root_installation_id", "")
    prepared: dict[str, dict[str, Any]] = {}
    interfaces: dict[str, dict[str, Any]] = {}
    for package in inventory:
        identifier = package.get("installation_id", "")
        if not identifier or identifier in prepared:
            raise WorkspaceRefusal("execution capture repeats or omits an installation ID")
        candidates = [
            value
            for (_, selected), value in worker.prepared_installations.items()
            if selected == identifier
        ]
        if not candidates:
            raise WorkspaceRefusal("execution package has not been installed on this machine")
        selected = candidates[0]
        selected.installed = selected.installed or package_installation.open_installation(
            Path(worker.options.install_root or ""),
            identifier,
        )
        # Worker-owned installed metadata is authoritative. The received interface is
        # a caller convenience, never a second implementation reading to compare.
        interfaces[identifier] = package_interface.read_bytes(
            selected.package_interface,
            "installed package",
        )
        prepared[identifier] = {"placement": selected.document, "installation_id": identifier}
    if root not in prepared:
        raise WorkspaceRefusal("execution capture root is not installed")
    targets: dict[tuple[str, str, str], tuple[str, str]] = {}
    for binding in capture.get("bindings", []):
        fields = {"caller_installation_id", "callee_installation_id", "module", "export"}
        if not isinstance(binding, dict) or not {*fields, "entrypoint"} <= set(binding):
            raise WorkspaceRefusal("execution callable binding has invalid fields")
        caller, callee = binding["caller_installation_id"], binding["callee_installation_id"]
        if caller not in prepared or callee not in prepared:
            raise WorkspaceRefusal("execution callable names an uninstalled package")
        target = binding["module"], binding["export"]
        # A repeated identical binding is one binding; one import bound two ways is ambiguous.
        if targets.setdefault((caller, *target), (callee, binding["entrypoint"])) != (
            callee,
            binding["entrypoint"],
        ):
            raise WorkspaceRefusal("execution callable inventory is ambiguous")
        entry = next(
            (
                entry
                for entry in [*interfaces[callee]["jobs"], *interfaces[callee]["entrypoints"]]
                if entry["name"] == binding["entrypoint"]
            ),
            None,
        )
        if entry is None:
            raise WorkspaceRefusal("execution callable is absent from the installed package")
        if entry.get("internal", False) and caller != callee:
            raise WorkspaceRefusal("internal_callable: child requires the same installation")
        invocable = entry.get("invocable", {})
        if (invocable.get("module"), invocable.get("export")) != target:
            raise WorkspaceRefusal("execution callable does not identify the installed export")
    if len(targets) > 4096:
        raise WorkspaceRefusal("execution callable inventory exceeds its bound")
    machine_model_defaults.verify(worker, request, capture, interfaces)
    spec = documents.read(request.offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    root = capture["root_installation_id"]
    state = request.prepared_state
    if "job" in spec:
        if (
            state.WhichOneof("mode") != "job"
            or state.job.installation_id != root
            or spec["job"].get("installation_id") != root
            or state.job.job_descriptor_id != spec["job"].get("job_descriptor_id")
        ):
            raise WorkspaceRefusal("execution root does not bind its prepared job directive")
    else:
        if state.WhichOneof("mode") != "placement_set":
            raise WorkspaceRefusal("execution inference has no exact prepared placement")
        placements = documents.read(
            state.placement_set.placement_set_canonical_bytes, pb.PlacementSet
        )
        if (
            documents.digest_of(state.placement_set.placement_set_canonical_bytes)
            != state.placement_set.placement_set_digest
        ):
            raise WorkspaceRefusal("execution placement digest changed")
        selected_entry = next(
            (
                entry
                for entry in placements["placements"]
                if entry["placement_id"] == request.offer.placement_id
            ),
            None,
        )
        held = prepared[root]["placement"]
        # The placement this machine prepared, named by the facts that bind execution; an
        # additive or reordered field from the owner's copy is not a different placement.
        if (
            selected_entry is None
            or installation_identity(selected_entry) != root
            or any(
                selected_entry.get(key) != held.get(key)
                for key in ("installation_id", "bindings_digest", "package_interface", "models")
            )
        ):
            raise WorkspaceRefusal("execution inference does not name the captured root")
    root_interface = interfaces[root]
    if "job" in spec:
        root_callable = next(
            (
                entry
                for entry in root_interface["jobs"]
                if package_interface.job_descriptor_id(root_interface, entry["name"])
                == spec["job"]["job_descriptor_id"]
            ),
            None,
        )
    else:
        binding = next(
            (
                entry
                for entry in prepared[root]["placement"]["entrypoints"]
                if entry["entrypoint_binding_digest"]
                == spec.get("serving", {}).get("entrypoint_binding_digest")
            ),
            None,
        )
        root_callable = next(
            (
                entry
                for entry in root_interface["entrypoints"]
                if binding is not None and entry["name"] == binding["name"]
            ),
            None,
        )
    if root_callable and root_callable.get("internal", False):
        raise WorkspaceRefusal(
            "internal_callable: an internal callable cannot be an execution root"
        )
    preparation: dict[str, Any] = {"installations": prepared, "state": state_identity(state)}
    assert worker.machine_calls is not None
    if sdk := worker.machine_calls.builtins.capture():
        preparation["runtime_builtin"] = sdk
    return canonical.write(preparation)


def state_identity(state: pb.DesiredWorkerState) -> str:
    """Strip observer/revision stamps, retaining every admitted placement/resource field."""
    copied = pb.DesiredWorkerState()
    copied.CopyFrom(state)
    for name in ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "revision"):
        copied.ClearField(name)
    return base64.b64encode(copied.SerializeToString(deterministic=True)).decode()


def restore(worker: Worker, preparation: dict[str, Any]) -> pb.DesiredWorkerState:
    """Reopen worker-owned installation lifetimes after restart."""
    for identifier, row in preparation["installations"].items():
        placement = worker._placement_from_entry(row["placement"], b"")
        placement.installed = package_installation.open_installation(
            Path(worker.options.install_root or ""),
            identifier,
        )
        worker.prepared_installations[(placement.placement_id, identifier)] = placement
    return pb.DesiredWorkerState.FromString(base64.b64decode(preparation["state"], validate=True))
