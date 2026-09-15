"""Verify captured callable inventory against Runtime's existing immutable preparations."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime.internal import (
    canonical,
    installed_interfaces,
    package_environment,
    package_interface,
)
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import PUBLISHED_MACHINE_CAPTURE_WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_model_defaults

if TYPE_CHECKING:
    from .session import Worker


def code_identity(placement: dict[str, Any]) -> str:
    """The established code identity for each origin, independent of Model bindings."""
    if placement.get("development"):
        return str(placement["development"].get("local_revision_digest", ""))
    return str(placement.get("environment_digest", ""))


def verify(worker: Worker, request: pb.MachineExecutionSubmit) -> bytes:
    raw = request.capture_canonical_bytes
    if (
        not 0 < len(raw) <= canonical.DOC_MAX_BYTES
        or documents.digest_of(raw) != request.capture_digest
    ):
        raise WorkspaceRefusal("execution capture does not bind its bounded canonical bytes")
    capture = documents.read(raw, pb.MachineExecutionCapture)
    published = capture.get("published_revisions", [])
    if published and request.claim.wire_minor < PUBLISHED_MACHINE_CAPTURE_WIRE_MINOR:
        raise WorkspaceRefusal("published execution capture requires wire54")
    if (
        set(capture) - {"published_revisions", "model_defaults"}
        != {"format", "root_revision_digest", "revisions", "bindings"}
        or not isinstance(capture["revisions"], list)
        or not isinstance(published, list)
        or not 1 <= len(capture["revisions"]) + len(published) <= 128
        or not isinstance(capture["bindings"], list)
        or len(capture["bindings"]) > 4096
    ):
        raise WorkspaceRefusal("execution capture has invalid inventory fields or bounds")
    revisions: dict[str, dict[str, Any]] = {}
    prepared: dict[str, dict[str, Any]] = {}
    interfaces: dict[str, dict[str, Any]] = {}
    exports: dict[str, set[tuple[str, str, str]]] = {}
    inventory = [
        (
            documents.spell(
                documents.embedded_document_digest(revision, "cozy.worker.v1.LocalPackageRevision")
            ),
            revision,
            False,
        )
        for revision in capture["revisions"]
    ]
    public_inventory = [
        (
            documents.spell(
                documents.embedded_document_digest(
                    revision["environment"], "cozy.worker.v1.Environment"
                )
            ),
            revision,
            True,
        )
        for revision in published
    ]
    for rows in (inventory, public_inventory):
        identities = [row[0] for row in rows]
        if identities != sorted(set(identities)):
            raise WorkspaceRefusal("execution capture needs sorted unique revisions")
    for digest, revision, is_published in [*inventory, *public_inventory]:
        if digest in revisions:
            raise WorkspaceRefusal("execution capture repeats a revision")
        revisions[digest] = revision
        candidates = [
            value
            for (_, selected), value in worker.prepared_revisions.items()
            if selected == digest
            and (
                digest != capture["root_revision_digest"]
                or request.prepared_state.WhichOneof("mode") == "job"
                or value.placement_id == request.offer.placement_id
            )
        ]
        if not candidates:
            raise WorkspaceRefusal(
                "execution capture revision has not been prepared on this machine"
            )
        selected = candidates[0]
        if selected.installed is None or not selected.installed.receipt_digest:
            raise WorkspaceRefusal("execution capture has no retained immutable environment")
        installed = package_environment.open_generation(
            Path(worker.options.install_root or ""),
            receipt_digest=selected.installed.receipt_digest,
            environment_digest=documents.spell(selected.environment_digest),
            python=Path(worker.options.python or selected.installed.python),
        )
        if is_published:
            if selected.development or any(
                revision[field] != selected.document.get(field)
                for field in ("package", "environment", "package_interface")
            ):
                raise WorkspaceRefusal("execution capture differs from its prepared publication")
        elif (
            revision["package"],
            revision["release"],
            revision["source_digest"],
            revision["package_interface"],
        ) != (
            selected.development.get("package"),
            selected.development.get("release"),
            selected.development.get("source_digest"),
            selected.document.get("package_interface"),
        ):
            raise WorkspaceRefusal("execution capture differs from its prepared revision")
        interface = revision["package_interface"]
        path = Path(worker.options.artifact_cache or "") / documents.raw(interface["digest"]).hex()
        with path.open("rb") as stream:
            interface_raw = stream.read(canonical.DOC_MAX_BYTES + 1)
        if (
            len(interface_raw) != interface["length"]
            or documents.spell(documents.digest_of(interface_raw)) != interface["digest"]
        ):
            raise WorkspaceRefusal("execution capture interface bytes are absent or changed")
        interfaces[digest] = package_interface.read_bytes(interface_raw, "execution capture")
        exports[digest] = {
            (row["interface_digest"], row["module"], row["export"])
            for row in installed_interfaces.read(installed)
        }
        prepared[digest] = {
            "placement": selected.document,
            "receipt_digest": installed.receipt_digest,
        }
    if not 1 <= len(revisions) <= 128 or capture.get("root_revision_digest") not in revisions:
        raise WorkspaceRefusal("execution capture needs sorted unique revisions and its exact root")
    keys = []
    bound: dict[str, set[tuple[str, str, str]]] = {digest: set() for digest in revisions}
    for binding in capture["bindings"]:
        if (
            not isinstance(binding, dict)
            or set(binding)
            != {
                "caller_revision_digest",
                "interface_digest",
                "module",
                "export",
                "callee_revision_digest",
                "entrypoint",
            }
            or any(
                not isinstance(value, str) or not value or len(value) > 256
                for value in binding.values()
            )
        ):
            raise WorkspaceRefusal("execution callable binding has invalid fields")
        caller, callee = binding["caller_revision_digest"], binding["callee_revision_digest"]
        target = (binding["interface_digest"], binding["module"], binding["export"])
        keys.append((caller, *target))
        if caller not in revisions or callee not in revisions:
            raise WorkspaceRefusal("execution callable names an uncaptured revision")
        is_self = (
            caller == callee
            and binding["interface_digest"] == revisions[caller]["package_interface"]["digest"]
        )
        if target not in exports[caller] and not is_self:
            raise WorkspaceRefusal("execution callable is outside the prepared caller inventory")
        if binding["interface_digest"] != revisions[callee]["package_interface"]["digest"]:
            raise WorkspaceRefusal("execution callable names a different callee interface")
        entry = next(
            (
                entry
                for entry in [*interfaces[callee]["jobs"], *interfaces[callee]["entrypoints"]]
                if entry["name"] == binding["entrypoint"]
            ),
            None,
        )
        if entry and entry.get("internal", False) and caller != callee:
            raise WorkspaceRefusal(
                "internal_callable: child requires the same installed package revision"
            )
        invocable = entry.get("invocable", {}) if entry else {}
        if (invocable.get("module"), invocable.get("export")) != target[1:]:
            raise WorkspaceRefusal("execution callable does not identify the prepared callee")
        if target in exports[caller]:
            bound[caller].add(target)
    if len(keys) > 4096 or keys != sorted(set(keys)) or bound != exports:
        raise WorkspaceRefusal("execution callable inventory is incomplete or ambiguous")
    machine_model_defaults.verify(worker, request, capture, interfaces)
    spec = documents.read(request.offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    root = capture["root_revision_digest"]
    state = request.prepared_state
    if "job" in spec:
        if (
            state.WhichOneof("mode") != "job"
            or state.job.build_id != root
            or spec["job"].get("build_id") != root
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
        if (
            selected_entry is None
            or code_identity(selected_entry) != root
            or selected_entry != prepared[root]["placement"]
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
    preparation: dict[str, Any] = {"revisions": prepared, "state": state_identity(state)}
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
    """Reopen Runtime-owned receipts; no source install IDs or laptop paths are admitted."""
    for digest, row in preparation["revisions"].items():
        placement = worker._placement_from_entry(row["placement"], b"")
        if code_identity(placement.document) != digest:
            raise WorkspaceRefusal("execution preparation revision changed")
        placement.installed = package_environment.open_generation(
            Path(worker.options.install_root or ""),
            receipt_digest=row["receipt_digest"],
            environment_digest=documents.spell(placement.environment_digest),
            python=Path(worker.options.python or sys.executable),
        )
        worker.prepared_revisions[(placement.placement_id, digest)] = placement
    return pb.DesiredWorkerState.FromString(base64.b64decode(preparation["state"], validate=True))
