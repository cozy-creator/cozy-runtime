"""Verify captured callable inventory against Runtime's existing immutable preparations."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime.internal import (
    canonical,
    package_installation,
    package_interface,
)
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_model_defaults, machine_model_resolve

if TYPE_CHECKING:
    from .session import Worker


class Installation(msgspec.Struct, frozen=True):
    """An installation an execution runs: the placement document it was prepared as."""

    placement: dict[str, Json]
    installation_id: str

    @property
    def interface(self) -> bytes:
        """The installed package interface's exact bytes."""
        return base64.b64decode(self.placement["package_interface"], validate=True)


class Builtin(msgspec.Struct, frozen=True):
    """Runtime's own operations installation, retained with every execution that may call it."""

    placement: dict[str, Json]
    installation_id: str
    runtime_version: str = ""


class Preparation(msgspec.Struct, frozen=True, omit_defaults=True):
    """What an accepted execution journals of this machine's preparation, and reopens."""

    installations: dict[str, Installation] = {}
    state: str = ""
    wire_minor: int | msgspec.UnsetType = msgspec.UNSET
    runtime_builtin: Builtin | None = None
    hub: str = ""  # the Hub its run came from, as `hub_key` spells it; "": the default
    account: str = ""  # the account owning the run: whose org-relative Models local/ code names
    entrypoint: str = ""

    def encode(self) -> bytes:
        return canonical.write(msgspec.to_builtins(self))


def preparation(document: Mapping[str, object]) -> Preparation:
    """A journaled preparation, decoded once where it is read back."""
    try:
        return msgspec.convert(document, Preparation)
    except msgspec.ValidationError as exc:
        raise WorkspaceRefusal(f"retained execution preparation is malformed: {exc}") from exc


def operation_identity(declared: package_interface.CallableDoc) -> str:
    """A callable's installed operation identity, or "" when it has none."""
    invocable = declared.invocable
    if invocable is msgspec.UNSET or invocable.operation_identity is msgspec.UNSET:
        return ""
    return invocable.operation_identity


def installation_identity(placement: Mapping[str, object]) -> str:
    """Opaque installed-generation lookup, never a code fingerprint."""
    return str(placement.get("installation_id", ""))


def verify(worker: Worker, request: pb.MachineExecutionSubmit) -> bytes:
    raw = request.capture_canonical_bytes
    if (
        not 0 < len(raw) <= canonical.DOC_MAX_BYTES
        or documents.digest_of(raw) != request.capture_digest
    ):
        raise WorkspaceRefusal("execution capture does not bind its bounded canonical bytes")
    capture = documents.parse(raw, pb.MachineExecutionCapture)
    if not 1 <= len(capture.installed_packages) <= 128:
        raise WorkspaceRefusal("execution capture needs bounded installed packages")
    root = capture.root_installation_id
    installations: dict[str, Installation] = {}
    interfaces: dict[str, package_interface.PackageInterface] = {}
    interface_bytes: dict[str, bytes] = {}
    for package in capture.installed_packages:
        identifier = package.installation_id
        if not identifier or identifier in installations:
            raise WorkspaceRefusal("execution capture repeats or omits an installation ID")
        selected = next(
            (
                value
                for (_, held, _), value in worker.prepared_installations.items()
                if held == identifier
            ),
            None,
        )
        if selected is None:
            raise WorkspaceRefusal("execution package has not been installed on this machine")
        selected.installed = selected.installed or package_installation.open_installation(
            Path(worker.options.install_root or ""),
            identifier,
        )
        # Worker-owned installed metadata is authoritative. The received interface is
        # a caller convenience, never a second implementation reading to compare.
        interface_bytes[identifier] = selected.package_interface
        interfaces[identifier] = package_interface.parse(
            selected.package_interface, "installed package"
        )
        installations[identifier] = Installation(documents.body(selected.document), identifier)
    if root not in installations:
        raise WorkspaceRefusal("execution capture root is not installed")
    if len(capture.deferred_installations) > 128:
        raise WorkspaceRefusal("execution capture has unbounded deferred installations")
    for row in capture.deferred_installations:
        if not row.key or row.key in interfaces or row.key != f"{row.package}@{row.release}":
            raise WorkspaceRefusal("execution capture repeats or misnames a deferred installation")
        if not row.preparation.package_interface:
            raise WorkspaceRefusal("deferred installation carries no release interface")
        interfaces[row.key] = package_interface.parse(
            row.preparation.package_interface, "deferred release"
        )
    targets: dict[tuple[str, str, str], tuple[str, str]] = {}
    for binding in capture.bindings:
        if not (binding.module and binding.export and binding.entrypoint):
            raise WorkspaceRefusal("execution callable binding has invalid fields")
        caller = binding.caller_installation_id or binding.caller_deferred_key
        callee = machine_model_defaults.callee(binding)
        if caller not in interfaces or callee not in interfaces:
            raise WorkspaceRefusal("execution callable names an uninstalled package")
        target = binding.module, binding.export
        # A repeated identical binding is one binding; one import bound two ways is ambiguous.
        if targets.setdefault((caller, *target), (callee, binding.entrypoint)) != (
            callee,
            binding.entrypoint,
        ):
            raise WorkspaceRefusal("execution callable inventory is ambiguous")
        interface = interfaces[callee]
        entry = next(
            (e for e in (*interface.jobs, *interface.entrypoints) if e.name == binding.entrypoint),
            None,
        )
        if entry is None:
            raise WorkspaceRefusal("execution callable is absent from the installed package")
        if entry.internal and caller != callee:
            raise WorkspaceRefusal("internal_callable: child requires the same installation")
        invocable = entry.invocable
        if invocable is msgspec.UNSET or (invocable.module, invocable.export) != target:
            raise WorkspaceRefusal("execution callable does not identify the installed export")
    if len(targets) > 4096:
        raise WorkspaceRefusal("execution callable inventory exceeds its bound")
    machine_model_defaults.verify(worker, capture)
    spec = documents.parse(request.offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    state = request.prepared_state
    root_interface = interfaces[root]
    if spec.HasField("job"):
        if (
            state.WhichOneof("mode") != "job"
            or state.job.installation_id != root
            or spec.job.installation_id != root
            or state.job.job_descriptor_id != spec.job.job_descriptor_id
        ):
            raise WorkspaceRefusal("execution root does not bind its prepared job directive")
        body = canonical.parse(interface_bytes[root])
        root_callable = next(
            (
                entry
                for entry in root_interface.jobs
                if package_interface.job_descriptor_id(body, entry.name)
                == spec.job.job_descriptor_id
            ),
            None,
        )
    else:
        if state.WhichOneof("mode") != "placement_set":
            raise WorkspaceRefusal("execution inference has no exact prepared placement")
        placements = documents.parse(
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
                for entry in placements.placements
                if entry.placement_id == request.offer.placement_id
            ),
            None,
        )
        # The placement this machine prepared, by the key it holds it under: the bindings
        # digest covers the entrypoints and models, and the installation its interface. A
        # release prepared for several model selections holds each, so concurrent roots of
        # one release never displace another.
        held = (
            worker.prepared_installations.get(
                (request.offer.placement_id, root, selected_entry.bindings_digest)
            )
            if selected_entry is not None and selected_entry.bindings_digest
            else None
        )
        if held is None or selected_entry is None or selected_entry.installation_id != root:
            raise WorkspaceRefusal("execution inference does not name the captured root")
        installations[root] = Installation(documents.body(held.document), root)
        # The held placement's bindings digest is the selected entry's: the same entrypoints.
        bound = next(
            (
                entry
                for entry in selected_entry.entrypoints
                if documents.spell(entry.entrypoint_binding_digest)
                == spec.serving.entrypoint_binding_digest
            ),
            None,
        )
        root_callable = next(
            (
                entry
                for entry in root_interface.entrypoints
                if bound is not None and entry.name == bound.name
            ),
            None,
        )
    if root_callable is not None and root_callable.internal:
        raise WorkspaceRefusal(
            "internal_callable: an internal callable cannot be an execution root"
        )
    from . import machine_model_choices

    if capture.model_choices:
        if root_callable is None:
            raise WorkspaceRefusal("model overrides require an exact root callable")
        machine_model_choices.resolve(capture, root_callable.name, interfaces)
    assert worker.machine_calls is not None
    return Preparation(
        installations=installations,
        entrypoint=root_callable.name if root_callable is not None else "",
        state=state_identity(state),
        # The minor this root and its descendants run under, whatever later streams say.
        wire_minor=min(WIRE_MINOR, request.claim.wire_minor),
        runtime_builtin=worker.machine_calls.builtins.capture(),
        account=request.account,
        hub=machine_model_resolve.hub_key(worker, request.hub),
    ).encode()


def state_identity(state: pb.DesiredWorkerState) -> str:
    """Strip observer/revision stamps, retaining every admitted placement/resource field."""
    copied = pb.DesiredWorkerState()
    copied.CopyFrom(state)
    for name in ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "revision"):
        copied.ClearField(name)
    return base64.b64encode(copied.SerializeToString(deterministic=True)).decode()


def restore(worker: Worker, document: Mapping[str, object]) -> pb.DesiredWorkerState:
    """Reopen worker-owned installation lifetimes after restart."""
    retained = preparation(document)
    for identifier, row in retained.installations.items():
        placement = worker._placement_from_entry(
            documents.from_body(row.placement, pb.Placement), b""
        )
        placement.installed = package_installation.open_installation(
            Path(worker.options.install_root or ""),
            identifier,
        )
        worker.prepared_installations[placement.prepared_key] = placement
    return pb.DesiredWorkerState.FromString(base64.b64decode(retained.state, validate=True))
