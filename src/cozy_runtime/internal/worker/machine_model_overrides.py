"""Resolve captured model choices through the machine's existing catalog and source paths."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime.internal import fill, package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    machine_adapter_views,
    machine_model_choices,
    machine_model_defaults,
    machine_model_resolve,
    machine_source_models,
)
from .machine_child_target import Target
from .workspace import WorkspaceRefusal
from .workspace_calls import Call

if TYPE_CHECKING:
    from .session import Worker


def captured(worker: Worker, owner: str, call: Call, target: Target) -> dict[str, pb.ModelChoice]:
    assert worker.executions is not None
    root = worker.executions.capture_root(owner, call.parent_request)
    capture = documents.from_body(
        worker.executions.capture(owner, root), pb.MachineExecutionCapture
    )
    if not capture.model_choices:
        return {}
    prepared = worker.executions.prepared(owner, root)
    interfaces = {
        row.installation_id: package_interface.parse(
            row.placement.package_interface, "prepared model choices"
        )
        for row in prepared.installations.values()
    }
    interfaces.update(
        {
            row.key: package_interface.parse(
                row.preparation.package_interface, "deferred model choices"
            )
            for row in capture.deferred_installations
        }
    )
    callee = target.callee or target.installation_id
    placement = target.prepared_installation["placement"]
    interfaces[callee] = package_interface.parse(
        base64.b64decode(placement["package_interface"], validate=True), "selected model choices"
    )
    choices = machine_model_choices.resolve(capture, prepared.entrypoint, interfaces)
    return {
        parameter: choice
        for (identity, entrypoint, parameter), choice in choices.items()
        if (identity, entrypoint) == (callee, target.entrypoint)
    }


def has_base(choice: pb.ModelChoice) -> bool:
    return bool(choice.repository or choice.source or choice.HasField("manifest"))


def _pin(worker: Worker, choice: pb.ModelChoice) -> dict[str, Json] | None:
    if not choice.HasField("manifest"):
        return None
    digest = documents.spell(choice.manifest.digest)
    length = choice.manifest.length
    if not length and worker.workspace is not None:
        try:
            length = len(fill.store(worker.workspace.store_root).manifest(digest)["manifest"])
        except fill.tensorfs_module().errors.Refusal:
            pass
    return {"digest": digest, "length": length}


def checkpoint(
    worker: Worker,
    choice: pb.ModelChoice,
    declaration: Mapping[str, Json],
    *,
    package: str,
    hub: str,
    owner: str,
    credentials: Mapping[str, str],
    note: machine_source_models.Note,
) -> dict[str, Json]:
    """Only source/canonical selection; this does not grant a GPU or construct a model."""
    if choice.source:
        return machine_source_models.slot(worker, choice, credentials, note)
    pin = _pin(worker, choice)
    if pin is not None and pin["length"]:
        authority = machine_model_resolve.registration(worker, hub)
        return {
            "parameter": choice.parameter,
            "repository": choice.repository,
            "manifest": pin,
            "release": choice.release,
            "lane": choice.lane,
            "public_origin": authority.origin if authority is not None else hub,
        }
    catalog = machine_model_resolve.own(worker, package, hub)
    selected: dict[str, Json] = {
        "repository": choice.repository,
        "release": choice.release,
        "lane": choice.lane,
    }
    if pin is not None:
        selected["manifest"] = pin
    return dict(
        machine_model_resolve.slot(
            catalog,
            package,
            declaration,
            choice.parameter,
            selected,
            machine_model_defaults.catalog_fits(machine_model_defaults.fitter(worker)),
            owner,
            note,
        )
    )


def apply(
    worker: Worker,
    base: dict[str, Json],
    choice: pb.ModelChoice,
    *,
    package: str,
    hub: str,
    owner: str,
    credentials: Mapping[str, str],
    note: machine_source_models.Note,
    check: Callable[[], None],
    cancellation: machine_model_defaults.Cancellation | None = None,
    native_base: bool = False,
) -> dict[str, Json]:
    if not choice.adapters:
        return base
    adapters = []
    for index, requested in enumerate(choice.adapters):
        selected = pb.ModelChoice(
            parameter=f"adapter_{index}",
            repository=requested.model,
            release=requested.release,
            lane=requested.lane,
            source=requested.source,
            profiles=requested.profiles,
        )
        if requested.manifest:
            selected.manifest.digest = documents.raw(requested.manifest)
        # An adapter is an independent data checkpoint. It inherits no model's GPU ladder.
        row = checkpoint(
            worker,
            selected,
            {"path": "adapter.models." + selected.parameter},
            package=package,
            hub=hub,
            owner=owner,
            credentials=credentials,
            note=note,
        )
        chosen = msgspec.convert(row, machine_model_defaults.Selected)
        scale = requested.scale or "1"
        ref = chosen.repository + (f"@{chosen.release}" if chosen.release else "")
        ref += f"/{chosen.lane}" if chosen.lane else ""
        note(
            f"{selected.parameter}: {ref} {chosen.manifest.digest} "
            f"scale {scale} on {requested.component}",
            0,
            0,
        )
        adapters.append(
            machine_adapter_views.Adapter(
                chosen, requested.component, requested.source_component or "adapter", scale
            )
        )
    note("preparing model adapters", 0, 0)
    return machine_adapter_views.compose(
        worker, base, adapters, check=check, cancellation=cancellation, native_base=native_base
    )


def context(worker: Worker, owner: str, call: Call) -> tuple[str, str, dict[str, str]]:
    """Accepted routing and ephemeral provider credentials, never part of a model record."""
    assert worker.executions is not None
    root = worker.executions.capture_root(owner, call.parent_request)
    prepared = worker.executions.prepared(owner, root)
    from .machine_sources import PROVIDERS

    with worker.machine_sources.lock:
        held = worker.machine_sources.credentials.get((owner, root), {})
        credentials = {PROVIDERS[key]: value for key, value in held.items() if key in PROVIDERS}
    return prepared.hub, prepared.account, credentials


def private(
    worker: Worker,
    request: pb.PreparePrivatePlacementRequest,
    prepared: pb.Placement,
    *,
    check: Callable[[], None],
) -> tuple[list[dict[str, Json]], list[pb.NativeModelBinding]]:
    """Resolve exact slots before a private capture exists, under the caller's current claim."""
    interface = package_interface.parse(prepared.package_interface, "private model overrides")
    package = (
        prepared.package.package if prepared.HasField("package") else prepared.development.package
    )
    allowed = {
        slot.path: slot for entrypoint in interface.entrypoints for slot in entrypoint.models
    }
    catalog = (
        documents.parse(request.download_delegation, pb.DownloadDelegation)
        if request.download_delegation
        else None
    )
    selections: dict[str, dict[str, Json]] = {
        row.slot: documents.body(row) for row in (() if catalog is None else catalog.models)
    }
    native = {row.slot: row for row in request.native_models}
    chosen: set[str] = set()
    credentials = machine_source_models.credentials(request.source_credentials)
    for original in request.model_choices:
        path = original.parameter
        if path not in allowed or path in chosen:
            raise WorkspaceRefusal(
                "model_override_invalid: private choices require one exact serving callable model slot"
            )
        chosen.add(path)
        choice = machine_model_choices.normalized(original)
        choice.parameter = allowed[path].parameter
        original_native = native.get(path)
        if has_base(choice):
            if original_native is not None:
                raise WorkspaceRefusal(
                    "model_override_conflict: an owned ModelArtifact already selects this base"
                )
            base = checkpoint(
                worker,
                choice,
                msgspec.to_builtins(allowed[path]),
                package=package,
                hub=request.hub,
                owner=request.owner,
                credentials=credentials,
                note=lambda *_: None,
            )
        elif original_native is not None:
            base = {
                "parameter": choice.parameter,
                "repository": original_native.model,
                "manifest": documents.body(original_native.manifest),
                "native": True,
            }
        elif path in selections:
            selected = selections[path]
            pin = pb.ModelChoice(
                parameter=choice.parameter,
                repository=str(selected["model"]),
                release=str(selected.get("release", "")),
                lane=str(selected.get("lane", "")),
            )
            pin.manifest.digest = documents.raw(str(selected["manifest"]))
            base = checkpoint(
                worker,
                pin,
                msgspec.to_builtins(allowed[path]),
                package=package,
                hub=request.hub,
                owner=request.owner,
                credentials=credentials,
                note=lambda *_: None,
            )
        else:
            raise WorkspaceRefusal(
                "model_override_invalid: adapter-only private choice has no selected base"
            )
        result = apply(
            worker,
            base,
            choice,
            package=package,
            hub=request.hub,
            owner=request.owner,
            credentials=credentials,
            note=lambda *_: None,
            check=check,
            native_base=original_native is not None,
        )
        selected_row = machine_model_defaults.decode([result])[0]
        selections[path] = {
            "package": package,
            "slot": path,
            "model": selected_row.repository,
            "manifest": selected_row.manifest.digest,
            "release": selected_row.release,
            "lane": selected_row.lane,
            "adapters": msgspec.to_builtins(selected_row.adapters),
            "composed": msgspec.to_builtins(selected_row.composed),
        }
        native.pop(path, None)
    return list(selections.values()), list(native.values())
