"""One privileged native source operation, launched on an inherited stdin channel.

No credentials or source URLs enter argv, environment, logs, or persistent state.
The parent owns acceptance and memoization; this child owns only bounded native work.
"""

from __future__ import annotations

import hashlib
from typing import TypedDict

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef, SourceArtifact
from cozy_runtime.author.sources import SOURCE_FILES_MAX_BYTES
from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import commit_files
from cozy_runtime.internal.worker.source_steps import (
    Answer,
    Convert,
    Download,
    Launch,
    Produced,
    Resolved,
    ResolveSource,
    Selection,
    SourceRefusal,
    SourceStep,
    View,
    report,
    serve,
)
from cozy_runtime.internal.worker.source_steps import Commit as CommitStep
from cozy_runtime.internal.worker.workspace_byte_outputs import manifest_members


class _Manifest(TypedDict):
    sha256: str
    length: int


class Receipt(TypedDict):
    """The fields read from TensorFS's derive receipt; the whole document is the receipt."""

    manifest: _Manifest


def materialize(store: fill.Store, step: Download) -> Produced:
    produced = store.materialize_source(
        step.native_owner,
        step.pin.members,
        allowed_hosts=step.pin.allowed_hosts,
        credential_hosts=step.pin.credential_hosts,
        credential=step.access["credential"],
        allow_local=step.access["allow_local"],
        progress=lambda present, whole: report("download", present, whole),
    )
    root = produced["artifact"]
    manifest = ObjectRef(root["manifest_digest"], root["manifest_length"])
    artifact = SourceArtifact(step.service_id, "source", manifest, root["receipt_digest"])
    return Produced(result=msgspec.to_builtins(artifact), native_receipt=root["receipt"])


def view(store: fill.Store, step: View) -> Produced:
    source, manifest = store.tree_root(step.source_owner), step.manifest
    if (
        not source
        or not source["complete"]
        or source["released"]
        or source["manifest_digest"] != manifest.digest
        or source["manifest_length"] != manifest.length
        or source["receipt_digest"] != step.receipt_digest
    ):
        raise ValueError("source view lost its exact native source")
    body = store.manifest(manifest.digest)["manifest"]
    size = sum(row.blob.length for row in manifest_members(body, SOURCE_FILES_MAX_BYTES))
    if size != step.view_bytes:
        raise ValueError("source view content bound changed")
    produced = store.create_tree_root(step.view_owner, manifest.digest, manifest.length)
    tree = {"asset_ref": manifest.digest, "kind": "tree", "digest": manifest.digest}
    return Produced(
        result={"files": {**tree, "size_bytes": size}}, native_receipt=produced["receipt"]
    )


def convert(store: fill.Store, step: Convert) -> Produced:
    observed = store.derived_lookup(step.native_owner)
    receipt: Receipt = (
        observed["receipt"]
        if observed.get("state") == "committed"
        else _compose(store, step, observed.get("writer_session_id"))
    )
    native_receipt = canonical_json.encode(receipt)
    manifest = ObjectRef("sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"])
    digest = "sha256:" + hashlib.sha256(native_receipt).hexdigest()
    artifact = ModelArtifact(step.service_id, "model", manifest, digest)
    # The genuine derived result now holds every inherited model byte independently.
    store.release_model_source(step.service_id)
    return Produced(result=msgspec.to_builtins(artifact), native_receipt=native_receipt)


def _compose(store: fill.Store, step: Convert, epoch: int | None) -> Receipt:
    """One reviewed profile per slot; several profiles compose one model whose components
    are the union of theirs."""
    retained = store.tree_root(step.source_owner)
    if retained is None:
        raise ValueError("conversion lost its retained source")
    total = sum(row["length"] for row in store.walk(retained["manifest_digest"]))
    checkpoints: list[tuple[str, str, int]] = []
    adopt, written = step.adopt, 0
    report("convert", 0, total)
    while True:
        prepared = store.prepare_source_artifact(
            step.source_owner,
            step.service_id,
            step.slots,
            checkpoints=checkpoints,
            adopt_from_operation_id=adopt,
            registry=step.registry,
        )
        adopt = None
        written += prepared["converted_bytes"]
        if prepared["complete"]:
            report("convert", total, total)
            break
        # Checkpoint heads carry resumed work; this process's passes carry the rest.
        durable = sum(row["bytes"] for row in prepared["checkpoints"])
        report("convert", min(max(durable, written), total), total)
        checkpoints = [
            (row["slot"], row["head"], row["head_length"]) for row in prepared["checkpoints"]
        ]
    converted = {row["slot"]: row for row in prepared["sources"]}
    if epoch:
        store.derived_fence(step.native_owner, epoch)
    sources: dict[str, tuple[str, int]] = {}
    targets: dict[str, dict[str, object]] = {}
    configs: dict[str, dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    assets: dict[str, list[str]] = {}
    for slot, _ in step.slots:
        part = converted[slot]
        sources[slot] = (part["manifest_digest"], part["manifest_length"])
        header = fill.tensorfs_module().parse_header(
            store.manifest(part["manifest_digest"])["header"]
        )
        for name, tensors in header["components"].items():
            if name in targets:
                raise SourceRefusal(
                    "model_source_components_overlap",
                    f"source profiles both produce component {name}",
                )
            targets[name] = {"source": slot, "source_component": name, "drop": [], "add": {}}
            order.extend((name, key) for key in tensors)
        for name in header["configs"]:
            if name in configs:
                raise SourceRefusal(
                    "model_source_configs_overlap", f"source profiles both carry config {name}"
                )
            configs[name] = {"kind": "copy", "source": slot, "source_config": name}
        if header["assets"]:
            assets[slot] = sorted(header["assets"])
    writer = store.begin_derived(
        step.native_owner,
        step.writer_epoch,
        sources,
        targets,
        configs,
        order,
        0,
        work_fingerprint=step.computation_digest,
    )
    receipt: Receipt = writer.commit()
    for slot, names in assets.items():
        inherited = store.inherit_model_assets(
            receipt["manifest"]["sha256"],
            receipt["manifest"]["length"],
            sources[slot][0],
            sources[slot][1],
            names,
        )
        receipt["manifest"] = {
            "sha256": str(inherited["manifest_id"]).removeprefix("sha256:"),
            "length": int(inherited["manifest_length"]),
        }
    return receipt


def execute(launch: Launch[SourceStep]) -> Answer:
    store = fill.store(launch.store)
    match launch.step:
        case ResolveSource() as step:
            selection = store.resolve_source(
                step.uri, step.carriers, profiles=step.profiles, files=step.files, **step.access
            )
            return Resolved(selection=msgspec.convert(selection, Selection))
        case Download() as step:
            return materialize(store, step)
        case Convert() as step:
            return convert(store, step)
        case View() as step:
            return view(store, step)
        case CommitStep() as step:
            return commit_files.execute(store, step)


if __name__ == "__main__":
    serve(Launch[SourceStep], execute)
