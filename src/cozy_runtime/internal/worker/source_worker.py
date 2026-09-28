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
from cozy_runtime.internal import fill, upload_child
from cozy_runtime.internal.worker import commit_files
from cozy_runtime.internal.worker.source_steps import (
    Answer,
    Composition,
    Convert,
    Download,
    Launch,
    Member,
    Prepared,
    Produced,
    Resolve,
    Resolved,
    ResolveSource,
    Selection,
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
        else _compose(store, step)
    )
    native_receipt = canonical_json.encode(receipt)
    manifest = ObjectRef("sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"])
    digest = "sha256:" + hashlib.sha256(native_receipt).hexdigest()
    artifact = ModelArtifact(step.service_id, "model", manifest, digest)
    # The genuine derived result now holds every inherited model byte independently.
    store.release_model_source(step.service_id)
    return Produced(result=msgspec.to_builtins(artifact), native_receipt=native_receipt)


def references(store: fill.Store, step: Convert) -> list[Member]:
    """The converters' pinned reference files (index, configs, tokenizers) for a source that
    carries no pipeline index, retained once per Store under the reference's own identity."""
    source = store.tree_root(step.source_owner)
    if source is None:
        raise ValueError("conversion lost its retained source")
    entries = msgspec.json.decode(store.manifest(source["manifest_digest"])["manifest"])
    if {row.get("path") for row in entries["entries"]} & set(upload_child.DIFFUSERS_INDEXES):
        return []
    rows: list[Member] = []
    profiles = [profile for _, profile in step.slots]
    converters = fill.tensorfs_module().source_profile_converters(
        store, profiles, registry=step.registry
    )
    for _, reference in converters:
        if not reference:
            continue
        resolve = Resolve(
            uri=reference,
            carriers=[],
            profiles=[],
            metadata=[],
            diffusers=True,
            registry=step.registry,
            access=step.access,
        )
        pinned = upload_child.reference_members(store, resolve, reference)
        owner = (
            "sha256:" + hashlib.sha256(b"converter-reference\0" + reference.encode()).hexdigest()
        )
        store.materialize_source(
            owner,
            pinned.members,
            allowed_hosts=pinned.allowed_hosts,
            credential_hosts=pinned.credential_hosts,
            credential=step.access["credential"],
            allow_local=step.access["allow_local"],
        )
        rows.extend(pinned.members)
    return rows


def _compose(store: fill.Store, step: Convert) -> Receipt:
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
    composition = Composition(
        native_owner=step.native_owner,
        writer_epoch=step.writer_epoch,
        computation_digest=step.computation_digest,
        metadata=references(store, step),
        recipe="",
    )
    upload_child.compose(store, composition, msgspec.convert(prepared["sources"], list[Prepared]))
    receipt: Receipt = store.derived_lookup(step.native_owner)["receipt"]
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
