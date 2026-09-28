"""Project typed child byte inputs from the parent's existing native capabilities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, NamedTuple

import msgspec

from cozy_runtime.author._media import admits, normalize
from cozy_runtime.internal.canonical import Json
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import byte_inputs, grants, machine_models, workspace_byte_outputs
from .workspace import Workspace, WorkspaceRefusal

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .calls import Calls


class Bound(msgspec.Struct, frozen=True):
    """A field's declared `asset_bound`; absent limits are the input plane's own."""

    max_bytes: int = byte_inputs.MAX_BYTES
    media_types: tuple[str, ...] = ()

    @property
    def limit(self) -> int:
        return min(self.max_bytes, byte_inputs.MAX_BYTES)

    def admits(self, media_type: str) -> bool:
        return not self.media_types or media_type in self.media_types


_UNDECLARED = Bound()


class Leaf(NamedTuple):
    name: str
    order: int
    kind: str  # "tree", or the asset kind of a file
    digest: str
    bound: Bound


class _Received(msgspec.Struct, frozen=True):
    """A byte capability the parent holds (`Calls.received_inputs` metadata)."""

    kind: str
    digest: str
    media_type: str
    length: int = 0  # a file's; a tree's is its manifest's


def _leaves(
    value: Json,
    schema: Json,
    path: tuple[str | int, ...] = (),
    order: int = 0,
    bound: Bound = _UNDECLARED,
) -> Iterator[Leaf]:
    if not isinstance(schema, dict):
        return
    if schema.get("asset") or schema.get("input") == "tree":
        if not path or any(isinstance(p, str) and (not p or "." in p) for p in path):
            raise WorkspaceRefusal("native child input has an ambiguous field path")
        name = ".".join(map(str, path))
        if name == "payload" or name.startswith(("tree:", "model:")):
            raise WorkspaceRefusal("native child input collides with a reserved binding")
        if not isinstance(value, str):
            raise WorkspaceRefusal("native child input must carry an exact digest reference")
        documents.raw(value)
        yield Leaf(name, order, str(schema.get("asset") or "tree"), value, bound)
    elif "fields" in schema and isinstance(value, dict):
        for field in schema["fields"]:
            if field["name"] in value:
                yield from _leaves(
                    value[field["name"]],
                    field["type"],
                    (*path, field["name"]),
                    order,
                    msgspec.convert(field["asset_bound"], Bound, strict=True)
                    if "asset_bound" in field
                    else bound,
                )
    elif "list" in schema and isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, schema["list"], (*path, index), index, bound)
    elif "tuple" in schema and isinstance(value, list):
        for index, (item, member) in enumerate(zip(value, schema["tuple"], strict=True)):
            yield from _leaves(item, member, (*path, index), index, bound)
    elif "map" in schema and isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, schema["map"]["value"], (*path, key), order, bound)
    elif "union" in schema:
        branches = schema["union"]
        if value is None and "null" in branches:
            return
        selected = [branch for branch in branches if branch != "null"]
        if tag := schema.get("tag_field"):
            selected = [
                branch
                for branch in selected
                if isinstance(branch, dict)
                and isinstance(value, dict)
                and branch.get("tag") == value.get(tag)
            ]
        if len(selected) == 1:
            yield from _leaves(value, selected[0], path, order, bound)
        elif any(list(_leaves(value, branch, path, order, bound)) for branch in selected):
            raise WorkspaceRefusal("ambiguous input union cannot grant native bytes")


def inputs(
    workspace: Workspace,
    owner: str,
    parent: AttemptRecord,
    recipient: str,
    schema: Json,
    arguments: Json,
    calls: Calls,
) -> tuple[list[pb.InputBinding], list[pb.InputAccess]]:
    """Acquire recipient custody before dispatch; existing grant loaders own byte access."""
    selected = list(_leaves(arguments, schema))
    if not selected:
        return [], []
    candidates = [
        (msgspec.convert(metadata, _Received, strict=True), source)
        for metadata, source in calls.received_inputs(parent)
    ]
    if parent.grant is not None:
        for entry in parent.grant.inputs.values():
            if entry.native_tree is not None:
                tree = entry.kind_mime == byte_inputs.TREE_MIME or entry.input_id.startswith(
                    "tree:"
                )
                media_type = (
                    parent.inputs[entry.input_id].media_type
                    if entry.input_id in parent.inputs
                    else normalize(entry.kind_mime)
                )
                candidates.append(
                    (
                        _Received(
                            "tree" if tree else "file",
                            documents.spell(entry.digest),
                            media_type,
                            entry.length,
                        ),
                        entry.native_tree,
                    )
                )
    bindings, access = [], []
    for name, order, kind, digest, bound in selected:
        matches = [
            (metadata, source)
            for metadata, source in candidates
            if metadata.digest == digest
            and (metadata.kind == "tree") == (kind == "tree")
            and (kind == "tree" or admits(kind, metadata.media_type))
            and bound.admits(metadata.media_type)
        ]
        if not matches:
            raise WorkspaceRefusal("native child input has no exact parent capability")
        metadata, source = matches[0]
        if source.source.content_bytes > bound.limit:
            raise WorkspaceRefusal("native child input exceeds its declared byte bound")
        media = metadata.media_type
        with workspace_byte_outputs.leased(workspace, owner, source) as (_, _, members):
            if kind == "tree":
                if digest != documents.spell(source.source.manifest.digest):
                    raise WorkspaceRefusal("native child tree differs from its source manifest")
                length, media = source.source.manifest.length, byte_inputs.TREE_MIME
            else:
                blob = byte_inputs.file_member(source, members)
                if (digest, metadata.length) != ("sha256:" + blob.sha256, blob.length):
                    raise WorkspaceRefusal("native child file differs from its source payload")
                length = blob.length
            retained = machine_models.retain_bytes(
                workspace, owner, recipient, "input/" + name, source.source
            )
        bindings.append(
            pb.InputBinding(
                input_id=name, digest=digest, length=length, kind_mime=media, order=order
            )
        )
        access.append(
            pb.InputAccess(
                input_id=name,
                native_tree=pb.NativeByteRetentionRequest(
                    source=retained.source, retention_id=retained.retention_id
                ),
            )
        )
    return bindings, access


def retain_root(
    workspace: Workspace,
    owner: str,
    offer: pb.AttemptOffer,
    schema: Json,
    arguments: Mapping[str, Json],
) -> None:
    """Acquire execution custody before acceptance without changing replay identity."""
    selected = list(_leaves(arguments, schema))
    native = {
        item.input_id: item.native_tree
        for item in offer.grant.inputs
        if item.HasField("native_tree")
    }
    if not selected and not native:
        return
    if set(native) != {leaf.name for leaf in selected}:
        raise WorkspaceRefusal("root byte inputs differ from the captured job arguments")
    spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    bound = grants.bind(spec, offer.grant, offer.invocation_spec_digest)
    # Validate every declared object before creating any execution recipient.
    for name, order, kind, digest, capacity in selected:
        source, entry = native[name], bound.inputs[name]
        if (
            source.source.content_bytes > capacity.limit
            or documents.spell(entry.digest) != digest
            or entry.order != order
        ):
            raise WorkspaceRefusal("root byte input changed its declared identity or capacity")
        with workspace_byte_outputs.leased(workspace, owner, source) as (_, _, members):
            if kind == "tree":
                if (
                    entry.kind_mime != byte_inputs.TREE_MIME
                    or entry.digest != source.source.manifest.digest
                    or entry.length != source.source.manifest.length
                ):
                    raise WorkspaceRefusal("root tree differs from its exact native manifest")
            else:
                blob = byte_inputs.file_member(source, members)
                if (
                    (digest, entry.length) != ("sha256:" + blob.sha256, blob.length)
                    or not admits(kind, entry.kind_mime)
                    or not capacity.admits(entry.kind_mime)
                ):
                    raise WorkspaceRefusal("root file differs from its exact native payload")
    for name, *_ in selected:
        machine_models.retain_bytes(
            workspace, owner, offer.request_id, "input/" + name, native[name].source
        )


def bind_root_grant(owner: str, offer: pb.AttemptOffer) -> None:
    """Dispatch a job root from the recipient `retain_root` owns.

    Only a job root's inputs are adopted at submission; a serving root dispatches from the
    submitter's own hold, which the submitter keeps until the execution ends.
    """
    if not documents.parse(offer.invocation_spec_canonical_bytes, pb.InvocationSpec).HasField(
        "job"
    ):
        return
    for item in offer.grant.inputs:
        if item.HasField("native_tree"):
            item.native_tree.retention_id = machine_models.byte_retention_id(
                owner, offer.request_id, "input/" + item.input_id, item.native_tree.source
            )
