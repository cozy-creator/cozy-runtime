"""Declared native output fields acquire independent parent custody before child ACK."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence

import msgspec

from cozy_runtime.author._call_results import result_at
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_models
from .workspace import Workspace, WorkspaceRefusal

#: The suffix of a list output's slot: `references.*` grants `references.0`, `references.1`, ...
LIST = ".*"


class _Bound(msgspec.Struct, frozen=True):
    max_bytes: int | None = None


class _Field(msgspec.Struct, frozen=True):
    name: str
    type: _Schema | str
    asset_bound: _Bound = _Bound()


class _Map(msgspec.Struct, frozen=True):
    key: _Schema | str
    value: _Schema | str


class _Schema(
    msgspec.Struct, frozen=True, rename={"tuple_": "tuple", "list_": "list", "map_": "map"}
):
    """Only output-slot declarations consumed here, with producer additions ignored."""

    asset: str = ""
    input: str = ""
    fields: tuple[_Field, ...] | None = None
    tuple_: tuple[_Schema | str, ...] | None = None
    list_: _Schema | str | None = None
    union: tuple[_Schema | str, ...] | None = None
    map_: _Map | None = None


def slot_of(output_id: str, slots: Collection[str]) -> str | None:
    """The granted slot an output id fills: itself, or the list slot it is an item of."""
    if output_id in slots and not output_id.endswith(LIST):
        return output_id
    head, _, index = output_id.rpartition(".")
    if head and index.isdigit() and str(int(index)) == index and head + LIST in slots:
        return head + LIST
    return None


def expand(fields: Sequence[tuple[str, int]], produced: Collection[str]) -> list[tuple[str, int]]:
    """Fixed slots as they are and each list slot as the items that were produced."""
    slots = {path for path, _ in fields}
    result = []
    for path, bound in fields:
        if not path.endswith(LIST):
            result.append((path, bound))
    for output_id in sorted(produced):
        slot = slot_of(output_id, slots)
        if slot is not None and slot.endswith(LIST):
            result.append((output_id, dict(fields)[slot]))
    return result


def paths(schema: object, path: str = "", bound: int = 256 << 20) -> Iterator[tuple[str, int]]:
    if not isinstance(schema, dict):
        return
    try:
        declared = msgspec.convert(schema, _Schema, strict=True)
    except msgspec.ValidationError as exc:
        raise WorkspaceRefusal("native byte result has an invalid output schema") from exc
    yield from _paths(declared, path, bound)


def _paths(schema: _Schema | str, path: str, bound: int) -> Iterator[tuple[str, int]]:
    if isinstance(schema, str):
        return
    if schema.asset or schema.input == "tree":
        if not path:
            raise WorkspaceRefusal("native byte result needs a named result field")
        yield path, bound
    elif schema.fields is not None:
        for field in schema.fields:
            size = field.asset_bound.max_bytes
            yield from _paths(
                field.type,
                path + "." + field.name if path else field.name,
                min(bound, size) if size is not None else bound,
            )
    elif schema.tuple_ is not None:
        for index, member in enumerate(schema.tuple_):
            yield from _paths(member, path + "." + str(index) if path else str(index), bound)
    elif isinstance(schema.list_, _Schema) and schema.list_.asset:
        # A list output: its items `path.0`, `path.1`, ... are granted by one `path.*` slot.
        if not path:
            raise WorkspaceRefusal("native byte result needs a named result field")
        yield path + LIST, bound
    elif schema.union is not None or schema.list_ is not None or schema.map_ is not None:
        # These shapes need output slots only known after executing the program. The
        # existing fixed-slot grant protocol cannot promise their cardinality.
        if _has_assets(schema):
            raise WorkspaceRefusal("dynamic native result fields require fixed output slots")


def _has_assets(schema: _Schema | str) -> bool:
    if isinstance(schema, str):
        return False
    return (
        bool(schema.asset)
        or schema.input == "tree"
        or any(_has_assets(field.type) for field in schema.fields or ())
        or any(_has_assets(member) for member in schema.tuple_ or ())
        or any(_has_assets(member) for member in schema.union or ())
        or (schema.list_ is not None and _has_assets(schema.list_))
        or (
            schema.map_ is not None
            and (_has_assets(schema.map_.key) or _has_assets(schema.map_.value))
        )
    )


def retain(
    workspace: Workspace,
    owner: str,
    parent: str,
    index: int,
    value: object,
    schema: object,
    outputs: Sequence[pb.OutputEntry],
    *,
    capture: bool = False,
) -> None:
    """Hold each typed output's native bytes for the parent. The entries are the decoded
    protobuf, so an empty output (proto3 `content_bytes` 0, omitted from its document) reads
    as 0 rather than a missing key."""
    by_id = {entry.output_id: entry for entry in outputs}
    fields = expand(list(paths(schema)), by_id)
    if capture:
        fields.append(("runtime.capture", 64 << 20))
    for path, _ in fields:
        entry = by_id.get(path)
        node = result_at(value, path) if path != "runtime.capture" else None
        if (
            entry is None
            or not entry.HasField("native_tree")
            or (
                path != "runtime.capture"
                and (
                    not isinstance(node, dict)
                    or node.get("digest") != documents.spell(entry.digest)
                )
            )
        ):
            raise WorkspaceRefusal("typed child output lacks its exact native result entry")
        if isinstance(node, dict) and node.get("size_bytes") != entry.native_tree.content_bytes:
            raise WorkspaceRefusal("typed child output length differs from native custody")
        machine_models.retain_bytes(
            workspace, owner, parent, f"call.{index}.output/{path}", entry.native_tree
        )


def grants(
    workspace: Workspace,
    owner: str,
    parent: str,
    index: int,
    schema: object,
    *,
    capture: bool = False,
) -> list[pb.ChildByteResultGrant]:
    result = []
    with workspace.locked() as db:
        prefix = f"call.{index}.output/"
        held = {
            row["path"].removeprefix(prefix)
            for row in db.execute(
                "SELECT path FROM execution_model_holds WHERE owner=? AND recipient=? "
                "AND substr(path,1,?)=?",
                (owner, parent, len(prefix), prefix),
            )
        }
        fields = expand(list(paths(schema)), held)
        if capture:
            fields.append(("runtime.capture", 64 << 20))
        for path, _ in fields:
            registered = db.execute(
                "SELECT retention FROM execution_model_holds "
                "WHERE owner=? AND recipient=? AND path=?",
                (owner, parent, prefix + path),
            ).fetchone()
            if registered is None:
                raise WorkspaceRefusal("typed child output has no registered parent custody")
            hold = pb.DerivedRetentionRequest.FromString(registered["retention"])
            row = db.execute(
                "SELECT h.*,b.content_bytes FROM holds h JOIN byte_outputs b "
                "ON b.id=h.transaction_id AND b.owner=h.owner "
                "WHERE h.id=? AND h.owner=? AND h.kind='tree' AND h.state='held'",
                (hold.retention_id, owner),
            ).fetchone()
            if row is None:
                raise WorkspaceRefusal("typed child output parent custody is no longer held")
            result.append(
                pb.ChildByteResultGrant(
                    output_id=path,
                    retention_id=hold.retention_id,
                    source=pb.NativeByteTreeRef(
                        producer_root_id=row["transaction_id"],
                        receipt_digest=row["native_digest"],
                        manifest=pb.Ref(digest=row["manifest"], length=row["manifest_length"]),
                        content_bytes=row["content_bytes"],
                    ),
                )
            )
    return result
