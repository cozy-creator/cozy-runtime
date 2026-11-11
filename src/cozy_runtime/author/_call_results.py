"""Decode host-granted results without widening the request file decoder."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import msgspec
from tensorfs.executor import ByteGrant

from cozy_runtime.author._decode import MediaReads
from cozy_runtime.author._files import (
    AudioFile,
    File,
    ImageFile,
    Tree,
    VideoFile,
    asset_inputs,
    bind,
    file_state,
    metadata_map,
)
from cozy_runtime.author._media import admits, normalize
from cozy_runtime.canonical_json import JsonValue

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ASSETS = {kind.kind: kind for kind in (File, ImageFile, AudioFile, VideoFile)}


class ByteResult(msgspec.Struct, frozen=True):
    """Consumed fields of a native result; producer additions remain readable."""

    asset_ref: str
    kind: str
    digest: str
    size_bytes: int


class FileResult(ByteResult, frozen=True):
    media_type: str


def result_at(value: object, path: str) -> object:
    for part in path.split(".") if path else ():
        if isinstance(value, list) and part.isdigit() and str(int(part)) == part:
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError("native result grant does not name a result field")
    return value


def decode(
    value: object,
    result_type: object,
    grants: Sequence[ByteGrant],
    *,
    request_id: str,
    guard: Callable[[], None],
    grant_token: object | None = None,
    media: MediaReads | None = None,
) -> object:
    """Every native result leaf is replaced from the separately verified host grant, a handle of
    the type its field declares: a `Latents` field's grant is a `Latents`, though on the wire it
    is a file like any other."""
    if len(grants) > 32:
        raise ValueError("native result grant inventory exceeds its bound")

    said: dict[str, dict[str, JsonValue]] = {}

    def handle(cls: type[File] | type[Tree], grant: ByteGrant) -> File | Tree:
        made: File | Tree
        if issubclass(cls, Tree):
            made = cls(grant.digest, digest=grant.digest, root=grant.local, attempt=request_id)
            made.size_bytes = grant.content_bytes
        else:
            made = cls(
                grant.digest,
                digest=grant.digest,
                media_type=grant.media_type,
                size_bytes=grant.length,
                local=grant.local,
                attempt=request_id,
            )
        made._read_guard = guard
        made._grant_token = grant_token
        if isinstance(made, File):
            # What the child said of its file stays with it: returned on, it's the parent's too.
            made._metadata = said.get(grant.output_id, {})
        return made

    verified: dict[str, ByteGrant] = {}
    handles: dict[str, File | Tree] = {}
    for grant in grants:
        output_id = grant.output_id
        if output_id in handles:
            raise ValueError("native result grant has a repeated or missing output ID")
        node = result_at(value, output_id)
        if not isinstance(node, dict):
            raise ValueError("native result field is not a file record")
        declared = msgspec.convert(node, ByteResult, strict=True)
        kind = declared.kind
        if kind not in _ASSETS and kind != "tree":
            raise ValueError("native result field has an invalid file shape")
        digest, size = declared.digest, declared.size_bytes
        if kind == "tree":
            expected_size = grant.content_bytes
            media_type = ""
        else:
            expected_size = grant.length
            media_type = msgspec.convert(node, FileResult, strict=True).media_type
            if media_type != grant.media_type or not admits(kind, normalize(media_type)):
                raise ValueError("native result differs from its verified host grant")
        if declared.asset_ref != digest:
            raise ValueError("native result reference differs from its byte identity")
        if (
            kind != grant.kind
            or _DIGEST.fullmatch(digest) is None
            or digest != grant.digest
            or size < 0
            or size != expected_size
            or not grant.local.is_absolute()
        ):
            raise ValueError("native result differs from its verified host grant")
        verified[output_id] = grant
        said[output_id] = metadata_map(node.get("metadata") or {})
        handles[output_id] = handle(Tree if kind == "tree" else _ASSETS[kind], grant)

    def replace(node: object, path: str) -> object:
        if path in handles:
            return handles[path]
        if isinstance(node, dict):
            return {
                key: replace(item, f"{path}.{key}" if path else key) for key, item in node.items()
            }
        if isinstance(node, list):
            return [
                replace(item, f"{path}.{index}" if path else str(index))
                for index, item in enumerate(node)
            ]
        return node

    def granted(target: type[object], item: object) -> File | Tree:
        path = next((path for path, held in handles.items() if held is item), None)
        if path is None or not isinstance(target, type) or not issubclass(target, (File, Tree)):
            raise TypeError("result file is not a verified grant of this attempt")
        held = handles[path]
        if not issubclass(target, type(held)) or target.kind != held.kind:
            raise TypeError("result file is not a verified grant of this attempt")
        if target is not type(held):
            # A subclass of its wire kind's class (`Latents` of `File`): the field's own type.
            held = handles[path] = handle(target, verified[path])
        return held

    result = msgspec.convert(replace(value, ""), type=result_type, dec_hook=granted, strict=True)
    # A received result becomes an input to the rest of the parent's Python
    # composition. Preserve its declared per-field media/decode limits through
    # the same concrete-value traversal used for ordinary input hydration.
    for item in asset_inputs(result):
        asset, bound = item.asset, item.bound
        if bound is not None:
            bound.check_kind(asset.kind)
            if (bound.max_bytes is not None and asset.size_bytes > bound.max_bytes) or (
                bound.media_types and asset.media_type not in bound.media_types
            ):
                raise ValueError("native result exceeds its declared file bound")
        assert asset._local is not None
        bind(
            asset,
            local=asset._local,
            attempt=request_id,
            media_type=asset.media_type,
            digest=asset.digest,
            length=asset.size_bytes,
            max_decoded_bytes=(bound.max_decoded_bytes or 0) if bound is not None else 0,
            input_id=item.input_id,
            order=item.order,
            read_guard=guard,
            source_state=file_state(asset._local),
            media=media,
        )
    return result
