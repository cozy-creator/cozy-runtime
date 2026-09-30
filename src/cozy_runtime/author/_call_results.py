"""Decode host-granted results without widening the request asset decoder."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author._assets import (
    Asset,
    AudioAsset,
    FileAsset,
    ImageAsset,
    Tree,
    VideoAsset,
    bind,
    file_state,
)
from cozy_runtime.author._capture import ExecutionObservation
from cozy_runtime.author._media import admits, normalize

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ASSETS = {kind.kind: kind for kind in (FileAsset, ImageAsset, AudioAsset, VideoAsset)}


class ByteResult(msgspec.Struct, frozen=True):
    """Consumed fields of a native result; producer additions remain readable."""

    asset_ref: str
    kind: str
    digest: str
    size_bytes: int


class FileResult(ByteResult, frozen=True):
    media_type: str


class _Grant(msgspec.Struct, frozen=True):
    """Fields consumed when reading a grant from another Runtime's worker."""

    output_id: str
    kind: str
    digest: str
    local: str


class _FileGrant(_Grant, frozen=True):
    length: int
    media_type: str


class _TreeGrant(_Grant, frozen=True):
    content_bytes: int


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
    grants: Sequence[Mapping[str, object]],
    *,
    request_id: str,
    guard: Callable[[], None],
    observation: ExecutionObservation | None,
    grant_token: object | None = None,
    python_type: type[msgspec.Struct] | None = None,
) -> tuple[Any, ExecutionObservation | None]:
    """Every native result leaf is replaced from the separately verified host grant."""
    if len(grants) > 32:
        raise ValueError("native result grant inventory exceeds its bound")
    handles: dict[str, Asset | Tree] = {}
    for raw_grant in grants:
        grant = msgspec.convert(raw_grant, _Grant, strict=True)
        output_id = grant.output_id
        if output_id in handles:
            raise ValueError("native result grant has a repeated or missing output ID")
        capture = output_id == "runtime.capture"
        node = result_at(value, output_id) if not capture else None
        if capture:
            if observation is None or observation.capture is None:
                raise ValueError("capture grant has no matching execution observation")
            expected_size = msgspec.convert(raw_grant, _TreeGrant, strict=True).content_bytes
            kind, size, media_type = "tree", expected_size, ""
            digest = grant.digest
        else:
            if not isinstance(node, dict):
                raise ValueError("native result field is not an asset record")
            declared = msgspec.convert(node, ByteResult, strict=True)
            kind = declared.kind
            if kind not in _ASSETS and kind != "tree":
                raise ValueError("native result field has an invalid asset shape")
            digest, size = declared.digest, declared.size_bytes
            if kind == "tree":
                expected_size = msgspec.convert(raw_grant, _TreeGrant, strict=True).content_bytes
                media_type = ""
            else:
                file_grant = msgspec.convert(raw_grant, _FileGrant, strict=True)
                expected_size = file_grant.length
                media_type = msgspec.convert(node, FileResult, strict=True).media_type
                if media_type != file_grant.media_type or not admits(kind, normalize(media_type)):
                    raise ValueError("native result differs from its verified host grant")
            if declared.asset_ref != digest:
                raise ValueError("native result reference differs from its byte identity")
        if (
            kind != grant.kind
            or _DIGEST.fullmatch(digest) is None
            or digest != grant.digest
            or size < 0
            or size != expected_size
            or not Path(grant.local).is_absolute()
        ):
            raise ValueError("native result differs from its verified host grant")
        if kind == "tree":
            tree = Tree(digest, digest=digest, root=Path(grant.local), attempt=request_id)
            tree.size_bytes = size
            tree._read_guard = guard
            tree._member_token = grant_token
            handle: Asset | Tree = tree
        else:
            asset: Asset = _ASSETS[kind](
                digest,
                digest=digest,
                media_type=media_type,
                size_bytes=size,
                local=Path(grant.local),
                attempt=request_id,
            )
            asset._read_guard = guard
            handle = asset
        handle._grant_token = grant_token
        handles[output_id] = handle
        if capture:
            assert observation is not None and observation.capture is not None
            observation = msgspec.structs.replace(
                observation, capture=msgspec.structs.replace(observation.capture, tree=handle)
            )

    if (
        observation is not None
        and observation.capture is not None
        and "runtime.capture" not in handles
    ):
        raise ValueError("capture observation has no verified native result grant")

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

    def granted(target: type[object], item: object) -> Asset | Tree:
        if (
            isinstance(target, type)
            and issubclass(target, (Asset, Tree))
            and isinstance(item, target)
            and any(item is handle for handle in handles.values())
        ):
            return item
        raise TypeError("result asset is not a verified grant of this attempt")

    result = msgspec.convert(replace(value, ""), type=result_type, dec_hook=granted, strict=True)
    if python_type is not None:
        # The generated schema above owns wire admission, including unknown
        # fields and tags. Then return the installed public Python type while
        # retaining each granted native handle as the same object.
        result = msgspec.convert(
            msgspec.to_builtins(
                result, builtin_types=(FileAsset, ImageAsset, AudioAsset, VideoAsset, Tree)
            ),
            type=python_type,
            dec_hook=granted,
            strict=True,
        )
    # A received result becomes an input to the rest of the parent's Python
    # composition. Preserve its declared per-field media/decode limits through
    # the same concrete-value traversal used for ordinary input hydration.
    from cozy_runtime.author._invoke import asset_inputs

    for item in asset_inputs(result):
        asset, bound = item.asset, item.bound
        if bound is not None:
            bound.check_kind(asset.kind)
            if (bound.max_bytes is not None and asset.size_bytes > bound.max_bytes) or (
                bound.media_types and asset.media_type not in bound.media_types
            ):
                raise ValueError("native result exceeds its declared asset bound")
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
        )
    return result, observation
