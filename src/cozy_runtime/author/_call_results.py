"""Decode host-granted results without widening the request asset decoder."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author._assets import Asset, AudioAsset, FileAsset, ImageAsset, Tree, VideoAsset
from cozy_runtime.author._capture import ExecutionObservation
from cozy_runtime.author._media import admits, normalize

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ASSETS = {kind.kind: kind for kind in (FileAsset, ImageAsset, AudioAsset, VideoAsset)}


def result_at(value: Any, path: str) -> Any:
    for part in path.split(".") if path else ():
        if isinstance(value, list) and part.isdigit() and str(int(part)) == part:
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError("native result grant does not name a result field")
    return value


def decode(
    value: Any,
    result_type: object,
    grants: list[dict[str, Any]],
    *,
    request_id: str,
    guard: Callable[[], None],
    observation: ExecutionObservation | None,
    grant_token: object | None = None,
) -> tuple[Any, ExecutionObservation | None]:
    """Every native result leaf is replaced from the separately verified host grant."""
    if len(grants) > 32:
        raise ValueError("native result grant inventory exceeds its bound")
    kind: Any
    handles: dict[str, Asset | Tree] = {}
    for grant in grants:
        output_id = grant.get("output_id")
        if not isinstance(output_id, str) or output_id in handles:
            raise ValueError("native result grant has a repeated or missing output ID")
        capture = output_id == "runtime.capture"
        node = result_at(value, output_id) if not capture else None
        if capture:
            if observation is None or observation.capture is None:
                raise ValueError("capture grant has no matching execution observation")
            kind, size, media_type = "tree", grant["content_bytes"], ""
            digest = grant["digest"]
        else:
            if not isinstance(node, dict):
                raise ValueError("native result field is not an asset record")
            kind = node.get("kind")
            allowed = {"asset_ref", "kind", "digest", "size_bytes"}
            if kind in _ASSETS:
                allowed.add("media_type")
            if set(node) != allowed or (kind not in _ASSETS and kind != "tree"):
                raise ValueError("native result field has an invalid asset shape")
            digest, size = node["digest"], node["size_bytes"]
            media_type = node.get("media_type", "")
            if node["asset_ref"] != digest:
                raise ValueError("native result reference differs from its byte identity")
        if (
            kind != grant.get("kind")
            or not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or digest != grant.get("digest")
            or type(size) is not int
            or size < 0
            or size != grant.get("content_bytes" if kind == "tree" else "length")
            or (
                kind in _ASSETS
                and (
                    media_type != grant.get("media_type") or not admits(kind, normalize(media_type))
                )
            )
            or not isinstance(grant.get("local"), str)
            or not Path(grant["local"]).is_absolute()
        ):
            raise ValueError("native result differs from its verified host grant")
        if kind == "tree":
            tree = Tree(digest, digest=digest, root=Path(grant["local"]), attempt=request_id)
            tree.size_bytes = size
            tree._read_guard = guard
            handle: Asset | Tree = tree
        else:
            asset = _ASSETS[kind](
                digest,
                digest=digest,
                media_type=media_type,
                size_bytes=size,
                local=Path(grant["local"]),
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

    def replace(node: Any, path: str) -> Any:
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

    def granted(target: type, item: Any) -> Any:
        if (
            isinstance(target, type)
            and issubclass(target, (Asset, Tree))
            and isinstance(item, target)
            and any(item is handle for handle in handles.values())
        ):
            return item
        raise TypeError("result asset is not a verified grant of this attempt")

    return msgspec.convert(
        replace(value, ""), type=result_type, dec_hook=granted, strict=True
    ), observation
