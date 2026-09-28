"""Typed projections minted by Runtime from an existing native Tree capability."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

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
from cozy_runtime.author._calls import _current
from cozy_runtime.author._errors import CapabilityError, ConformanceError, RuntimeFailure
from cozy_runtime.author._executor_requests import Member, TreeMember
from cozy_runtime.author._markers import AssetBound
from cozy_runtime.author._media import admits, normalize


def project[A: Asset](tree: Tree, path: str, asset_type: type[A], *, bound: AssetBound) -> A:
    broker = _current.get()
    if (
        broker is None
        or broker.closed
        or broker.context is None
        or tree._member_token is not broker.grant_token
        or tree._attempt != broker.request_id
        or not tree.hydrated
    ):
        raise CapabilityError(
            "Tree member projection requires this attempt's native Tree",
            code="tree_member_ungranted",
        )
    if asset_type not in (ImageAsset, VideoAsset, AudioAsset, FileAsset):
        raise ConformanceError(
            "Tree.member requires ImageAsset, VideoAsset, AudioAsset or FileAsset"
        )
    if (
        not isinstance(bound, AssetBound)
        or type(bound.max_bytes) is not int
        or bound.max_bytes <= 0
    ):
        raise ConformanceError(
            "Tree.member requires AssetBound(max_bytes=...) with a positive integer"
        )
    bound.check_kind(asset_type.kind)
    if (
        not isinstance(path, str)
        or not path
        or len(path.encode()) > 4096
        or "\\" in path
        or "\x00" in path
        or PurePosixPath(path).is_absolute()
        or str(PurePosixPath(path)) != path
        or any(part in (".", "..") for part in PurePosixPath(path).parts)
    ):
        raise CapabilityError(
            "Tree member path must be a canonical relative file path", code="tree_member_path"
        )

    def guard() -> None:
        if broker.closed:
            raise CapabilityError("Tree member escaped its attempt", code="escaped_handle")
        assert broker.context is not None
        broker.context.raise_if_cancelled()
        if tree._read_guard is not None:
            tree._read_guard()

    guard()
    reply = broker.exchange(
        TreeMember(
            tree=tree.digest,
            path=path,
            asset_kind=asset_type.kind,
            max_bytes=bound.max_bytes,
            media_types=tuple(bound.media_types),
        ),
        Member,
    )
    if not reply.ok:
        raise RuntimeFailure(reply.detail, code=reply.code)
    digest, length, media, local = reply.digest, reply.length, reply.media_type, reply.local
    if (
        reply.tree != tree.digest
        or reply.path != path
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or not 0 <= length <= bound.max_bytes
        or normalize(media) != media
        or not admits(asset_type.kind, media)
        or (bound.media_types and media not in bound.media_types)
        or not Path(local).is_absolute()
        or reply.file_state is None
    ):
        raise RuntimeFailure(
            "Tree member differs from Runtime's verified projection", code="tree_member_invalid"
        )
    guard()
    actual = file_state(Path(local))
    if actual != reply.file_state:
        raise RuntimeFailure("Tree member changed before binding", code="input_changed")
    asset = asset_type(digest)
    bind(
        asset,
        local=Path(local),
        attempt=broker.request_id,
        media_type=media,
        digest=digest,
        length=length,
        max_decoded_bytes=bound.max_decoded_bytes or 0,
        input_id=path,
        read_guard=guard,
        source_state=actual,
    )
    return asset
