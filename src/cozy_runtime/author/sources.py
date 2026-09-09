"""Awaitable native source operations for ordinary private Python scripts.

The admitted deployment adapter owns credentials and source acceptance. Calls use
TensorFS locally or on the private worker and never publish the script or its sources.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, cast

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact, SourceArtifact
from cozy_runtime.author._assets import Tree
from cozy_runtime.author._calls import _current
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._markers import AssetBound

SOURCE_FILES_MAX_BYTES = 64 << 20


class _SourceFilesResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    files: Annotated[Tree, AssetBound(max_bytes=SOURCE_FILES_MAX_BYTES)]


def _call(operation: str, arguments: dict[str, object]) -> Awaitable[object]:
    broker = _current.get()
    if broker is None:
        raise CapabilityError(
            "source operation has no admitted attempt", code="child_broker_absent"
        )
    return broker.reserve("", __name__, operation, arguments)


def download_huggingface(
    repository: str,
    *,
    revision: str,
    profiles: tuple[str, ...] = (),
    carriers: tuple[str, ...] = (),
    files: tuple[str, ...] = (),
) -> Awaitable[SourceArtifact]:
    """Retain reviewed carriers or exact ordinary files at an immutable HF revision."""
    return cast(
        Awaitable[SourceArtifact],
        _call(
            "download_huggingface",
            {
                "repository": repository,
                "revision": revision,
                "profiles": profiles,
                "carriers": carriers,
                "files": files,
            },
        ),
    )


def download_civitai(version: int, *, file: str = "") -> Awaitable[SourceArtifact]:
    """Retain the selected Civitai file after pinning its SHA-256 and exact length."""
    return cast(
        Awaitable[SourceArtifact], _call("download_civitai", {"version": version, "file": file})
    )


def convert_cozytensors(source: SourceArtifact, *, profile: str) -> Awaitable[ModelArtifact]:
    """Convert retained source bytes with one reviewed native TensorFS profile."""
    return cast(
        Awaitable[ModelArtifact],
        _call("convert_cozytensors", {"source": source, "profile": profile}),
    )


async def source_files(source: SourceArtifact) -> Tree:
    """Grant a readonly view of a retained ordinary source, bounded to 64 MiB."""
    result = cast(_SourceFilesResult, await _call("source_files", {"source": source}))
    return result.files
