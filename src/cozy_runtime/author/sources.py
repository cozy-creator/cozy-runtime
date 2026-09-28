"""Awaitable native source operations for ordinary private Python scripts.

The admitted deployment adapter owns credentials and source acceptance. Calls use
TensorFS locally or on the private worker and never publish the script or its sources;
an upload publishes only its converted checkpoint to the admitted destination.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated, cast

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact, SourceArtifact
from cozy_runtime.author._assets import Tree
from cozy_runtime.author._calls import _current
from cozy_runtime.author._errors import CapabilityError, UnsupportedInput
from cozy_runtime.author._markers import AssetBound
from cozy_runtime.author.publication import CheckpointRef

SOURCE_FILES_MAX_BYTES = 64 << 20


class _SourceFilesResult(msgspec.Struct, frozen=True):
    files: Annotated[Tree, AssetBound(max_bytes=SOURCE_FILES_MAX_BYTES)]


def _call(operation: str, arguments: dict[str, object]) -> Awaitable[object]:
    broker = _current.get()
    if broker is None:
        raise CapabilityError(
            "source operation has no admitted attempt", code="child_broker_absent"
        )
    return broker.reserve(__name__, operation, arguments)


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


def convert_cozytensors(
    source: SourceArtifact, *, profiles: tuple[str, ...]
) -> Awaitable[ModelArtifact]:
    """Convert retained source bytes with reviewed native TensorFS profiles whose disjoint
    components compose one model."""
    if not profiles or len(set(profiles)) != len(profiles):
        raise UnsupportedInput(
            "convert_cozytensors takes distinct profiles", code="model_source_profiles_invalid"
        )
    return cast(
        Awaitable[ModelArtifact],
        _call("convert_cozytensors", {"source": source, "profiles": profiles}),
    )


async def source_files(source: SourceArtifact) -> Tree:
    """Grant a readonly view of a retained ordinary source, bounded to 64 MiB."""
    result = cast(_SourceFilesResult, await _call("source_files", {"source": source}))
    return result.files


async def ingest_huggingface(repository: str, *, revision: str) -> ModelArtifact:
    """Prepare a complete native model using its reviewed model-owned recipe.

    Each source, conversion and metadata derivation retains its own receipt.
    Upload and release publication remain explicit separate operations.
    """
    from cozy_runtime.derive.operations import prepare_model
    from cozy_runtime.models.ingestion import require_recipe

    recipe = require_recipe(repository, revision)
    source = await download_huggingface(
        recipe.repository, revision=recipe.revision, carriers=recipe.carriers
    )
    # The metadata never depends on the conversion, so both run at once. They are
    # reserved in the same order as before, so their call indices are unchanged.
    model, metadata = await asyncio.gather(
        convert_cozytensors(source, profiles=(recipe.profile,)),
        download_huggingface(
            recipe.repository, revision=recipe.revision, files=tuple(recipe.metadata)
        ),
    )
    # The managed caller accepts a receipt; the execution signature deliberately
    # declares the Model that Runtime injects from that receipt at admission.
    prepare = cast(Callable[..., Awaitable[ModelArtifact]], prepare_model)
    return await prepare(source=model, metadata=await source_files(metadata), recipe=recipe.name)


def _profiles(profiles: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(profiles)) != len(profiles):
        raise UnsupportedInput(
            "source profiles must be distinct", code="model_source_profiles_invalid"
        )
    return profiles


def upload_huggingface(
    repository: str,
    *,
    revision: str,
    destination: str,
    profiles: tuple[str, ...] = (),
    carriers: tuple[str, ...] = (),
) -> Awaitable[CheckpointRef]:
    """Download, convert and publish one checkpoint within the machine's free disk.

    Reviewed profiles select and convert the source; carriers narrow the selection to
    exact members. With no profile, the model's reviewed recipe decides and attaches its
    metadata, or the machine's TensorFS selects the profile the headers match. Bodies
    stream through a window sized to the observed disk.
    """
    return cast(
        Awaitable[CheckpointRef],
        _call(
            "upload_huggingface",
            {
                "repository": repository,
                "revision": revision,
                "destination": destination,
                "profiles": _profiles(profiles),
                "carriers": carriers,
            },
        ),
    )


def upload_civitai(
    version: int, *, destination: str, profiles: tuple[str, ...] = (), file: str = ""
) -> Awaitable[CheckpointRef]:
    """Download, convert and publish one Civitai file within the machine's free disk.

    With no profile, the machine's TensorFS selects the one the file's header matches.
    """
    return cast(
        Awaitable[CheckpointRef],
        _call(
            "upload_civitai",
            {
                "version": version,
                "destination": destination,
                "profiles": _profiles(profiles),
                "file": file,
            },
        ),
    )
