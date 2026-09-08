"""Awaitable native source operations for ordinary private Python scripts.

The admitted deployment adapter owns credentials and source acceptance. Calls use
TensorFS locally or on the private worker and never publish the script or its sources.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import cast

from cozy_runtime.author._artifacts import ModelArtifact, SourceArtifact
from cozy_runtime.author._calls import _current
from cozy_runtime.author._errors import CapabilityError


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
) -> Awaitable[SourceArtifact]:
    """Retain the reviewed carrier selection at an immutable HuggingFace revision."""
    return cast(
        Awaitable[SourceArtifact],
        _call(
            "download_huggingface",
            {
                "repository": repository,
                "revision": revision,
                "profiles": profiles,
                "carriers": carriers,
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
