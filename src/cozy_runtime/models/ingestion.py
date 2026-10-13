"""Reviewed model-owned source recipes; no inference imports or network effects."""

from __future__ import annotations

from dataclasses import dataclass

from cozy_runtime.author import UnsupportedInput
from cozy_runtime.models.birefnet import ingestion as birefnet


@dataclass(frozen=True)
class Recipe:
    name: str
    repository: str
    revision: str
    profile: str
    carriers: tuple[str, ...]
    metadata: dict[str, tuple[int, str]]


def huggingface_recipe(repository: str, revision: str) -> Recipe | None:
    if repository.lower() != birefnet.REPOSITORY.lower():
        return None
    if revision != birefnet.REVISION:
        raise UnsupportedInput(
            "BiRefNet ingestion requires its reviewed immutable revision " + birefnet.REVISION,
            code="model_ingestion_revision",
        )
    return Recipe(
        "birefnet/original/1",
        birefnet.REPOSITORY,
        birefnet.REVISION,
        "as-is/1",
        ("model.safetensors",),
        dict(birefnet.FILES),
    )
