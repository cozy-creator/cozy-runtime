"""Reviewed model-owned source recipes; no inference imports or network effects."""

from __future__ import annotations

from dataclasses import dataclass

from cozy_runtime.author import UnsupportedInput


@dataclass(frozen=True)
class Recipe:
    name: str
    repository: str
    revision: str
    profile: str
    carriers: tuple[str, ...]
    metadata: dict[str, tuple[int, str]]


def huggingface_recipe(repository: str, revision: str) -> Recipe | None:
    from cozy_runtime.models.qwen_image21 import ingestion as qwen

    if repository.lower() != qwen.REPOSITORY.lower():
        return None
    if revision != qwen.REVISION:
        raise UnsupportedInput(
            "Qwen ingestion requires its reviewed immutable revision " + qwen.REVISION,
            code="model_ingestion_revision",
        )
    return Recipe(
        "qwen-image-2.1/original/1",
        qwen.REPOSITORY,
        qwen.REVISION,
        "hf/qwen/qwen-image-2.1/original/1",
        (
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "text_encoder/model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
        dict(qwen.FILES),
    )


def require_recipe(repository: str, revision: str) -> Recipe:
    recipe = huggingface_recipe(repository, revision)
    if recipe is None:
        raise UnsupportedInput(
            f"No reviewed complete-model ingestion recipe for {repository}",
            code="model_ingestion_recipe_unavailable",
        )
    return recipe
