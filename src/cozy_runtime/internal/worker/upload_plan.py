"""An upload's identity from its request alone: recipe, conversion slots and carriers."""

from __future__ import annotations

import re
from typing import Any

from cozy_runtime.models.ingestion import Recipe, huggingface_recipe

_CIVITAI_MEMBER = re.compile(r"civitai/files/[1-9][0-9]*")
# Upload refusals a caller can act on; everything else is a native TensorFS code or opaque.
CODES = frozenset(
    {
        "model_ingestion_recipe_unavailable",
        "model_ingestion_revision",
        "model_source_profiles_invalid",
        "model_source_components_overlap",
        "model_source_configs_overlap",
    }
)


class UploadRefusal(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def carrier(member: str) -> bool:
    """Members whose header plans the conversion; ordinary files never are."""
    return (
        member.endswith((".safetensors", ".safetensors.index.json"))
        or _CIVITAI_MEMBER.fullmatch(member) is not None
    )


def recipe_of(arguments: dict[str, Any], operation: str) -> Recipe | None:
    if operation != "upload_huggingface" or arguments["profiles"] or arguments["carriers"]:
        return None
    # An unreviewed revision of a reviewed repository refuses with its own code.
    recipe = huggingface_recipe(arguments["repository"], arguments["revision"])
    if recipe is None:
        raise UploadRefusal(
            "model_ingestion_recipe_unavailable",
            f"no reviewed recipe or profile converts {arguments['repository']}",
        )
    return recipe


def slots(arguments: dict[str, Any], recipe: Recipe | None) -> list[tuple[str, str]]:
    """One reviewed profile converts into ``model``; several compose ``part<i>`` slots."""
    profiles = (recipe.profile,) if recipe is not None else tuple(arguments["profiles"])
    if not profiles or len(set(profiles)) != len(profiles):
        raise UploadRefusal("model_source_profiles_invalid", "name distinct reviewed profiles")
    if len(profiles) == 1:
        return [("model", profiles[0])]
    return [(f"part{index}", profile) for index, profile in enumerate(profiles)]


def source_uri(operation: str, arguments: dict[str, Any]) -> str:
    if operation == "upload_huggingface":
        return f"hf://{arguments['repository']}@{arguments['revision']}"
    return f"civitai://{arguments['version']}"
