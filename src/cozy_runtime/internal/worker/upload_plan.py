"""An upload's identity from its request alone: recipe, conversion slots and carriers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from cozy_runtime.internal.source_interfaces import UploadCivitai, UploadHuggingFace
from cozy_runtime.internal.worker.source_steps import SourceRefusal
from cozy_runtime.models.ingestion import Recipe, huggingface_recipe

type Request = UploadHuggingFace | UploadCivitai

_CIVITAI_MEMBER = re.compile(r"civitai/files/[1-9][0-9]*")
# TensorFS's profile for a source no reviewed profile recognizes: its keys, stored as-is.
AS_IS = "as-is/1"
AS_IS_NOTE = "unrecognized layout; stored as-is; packages may need to normalize it"

# The TensorFS bindings an upload calls, and the first release that carries them all.
BINDINGS = (
    "read_source_heads",
    "prepare_selected_source",
    "record_model_source_custody",
    "model_source_objects",
)
TENSORFS_FLOOR = "0.3.60"


def require_bindings(tensorfs: object, operation: str) -> None:
    """Refuse before any byte moves when the installed TensorFS predates uploads."""
    if not all(callable(getattr(tensorfs, name, None)) for name in BINDINGS):
        raise SourceRefusal(
            "tensorfs_capability_missing", f"{operation} needs tensorfs >= {TENSORFS_FLOOR}"
        )


def carrier(member: str) -> bool:
    """Members whose header plans the conversion; ordinary files never are."""
    return (
        member.endswith((".safetensors", ".safetensors.index.json"))
        or _CIVITAI_MEMBER.fullmatch(member) is not None
    )


def recipe_of(request: Request) -> Recipe | None:
    if not isinstance(request, UploadHuggingFace) or request.profiles or request.carriers:
        return None
    # An unreviewed revision of a reviewed repository refuses with its own code.
    return huggingface_recipe(request.repository, request.revision)


def slots(profiles: Sequence[str]) -> list[tuple[str, str]]:
    """One reviewed profile converts into ``model``; several compose ``part<i>`` slots.

    None named is empty: the machine selects the profile from the headers.
    """
    if len(set(profiles)) != len(profiles):
        raise SourceRefusal("model_source_profiles_invalid", "name distinct reviewed profiles")
    if len(profiles) == 1:
        return [("model", profiles[0])]
    return [(f"part{index}", profile) for index, profile in enumerate(profiles)]


def source_uri(request: Request) -> str:
    if isinstance(request, UploadHuggingFace):
        return f"hf://{request.repository}@{request.revision}"
    return f"civitai://{request.version}"
