"""Source-authored model selection metadata; names are resolved by the granting host.

A lane may omit its org (`model@release/lane`): it names a model of the package's own
account. The publishing host writes that account into the published interface."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from cozy_runtime.author._errors import ConformanceError

# These are declaration shapes, not a catalog client or a GPU matching implementation.
_SLUG = r"[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?"
_REFERENCE = re.compile(
    rf"(?P<model>(?:{_SLUG}/)?{_SLUG})@(?P<release>[A-Za-z0-9][A-Za-z0-9._+!-]{{0,63}})"
    r"/(?P<lane>[A-Za-z0-9][A-Za-z0-9._+!-]{0,127})"
)
_DIGEST = re.compile(r"[0-9a-f]{64}")

type DefaultLadder = tuple[tuple[str, int, str], ...]


def default_ladder(value: object) -> DefaultLadder:
    """Validate and freeze one ordered ladder without resolving any reference."""
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 32:
        raise ConformanceError(
            "a model default ladder has 1 through 32 rungs", code="model_defaults"
        )
    out: list[tuple[str, int, str]] = []
    target: tuple[str, str] | None = None
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) not in (
            {"gpu", "lane"},
            {"gpu", "gpus", "lane"},
        ):
            raise ConformanceError(
                "each model default rung carries gpu, optional gpus, and lane",
                code="model_defaults",
            )
        gpu, lane = row["gpu"], row["lane"]
        gpus = 0
        if "gpus" in row:
            count = row["gpus"]
            if type(count) is not int or count < 1:
                raise ConformanceError(
                    "default gpus must be a positive integer", code="model_defaults"
                )
            gpus = count
        if (
            not isinstance(gpu, str)
            or not 1 <= len(gpu.encode()) <= 64
            or gpu != gpu.strip()
            or (gpu == "*" and index != len(value) - 1)
            or (gpu != "*" and not any(char.isalnum() for char in gpu))
        ):
            raise ConformanceError(
                "default gpu must be 1..64 bytes with a letter/digit token; "
                "'*' is allowed only last",
                code="model_defaults",
            )
        reference = _REFERENCE.fullmatch(lane) if isinstance(lane, str) else None
        if reference is None or _DIGEST.fullmatch(reference["release"]):
            raise ConformanceError(
                "default lane must name [org/]model@release/lane with an explicit release and lane",
                code="model_defaults",
            )
        current = reference["model"], reference["release"]
        if target is not None and current != target:
            raise ConformanceError(
                "one model default ladder must share a model and release; "
                "mixed targets are unsupported",
                code="model_defaults",
            )
        target = current
        out.append((gpu, gpus, lane))
    return tuple(out)


def reference(lane: str) -> tuple[str, str, str]:
    """One validated rung's ([org/]model, release, lane)."""
    match = _REFERENCE.fullmatch(lane)
    if match is None:
        raise ConformanceError(
            "default lane is not [org/]model@release/lane", code="model_defaults"
        )
    return match["model"], match["release"], match["lane"]


def model_defaults(value: object) -> dict[str, DefaultLadder]:
    """Copy a decorator declaration so later mutations cannot rewrite registered defaults."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 16:
        raise ConformanceError(
            "defaults must map at most 16 Model arguments to ladders", code="model_defaults"
        )
    out = {}
    for parameter, ladder in value.items():
        if not isinstance(parameter, str) or not parameter.isidentifier():
            raise ConformanceError(
                "model defaults keys must be argument names", code="model_defaults"
            )
        out[parameter] = default_ladder(ladder)
    return out


def check_model_arguments(defaults: Mapping[str, DefaultLadder], parameters: Sequence[str]) -> None:
    unknown = sorted(set(defaults) - set(parameters))
    if unknown:
        raise ConformanceError(
            "defaults name arguments that are not injected Models: " + ", ".join(unknown),
            code="model_defaults",
            fields=unknown,
        )
