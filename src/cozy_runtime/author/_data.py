"""Interface values a package reads from its own committed data asset.

The admitted values of a wire enum are often a fact of a committed TABLE rather than of
code: MiniMax-H3's served denoise steps are its task plan's, and decision #705 forbids a
second copy of them in serving code. Spelling that derivation as a call into package code
puts the interface behind execution, which the static reader refuses (#713, cr-114).

This reads the DATA — never code. `describe` and the imported app take the values from the
same bytes through this same function, so the two derivations cannot drift, and the asset
that ships is the one owner of the fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author._errors import ConformanceError


def data_values(anchor: str, asset: str, *path: str) -> tuple[Any, ...]:
    """The values a committed JSON asset lists at `path`, as a tuple.

    `anchor` is the calling module's `__file__`; `asset` is a relative path under that
    module's own directory, so source tree and installed wheel read the same bytes. Each
    `path` step indexes an object by key. Exactly one step reaches a JSON list — the rows —
    and any remaining steps select a field of each row::

        SUPPORTED_STEPS = data_values(__file__, "plan.json", "schedules", "evaluations")

    Only JSON scalars are interface content; a row that is not one is refused.
    """
    value = _read(anchor, asset)
    for depth, _ in enumerate(path):
        if isinstance(value, list):
            return tuple(
                _scalar(_dig(row, path[depth:], asset, path), asset, path) for row in value
            )
        value = _dig(value, path[depth : depth + 1], asset, path)
    if not isinstance(value, list):
        raise ConformanceError(
            f"{asset}:{'.'.join(path) or '(document)'} is a {type(value).__name__}, not the "
            "list of values an interface reads",
            code="data_asset",
        )
    return tuple(_scalar(row, asset, path) for row in value)


def _read(anchor: str, asset: str) -> Any:
    base = Path(anchor).resolve().parent
    target = (base / asset).resolve()
    if Path(asset).is_absolute() or not target.is_relative_to(base):
        raise ConformanceError(
            f"data asset {asset!r} resolves outside {base}: a package reads only data that "
            "ships beside the module naming it",
            code="data_asset",
        )
    try:
        return canonical_json.decode(target.read_bytes())
    except OSError as exc:
        raise ConformanceError(
            f"data asset {asset!r} is unreadable: {exc}", code="data_asset"
        ) from exc
    except ValueError as exc:
        raise ConformanceError(
            f"data asset {asset!r} is not readable JSON: {exc}", code="data_asset"
        ) from exc


def _dig(value: Any, keys: Sequence[str], asset: str, path: Sequence[str]) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ConformanceError(
                f"data asset {asset!r} has no {'.'.join(path)}", code="data_asset"
            )
        value = value[key]
    return value


def _scalar(value: Any, asset: str, path: Sequence[str]) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise ConformanceError(
        f"data asset {asset!r} lists a {type(value).__name__} at {'.'.join(path)}; only JSON "
        "scalars are interface content",
        code="data_asset",
    )
