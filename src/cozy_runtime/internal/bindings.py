"""Binding resolution: `package.toml` defaults -> `--model` -> the deploy binding.

Code states CAPABILITY; selection is a BINDING (§1.0/§1.1). Author code never names a model,
release, checkpoint or adapter — so this module lives outside the author fence, reads the
package repository's `package.toml`, and speaks the same one binding vocabulary the deploy
side does.

`package.toml` is a DEFAULT, never a source of truth: it governs only when nothing else
speaks (a bare-venv `run` with no `--model`). Production authority is the hub-side binding
alone. Two spellings, one schema: `[bindings."<ModelClass>"]` covers every parameter of that
class, `[bindings."<callable>.models.<param>"]` names one parameter specifically, and the
path spelling wins where both speak.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cozy_runtime.author import ConformanceError


class BindingSlot(Protocol):
    @property
    def path(self) -> str: ...

    @property
    def param(self) -> str: ...

    @property
    def class_key(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Binding:
    """One resolved slot: which artifact a binding path loads, and who said so."""

    path: str
    ref: str
    lane: str | None
    source: str
    """Where it came from, most authoritative last."""

    def __str__(self) -> str:
        lane = f" --lane={self.lane}" if self.lane else ""
        return f"{self.path} -> {self.ref}{lane}  [{self.source}]"


def read_table(package_toml: Path) -> Mapping[str, Mapping[str, object]]:
    """The package repository's unresolved `[bindings.*]` table."""
    data = tomllib.loads(package_toml.read_text())
    table = data.get("bindings", {})
    if not isinstance(table, dict):
        raise ConformanceError(
            f"{package_toml}: [bindings] must be a table of binding paths", code="bindings"
        )
    return {str(k): dict(v) for k, v in table.items() if isinstance(v, dict)}


def resolve(
    slots: Sequence[BindingSlot],
    table: Mapping[str, Mapping[str, object]],
    *,
    overrides: Mapping[str, str] | None = None,
    deploy: Mapping[str, str] | None = None,
) -> dict[str, Binding]:
    """Resolve every declared slot, or refuse naming the slot nothing binds."""
    ladder = ((overrides or {}, "--model"), (deploy or {}, "deploy binding"))
    known = {s.path for s in slots} | {s.class_key for s in slots}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ConformanceError(
            f"package.toml binds {', '.join(unknown)}, which this release declares no slot "
            f"for (slots: {', '.join(sorted(s.path for s in slots))})",
            code="unknown_binding",
            fields=unknown,
        )
    resolved: dict[str, Binding] = {}
    for slot in slots:
        chosen: Binding | None = None
        for key, source in (
            (slot.class_key, "package.toml:class"),
            (slot.path, "package.toml:path"),
        ):
            entry = table.get(key)
            if entry is not None:
                chosen = Binding(slot.path, _ref(entry, key), _lane(entry), source)
        for override, source in ladder:
            ref = override.get(slot.path) or override.get(slot.class_key)
            if ref is not None:
                chosen = Binding(slot.path, ref, chosen.lane if chosen else None, source)
        if chosen is None:
            raise ConformanceError(
                f"{slot.path} ({slot.class_key}) has no binding: code states capability, "
                "bindings state selection — name it in package.toml or pass --model",
                code="unbound_slot",
                fields=[slot.path],
            )
        resolved[slot.path] = chosen
    return resolved


def _ref(entry: Mapping[str, object], key: str) -> str:
    unknown = sorted(set(entry) - {"model", "release", "lane"})
    if unknown:
        raise ConformanceError(
            f"[bindings.{key!r}] has unknown fields: {', '.join(unknown)}",
            code="bindings",
            fields=[key, *unknown],
        )
    model = entry.get("model")
    if not isinstance(model, str) or not model:
        raise ConformanceError(
            f"[bindings.{key!r}] declares no model=", code="bindings", fields=[key]
        )
    release = entry.get("release")
    return f"{model}@{release}" if isinstance(release, str) and release else model


def _lane(entry: Mapping[str, object]) -> str | None:
    lane = entry.get("lane")
    return lane if isinstance(lane, str) else None
