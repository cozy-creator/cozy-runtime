"""Semantic-version compatibility for two valid PackageDescriptors.

This is Runtime's decision because Runtime owns the descriptor grammar.  It is an
ordinary comparison result, not another stored document or release identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from cozy_runtime.internal import descriptor
from cozy_runtime.internal.canonical import Json

Compatibility = Literal["patch", "minor", "major"]
MAX_REASONS = 16


@dataclass(frozen=True, slots=True)
class Result:
    classification: Compatibility
    reasons: tuple[str, ...]


def classify_bytes(previous: bytes, candidate: bytes) -> Result:
    """Parse and classify two exact descriptor byte strings."""
    before = descriptor.read_bytes(previous, "previous package descriptor")
    after = descriptor.read_bytes(candidate, "candidate package descriptor")
    return _classify(before, after)


def _classify(previous: dict[str, Json], candidate: dict[str, Json]) -> Result:
    major: list[str] = []
    minor: list[str] = []

    if previous["application"] != candidate["application"]:
        major.append("application changed")

    before = _callables(previous, "previous package descriptor")
    after = _callables(candidate, "candidate package descriptor")
    for name in sorted(before):
        old_kind, old = before[name]
        current = after.get(name)
        if current is None:
            major.append(f"removed {old_kind} {name}")
            continue
        new_kind, new = current
        if old_kind != new_kind:
            major.append(f"callable {name} changed kind from {old_kind} to {new_kind}")
            continue

        for key, label in (
            ("result", "result schema"),
            ("publishes", "job publication metadata"),
            ("resources", "job resource metadata"),
        ):
            if old.get(key) != new.get(key):
                major.append(f"{old_kind} {name} changed {label}")
        if old.get("models", []) != new.get("models", []):
            major.append(f"{old_kind} {name} changed model metadata")

        additions, incompatible = _request_additions(
            old["request"], new["request"], f"{old_kind} {name} request"
        )
        if incompatible is not None:
            major.append(incompatible)
        else:
            minor.extend(additions)

    for name in sorted(set(after) - set(before)):
        kind, _ = after[name]
        minor.append(f"added {kind} {name}")

    if major:
        return Result("major", _bounded(major))
    if minor:
        return Result("minor", _bounded(minor))
    return Result("patch", ("public interface identical",))


def _callables(
    body: dict[str, Json], source: str
) -> dict[str, tuple[Literal["entrypoint", "job"], dict[str, Json]]]:
    found: dict[str, tuple[Literal["entrypoint", "job"], dict[str, Json]]] = {}
    for collection, kind in (("entrypoints", "entrypoint"), ("jobs", "job")):
        rows = cast(list[Json], body[collection])
        for value in rows:
            row = cast(dict[str, Json], value)
            name = cast(str, row["name"])
            if name in found:
                raise descriptor.StalePackageDescriptor(
                    f"{source}: callable name {name!r} is used by both an entrypoint and a job",
                    code="malformed_package_descriptor",
                )
            found[name] = (cast(Literal["entrypoint", "job"], kind), row)
    return found


def _request_additions(previous: Json, candidate: Json, path: str) -> tuple[list[str], str | None]:
    """Return added optional fields, or one exact reason the request is incompatible.

    Existing schema structure must remain identical.  The sole relaxation is adding a
    field marked ``optional`` or ``omissible`` to any existing request struct.
    """
    if previous == candidate:
        return [], None
    if not isinstance(previous, dict) or not isinstance(candidate, dict):
        return [], f"{path} schema changed"

    if "fields" in previous or "fields" in candidate:
        if "fields" not in previous or "fields" not in candidate:
            return [], f"{path} schema changed"
        old_meta = {key: value for key, value in previous.items() if key != "fields"}
        new_meta = {key: value for key, value in candidate.items() if key != "fields"}
        if old_meta != new_meta:
            return [], f"{path} schema changed"
        old_fields = cast(list[Json], previous["fields"])
        new_fields = cast(list[Json], candidate["fields"])
        old_by_name = {cast(str, cast(dict[str, Json], row)["name"]): row for row in old_fields}
        new_by_name = {cast(str, cast(dict[str, Json], row)["name"]): row for row in new_fields}
        old_names = list(old_by_name)
        retained_names = [name for name in new_by_name if name in old_by_name]
        if retained_names != old_names:
            return [], f"{path} removed or reordered an existing field"

        additions: list[str] = []
        for name in old_names:
            old_field = cast(dict[str, Json], old_by_name[name])
            new_value = new_by_name.get(name)
            if new_value is None:
                return [], f"{path} removed field {name}"
            new_field = cast(dict[str, Json], new_value)
            old_field_meta = {key: value for key, value in old_field.items() if key != "type"}
            new_field_meta = {key: value for key, value in new_field.items() if key != "type"}
            if old_field_meta != new_field_meta:
                return [], f"{path} changed field {name}"
            nested, incompatible = _request_additions(
                old_field["type"], new_field["type"], f"{path}.{name}"
            )
            if incompatible is not None:
                return [], incompatible
            additions.extend(nested)

        for name in new_by_name:
            if name in old_by_name:
                continue
            field = cast(dict[str, Json], new_by_name[name])
            wire = field.get("wire")
            if wire not in {"optional", "omissible"}:
                return [], f"{path} added required field {name}"
            additions.append(f"{path} added {wire} field {name}")
        return additions, None

    if set(previous) != set(candidate):
        return [], f"{path} schema changed"
    if "list" in previous:
        return _request_additions(previous["list"], candidate["list"], f"{path}[]")
    if "union" in previous:
        if {key: value for key, value in previous.items() if key != "union"} != {
            key: value for key, value in candidate.items() if key != "union"
        }:
            return [], f"{path} schema changed"
        old_branches = cast(list[Json], previous["union"])
        new_branches = cast(list[Json], candidate["union"])
        if len(old_branches) != len(new_branches):
            return [], f"{path} schema changed"
        union_additions: list[str] = []
        for index, (old, new) in enumerate(zip(old_branches, new_branches, strict=True)):
            nested, incompatible = _request_additions(old, new, f"{path}.union[{index}]")
            if incompatible is not None:
                return [], incompatible
            union_additions.extend(nested)
        return union_additions, None
    return [], f"{path} schema changed"


def _bounded(reasons: list[str]) -> tuple[str, ...]:
    if len(reasons) <= MAX_REASONS:
        return tuple(reasons)
    omitted = len(reasons) - MAX_REASONS + 1
    return (*reasons[: MAX_REASONS - 1], f"{omitted} more changes")
