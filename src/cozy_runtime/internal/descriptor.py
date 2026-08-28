"""The reviewed endpoint descriptor generated from one frozen application.

The committed ``endpoint.descriptor.json`` is a readable projection. Its one identity is
SHA-256 over the canonical JSON encoding of its content, so indentation, whitespace and key
order can never change ``descriptor_digest``. The digest is derived and is not embedded in
the document it identifies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from cozy_runtime.author import ConformanceError, ModelBinding, Surface
from cozy_runtime.internal import canonical, schema
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.discovery import Discovered

SCHEMA = "cozy.endpoint.descriptor/1"
FILE = "endpoint.descriptor.json"
DIGEST_KEY = "descriptor_digest"

#: The complete top-level field set. A reader admits EXACTLY these — never a superset.
TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "application",
        "entrypoints",
        "jobs",
    }
)


class StaleDescriptor(ConformanceError):
    """The committed descriptor is not what this release derives. Exit 13."""

    default_code = "stale_descriptor"


# ------------------------------------------------------------------------ the document


def build(found: Discovered) -> dict[str, Json]:
    """Derive one descriptor from the frozen application."""
    entrypoints: list[Json] = []
    jobs: list[Json] = []
    for surface in found.surfaces:
        doc = _callable(surface)
        (jobs if surface.kind == "job" else entrypoints).append(doc)
    body: dict[str, Json] = {
        "format": SCHEMA,
        "application": found.application,
        "entrypoints": sorted(entrypoints, key=_by_name),
        "jobs": sorted(jobs, key=_by_name),
    }
    assert set(body) == TOP_LEVEL_KEYS, sorted(set(body) ^ TOP_LEVEL_KEYS)
    _fixed_point(body, _binding_values(found.bindings))
    return body


def _by_name(doc: Json) -> str:
    assert isinstance(doc, dict)
    name = doc["name"]
    assert isinstance(name, str)
    return name


def _callable(surface: Surface) -> dict[str, Json]:
    doc: dict[str, Json] = {
        "name": surface.name,
        "request": schema.render(surface.payload_type),
        "result": schema.render(surface.result_type),
    }
    if surface.kind == "entrypoint":
        doc["hidden"] = surface.hidden
    models = [_model(binding) for binding in surface.model_bindings]
    if models:
        doc["models"] = models
    if surface.kind == "job":
        doc["publishes"] = surface.publishes
        if surface.resources is not None and surface.resources.gpu_count is not None:
            doc["resources"] = {"gpu_count": surface.resources.gpu_count}
    return doc


def _model(binding: ModelBinding) -> dict[str, Json]:
    """One model slot. The PATH is the binding point, the CLASS carries admissibility, and
    the per-public-method component sets are the ComponentUseContracts. Which artifact ever
    satisfies it is a binding-clock fact and is not here."""
    doc: dict[str, Json] = {
        "path": binding.path,
        "param": binding.param,
        "class": binding.class_key,
        "stamps": dict(binding.stamps),
        # Declaration order encodes no control flow and no residency choice (§1.1), so the
        # contract sorts — a reordered method body is not a surface change.
        "component_use": {
            method: sorted(names) for method, names in sorted(binding.components.items())
        },
    }
    return doc


# -------------------------------------------------------------------- fixed-point scan

_MACHINE_PATH = re.compile(r"(^/|^[A-Za-z]:[\\/]|/home/|\\Users\\|site-packages|\.venv/)")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_DIGEST = re.compile(r"^(sha256:)?[0-9a-f]{64}$")

#: Keys that can only carry a fact from a LATER clock: a binding, a candidate, a build.
_FORBIDDEN_KEYS = frozenset(
    {
        "release",
        "release_digest",
        "built_at",
        "build_id",
        "timestamp",
        "created_at",
        "generated_at",
        "model",
        "repo",
        "lane",
        "checkpoint",
        "checkpoint_ref",
        "artifact",
        "snapshot",
        "candidate",
        "destinations",
        "topology_digest",
    }
)


def _binding_values(bindings: Mapping[str, Mapping[str, object]]) -> frozenset[str]:
    """Every spelling of a binding an endpoint.toml states — the parts AND the composed ref,
    because `model` and `release` are stored apart but named together everywhere else."""
    values: set[str] = set()
    for entry in bindings.values():
        for value in entry.values():
            if isinstance(value, str):
                values.add(value)
        model, release = entry.get("model"), entry.get("release")
        if isinstance(model, str):
            values.add(f"{model}@{release}" if isinstance(release, str) else model)
    return frozenset(values)


def _fixed_point(value: Json, bindings: frozenset[str], path: str = "") -> None:
    """The descriptor's fixed point, proven over the built document (§1.0)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_KEYS:
                raise ConformanceError(
                    f"{path}.{key}: the descriptor is the SOURCE-STABLE surface — release "
                    "digests, timestamps, machine paths, mutable bindings and "
                    "candidate-selected facts live on other clocks (cr-004)",
                    code="fixed_point",
                    fields=[f"{path}.{key}"],
                )
            _fixed_point(item, bindings, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _fixed_point(item, bindings, f"{path}[{index}]")
    elif isinstance(value, str):
        if value in bindings:
            raise ConformanceError(
                f"{path}: {value!r} is an endpoint.toml binding value — bindings are MUTABLE "
                "and post-deploy authority is the hub's, so no binding can key a release",
                code="fixed_point",
                fields=[path],
            )
        for pattern, why in (
            (_MACHINE_PATH, "a machine path"),
            (_TIMESTAMP, "a build timestamp"),
            (_DIGEST, "a digest of something on another clock"),
        ):
            if pattern.search(value):
                raise ConformanceError(
                    f"{path}: {value!r} is {why}; the descriptor must reproduce byte-identically "
                    "in every environment of one release",
                    code="fixed_point",
                    fields=[path],
                )


# ------------------------------------------------------------------------ identity + file


def render(doc: Mapping[str, Json]) -> bytes:
    """The deterministic, readable repository projection."""
    return canonical.pretty(dict(doc)).encode("utf-8")


def canonical_bytes(doc: Mapping[str, Json]) -> bytes:
    """The semantic descriptor bytes used for identity and control-plane transport."""
    return canonical.write(dict(doc))


def descriptor_digest(doc: Mapping[str, Json]) -> str:
    """The descriptor's formatting-invariant semantic identity."""
    return canonical.digest(dict(doc))


def job_descriptor_id(body: Mapping[str, Json], name: str) -> str:
    """`job_descriptor_id` — sha256 of ONE job's exact callable, schemas, effects, resource
    caps and publication contract (README §4).

    DERIVED, never stored in the document: a digest of a subject that lives on this same
    clock is still a digest, and `_fixed_point` refuses one anywhere in the body — for the
    good reason that a reader who finds a `sha256:` string in a source-stable surface has no
    way to know which clock it came from. The runner computes it when it needs it, from the
    document, and every environment of one release computes the same one.
    """
    jobs = body.get("jobs", [])
    assert isinstance(jobs, list)
    for entry in jobs:
        assert isinstance(entry, dict)
        if entry["name"] == name:
            return canonical.digest({"format": "cozy.runtime.JobDescriptor/1", **entry})
    raise ConformanceError(
        f"this release registers no job named {name!r}: it has "
        f"{', '.join(str(j['name']) for j in jobs if isinstance(j, dict)) or 'no jobs'}",
        code="unknown_job",
    )


def path_of(project: Path) -> Path:
    return project / FILE


def write(project: Path, doc: Mapping[str, Json]) -> Path:
    target = path_of(project)
    target.write_bytes(render(doc))
    return target


def compare(left: Mapping[str, Json], right: Mapping[str, Json], names: tuple[str, str]) -> None:
    """Refuse unless two descriptors have the same content."""
    if canonical_bytes(left) == canonical_bytes(right):
        return
    detail = "; ".join(_differences(left, right)) or "the documents differ"
    raise StaleDescriptor(
        f"{names[0]} and {names[1]} are different documents: {detail}",
        code="stale_descriptor",
    )


def _differences(left: Json, right: Json, path: str = "") -> list[str]:
    """The first few disagreements, deepest key named — a reviewer needs the FIELD."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append(f"{path}.{key} only in the second")
            elif key not in right:
                out.append(f"{path}.{key} only in the first")
            else:
                out += _differences(left[key], right[key], f"{path}.{key}")
            if len(out) >= 4:
                break
        return out[:4]
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return [f"{path or 'root'}: {len(left)} vs {len(right)} entries"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out += _differences(a, b, f"{path}[{index}]")
            if len(out) >= 4:
                break
        return out[:4]
    if left != right:
        return [f"{path or 'root'}: {left!r} vs {right!r}"]
    return []


def read_bytes(raw: bytes, source: str = FILE) -> dict[str, Json]:
    """Read one descriptor projection without making its formatting part of identity."""
    try:
        parsed = canonical.parse(raw)
    except canonical.CanonicalError as exc:
        raise StaleDescriptor(f"{source}: {exc}", code="malformed_descriptor") from exc
    if not isinstance(parsed, dict):
        raise StaleDescriptor(f"{source} is not a JSON object", code="malformed_descriptor")
    fields = set(parsed)
    if fields != TOP_LEVEL_KEYS:
        missing = sorted(TOP_LEVEL_KEYS - fields)
        unknown = sorted(fields - TOP_LEVEL_KEYS)
        problems = []
        if missing:
            problems.append(f"missing {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown {', '.join(unknown)}")
        raise StaleDescriptor(
            f"{source} has {'; '.join(problems)}: the descriptor fields are exactly "
            f"{', '.join(sorted(TOP_LEVEL_KEYS))}",
            code="unknown_field",
        )
    if parsed["format"] != SCHEMA:
        raise StaleDescriptor(
            f"{source}: format {parsed['format']!r} is not {SCHEMA!r}",
            code="unknown_format",
        )
    return parsed


def read(project: Path) -> dict[str, Json]:
    """The committed descriptor, parsed. A missing one is a refusal, not an empty default."""
    target = path_of(project)
    if not target.is_file():
        raise StaleDescriptor(
            f"{FILE} is not committed in {project} — the endpoint is valid only once its "
            "descriptor is derived and committed (§1.0)",
            code="missing_descriptor",
        )
    return read_bytes(target.read_bytes(), str(target))


def counts(body: Mapping[str, Json]) -> dict[str, int]:
    """Aggregates ride the listing, so no follow-up call is needed."""
    entrypoints = body.get("entrypoints", [])
    jobs = body.get("jobs", [])
    assert isinstance(entrypoints, list) and isinstance(jobs, list)
    slots = 0
    for doc in [*entrypoints, *jobs]:
        assert isinstance(doc, dict)
        models = doc.get("models", [])
        assert isinstance(models, list)
        slots += len(models)
    return {"entrypoints": len(entrypoints), "jobs": len(jobs), "model_bindings": slots}
