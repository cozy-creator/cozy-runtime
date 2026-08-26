"""EndpointDescriptor and JobDescriptor — the SCHEMA HOME (cr-003).

What the platform emits by EXECUTION, never hand-authored as a second document (§1.0): the
source-stable surface of one release. Callable names and schemas, signature-derived model
binding paths with each role class's admissibility stamps, the AuthorCapabilitySet derived
from each signature, per-public-method ComponentUseContracts, the config and secret schemas,
the protocol range, the Context/service facts endpoints rely on, and the surface digest.

Three things are deliberately NOT here, each a red arm:

* No candidate code topology. A release is built before any deployment binding exists, and
  an open slot binds checkpoints that did not exist at build. Destinations, ties, buffers,
  component membership and cache declarations belong to the ModelConstructionContract,
  derived per exact (release, factory, artifact snapshot, configs, variant) — cr-004.
* No fixed-point content: the enclosing release digest, a build timestamp, a machine path,
  or any mutable-binding/candidate-selected fact. `_fixed_point` proves it over the built
  document rather than trusting the builders above it.
* No execution spelling. `def` and `async def` describe IDENTICALLY: the spelling picks
  where the body runs (§1.2), and a caller cannot observe it.

Identity is `surface_digest` = sha256 over the canonical bytes of the body — every key
except `surface_digest` itself, which is the one self-reference a document cannot contain.
The committed `endpoint.descriptor.json` is a pretty PROJECTION of that body: a checked
cache, never authority.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime import PYTHON_CONTRACT_VERSION, WIRE_PROTOCOL_PACKAGE
from cozy_runtime.author import ConformanceError, Context, ModelBinding, Surface
from cozy_runtime.author._demand import feature_specs
from cozy_runtime.author._errors import NEUTRAL_ERROR_SET
from cozy_runtime.author._services import SERVICES
from cozy_runtime.internal import canonical, schema
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.protocol import WIRE_MINOR

SCHEMA = "cozy.endpoint.descriptor/1"
FILE = "endpoint.descriptor.json"
DIGEST_KEY = "surface_digest"

#: The complete top-level field set. A reader admits EXACTLY these — never a superset.
TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "application",
        "protocol",
        "context_facts",
        "services",
        "error_model",
        "config_schema",
        "secret_schema",
        "entrypoints",
        "jobs",
        DIGEST_KEY,
    }
)


class StaleDescriptor(ConformanceError):
    """The committed descriptor is not what this release derives. Exit 13."""

    default_code = "stale_descriptor"


# ------------------------------------------------------------------------ the document


def build(found: Discovered) -> dict[str, Json]:
    """The descriptor BODY for one discovered release. Pure function of the frozen app."""
    settings: dict[str, Json] = {}
    secrets: dict[str, Json] = {}
    entrypoints: list[Json] = []
    jobs: list[Json] = []
    for surface in found.surfaces:
        doc = _callable(surface, settings, secrets)
        (jobs if surface.kind == "job" else entrypoints).append(doc)
    body: dict[str, Json] = {
        "schema": SCHEMA,
        "application": found.application,
        "protocol": _protocol(),
        "context_facts": _context_facts(),
        "services": _services(),
        "error_model": dict(NEUTRAL_ERROR_SET),
        "config_schema": settings,
        "secret_schema": secrets,
        "entrypoints": sorted(entrypoints, key=_by_name),
        "jobs": sorted(jobs, key=_by_name),
    }
    # The declared field set and the built one are the same fact, proven here rather than
    # kept in step by review.
    assert set(body) == TOP_LEVEL_KEYS - {DIGEST_KEY}, sorted(set(body) ^ TOP_LEVEL_KEYS)
    _fixed_point(body, _binding_values(found.bindings))
    return body


def _by_name(doc: Json) -> str:
    assert isinstance(doc, dict)
    name = doc["name"]
    assert isinstance(name, str)
    return name


def _callable(
    surface: Surface, settings: dict[str, Json], secrets: dict[str, Json]
) -> dict[str, Json]:
    doc: dict[str, Json] = {
        "name": surface.name,
        "kind": surface.kind,
        # The AuthorCapabilitySet: a pure function of the parameter list. It names powers
        # granted to AUTHOR code — journaling, fetching and mandatory telemetry stay
        # runtime-owned policy and are deliberately not here (§1.3).
        "capabilities": sorted(surface.capabilities),
        "gpu": surface.gpu,
        "request": schema.render(surface.payload_type),
        "result": schema.render(surface.result_type),
    }
    models = [_model(binding) for binding in surface.model_bindings]
    if models:
        doc["models"] = models
    features = [
        {"axis": spec.axis, "field": spec.field, "source": spec.source}
        for spec in feature_specs(surface.payload_type)
    ]
    if features:
        doc["request_features"] = sorted(features, key=lambda f: (f["axis"], f["field"]))
    if surface.kind == "job":
        # WHAT THE BODY MAY WRITE, as its own declaration. Grants mint off these two lines
        # and never off the kind (§2): a job that declares neither is granted no destination.
        doc["publishes"] = surface.publishes
        doc["emits_media"] = surface.emits_media
    if surface.preflight is not None:
        doc["preflight"] = {
            "identity": surface.preflight.identity,
            "facts": schema.render(surface.preflight.result_type),
        }
    for param in surface.params:
        if param.capability == "settings" and param.arg is not None:
            doc["settings"] = _register(settings, param.arg)
        elif param.capability == "secrets" and param.arg is not None:
            doc["secrets"] = _register(secrets, param.arg)
    if surface.resources is not None:
        doc["resources"] = _resources(surface)
    if surface.demand_bound is not None:
        bound = surface.demand_bound
        declared: dict[str, Json] = {"provenance": bound.provenance, "reason": bound.reason}
        if bound.vram_bytes is not None:
            declared["vram_bytes"] = bound.vram_bytes
        if bound.host_ram_bytes is not None:
            declared["host_ram_bytes"] = bound.host_ram_bytes
        doc["demand"] = declared
    return doc


def _register(into: dict[str, Json], declared: object) -> str:
    """Typed deployment schemas live ONCE at the root; a callable references one by name."""
    rendered = schema.render(declared)
    assert isinstance(rendered, dict)
    name = str(rendered.get("struct", getattr(declared, "__name__", "schema")))
    existing = into.get(name)
    if existing is not None and existing != rendered:
        raise ConformanceError(
            f"two different declared schemas are both named {name!r}: the deployment config "
            "plane keys on the name, so one name is one schema",
            code="schema_collision",
        )
    into[name] = rendered
    return name


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


def _resources(surface: Surface) -> dict[str, Json]:
    override = surface.resources
    assert override is not None
    doc: dict[str, Json] = {}
    if override.gpu_count is not None:
        doc["gpu_count"] = override.gpu_count
    if override.placement is not None:
        doc["placement"] = override.placement
    if override.requires is not None:
        doc["requires"] = override.requires
    return doc


def _protocol() -> dict[str, Json]:
    """The declared RANGE this release speaks. Scripts gate on capability tokens, never on
    version strings (cozy-runtime-cli.md) — this is what a coordinator checks admissibility
    against, not what anyone parses for features."""
    return {
        "wire_package": WIRE_PROTOCOL_PACKAGE,
        "wire_minor_min": 0,
        "wire_minor_max": WIRE_MINOR,
        "python_contract_min": PYTHON_CONTRACT_VERSION,
        "python_contract_max": PYTHON_CONTRACT_VERSION,
    }


def _context_facts() -> list[Json]:
    """The committed Context inventory, READ from the class rather than restated.

    Every fact here has a live conformance exercise (`internal/conformance.py`): the v1
    writer-drift lesson is that `ctx.boot_warmup`'s writer was deleted while the contract
    stood, silently stripping warmup arms fleet-wide.
    """
    facts: dict[str, Json] = {}
    for name in [*Context.__dataclass_fields__, *dir(Context)]:
        if name.startswith("_") or name in facts:
            continue
        member = inspect.getattr_static(Context, name, None)
        facts[name] = {"name": name, "kind": "method" if callable(member) else "attribute"}
    return sorted(facts.values(), key=_by_name)


def _services() -> dict[str, Json]:
    """capability -> the service type and the members author code may call."""
    out: dict[str, Json] = {}
    for capability, service in sorted(SERVICES.items()):
        members = sorted(
            name
            for name, value in vars(service).items()
            if not name.startswith("_") and callable(value)
        )
        out[capability] = {"service": service.__name__, "members": members}
    return out


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
    because `repo` and `release` are stored apart but named together everywhere else."""
    values: set[str] = set()
    for entry in bindings.values():
        for value in entry.values():
            if isinstance(value, str):
                values.add(value)
        repo, release = entry.get("repo"), entry.get("release")
        if isinstance(repo, str):
            values.add(f"{repo}@{release}" if isinstance(release, str) else repo)
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


def surface_digest(body: Mapping[str, Json]) -> str:
    """sha256 over the canonical bytes of everything EXCEPT the digest field itself."""
    return canonical.digest({k: v for k, v in body.items() if k != DIGEST_KEY})


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


def document(body: Mapping[str, Json]) -> dict[str, Json]:
    """The committed document: the body plus its own digest, the one derived line a
    reviewer reads to know the surface contract moved."""
    return {**body, DIGEST_KEY: surface_digest(body)}


def path_of(project: Path) -> Path:
    return project / FILE


def write(project: Path, doc: Mapping[str, Json]) -> Path:
    target = path_of(project)
    target.write_text(canonical.pretty(dict(doc)))
    return target


@dataclass(frozen=True, slots=True)
class Divergence:
    """Which pair disagreed, and on what."""

    left: str
    right: str
    detail: str


def compare(left: Mapping[str, Json], right: Mapping[str, Json], names: tuple[str, str]) -> None:
    """Refuse unless two descriptors are the same document, naming the pair and the field."""
    left_body = {k: v for k, v in left.items() if k != DIGEST_KEY}
    right_body = {k: v for k, v in right.items() if k != DIGEST_KEY}
    if canonical.write(left_body) == canonical.write(right_body):
        declared = left.get(DIGEST_KEY)
        derived = surface_digest(right_body)
        if declared is not None and declared != derived:
            raise StaleDescriptor(
                f"{names[0]} carries {DIGEST_KEY} {declared} but its own body digests to "
                f"{derived} — the committed digest line was edited without the body",
                code="digest_mismatch",
            )
        return
    detail = "; ".join(_differences(left_body, right_body)) or "the documents differ"
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


def read(project: Path) -> dict[str, Json]:
    """The committed descriptor, parsed. A missing one is a refusal, not an empty default."""
    target = path_of(project)
    if not target.is_file():
        raise StaleDescriptor(
            f"{FILE} is not committed in {project} — the endpoint is valid only once its "
            "descriptor is derived and committed (§1.0)",
            code="missing_descriptor",
        )
    try:
        parsed = canonical.parse(target.read_bytes())
    except canonical.CanonicalError as exc:
        raise StaleDescriptor(f"{FILE}: {exc}", code="malformed_descriptor") from exc
    if not isinstance(parsed, dict):
        raise StaleDescriptor(f"{FILE} is not a JSON object", code="malformed_descriptor")
    # EXACT field set, never a superset (v1's graph-document lesson: an unknown key read as
    # a legal older document is how a reader goes quiet instead of refusing).
    unknown = sorted(set(parsed) - TOP_LEVEL_KEYS)
    if unknown:
        raise StaleDescriptor(
            f"{FILE} carries unknown top-level field(s) {', '.join(unknown)}: the descriptor "
            f"schema is exactly {', '.join(sorted(TOP_LEVEL_KEYS))}",
            code="unknown_field",
        )
    declared = parsed.get(DIGEST_KEY)
    if not isinstance(declared, str) or not canonical.DIGEST_RE.match(declared):
        raise StaleDescriptor(
            f"{FILE}: {DIGEST_KEY} {declared!r} is not `sha256:<64 lowercase hex>` — one digest "
            "spelling, shared with tensorfs (tfs-013)",
            code="malformed_digest",
        )
    return parsed


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
