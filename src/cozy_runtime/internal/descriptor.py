"""The package descriptor generated from one frozen application.

Its one identity is SHA-256 over canonical JSON. Creator obtains those exact bytes from
`cozy-runtime --json describe`; nothing is written into the package repository.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import NoReturn, cast

from cozy_runtime.author import ConformanceError, ModelBinding, Surface
from cozy_runtime.author._markers import Resources
from cozy_runtime.author._production import Production, ProductionContract, ProductionStep
from cozy_runtime.internal import canonical, schema
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.discovery import Discovered

SCHEMA = "cozy.package.descriptor/1"
DIGEST_KEY = "package_descriptor_digest"

#: The complete top-level field set. A reader admits EXACTLY these — never a superset.
TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "application",
        "entrypoints",
        "jobs",
        "model_productions",
    }
)


class StalePackageDescriptor(ConformanceError):
    """Supplied descriptor bytes are malformed or differ from the installed release."""

    default_code = "stale_package_descriptor"


# ------------------------------------------------------------------------ the document


def build(found: Discovered) -> dict[str, Json]:
    """Derive one descriptor from the frozen application."""
    entrypoints: list[Json] = []
    jobs: list[Json] = []
    for surface in found.surfaces:
        if surface.hidden:
            continue
        doc = _callable(surface)
        (jobs if surface.kind == "job" else entrypoints).append(doc)
    body: dict[str, Json] = {
        "format": SCHEMA,
        "application": found.application,
        "entrypoints": sorted(entrypoints, key=_by_name),
        "jobs": sorted(jobs, key=_by_name),
        "model_productions": sorted(
            (_production(production) for production in found.app.productions()), key=_by_name
        ),
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
    models = [_model(binding) for binding in surface.model_bindings]
    if models:
        doc["models"] = models
    if surface.kind == "job":
        doc["publishes"] = surface.publishes
        if surface.artifact_outputs:
            doc["artifact_outputs"] = [
                {
                    "output_id": output.name,
                    "mime_type": "application/vnd.cozy.model-manifest",
                    "max_bytes": output.max_new_bytes,
                }
                for output in sorted(surface.artifact_outputs, key=lambda row: row.name)
            ]
        if surface.resources is not None:
            resources = _resources(surface.resources)
            if resources:
                doc["resources"] = resources
    return doc


def _model(binding: ModelBinding) -> dict[str, Json]:
    """One model slot. The PATH is the binding point, the CLASS carries admissibility, and
    the per-public-method component sets are the ComponentUseContracts. Which artifact ever
    satisfies it is a binding-clock fact and is not here."""
    doc: dict[str, Json] = {
        "path": binding.path,
        "class": binding.class_key,
        "stamps": dict(binding.stamps),
        # Declaration order encodes no control flow and no residency choice (§1.1), so the
        # contract sorts — a reordered method body is not a surface change.
        "component_use": {
            method: sorted(names) for method, names in sorted(binding.components.items())
        },
    }
    return doc


def _resources(resources: Resources) -> dict[str, Json]:
    out: dict[str, Json] = {}
    if resources.gpu_count is not None:
        out["gpu_count"] = resources.gpu_count
    if resources.placement is not None:
        out["placement"] = resources.placement
    if resources.requires is not None:
        out["requires"] = resources.requires
    return out


def _production(production: Production) -> dict[str, Json]:
    return {
        "name": production.name,
        "sources": dict(sorted(production.sources.items())),
        "steps": [
            _production_step(step) for step in sorted(production.steps, key=lambda row: row.name)
        ],
        "outputs": [
            {
                "name": output.name,
                "source": output.source,
                "lane_key": output.lane_key,
                "required_contract": _production_contract(output.required),
            }
            for output in sorted(production.outputs, key=lambda row: row.name)
        ],
    }


def _production_step(step: ProductionStep) -> dict[str, Json]:
    doc: dict[str, Json] = {
        "name": step.name,
        "callable": step.callable,
        "models": dict(sorted(step.models.items())),
        "outputs": sorted(step.outputs),
    }
    if step.resources is not None:
        doc["resources"] = _resources(step.resources)
    return doc


def _production_contract(contract: ProductionContract) -> dict[str, Json]:
    return {
        "topology_digest": contract.topology_digest,
        "encodings": sorted(contract.encodings),
    }


# -------------------------------------------------------------------- fixed-point scan

_MACHINE_PATH = re.compile(r"(^/|^[A-Za-z]:[\\/]|/home/|\\Users\\|site-packages|\.venv/)")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_DIGEST = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")

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
    }
)


def _binding_values(bindings: Mapping[str, Mapping[str, object]]) -> frozenset[str]:
    """Every spelling of a binding a package.toml states — the parts AND the composed ref,
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
        # A production output declares the exact byte-level result the package promises to
        # derive. Those two digest-shaped values are SOURCE facts, unlike a selected release,
        # model, build, or candidate. The contract's closed parser below is the only place
        # this exception exists and validates both values as exact sha256 identities.
        if path.endswith(".required_contract.topology_digest") or re.search(
            r"\.required_contract\.encodings\[\d+\]$", path
        ):
            return
        if value in bindings:
            raise ConformanceError(
                f"{path}: {value!r} is a package.toml binding value — bindings are MUTABLE "
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


def canonical_bytes(doc: Mapping[str, Json]) -> bytes:
    """The semantic descriptor bytes used for identity and control-plane transport."""
    return canonical.write(dict(doc))


def package_descriptor_digest(doc: Mapping[str, Json]) -> str:
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
            encoded = canonical.write(entry)
            return (
                "sha256:" + hashlib.sha256(b"cozy.runtime.job-descriptor\0" + encoded).hexdigest()
            )
    raise ConformanceError(
        f"this release registers no job named {name!r}: it has "
        f"{', '.join(str(j['name']) for j in jobs if isinstance(j, dict)) or 'no jobs'}",
        code="unknown_job",
    )


def compare(left: Mapping[str, Json], right: Mapping[str, Json], names: tuple[str, str]) -> None:
    """Refuse unless two descriptors have the same content."""
    if canonical_bytes(left) == canonical_bytes(right):
        return
    detail = "; ".join(_differences(left, right)) or "the documents differ"
    raise StalePackageDescriptor(
        f"{names[0]} and {names[1]} are different documents: {detail}",
        code="stale_package_descriptor",
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


def read_bytes(raw: bytes, source: str = "package descriptor") -> dict[str, Json]:
    """Read one descriptor projection without making its formatting part of identity."""
    try:
        parsed = canonical.parse(raw)
    except canonical.CanonicalError as exc:
        raise StalePackageDescriptor(
            f"{source}: {exc}", code="malformed_package_descriptor"
        ) from exc
    if not isinstance(parsed, dict):
        raise StalePackageDescriptor(
            f"{source} is not a JSON object", code="malformed_package_descriptor"
        )
    fields = set(parsed)
    if fields != TOP_LEVEL_KEYS:
        missing = sorted(TOP_LEVEL_KEYS - fields)
        unknown = sorted(fields - TOP_LEVEL_KEYS)
        problems = []
        if missing:
            problems.append(f"missing {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown {', '.join(unknown)}")
        raise StalePackageDescriptor(
            f"{source} has {'; '.join(problems)}: the descriptor fields are exactly "
            f"{', '.join(sorted(TOP_LEVEL_KEYS))}",
            code="unknown_field",
        )
    if parsed["format"] != SCHEMA:
        raise StalePackageDescriptor(
            f"{source}: format {parsed['format']!r} is not {SCHEMA!r}",
            code="unknown_format",
        )
    _validate(parsed, source)
    return parsed


def _validate(body: Mapping[str, Json], source: str) -> None:
    """Read the one current nested grammar. This is a hardcut, not a legacy decoder."""
    application = body["application"]
    if not isinstance(application, str) or not application:
        _invalid(source, "application", "must be a non-empty string")
    _productions(body["model_productions"], source)
    for collection, kind in (("entrypoints", "entrypoint"), ("jobs", "job")):
        rows = body[collection]
        if not isinstance(rows, list):
            _invalid(source, collection, "must be an array")
        names: set[str] = set()
        for index, row in enumerate(rows):
            path = f"{collection}[{index}]"
            required = {"name", "request", "result"}
            optional = {"models"}
            if kind == "job":
                required.add("publishes")
                optional.update(("resources", "artifact_outputs"))
            item = _closed(row, required, optional, source, path)
            name = item["name"]
            if not isinstance(name, str) or not name or name in names:
                _invalid(source, f"{path}.name", "must be non-empty and unique")
            names.add(name)
            _schema(item["request"], source, f"{path}.request")
            _schema(item["result"], source, f"{path}.result")
            models = item.get("models", [])
            if not isinstance(models, list) or len(models) > 16:
                _invalid(source, f"{path}.models", "must be an array of at most 16 slots")
            for model_index, model in enumerate(models):
                model_path = f"{path}.models[{model_index}]"
                slot = _closed(
                    model,
                    {"class", "component_use", "path", "stamps"},
                    set(),
                    source,
                    model_path,
                )
                prefix = f"{name}.models."
                if (
                    not isinstance(slot["path"], str)
                    or not slot["path"].startswith(prefix)
                    or not slot["path"][len(prefix) :]
                ):
                    _invalid(source, f"{model_path}.path", f"must start with {prefix!r}")
                if not isinstance(slot["class"], str) or not slot["class"]:
                    _invalid(source, f"{model_path}.class", "must be a non-empty string")
                _string_map(slot["stamps"], source, f"{model_path}.stamps")
                uses = slot["component_use"]
                if not isinstance(uses, dict):
                    _invalid(source, f"{model_path}.component_use", "must be an object")
                for method, components in uses.items():
                    if (
                        not isinstance(method, str)
                        or not isinstance(components, list)
                        or not all(isinstance(component, str) for component in components)
                    ):
                        _invalid(
                            source,
                            f"{model_path}.component_use",
                            "must map method names to component-name arrays",
                        )
            if kind == "job":
                if not isinstance(item["publishes"], bool):
                    _invalid(source, f"{path}.publishes", "must be a boolean")
                if "resources" in item:
                    _production_resources(item["resources"], source, f"{path}.resources")
                if "artifact_outputs" in item:
                    artifact_outputs = item["artifact_outputs"]
                    if (
                        not isinstance(artifact_outputs, list)
                        or not 1 <= len(artifact_outputs) <= 16
                    ):
                        _invalid(
                            source,
                            f"{path}.artifact_outputs",
                            "must contain 1 through 16 outputs",
                        )
                    artifact_names: set[str] = set()
                    for output_index, output_raw in enumerate(artifact_outputs):
                        output_path = f"{path}.artifact_outputs[{output_index}]"
                        output = _closed(
                            output_raw,
                            {"output_id", "mime_type", "max_bytes"},
                            set(),
                            source,
                            output_path,
                        )
                        output_id = _portable_name(
                            output["output_id"], source, f"{output_path}.output_id"
                        )
                        if output_id in artifact_names:
                            _invalid(source, f"{output_path}.output_id", "must be unique")
                        artifact_names.add(output_id)
                        if output["mime_type"] != "application/vnd.cozy.model-manifest":
                            _invalid(source, f"{output_path}.mime_type", "is not the model MIME")
                        maximum = output["max_bytes"]
                        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
                            _invalid(source, f"{output_path}.max_bytes", "must be non-negative")


def _productions(value: Json, source: str) -> None:
    if not isinstance(value, list) or len(value) > 16:
        _invalid(source, "model_productions", "must be an array of at most 16 entries")
    names: set[str] = set()
    for index, raw in enumerate(value):
        path = f"model_productions[{index}]"
        production = _closed(raw, {"name", "sources", "steps", "outputs"}, set(), source, path)
        name = _portable_name(production["name"], source, f"{path}.name")
        if name in names:
            _invalid(source, f"{path}.name", "must be unique")
        names.add(name)
        roots = _string_map_value(production["sources"], source, f"{path}.sources", 16)
        if not roots:
            _invalid(source, f"{path}.sources", "must contain at least one source slot")
        for slot, profile in roots.items():
            _contract_value(profile, source, f"{path}.sources.{slot}")
        steps = production["steps"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
            _invalid(source, f"{path}.steps", "must contain 1 through 64 steps")
        step_outputs: dict[str, set[str]] = {}
        dependencies: dict[str, set[str]] = {}
        for step_index, step_raw in enumerate(steps):
            step_path = f"{path}.steps[{step_index}]"
            step = _closed(
                step_raw,
                {"name", "callable", "models", "outputs"},
                {"resources"},
                source,
                step_path,
            )
            step_name = _portable_name(step["name"], source, f"{step_path}.name")
            if step_name in step_outputs:
                _invalid(source, f"{step_path}.name", "must be unique")
            callable_name = step["callable"]
            if not isinstance(callable_name, str) or _CALLABLE.fullmatch(callable_name) is None:
                _invalid(source, f"{step_path}.callable", "must be org/package/function")
            outputs = _portable_names(step["outputs"], source, f"{step_path}.outputs", 8)
            step_outputs[step_name] = set(outputs)
            dependencies[step_name] = set()
            models = _string_map_value(step["models"], source, f"{step_path}.models", 16)
            if not models:
                _invalid(source, f"{step_path}.models", "must not be empty")
            if "resources" in step:
                _production_resources(step["resources"], source, f"{step_path}.resources")

        def edge(
            reference: str,
            edge_path: str,
            root_names: frozenset[str] = frozenset(roots),
            outputs_by_step: dict[str, set[str]] = step_outputs,
        ) -> str | None:
            if reference in root_names:
                return None
            step_name, separator, output = reference.partition(".")
            if not separator or output not in outputs_by_step.get(step_name, set()):
                _invalid(source, edge_path, f"references unknown edge {reference!r}")
            return step_name

        for step_index, step_raw in enumerate(steps):
            assert isinstance(step_raw, dict)
            step_name = str(step_raw["name"])
            models = step_raw["models"]
            assert isinstance(models, dict)
            for model_name, reference in models.items():
                dependency = edge(str(reference), f"{path}.steps[{step_index}].models.{model_name}")
                if dependency is not None:
                    dependencies[step_name].add(dependency)

        outputs = production["outputs"]
        if not isinstance(outputs, list) or not 1 <= len(outputs) <= 16:
            _invalid(source, f"{path}.outputs", "must contain 1 through 16 required outputs")
        output_names: set[str] = set()
        lane_keys: set[str] = set()
        required_steps: set[str] = set()

        def retain(
            step_name: str,
            visiting: set[str],
            production_path: str = path,
            graph: dict[str, set[str]] = dependencies,
            retained: set[str] = required_steps,
        ) -> None:
            if step_name in visiting:
                _invalid(source, production_path, f"contains a cycle through {step_name}")
            if step_name in retained:
                return
            visiting.add(step_name)
            for dependency in graph[step_name]:
                retain(dependency, visiting)
            visiting.remove(step_name)
            retained.add(step_name)

        for output_index, output_raw in enumerate(outputs):
            output_path = f"{path}.outputs[{output_index}]"
            output = _closed(
                output_raw,
                {"name", "source", "lane_key", "required_contract"},
                set(),
                source,
                output_path,
            )
            output_name = _portable_name(output["name"], source, f"{output_path}.name")
            lane_key = _portable_name(output["lane_key"], source, f"{output_path}.lane_key")
            if output_name in output_names or lane_key in lane_keys:
                _invalid(source, output_path, "repeats an output name or lane key")
            output_names.add(output_name)
            lane_keys.add(lane_key)
            selected = edge(str(output["source"]), f"{output_path}.source")
            if selected is None:
                _invalid(source, f"{output_path}.source", "must name a step output")
            retain(selected, set())
            _production_contract_value(
                output["required_contract"], source, f"{output_path}.required_contract"
            )
        disconnected = sorted(set(step_outputs) - required_steps)
        if disconnected:
            _invalid(source, path, f"contains disconnected steps {disconnected}")


_PRODUCTION_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_CONTRACT_VALUE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}(?:/[a-z0-9][a-z0-9._+-]{0,63})*$")
_CALLABLE = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,63}/"
    r"[a-z][a-z0-9_-]{0,63}$"
)


def _portable_name(value: Json, source: str, path: str) -> str:
    if not isinstance(value, str) or _PRODUCTION_NAME.fullmatch(value) is None:
        _invalid(source, path, "must be a portable lowercase name")
    return value


def _portable_names(value: Json, source: str, path: str, cap: int) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= cap:
        _invalid(source, path, f"must contain 1 through {cap} names")
    out = [_portable_name(item, source, path) for item in value]
    if len(set(out)) != len(out):
        _invalid(source, path, "must be unique")
    return out


def _contract_value(value: Json, source: str, path: str) -> str:
    if not isinstance(value, str) or _CONTRACT_VALUE.fullmatch(value) is None:
        _invalid(source, path, "is not a bounded contract token")
    return value


def _string_map_value(value: Json, source: str, path: str, cap: int) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > cap:
        _invalid(source, path, f"must be an object with at most {cap} entries")
    out: dict[str, str] = {}
    for key, item in value.items():
        _portable_name(key, source, path)
        if not isinstance(item, str) or not item:
            _invalid(source, f"{path}.{key}", "must be a non-empty string")
        out[key] = item
    return out


def _production_resources(value: Json, source: str, path: str) -> None:
    resources = _closed(value, set(), {"gpu_count", "placement", "requires"}, source, path)
    if not resources:
        _invalid(source, path, "must not be empty")
    gpu_count = resources.get("gpu_count")
    if gpu_count is not None and (
        not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 1
    ):
        _invalid(source, f"{path}.gpu_count", "must be positive")
    if resources.get("placement") not in (None, "single_node", "any"):
        _invalid(source, f"{path}.placement", "must be single_node or any")
    requires = resources.get("requires")
    if requires is not None and (not isinstance(requires, str) or not requires):
        _invalid(source, f"{path}.requires", "must be a non-empty string")


def _production_contract_value(value: Json, source: str, path: str) -> None:
    contract = _closed(value, {"topology_digest", "encodings"}, set(), source, path)
    topology = contract["topology_digest"]
    encodings = contract["encodings"]
    if not isinstance(topology, str) or _SHA256.fullmatch(topology) is None:
        _invalid(source, f"{path}.topology_digest", "must be an exact sha256 digest")
    if (
        not isinstance(encodings, list)
        or not 1 <= len(encodings) <= 32
        or encodings != sorted(set(encodings))
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in encodings
        )
    ):
        _invalid(source, f"{path}.encodings", "must be sorted unique exact sha256 digests")


def _schema(value: Json, source: str, path: str, union_tag_field: str | None = None) -> None:
    if isinstance(value, str):
        if value not in _SCALARS:
            _invalid(source, path, f"unknown scalar {value!r}")
        return
    if not isinstance(value, dict):
        _invalid(source, path, "must be a scalar or schema object")
    forms = set(value) & _SCHEMA_FORMS
    if len(forms) != 1:
        _invalid(source, path, "must carry exactly one schema form")
    form = next(iter(forms))
    if form == "fields":
        optional = {"tag", "tag_field"}
        struct = _closed(value, {"fields"}, optional, source, path)
        tag = struct.get("tag")
        tag_field = struct.get("tag_field")
        active_tag_field: Json
        if union_tag_field is not None:
            if tag is None or tag_field is not None:
                _invalid(source, path, "a tagged-union member carries only its tag")
            active_tag_field = union_tag_field
        elif (tag is None) != (tag_field is None):
            _invalid(source, path, "a standalone tag and tag_field must appear together")
        else:
            active_tag_field = tag_field
        if active_tag_field is not None and (
            not isinstance(active_tag_field, str) or not active_tag_field
        ):
            _invalid(source, f"{path}.tag_field", "must be a non-empty string")
        fields = struct["fields"]
        if not isinstance(fields, list):
            _invalid(source, f"{path}.fields", "must be an array")
        names: set[str] = set()
        for index, value_field in enumerate(fields):
            field_path = f"{path}.fields[{index}]"
            field = _closed(
                value_field,
                {"name", "type"},
                {"asset_bound", "constraints", "wire"},
                source,
                field_path,
            )
            name = field["name"]
            if not isinstance(name, str) or not name or name in names or name == active_tag_field:
                _invalid(
                    source,
                    f"{field_path}.name",
                    "must be non-empty, unique, and not repeat the tag field",
                )
            names.add(name)
            wire = field.get("wire")
            if wire is not None and wire not in {"optional", "omissible"}:
                _invalid(
                    source,
                    f"{field_path}.wire",
                    "may only be optional or omissible; required is the default",
                )
            _schema(field["type"], source, f"{field_path}.type")
            if "constraints" in field:
                _subset_object(
                    field["constraints"], _CONSTRAINT_KEYS, source, f"{field_path}.constraints"
                )
            if "asset_bound" in field:
                _subset_object(
                    field["asset_bound"], _ASSET_BOUND_KEYS, source, f"{field_path}.asset_bound"
                )
        return
    if form == "union":
        union = _closed(value, {"union"}, {"tag_field"}, source, path)
        branches = union["union"]
        if not isinstance(branches, list) or not branches:
            _invalid(source, f"{path}.union", "must be a non-empty array")
        tag_field = union.get("tag_field")
        if tag_field is not None and (not isinstance(tag_field, str) or not tag_field):
            _invalid(source, f"{path}.tag_field", "must be a non-empty string")
        for index, branch in enumerate(branches):
            _schema(branch, source, f"{path}.union[{index}]", tag_field)
        return
    exact = _closed(value, {form}, set(), source, path)
    child = exact[form]
    if form == "list":
        _schema(child, source, f"{path}.list")
    elif form == "literal":
        if not isinstance(child, list):
            _invalid(source, f"{path}.literal", "must be an array")
    elif not isinstance(child, str) or not child:
        _invalid(source, f"{path}.{form}", "must be a non-empty string")


def _closed(
    value: Json,
    required: set[str],
    optional: set[str],
    source: str,
    path: str,
) -> dict[str, Json]:
    if not isinstance(value, dict):
        _invalid(source, path, "must be an object")
    fields = set(value)
    if fields != required | (fields & optional):
        missing = sorted(required - fields)
        unknown = sorted(fields - required - optional)
        detail = ", ".join(
            [*(f"missing {name}" for name in missing), *(f"unknown {name}" for name in unknown)]
        )
        _invalid(source, path, detail)
    return cast(dict[str, Json], value)


def _string_map(value: Json, source: str, path: str) -> None:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        _invalid(source, path, "must map strings to strings")


def _subset_object(value: Json, allowed: set[str], source: str, path: str) -> None:
    if not isinstance(value, dict) or not value or not set(value) <= allowed:
        _invalid(source, path, f"must be a non-empty object using {sorted(allowed)}")


def _invalid(source: str, path: str, detail: str) -> NoReturn:
    raise StalePackageDescriptor(f"{source}: {path} {detail}", code="malformed_package_descriptor")


_SCALARS = frozenset({"bool", "float", "int", "null", "str"})
_SCHEMA_FORMS = frozenset({"asset", "fields", "input", "list", "literal", "union"})
_CONSTRAINT_KEYS = {"ge", "gt", "le", "max_length", "min_length"}
_ASSET_BOUND_KEYS = {"max_bytes", "media_types"}


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
    productions = body.get("model_productions", [])
    assert isinstance(productions, list)
    return {
        "entrypoints": len(entrypoints),
        "jobs": len(jobs),
        "model_bindings": slots,
        "model_productions": len(productions),
    }
