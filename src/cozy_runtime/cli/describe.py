"""`cozy-runtime describe` — the PackageInterface from SOURCE, executing no package code.

Always read-only. Full `--json` output is the canonical publication input; there is no
committed interface projection and therefore no writer or staleness-check mode.

The default reader is `internal/static_interface.py`: `ast` over the package tree, a closed
vocabulary, typed refusals (decision #713 — package code is untrusted). Weights are
structurally unreachable: nothing here imports the package, let alone constructs a `Loader`.

`--conformance` is the locked-environment ORACLE: it additionally imports the app in THIS
interpreter — which must be the package's own locked venv — and refuses unless the two
derivations are byte-identical. That is what a package's own CI runs; publish runs the
static reader alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.exits import Exit


def run(function: str | None, opts: Options) -> Result:
    project = Path(opts.dir).expanduser()
    rows: tuple[tuple[str, str], ...] = ()
    try:
        if opts.builtin:
            body = static_interface.build_builtin(opts.builtin)
        elif opts.package_interface:
            interface = Path(opts.package_interface).expanduser()
            try:
                raw = interface.read_bytes()
            except OSError as exc:
                raise ConformanceError(
                    f"cannot read exact package interface {interface}: {exc}",
                    code="package_interface_unreadable",
                ) from exc
            body = package_interface.read_bytes(raw, str(interface))
        elif opts.distribution:
            body = static_interface.build_installed(
                opts.distribution, environment_python=Path(opts.environment_python)
            )
        else:
            body = static_interface.build(
                project,
                environment_python=Path(opts.environment_python)
                if opts.environment_python
                else None,
            )
            if opts.conformance:
                rows = (("conformance", _conformance(project, body)),)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "a package's surface must be valid before it can be described (§1.0)",
            Exit.validation,
        ) from exc
    if function is not None:
        return _one(body, function)
    listed = _list(body)
    return Result(
        rows + listed.rows, next=listed.next, document=body, canonical_json=listed.canonical_json
    )


def _conformance(project: Path, static: Mapping[str, Json]) -> str:
    """Import the app here and refuse unless it agrees with the source reading."""
    from cozy_runtime.internal.discovery import assert_locked_environment, discover

    environment = assert_locked_environment(project)
    imported = package_interface.source_facts(package_interface.build(discover(project)))
    package_interface.compare(static, imported, ("the source reading", "the imported app"))
    return f"equal: imported {static['application']} ({environment})"


def _list(body: Mapping[str, Json]) -> Result:
    rows: list[tuple[str, str]] = []
    for kind in ("entrypoints", "jobs"):
        entries = body[kind]
        assert isinstance(entries, list)
        for doc in entries:
            assert isinstance(doc, dict)
            rows.append((str(doc["name"]), _one_line(doc, kind[:-1])))
            if kind == "jobs":
                # THE ID A RecordOwner DISPATCHES BY. It is derived, never stored in the
                # document (a digest inside a source-stable surface has no clock a reader
                # can name), so the only way anyone else gets it is by re-deriving the
                # canonical form — which cozy-creator was doing in Go. Printing it here is
                # what deletes that second implementation (cl-004).
                rows.append(
                    (
                        f"{doc['name']}.job_descriptor_id",
                        package_interface.job_descriptor_id(body, str(doc["name"])),
                    )
                )
    counts = package_interface.counts(body)
    # Empty states are definitive, and aggregates ride the listing.
    rows.append(("counts", " · ".join(_plural(n, k) for k, n in counts.items())))
    rows.append((package_interface.DIGEST_KEY, package_interface.package_interface_digest(body)))
    return Result(
        rows=tuple(rows),
        document=body,
        canonical_json=package_interface.canonical_bytes(body),
        next=("cozy-runtime describe <function> — that function's schemas",),
    )


def _plural(count: int, kind: str) -> str:
    """Empty states are definitive — "0 jobs", never a blank."""
    word = kind.replace("_", " ")
    return f"{count} {word if count != 1 else word[:-1]}"


def _one_line(doc: Mapping[str, Json], kind: str) -> str:
    request = doc["request"]
    result = doc["result"]
    assert isinstance(request, dict) and isinstance(result, dict)
    slots = doc.get("models", [])
    assert isinstance(slots, list)
    parts = [
        kind,
        f"{len(_as_list(request['fields']))} request fields",
        "model artifact result"
        if result.get("input") == "model"
        else f"{len(_as_list(result['fields']))} result fields",
    ]
    if slots:
        parts.append(f"{len(slots)} model binding{'s' if len(slots) != 1 else ''}")
    return " · ".join(parts)


def _one(body: Mapping[str, Json], function: str) -> Result:
    entrypoints = body["entrypoints"]
    jobs = body["jobs"]
    assert isinstance(entrypoints, list) and isinstance(jobs, list)
    for kind, docs in (("entrypoint", entrypoints), ("job", jobs)):
        for doc in docs:
            assert isinstance(doc, dict)
            if doc["name"] == function:
                return _schema_view(doc, body, kind)
    known = sorted(str(d["name"]) for d in [*entrypoints, *jobs] if isinstance(d, dict))
    raise CliError(
        "not_found",
        f"no registered callable named {function!r}",
        f"this release registers: {', '.join(known) or '0 functions'}",
        Exit.not_found,
        next=("cozy-runtime describe — the full list",),
    )


def _schema_view(doc: Mapping[str, Json], body: Mapping[str, Json], kind: str) -> Result:
    rows: list[tuple[str, str]] = [("function", f"{doc['name']} ({kind})")]
    view: dict[str, Json] = dict(doc)
    if kind == "job":
        # DERIVED here, and only here. It may not enter the published interface — a
        # `sha256:` inside a source-stable surface leaves a reader no way to know which
        # clock it came from — but a RecordOwner dispatches by it, so a verb that would not
        # say it forces every consumer to re-derive the canonical form (cl-004 measured one
        # doing exactly that, in Go).
        view["job_descriptor_id"] = package_interface.job_descriptor_id(body, str(doc["name"]))
        rows.append(("job_descriptor_id", str(view["job_descriptor_id"])))
    lines: list[str] = ["request:"]
    lines += _fields(doc["request"])
    lines.append("result:")
    lines += _fields(doc["result"])
    for slot in _as_list(doc.get("models", [])):
        assert isinstance(slot, dict)
        lines.append(
            f"model {slot['path']} -> {slot['class']}  "
            f"encoded_leaves: {slot.get('encoded_leaves', 'refuse')}"
            + (f"  fusion: {slot['fusion']}" if "fusion" in slot else "")
        )
        components = slot["component_use"]
        assert isinstance(components, dict)
        for method, names in sorted(components.items()):
            lines.append(f"    {method}: {', '.join(str(n) for n in _as_list(names))}")
    return Result(rows=tuple(rows), lines=tuple(lines), document=view)


def _as_list(value: Json) -> Sequence[Json]:
    assert isinstance(value, list)
    return value


def _fields(struct: Json) -> list[str]:
    assert isinstance(struct, dict)
    if struct.get("input") == "model":
        return ["    ModelArtifact: exact retained native model result"]
    out = []
    for field in _as_list(struct["fields"]):
        assert isinstance(field, dict)
        bits = [
            f"    {field['name']}: {_type(field['type'])}",
            str(field.get("wire", "required")),
        ]
        if (constraints := field.get("constraints")) is not None:
            assert isinstance(constraints, dict)
            bits.append("(" + ", ".join(f"{k}={v}" for k, v in sorted(constraints.items())) + ")")
        out.append("  ".join(bits))
    return out


def _type(rendered: Json) -> str:
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, dict):
        if "fields" in rendered:
            return "object"
        if "asset" in rendered:
            return f"{rendered['asset']} asset"
        if "literal" in rendered:
            return "|".join(repr(v) for v in _as_list(rendered["literal"]))
        if "union" in rendered:
            return "|".join(_type(v) for v in _as_list(rendered["union"]))
        if "list" in rendered:
            return f"list[{_type(rendered['list'])}]"
    return "?"
