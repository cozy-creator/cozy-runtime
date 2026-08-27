"""`cozy-runtime describe` — emit the EndpointDescriptor without loading weights.

READ-ONLY by default: an inspection command never silently dirties a repository
(cozy-runtime-cli.md). `--write-descriptor` commits the pretty descriptor, `--check` is the
CI/build staleness verb and exits 13 naming which pair diverged.

Weights are structurally unreachable here, not merely unrequested: discovery imports the app
and freezes it, and a model parameter is a TYPE. Nothing on this path constructs a `Loader`,
so no census, no fill and no backend call can happen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit

FLAGS = ("--write-descriptor", "--check")


def run(function: str | None, opts: Options, *, write: bool, check: bool) -> Result:
    if write and check:
        raise CliError(
            "usage",
            "--write-descriptor and --check are opposites",
            "--check verifies the committed descriptor; --write-descriptor replaces it",
            Exit.usage,
        )
    project = Path(opts.dir).expanduser()
    try:
        found = discover(project)
        body = desc.build(found)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "an endpoint's surface must be valid before it can be described (§1.0)",
            Exit.validation,
        ) from exc
    document = desc.document(body)

    if check:
        return _check(project, document)
    if write:
        target = desc.write(project, document)
        return Result(
            rows=(
                ("wrote", str(target.relative_to(project) if project != Path() else target)),
                ("application", str(body["application"])),
                (desc.DIGEST_KEY, str(document[desc.DIGEST_KEY])),
                ("bytes", str(target.stat().st_size)),
            ),
            document=document,
            next=(f"git add {desc.FILE} — the surface contract is reviewed in the diff",),
        )
    if function is not None:
        return _one(body, function)
    return _list(body, document)


def _check(project: Path, derived: Mapping[str, Json]) -> Result:
    try:
        committed = desc.read(project)
        desc.compare(committed, derived, ("the committed descriptor", "this release"))
    except desc.StaleDescriptor as exc:
        raise CliError(
            "conflict",
            str(exc),
            "run `cozy-runtime describe --write-descriptor` and commit the result; if the "
            "source did not change, the two environments derive different surfaces and one "
            "of them is broken",
            Exit.conflict,
        ) from exc
    return Result(
        rows=(
            ("descriptor", "current"),
            (desc.DIGEST_KEY, str(derived[desc.DIGEST_KEY])),
        )
    )


def _list(body: Mapping[str, Json], document: Mapping[str, Json]) -> Result:
    rows: list[tuple[str, str]] = []
    for kind in ("entrypoints", "jobs"):
        entries = body[kind]
        assert isinstance(entries, list)
        for doc in entries:
            assert isinstance(doc, dict)
            rows.append((str(doc["name"]), _one_line(doc)))
            if kind == "jobs":
                # THE ID A RecordOwner DISPATCHES BY. It is derived, never stored in the
                # document (a digest inside a source-stable surface has no clock a reader
                # can name), so the only way anyone else gets it is by re-deriving the
                # canonical form — which cozy-creator was doing in Go. Printing it here is
                # what deletes that second implementation (cl-004).
                rows.append(
                    (
                        f"{doc['name']}.job_descriptor_id",
                        desc.job_descriptor_id(body, str(doc["name"])),
                    )
                )
    counts = desc.counts(body)
    # Empty states are definitive, and aggregates ride the listing.
    rows.append(("counts", " · ".join(_plural(n, k) for k, n in counts.items())))
    rows.append((desc.DIGEST_KEY, str(document[desc.DIGEST_KEY])))
    return Result(
        rows=tuple(rows),
        document=document,
        next=(
            "cozy-runtime describe <function> — that function's schemas",
            "cozy-runtime describe --write-descriptor — commit the surface contract",
        ),
    )


def _plural(count: int, kind: str) -> str:
    """Empty states are definitive — "0 jobs", never a blank."""
    word = kind.replace("_", " ")
    return f"{count} {word if count != 1 else word[:-1]}"


def _one_line(doc: Mapping[str, Json]) -> str:
    request = doc["request"]
    result = doc["result"]
    assert isinstance(request, dict) and isinstance(result, dict)
    caps = doc["capabilities"]
    assert isinstance(caps, list)
    slots = doc.get("models", [])
    assert isinstance(slots, list)
    parts = [
        f"{doc['kind']}",
        f"({request['struct']}) -> {result['struct']}",
        f"caps: {', '.join(str(c) for c in caps) or 'none'}",
        f"gpu={str(doc['gpu']).lower()}",
    ]
    if slots:
        parts.append(f"{len(slots)} model binding{'s' if len(slots) != 1 else ''}")
    return " · ".join(parts)


def _one(body: Mapping[str, Json], function: str) -> Result:
    entrypoints = body["entrypoints"]
    jobs = body["jobs"]
    assert isinstance(entrypoints, list) and isinstance(jobs, list)
    for doc in [*entrypoints, *jobs]:
        assert isinstance(doc, dict)
        if doc["name"] == function:
            return _schema_view(doc, body)
    known = sorted(str(d["name"]) for d in [*entrypoints, *jobs] if isinstance(d, dict))
    raise CliError(
        "not_found",
        f"no registered callable named {function!r}",
        f"this release registers: {', '.join(known) or '0 functions'}",
        Exit.not_found,
        next=("cozy-runtime describe — the full list",),
    )


def _schema_view(doc: Mapping[str, Json], body: Mapping[str, Json]) -> Result:
    rows: list[tuple[str, str]] = [
        ("function", f"{doc['name']} ({doc['kind']})"),
        ("capabilities", ", ".join(str(c) for c in _as_list(doc["capabilities"])) or "none"),
        ("gpu", str(doc["gpu"]).lower()),
    ]
    view: dict[str, Json] = dict(doc)
    if doc["kind"] == "job":
        # DERIVED here, and only here. It may not enter the committed descriptor — a
        # `sha256:` inside a source-stable surface leaves a reader no way to know which
        # clock it came from — but a RecordOwner dispatches by it, so a verb that would not
        # say it forces every consumer to re-derive the canonical form (cl-004 measured one
        # doing exactly that, in Go).
        view["job_descriptor_id"] = desc.job_descriptor_id(body, str(doc["name"]))
        rows.append(("job_descriptor_id", str(view["job_descriptor_id"])))
    lines: list[str] = ["request:"]
    lines += _fields(doc["request"])
    lines.append("result:")
    lines += _fields(doc["result"])
    for slot in _as_list(doc.get("models", [])):
        assert isinstance(slot, dict)
        stamps = slot["stamps"]
        assert isinstance(stamps, dict)
        rendered = ",".join(f"{k}={v}" for k, v in sorted(stamps.items())) or "none"
        lines.append(f"model {slot['path']} -> {slot['class']}  stamps: {rendered}")
        components = slot["component_use"]
        assert isinstance(components, dict)
        for method, names in sorted(components.items()):
            lines.append(f"    {method}: {', '.join(str(n) for n in _as_list(names))}")
    if (preflight := doc.get("preflight")) is not None:
        assert isinstance(preflight, dict)
        lines.append(f"preflight {preflight['identity']}:")
        lines += _fields(preflight["facts"])
    return Result(rows=tuple(rows), lines=tuple(lines), document=view)


def _as_list(value: Json) -> Sequence[Json]:
    assert isinstance(value, list)
    return value


def _fields(struct: Json) -> list[str]:
    assert isinstance(struct, dict)
    out = []
    for field in _as_list(struct["fields"]):
        assert isinstance(field, dict)
        bits = [f"    {field['name']}: {_type(field['type'])}", str(field["wire"])]
        if "default" in field:
            bits.append(f"= {field['default']!r}")
        if (constraints := field.get("constraints")) is not None:
            assert isinstance(constraints, dict)
            bits.append("(" + ", ".join(f"{k}={v}" for k, v in sorted(constraints.items())) + ")")
        if (axes := field.get("shape_axes")) is not None:
            bits.append(f"axes: {', '.join(str(a) for a in _as_list(axes))}")
        out.append("  ".join(bits))
    return out


def _type(rendered: Json) -> str:
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, dict):
        for key in ("struct", "struct_ref", "enum"):
            if key in rendered:
                return str(rendered[key])
        if "asset" in rendered:
            return f"{rendered['asset']} asset"
        if "literal" in rendered:
            return "|".join(repr(v) for v in _as_list(rendered["literal"]))
        if "union" in rendered:
            return "|".join(_type(v) for v in _as_list(rendered["union"]))
        if "list" in rendered:
            return f"list[{_type(rendered['list'])}]"
        if "set" in rendered:
            return f"set[{_type(rendered['set'])}]"
        if "tuple" in rendered:
            return f"tuple[{', '.join(_type(v) for v in _as_list(rendered['tuple']))}]"
        if "map" in rendered:
            return f"dict[{', '.join(_type(v) for v in _as_list(rendered['map']))}]"
    return "?"
