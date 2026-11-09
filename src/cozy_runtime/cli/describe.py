"""`cozy-runtime describe` — the PackageInterface from SOURCE, executing no package code.

Always read-only. Full `--json` output is the canonical publication input; there is no
committed interface projection and therefore no writer or staleness-check mode.

The one reader is `internal/static_interface.py`: `ast` over the package tree, a closed
vocabulary, typed refusals (decision #713 — package code is untrusted). Nothing here imports
the package, so describing never runs it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.exits import Exit


def run(function: str | None, opts: Options) -> Result:
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
                Path(opts.dir).expanduser(),
                environment_python=Path(opts.environment_python)
                if opts.environment_python
                else None,
            )
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "a package's surface must be valid before it can be described",
            Exit.validation,
        ) from exc
    if function is not None:
        return _one(body, function)
    return _list(body)


def _list(body: Mapping[str, Json]) -> Result:
    rows: list[tuple[str, str]] = []
    for doc in _as_list(body["entrypoints"]):
        assert isinstance(doc, dict)
        rows.append((str(doc["name"]), _one_line(doc)))
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
    """Empty states are definitive — "0 functions", never a blank."""
    word = kind.replace("_", " ")
    return f"{count} {word if count != 1 else word[:-1]}"


def _one_line(doc: Mapping[str, Json]) -> str:
    request = doc["request"]
    result = doc["result"]
    assert isinstance(request, dict) and isinstance(result, dict)
    slots = doc.get("models", [])
    assert isinstance(slots, list)
    parts = [
        *(
            [f"tool {doc['tool'].get('label') or doc['name']}"]
            if isinstance(doc.get("tool"), dict)
            else []
        ),
        f"{len(_as_list(request['fields']))} request fields",
        "model artifact result"
        if result.get("input") == "model"
        else f"{len(_as_list(result['fields']))} result fields",
    ]
    if slots:
        parts.append(f"{len(slots)} model binding{'s' if len(slots) != 1 else ''}")
    if doc.get("weights_outputs"):
        parts.append("writes weights")
    if doc.get("accelerator") is False:
        parts.append("CPU only")
    if calls := _as_list(doc.get("calls", [])):
        parts.append(
            "calls "
            + ", ".join(f"{c['package']}/{c['function']}" for c in calls if isinstance(c, dict))
        )
    return " · ".join(parts)


def _one(body: Mapping[str, Json], function: str) -> Result:
    entrypoints = _as_list(body["entrypoints"])
    for doc in entrypoints:
        assert isinstance(doc, dict)
        if doc["name"] == function:
            return _schema_view(doc, body)
    known = sorted(str(d["name"]) for d in entrypoints if isinstance(d, dict))
    raise CliError(
        "not_found",
        f"no registered callable named {function!r}",
        f"this release registers: {', '.join(known) or '0 functions'}",
        Exit.not_found,
        next=("cozy-runtime describe — the full list",),
    )


def _schema_view(doc: Mapping[str, Json], body: Mapping[str, Json]) -> Result:
    # The operation identity is derived here, never stored in the source-stable interface.
    view: dict[str, Json] = {
        **doc,
        "descriptor_id": package_interface.descriptor_id(body, str(doc["name"])),
    }
    rows: list[tuple[str, str]] = [
        ("function", str(doc["name"])),
        ("descriptor_id", str(view["descriptor_id"])),
    ]
    lines: list[str] = ["request:"]
    lines += _fields(doc["request"])
    lines.append("result:")
    lines += _fields(doc["result"])
    for slot in _as_list(doc.get("models", [])):
        assert isinstance(slot, dict)
        runs = ", ".join(map(str, _as_list(slot.get("components", []))))
        lines.append(
            f"{'weights' if slot.get('weights') else 'model'} {slot['path']} -> {slot['class']}  "
            f"encoded_leaves: {slot.get('encoded_leaves', 'refuse')}"
            + "".join(
                f"  {key}: {slot[key]}" for key in ("default", "min_bits", "phase") if key in slot
            )
            + (f"  runs: {runs}" if runs else "")
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
            return f"{rendered['asset']} file"
        if "literal" in rendered:
            return "|".join(repr(v) for v in _as_list(rendered["literal"]))
        if "union" in rendered:
            return "|".join(_type(v) for v in _as_list(rendered["union"]))
        if "list" in rendered:
            return f"list[{_type(rendered['list'])}]"
    return "?"
