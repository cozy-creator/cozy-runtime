"""The `run`/`job` payload grammar and the descriptor-derived output field paths.

Grammar (cozy-runtime-cli.md): a positional primary field, `key=value` scalars,
`key=@file` file/asset inputs, `key:=<json>` raw JSON for nested values, and `--in <file>`
supplying the whole payload as JSON with `key=value` merged on top.

Typing is the DESCRIPTOR's, not this module's: a scalar's spelling is decided by the field's
rendered schema, so `steps=20` becomes an int because the schema says int — never because
the string looked numeric. A key the request struct does not declare refuses here rather
than at the executor, because a typo should cost a millisecond and not a model load.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cozy_runtime.cli.io import CliError
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.exits import Exit


def fields(struct: Json) -> dict[str, Json]:
    """name -> rendered type, for one rendered struct."""
    assert isinstance(struct, dict)
    out: dict[str, Json] = {}
    for item in struct.get("fields", []):
        assert isinstance(item, dict)
        out[str(item["name"])] = item["type"]
    return out


def primary_field(struct: Json) -> str | None:
    """The FIRST declared field — what a positional argument fills.

    "First" is the author's declaration order, which msgspec preserves, so the positional
    is the field the author put first rather than a name this module blesses.
    """
    assert isinstance(struct, dict)
    declared = struct.get("fields", [])
    assert isinstance(declared, list)
    for item in declared:
        assert isinstance(item, dict)
        return str(item["name"])
    return None


def asset_paths(struct: Json, prefix: str = "") -> tuple[str, ...]:
    """Every asset-typed field path in a rendered result struct, dotted for nesting.

    These ARE the output ids: the coordinator grants one destination per asset field path,
    and a set that does not match what the handler produced refuses the tail whole. Deriving
    them from the descriptor is what makes `--out` a consequence of the endpoint's declared
    result rather than a convention the CLI and the endpoint both have to remember.
    """
    assert isinstance(struct, dict)
    found: list[str] = []
    for item in struct.get("fields", []):
        assert isinstance(item, dict)
        name = f"{prefix}{item['name']}"
        rendered = item["type"]
        if isinstance(rendered, dict):
            if "asset" in rendered:
                found.append(name)
            elif "fields" in rendered:
                found.extend(asset_paths(rendered, f"{name}."))
    return tuple(found)


def parse(
    args: Sequence[str],
    request: Json,
    *,
    infile: str = "",
    project: Path | None = None,
) -> dict[str, Any]:
    """argv payload terms -> the request document, typed against the rendered schema."""
    declared = fields(request)
    document: dict[str, Any] = {}
    if infile:
        path = Path(infile).expanduser()
        if not path.is_file():
            raise CliError(
                "not_found",
                f"--in {infile}: no such file",
                "--in takes one JSON file holding the whole payload",
                Exit.not_found,
            )
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise CliError(
                "validation",
                f"--in {infile}: {exc}",
                "the file must hold one JSON object",
                Exit.validation,
            ) from exc
        if not isinstance(loaded, dict):
            raise CliError(
                "validation",
                f"--in {infile} holds {type(loaded).__name__}, not an object",
                "the file must hold one JSON object whose keys are request fields",
                Exit.validation,
            )
        document.update(loaded)

    positional: list[str] = []
    for term in args:
        colon = term.find(":=")
        # `key:=json` only when the FIRST `=` is the one in `:=` — `a=b:=c` is a scalar.
        if colon >= 0 and term.find("=") == colon + 1:
            key, _, raw = term.partition(":=")
            document[key] = _json(key, raw)
        elif "=" in term:
            key, _, raw = term.partition("=")
            if raw.startswith("@"):
                document[key] = _file(key, raw[1:], project)
            else:
                document[key] = _scalar(key, raw, declared, request)
        else:
            positional.append(term)

    if positional:
        if len(positional) > 1:
            raise CliError(
                "usage",
                f"{len(positional)} positional values, but exactly one primary field exists",
                "pass the rest as key=value: `cozy-runtime run <fn> <primary> k=v k:=json`",
                Exit.usage,
            )
        name = primary_field(request)
        if name is None:
            raise CliError(
                "usage",
                f"{positional[0]!r} was passed positionally, but the request struct declares "
                "no fields",
                "this function takes no payload",
                Exit.usage,
            )
        document[name] = _scalar(name, positional[0], declared, request)

    unknown = sorted(set(document) - set(declared))
    if unknown:
        raise CliError(
            "validation",
            f"the request declares no field(s) {', '.join(unknown)}",
            f"declared fields: {', '.join(sorted(declared)) or 'none'}",
            Exit.validation,
            next=("cozy-runtime describe <function> — the request schema",),
        )
    return document


def _json(key: str, raw: str) -> Json:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(
            "usage",
            f"{key}:= is not valid JSON: {exc}",
            f"`{key}:=<json>` takes a raw JSON value; use `{key}=<text>` for a plain string",
            Exit.usage,
        ) from exc


def _file(key: str, spelled: str, project: Path | None) -> str:
    path = Path(spelled).expanduser()
    if not path.is_absolute() and project is not None:
        path = (project / path).resolve() if not path.exists() else path.resolve()
    if not path.is_file():
        raise CliError(
            "not_found",
            f"{key}=@{spelled}: no such file",
            "`key=@file` reads that file as the field's value",
            Exit.not_found,
        )
    return str(path)


def _scalar(key: str, raw: str, declared: Mapping[str, Json], request: Json) -> Any:
    """One `key=value` term, spelled by the FIELD's declared type."""
    rendered = declared.get(key)
    if rendered is None:
        # Unknown keys are reported together, after the whole payload is parsed, so a user
        # with two typos learns about both.
        return raw
    return _coerce(key, raw, rendered)


def _coerce(key: str, raw: str, rendered: Json) -> Any:
    if isinstance(rendered, dict) and "union" in rendered:
        union = rendered["union"]
        members = [m for m in union if m != "null"] if isinstance(union, list) else []
        if raw.lower() in ("null", "none"):
            return None
        if len(members) == 1:
            return _coerce(key, raw, members[0])
        return raw
    if rendered == "int":
        return _int(key, raw)
    if rendered == "float":
        return _float(key, raw)
    if rendered == "bool":
        return _bool(key, raw)
    if rendered == "str":
        return raw
    if isinstance(rendered, dict) and ("list" in rendered or "set" in rendered):
        inner = rendered.get("list", rendered.get("set"))
        assert inner is not None
        return [_coerce(key, part, inner) for part in raw.split(",") if part]
    return raw


def _int(key: str, raw: str) -> int:
    try:
        return int(raw, 10)
    except ValueError:
        raise CliError(
            "validation",
            f"{key}={raw!r}: the request declares {key} as int",
            f"pass a whole number, e.g. {key}=20",
            Exit.validation,
        ) from None


def _float(key: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise CliError(
            "validation",
            f"{key}={raw!r}: the request declares {key} as float",
            f"pass a number, e.g. {key}=5.0",
            Exit.validation,
        ) from None


def _bool(key: str, raw: str) -> bool:
    lowered = raw.lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise CliError(
        "validation",
        f"{key}={raw!r}: the request declares {key} as bool",
        f"pass {key}=true or {key}=false",
        Exit.validation,
    )
