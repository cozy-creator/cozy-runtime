"""The boot warm pass's payload synthesizer — ONE synthesizer, over SCHEMA DATA.

§3.3: one attempted warm case per warmable entrypoint, through the ordinary dispatch path,
with a payload synthesized deterministically and boundedly from the entrypoint's own schema
at neutral defaults. There is no warmup DSL and there will not be one: a knob here would be
a second declaration surface, free to drift from the schema it claims to describe.

The input is `internal/schema.render`'s canonical schema DOCUMENT — the same data the
package interface publishes — not a Python type. That is deliberate and load-bearing:

* it is the same artifact a reader can inspect, so "what would the warm pass send" is
  answerable without importing anything;
* the walk branches on the document's own tags (`str`, `{"asset": …}`, `{"struct": …}`,
  `{"union": […]}`) rather than on type NAMES, so a renamed class changes nothing and an
  unrenderable type never reaches here;
* nested structs, lists and unions are walked, not skipped — a required nested struct is
  synthesized field by field, and only a leaf with no admissible neutral value stops it.

Every stop is NAMED. `synthesize` returns `(payload, reason)`; a non-empty reason is the
typed `skipped` reason the durable outcome carries, and it names the exact field path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Asset kinds with a real, tiny, deterministic fixture. `image` and `file` are producible
#: because the SDK owns an actual PNG encoder and raw bytes need no encoder at all.
#: `audio` and `video` encoders exist (cr-017's codec plane) but live behind the `media`
#: extra, which the deciding environment may not have — a fixture that exists only in some
#: venvs would make boot behavior env-dependent, so a required audio/video input stays
#: `skipped` with its reason until the warm pass has an unconditional producer.
_ASSET_FIXTURES = ("image", "file")

#: Bounds for a synthesized number when the schema declares them. Neutral means the
#: SMALLEST legal value, never an invented middle: a warm case pays the first-call tax on
#: the cheapest shape the contract admits.
_NUMBER_KEYS = (("ge", 0), ("gt", 1), ("le", 0), ("lt", -1))


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def synthesize(schema: Any, spool: Path) -> tuple[dict[str, Any], str]:
    """A neutral-defaults payload for one entrypoint, or the reason there is none."""
    try:
        value = _value(schema, "", spool)
    except _Stop as stop:
        return {}, stop.reason
    if not isinstance(value, dict):
        return {}, "the payload schema is not a struct"
    return value, ""


def _value(node: Any, path: str, spool: Path) -> Any:
    if isinstance(node, str):
        return _primitive(node, path)
    if not isinstance(node, dict):
        raise _Stop(f"{path or 'payload'}: unrecognized schema node")
    if "fields" in node:
        return _struct(node, path, spool)
    if "asset" in node:
        return _asset(str(node["asset"]), path, spool)
    if "enum" in node:
        values = list(node.get("values", ()))
        if not values:
            raise _Stop(f"{path}: enum declares no values")
        return values[0]
    if "literal" in node:
        members = list(node["literal"])
        if not members:
            raise _Stop(f"{path}: empty literal")
        return members[0]
    if "union" in node:
        # `null` first: an optional field's neutral value is its absence-equivalent.
        members = list(node["union"])
        if "null" in members:
            return None
        for member in members:
            try:
                return _value(member, path, spool)
            except _Stop:
                continue
        raise _Stop(f"{path}: no union member is synthesizable")
    if "list" in node or "set" in node:
        return []  # the neutral list is the empty one
    if "map" in node:
        return {}
    if "tuple" in node:
        return [_value(member, f"{path}[{i}]", spool) for i, member in enumerate(node["tuple"])]
    if "struct_ref" in node:
        raise _Stop(f"{path}: recursive struct has no bounded neutral value")
    raise _Stop(f"{path or 'payload'}: unrecognized schema node")


def _struct(node: dict[str, Any], path: str, spool: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in node.get("fields", ()):
        name = str(field["name"])
        where = f"{path}.{name}" if path else name
        wire = field.get("wire", "required")
        if wire in ("optional", "omissible"):
            # A default or an omissible field is left OUT: the neutral case is the one the
            # deployment would serve for a caller who said nothing.
            continue
        value = _value(field["type"], where, spool)
        constraints = field.get("constraints") or {}
        out[name] = _clamp(value, constraints, where)
    return out


def _clamp(value: Any, constraints: dict[str, Any], path: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, str):
            minimum = constraints.get("min_length")
            if isinstance(minimum, int) and len(value) < minimum:
                return value * (minimum // max(len(value), 1) + 1)
        return value
    for key, offset in _NUMBER_KEYS:
        bound = constraints.get(key)
        if bound is None:
            continue
        candidate = bound + offset
        if key in ("ge", "gt"):
            return candidate if isinstance(value, int) else float(candidate)
        if value > candidate:
            return candidate if isinstance(value, int) else float(candidate)
    del path
    return value


def _primitive(name: str, path: str) -> Any:
    if name == "str":
        return "warm"
    if name == "int":
        return 1
    if name == "float":
        return 1.0
    if name == "bool":
        return False
    if name == "null":
        return None
    if name == "bytes":
        return ""  # base64 of zero bytes
    raise _Stop(f"{path}: {name} has no neutral value")


def _asset(kind: str, path: str, spool: Path) -> str:
    if kind not in _ASSET_FIXTURES:
        raise _Stop(
            f"required {kind} field {path!r} has no synthesizable fixture: this release has "
            f"no {kind} encoder at all (cr-017 owns the codec plane), and a warm case is "
            "never fed a file no part of this build could have produced"
        )
    spool.mkdir(parents=True, exist_ok=True)
    stem = path.replace(".", "-") or "asset"
    target = spool / f"warm-{stem}.{'png' if kind == 'image' else 'bin'}"
    target.write_bytes(_tiny_png() if kind == "image" else b"cozy-warm")
    return f"file://{target}"


def _tiny_png() -> bytes:
    """A REAL 8x8 PNG from the SDK's own encoder — the same encoder a handler would use."""
    from cozy_runtime.author._services import ImageFrame, encode_png

    return encode_png(ImageFrame(8, 8, bytes(8 * 8 * 3)))
