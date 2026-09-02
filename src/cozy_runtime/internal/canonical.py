"""The ONE canonical-JSON layer for cozy-runtime documents: RFC 8785 (JCS).

tensorfs-v2 owns the writer RULES (tfs-013, `crates/tensorfs-core/src/canon.rs`); this is
the same profile in Python for the documents cozy-runtime authors — PackageInterface,
EnvironmentReceipt, and PackageEnvironment. Shared core: sorted
object keys
(UTF-16 code-unit order), no insignificant whitespace, shortest round-trip numbers,
duplicate-key refusal, lowercase `sha256:` digests, and STORED bytes are canonical bytes.

Recorded divergence from tensorfs's PROFILE (not from RFC 8785): tensorfs restricts its own
header codec to integers, printable ASCII and no `null`. A package interface legitimately carries
float bounds (`le=20.0`), `None` defaults and non-ASCII literals, so this writer admits
those three while retaining the I-JSON safe-number magnitude for every numeric value. The
seam is exercised live:
`scripts/verify.py` runs this codec over tensorfs's own `vectors/cases/canonical` corpus and
reports, per case, whether the refusal is shared or profile-only.

There is no hand-sorted construction path — every object sorts at WRITE time, which is the
stronger form of tfs's `Value::map` lesson (decisions row 175): a projection cannot disagree
with the document it projects because neither one chooses an order.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

#: I-JSON interoperable numeric range. A value outside it refuses regardless of whether its
#: source token used integer, decimal, or exponent notation.
INT_MAX = (1 << 53) - 1
INT_MIN = -INT_MAX

DEPTH_MAX = 32
DOC_MAX_BYTES = 8 * 1024 * 1024

#: The one digest spelling, shared with tensorfs's `ids::prefixed`.
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

#: A JSON document as plain Python: dict, list, str, int, float, bool or None.
Json = Any


class CanonicalError(Exception):
    """A canonicalization refusal: the value or bytes are not admissible here."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- writer


def write(value: Json) -> bytes:
    """The canonical bytes of `value`. These are what gets stored and what gets hashed."""
    out: list[str] = []
    _emit(value, out, 0)
    data = "".join(out).encode("utf-8")
    if len(data) > DOC_MAX_BYTES:
        raise CanonicalError("size_cap", f"{len(data)} bytes exceeds cap {DOC_MAX_BYTES}")
    return data


def digest(value: Json) -> str:
    """`sha256:<lowercase hex>` of the canonical bytes — a document's one identity."""
    return "sha256:" + hashlib.sha256(write(value)).hexdigest()


def _emit(value: Json, out: list[str], depth: int) -> None:
    if depth > DEPTH_MAX:
        raise CanonicalError("depth_cap", f"nesting deeper than {DEPTH_MAX}")
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        if not INT_MIN <= value <= INT_MAX:
            raise CanonicalError("number_range", f"{value} is outside the interoperable range")
        out.append(str(value))
    elif isinstance(value, float):
        out.append(_number(value))
    elif isinstance(value, str):
        out.append(_string(value))
    elif isinstance(value, Mapping):
        out.append("{")
        for index, (key, item) in enumerate(_sorted(value)):
            if index:
                out.append(",")
            out.append(_string(key))
            out.append(":")
            _emit(item, out, depth + 1)
        out.append("}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _emit(item, out, depth + 1)
        out.append("]")
    else:
        raise CanonicalError("wrong_type", f"{type(value).__name__} is not a JSON type")


def _sorted(mapping: Mapping[str, Json]) -> list[tuple[str, Json]]:
    """JCS orders keys by UTF-16 code unit — never by insertion, never by hand."""
    pairs = []
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise CanonicalError("key_grammar", f"object key {key!r} is not a string")
        pairs.append((key, item))
    # JCS sorts by UTF-16 code unit, which differs from Python's code-point order only once
    # a key leaves the BMP. `isascii()` is a single C-level check and every package interface key is
    # an identifier, so the exact-but-costly encode runs only when a key is not ASCII.
    if all(key.isascii() for key, _ in pairs):
        pairs.sort(key=lambda kv: kv[0])
    else:
        pairs.sort(key=lambda kv: kv[0].encode("utf-16-be"))
    for a, b in itertools.pairwise(pairs):
        if a[0] == b[0]:
            raise CanonicalError("duplicate_key", f"key {a[0]!r} appears twice")
    return pairs


_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


#: Almost every string in a package interface is an identifier, a type name or a sentence — none
#: of them need escaping. Checking once beats branching per character: this fast path plus
#: the ASCII key sort took `canonical.write` over the real H3 package interface from 931 us to
#: 525 us (measured, this box). Both paths produce the same canonical digest.
_NEEDS_ESCAPE = re.compile(r'["\\\x00-\x1f]')


def _string(value: str) -> str:
    if _NEEDS_ESCAPE.search(value) is None:
        return f'"{value}"'
    out = ['"']
    for char in value:
        escape = _ESCAPES.get(char)
        if escape is not None:
            out.append(escape)
        elif char < "\x20":
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _number(value: float) -> str:
    """ECMAScript `Number::toString` (ES6 7.1.12.1), which is RFC 8785's number rule.

    The whole algorithm runs on the SHORTEST round-tripping digit string — `repr` gives it,
    the same Grisu/Ryu-class result V8 gives — and never on the value's exact expansion.
    The two are different numbers above 2^53: `1.8645733457839102e20` has shortest digits
    `18645733457839102` and serializes as `186457334578391020000`, while its exact binary
    value expands to `186457334578391023616`. An `is_integer()` fast path returning
    `str(int(value))` printed the second, which is a DIFFERENT DOCUMENT under any canonical
    reader — 12,715 of 50,000 sampled doubles diverged from `JSON.stringify`. There is no
    fast path here now, on purpose: a second spelling of one rule is how it drifted in.
    """
    if math.isnan(value) or math.isinf(value):
        raise CanonicalError("number_range", f"{value!r} has no JSON spelling")
    if abs(value) > INT_MAX:
        raise CanonicalError(
            "number_range",
            f"{value!r} is outside the interoperable numeric range +/-{INT_MAX}",
        )
    if value == 0:
        return "0"
    sign, raw, exponent = Decimal(repr(value)).as_tuple()
    digits = "".join(str(d) for d in raw).rstrip("0") or "0"
    assert isinstance(exponent, int)
    exponent += len(raw) - len(digits)
    k, n = len(digits), len(digits) + exponent
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * -n + digits
    else:
        mantissa = digits[0] if k == 1 else digits[0] + "." + digits[1:]
        body = f"{mantissa}e{'+' if n > 0 else '-'}{abs(n - 1)}"
    return ("-" if sign else "") + body


# --------------------------------------------------------------------------- reader


def parse(data: bytes) -> Json:
    """Parse with the profile's refusals: size cap, duplicate keys, no NaN/Infinity."""
    if len(data) > DOC_MAX_BYTES:
        raise CanonicalError("size_cap", f"{len(data)} bytes exceeds cap {DOC_MAX_BYTES}")
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_no_constants,
            parse_int=_parse_int,
            parse_float=_parse_float,
        )
    except UnicodeDecodeError as exc:
        raise CanonicalError("malformed_json", f"not UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CanonicalError("malformed_json", f"{exc.msg} at offset {exc.pos}") from exc


def parse_canonical(data: bytes) -> Json:
    """Parse AND require the bytes to be exactly the canonical encoding of their content."""
    value = parse(data)
    if write(value) != data:
        raise CanonicalError(
            "noncanonical_encoding", "bytes are not the canonical encoding of their own content"
        )
    return value


def _pairs(pairs: list[tuple[str, Json]]) -> dict[str, Json]:
    seen: dict[str, Json] = {}
    for key, value in pairs:
        if key in seen:
            raise CanonicalError("duplicate_key", f"key {key!r} appears twice")
        seen[key] = value
    return seen


def _no_constants(name: str) -> Json:
    raise CanonicalError("number_range", f"{name} is not admitted")


def _parse_int(token: str) -> int:
    value = int(token)
    if not INT_MIN <= value <= INT_MAX:
        raise CanonicalError("number_range", f"{token} is outside the interoperable numeric range")
    return value


def _parse_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value) or abs(value) > INT_MAX:
        raise CanonicalError("number_range", f"{token} is outside the interoperable numeric range")
    return value
