"""The digest plane: `cozy.worker.v1` messages <-> their canonical DOCUMENT bytes.

Identity in this protocol is never marshaled protobuf (01 §header): a digest that fences
MEANING is the SHA-256 of a document's exact canonical bytes under tfs-013's writer, and the
bytes travel IN the message so a verifier recomputes rather than re-canonicalizes.

This module is the WORKER's authoring half — `internal/canonical.py` is the writer, and the
mapping below is proto3 -> JSON value:

  * set fields only (a field at its proto3 default is OMITTED — absence-default equals
    pre-introduction behaviour, R8, so omission is the one spelling of that meaning),
  * enums spell as their NUMBER (R2: numbers are normative, names are not),
  * BYTES NAMING LAW: a `bytes` field named `digest`/`*_digest` is a 32-byte SHA-256 spelled
    `sha256:<hex>`; every other `bytes` field is opaque payload spelled base64,
  * a `format` tag (`<full.Name>/1`) domain-separates two document kinds with equal fields.

The reading half refuses non-canonical encoding, floats and a wrong `format`. Keys this binding does
not know are additive fields from a newer writer: the digest still covers the exact bytes, and
the reader acts on the fields it consumes.

Profile note, checked live rather than asserted (`scripts/worker-live.py fixtures` renders
th-024's frozen canonical fixtures through THIS writer and compares bytes + ids):
cozy-runtime's writer admits floats, `null` and non-ASCII because a package interface needs them;
the protocol profile does not, so `_ascii` refuses a non-printable-ASCII field here instead
of spelling it. Within the admissible profile the two writers are byte-identical.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any

from google.protobuf.descriptor import FieldDescriptor as FD
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import Message

from cozy_runtime.internal import canonical

INT_MAX = (1 << 53) - 1
INT_MIN = -INT_MAX

_INT_TYPES = frozenset(
    {
        FD.TYPE_INT32,
        FD.TYPE_INT64,
        FD.TYPE_UINT32,
        FD.TYPE_UINT64,
        FD.TYPE_SINT32,
        FD.TYPE_SINT64,
        FD.TYPE_FIXED32,
        FD.TYPE_FIXED64,
        FD.TYPE_SFIXED32,
        FD.TYPE_SFIXED64,
    }
)
_FLOAT_TYPES = frozenset({FD.TYPE_FLOAT, FD.TYPE_DOUBLE})
type _Scalar = bool | int | str | bytes
_INT_BOUNDS = {
    **dict.fromkeys(
        (FD.TYPE_INT32, FD.TYPE_SINT32, FD.TYPE_SFIXED32, FD.TYPE_ENUM), (-(1 << 31), (1 << 31) - 1)
    ),
    **dict.fromkeys((FD.TYPE_UINT32, FD.TYPE_FIXED32), (0, (1 << 32) - 1)),
    **dict.fromkeys((FD.TYPE_UINT64, FD.TYPE_FIXED64), (0, INT_MAX)),
    **dict.fromkeys((FD.TYPE_INT64, FD.TYPE_SINT64, FD.TYPE_SFIXED64), (INT_MIN, INT_MAX)),
}

# Closed collection fields have one spelling even when empty (worker-protocol
# scripts/canonical.py EXPLICIT_REPEATED; the same table in Creator's canonical writer).
EXPLICIT_REPEATED: dict[str, tuple[str, ...]] = {
    "cozy.worker.v1.DownloadDelegation": ("models", "packages"),
    "cozy.worker.v1.Entrypoint": ("slots",),
    "cozy.worker.v1.Placement": ("entrypoints", "models"),
    "cozy.worker.v1.MachineExecutionCapture": ("bindings", "installed_packages"),
    "cozy.worker.v1.Slot": ("components", "stamps"),
    "cozy.worker.v1.Stamp": ("values",),
}


def doc_format(full_name: str) -> str:
    """The pre-release `format` tag for a document kind: `<full.Name>/1`."""
    return f"{full_name}/{2 if full_name == 'cozy.worker.v1.MachineExecutionCapture' else 1}"


class DocumentError(Exception):
    """A canonical-document refusal, named by code so a caller can classify it."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- digests


def spell(raw: bytes) -> str:
    """The 32 raw bytes protobuf transports -> `sha256:<hex>`. One identity, two spellings."""
    if len(raw) != 32:
        raise DocumentError("malformed_digest", f"{len(raw)} B is not a 32-byte SHA-256")
    return "sha256:" + raw.hex()


def raw(spelled: str) -> bytes:
    """`sha256:<hex>` -> the 32 raw bytes protobuf transports."""
    if canonical.DIGEST_RE.fullmatch(spelled) is None:
        raise DocumentError("malformed_digest", f"{spelled!r} is not a lowercase sha256: digest")
    return bytes.fromhex(spelled[7:])


def digest_of(data: bytes) -> bytes:
    """The raw 32-byte SHA-256 of exactly these bytes — the only digest a receiver computes."""
    return hashlib.sha256(data).digest()


# --------------------------------------------------------------------------- writer


def _ascii(text: str, where: str) -> str:
    for char in text:
        if not 0x20 <= ord(char) <= 0x7E:
            raise DocumentError(
                "non_ascii_field", f"U+{ord(char):04X} in {where}: fields are printable ASCII"
            )
    return text


def _field(fd: FD, value: Any, where: str) -> Any:
    if fd.type == FD.TYPE_BOOL:
        return bool(value)
    if fd.type == FD.TYPE_ENUM:
        return int(value)
    if fd.type in _INT_TYPES:
        if not INT_MIN <= value <= INT_MAX:
            raise DocumentError("number_range", f"{where}={value} is outside the safe range")
        return int(value)
    if fd.type in _FLOAT_TYPES:
        raise DocumentError("non_integer_number", f"{where} is a float: documents are integer-only")
    if fd.type == FD.TYPE_STRING:
        return _ascii(str(value), where)
    if fd.type == FD.TYPE_BYTES:
        if fd.name == "digest" or fd.name.endswith(("_digest", "_digests")):
            return spell(bytes(value))
        return base64.b64encode(bytes(value)).decode()
    if fd.type == FD.TYPE_MESSAGE:
        return body(value)
    raise DocumentError("wrong_type", f"{where}: proto type {fd.type} has no canonical form")


def body(message: Message) -> dict[str, Any]:
    """Set fields plus the explicit empty collections, in no particular order — the writer sorts."""
    out: dict[str, Any] = {}
    for fd, value in message.ListFields():
        where = fd.full_name
        out[fd.name] = (
            [_field(fd, item, where) for item in value]
            if fd.is_repeated
            else _field(fd, value, where)
        )
    for name in EXPLICIT_REPEATED.get(message.DESCRIPTOR.full_name, ()):
        out.setdefault(name, [])
    return out


def document(message: Message) -> dict[str, Any]:
    """The document: the message's set fields plus the `format` tag that separates kinds."""
    rendered = body(message)
    if "format" in rendered:
        raise DocumentError("unknown_field", "`format` is reserved for the document tag")
    rendered["format"] = doc_format(message.DESCRIPTOR.full_name)
    return rendered


def canonical_bytes(message: Message) -> bytes:
    return canonical.write(document(message))


def identity(message: Message) -> tuple[bytes, bytes]:
    """`(canonical bytes, raw 32-byte digest)` — what a message carries as a fenced pair."""
    data = canonical_bytes(message)
    return data, digest_of(data)


# --------------------------------------------------------------------------- reader


def parse[M: Message](data: bytes, cls: type[M]) -> M:
    """Exact canonical bytes -> the typed message, decoded once at this boundary.

    The bytes stay the document's identity: a caller that names the document digests `data`,
    never a re-encoding of the message. A field the writer omitted reads as its proto3
    default, which is exactly what omission means. Keys this binding does not know are
    additive fields from a newer writer and are skipped; a known key of the wrong JSON type
    is refused here by name, not discovered later as a KeyError in the consumer.
    """
    if len(data) > canonical.DOC_MAX_BYTES:
        raise DocumentError("size_cap", f"{len(data)} B over the document cap")
    try:
        value = canonical.parse_canonical(data)
    except canonical.CanonicalError as exc:
        raise DocumentError(exc.code, exc.detail) from exc
    if not isinstance(value, dict):
        raise DocumentError("wrong_type", "a document is a JSON object")
    want = doc_format(cls.DESCRIPTOR.full_name)
    if value.get("format") != want:
        raise DocumentError("unknown_format", f"{value.get('format')!r} is not {want!r}")
    return from_body({key: item for key, item in value.items() if key != "format"}, cls)


def from_body[M: Message](value: object, cls: type[M]) -> M:
    """A message's fields as another document embeds them -> the typed message, refused as
    `parse` refuses."""
    _integers_only(value)
    message = cls()
    _fill(message, _object(value, cls.DESCRIPTOR.full_name))
    return message


def _fill(message: Message, fields: dict[str, object]) -> None:
    for name, value in fields.items():
        fd = message.DESCRIPTOR.fields_by_name.get(name)
        if fd is None:
            continue
        where = fd.full_name
        oneof = fd.containing_oneof
        if oneof is not None and message.WhichOneof(oneof.name) not in (None, name):
            raise DocumentError("oneof_conflict", f"{where} is one choice of several set")
        if fd.is_repeated:
            if not isinstance(value, list):
                raise DocumentError("wrong_type", f"{where} is not a list")
            if fd.type == FD.TYPE_MESSAGE:
                messages: RepeatedCompositeFieldContainer[Message] = getattr(message, name)
                for item in value:
                    _fill(messages.add(), _object(item, where))
            else:
                scalars: RepeatedScalarFieldContainer[_Scalar] = getattr(message, name)
                scalars.extend(_scalar(fd.type, name, item, where) for item in value)
        elif fd.type == FD.TYPE_MESSAGE:
            child: Message = getattr(message, name)
            child.SetInParent()
            _fill(child, _object(value, where))
        else:
            setattr(message, name, _scalar(fd.type, name, value, where))


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DocumentError("wrong_type", f"{where} is not an object")
    return value


def _scalar(kind: int, name: str, value: object, where: str) -> _Scalar:
    if kind == FD.TYPE_BOOL and isinstance(value, bool):
        return value
    if kind in _INT_BOUNDS and isinstance(value, int) and not isinstance(value, bool):
        low, high = _INT_BOUNDS[kind]
        if not low <= value <= high:
            raise DocumentError("number_range", f"{where}={value} is outside its field")
        return value
    if kind == FD.TYPE_STRING and isinstance(value, str):
        return value
    if kind == FD.TYPE_BYTES and isinstance(value, str):
        if name == "digest" or name.endswith(("_digest", "_digests")):
            return raw(value)
        try:
            return base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise DocumentError("malformed_bytes", f"{where} is not base64") from exc
    raise DocumentError("wrong_type", f"{where} is not a proto type {kind} value")


def _integers_only(value: object) -> None:
    """A float has no single canonical spelling across writers, so it is an ambiguous encoding."""
    if isinstance(value, float):
        raise DocumentError("non_integer_number", "documents are integer-only")
    if isinstance(value, dict):
        for item in value.values():
            _integers_only(item)
    elif isinstance(value, list):
        for item in value:
            _integers_only(item)


def read(data: bytes, cls: type[Message]) -> dict[str, Any]:
    """Exact canonical bytes -> the untyped document. Prefer `parse`, which returns the message."""
    if len(data) > canonical.DOC_MAX_BYTES:
        raise DocumentError("size_cap", f"{len(data)} B over the document cap")
    try:
        value = canonical.parse_canonical(data)
    except canonical.CanonicalError as exc:
        raise DocumentError(exc.code, exc.detail) from exc
    if not isinstance(value, dict):
        raise DocumentError("wrong_type", "a document is a JSON object")
    _integers_only(value)
    want = doc_format(cls.DESCRIPTOR.full_name)
    if value.get("format") != want:
        raise DocumentError("unknown_format", f"{value.get('format')!r} is not {want!r}")
    return value
