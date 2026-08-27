"""Exact byte and wheel mechanics shared by repository proof drivers."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import Any


class ProofFailure(RuntimeError):
    """Observed proof evidence did not match its contract."""


@dataclass(frozen=True, slots=True)
class Wheel:
    distribution: str
    version: str
    filename: str
    digest: str
    length: int
    tags: tuple[str, ...]


def inventory(path: Path) -> Wheel:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_name = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_name))
    raw = path.read_bytes()
    return Wheel(
        distribution=str(metadata["Name"]),
        version=str(metadata["Version"]),
        filename=path.name,
        digest=digest(raw),
        length=len(raw),
        tags=tuple(sorted(wheel_metadata.get_all("Tag") or [])),
    )


def wheel(entry: Wheel) -> dict[str, Any]:
    return {
        "digest": entry.digest,
        "distribution": entry.distribution,
        "filename": entry.filename,
        "length": entry.length,
        "tags": list(entry.tags),
        "version": entry.version,
    }


def ref(raw: bytes) -> dict[str, Any]:
    return {"digest": digest(raw), "length": len(raw)}


def digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    """Encode integer/string-only proof documents without Runtime's writer."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def parse_json(raw: bytes) -> object:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ProofFailure(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofFailure(f"invalid JSON: {exc}") from exc


def parse_canonical(raw: bytes) -> object:
    value = parse_json(raw)
    require(canonical(value) == raw, "document is not its own canonical JSON encoding")
    return value


def json_file(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_value) + "\n")
    return path.resolve()


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProofFailure(f"cannot read JSON {path}: {exc}") from exc


def write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path.resolve()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ProofFailure(detail)


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
