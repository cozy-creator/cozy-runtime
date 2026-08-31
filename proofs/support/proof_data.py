"""Exact byte and wheel mechanics shared by repository proof drivers."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

_IMPORT_ROOT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
    import_roots: tuple[str, ...]


def inventory(path: Path) -> Wheel:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        wheel_name = next(name for name in names if name.endswith(".dist-info/WHEEL"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_name))
    raw = path.read_bytes()
    return Wheel(
        distribution=normalize_distribution(str(metadata["Name"])),
        version=str(metadata["Version"]),
        filename=path.name,
        digest=digest(raw),
        length=len(raw),
        tags=tuple(sorted(wheel_metadata.get_all("Tag") or [])),
        import_roots=_wheel_import_roots(names),
    )


def wheel(entry: Wheel, *, import_roots: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "digest": entry.digest,
        "distribution": entry.distribution,
        "filename": entry.filename,
        "length": entry.length,
        "tags": list(entry.tags),
        "version": entry.version,
    }
    if import_roots:
        row["import_roots"] = list(entry.import_roots)
    return row


def normalize_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _wheel_import_roots(names: list[str]) -> tuple[str, ...]:
    roots: set[str] = set()
    for name in names:
        path = PurePosixPath(name)
        if not path.parts:
            continue
        if path.parts[0].endswith(".data"):
            if len(path.parts) < 3 or path.parts[1] not in {"purelib", "platlib"}:
                continue
            path = PurePosixPath(*path.parts[2:])
        if len(path.parts) == 1:
            filename = path.name
            if not filename.endswith((".py", ".pyi", ".so", ".pyd", ".dylib", ".dll")):
                continue
        elif not path.name.endswith((".py", ".pyi", ".so", ".pyd", ".dylib", ".dll")):
            continue
        first = path.parts[0]
        if first.endswith((".dist-info", ".egg-info", ".libs")) or first == "__pycache__":
            continue
        root = first.removesuffix(".py").removesuffix(".pyi").partition(".")[0]
        if _IMPORT_ROOT_RE.fullmatch(root) is not None:
            roots.add(root)
    return tuple(sorted(roots))


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
