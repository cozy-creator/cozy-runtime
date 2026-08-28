"""Exact byte and wheel mechanics shared by repository proof drivers."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, cast

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


def installed_base_ownership() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Measure the proof driver's complete installed distribution/import-root ownership."""

    versions: dict[str, str] = {}
    for installed in importlib.metadata.distributions():
        name = normalize_distribution(str(installed.metadata["Name"] or ""))
        version = str(installed.version)
        if not name or not version or name in versions:
            raise ProofFailure(f"installed distribution inventory is ambiguous: {name!r}")
        versions[name] = version
    providers: dict[str, set[str]] = {}
    for root, names in importlib.metadata.packages_distributions().items():
        if _IMPORT_ROOT_RE.fullmatch(root) is None:
            continue
        normalized = {normalize_distribution(name) for name in names}
        if not normalized or not normalized.issubset(versions):
            raise ProofFailure(f"installed import root has unknown providers: {root!r}")
        providers.setdefault(root, set()).update(normalized)
    for root in sys.stdlib_module_names:
        if _IMPORT_ROOT_RE.fullmatch(root) is not None:
            providers.setdefault(root, set()).add("python-stdlib")
    distributions = [
        {"distribution": name, "version": version} for name, version in sorted(versions.items())
    ]
    roots: list[dict[str, object]] = [
        {"providers": sorted(names), "root": root} for root, names in sorted(providers.items())
    ]
    return distributions, roots


def normalize_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def pack_installed(root: Path, distribution: str) -> Path:
    """Repack one installed proof distribution into deterministic exact wheel bytes."""

    installed = importlib.metadata.distribution(distribution)
    files: dict[str, bytes] = {}
    record = ""
    for member in installed.files or ():
        name = PurePosixPath(str(member)).as_posix()
        if ".." in PurePosixPath(name).parts or name.endswith(
            ("/INSTALLER", "/RECORD", "/REQUESTED", "/direct_url.json", "/uv_cache.json")
        ):
            continue
        path = cast(Path, installed.locate_file(member))
        if path.is_file():
            files[name] = path.read_bytes()
        if name.endswith(".dist-info/METADATA"):
            record = str(PurePosixPath(name).parent / "RECORD")
    require(
        bool(record and files), f"installed distribution has no wheel inventory: {distribution}"
    )
    rows = [[name, _record_hash(raw), str(len(raw))] for name, raw in sorted(files.items())]
    rows.append([record, "", ""])
    rendered = io.StringIO(newline="")
    csv.writer(rendered, lineterminator="\n").writerows(rows)
    files[record] = rendered.getvalue().encode()
    wheel_meta = next(raw for name, raw in files.items() if name.endswith(".dist-info/WHEEL"))
    tags = BytesParser().parsebytes(wheel_meta).get_all("Tag") or []
    require(bool(tags), f"installed distribution has no wheel tag: {distribution}")
    normalized = str(installed.metadata["Name"]).replace("-", "_")
    path = root / f"{normalized}-{installed.version}-{tags[0]}.whl"
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as wheel_file:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            wheel_file.writestr(info, raw)
    return path


def _record_hash(raw: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
    return "sha256=" + encoded


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
        first = path.parts[0]
        if first.endswith((".dist-info", ".egg-info", ".libs")) or first == "__pycache__":
            continue
        root = first.partition(".")[0]
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
