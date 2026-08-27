"""Exact, offline endpoint-environment materialization.

This module is deliberately below the release-document adapter.  th-044 owns the canonical
``WheelhouseManifest`` / ``ResolvedWheelSet`` shapes and th-046 owns the endpoint-bundle shape;
none of those shapes is frozen yet.  Their eventual reader must turn those already-verified
canonical bytes into :class:`EnvironmentInput`.  Keeping that join typed here prevents this
runtime from quietly inventing a second resolver or a provisional manifest grammar.

The mechanism itself has one meaning:

    exact canonical identities + exact wheel inventory + exact local byte grants
        -> immutable venv generation + measured InstalledEnvironmentReceipt

It never downloads, resolves, builds, reads an index, imports a source tree, or uses host
site-packages.  A staging directory is not selectable.  Only a complete generation containing
its canonical receipt is atomically renamed into the receipt-keyed generation namespace.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

RECEIPT_FORMAT = "cozy.worker.v1.InstalledEnvironmentReceipt/1"
RECEIPT_NAME = "installed-environment-receipt.json"
MAX_WHEEL_BYTES = 8 << 30
MAX_WHEEL_MEMBERS = 200_000
MAX_WHEEL_EXPANDED_BYTES = 32 << 30
MAX_WHEEL_EXPANSION = 64
MAX_METADATA_BYTES = 4 << 20
PINNED_UV_VERSION = "uv 0.9.18"
PINNED_UV_LENGTH = 55_339_456
PINNED_UV_DIGEST = "sha256:7f73d66b9767a281c2cae604b7cd0b6eaa1f0dda5ae28b9067e6de745d9593eb"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NORMALIZE = re.compile(r"[-_.]+")
_GLIBC = re.compile(r"^glibc([0-9]+)\.([0-9]+)$")


class EnvironmentRefusal(Exception):
    """A fail-closed materialization verdict with a stable machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class PlatformTarget:
    os_arch: str
    libc: str
    python_abi: str
    accelerator_backend: str
    accelerator_abi: str

    @classmethod
    def read(cls, value: object) -> PlatformTarget:
        if not isinstance(value, dict):
            raise EnvironmentRefusal("platform_target_invalid", "platform_target is not an object")
        keys = {"os_arch", "libc", "python_abi", "accelerator_backend", "accelerator_abi"}
        if set(value) != keys or not all(isinstance(value[k], str) and value[k] for k in keys):
            raise EnvironmentRefusal(
                "platform_target_invalid", "platform_target must contain its five non-empty axes"
            )
        return cls(**{key: value[key] for key in keys})

    def document(self) -> dict[str, str]:
        return {
            "os_arch": self.os_arch,
            "libc": self.libc,
            "python_abi": self.python_abi,
            "accelerator_backend": self.accelerator_backend,
            "accelerator_abi": self.accelerator_abi,
        }


@dataclass(frozen=True, slots=True)
class WheelInventoryEntry:
    """One row decoded from the canonical bundle/wheelhouse documents by their owner."""

    distribution: str
    version: str
    filename: str
    digest: str
    length: int
    tags: tuple[str, ...]
    owner: str
    """``project`` | ``bundle`` | ``wheelhouse``; receipt identity ignores storage owner."""


@dataclass(frozen=True, slots=True)
class BlobGrant:
    """The local bytes authorized for one inventory digest; paths never enter a document."""

    digest: str
    length: int
    path: Path


@dataclass(frozen=True, slots=True)
class EnvironmentInput:
    environment_spec_bytes: bytes
    environment_spec_digest: str
    endpoint_bundle_bytes: bytes
    wheelhouse_manifest_bytes: bytes
    wheels: tuple[WheelInventoryEntry, ...]
    grants: tuple[BlobGrant, ...]


@dataclass(frozen=True, slots=True)
class InstalledEnvironment:
    generation: Path
    python: Path
    receipt_bytes: bytes
    receipt_digest: str
    reused: bool


@dataclass(frozen=True, slots=True)
class _InspectedWheel:
    entry: WheelInventoryEntry
    distribution: str
    version: str
    tags: tuple[str, ...]
    path: Path


def materialize(
    source: EnvironmentInput,
    root: Path,
    *,
    uv: Path,
    python: Path = Path(sys.executable),
    base_env: Mapping[str, str] | None = None,
    expected_receipt_digest: str = "",
) -> InstalledEnvironment:
    """Install and atomically publish one exact generation, or refuse without selecting it."""

    spec, target = _verify_input(source)
    env = _sealed_env(base_env or {})
    _verify_uv(uv, env)
    _verify_host_target(target, python, env)
    inspected = _verify_grants(source)
    if expected_receipt_digest and not _DIGEST.fullmatch(expected_receipt_digest):
        raise EnvironmentRefusal("environment_receipt_digest_invalid", expected_receipt_digest)

    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    expected_generation = generations / expected_receipt_digest.removeprefix("sha256:")
    if expected_receipt_digest and expected_generation.exists():
        return verify_generation(
            expected_generation,
            receipt_digest=expected_receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
            reused=True,
        )

    staging = root / f".staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=False, mode=0o700)
        venv = staging / "venv"
        install_wheels = _stage_wheels(inspected, staging / "wheelhouse")
        _run(
            [
                str(uv),
                "venv",
                "--python",
                str(python),
                "--no-python-downloads",
                "--no-project",
                "--no-config",
                "--no-cache",
                "--offline",
                str(venv),
            ],
            env,
            "venv_create_failed",
        )
        _strip_uv_venv_files(venv)
        venv_python = _venv_python(venv)
        _run(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(venv_python),
                "--no-python-downloads",
                "--no-deps",
                "--no-build",
                "--no-index",
                "--no-config",
                "--no-cache",
                "--offline",
                *(str(w.path) for w in install_wheels),
            ],
            env,
            "wheel_install_failed",
        )
        _strip_installer_paths(venv)
        records = _verify_installed(
            venv, {wheel.distribution: (wheel.version, "") for wheel in install_wheels}
        )
        _verify_installed_target(venv_python, target, env)
        receipt_bytes = canonical.write(_receipt(source, spec, target, install_wheels, records))
        receipt_digest = _sha(receipt_bytes)
        if expected_receipt_digest and receipt_digest != expected_receipt_digest:
            raise EnvironmentRefusal(
                "environment_receipt_mismatch",
                f"measured {receipt_digest}, expected {expected_receipt_digest}",
            )
        final = generations / receipt_digest.removeprefix("sha256:")
        _remove_tree(staging / "wheelhouse")
        receipt_path = staging / RECEIPT_NAME
        _write_durable(receipt_path, receipt_bytes)
        _make_immutable(staging)
        _fsync_dir(staging)
        try:
            os.rename(staging, final)
            final.chmod(final.stat().st_mode & ~0o222)
        except FileExistsError:
            _remove_tree(staging)
        _fsync_dir(generations)
        return verify_generation(
            final,
            receipt_digest=receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
            reused=False,
        )
    except Exception:
        if staging.exists():
            _remove_tree(staging)
        raise


def verify_generation(
    generation: Path,
    *,
    receipt_digest: str,
    environment_spec_digest: str,
    reused: bool = True,
) -> InstalledEnvironment:
    """Re-measure a published generation before it is allowed to call itself staged."""

    receipt_path = generation / RECEIPT_NAME
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as exc:
        raise EnvironmentRefusal("environment_receipt_absent", str(exc)) from exc
    if _sha(receipt_bytes) != receipt_digest:
        raise EnvironmentRefusal("environment_receipt_corrupt", str(receipt_path))
    try:
        value = canonical.parse_canonical(receipt_bytes)
    except canonical.CanonicalError as exc:
        raise EnvironmentRefusal("environment_receipt_invalid", str(exc)) from exc
    top_keys = {
        "format",
        "environment_spec_digest",
        "endpoint_bundle_digest",
        "project_wheel_digest",
        "wheelhouse_manifest_digest",
        "platform_target",
        "distributions",
    }
    if (
        not isinstance(value, dict)
        or set(value) != top_keys
        or value.get("format") != RECEIPT_FORMAT
    ):
        raise EnvironmentRefusal("environment_receipt_invalid", "wrong or absent format")
    if generation.name != receipt_digest.removeprefix("sha256:"):
        raise EnvironmentRefusal("environment_generation_wrong_key", str(generation))
    for field in (
        "environment_spec_digest",
        "endpoint_bundle_digest",
        "project_wheel_digest",
        "wheelhouse_manifest_digest",
    ):
        if not isinstance(value[field], str) or not _DIGEST.fullmatch(value[field]):
            raise EnvironmentRefusal("environment_receipt_invalid", f"{field} is not a digest")
    PlatformTarget.read(value["platform_target"])
    if value.get("environment_spec_digest") != environment_spec_digest:
        raise EnvironmentRefusal(
            "environment_receipt_wrong_spec", f"receipt does not name {environment_spec_digest}"
        )
    rows = value.get("distributions")
    if not isinstance(rows, list):
        raise EnvironmentRefusal("environment_receipt_invalid", "distributions is not a list")
    expected: dict[str, tuple[str, str]] = {}
    prior = ""
    for row in rows:
        row_keys = {"distribution", "version", "wheel_digest", "record_digest"}
        if not isinstance(row, dict) or set(row) != row_keys:
            raise EnvironmentRefusal("environment_receipt_invalid", "distribution row invalid")
        if (
            not isinstance(row["distribution"], str)
            or not isinstance(row["version"], str)
            or not isinstance(row["wheel_digest"], str)
            or not isinstance(row["record_digest"], str)
        ):
            raise EnvironmentRefusal("environment_receipt_invalid", "distribution row types")
        distribution = row["distribution"]
        version = row["version"]
        wheel_digest = row["wheel_digest"]
        record_digest = row["record_digest"]
        if (
            not distribution
            or distribution != _name(distribution)
            or distribution <= prior
            or not version
            or not _DIGEST.fullmatch(wheel_digest)
            or not _DIGEST.fullmatch(record_digest)
        ):
            raise EnvironmentRefusal(
                "environment_receipt_invalid", f"invalid or unsorted row {distribution!r}"
            )
        prior = distribution
        expected[distribution] = (version, record_digest)
    venv = generation / "venv"
    _verify_installed(venv, expected)
    if generation.stat().st_mode & 0o222:
        raise EnvironmentRefusal("environment_generation_mutable", str(generation))
    for path in generation.rglob("*"):
        if not path.is_symlink() and path.stat().st_mode & 0o222:
            raise EnvironmentRefusal("environment_generation_mutable", str(path))
    return InstalledEnvironment(
        generation=generation,
        python=_venv_python(venv),
        receipt_bytes=receipt_bytes,
        receipt_digest=receipt_digest,
        reused=reused,
    )


def _verify_input(source: EnvironmentInput) -> tuple[dict[str, Any], PlatformTarget]:
    if _sha(source.environment_spec_bytes) != source.environment_spec_digest:
        raise EnvironmentRefusal("environment_spec_digest_mismatch", source.environment_spec_digest)
    try:
        spec = documents.read(source.environment_spec_bytes, pb.EndpointEnvironmentSpec)
    except (documents.DocumentError, canonical.CanonicalError) as exc:
        raise EnvironmentRefusal("environment_spec_invalid", str(exc)) from exc
    target = PlatformTarget.read(spec.get("platform_target"))
    for field, data in (
        ("endpoint_bundle_digest", source.endpoint_bundle_bytes),
        ("wheelhouse_manifest_digest", source.wheelhouse_manifest_bytes),
    ):
        expected = spec.get(field)
        if not isinstance(expected, str) or _sha(data) != expected:
            raise EnvironmentRefusal(
                f"{field}_mismatch", f"measured {_sha(data)}, expected {expected}"
            )
        try:
            value = canonical.parse_canonical(data)
        except canonical.CanonicalError as exc:
            raise EnvironmentRefusal(f"{field}_invalid", str(exc)) from exc
        if not isinstance(value, dict) or not isinstance(value.get("format"), str):
            raise EnvironmentRefusal(f"{field}_invalid", "canonical document has no format")
    return spec, target


def _verify_grants(source: EnvironmentInput) -> tuple[_InspectedWheel, ...]:
    expected = {wheel.digest: wheel for wheel in source.wheels}
    granted = {grant.digest: grant for grant in source.grants}
    if len(expected) != len(source.wheels):
        raise EnvironmentRefusal("wheel_inventory_duplicate_digest", "wheel digests are not unique")
    if len(granted) != len(source.grants):
        raise EnvironmentRefusal("blob_grant_duplicate_digest", "grant digests are not unique")
    if missing := sorted(set(expected) - set(granted)):
        raise EnvironmentRefusal("wheel_blob_missing", missing[0])
    if extra := sorted(set(granted) - set(expected)):
        raise EnvironmentRefusal("wheel_blob_extra", extra[0])
    project_digest = _digest_field(source.environment_spec_bytes, "project_wheel_digest")
    projects = [wheel for wheel in source.wheels if wheel.owner == "project"]
    if len(projects) != 1 or projects[0].digest != project_digest:
        raise EnvironmentRefusal(
            "project_wheel_mismatch", "inventory must contain exactly the spec-named project wheel"
        )
    if any(wheel.owner not in {"project", "bundle", "wheelhouse"} for wheel in source.wheels):
        raise EnvironmentRefusal(
            "wheel_owner_invalid", "wheel owner is not project/bundle/wheelhouse"
        )
    inspected = tuple(_inspect_wheel(wheel, granted[wheel.digest]) for wheel in source.wheels)
    names = [wheel.distribution for wheel in inspected]
    if len(names) != len(set(names)):
        raise EnvironmentRefusal("wheel_distribution_duplicate", "normalized names are not unique")
    return tuple(sorted(inspected, key=lambda wheel: wheel.distribution))


def _inspect_wheel(entry: WheelInventoryEntry, grant: BlobGrant) -> _InspectedWheel:
    if not _DIGEST.fullmatch(entry.digest) or entry.length <= 0 or entry.length > MAX_WHEEL_BYTES:
        raise EnvironmentRefusal("wheel_inventory_invalid", entry.filename)
    if grant.digest != entry.digest or grant.length != entry.length:
        raise EnvironmentRefusal("wheel_grant_mismatch", entry.digest)
    path = grant.path
    try:
        info = path.lstat()
    except OSError as exc:
        raise EnvironmentRefusal("wheel_blob_unreadable", str(exc)) from exc
    if not path.is_absolute() or not stat.S_ISREG(info.st_mode) or info.st_size != entry.length:
        raise EnvironmentRefusal("wheel_blob_invalid", str(path))
    parts = entry.filename.removesuffix(".whl").split("-")
    if (
        len(parts) < 5
        or _name(parts[0]) != _name(entry.distribution)
        or parts[1].replace("_", "-") != entry.version.replace("_", "-")
    ):
        raise EnvironmentRefusal(
            "wheel_filename_mismatch", f"{entry.filename!r} does not name its distribution/version"
        )
    digest = _hash_file(path)
    if digest != entry.digest:
        raise EnvironmentRefusal("wheel_digest_mismatch", f"{path}: {digest}")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_WHEEL_MEMBERS:
                raise EnvironmentRefusal("wheel_member_cap", entry.filename)
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise EnvironmentRefusal("wheel_member_duplicate", entry.filename)
            expanded = sum(member.file_size for member in members)
            if expanded > min(MAX_WHEEL_EXPANDED_BYTES, entry.length * MAX_WHEEL_EXPANSION):
                raise EnvironmentRefusal(
                    "wheel_expansion_cap",
                    f"{entry.filename}: {expanded} expanded bytes from {entry.length}",
                )
            for member in members:
                pure = PurePosixPath(member.filename)
                if pure.is_absolute() or ".." in pure.parts or "\\" in member.filename:
                    raise EnvironmentRefusal("wheel_member_path", member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise EnvironmentRefusal("wheel_member_symlink", member.filename)
            metadata_names = [
                member.filename
                for member in members
                if member.filename.endswith(".dist-info/METADATA")
            ]
            wheel_names = [
                member.filename
                for member in members
                if member.filename.endswith(".dist-info/WHEEL")
            ]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise EnvironmentRefusal("wheel_metadata_ambiguous", entry.filename)
            metadata = _zip_read(archive, metadata_names[0])
            wheel_meta = _zip_read(archive, wheel_names[0])
    except (OSError, zipfile.BadZipFile) as exc:
        raise EnvironmentRefusal("wheel_zip_invalid", f"{entry.filename}: {exc}") from exc
    parsed = BytesParser().parsebytes(metadata)
    wheel_parsed = BytesParser().parsebytes(wheel_meta)
    distribution = _name(parsed.get("Name", ""))
    version = parsed.get("Version", "")
    tags = tuple(sorted(wheel_parsed.get_all("Tag", [])))
    if (
        distribution != _name(entry.distribution)
        or version != entry.version
        or tags != tuple(sorted(entry.tags))
        or not tags
    ):
        raise EnvironmentRefusal(
            "wheel_metadata_mismatch",
            f"{entry.filename}: measured {distribution}=={version} tags={tags}",
        )
    return _InspectedWheel(entry, distribution, version, tags, path)


def _verify_installed(venv: Path, expected: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    cfg = venv / "pyvenv.cfg"
    try:
        cfg_text = cfg.read_text()
    except OSError as exc:
        raise EnvironmentRefusal("venv_invalid", str(exc)) from exc
    if "include-system-site-packages = false" not in cfg_text.lower():
        raise EnvironmentRefusal("system_site_packages_enabled", str(cfg))
    sites = list(venv.glob("lib/python*/site-packages")) + list(venv.glob("Lib/site-packages"))
    if len(sites) != 1:
        raise EnvironmentRefusal("site_packages_ambiguous", str(venv))
    site = sites[0]
    if site.is_symlink():
        raise EnvironmentRefusal("installed_unowned_symlink", str(site))
    measured: dict[str, tuple[str, Path]] = {}
    for dist_info in site.glob("*.dist-info"):
        metadata_path = dist_info / "METADATA"
        if not metadata_path.is_file():
            raise EnvironmentRefusal("installed_metadata_missing", str(dist_info))
        parsed = BytesParser().parsebytes(metadata_path.read_bytes())
        name, version = _name(parsed.get("Name", "")), parsed.get("Version", "")
        if not name or name in measured:
            raise EnvironmentRefusal("installed_distribution_duplicate", name or str(dist_info))
        measured[name] = (version, dist_info)
    if set(measured) != set(expected):
        raise EnvironmentRefusal(
            "installed_distribution_set_mismatch",
            f"missing={sorted(set(expected) - set(measured))}, "
            f"extra={sorted(set(measured) - set(expected))}",
        )
    records: dict[str, str] = {}
    owned: set[Path] = set()
    for name, (expected_version, expected_record_digest) in expected.items():
        version, dist_info = measured[name]
        if version != expected_version:
            raise EnvironmentRefusal(
                "installed_version_mismatch", f"{name}: {version!r} != {expected_version!r}"
            )
        record = dist_info / "RECORD"
        distribution_owned = _verify_record(venv, site, record)
        if duplicate := owned & distribution_owned:
            raise EnvironmentRefusal("installed_record_duplicate_ownership", str(min(duplicate)))
        owned.update(distribution_owned)
        record_digest = _hash_file(record)
        if expected_record_digest and record_digest != expected_record_digest:
            raise EnvironmentRefusal("installed_record_digest_mismatch", name)
        records[name] = record_digest
    for path in site.rglob("*"):
        if path.is_symlink():
            raise EnvironmentRefusal("installed_unowned_symlink", str(path))
        if path.is_file() and path.resolve() not in owned:
            raise EnvironmentRefusal("installed_unrecorded_file", str(path))
    return records


def _strip_installer_paths(venv: Path) -> None:
    """Delete uv's local staging metadata; a host path is not environment data."""

    sites = list(venv.glob("lib/python*/site-packages")) + list(venv.glob("Lib/site-packages"))
    if len(sites) != 1:
        raise EnvironmentRefusal("site_packages_ambiguous", str(venv))
    site = sites[0]
    for dist_info in site.glob("*.dist-info"):
        generated = [
            path
            for name in ("direct_url.json", "uv_cache.json")
            if (path := dist_info / name).exists()
        ]
        if not generated:
            continue
        record = dist_info / "RECORD"
        try:
            with record.open(newline="") as stream:
                rows = list(csv.reader(stream))
            relative = {path.relative_to(site).as_posix() for path in generated}
            filtered = [row for row in rows if row and row[0] not in relative]
            if len(filtered) != len(rows) - len(generated):
                raise EnvironmentRefusal("installed_path_metadata_unrecorded", str(dist_info))
            filtered.sort(key=lambda row: row[0])
            for path in generated:
                path.unlink()
            temporary = record.with_suffix(".tmp")
            with temporary.open("w", newline="") as stream:
                csv.writer(stream, lineterminator="\n").writerows(filtered)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, record)
        except OSError as exc:
            raise EnvironmentRefusal("installed_path_metadata_cleanup_failed", str(exc)) from exc


def _strip_uv_venv_files(venv: Path) -> None:
    """Remove uv's unowned bootstrap before wheels can claim either path."""

    sites = list(venv.glob("lib/python*/site-packages")) + list(venv.glob("Lib/site-packages"))
    if len(sites) != 1:
        raise EnvironmentRefusal("site_packages_ambiguous", str(venv))
    for name in ("_virtualenv.pth", "_virtualenv.py"):
        path = sites[0] / name
        if path.exists():
            path.unlink()


def _verify_record(venv: Path, site: Path, record: Path) -> set[Path]:
    if not record.is_file():
        raise EnvironmentRefusal("installed_record_missing", str(record))
    root = venv.resolve()
    seen: set[Path] = set()
    with record.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 3 or not row[0]:
                raise EnvironmentRefusal("installed_record_invalid", f"{record}: {row!r}")
            unresolved = site / row[0]
            if unresolved.is_symlink():
                raise EnvironmentRefusal("installed_record_symlink", row[0])
            path = unresolved.resolve()
            if path in seen or (path != root and root not in path.parents):
                raise EnvironmentRefusal("installed_record_path", row[0])
            seen.add(path)
            if not path.is_file():
                raise EnvironmentRefusal("installed_record_entry_missing", row[0])
            if not row[1]:
                if path != record.resolve():
                    raise EnvironmentRefusal("installed_record_unhashed", row[0])
                continue
            algorithm, separator, encoded = row[1].partition("=")
            if algorithm != "sha256" or not separator:
                raise EnvironmentRefusal("installed_record_hash", row[1])
            actual = base64.urlsafe_b64encode(_hash_raw(path)).rstrip(b"=").decode()
            if actual != encoded or not row[2].isdigit() or path.stat().st_size != int(row[2]):
                raise EnvironmentRefusal("installed_record_mismatch", row[0])
    return seen


def _receipt(
    source: EnvironmentInput,
    spec: dict[str, Any],
    target: PlatformTarget,
    wheels: Sequence[_InspectedWheel],
    records: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "format": RECEIPT_FORMAT,
        "environment_spec_digest": source.environment_spec_digest,
        "endpoint_bundle_digest": spec["endpoint_bundle_digest"],
        "project_wheel_digest": spec["project_wheel_digest"],
        "wheelhouse_manifest_digest": spec["wheelhouse_manifest_digest"],
        "platform_target": target.document(),
        "distributions": [
            {
                "distribution": wheel.distribution,
                "version": wheel.version,
                "wheel_digest": wheel.entry.digest,
                "record_digest": records[wheel.distribution],
            }
            for wheel in wheels
        ],
    }


def _verify_host_target(target: PlatformTarget, python: Path, env: Mapping[str, str]) -> None:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        arch = machine
    if f"{system}/{arch}" != target.os_arch:
        raise EnvironmentRefusal("platform_os_arch_mismatch", f"host is {system}/{arch}")
    result = _run(
        [
            str(python),
            "-I",
            "-c",
            "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
        ],
        env,
        "platform_python_probe_failed",
    )
    abi = result.stdout.strip()
    if abi != target.python_abi:
        raise EnvironmentRefusal("platform_python_abi_mismatch", f"host is {abi}")
    if system == "linux":
        match = _GLIBC.fullmatch(target.libc)
        measured_name, measured_version = platform.libc_ver()
        if not match or measured_name != "glibc":
            raise EnvironmentRefusal("platform_libc_unsupported", target.libc)
        required = (int(match.group(1)), int(match.group(2)))
        measured = tuple(int(part) for part in measured_version.split(".")[:2])
        if measured < required:
            raise EnvironmentRefusal("platform_libc_mismatch", f"host glibc is {measured_version}")
    elif target.libc != "none":
        raise EnvironmentRefusal("platform_libc_mismatch", f"{system} requires libc=none")
    if target.accelerator_backend == "none" and target.accelerator_abi != "none":
        raise EnvironmentRefusal("platform_accelerator_invalid", "none backend requires none ABI")
    if target.accelerator_backend not in {"none", "cuda", "mps"}:
        raise EnvironmentRefusal("platform_accelerator_unsupported", target.accelerator_backend)


def _verify_installed_target(python: Path, target: PlatformTarget, env: Mapping[str, str]) -> None:
    if target.accelerator_backend == "none":
        return
    probe = (
        "import json,torch; "
        "print(json.dumps({'cuda':getattr(torch.version,'cuda',None),"
        "'cuda_available':torch.cuda.is_available(),"
        "'mps_available':bool(getattr(getattr(torch.backends,'mps',None),'is_available',lambda:False)())}))"
    )
    result = _run([str(python), "-I", "-c", probe], env, "accelerator_probe_failed")
    measured = json.loads(result.stdout)
    if target.accelerator_backend == "cuda":
        cuda = measured.get("cuda")
        abi = "cu" + str(cuda).replace(".", "") if cuda else ""
        if abi != target.accelerator_abi or not measured.get("cuda_available"):
            raise EnvironmentRefusal(
                "platform_accelerator_mismatch", f"installed torch reports cuda={cuda!r}"
            )
    elif not measured.get("mps_available") or target.accelerator_abi != "none":
        raise EnvironmentRefusal("platform_accelerator_mismatch", "installed torch has no MPS")


def _digest_field(spec_bytes: bytes, field: str) -> str:
    value = documents.read(spec_bytes, pb.EndpointEnvironmentSpec).get(field)
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise EnvironmentRefusal("environment_spec_invalid", f"{field} is not a digest")
    return value


def _sealed_env(base: Mapping[str, str]) -> dict[str, str]:
    erased = ("PYTHON", "PIP_", "UV_", "VIRTUAL_ENV", "CONDA")
    env = {key: value for key, value in base.items() if not key.startswith(erased)}
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_NO_CONFIG": "1",
            "UV_NO_CACHE": "1",
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    return env


def _verify_uv(uv: Path, env: Mapping[str, str]) -> None:
    try:
        info = uv.lstat()
    except OSError as exc:
        raise EnvironmentRefusal("uv_unreadable", str(exc)) from exc
    if (
        not uv.is_absolute()
        or not stat.S_ISREG(info.st_mode)
        or info.st_size != PINNED_UV_LENGTH
        or _hash_file(uv) != PINNED_UV_DIGEST
    ):
        raise EnvironmentRefusal("uv_identity_mismatch", str(uv))
    result = _run([str(uv), "--version"], env, "uv_version_unreadable")
    if result.stdout.strip() != PINNED_UV_VERSION:
        raise EnvironmentRefusal("uv_identity_mismatch", result.stdout.strip())


def _stage_wheels(
    wheels: Sequence[_InspectedWheel], wheelhouse: Path
) -> tuple[_InspectedWheel, ...]:
    """Copy the verified CAS bytes to their exact wheel filenames, then rehash the copies.

    CAS object paths need not end in ``.whl`` and must never enter an identity document.  uv,
    correctly, does require a wheel filename to apply wheel semantics.  The private staging
    wheelhouse is that path-only join; rehashing after the copy closes the source mutation window.
    """

    wheelhouse.mkdir(mode=0o700)
    staged: list[_InspectedWheel] = []
    for wheel in wheels:
        target = wheelhouse / wheel.entry.filename
        try:
            with wheel.path.open("rb") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=4 << 20)
        except OSError as exc:
            raise EnvironmentRefusal("wheel_stage_failed", str(exc)) from exc
        if target.stat().st_size != wheel.entry.length or _hash_file(target) != wheel.entry.digest:
            raise EnvironmentRefusal("wheel_stage_mismatch", wheel.entry.digest)
        staged.append(
            _InspectedWheel(wheel.entry, wheel.distribution, wheel.version, wheel.tags, target)
        )
    return tuple(staged)


def _run(argv: list[str], env: Mapping[str, str], code: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(argv, env=env, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise EnvironmentRefusal(code, str(exc)) from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise EnvironmentRefusal(code, detail)
    return result


def _zip_read(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_METADATA_BYTES:
        raise EnvironmentRefusal("wheel_metadata_cap", name)
    return archive.read(info)


def _hash_file(path: Path) -> str:
    return "sha256:" + _hash_raw(path).hex()


def _hash_raw(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    return digest.digest()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _name(value: str) -> str:
    return _NORMALIZE.sub("-", value).lower()


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


def _remove_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    root.chmod(root.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(root)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
