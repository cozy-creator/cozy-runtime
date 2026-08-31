"""Small package overlays over the immutable base-worker-image Python environment.

The base worker image already owns Python, Torch/CUDA, cozy-runtime, TensorFS, and their
common dependencies. A placement therefore materializes only the package project wheel
and optional package-owned custom wheels. It never creates a venv, resolves a dependency,
copies a base package, or re-hashes the base environment's RECORD tree.

The OCI image digest protects image-resident bytes. This module protects bytes that arrive
after boot: exact wheel grants are hashed once, inspected, extracted into private staging,
and atomically published. Reuse reads the small receipt and image base-family manifest; it
does not walk or hash the installed tree again.
"""

from __future__ import annotations

import base64
import compileall
import csv
import errno
import hashlib
import os
import py_compile
import re
import shutil
import stat
import sys
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, cast

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, parse_tag
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import canonical, native_wheel

RECEIPT_FORMAT = "cozy.runtime.EnvironmentReceipt/1"
CONTENT_IDENTITY_NAME = "package-overlay-content.json"
WHEEL_EXTRACTION_COMPLETE = ".complete"
OVERLAY_DIR = "site-packages"
DEFAULT_BASE_MANIFEST = Path("/opt/cozy/base-runtime/WheelhouseManifest.json")
WHEELHOUSE_MANIFEST_FORMAT = "WheelhouseManifest/1"
MAX_WHEEL_BYTES = 2 << 30
MAX_WHEEL_MEMBERS = 100_000
MAX_WHEEL_EXPANDED_BYTES = 8 << 30
MAX_WHEEL_EXPANSION = 64
MAX_METADATA_BYTES = 4 << 20

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NORMALIZE = re.compile(r"[-_.]+")
_NORMALIZED_DISTRIBUTION = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}$")
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,255}\.whl$")
_IMPORT_ROOT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WHEEL_TAG = re.compile(r"^[a-z0-9][a-z0-9._]*-[a-z0-9][a-z0-9._]*-[a-z0-9][a-z0-9._]*$")
_SAFE_MEMBER = re.compile(r"^[^/][^\x00]*$")
_NATIVE_SUFFIXES = frozenset({".dll", ".dylib", ".pyd", ".so"})
_BUILD_SOURCE_SUFFIXES = frozenset(
    {
        ".a",
        ".asm",
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".cxx",
        ".h",
        ".hip",
        ".hpp",
        ".o",
        ".obj",
        ".rs",
    }
)
_BUILD_INPUT_NAMES = frozenset(
    {
        "build.ninja",
        "cargo.toml",
        "cmakelists.txt",
        "makefile",
        "meson.build",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
    }
)
_PLATFORM_TARGET_FIELDS = frozenset(
    {"accelerator_abi", "accelerator_backend", "libc", "os_arch", "python_abi"}
)
_COMPATIBILITY_PROFILE_FIELDS = frozenset(
    {"accelerator_build", "os_cpu", "python_abi", "torch_release"}
)


class OverlayRefusal(Exception):
    """A fail-closed package-overlay verdict with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Wheel:
    distribution: str
    version: str
    filename: str
    digest: str
    length: int
    tags: tuple[str, ...]
    owner: str
    """Exactly ``project``, ``custom``, or image-owned ``base``."""
    import_roots: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BaseManifest:
    """The complete immutable Python ownership of one base worker image."""

    platform_target: tuple[tuple[str, str], ...]
    compatibility_profile: tuple[tuple[str, str], ...]
    minimum_cuda_version: str
    minimum_driver_version: str
    wheels: tuple[Wheel, ...]
    base_distributions: tuple[tuple[str, str], ...]
    base_import_roots: tuple[tuple[str, tuple[str, ...]], ...]
    ownership: str = "image"

    @property
    def distribution_versions(self) -> dict[str, str]:
        return dict(self.base_distributions)

    @property
    def root_providers(self) -> dict[str, tuple[str, ...]]:
        return dict(self.base_import_roots)


@dataclass(frozen=True, slots=True)
class BlobGrant:
    digest: str
    length: int
    path: Path
    verified: VerifiedFile | None = None


@dataclass(frozen=True, slots=True)
class VerifiedFile:
    """A same-boot receipt for one already-hashed immutable cache file."""

    digest: str
    length: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def record(cls, path: Path, digest: str, length: int) -> VerifiedFile:
        try:
            info = path.lstat()
        except OSError as exc:
            raise OverlayRefusal("overlay_wheel_unreadable", str(path)) from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size != length:
            raise OverlayRefusal("overlay_wheel_identity_changed", str(path))
        return cls(
            digest=digest,
            length=length,
            device=info.st_dev,
            inode=info.st_ino,
            mtime_ns=info.st_mtime_ns,
            ctime_ns=info.st_ctime_ns,
        )

    def matches(self, path: Path, digest: str, length: int) -> bool:
        try:
            info = path.lstat()
        except OSError:
            return False
        return (
            self.digest == digest
            and self.length == length
            and stat.S_ISREG(info.st_mode)
            and info.st_size == length
            and info.st_dev == self.device
            and info.st_ino == self.inode
            and info.st_mtime_ns == self.mtime_ns
            and info.st_ctime_ns == self.ctime_ns
        )


@dataclass(frozen=True, slots=True)
class Input:
    environment_digest: str
    base_family_digest: str
    project_wheel_digest: str
    wheels: tuple[Wheel, ...]
    grants: tuple[BlobGrant, ...]


@dataclass(frozen=True, slots=True)
class InstalledOverlay:
    generation: Path
    import_root: Path
    python: Path
    receipt_bytes: bytes
    receipt_digest: str
    content_digest: str
    reused: bool


@dataclass(frozen=True, slots=True)
class _Inspected:
    wheel: Wheel
    path: Path
    members: tuple[zipfile.ZipInfo, ...]
    import_roots: frozenset[str]
    requires_python: str | None
    requirements: tuple[Requirement, ...]
    native_members: tuple[str, ...]


def measure_wheel(
    path: Path,
    owner: str,
    base: BaseManifest,
    *,
    allow_native_custom: bool = False,
) -> tuple[Wheel, VerifiedFile]:
    """Derive one local wheel fact, then run the ordinary full wheel inspection once.

    Native custom wheels stay closed by default. The same-user local preparation command
    opts in because installing a package already authorizes its Python code to execute with
    that user's authority; the pod worker never opts in.
    """

    if owner not in {"project", "custom"} or not path.is_absolute():
        raise OverlayRefusal("overlay_wheel_invalid", str(path))
    try:
        info = path.lstat()
        distribution, version, _build, tags = parse_wheel_filename(path.name)
        with zipfile.ZipFile(path) as archive:
            roots = _import_roots({member.filename for member in archive.infolist()})
    except (OSError, InvalidWheelFilename, zipfile.BadZipFile) as exc:
        raise OverlayRefusal("overlay_wheel_invalid", path.name) from exc
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_WHEEL_BYTES:
        raise OverlayRefusal("overlay_wheel_invalid", path.name)
    wheel = Wheel(
        distribution=_name(str(distribution)),
        version=str(version),
        filename=path.name,
        digest=_hash_file(path),
        length=info.st_size,
        tags=tuple(sorted(str(tag) for tag in tags)),
        owner=owner,
        import_roots=tuple(sorted(roots)),
    )
    inspected = _inspect(wheel, path, base, allow_native_custom=allow_native_custom)
    return inspected.wheel, VerifiedFile.record(path, wheel.digest, wheel.length)


def expected_receipt_bytes(source: Input) -> bytes:
    """Canonical receipt implied by already-joined release documents.

    The verified-file intake already hashed and installed the wheel bytes. The acceptance
    observer needs only to prove that the small receipt it names is exactly the one those same
    EES/Bundle/RWS/WHM facts imply; it must not reopen or re-hash the wheels to do that.
    """

    return canonical.write(_receipt(source, source.wheels))


def materialize(
    source: Input,
    root: Path,
    *,
    python: Path = Path(sys.executable),
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    expected_receipt_digest: str = "",
    allow_native_custom: bool = False,
) -> InstalledOverlay:
    """Verify and atomically publish package content plus a placement receipt.

    Installed bytes are keyed only by the base family and sorted wheel digests.  The
    environment spec remains in the small placement receipt, so two specs with identical
    wheels join the same compiled content without weakening the receipt join.

    ``allow_native_custom`` belongs to the local same-user lane. Its default is the rented-pod
    boundary, where native dependency wheels require separate hardware-class proof.
    """

    _require_digest(source.environment_digest, "environment_digest")
    _require_digest(source.base_family_digest, "base_family_digest")
    _require_digest(source.project_wheel_digest, "project_wheel_digest")
    if expected_receipt_digest:
        _require_digest(expected_receipt_digest, "expected_receipt_digest")

    base = _read_base_manifest(base_manifest, source.base_family_digest)
    wheels = _verify_inventory(source, base)
    receipt_bytes = canonical.write(_receipt(source, wheels))
    receipt_digest = _sha(receipt_bytes)
    if expected_receipt_digest and receipt_digest != expected_receipt_digest:
        raise OverlayRefusal(
            "overlay_receipt_mismatch",
            f"derived {receipt_digest}, expected {expected_receipt_digest}",
        )
    content_bytes = canonical.write(_content_identity(source, wheels))
    content_digest = _sha(content_bytes)
    contents = root / "contents"
    wheel_pools = root / "wheel-pools"
    receipts = root / "receipts"
    final = contents / content_digest.removeprefix("sha256:")
    receipt_path = receipts / f"{receipt_digest.removeprefix('sha256:')}.json"
    if receipt_path.exists():
        return open_generation(
            root,
            receipt_digest=receipt_digest,
            environment_digest=source.environment_digest,
            base_manifest=base_manifest,
            reused=True,
            python=python,
        )
    reused = final.exists()
    inspected = (
        None
        if reused
        else _verify_grants(
            source,
            wheels,
            base,
            allow_native_custom=allow_native_custom,
        )
    )
    _require_owned_root(root)
    _remove_dead_staging(root)
    contents.mkdir(parents=True, exist_ok=True, mode=0o700)
    wheel_pools.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The executor receives one exact import-root path. Search without directory listing is
    # sufficient to reach it; write remains root-only and receipts remain fully private.
    contents.chmod(0o711)
    wheel_pools.chmod(0o700)
    receipts.chmod(0o700)
    if not reused:
        assert inspected is not None
        pools = {
            item.wheel.digest: _publish_wheel_pool(wheel_pools, source.base_family_digest, item)
            for item in inspected
        }
        staging = root / f".overlay-staging-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            site = staging / OVERLAY_DIR
            site.mkdir(parents=True, mode=0o700)
            _compose_wheel_pools(pools, wheels, site)
            # Package code is small, but it should still pay compilation once at intake
            # rather than once per executor. Base bytecode is already baked in the image.
            if not compileall.compile_dir(
                site,
                quiet=1,
                force=True,
                workers=1,
                invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
                stripdir=str(site),
                prependdir=str(final / OVERLAY_DIR),
            ):
                raise OverlayRefusal("overlay_bytecode_failed", str(site))
            _write_once(staging / CONTENT_IDENTITY_NAME, content_bytes)
            _make_read_only(staging)
            _fsync_dir(staging)
            try:
                os.rename(staging, final)
                final.chmod(0o555)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                _remove_tree(staging)
                reused = True
            _fsync_dir(contents)
        except Exception:
            if staging.exists():
                _remove_tree(staging)
            raise
    _publish_receipt(receipt_path, receipt_bytes, receipts)
    return open_generation(
        root,
        receipt_digest=receipt_digest,
        environment_digest=source.environment_digest,
        base_manifest=base_manifest,
        reused=reused,
        python=python,
    )


def open_generation(
    root: Path,
    *,
    receipt_digest: str,
    environment_digest: str,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    reused: bool = True,
    python: Path = Path(sys.executable),
) -> InstalledOverlay:
    """Open immutable content through one exact placement receipt without tree hashes.

    The receipt and base-family manifest are small identity documents. The package wheel
    bytes were hashed before atomic publication; the official image's OCI digest already
    protects its base environment. Re-reading every installed file would add startup cost
    without adding an independent trust boundary.
    """

    _require_digest(receipt_digest, "receipt_digest")
    _require_digest(environment_digest, "environment_digest")
    receipt_path = root / "receipts" / f"{receipt_digest.removeprefix('sha256:')}.json"
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as exc:
        raise OverlayRefusal("overlay_receipt_absent", str(exc)) from exc
    if _sha(receipt_bytes) != receipt_digest:
        raise OverlayRefusal("overlay_receipt_corrupt", str(receipt_path))
    value = parse_environment_receipt(receipt_bytes)
    if value.get("environment_digest") != environment_digest:
        raise OverlayRefusal("overlay_receipt_wrong_environment", environment_digest)
    base_digest = value.get("base_family_digest")
    if not isinstance(base_digest, str):
        raise OverlayRefusal("overlay_receipt_invalid", "base family is not a digest")
    _read_base_manifest(base_manifest, base_digest)
    content_digest = _content_digest_from_receipt(value)
    generation = root / "contents" / content_digest.removeprefix("sha256:")
    try:
        identity_bytes = (generation / CONTENT_IDENTITY_NAME).read_bytes()
    except OSError as exc:
        raise OverlayRefusal("overlay_content_absent", str(exc)) from exc
    if _sha(identity_bytes) != content_digest:
        raise OverlayRefusal("overlay_content_identity_corrupt", str(generation))
    try:
        identity = canonical.parse_canonical(identity_bytes)
    except canonical.CanonicalError as exc:
        raise OverlayRefusal("overlay_content_identity_invalid", str(exc)) from exc
    if identity != _content_identity_from_receipt(value):
        raise OverlayRefusal("overlay_content_identity_mismatch", str(generation))
    import_root = generation / OVERLAY_DIR
    if not import_root.is_dir():
        raise OverlayRefusal("overlay_import_root_absent", str(import_root))
    owner = os.geteuid()
    boundaries = (root, root / "contents", root / "receipts")
    if (
        any(path.stat().st_uid != owner or path.stat().st_mode & 0o022 for path in boundaries)
        or generation.stat().st_uid != owner
        or import_root.stat().st_uid != owner
        or receipt_path.stat().st_uid != owner
        or generation.stat().st_mode & 0o222
        or import_root.stat().st_mode & 0o222
        or receipt_path.stat().st_mode & 0o222
    ):
        raise OverlayRefusal("overlay_generation_mutable", str(generation))
    if not python.is_file():
        raise OverlayRefusal("overlay_python_absent", str(python))
    return InstalledOverlay(
        generation=generation,
        import_root=import_root,
        python=python,
        receipt_bytes=receipt_bytes,
        receipt_digest=receipt_digest,
        content_digest=content_digest,
        reused=reused,
    )


def parse_environment_receipt(raw: bytes) -> dict[str, Any]:
    """Read the one exact installed-environment observation."""

    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise OverlayRefusal("overlay_receipt_invalid", str(exc)) from exc
    fields = {
        "base_family_digest",
        "environment_digest",
        "format",
        "overlay_wheels",
        "project_wheel_digest",
    }
    if not isinstance(value, dict) or set(value) != fields or value.get("format") != RECEIPT_FORMAT:
        raise OverlayRefusal("overlay_receipt_invalid", "wrong fields or format")
    for field in ("base_family_digest", "environment_digest", "project_wheel_digest"):
        item = value.get(field)
        if not isinstance(item, str):
            raise OverlayRefusal("overlay_receipt_invalid", f"{field} is not a digest")
        _require_digest(item, field)
    rows = value.get("overlay_wheels")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise OverlayRefusal("overlay_receipt_invalid", "overlay_wheels is empty or oversized")
    distributions: list[str] = []
    digests: set[str] = set()
    projects = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "digest",
            "distribution",
            "owner",
            "version",
        }:
            raise OverlayRefusal("overlay_receipt_invalid", f"overlay_wheels[{index}] fields")
        digest = row.get("digest")
        distribution = row.get("distribution")
        owner = row.get("owner")
        version = row.get("version")
        if (
            not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or digest in digests
            or not isinstance(distribution, str)
            or not normalized_distribution(distribution)
            or not isinstance(version, str)
            or not valid_wheel_version(version)
            or owner not in {"custom", "project"}
        ):
            raise OverlayRefusal("overlay_receipt_invalid", f"overlay_wheels[{index}] values")
        digests.add(digest)
        distributions.append(distribution)
        projects += owner == "project"
    if distributions != sorted(set(distributions)) or projects != 1:
        raise OverlayRefusal(
            "overlay_receipt_invalid", "overlay wheels are not sorted unique with one project"
        )
    project_digest = value["project_wheel_digest"]
    if not any(row["owner"] == "project" and row["digest"] == project_digest for row in rows):
        raise OverlayRefusal("overlay_receipt_invalid", "project wheel digest is not the project")
    return value


def _read_base_manifest(path: Path, expected_digest: str) -> BaseManifest:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OverlayRefusal("base_family_manifest_unreadable", str(exc)) from exc
    return parse_base_manifest(raw, expected_digest)


def parse_base_manifest(raw: bytes, expected_digest: str) -> BaseManifest:
    """Strictly parse one canonical complete base-worker-image ownership manifest."""

    _require_digest(expected_digest, "base_family_digest")
    if _sha(raw) != expected_digest:
        raise OverlayRefusal("base_family_mismatch", f"manifest does not name {expected_digest}")
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise OverlayRefusal("base_family_manifest_invalid", str(exc)) from exc
    fields = {
        "base_distributions",
        "base_import_roots",
        "compatibility_profile",
        "format",
        "minimum_cuda_version",
        "minimum_driver_version",
        "platform_target",
        "wheels",
    }
    shapes = {frozenset(fields), frozenset({*fields, "ownership"})}
    if not isinstance(value, dict) or set(value) not in shapes:
        raise OverlayRefusal("base_family_manifest_invalid", "wrong fields")
    ownership = value.get("ownership", "image")
    if ownership not in {"image", "observed-local"}:
        raise OverlayRefusal("base_family_manifest_invalid", "invalid ownership")
    if value["format"] != WHEELHOUSE_MANIFEST_FORMAT:
        raise OverlayRefusal("base_family_manifest_invalid", "wrong format")
    platform = value["platform_target"]
    if (
        not isinstance(platform, dict)
        or set(platform) != _PLATFORM_TARGET_FIELDS
        or not all(isinstance(item, str) and item for item in platform.values())
    ):
        raise OverlayRefusal("base_family_manifest_invalid", "invalid platform_target")
    profile = value["compatibility_profile"]
    accelerator_build = profile.get("accelerator_build") if isinstance(profile, dict) else None
    cpu_profile = (
        accelerator_build == "cpu"
        and platform.get("accelerator_backend") == "none"
        and platform.get("accelerator_abi") == "none"
    )
    accelerator_profile = (
        isinstance(profile, dict)
        and accelerator_build == platform.get("accelerator_abi")
        and platform.get("accelerator_backend") == "cuda"
    )
    if (
        not isinstance(profile, dict)
        or set(profile) != _COMPATIBILITY_PROFILE_FIELDS
        or not all(isinstance(item, str) and item for item in profile.values())
        or not (cpu_profile or accelerator_profile)
        or profile["python_abi"] != platform["python_abi"]
        or profile["os_cpu"] != "linux-x86"
        or platform["os_arch"] != "linux/amd64"
        or (
            re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", profile["torch_release"]) is None
            and not (ownership == "observed-local" and profile["torch_release"] == "none")
        )
    ):
        raise OverlayRefusal("base_family_manifest_invalid", "invalid compatibility_profile")
    minimum_cuda = value["minimum_cuda_version"]
    minimum_driver = value["minimum_driver_version"]
    cuda_floors = (
        isinstance(minimum_cuda, str)
        and re.fullmatch(r"[0-9]+\.[0-9]+", minimum_cuda) is not None
        and isinstance(minimum_driver, str)
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", minimum_driver) is not None
    )
    cpu_floors = minimum_cuda == "none" and minimum_driver == "none"
    if not ((cpu_profile and cpu_floors) or (accelerator_profile and cuda_floors)):
        raise OverlayRefusal("base_family_manifest_invalid", "invalid CUDA or driver floor")
    distributions = _base_distribution_rows(value["base_distributions"])
    roots = _base_import_root_rows(value["base_import_roots"], dict(distributions))
    if ownership == "observed-local":
        if value["wheels"] != [] or not cpu_profile or profile["torch_release"] != "none":
            raise OverlayRefusal("base_family_manifest_invalid", "invalid observed-local base")
        wheels: tuple[Wheel, ...] = ()
    else:
        wheels = _base_wheel_rows(value["wheels"], dict(distributions), dict(roots))
    return BaseManifest(
        platform_target=tuple(sorted(platform.items())),
        compatibility_profile=tuple(sorted(profile.items())),
        minimum_cuda_version=minimum_cuda,
        minimum_driver_version=minimum_driver,
        wheels=wheels,
        base_distributions=distributions,
        base_import_roots=roots,
        ownership=ownership,
    )


def _base_distribution_rows(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise OverlayRefusal("base_family_manifest_invalid", "base_distributions is not a list")
    rows: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"distribution", "version"}:
            raise OverlayRefusal("base_family_manifest_invalid", "invalid distribution row")
        distribution = item["distribution"]
        version = item["version"]
        if (
            not isinstance(distribution, str)
            or not normalized_distribution(distribution)
            or distribution == "python-stdlib"
            or not isinstance(version, str)
            or not valid_wheel_version(version)
        ):
            raise OverlayRefusal("base_family_manifest_invalid", "invalid distribution row")
        rows.append((distribution, version))
    if rows != sorted(rows) or len({name for name, _ in rows}) != len(rows):
        raise OverlayRefusal(
            "base_family_manifest_invalid", "base_distributions is not sorted unique"
        )
    return tuple(rows)


def _base_import_root_rows(
    value: object, distributions: dict[str, str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, list) or not value:
        raise OverlayRefusal("base_family_manifest_invalid", "base_import_roots is not a list")
    rows: list[tuple[str, tuple[str, ...]]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"providers", "root"}:
            raise OverlayRefusal("base_family_manifest_invalid", "invalid import-root row")
        root = item["root"]
        providers = item["providers"]
        if (
            not isinstance(root, str)
            or not valid_import_root(root)
            or not isinstance(providers, list)
            or not providers
            or not all(
                isinstance(provider, str)
                and (provider == "python-stdlib" or normalized_distribution(provider))
                and (provider == "python-stdlib" or provider in distributions)
                for provider in providers
            )
            or providers != sorted(set(providers))
        ):
            raise OverlayRefusal("base_family_manifest_invalid", "invalid import-root row")
        rows.append((root, tuple(providers)))
    if rows != sorted(rows) or len({root for root, _ in rows}) != len(rows):
        raise OverlayRefusal(
            "base_family_manifest_invalid", "base_import_roots is not sorted unique"
        )
    return tuple(rows)


def _base_wheel_rows(
    value: object,
    distributions: dict[str, str],
    root_providers: dict[str, tuple[str, ...]],
) -> tuple[Wheel, ...]:
    if not isinstance(value, list) or not value:
        raise OverlayRefusal("base_family_manifest_invalid", "wheels is not a list")
    wheels: list[Wheel] = []
    for item in value:
        keys = {
            "digest",
            "distribution",
            "filename",
            "import_roots",
            "length",
            "tags",
            "version",
        }
        if not isinstance(item, dict) or set(item) != keys:
            raise OverlayRefusal("base_family_manifest_invalid", "invalid wheel row")
        distribution = item["distribution"]
        version = item["version"]
        digest = item["digest"]
        filename = item["filename"]
        length = item["length"]
        tags = item["tags"]
        import_roots = item["import_roots"]
        if (
            not isinstance(distribution, str)
            or not normalized_distribution(distribution)
            or not isinstance(version, str)
            or not valid_wheel_version(version)
            or distributions.get(distribution) != version
            or not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or not isinstance(filename, str)
            or not valid_wheel_filename(filename)
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            or length > MAX_WHEEL_BYTES
            or not isinstance(tags, list)
            or not tags
            or not all(isinstance(tag, str) and valid_wheel_tag(tag) for tag in tags)
            or tags != sorted(set(tags))
            or not isinstance(import_roots, list)
            or not all(isinstance(root, str) and valid_import_root(root) for root in import_roots)
            or import_roots != sorted(set(import_roots))
            or any(distribution not in root_providers.get(root, ()) for root in import_roots)
        ):
            raise OverlayRefusal(
                "base_family_manifest_invalid",
                f"invalid wheel row: distribution={distribution!r} version={version!r} "
                f"filename={filename!r}",
            )
        wheels.append(
            Wheel(
                distribution,
                version,
                filename,
                digest,
                length,
                tuple(tags),
                "base",
                tuple(import_roots),
            )
        )
    if [wheel.distribution for wheel in wheels] != sorted(
        wheel.distribution for wheel in wheels
    ) or len({wheel.distribution for wheel in wheels}) != len(wheels):
        raise OverlayRefusal("base_family_manifest_invalid", "wheels is not sorted unique")
    return tuple(wheels)


def _verify_inventory(source: Input, base: BaseManifest) -> tuple[Wheel, ...]:
    if not source.wheels or len(source.wheels) > 128:
        raise OverlayRefusal("overlay_wheel_set_invalid", str(len(source.wheels)))
    names: set[str] = set()
    digests: set[str] = set()
    projects = 0
    ordered: list[Wheel] = []
    for wheel in source.wheels:
        name = _name(wheel.distribution)
        _require_digest(wheel.digest, "wheel.digest")
        if (
            not name
            or not normalized_distribution(wheel.distribution)
            or not valid_wheel_version(wheel.version)
            or not valid_wheel_filename(wheel.filename)
            or wheel.length <= 0
            or wheel.length > MAX_WHEEL_BYTES
            or not wheel.tags
            or not all(valid_wheel_tag(tag) for tag in wheel.tags)
            or tuple(sorted(set(wheel.tags))) != wheel.tags
            or wheel.owner not in {"project", "custom"}
        ):
            raise OverlayRefusal("overlay_wheel_invalid", wheel.filename)
        if wheel.owner == "project" and wheel.tags != ("py3-none-any",):
            raise OverlayRefusal(
                "overlay_project_wheel_not_pure",
                f"{wheel.filename} tags={','.join(wheel.tags)}; project must be py3-none-any",
            )
        base_version = base.distribution_versions.get(name)
        if base_version is not None:
            raise OverlayRefusal(
                "overlay_shadows_base_distribution",
                f"distribution={name} overlay_version={wheel.version} image_version={base_version}",
            )
        if wheel.import_roots != tuple(sorted(set(wheel.import_roots))) or not all(
            valid_import_root(root) for root in wheel.import_roots
        ):
            raise OverlayRefusal("overlay_wheel_invalid", wheel.filename)
        _require_compatible_tags(wheel, base)
        for root in wheel.import_roots:
            providers = base.root_providers.get(root)
            if providers is not None:
                raise _base_import_collision(wheel, root, providers)
        if name in names or wheel.digest in digests:
            raise OverlayRefusal("overlay_wheel_duplicate", name)
        names.add(name)
        digests.add(wheel.digest)
        projects += wheel.owner == "project"
        ordered.append(wheel)
    if projects != 1 or source.project_wheel_digest not in {
        row.digest for row in ordered if row.owner == "project"
    }:
        raise OverlayRefusal("overlay_project_wheel_mismatch", source.project_wheel_digest)
    return tuple(sorted(ordered, key=lambda row: row.distribution))


def _verify_grants(
    source: Input,
    wheels: Sequence[Wheel],
    base: BaseManifest,
    *,
    allow_native_custom: bool,
) -> tuple[_Inspected, ...]:
    grants = {grant.digest: grant for grant in source.grants}
    expected = {wheel.digest for wheel in wheels}
    if len(grants) != len(source.grants) or set(grants) != expected:
        raise OverlayRefusal(
            "overlay_grant_set_mismatch",
            f"missing={sorted(expected - set(grants))} extra={sorted(set(grants) - expected)}",
        )
    inspected: list[_Inspected] = []
    for wheel in wheels:
        grant = grants[wheel.digest]
        if grant.length != wheel.length:
            raise OverlayRefusal("overlay_grant_length_mismatch", wheel.digest)
        try:
            info = grant.path.lstat()
        except OSError as exc:
            raise OverlayRefusal("overlay_wheel_unreadable", str(grant.path)) from exc
        verified = grant.verified
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size != wheel.length
            or verified is None
            or not verified.matches(grant.path, wheel.digest, wheel.length)
        ):
            # The trusted local proof seat has no acquisition receipt, so it verifies here.
            # Production acquisition always supplies one and therefore never hashes a wheel
            # a second time after fetch_exact already admitted it.
            if info.st_size != wheel.length or _hash_file(grant.path) != wheel.digest:
                raise OverlayRefusal("overlay_wheel_digest_mismatch", wheel.digest)
        elif verified.digest != wheel.digest:
            raise OverlayRefusal("overlay_wheel_digest_mismatch", wheel.digest)
        inspected.append(
            _inspect(
                wheel,
                grant.path,
                base,
                allow_native_custom=allow_native_custom,
            )
        )
    root_owners: dict[str, str] = {}
    for item in inspected:
        for root in item.import_roots:
            prior = root_owners.get(root)
            if prior is not None:
                raise OverlayRefusal(
                    "overlay_import_root_duplicate",
                    f"root={root} distributions={prior},{item.wheel.distribution}",
                )
            root_owners[root] = item.wheel.distribution
    _verify_python_requirements(inspected, base)
    return tuple(inspected)


def _inspect(
    wheel: Wheel,
    path: Path,
    base: BaseManifest,
    *,
    allow_native_custom: bool = False,
) -> _Inspected:
    try:
        filename_name, filename_version, _build, filename_tags = parse_wheel_filename(
            wheel.filename
        )
    except InvalidWheelFilename as exc:
        raise OverlayRefusal("wheel_filename_mismatch", wheel.filename) from exc
    declared_tags: set[Tag] = set()
    for tag in wheel.tags:
        declared_tags.update(parse_tag(tag))
    if (
        _name(filename_name) != wheel.distribution
        or filename_version != Version(wheel.version)
        or filename_tags != declared_tags
    ):
        raise OverlayRefusal("wheel_filename_mismatch", wheel.filename)
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(archive.infolist())
            if not members or len(members) > MAX_WHEEL_MEMBERS:
                raise OverlayRefusal("overlay_wheel_member_count", wheel.filename)
            names: set[str] = set()
            native_members: list[str] = []
            expanded = 0
            for member in members:
                name = member.filename
                posix = PurePosixPath(name)
                if (
                    not _SAFE_MEMBER.fullmatch(name)
                    or "\\" in name
                    or posix.is_absolute()
                    or ".." in posix.parts
                    or name in names
                ):
                    raise OverlayRefusal("overlay_wheel_member_invalid", name)
                names.add(name)
                expanded += member.file_size
                if (
                    expanded > MAX_WHEEL_EXPANDED_BYTES
                    or (member.compress_size == 0 and member.file_size > 0)
                    or (
                        member.compress_size > 0
                        and member.file_size > member.compress_size * MAX_WHEEL_EXPANSION
                    )
                ):
                    raise OverlayRefusal("overlay_wheel_expansion_cap", wheel.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise OverlayRefusal("overlay_wheel_symlink", name)
                if not member.is_dir() and PurePosixPath(name).suffix == ".pth":
                    raise OverlayRefusal("overlay_pth_forbidden", name)
                if member.is_dir():
                    continue
                installed = _install_path(name)
                if installed is None and ".data/" in name:
                    raise OverlayRefusal("overlay_wheel_install_scheme_forbidden", name)
                basename = posix.name.lower()
                suffix = posix.suffix.lower()
                project_metadata = (
                    wheel.owner == "project"
                    and len(posix.parts) == 1
                    and basename in {"pyproject.toml", "uv.lock"}
                )
                if (
                    basename in _BUILD_INPUT_NAMES or suffix in _BUILD_SOURCE_SUFFIXES
                ) and not project_metadata:
                    raise OverlayRefusal("overlay_native_build_input_forbidden", name)
                magic = archive.open(member).read(4)
                native = suffix in _NATIVE_SUFFIXES or magic == b"\x7fELF"
                if wheel.owner == "project" and native:
                    raise OverlayRefusal("overlay_project_wheel_native", name)
                if native:
                    assert installed is not None
                    native_members.append(installed.as_posix())
            dist_infos = sorted(
                name
                for name in names
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            )
            if len(dist_infos) != 1:
                raise OverlayRefusal("overlay_wheel_metadata_missing", wheel.filename)
            wheel_infos = sorted(
                name for name in names if name.count("/") == 1 and name.endswith(".dist-info/WHEEL")
            )
            if len(wheel_infos) != 1:
                raise OverlayRefusal("overlay_wheel_metadata_missing", wheel.filename)
            metadata = archive.read(dist_infos[0])
            if len(metadata) > MAX_METADATA_BYTES:
                raise OverlayRefusal("overlay_wheel_metadata_oversized", wheel.filename)
            parsed = BytesParser().parsebytes(metadata)
            parsed_wheel = BytesParser().parsebytes(archive.read(wheel_infos[0]))
            if (
                _name(parsed.get("Name", "")) != wheel.distribution
                or parsed.get("Version") != wheel.version
                or tuple(sorted(parsed_wheel.get_all("Tag", []))) != wheel.tags
            ):
                raise OverlayRefusal("overlay_wheel_metadata_mismatch", wheel.filename)
            root_is_pure = parsed_wheel.get("Root-Is-Purelib", "").strip().lower()
            if wheel.owner == "project" and root_is_pure != "true":
                raise OverlayRefusal(
                    "overlay_project_wheel_not_pure", "Root-Is-Purelib is not true"
                )
            if native_members and root_is_pure != "false":
                raise OverlayRefusal(
                    "overlay_native_wheel_metadata_invalid",
                    f"{wheel.filename} carries native members but Root-Is-Purelib is not false",
                )
            if (
                wheel.owner == "custom"
                and native_members
                and (not allow_native_custom or base.ownership != "observed-local")
            ):
                raise OverlayRefusal(
                    "overlay_custom_wheel_native_unsupported",
                    f"{wheel.filename}: custom native wheels are not admitted yet",
                )
            if native_members and any(
                tag.interpreter.startswith("py") or tag.abi == "none" or tag.platform == "any"
                for tag in declared_tags
            ):
                raise OverlayRefusal(
                    "overlay_native_wheel_tag_not_concrete",
                    f"{wheel.filename}: native custom wheels require concrete CPython ABI and "
                    "platform tags",
                )
            record_name = dist_infos[0].removesuffix("METADATA") + "RECORD"
            if record_name not in names:
                raise OverlayRefusal("overlay_wheel_record_missing", wheel.filename)
            _verify_record(archive, record_name, names)
            roots = _import_roots(names)
            measured_roots = tuple(sorted(roots))
            if wheel.import_roots != measured_roots:
                raise OverlayRefusal(
                    "overlay_wheel_import_roots_mismatch",
                    f"distribution={wheel.distribution} declared={list(wheel.import_roots)} "
                    f"measured={list(measured_roots)}",
                )
            for root in measured_roots:
                providers = base.root_providers.get(root)
                if providers is not None:
                    raise _base_import_collision(wheel, root, providers)
            requires_python = parsed.get("Requires-Python")
            requirements: list[Requirement] = []
            try:
                for value in parsed.get_all("Requires-Dist", []):
                    requirements.append(Requirement(value))
            except InvalidRequirement as exc:
                raise OverlayRefusal(
                    "overlay_requirement_invalid", f"{wheel.distribution}: {exc}"
                ) from exc
            return _Inspected(
                wheel,
                path,
                members,
                roots,
                requires_python,
                tuple(requirements),
                tuple(sorted(native_members)),
            )
    except OverlayRefusal:
        raise
    except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise OverlayRefusal("overlay_wheel_invalid", wheel.filename) from exc


def _verify_record(archive: zipfile.ZipFile, record_name: str, names: set[str]) -> None:
    raw = archive.read(record_name)
    try:
        rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    except (UnicodeError, csv.Error) as exc:
        raise OverlayRefusal("overlay_wheel_record_invalid", record_name) from exc
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in recorded or row[0] not in names:
            raise OverlayRefusal("overlay_wheel_record_invalid", record_name)
        recorded.add(row[0])
        if row[0] == record_name:
            if row[1] or row[2]:
                raise OverlayRefusal("overlay_wheel_record_invalid", record_name)
            continue
        if not row[1] or not row[2].isdigit():
            raise OverlayRefusal("overlay_wheel_record_unhashed", row[0])
        algorithm, separator, encoded = row[1].partition("=")
        payload = archive.read(row[0])
        got = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        if algorithm != "sha256" or not separator or encoded != got or int(row[2]) != len(payload):
            raise OverlayRefusal("overlay_wheel_record_mismatch", row[0])
    files = {name for name in names if not name.endswith("/")}
    if recorded != files:
        raise OverlayRefusal("overlay_wheel_record_incomplete", record_name)


def _import_roots(names: set[str]) -> frozenset[str]:
    """Measure Python import roots without mistaking wheel data for modules.

    A wheel may deliberately install repository metadata beside its packages (the package
    project carries ``package.toml``).  A top-level filename is importable only when it is a
    Python or native module; a directory is a package root only when the wheel installs an
    importable member below it.  The files are still extracted -- this function classifies
    ownership only.
    """

    roots: set[str] = set()
    for name in names:
        installed = _install_path(name)
        if installed is None:
            continue
        if len(installed.parts) == 1:
            filename = installed.name
            if not filename.endswith((".py", ".pyi", *_NATIVE_SUFFIXES)):
                continue
        elif not installed.name.endswith((".py", ".pyi", *_NATIVE_SUFFIXES)):
            continue
        first = installed.parts[0]
        if first.endswith((".dist-info", ".egg-info", ".libs")) or first == "__pycache__":
            continue
        root = first.removesuffix(".py").removesuffix(".pyi").partition(".")[0]
        if valid_import_root(root):
            roots.add(root)
    return frozenset(roots)


def _base_import_collision(wheel: Wheel, root: str, providers: tuple[str, ...]) -> OverlayRefusal:
    return OverlayRefusal(
        "overlay_shadows_base_import",
        f"root={root} overlay_distribution={wheel.distribution} "
        f"image_providers={','.join(providers)}",
    )


def _target_python(base: BaseManifest) -> tuple[int, int]:
    target = dict(base.platform_target)
    match = re.fullmatch(r"cp([0-9])([0-9]{2})", target.get("python_abi", ""))
    if match is None:
        raise OverlayRefusal(
            "base_family_manifest_invalid", "python_abi is not the cpNNN profile spelling"
        )
    return int(match.group(1)), int(match.group(2))


def _target_platforms(base: BaseManifest) -> frozenset[str]:
    target = dict(base.platform_target)
    os_arch = target.get("os_arch")
    libc = target.get("libc")
    if os_arch != "linux/amd64":
        raise OverlayRefusal(
            "overlay_platform_unsupported", f"semantic wheel admission has no seat for {os_arch}"
        )
    match = re.fullmatch(r"glibc([0-9]+)\.([0-9]+)", libc or "")
    if match is None or int(match.group(1)) != 2:
        raise OverlayRefusal(
            "base_family_manifest_invalid", f"linux/x86 base has unsupported libc {libc!r}"
        )
    minor = int(match.group(2))
    platforms = {"any", "linux_x86_64"}
    platforms.update(f"manylinux_2_{floor}_x86_64" for floor in range(5, minor + 1))
    if minor >= 17:
        platforms.add("manylinux2014_x86_64")
    if minor >= 12:
        platforms.add("manylinux2010_x86_64")
    if minor >= 5:
        platforms.add("manylinux1_x86_64")
    return frozenset(platforms)


def _compatible_tags(base: BaseManifest) -> frozenset[Tag]:
    major, minor = _target_python(base)
    interpreter = f"cp{major}{minor}"
    platforms = _target_platforms(base)
    tags: set[Tag] = set()
    for platform in platforms:
        tags.add(Tag(interpreter, interpreter, platform))
        tags.add(Tag(interpreter, "none", platform))
        for stable_minor in range(2, minor + 1):
            tags.add(Tag(f"cp{major}{stable_minor}", "abi3", platform))
        tags.add(Tag(f"py{major}{minor}", "none", platform))
        tags.add(Tag(f"py{major}", "none", platform))
    return frozenset(tags)


def _require_compatible_tags(wheel: Wheel, base: BaseManifest) -> None:
    declared: set[Tag] = set()
    try:
        for value in wheel.tags:
            declared.update(parse_tag(value))
    except ValueError as exc:
        raise OverlayRefusal("overlay_wheel_tag_invalid", wheel.filename) from exc
    if declared.isdisjoint(_compatible_tags(base)):
        raise OverlayRefusal(
            "overlay_wheel_platform_incompatible",
            f"distribution={wheel.distribution} tags={','.join(wheel.tags)} "
            f"target={dict(base.platform_target)}",
        )


def _marker_environment(base: BaseManifest) -> dict[str, str]:
    major, minor = _target_python(base)
    environment = cast(dict[str, str], default_environment())
    environment.update(
        {
            "implementation_name": "cpython",
            "implementation_version": f"{major}.{minor}.0",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "python_full_version": f"{major}.{minor}.0",
            "python_version": f"{major}.{minor}",
            "sys_platform": "linux",
        }
    )
    return environment


def _verify_python_requirements(inspected: Sequence[_Inspected], base: BaseManifest) -> None:
    environment = _marker_environment(base)
    python = Version(environment["python_full_version"])
    versions = base.distribution_versions
    versions.update({item.wheel.distribution: item.wheel.version for item in inspected})
    for item in inspected:
        if item.requires_python:
            try:
                supported = SpecifierSet(item.requires_python).contains(python, prereleases=True)
            except InvalidSpecifier as exc:
                raise OverlayRefusal(
                    "overlay_requires_python_invalid",
                    f"{item.wheel.distribution}: {item.requires_python}",
                ) from exc
            if not supported:
                raise OverlayRefusal(
                    "overlay_python_incompatible",
                    f"distribution={item.wheel.distribution} "
                    f"requires_python={item.requires_python} "
                    f"target={python}",
                )
        for requirement in item.requirements:
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            name = _name(requirement.name)
            installed = versions.get(name)
            if installed is None:
                raise OverlayRefusal(
                    "overlay_dependency_missing",
                    f"distribution={item.wheel.distribution} requirement={requirement}",
                )
            try:
                accepted = not requirement.specifier or requirement.specifier.contains(
                    Version(installed), prereleases=True
                )
            except InvalidVersion as exc:
                raise OverlayRefusal(
                    "base_family_manifest_invalid", f"{name} has invalid version {installed!r}"
                ) from exc
            if not accepted:
                raise OverlayRefusal(
                    "overlay_dependency_base_conflict",
                    f"distribution={item.wheel.distribution} requirement={requirement} "
                    f"base_or_overlay_version={installed}",
                )


def _extract(wheels: Sequence[_Inspected], site: Path) -> None:
    written: set[Path] = set()
    for inspected in wheels:
        with zipfile.ZipFile(inspected.path) as archive:
            for member in inspected.members:
                if member.is_dir():
                    continue
                relative = _install_path(member.filename)
                if relative is None:
                    continue
                target = site / relative
                if target in written:
                    raise OverlayRefusal("overlay_install_path_collision", relative.as_posix())
                written.add(target)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with archive.open(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1 << 20)
                target.chmod(0o500 if target.suffix in {".so", ".pyd", ".dll"} else 0o400)


def _wheel_pool_digest(base_family_digest: str, wheel_digest: str) -> str:
    """Name a private extraction cache without inventing a portable document format."""

    preimage = (
        b"cozy.runtime.wheel-extraction\0"
        + base_family_digest.encode("ascii")
        + b"\0"
        + wheel_digest.encode("ascii")
    )
    return _sha(preimage)


def _publish_wheel_pool(pools: Path, base_family_digest: str, inspected: _Inspected) -> Path:
    """Extract one verified wheel once under the image/runtime extraction semantics."""

    pool_digest = _wheel_pool_digest(base_family_digest, inspected.wheel.digest)
    final = pools / pool_digest.removeprefix("sha256:")
    if final.exists():
        marker = final / WHEEL_EXTRACTION_COMPLETE
        try:
            marker_info = marker.lstat()
            complete = marker.read_bytes()
        except OSError as exc:
            raise OverlayRefusal("overlay_wheel_pool_invalid", str(exc)) from exc
        if (
            complete
            or not stat.S_ISREG(marker_info.st_mode)
            or marker_info.st_uid != os.geteuid()
            or marker_info.st_mode & 0o222
            or final.stat().st_uid != os.geteuid()
            or not (final / "tree").is_dir()
        ):
            raise OverlayRefusal("overlay_wheel_pool_invalid", str(final))
        return final / "tree"
    staging = pools.parent / f".wheel-staging-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        tree = staging / "tree"
        tree.mkdir(parents=True, mode=0o700)
        _extract((inspected,), tree)
        if inspected.native_members:
            try:
                native_wheel.inspect_tree(tree, inspected.native_members)
            except native_wheel.NativeWheelRefusal as exc:
                raise OverlayRefusal(exc.code, exc.detail) from exc
        _write_once(staging / WHEEL_EXTRACTION_COMPLETE, b"")
        _make_read_only(staging)
        _fsync_dir(staging)
        try:
            os.rename(staging, final)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _remove_tree(staging)
        _fsync_dir(pools)
    except Exception:
        if staging.exists():
            _remove_tree(staging)
        raise
    return final / "tree"


def _compose_wheel_pools(pools: Mapping[str, Path], wheels: Sequence[Wheel], site: Path) -> None:
    """Compose one content generation with hardlinks to immutable per-wheel trees."""

    written: set[Path] = set()
    for wheel in sorted(wheels, key=lambda item: item.digest):
        pool = pools[wheel.digest]
        if not pool.is_dir():
            raise OverlayRefusal("overlay_wheel_pool_invalid", str(pool))
        for source in sorted(pool.rglob("*")):
            relative = source.relative_to(pool)
            target = site / relative
            if source.is_symlink():
                raise OverlayRefusal("overlay_wheel_pool_invalid", str(source))
            if source.is_dir():
                if target.exists() and not target.is_dir():
                    raise OverlayRefusal("overlay_install_path_collision", relative.as_posix())
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                continue
            if target in written or target.exists():
                raise OverlayRefusal("overlay_install_path_collision", relative.as_posix())
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.link(source, target)
            written.add(target)


def _install_path(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    first = path.parts[0]
    if first.endswith(".data"):
        if len(path.parts) < 3 or path.parts[1] not in {"purelib", "platlib"}:
            return None
        return PurePosixPath(*path.parts[2:])
    return path


def _receipt(source: Input, wheels: Sequence[Wheel]) -> dict[str, Any]:
    return {
        "base_family_digest": source.base_family_digest,
        "environment_digest": source.environment_digest,
        "format": RECEIPT_FORMAT,
        "overlay_wheels": [
            {
                "digest": wheel.digest,
                "distribution": wheel.distribution,
                "owner": wheel.owner,
                "version": wheel.version,
            }
            for wheel in wheels
        ],
        "project_wheel_digest": source.project_wheel_digest,
    }


def _content_identity(source: Input, wheels: Sequence[Wheel]) -> dict[str, Any]:
    return {
        "base_family_digest": source.base_family_digest,
        "wheel_digests": sorted(wheel.digest for wheel in wheels),
    }


def _content_identity_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    rows = receipt.get("overlay_wheels")
    if not isinstance(rows, list) or not rows:
        raise OverlayRefusal("overlay_receipt_invalid", "overlay_wheels is empty")
    digests: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "digest",
            "distribution",
            "owner",
            "version",
        }:
            raise OverlayRefusal("overlay_receipt_invalid", "overlay wheel row is not closed")
        digest = row.get("digest")
        if not isinstance(digest, str):
            raise OverlayRefusal("overlay_receipt_invalid", "overlay wheel digest is absent")
        _require_digest(digest, "overlay_wheels.digest")
        digests.append(digest)
    if len(set(digests)) != len(digests):
        raise OverlayRefusal("overlay_receipt_invalid", "overlay wheel digests repeat")
    base = receipt.get("base_family_digest")
    if not isinstance(base, str):
        raise OverlayRefusal("overlay_receipt_invalid", "base family is absent")
    _require_digest(base, "base_family_digest")
    return {
        "base_family_digest": base,
        "wheel_digests": sorted(digests),
    }


def _content_digest_from_receipt(receipt: Mapping[str, Any]) -> str:
    return _sha(canonical.write(_content_identity_from_receipt(receipt)))


def _require_owned_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = root.lstat()
    if not root.is_absolute() or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise OverlayRefusal("overlay_root_owner_mismatch", str(root))
    root.chmod(0o711)


def _remove_dead_staging(root: Path) -> None:
    """Remove only staging owned by a process identity that is observably gone."""

    for prefix in (".overlay-staging-", ".wheel-staging-"):
        for path in root.glob(prefix + "*"):
            remainder = path.name.removeprefix(prefix)
            pid_raw, separator, _nonce = remainder.partition("-")
            if not separator or not pid_raw.isdigit():
                continue
            pid = int(pid_raw)
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                info = path.lstat()
                if (
                    stat.S_ISDIR(info.st_mode)
                    and not stat.S_ISLNK(info.st_mode)
                    and info.st_uid == os.geteuid()
                ):
                    _remove_tree(path)
            except PermissionError:
                continue


def _publish_receipt(path: Path, payload: bytes, receipts: Path) -> None:
    temporary = receipts / f".receipt-{uuid.uuid4().hex}"
    _write_once(temporary, payload)
    temporary.chmod(0o400)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise OverlayRefusal("overlay_receipt_conflict", str(path)) from None
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_dir(receipts)


def _require_digest(value: str, name: str) -> None:
    if not _DIGEST.fullmatch(value):
        raise OverlayRefusal("overlay_digest_invalid", f"{name}={value!r}")


def _name(value: str) -> str:
    return _NORMALIZE.sub("-", value).lower()


def normalized_distribution(value: str) -> bool:
    return _NORMALIZED_DISTRIBUTION.fullmatch(value) is not None and _name(value) == value


def valid_wheel_version(value: str) -> bool:
    return _VERSION.fullmatch(value) is not None


def valid_wheel_filename(value: str) -> bool:
    return _FILENAME.fullmatch(value) is not None and PurePosixPath(value).name == value


def valid_import_root(value: str) -> bool:
    return _IMPORT_ROOT.fullmatch(value) is not None


def valid_wheel_tag(value: str) -> bool:
    return _WHEEL_TAG.fullmatch(value) is not None


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    measured = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b""):
                measured.update(chunk)
    except OSError as exc:
        raise OverlayRefusal("overlay_wheel_unreadable", str(path)) from exc
    return "sha256:" + measured.hexdigest()


def _write_once(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_symlink():
            mode = path.stat().st_mode
            path.chmod(0o555 if stat.S_ISDIR(mode) or mode & 0o111 else 0o444)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        if not item.is_symlink():
            item.chmod(item.stat().st_mode | stat.S_IWUSR)
    path.chmod(path.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(path)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
