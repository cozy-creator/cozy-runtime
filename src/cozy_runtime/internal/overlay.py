"""Small endpoint overlays over the immutable base-worker-image Python environment.

The base worker image already owns Python, Torch/CUDA, cozy-runtime, TensorFS, and their
common dependencies. A placement therefore materializes only the endpoint project wheel
and optional endpoint-owned custom wheels. It never creates a venv, resolves a dependency,
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
from collections.abc import Sequence
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from cozy_runtime.internal import canonical

RECEIPT_FORMAT = "cozy.runtime.EndpointOverlayReceipt/1"
RECEIPT_NAME = "endpoint-overlay-receipt.json"
OVERLAY_DIR = "site-packages"
DEFAULT_BASE_MANIFEST = Path("/opt/cozy/base-runtime/WheelhouseManifest.json")
WHEELHOUSE_MANIFEST_FORMAT = "WheelhouseManifest/2"
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
_PLATFORM_TARGET_FIELDS = frozenset(
    {"accelerator_abi", "accelerator_backend", "libc", "os_arch", "python_abi"}
)


class OverlayRefusal(Exception):
    """A fail-closed endpoint-overlay verdict with a stable code."""

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
    wheels: tuple[Wheel, ...]
    base_distributions: tuple[tuple[str, str], ...]
    base_import_roots: tuple[tuple[str, tuple[str, ...]], ...]

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


@dataclass(frozen=True, slots=True)
class Input:
    environment_spec_digest: str
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
    reused: bool


@dataclass(frozen=True, slots=True)
class _Inspected:
    wheel: Wheel
    path: Path
    members: tuple[zipfile.ZipInfo, ...]
    import_roots: frozenset[str]


def materialize(
    source: Input,
    root: Path,
    *,
    python: Path = Path(sys.executable),
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    expected_receipt_digest: str = "",
) -> InstalledOverlay:
    """Verify and atomically publish only endpoint-owned packages."""

    _require_digest(source.environment_spec_digest, "environment_spec_digest")
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
    generations = root / "generations"
    final = generations / receipt_digest.removeprefix("sha256:")
    if final.exists():
        return open_generation(
            final,
            receipt_digest=receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
            base_manifest=base_manifest,
            reused=True,
            python=python,
        )

    inspected = _verify_grants(source, wheels, base)
    generations.mkdir(parents=True, exist_ok=True)
    staging = root / f".overlay-staging-{uuid.uuid4().hex}"
    try:
        site = staging / OVERLAY_DIR
        site.mkdir(parents=True, mode=0o700)
        _extract(inspected, site)
        # Endpoint code is small, but it should still pay compilation once at intake rather
        # than once per executor. The base image's much larger bytecode is already baked.
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
        _write_once(staging / RECEIPT_NAME, receipt_bytes)
        _make_read_only(staging)
        _fsync_dir(staging)
        try:
            os.rename(staging, final)
            final.chmod(0o555)
            reused = False
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _remove_tree(staging)
            reused = True
        _fsync_dir(generations)
        return open_generation(
            final,
            receipt_digest=receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
            base_manifest=base_manifest,
            reused=reused,
            python=python,
        )
    except Exception:
        if staging.exists():
            _remove_tree(staging)
        raise


def open_generation(
    generation: Path,
    *,
    receipt_digest: str,
    environment_spec_digest: str,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    reused: bool = True,
    python: Path = Path(sys.executable),
) -> InstalledOverlay:
    """Open a complete overlay without re-hashing its installed files.

    The receipt and base-family manifest are small identity documents. The endpoint wheel
    bytes were hashed before atomic publication; the official image's OCI digest already
    protects its base environment. Re-reading every installed file would add startup cost
    without adding an independent trust boundary.
    """

    _require_digest(receipt_digest, "receipt_digest")
    _require_digest(environment_spec_digest, "environment_spec_digest")
    try:
        receipt_bytes = (generation / RECEIPT_NAME).read_bytes()
    except OSError as exc:
        raise OverlayRefusal("overlay_receipt_absent", str(exc)) from exc
    if _sha(receipt_bytes) != receipt_digest:
        raise OverlayRefusal("overlay_receipt_corrupt", str(generation / RECEIPT_NAME))
    try:
        value = canonical.parse_canonical(receipt_bytes)
    except canonical.CanonicalError as exc:
        raise OverlayRefusal("overlay_receipt_invalid", str(exc)) from exc
    if not isinstance(value, dict) or set(value) != {
        "base_family_digest",
        "environment_spec_digest",
        "format",
        "overlay_wheels",
        "project_wheel_digest",
    }:
        raise OverlayRefusal("overlay_receipt_invalid", "wrong fields")
    if value.get("format") != RECEIPT_FORMAT:
        raise OverlayRefusal("overlay_receipt_invalid", "wrong format")
    if value.get("environment_spec_digest") != environment_spec_digest:
        raise OverlayRefusal("overlay_receipt_wrong_spec", environment_spec_digest)
    base_digest = value.get("base_family_digest")
    if not isinstance(base_digest, str):
        raise OverlayRefusal("overlay_receipt_invalid", "base family is not a digest")
    _read_base_manifest(base_manifest, base_digest)
    if generation.name != receipt_digest.removeprefix("sha256:"):
        raise OverlayRefusal("overlay_generation_wrong_key", str(generation))
    import_root = generation / OVERLAY_DIR
    if not import_root.is_dir():
        raise OverlayRefusal("overlay_import_root_absent", str(import_root))
    if generation.stat().st_mode & 0o222 or import_root.stat().st_mode & 0o222:
        raise OverlayRefusal("overlay_generation_mutable", str(generation))
    if not python.is_file():
        raise OverlayRefusal("overlay_python_absent", str(python))
    return InstalledOverlay(
        generation=generation,
        import_root=import_root,
        python=python,
        receipt_bytes=receipt_bytes,
        receipt_digest=receipt_digest,
        reused=reused,
    )


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
        "format",
        "platform_target",
        "wheels",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise OverlayRefusal("base_family_manifest_invalid", "wrong fields")
    if value["format"] != WHEELHOUSE_MANIFEST_FORMAT:
        raise OverlayRefusal("base_family_manifest_invalid", "wrong format")
    platform = value["platform_target"]
    if (
        not isinstance(platform, dict)
        or set(platform) != _PLATFORM_TARGET_FIELDS
        or not all(isinstance(item, str) and item for item in platform.values())
    ):
        raise OverlayRefusal("base_family_manifest_invalid", "invalid platform_target")
    distributions = _base_distribution_rows(value["base_distributions"])
    roots = _base_import_root_rows(value["base_import_roots"], dict(distributions))
    wheels = _base_wheel_rows(value["wheels"], dict(distributions), dict(roots))
    return BaseManifest(
        platform_target=tuple(sorted(platform.items())),
        wheels=wheels,
        base_distributions=distributions,
        base_import_roots=roots,
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
    source: Input, wheels: Sequence[Wheel], base: BaseManifest
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
            size = grant.path.stat().st_size
        except OSError as exc:
            raise OverlayRefusal("overlay_wheel_unreadable", str(grant.path)) from exc
        if size != wheel.length or _hash_file(grant.path) != wheel.digest:
            raise OverlayRefusal("overlay_wheel_digest_mismatch", wheel.digest)
        inspected.append(_inspect(wheel, grant.path, base))
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
    return tuple(inspected)


def _inspect(wheel: Wheel, path: Path, base: BaseManifest) -> _Inspected:
    filename = wheel.filename.removesuffix(".whl").split("-")
    if (
        len(filename) < 5
        or _name(filename[0]) != wheel.distribution
        or filename[1].replace("_", "-") != wheel.version.replace("_", "-")
    ):
        raise OverlayRefusal("wheel_filename_mismatch", wheel.filename)
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(archive.infolist())
            if not members or len(members) > MAX_WHEEL_MEMBERS:
                raise OverlayRefusal("overlay_wheel_member_count", wheel.filename)
            names: set[str] = set()
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
            return _Inspected(wheel, path, members, roots)
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
    roots: set[str] = set()
    for name in names:
        installed = _install_path(name)
        if installed is None:
            continue
        first = installed.parts[0]
        if first.endswith((".dist-info", ".egg-info", ".libs")) or first == "__pycache__":
            continue
        root = first.partition(".")[0]
        if valid_import_root(root):
            roots.add(root)
    return frozenset(roots)


def _base_import_collision(wheel: Wheel, root: str, providers: tuple[str, ...]) -> OverlayRefusal:
    return OverlayRefusal(
        "overlay_shadows_base_import",
        f"root={root} overlay_distribution={wheel.distribution} "
        f"image_providers={','.join(providers)}",
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
        "environment_spec_digest": source.environment_spec_digest,
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
