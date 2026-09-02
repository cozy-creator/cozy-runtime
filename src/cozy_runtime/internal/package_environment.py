"""Per-package uv environments over the immutable base-worker-image Python environment.

Runtime observes the actual interpreter's standard library and narrow protected platform families:
Python, cozy-runtime, msgspec, TensorFS, and, when present, Torch/Triton/CUDA/NVIDIA. A placement
creates one ordinary uv venv containing only the package project wheel and package-owned dependency
wheels. It never re-resolves the lock, copies a protected package, or re-hashes the base
environment's RECORD tree. The one deliberate exception is cr-070's boot-install slow path: an
image-carried distribution whose observed version falls outside the wheels' declared ranges is
installed in range from the index, alone (`--no-deps`), logged, and re-verified on every open.

The OCI image digest protects image-resident bytes. This module protects bytes that arrive
after boot: exact wheel grants are hashed once, inspected, installed by uv into private staging,
and atomically published. Reuse reads only the small receipt; actual base compatibility is
re-observed before admission and Runtime does not walk or hash the installed tree again.
"""

from __future__ import annotations

import base64
import csv
import errno
import hashlib
import json
import os
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
from typing import Any, cast

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, parse_tag
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import base_observation, canonical, native_wheel

RECEIPT_FORMAT = "cozy.runtime.EnvironmentReceipt/1"
CONTENT_IDENTITY_NAME = "package-environment-content.json"
MAX_WHEEL_BYTES = 2 << 30
MAX_WHEEL_MEMBERS = 100_000
MAX_WHEEL_EXPANDED_BYTES = 8 << 30
MAX_WHEEL_EXPANSION = 64
MAX_METADATA_BYTES = 4 << 20
MAX_PACKAGE_WHEELS = 129

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


class EnvironmentRefusal(Exception):
    """A fail-closed package-environment verdict with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def observe_base(python: Path = Path(sys.executable)) -> base_observation.Observation:
    try:
        return base_observation.from_python(python)
    except base_observation.ObservationRefusal as exc:
        raise EnvironmentRefusal("package_environment_base_invalid", str(exc)) from exc


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
class BlobGrant:
    digest: str
    length: int
    path: Path
    verified: VerifiedFile | None = None


@dataclass(frozen=True, slots=True)
class BootInstall:
    """One image-carried distribution whose base version falls outside a declared range.

    The pod venv installs an in-range version at prepare time — the boot-install slow path
    (cr-070, xs-008): allowed, non-ideal, expected never, logged. The venv copy then shadows
    the image's through ordinary venv precedence.
    """

    distribution: str
    specifier: str
    extras: tuple[str, ...]
    image_version: str

    @property
    def requirement(self) -> str:
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        return f"{self.distribution}{extras}{self.specifier}"


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
            raise EnvironmentRefusal("package_environment_wheel_unreadable", str(path)) from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size != length:
            raise EnvironmentRefusal("package_environment_wheel_identity_changed", str(path))
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
    project_wheel_digest: str
    wheels: tuple[Wheel, ...]
    grants: tuple[BlobGrant, ...]


@dataclass(frozen=True, slots=True)
class InstalledEnvironment:
    generation: Path
    site_packages: Path
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
    base: base_observation.Observation,
    *,
    allow_base_owned: bool = False,
) -> tuple[Wheel, VerifiedFile]:
    """Derive one local wheel fact, then run the ordinary full wheel inspection once."""

    if owner not in {"project", "custom"} or not path.is_absolute():
        raise EnvironmentRefusal("package_environment_wheel_invalid", str(path))
    try:
        info = path.lstat()
        distribution, version, _build, tags = parse_wheel_filename(path.name)
        with zipfile.ZipFile(path) as archive:
            roots = _import_roots({member.filename for member in archive.infolist()})
    except (OSError, InvalidWheelFilename, zipfile.BadZipFile) as exc:
        raise EnvironmentRefusal("package_environment_wheel_invalid", path.name) from exc
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_WHEEL_BYTES:
        raise EnvironmentRefusal("package_environment_wheel_invalid", path.name)
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
    inspected = _inspect(
        wheel,
        path,
        base,
        allow_base_owned=allow_base_owned,
    )
    return inspected.wheel, VerifiedFile.record(path, wheel.digest, wheel.length)


def preflight_environment(
    project: Path,
    dependencies: Sequence[Path],
    base: base_observation.Observation | None = None,
) -> tuple[base_observation.Observation, tuple[Wheel, ...], tuple[Wheel, ...]]:
    """Apply rental package admission without installing or executing code."""

    base = base or base_observation.current()
    project_row = measure_wheel(project, "project", base)
    observed = [project_row[0]]
    measured = [(project_row[0], project_row[1], project)]
    for path in dependencies:
        row = measure_wheel(path, "custom", base, allow_base_owned=True)
        observed.append(row[0])
        if not base_owned_wheel(row[0], base):
            measured.append((row[0], row[1], path))
    wheels = tuple(row[0] for row in measured)
    source = Input(
        environment_digest=_sha(canonical.write({"wheels": [wheel.digest for wheel in wheels]})),
        project_wheel_digest=project_row[0].digest,
        wheels=wheels,
        grants=tuple(
            BlobGrant(wheel.digest, wheel.length, path, receipt)
            for wheel, receipt, path in measured
        ),
    )
    ordered = _verify_inventory(source, base)
    inspected = _verify_grants(source, ordered, base)
    _verify_python_requirements(inspected, base)
    return base, ordered, tuple(observed)


def base_owned_wheel(wheel: Wheel, base: base_observation.Observation) -> bool:
    """Identify a redundant worker-image distribution by its complete import ownership.

    Package metadata, not the supplied wheel version, constrains the worker-image version.
    The caller omits a base-owned wheel and the ordinary requirement check below then admits
    or refuses the version actually provided by the worker image.
    """

    if wheel.owner != "custom":
        return False
    version = base.distribution_versions.get(wheel.distribution)
    if version is None or not base_observation.protected_distribution(wheel.distribution):
        return False
    roots = tuple(
        sorted(
            root
            for root, providers in base.root_providers.items()
            if wheel.distribution in providers
        )
    )
    try:
        Version(version)
        Version(wheel.version)
    except InvalidVersion as exc:
        raise EnvironmentRefusal(
            "package_environment_base_invalid",
            f"distribution={wheel.distribution} worker_image_version={version!r}",
        ) from exc
    if roots != wheel.import_roots:
        raise EnvironmentRefusal(
            "package_environment_base_distribution_import_roots_conflict",
            f"package supplied {wheel.distribution} {wheel.version} with import roots "
            f"{wheel.import_roots}, but the worker image provides {wheel.distribution} "
            f"{version} with import roots {roots}; worker-image distributions cannot be replaced",
        )
    return True


def _allows_observed_wheel(
    wheel: Wheel,
    base: base_observation.Observation,
    *,
    allow_base_owned: bool,
    allow_installed_package: bool,
) -> bool:
    """Distinguish immutable image ownership from an already-complete package venv."""

    if allow_base_owned and base_owned_wheel(wheel, base):
        return True
    version = base.distribution_versions.get(wheel.distribution)
    if (
        not allow_installed_package
        or version is None
        or base_observation.protected_distribution(wheel.distribution)
    ):
        return False
    try:
        return Version(version) == Version(wheel.version)
    except InvalidVersion as exc:
        raise EnvironmentRefusal(
            "package_environment_base_invalid",
            f"distribution={wheel.distribution} installed_version={version!r}",
        ) from exc


def materialize(
    source: Input,
    root: Path,
    *,
    python: Path = Path(sys.executable),
    base: base_observation.Observation | None = None,
    native_environment: Mapping[str, str] | None = None,
    native_device_index: int = 0,
) -> InstalledEnvironment:
    """Verify exact wheels and atomically publish one ordinary uv virtual environment.

    Content identity is the interpreter ABI/platform, uv version, and sorted package-owned
    wheel digests. The actual base is re-observed for admission and is not part of that identity.

    A native dependency wheel has no bypass. Its exact installed bytes are statically inspected,
    joined to the observed interpreter/libc/loader/GPU seat, then loaded and imported in the
    bounded native operator child before the generation can be published or reused.

    An image-carried distribution whose observed version falls outside the wheels' declared
    ranges is boot-installed into the venv from the index (cr-070) rather than refused. The
    boot plan is a pure function of the exact wheels and the re-observed base — the same
    inputs every admission already re-derives — so it is deliberately NOT part of the content
    identity, and every open of the generation re-verifies the installed versions in range.
    """

    _require_digest(source.environment_digest, "environment_digest")
    _require_digest(source.project_wheel_digest, "project_wheel_digest")
    observed = base or observe_base(python)
    wheels = _verify_inventory(source, observed, venv_precedence=True)
    uv = _uv_binary()
    current_uv = _observed_uv_version(uv)
    if current_uv != observed.uv_version:
        raise EnvironmentRefusal(
            "package_environment_uv_changed",
            f"observed={observed.uv_version} current={current_uv}",
        )
    content_bytes = canonical.write(
        _content_identity(wheels, observed.interpreter_identity, observed.uv_version)
    )
    content_digest = _sha(content_bytes)
    receipt_bytes = canonical.write(_receipt(source, wheels, content_digest))
    receipt_digest = _sha(receipt_bytes)
    contents = root / "contents"
    receipts = root / "receipts"
    final = contents / content_digest.removeprefix("sha256:")
    receipt_path = receipts / f"{receipt_digest.removeprefix('sha256:')}.json"
    inspected = _verify_grants(source, wheels, observed)
    boot_installs = _verify_python_requirements(inspected, observed, boot=True)
    if receipt_path.exists():
        installed = open_generation(
            root,
            receipt_digest=receipt_digest,
            environment_digest=source.environment_digest,
            reused=True,
            python=python,
        )
        booted = _verify_boot_installs(installed.site_packages, boot_installs, reused=True)
        _qualify_native_environment(
            installed,
            inspected,
            observed,
            native_environment,
            native_device_index,
            booted,
        )
        return installed
    reused = final.exists()
    _require_owned_root(root)
    _remove_dead_staging(root)
    contents.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The executor receives only the venv interpreter. Search without directory listing is
    # sufficient to reach it; write remains root-only and receipts remain fully private.
    contents.chmod(0o711)
    receipts.chmod(0o700)
    if not reused:
        staging = root / f".environment-staging-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            _install_with_uv(uv, python, staging, inspected, boot_installs)
            staged = _installed_at(
                staging,
                content_digest=content_digest,
                receipt_bytes=receipt_bytes,
                receipt_digest=receipt_digest,
                reused=False,
                base=observed,
            )
            booted = _verify_boot_installs(staged.site_packages, boot_installs, reused=False)
            _qualify_native_environment(
                staged,
                inspected,
                observed,
                native_environment,
                native_device_index,
                booted,
            )
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
    installed = open_generation(
        root,
        receipt_digest=receipt_digest,
        environment_digest=source.environment_digest,
        reused=reused,
        python=python,
    )
    if reused:
        booted = _verify_boot_installs(installed.site_packages, boot_installs, reused=True)
        _qualify_native_environment(
            installed,
            inspected,
            observed,
            native_environment,
            native_device_index,
            booted,
        )
    return installed


def verify_inventory(source: Input, base: base_observation.Observation) -> None:
    """Refuse incompatible wheel facts before the worker downloads their bytes.

    This is the acquisition lane's pre-download mirror of `materialize`, so it applies the
    same venv-precedence rule: a package copy of a non-protected base distribution installs
    and shadows rather than refusing.
    """

    _verify_inventory(source, base, venv_precedence=True)


def verify_existing_environment(
    source: Input,
    *,
    base: base_observation.Observation,
    installed: InstalledEnvironment,
    native_environment: Mapping[str, str] | None = None,
    native_device_index: int = 0,
) -> None:
    """Verify exact wheel facts against an already-complete local uv environment."""

    wheels = _verify_inventory(
        source,
        base,
        allow_base_owned=True,
        allow_installed_package=True,
    )
    inspected = _verify_grants(
        source,
        wheels,
        base,
        allow_base_owned=True,
    )
    _verify_python_requirements(inspected, base)
    _qualify_native_environment(
        installed,
        inspected,
        base,
        native_environment,
        native_device_index,
    )


def _installed_at(
    generation: Path,
    *,
    content_digest: str,
    receipt_bytes: bytes,
    receipt_digest: str,
    reused: bool,
    base: base_observation.Observation,
) -> InstalledEnvironment:
    major, minor = _target_python(base)
    site_packages = generation / "lib" / f"python{major}.{minor}" / "site-packages"
    python = generation / "bin" / "python"
    if not site_packages.is_dir() or not python.is_file():
        raise EnvironmentRefusal("package_environment_incomplete", str(generation))
    return InstalledEnvironment(
        generation=generation,
        site_packages=site_packages,
        python=python,
        receipt_bytes=receipt_bytes,
        receipt_digest=receipt_digest,
        content_digest=content_digest,
        reused=reused,
    )


def _qualify_native_environment(
    installed: InstalledEnvironment,
    inspected: Sequence[_Inspected],
    base: base_observation.Observation,
    environment: Mapping[str, str] | None,
    device_index: int,
    boot_versions: Mapping[str, str] | None = None,
) -> None:
    native = tuple(item for item in inspected if item.native_members)
    if not native:
        return
    if not environment:
        raise EnvironmentRefusal(
            "package_environment_native_proof_environment_absent",
            "native dependency wheels require the worker's sealed execution environment",
        )
    members = tuple(sorted({member for item in native for member in item.native_members}))
    roots = tuple(sorted({root for item in native for root in item.import_roots}))
    if not roots:
        raise EnvironmentRefusal(
            "package_environment_native_import_roots_absent",
            ",".join(item.wheel.filename for item in native),
        )
    try:
        inspection = native_wheel.inspect_tree(installed.site_packages, members)
        gpu = native_wheel.observe_gpu(device_index) if base.accelerator_backend != "none" else None
        seat = native_wheel.observe_seat(installed.python, environment)
        venv_versions = {
            **{item.wheel.distribution: item.wheel.version for item in inspected},
            **(boot_versions or {}),
        }
        native_wheel.verify_seat(base, seat, gpu=gpu, venv_versions=venv_versions)
        native_wheel.qualify_host(
            inspection,
            gpu_sm=gpu["sm"] if gpu is not None else None,
        )
        request = json.dumps(
            {
                "members": list(members),
                "roots": list(roots),
                "site_packages": str(installed.site_packages),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        native_wheel.qualify_operator(
            python=installed.python,
            fixture="cozy_runtime.internal.native_dependency_probe:probe",
            expected_result_digest=canonical.digest(
                {"members": list(members), "roots": list(roots)}
            ),
            environment=environment,
            request=request,
            # The runaway fence for a dlopen-every-member, import-every-root probe: a spin
            # is stopped by the kernel at this many CPU-seconds; a slow cold page-in is not
            # a spin and is not stopped by anything but its own exit.
            cpu_seconds=120,
        )
    except native_wheel.NativeWheelRefusal as exc:
        raise EnvironmentRefusal(exc.code, exc.detail) from exc


def open_existing_environment(python: Path, content_digest: str) -> InstalledEnvironment:
    """Open one ordinary local venv without installing or changing it."""

    _require_digest(content_digest, "content_digest")
    if not python.is_absolute() or python.parent.name != "bin" or not python.is_file():
        raise EnvironmentRefusal("package_environment_python_invalid", str(python))
    generation = python.parent.parent
    if not (generation / "pyvenv.cfg").is_file():
        raise EnvironmentRefusal("package_environment_venv_invalid", str(generation))
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json,sys,sysconfig; print(json.dumps({"
            "'base_prefix':sys.base_prefix,'prefix':sys.prefix,"
            "'purelib':sysconfig.get_path('purelib')},sort_keys=True,separators=(',',':')))",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise EnvironmentRefusal(
            "package_environment_venv_invalid",
            result.stderr.strip()[-1000:] or "venv probe returned invalid JSON",
        ) from exc
    site = Path(str(observed.get("purelib", ""))) if isinstance(observed, dict) else Path()
    if (
        result.returncode != 0
        or not isinstance(observed, dict)
        or observed.get("prefix") != str(generation)
        or observed.get("base_prefix") == str(generation)
        or not site.is_absolute()
        or not site.is_relative_to(generation)
        or not site.is_dir()
    ):
        raise EnvironmentRefusal(
            "package_environment_venv_invalid", f"python={python} observed={observed!r}"
        )
    return InstalledEnvironment(
        generation=generation,
        site_packages=site,
        python=python,
        receipt_bytes=b"",
        receipt_digest="",
        content_digest=content_digest,
        reused=True,
    )


def open_generation(
    root: Path,
    *,
    receipt_digest: str,
    environment_digest: str,
    reused: bool = True,
    python: Path = Path(sys.executable),
) -> InstalledEnvironment:
    """Open immutable content through one exact placement receipt without tree hashes.

    The package wheel bytes were hashed before atomic publication. Re-reading every installed
    file would add startup cost without adding an independent trust boundary.
    """

    _require_digest(receipt_digest, "receipt_digest")
    _require_digest(environment_digest, "environment_digest")
    receipt_path = root / "receipts" / f"{receipt_digest.removeprefix('sha256:')}.json"
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as exc:
        raise EnvironmentRefusal("package_environment_receipt_absent", str(exc)) from exc
    if _sha(receipt_bytes) != receipt_digest:
        raise EnvironmentRefusal("package_environment_receipt_corrupt", str(receipt_path))
    value = parse_environment_receipt(receipt_bytes)
    if value.get("environment_digest") != environment_digest:
        raise EnvironmentRefusal(
            "package_environment_receipt_wrong_environment", environment_digest
        )
    content_digest = value.get("content_digest")
    if not isinstance(content_digest, str):
        raise EnvironmentRefusal("package_environment_receipt_invalid", "content digest is absent")
    _require_digest(content_digest, "content_digest")
    generation = root / "contents" / content_digest.removeprefix("sha256:")
    try:
        identity_bytes = (generation / CONTENT_IDENTITY_NAME).read_bytes()
    except OSError as exc:
        raise EnvironmentRefusal("package_environment_content_absent", str(exc)) from exc
    if _sha(identity_bytes) != content_digest:
        raise EnvironmentRefusal("package_environment_content_identity_corrupt", str(generation))
    try:
        identity = canonical.parse_canonical(identity_bytes)
    except canonical.CanonicalError as exc:
        raise EnvironmentRefusal("package_environment_content_identity_invalid", str(exc)) from exc
    if not _content_matches_receipt(identity, value):
        raise EnvironmentRefusal("package_environment_content_identity_mismatch", str(generation))
    if not isinstance(identity, dict) or not isinstance(identity.get("interpreter"), dict):
        raise EnvironmentRefusal(
            "package_environment_content_identity_invalid", "interpreter identity is absent"
        )
    python_abi = identity["interpreter"].get("python_abi")
    match = re.fullmatch(r"cp([0-9])([0-9]{2})", str(python_abi))
    if match is None:
        raise EnvironmentRefusal(
            "package_environment_content_identity_invalid", "python ABI is invalid"
        )
    major, minor = int(match.group(1)), int(match.group(2))
    site_packages = generation / "lib" / f"python{major}.{minor}" / "site-packages"
    environment_python = generation / "bin" / "python"
    if not site_packages.is_dir() or not environment_python.is_file():
        raise EnvironmentRefusal("package_environment_incomplete", str(generation))
    owner = os.geteuid()
    boundaries = (root, root / "contents", root / "receipts")
    if (
        any(path.stat().st_uid != owner or path.stat().st_mode & 0o022 for path in boundaries)
        or generation.stat().st_uid != owner
        or site_packages.stat().st_uid != owner
        or receipt_path.stat().st_uid != owner
        or generation.stat().st_mode & 0o222
        or site_packages.stat().st_mode & 0o222
        or receipt_path.stat().st_mode & 0o222
    ):
        raise EnvironmentRefusal("package_environment_generation_mutable", str(generation))
    if not python.is_file():
        raise EnvironmentRefusal("package_environment_base_python_absent", str(python))
    return InstalledEnvironment(
        generation=generation,
        site_packages=site_packages,
        python=environment_python,
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
        raise EnvironmentRefusal("package_environment_receipt_invalid", str(exc)) from exc
    fields = {
        "content_digest",
        "environment_digest",
        "format",
        "package_wheels",
        "project_wheel_digest",
    }
    if not isinstance(value, dict) or set(value) != fields or value.get("format") != RECEIPT_FORMAT:
        raise EnvironmentRefusal("package_environment_receipt_invalid", "wrong fields or format")
    for field in (
        "content_digest",
        "environment_digest",
        "project_wheel_digest",
    ):
        item = value.get(field)
        if not isinstance(item, str):
            raise EnvironmentRefusal(
                "package_environment_receipt_invalid", f"{field} is not a digest"
            )
        _require_digest(item, field)
    rows = value.get("package_wheels")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PACKAGE_WHEELS:
        raise EnvironmentRefusal(
            "package_environment_receipt_invalid", "package_wheels is empty or oversized"
        )
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
            raise EnvironmentRefusal(
                "package_environment_receipt_invalid", f"package_wheels[{index}] fields"
            )
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
            raise EnvironmentRefusal(
                "package_environment_receipt_invalid", f"package_wheels[{index}] values"
            )
        digests.add(digest)
        distributions.append(distribution)
        projects += owner == "project"
    if distributions != sorted(set(distributions)) or projects != 1:
        raise EnvironmentRefusal(
            "package_environment_receipt_invalid",
            "package wheels are not sorted unique with one project",
        )
    project_digest = value["project_wheel_digest"]
    if not any(row["owner"] == "project" and row["digest"] == project_digest for row in rows):
        raise EnvironmentRefusal(
            "package_environment_receipt_invalid", "project wheel digest is not the project"
        )
    return value


def _verify_inventory(
    source: Input,
    base: base_observation.Observation,
    *,
    allow_base_owned: bool = False,
    allow_installed_package: bool = False,
    venv_precedence: bool = False,
) -> tuple[Wheel, ...]:
    if not source.wheels or len(source.wheels) > MAX_PACKAGE_WHEELS:
        raise EnvironmentRefusal("package_environment_wheel_set_invalid", str(len(source.wheels)))
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
            raise EnvironmentRefusal("package_environment_wheel_invalid", wheel.filename)
        if wheel.owner == "project" and wheel.tags != ("py3-none-any",):
            raise EnvironmentRefusal(
                "package_environment_project_wheel_not_pure",
                f"{wheel.filename} tags={','.join(wheel.tags)}; project must be py3-none-any",
            )
        base_version = base.distribution_versions.get(name)
        if base_version is None and base_observation.protected_distribution(name):
            raise EnvironmentRefusal(
                "package_environment_second_protected_family",
                f"distribution={name} is worker-image-owned and absent from this worker",
            )
        if (
            base_version is not None
            and not _allows_observed_wheel(
                wheel,
                base,
                allow_base_owned=allow_base_owned,
                allow_installed_package=allow_installed_package,
            )
            # A fresh `--system-site-packages` venv legitimately shadows a NON-protected
            # base copy: the venv wins the import by ordinary precedence (cr-070). The
            # protected worker-image families keep their fence.
            and not (venv_precedence and not base_observation.protected_distribution(name))
        ):
            raise EnvironmentRefusal(
                "package_environment_shadows_base_distribution",
                f"distribution={name} package_version={wheel.version} image_version={base_version}",
            )
        if wheel.import_roots != tuple(sorted(set(wheel.import_roots))) or not all(
            valid_import_root(root) for root in wheel.import_roots
        ):
            raise EnvironmentRefusal("package_environment_wheel_invalid", wheel.filename)
        _require_compatible_tags(wheel, base)
        for root in wheel.import_roots:
            providers = base.root_providers.get(root)
            if providers is not None and not (allow_base_owned and wheel.distribution in providers):
                raise _base_import_collision(wheel, root, providers)
        if name in names or wheel.digest in digests:
            raise EnvironmentRefusal("package_environment_wheel_duplicate", name)
        names.add(name)
        digests.add(wheel.digest)
        projects += wheel.owner == "project"
        ordered.append(wheel)
    if projects != 1 or source.project_wheel_digest not in {
        row.digest for row in ordered if row.owner == "project"
    }:
        raise EnvironmentRefusal(
            "package_environment_project_wheel_mismatch", source.project_wheel_digest
        )
    return tuple(sorted(ordered, key=lambda row: row.distribution))


def _verify_grants(
    source: Input,
    wheels: Sequence[Wheel],
    base: base_observation.Observation,
    *,
    allow_base_owned: bool = False,
) -> tuple[_Inspected, ...]:
    grants = {grant.digest: grant for grant in source.grants}
    expected = {wheel.digest for wheel in wheels}
    if len(grants) != len(source.grants) or set(grants) != expected:
        raise EnvironmentRefusal(
            "package_environment_grant_set_mismatch",
            f"missing={sorted(expected - set(grants))} extra={sorted(set(grants) - expected)}",
        )
    inspected: list[_Inspected] = []
    for wheel in wheels:
        grant = grants[wheel.digest]
        if grant.length != wheel.length:
            raise EnvironmentRefusal("package_environment_grant_length_mismatch", wheel.digest)
        try:
            info = grant.path.lstat()
        except OSError as exc:
            raise EnvironmentRefusal(
                "package_environment_wheel_unreadable", str(grant.path)
            ) from exc
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
                raise EnvironmentRefusal("package_environment_wheel_digest_mismatch", wheel.digest)
        elif verified.digest != wheel.digest:
            raise EnvironmentRefusal("package_environment_wheel_digest_mismatch", wheel.digest)
        inspected.append(
            _inspect(
                wheel,
                grant.path,
                base,
                allow_base_owned=allow_base_owned,
            )
        )
    root_owners: dict[str, str] = {}
    for item in inspected:
        for root in item.import_roots:
            prior = root_owners.get(root)
            if prior is not None:
                raise EnvironmentRefusal(
                    "package_environment_import_root_duplicate",
                    f"root={root} distributions={prior},{item.wheel.distribution}",
                )
            root_owners[root] = item.wheel.distribution
    return tuple(inspected)


def _inspect(
    wheel: Wheel,
    path: Path,
    base: base_observation.Observation,
    *,
    allow_base_owned: bool = False,
) -> _Inspected:
    try:
        filename_name, filename_version, _build, filename_tags = parse_wheel_filename(
            wheel.filename
        )
    except InvalidWheelFilename as exc:
        raise EnvironmentRefusal("wheel_filename_mismatch", wheel.filename) from exc
    declared_tags: set[Tag] = set()
    for tag in wheel.tags:
        declared_tags.update(parse_tag(tag))
    if (
        _name(filename_name) != wheel.distribution
        or filename_version != Version(wheel.version)
        or filename_tags != declared_tags
    ):
        raise EnvironmentRefusal("wheel_filename_mismatch", wheel.filename)
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(archive.infolist())
            if not members or len(members) > MAX_WHEEL_MEMBERS:
                raise EnvironmentRefusal("package_environment_wheel_member_count", wheel.filename)
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
                    raise EnvironmentRefusal("package_environment_wheel_member_invalid", name)
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
                    raise EnvironmentRefusal(
                        "package_environment_wheel_expansion_cap", wheel.filename
                    )
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise EnvironmentRefusal("package_environment_wheel_symlink", name)
                if not member.is_dir() and PurePosixPath(name).suffix == ".pth":
                    raise EnvironmentRefusal("package_environment_pth_forbidden", name)
                if member.is_dir():
                    continue
                installed = _install_path(name)
                if installed is None and ".data/" in name:
                    raise EnvironmentRefusal(
                        "package_environment_wheel_install_scheme_forbidden", name
                    )
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
                    raise EnvironmentRefusal(
                        "package_environment_native_build_input_forbidden", name
                    )
                magic = archive.open(member).read(4)
                native = suffix in _NATIVE_SUFFIXES or magic == b"\x7fELF"
                if wheel.owner == "project" and native:
                    raise EnvironmentRefusal("package_environment_project_wheel_native", name)
                if native:
                    assert installed is not None
                    native_members.append(installed.as_posix())
            dist_infos = sorted(
                name
                for name in names
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            )
            if len(dist_infos) != 1:
                raise EnvironmentRefusal(
                    "package_environment_wheel_metadata_missing", wheel.filename
                )
            wheel_infos = sorted(
                name for name in names if name.count("/") == 1 and name.endswith(".dist-info/WHEEL")
            )
            if len(wheel_infos) != 1:
                raise EnvironmentRefusal(
                    "package_environment_wheel_metadata_missing", wheel.filename
                )
            metadata = archive.read(dist_infos[0])
            if len(metadata) > MAX_METADATA_BYTES:
                raise EnvironmentRefusal(
                    "package_environment_wheel_metadata_oversized", wheel.filename
                )
            parsed = BytesParser().parsebytes(metadata)
            parsed_wheel = BytesParser().parsebytes(archive.read(wheel_infos[0]))
            if (
                _name(parsed.get("Name", "")) != wheel.distribution
                or parsed.get("Version") != wheel.version
                or tuple(sorted(parsed_wheel.get_all("Tag", []))) != wheel.tags
            ):
                raise EnvironmentRefusal(
                    "package_environment_wheel_metadata_mismatch", wheel.filename
                )
            root_is_pure = parsed_wheel.get("Root-Is-Purelib", "").strip().lower()
            if wheel.owner == "project" and root_is_pure != "true":
                raise EnvironmentRefusal(
                    "package_environment_project_wheel_not_pure", "Root-Is-Purelib is not true"
                )
            if native_members and root_is_pure != "false":
                raise EnvironmentRefusal(
                    "package_environment_native_wheel_metadata_invalid",
                    f"{wheel.filename} carries native members but Root-Is-Purelib is not false",
                )
            if native_members and any(
                tag.interpreter.startswith("py") or tag.abi == "none" or tag.platform == "any"
                for tag in declared_tags
            ):
                raise EnvironmentRefusal(
                    "package_environment_native_wheel_tag_not_concrete",
                    f"{wheel.filename}: native custom wheels require concrete CPython ABI and "
                    "platform tags",
                )
            if native_members and any(".data/" in name for name in names):
                raise EnvironmentRefusal(
                    "package_environment_native_install_scheme_forbidden",
                    f"{wheel.filename}: native wheels may not use a .data install scheme",
                )
            record_name = dist_infos[0].removesuffix("METADATA") + "RECORD"
            if record_name not in names:
                raise EnvironmentRefusal("package_environment_wheel_record_missing", wheel.filename)
            _verify_record(archive, record_name, names)
            roots = _import_roots(names)
            measured_roots = tuple(sorted(roots))
            if wheel.import_roots != measured_roots:
                raise EnvironmentRefusal(
                    "package_environment_wheel_import_roots_mismatch",
                    f"distribution={wheel.distribution} declared={list(wheel.import_roots)} "
                    f"measured={list(measured_roots)}",
                )
            for root in measured_roots:
                providers = base.root_providers.get(root)
                if providers is not None and not (
                    allow_base_owned and wheel.distribution in providers
                ):
                    raise _base_import_collision(wheel, root, providers)
            requires_python = parsed.get("Requires-Python")
            requirements: list[Requirement] = []
            try:
                for value in parsed.get_all("Requires-Dist", []):
                    requirements.append(Requirement(value))
            except InvalidRequirement as exc:
                raise EnvironmentRefusal(
                    "package_environment_requirement_invalid", f"{wheel.distribution}: {exc}"
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
    except EnvironmentRefusal:
        raise
    except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise EnvironmentRefusal("package_environment_wheel_invalid", wheel.filename) from exc


def _verify_record(archive: zipfile.ZipFile, record_name: str, names: set[str]) -> None:
    raw = archive.read(record_name)
    try:
        rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    except (UnicodeError, csv.Error) as exc:
        raise EnvironmentRefusal("package_environment_wheel_record_invalid", record_name) from exc
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in recorded or row[0] not in names:
            raise EnvironmentRefusal("package_environment_wheel_record_invalid", record_name)
        recorded.add(row[0])
        if row[0] == record_name:
            if row[1] or row[2]:
                raise EnvironmentRefusal("package_environment_wheel_record_invalid", record_name)
            continue
        if not row[1] or not row[2].isdigit():
            raise EnvironmentRefusal("package_environment_wheel_record_unhashed", row[0])
        algorithm, separator, encoded = row[1].partition("=")
        payload = archive.read(row[0])
        got = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        if algorithm != "sha256" or not separator or encoded != got or int(row[2]) != len(payload):
            raise EnvironmentRefusal("package_environment_wheel_record_mismatch", row[0])
    files = {name for name in names if not name.endswith("/")}
    if recorded != files:
        raise EnvironmentRefusal("package_environment_wheel_record_incomplete", record_name)


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


def _base_import_collision(
    wheel: Wheel, root: str, providers: tuple[str, ...]
) -> EnvironmentRefusal:
    return EnvironmentRefusal(
        "package_environment_shadows_base_import",
        f"root={root} package_distribution={wheel.distribution} "
        f"image_providers={','.join(providers)}",
    )


def _target_python(base: base_observation.Observation) -> tuple[int, int]:
    match = re.fullmatch(r"cp([0-9])([0-9]{2})", base.python_abi)
    if match is None:
        raise EnvironmentRefusal(
            "package_environment_base_invalid", "Python ABI is not the cpNNN spelling"
        )
    return int(match.group(1)), int(match.group(2))


def _target_platforms(base: base_observation.Observation) -> frozenset[str]:
    os_arch = base.os_arch
    libc = base.libc
    if os_arch != "linux/amd64":
        raise EnvironmentRefusal(
            "package_environment_platform_unsupported",
            f"semantic wheel admission has no seat for {os_arch}",
        )
    match = re.fullmatch(r"glibc([0-9]+)\.([0-9]+)", libc or "")
    if match is None or int(match.group(1)) != 2:
        raise EnvironmentRefusal(
            "package_environment_base_invalid", f"Linux/x86 base has unsupported libc {libc!r}"
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


def _compatible_tags(base: base_observation.Observation) -> frozenset[Tag]:
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


def _require_compatible_tags(wheel: Wheel, base: base_observation.Observation) -> None:
    declared: set[Tag] = set()
    try:
        for value in wheel.tags:
            declared.update(parse_tag(value))
    except ValueError as exc:
        raise EnvironmentRefusal("package_environment_wheel_tag_invalid", wheel.filename) from exc
    if declared.isdisjoint(_compatible_tags(base)):
        raise EnvironmentRefusal(
            "package_environment_wheel_platform_incompatible",
            f"distribution={wheel.distribution} tags={','.join(wheel.tags)} "
            f"target={base.interpreter_identity}",
        )


def _marker_environment(base: base_observation.Observation) -> dict[str, str]:
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


def _verify_python_requirements(
    inspected: Sequence[_Inspected],
    base: base_observation.Observation,
    *,
    boot: bool = False,
) -> tuple[BootInstall, ...]:
    """Verify Requires-Python and every marker-true Requires-Dist against the joined venv.

    With ``boot`` — the fresh-venv build lane — a requirement naming an image-carried
    distribution the package does not itself provide, whose observed image version falls
    outside the range, becomes one BootInstall row instead of a refusal (cr-070): the venv
    installs an in-range version that shadows the image's copy. The row's specifier is the
    intersection of EVERY declared range naming that distribution, so the chosen version
    satisfies all of them or the install itself refuses.
    """

    environment = _marker_environment(base)
    python = Version(environment["python_full_version"])
    base_versions = base.distribution_versions
    package_versions = {item.wheel.distribution: item.wheel.version for item in inspected}
    versions = {**base_versions, **package_versions}
    rejected: set[str] = set()
    merged: dict[str, tuple[SpecifierSet, frozenset[str]]] = {}
    for item in inspected:
        if item.requires_python:
            try:
                supported = SpecifierSet(item.requires_python).contains(python, prereleases=True)
            except InvalidSpecifier as exc:
                raise EnvironmentRefusal(
                    "package_environment_requires_python_invalid",
                    f"{item.wheel.distribution}: {item.requires_python}",
                ) from exc
            if not supported:
                raise EnvironmentRefusal(
                    "package_environment_python_incompatible",
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
                raise EnvironmentRefusal(
                    "package_environment_dependency_missing",
                    f"distribution={item.wheel.distribution} requirement={requirement}",
                )
            try:
                accepted = not requirement.specifier or requirement.specifier.contains(
                    Version(installed), prereleases=True
                )
            except InvalidVersion as exc:
                raise EnvironmentRefusal(
                    "package_environment_base_invalid",
                    f"{name} has invalid version {installed!r}",
                ) from exc
            bootable = boot and name in base_versions and name not in package_versions
            if bootable:
                specifier, extras = merged.get(name, (SpecifierSet(), frozenset()))
                merged[name] = (
                    specifier & requirement.specifier,
                    extras | frozenset(requirement.extras),
                )
            if accepted:
                continue
            if bootable:
                rejected.add(name)
                continue
            if name in base_versions and name not in package_versions:
                detail = (
                    f"package distribution {item.wheel.distribution} requires {requirement}, "
                    f"but the worker image provides {name} {installed}"
                )
            else:
                detail = (
                    f"package distribution {item.wheel.distribution} requires {requirement}, "
                    f"but the package environment provides {name} {installed}"
                )
            raise EnvironmentRefusal(
                "package_environment_dependency_base_conflict",
                detail,
            )
    return tuple(
        BootInstall(
            distribution=name,
            specifier=str(merged[name][0]),
            extras=tuple(sorted(merged[name][1])),
            image_version=base_versions[name],
        )
        for name in sorted(rejected)
    )


def _install_path(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    first = path.parts[0]
    if first.endswith(".data"):
        if len(path.parts) < 3 or path.parts[1] not in {"purelib", "platlib"}:
            return None
        return PurePosixPath(*path.parts[2:])
    return path


def _uv_binary() -> Path:
    found = shutil.which("uv")
    if found is None:
        raise EnvironmentRefusal(
            "package_environment_uv_absent", "uv is not installed in the worker base"
        )
    path = Path(found)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise EnvironmentRefusal("package_environment_uv_invalid", str(path))
    return path


def _observed_uv_version(uv: Path) -> str:
    result = subprocess.run(
        [str(uv), "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    version = result.stdout.strip()
    if (
        result.returncode != 0
        or re.fullmatch(r"uv [0-9]+(?:\.[0-9]+){1,3}(?: \([^\n]+\))?", version) is None
    ):
        raise EnvironmentRefusal(
            "package_environment_uv_invalid",
            result.stderr.strip()[-1000:] or f"uv exited {result.returncode}",
        )
    return version.split()[1]


def _install_with_uv(
    uv: Path,
    python: Path,
    staging: Path,
    inspected: Sequence[_Inspected],
    boot_installs: Sequence[BootInstall] = (),
) -> None:
    commands = [
        [
            str(uv),
            "venv",
            "--no-project",
            "--no-config",
            "--offline",
            "--no-python-downloads",
            "--link-mode",
            "clone",
            "--python",
            str(python),
            "--system-site-packages",
            str(staging),
        ],
        [
            str(uv),
            "pip",
            "install",
            "--no-config",
            "--offline",
            "--no-index",
            "--no-deps",
            "--no-build",
            "--link-mode",
            "clone",
            "--compile-bytecode",
            "--python",
            str(staging / "bin" / "python"),
            *(str(item.path) for item in sorted(inspected, key=lambda item: item.wheel.digest)),
        ],
    ]
    if boot_installs:
        # The boot-install slow path (cr-070): ONE online resolve of exactly the named
        # image-owned distributions, `--no-deps` so no digest-verified lock wheel is ever
        # replaced by a resolver choice. `_verify_boot_installs` then proves each installed
        # version in range, and the descriptor derivation exercises the joined venv for real.
        commands.append(
            [
                str(uv),
                "pip",
                "install",
                "--no-config",
                "--no-deps",
                "--no-build",
                "--link-mode",
                "clone",
                "--compile-bytecode",
                "--python",
                str(staging / "bin" / "python"),
                *(row.requirement for row in boot_installs),
            ]
        )
    for command in commands:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()[-2000:] or result.stdout.strip()[-2000:]
            raise EnvironmentRefusal(
                "package_environment_uv_failed",
                detail or f"uv exited {result.returncode}",
            )


def _verify_boot_installs(
    site_packages: Path, boot_installs: Sequence[BootInstall], *, reused: bool
) -> dict[str, str]:
    """Prove every boot-installed distribution present and in range, then say so.

    Runs on EVERY open of a generation that needs boot rows — fresh or reused — because the
    rows are re-derived from the wheels and the re-observed base rather than carried in the
    content identity. A generation that predates this image's needs is named, not served.
    """

    chosen_versions: dict[str, str] = {}
    if not boot_installs:
        return chosen_versions
    installed: dict[str, str] = {}
    for entry in site_packages.glob("*.dist-info"):
        name, separator, version = entry.name.removesuffix(".dist-info").rpartition("-")
        if separator and name:
            installed[_name(name)] = version
    for row in boot_installs:
        chosen = installed.get(row.distribution)
        if chosen is None:
            raise EnvironmentRefusal(
                "package_environment_boot_install_missing",
                f"distribution={row.distribution} required={row.specifier} "
                f"image_version={row.image_version}; the generation predates this need",
            )
        try:
            in_range = SpecifierSet(row.specifier).contains(Version(chosen), prereleases=True)
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise EnvironmentRefusal(
                "package_environment_boot_install_invalid",
                f"distribution={row.distribution} required={row.specifier} chosen={chosen!r}",
            ) from exc
        if not in_range:
            raise EnvironmentRefusal(
                "package_environment_boot_install_out_of_range",
                f"distribution={row.distribution} required={row.specifier} chosen={chosen}",
            )
        chosen_versions[row.distribution] = chosen
        print(
            f"[package-environment] boot-install distribution={row.distribution} "
            f"image_version={row.image_version} required={row.specifier} "
            f"chosen={chosen} reused={str(reused).lower()}",
            flush=True,
        )
    return chosen_versions


def _receipt(source: Input, wheels: Sequence[Wheel], content_digest: str) -> dict[str, Any]:
    return {
        "content_digest": content_digest,
        "environment_digest": source.environment_digest,
        "format": RECEIPT_FORMAT,
        "package_wheels": [
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


def _content_identity(
    wheels: Sequence[Wheel], interpreter: Mapping[str, str], uv_version: str
) -> dict[str, Any]:
    return {
        "format": "cozy.runtime.PackageEnvironment/1",
        "interpreter": dict(interpreter),
        "uv_version": uv_version,
        "wheel_digests": sorted(wheel.digest for wheel in wheels),
    }


def _content_matches_receipt(identity: object, receipt: Mapping[str, Any]) -> bool:
    rows = receipt.get("package_wheels")
    if not isinstance(rows, list) or not rows:
        raise EnvironmentRefusal("package_environment_receipt_invalid", "package_wheels is empty")
    digests: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "digest",
            "distribution",
            "owner",
            "version",
        }:
            raise EnvironmentRefusal(
                "package_environment_receipt_invalid", "package wheel row is not closed"
            )
        digest = row.get("digest")
        if not isinstance(digest, str):
            raise EnvironmentRefusal(
                "package_environment_receipt_invalid", "package wheel digest is absent"
            )
        _require_digest(digest, "package_wheels.digest")
        digests.append(digest)
    if len(set(digests)) != len(digests):
        raise EnvironmentRefusal(
            "package_environment_receipt_invalid", "package wheel digests repeat"
        )
    return (
        isinstance(identity, dict)
        and set(identity) == {"format", "interpreter", "uv_version", "wheel_digests"}
        and identity.get("format") == "cozy.runtime.PackageEnvironment/1"
        and isinstance(identity.get("interpreter"), dict)
        and set(identity["interpreter"]) == {"libc", "os_arch", "python_abi"}
        and all(isinstance(value, str) and value for value in identity["interpreter"].values())
        and isinstance(identity.get("uv_version"), str)
        and bool(identity["uv_version"])
        and identity.get("wheel_digests") == sorted(digests)
    )


def _require_owned_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = root.lstat()
    if not root.is_absolute() or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise EnvironmentRefusal("package_environment_root_owner_mismatch", str(root))
    root.chmod(0o711)


def _remove_dead_staging(root: Path) -> None:
    """Remove only staging owned by a process identity that is observably gone."""

    for prefix in (".environment-staging-",):
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
            raise EnvironmentRefusal("package_environment_receipt_conflict", str(path)) from None
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_dir(receipts)


def _require_digest(value: str, name: str) -> None:
    if not _DIGEST.fullmatch(value):
        raise EnvironmentRefusal("package_environment_digest_invalid", f"{name}={value!r}")


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
        raise EnvironmentRefusal("package_environment_wheel_unreadable", str(path)) from exc
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
