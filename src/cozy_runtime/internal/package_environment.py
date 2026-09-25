"""Complete private package environments installed with uv and atomically retained.

Exact artifact identities own the dependency closure. Image inventory supplies only
interpreter compatibility facts; it never supplies missing package distributions.
"""

from __future__ import annotations

import base64
import csv
import errno
import hashlib
import io
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import Tag, compatible_tags, cpython_tags, parse_tag
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import base_observation, canonical, storage_admission
from cozy_runtime.internal.config import package_install_environment
from cozy_runtime.protocol import weights_limits

RECEIPT_FORMAT = "cozy.runtime.EnvironmentReceipt/1"
LOCKED_RECEIPT_FORMAT = "cozy.runtime.LockedEnvironmentReceipt/1"
LOCKED_CONTENT_FORMAT = "cozy.runtime.LockedPackageEnvironment/2"
CONTENT_IDENTITY_NAME = "package-environment-content.json"
MAX_WHEEL_BYTES = 2 << 30
MAX_WHEEL_MEMBERS = 100_000
MAX_WHEEL_EXPANDED_BYTES = 8 << 30
MAX_WHEEL_EXPANSION = 64
MAX_METADATA_BYTES = 4 << 20
MAX_PACKAGE_WHEELS = weights_limits.MAX_LOCAL_PACKAGE_FILES

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NORMALIZE = re.compile(r"[-_.]+")
_NORMALIZED_DISTRIBUTION = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}$")
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,255}\.whl$")
_IMPORT_ROOT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WHEEL_TAG = re.compile(r"^[a-z0-9][a-z0-9._]*-[a-z0-9][a-z0-9._]*-[a-z0-9][a-z0-9._]*$")
_SAFE_MEMBER = re.compile(r"^[^/][^\x00]*$")
_NATIVE_SUFFIXES = frozenset({".dll", ".dylib", ".pyd", ".so"})


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
    image_seed: bool = False


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
    image_seed: bool = False


def wheel_name_version(filename: str) -> tuple[str, str] | None:
    """(normalized distribution, version) a wheel filename declares, or None."""

    try:
        name, version, _build, _tags = parse_wheel_filename(filename)
    except (InvalidWheelFilename, InvalidVersion):
        return None
    return str(name), str(version)


def measure_wheel(
    path: Path, owner: str, base: base_observation.Observation, *, allow_base_owned: bool = False
) -> tuple[Wheel, VerifiedFile]:
    inspected, verified = _measure_inspected_wheel(
        path, owner, base, allow_base_owned=allow_base_owned
    )
    return inspected.wheel, verified


def _measure_inspected_wheel(
    path: Path,
    owner: str,
    base: base_observation.Observation,
    *,
    allow_base_owned: bool = False,
    verified: VerifiedFile | None = None,
) -> tuple[_Inspected, VerifiedFile]:
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
        # The authenticated supervisor already checked the whole wheel digest.
        # Reuse the existing same-boot immutable-file receipt exactly as the
        # ordinary local wheel grant path does; full ZIP admission still runs.
        digest=(
            verified.digest
            if verified is not None and verified.matches(path, verified.digest, info.st_size)
            else _hash_file(path)
        ),
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
    return inspected, VerifiedFile.record(path, wheel.digest, wheel.length)


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


def materialize(
    source: Input,
    root: Path,
    *,
    python: Path = Path(sys.executable),
    base: base_observation.Observation | None = None,
    dependency_python: Path | None = None,
    exact_dependencies: bool = False,
    dependency_cache: Path | None = None,
) -> InstalledEnvironment:
    """Verify exact wheels and atomically publish one ordinary uv virtual environment.

    Content identity is the interpreter ABI/platform, uv version, and sorted package-owned
    wheel digests. The actual base is re-observed for admission and is not part of that identity.

    A native dependency wheel installs like any other wheel (cr-093). Whether its extension
    actually loads and launches is answered where it can only be answered — by the executor
    that imports it — and, when an author wants an offline proof, by `cozy-native-wheel-proof`.

    The selected wheels must supply their complete closure, including Runtime and ordinary
    transitive dependencies. Image distributions never fill missing requirements.
    """

    _require_digest(source.environment_digest, "environment_digest")
    _require_digest(source.project_wheel_digest, "project_wheel_digest")
    if dependency_python is not None:
        _require_same_interpreter(python, dependency_python)
        if storage_admission.enabled():
            raise EnvironmentRefusal(
                "package_environment_dependency_snapshot_local_only",
                "rented execution requires the declared exact dependency wheels",
            )
        return _materialize_dependency_snapshot(
            source,
            root,
            python,
            dependency_python,
            base,
            dependency_cache=dependency_cache,
        )
    observed = base or observe_base(python)
    _require_observed_interpreter(python, observed)
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
    _verify_python_requirements(inspected, observed)
    if receipt_path.exists():
        try:
            installed = open_generation(
                root,
                receipt_digest=receipt_digest,
                environment_digest=source.environment_digest,
                reused=True,
                python=python,
            )
        except EnvironmentRefusal as exc:
            if exc.code != "package_environment_content_absent":
                raise
            # The receipt is valid but its generation may have been evicted. Rebuild
            # only an absent exact generation from the freshly verified wheels.
            # An existing incomplete/malformed generation is never overwritten.
            try:
                final.lstat()
            except FileNotFoundError:
                pass
            else:
                raise
        else:
            _verify_prefix_resources(installed.generation, inspected)
            if exact_dependencies:
                _verify_installed_requirements(installed.site_packages, inspected, observed)
            return installed
    reused = final.exists()
    with storage_admission.admit(
        _wheel_write_bound(root, inspected, reused=reused),
        _wheel_archive_write_bound(
            dependency_cache or root / "wheel-cache", inspected, reused=reused
        ),
    ):
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
                _warn_accelerator_stack(
                    {item.wheel.distribution: item.wheel.version for item in inspected}, observed
                )
                _install_with_uv(uv, python, staging, inspected, cache_root=dependency_cache)
                staged = _installed_at(
                    staging,
                    content_digest=content_digest,
                    receipt_bytes=receipt_bytes,
                    receipt_digest=receipt_digest,
                    reused=False,
                    base=observed,
                )
                _verify_prefix_resources(staging, inspected)
                if exact_dependencies:
                    _verify_installed_requirements(staged.site_packages, inspected, observed)
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
            _verify_prefix_resources(installed.generation, inspected)
        if exact_dependencies:
            _verify_installed_requirements(installed.site_packages, inspected, observed)
        return installed


def materialize_captured_builtin(
    source: Input,
    root: Path,
    *,
    captured: InstalledEnvironment,
    python: Path,
    base: base_observation.Observation | None = None,
    dependency_cache: Path | None = None,
) -> InstalledEnvironment:
    """Rebind only the sealed Runtime builtin closure to its ordinary placement.

    This trusted worker path has already captured SDK bytes and the numerical
    supplement. Arbitrary rented packages still require exact dependency wheels.
    """
    from cozy_runtime.internal import _installed_dependencies, builtin_operations

    receipt = parse_environment_receipt(captured.receipt_bytes)
    if (
        len(source.wheels) != 1
        or source.wheels[0].distribution != builtin_operations.NAME
        or source.wheels[0].owner != "project"
        or receipt["project_wheel_digest"] != source.project_wheel_digest
        or receipt["package_wheels"]
        != _receipt(source, source.wheels, captured.content_digest)["package_wheels"]
    ):
        raise EnvironmentRefusal("builtin_snapshot_mismatch", "captured carrier differs")
    held = open_generation(
        captured.generation.parent.parent,
        receipt_digest=captured.receipt_digest,
        environment_digest=receipt["environment_digest"],
        python=python,
    )
    if held.generation != captured.generation or held.content_digest != captured.content_digest:
        raise EnvironmentRefusal("builtin_snapshot_mismatch", "captured SDK generation differs")
    _require_same_interpreter(python, held.python)
    identity = canonical.parse_canonical((held.generation / CONTENT_IDENTITY_NAME).read_bytes())
    manifest = _installed_dependencies.read_manifest(
        held.generation, identity["dependency_snapshot"]
    )
    observed = base or observe_base(python)
    wheels = _verify_inventory(source, observed, venv_precedence=True)
    inspected = _verify_grants(source, wheels, observed)
    # Reserve copied payloads plus directories/metadata even without shared inodes.
    files = manifest["files"]
    nodes = {str(parent) for row in files for parent in PurePosixPath(row["path"]).parents}
    count = len(files) + len(nodes)
    payload = sum(row["length"] for row in files)
    own = _wheel_write_bound(root, inspected)
    with storage_admission.admit(
        storage_admission.Write(
            root, own.bytes + 2 * payload + count * (128 << 10), own.inodes + 2 * count
        ),
        storage_admission.Write(
            dependency_cache or root / "wheel-cache", 2 * payload + (1 << 20), 2 * count + 128
        ),
    ):
        return _materialize_dependency_snapshot(
            source, root, python, held.python, observed, dependency_cache=dependency_cache
        )


def accelerator_warnings(
    versions: Mapping[str, str], base: base_observation.Observation
) -> tuple[str, ...]:
    return tuple(
        f"Degraded accelerator preparation: private {name} {selected}; "
        f"image {base.distribution_versions.get(name, 'none')}. "
        "Startup/disk use may increase; image optimizations may be unavailable. "
        "Native compatibility is checked when loading."
        for name in ("torch", "triton")
        if (selected := versions.get(name)) is not None
        and selected != base.distribution_versions.get(name)
    )


def _warn_accelerator_stack(
    versions: Mapping[str, str], base: base_observation.Observation
) -> None:
    for message in accelerator_warnings(versions, base):
        logging.getLogger(__name__).warning(message)


def _wheel_write_bound(
    root: Path, inspected: Sequence[_Inspected], *, reused: bool = False
) -> storage_admission.Write:
    # Two expanded copies cover uv staging and the installed generation even
    # when reflinks/hardlinks are unavailable. Reserve directory/record overhead
    # too. Precompiled Python bytecode is omitted: import can compile in memory.
    members = [member for wheel in inspected for member in wheel.members] if not reused else []
    nodes = {str(parent) for member in members for parent in PurePosixPath(member.filename).parents}
    count = len(members) + len(nodes)
    expanded = sum(
        member.file_size
        for item in inspected
        if not item.image_seed and not reused
        for member in item.members
    )
    return storage_admission.Write(
        root,
        2 * expanded + (count * (128 << 10)) + (1 << 20),
        2 * count + 128,
    )


def _wheel_archive_write_bound(
    cache: Path, inspected: Sequence[_Inspected], *, reused: bool = False
) -> storage_admission.Write:
    missing = [
        item
        for item in inspected
        if not reused
        and not item.image_seed
        and not _wheel_archive_path(cache, item.wheel).exists()
    ]
    return storage_admission.Write(
        cache, sum(item.wheel.length for item in missing) + (1 << 20), 3 * len(missing) + 8
    )


def _verify_installed_requirements(
    site_packages: Path,
    inspected: Sequence[_Inspected],
    base: base_observation.Observation,
) -> None:
    """Verify actual installed metadata against the captured wheel and requirement roster."""
    installed: dict[str, str] = {}
    for info in site_packages.glob("*.dist-info"):
        path = info / "METADATA"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
            raise EnvironmentRefusal("package_environment_installed_metadata_invalid", info.name)
        metadata = BytesParser().parsebytes(path.read_bytes())
        name, version = _name(str(metadata.get("Name", ""))), str(metadata.get("Version", ""))
        if not name or not version or name in installed:
            raise EnvironmentRefusal("package_environment_installed_metadata_invalid", info.name)
        installed[name] = version
    for item in inspected:
        if installed.get(item.wheel.distribution) != item.wheel.version:
            raise EnvironmentRefusal(
                "package_environment_installed_dependency_changed",
                f"distribution={item.wheel.distribution} expected={item.wheel.version} "
                f"actual={installed.get(item.wheel.distribution)!r}",
            )
    joined = installed
    environment = _marker_environment(base)
    for item in inspected:
        for requirement in item.requirements:
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            actual = joined.get(_name(requirement.name))
            try:
                accepted = actual is not None and requirement.specifier.contains(
                    Version(actual), prereleases=True
                )
            except InvalidVersion:
                accepted = False
            if not accepted:
                raise EnvironmentRefusal(
                    "package_environment_installed_dependency_changed",
                    f"distribution={item.wheel.distribution} "
                    f"requires={requirement} actual={actual!r}",
                )


@dataclass(frozen=True, slots=True)
class LockedRow:
    """One exact hash-pinned requirement row of a locked-requirements export."""

    text: str
    name: str
    version: str
    url: str = ""
    hashes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LockedRequirements:
    """The release's exact requirements export (wire 30): identity over the raw bytes."""

    digest: str
    length: int
    indexes: tuple[str, ...]
    rows: tuple[LockedRow, ...]


_INDEX_DIRECTIVES = ("--index-url", "--extra-index-url")


def read_locked_requirements(raw: bytes) -> LockedRequirements:
    """Admit one locked-requirements export.

    The only admitted rows are blank/comment lines, https index directives, and exact
    `name==version` requirements carrying at least one `--hash=sha256:`. Anything a resolver
    would have freedom over — a range, a bare name, a path or URL row, an editable — refuses.
    """

    from cozy_runtime.protocol import weights_limits

    ceiling = int(weights_limits.MAX_LOCKED_REQUIREMENTS_BYTES)
    if not 0 < len(raw) <= ceiling:
        raise EnvironmentRefusal(
            "package_environment_lock_invalid", f"{len(raw)} B is empty or over {ceiling} B"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EnvironmentRefusal("package_environment_lock_invalid", str(exc)) from exc
    logical: list[str] = []
    pending = ""
    for line in text.splitlines():
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        logical.append(pending + line)
        pending = ""
    if pending:
        logical.append(pending)
    indexes: list[str] = []
    rows: list[LockedRow] = []
    prior = ""
    for number, row in enumerate((entry.strip() for entry in logical), start=1):

        def refuse(detail: str, number: int = number) -> EnvironmentRefusal:
            return EnvironmentRefusal(
                "package_environment_lock_invalid", f"line {number}: {detail}"
            )

        if not row or row.startswith("#"):
            continue
        directive, _, address = row.partition(" ")
        if directive in _INDEX_DIRECTIVES:
            address = address.strip()
            if not address.startswith("https://") and not address.startswith(
                ("http://127.0.0.1:", "http://127.0.0.1/", "http://[::1]:", "http://[::1]/")
            ):
                raise refuse(f"index {address!r} is not an https or loopback URL")
            indexes.append(address)
            continue
        if row.startswith("-"):
            raise refuse(f"option {row.split()[0]!r} is outside the locked vocabulary")
        requirement_text, *options = row.split(" --")
        options = ["--" + option.strip() for option in options]
        if not options or any(not option.startswith("--hash=sha256:") for option in options):
            raise refuse("every requirement row carries only --hash=sha256: options")
        if any(len(option) != len("--hash=sha256:") + 64 for option in options):
            raise refuse("a --hash value is not 64 hex")
        try:
            requirement = Requirement(requirement_text.strip())
        except InvalidRequirement as exc:
            raise refuse(str(exc)) from exc
        clauses = list(requirement.specifier)
        name = _NORMALIZE.sub("-", requirement.name).lower()
        if requirement.extras:
            raise refuse("locked requirements cannot add unpinned extras")
        if requirement.url:
            parsed = urlsplit(requirement.url)
            selected = wheel_name_version(unquote(Path(parsed.path).name))
            if (
                parsed.scheme not in {"https", "http"}
                or (parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1"})
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
                or selected is None
                or selected[0] != name
                or clauses
            ):
                raise refuse("direct requirement must name one exact public wheel")
            version = selected[1]
        elif len(clauses) == 1 and clauses[0].operator == "==":
            version = clauses[0].version
        else:
            raise refuse("requirement is not one exact wheel or == pin")
        if name <= prior:
            raise refuse(f"{name!r} is out of order or repeats a name")
        prior = name
        hashes = tuple(option.removeprefix("--hash=") for option in options)
        rows.append(
            LockedRow(
                text=row, name=name, version=version, url=requirement.url or "", hashes=hashes
            )
        )
    if not rows:
        raise EnvironmentRefusal("package_environment_lock_invalid", "the export pins nothing")
    return LockedRequirements(
        digest=_sha(raw),
        length=len(raw),
        indexes=tuple(indexes),
        rows=tuple(rows),
    )


def wheel_receipt_digest(
    base: base_observation.Observation,
    *,
    wheels: Sequence[Wheel],
    environment_digest: str,
    dependency_snapshot: Mapping[str, Any] | None = None,
) -> str:
    """Derive a wheel installation receipt from placement and frozen snapshot facts.

    Materialization sorts the complete inventory by distribution before publishing
    its receipt, including the project wheel among the dependency wheels.
    A captured local SDK adds its immutable snapshot ref to the same content identity.
    """

    _require_digest(environment_digest, "environment_digest")
    wheels = sorted(wheels, key=lambda wheel: wheel.distribution)
    project = next((wheel for wheel in wheels if wheel.owner == "project"), None)
    if project is None:
        raise EnvironmentRefusal(
            "package_environment_project_wheel_absent", "no project wheel fact"
        )
    identity = _content_identity(wheels, base.interpreter_identity, base.uv_version)
    if dependency_snapshot is not None:
        identity["dependency_snapshot"] = dict(dependency_snapshot)
    content_bytes = canonical.write(identity)
    receipt = {
        "content_digest": _sha(content_bytes),
        "environment_digest": environment_digest,
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
        "project_wheel_digest": project.digest,
    }
    return _sha(canonical.write(receipt))


def locked_identity(
    base: base_observation.Observation,
    *,
    locked_digest: str,
    locked_length: int,
    environment_digest: str,
) -> tuple[str, str]:
    """(content_digest, receipt_digest) for a locked generation, from placement facts alone.

    Deterministic in the placement's Environment.locked_requirements ref plus the observed
    base, so a serving worker reopens the generation its prepare published without holding
    the export bytes (proto-030 deletes them after venv creation)."""

    content_bytes = canonical.write(
        _locked_content_identity(locked_digest, base.interpreter_identity, base.uv_version)
    )
    content_digest = _sha(content_bytes)
    receipt_bytes = canonical.write(
        _locked_receipt(environment_digest, content_digest, locked_digest, locked_length)
    )
    return content_digest, _sha(receipt_bytes)


def materialize_locked(
    locked: LockedRequirements,
    install_rows: Sequence[LockedRow],
    root: Path,
    *,
    environment_digest: str,
    python: Path = Path(sys.executable),
    base: base_observation.Observation | None = None,
    dependency_python: Path | None = None,
    dependency_cache: Path | None = None,
) -> InstalledEnvironment:
    """Materialize one venv from the release's exact hash-pinned export (xs-017).

    `install_rows` contains the complete export. Every artifact must match its selected
    hash; indexes have no authority over bytes. A shared uv cache reuses immutable artifacts.
    Private metadata closure is checked before import, including on reuse. A captured local
    SDK closure is copied into the generation and becomes part of its content identity.
    """

    from cozy_runtime.internal import package_requirements

    _require_digest(environment_digest, "environment_digest")
    if not locked.indexes:
        raise EnvironmentRefusal(
            "package_environment_lock_index_absent",
            "the export names no --index-url/--extra-index-url",
        )
    observed = base or observe_base(python)
    _require_observed_interpreter(python, observed)
    if dependency_python is not None:
        _require_same_interpreter(python, dependency_python)
    uv = _uv_binary()
    current_uv = _observed_uv_version(uv)
    if current_uv != observed.uv_version:
        raise EnvironmentRefusal(
            "package_environment_uv_changed",
            f"observed={observed.uv_version} current={current_uv}",
        )
    content_bytes = canonical.write(
        _locked_content_identity(locked.digest, observed.interpreter_identity, observed.uv_version)
    )
    content_digest = _sha(content_bytes)
    receipt_bytes = canonical.write(
        _locked_receipt(environment_digest, content_digest, locked.digest, locked.length)
    )
    receipt_digest = _sha(receipt_bytes)
    contents = root / "contents"
    receipts = root / "receipts"
    final = contents / content_digest.removeprefix("sha256:")
    receipt_path = receipts / f"{receipt_digest.removeprefix('sha256:')}.json"
    if dependency_python is None and receipt_path.exists():
        installed = open_generation(
            root,
            receipt_digest=receipt_digest,
            environment_digest=environment_digest,
            reused=True,
            python=python,
        )
        package_requirements.validate(installed.site_packages, python, observed)
        _release_locked_wheels(install_rows)
        return installed
    reused = dependency_python is None and final.exists()
    with storage_admission.admit(storage_admission.Write(root, 1 << 20, 128)):
        _require_owned_root(root)
        _remove_dead_staging(root)
        contents.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
        contents.chmod(0o711)
        receipts.chmod(0o700)
    if not reused:
        staging = root / f".environment-staging-{os.getpid()}-{uuid.uuid4().hex}"
        cache = root / f".uv-cache-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            manifest = b""

            def copy_dependencies(target: Path) -> None:
                nonlocal manifest
                from cozy_runtime.internal import _installed_dependencies

                assert dependency_python is not None
                manifest = _installed_dependencies.snapshot(
                    dependency_python,
                    target,
                    {row.name for row in install_rows},
                    cache=dependency_cache,
                )

            _warn_accelerator_stack({row.name: row.version for row in install_rows}, observed)
            _install_locked(
                uv,
                python,
                staging,
                cache,
                locked.indexes,
                install_rows,
                dependencies=copy_dependencies if dependency_python is not None else None,
                cache_root=dependency_cache,
            )
            if dependency_python is not None:
                from cozy_runtime.internal import _installed_dependencies

                _installed_dependencies.verify_files(staging, manifest)
                identity = _locked_content_identity(
                    locked.digest, observed.interpreter_identity, observed.uv_version
                )
                identity["dependency_snapshot"] = {
                    "digest": _sha(manifest),
                    "length": len(manifest),
                }
                content_bytes = canonical.write(identity)
                content_digest = _sha(content_bytes)
                receipt_bytes = canonical.write(
                    _locked_receipt(
                        environment_digest, content_digest, locked.digest, locked.length
                    )
                )
                receipt_digest = _sha(receipt_bytes)
                final = contents / content_digest.removeprefix("sha256:")
                receipt_path = receipts / f"{receipt_digest.removeprefix('sha256:')}.json"
                _write_once(staging / _installed_dependencies.MANIFEST_NAME, manifest)
            # Validates the staged venv's shape; the typed refusal is the point here.
            staged = _installed_at(
                staging,
                content_digest=content_digest,
                receipt_bytes=receipt_bytes,
                receipt_digest=receipt_digest,
                reused=False,
                base=observed,
            )
            package_requirements.validate(staged.site_packages, python, observed)
            with storage_admission.admit(storage_admission.Write(root, 1 << 20, 128)):
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
        finally:
            if cache.exists():
                _remove_tree(cache)
    with storage_admission.admit(storage_admission.Write(root, 1 << 20, 128)):
        _publish_receipt(receipt_path, receipt_bytes, receipts)
    installed = open_generation(
        root,
        receipt_digest=receipt_digest,
        environment_digest=environment_digest,
        reused=reused,
        python=python,
    )
    if reused:
        package_requirements.validate(installed.site_packages, python, observed)
    _release_locked_wheels(install_rows)
    return installed


def _require_dependency_cache_root(cache: Path) -> None:
    if not cache.is_absolute() or cache.resolve() != cache:
        raise EnvironmentRefusal("package_environment_dependency_cache_invalid", str(cache))
    try:
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = cache.lstat()
    except OSError as exc:
        raise EnvironmentRefusal(
            "package_environment_dependency_cache_invalid", str(cache)
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or (
        os.name == "posix" and (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700)
    ):
        raise EnvironmentRefusal("package_environment_dependency_cache_invalid", str(cache))


def _install_locked(
    uv: Path,
    python: Path,
    staging: Path,
    cache: Path,
    indexes: Sequence[str],
    install_rows: Sequence[LockedRow],
    *,
    dependencies: Callable[[Path], None] | None = None,
    cache_root: Path | None = None,
) -> None:
    if storage_admission.enabled():
        _install_locked_bounded(
            uv, python, staging, install_rows, dependencies=dependencies, cache_root=cache_root
        )
        return
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    requirements = cache / "locked-requirements.txt"
    requirements.write_text("".join(row.text + "\n" for row in install_rows))
    if cache_root is not None:
        _require_dependency_cache_root(cache_root)
    shared_cache = (cache_root or staging.parent) / "uv-cache"
    shared_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    scoped = ["--cache-dir", str(shared_cache)]
    commands = [
        [
            str(uv),
            "venv",
            "--no-project",
            "--no-config",
            "--offline",
            "--no-python-downloads",
            "--link-mode",
            "hardlink",
            "--python",
            str(python),
            *scoped,
            str(staging),
        ],
        [
            str(uv),
            "pip",
            "install",
            "--no-config",
            *scoped,
            "--require-hashes",
            "--no-deps",
            "--no-build",
            "--link-mode",
            "hardlink",
            "--compile-bytecode",
            "--python",
            str(staging / "bin" / "python"),
            "--index-url",
            indexes[0],
            *(part for index in indexes[1:] for part in ("--extra-index-url", index)),
            "--requirements",
            str(requirements),
        ],
    ]
    for index, command in enumerate(commands):
        if index == 1 and dependencies is not None:
            dependencies(staging)
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            env=package_install_environment(),
        )
        if result.returncode != 0:
            detail = result.stderr.strip()[-2000:] or result.stdout.strip()[-2000:]
            raise EnvironmentRefusal(
                "package_environment_uv_failed",
                detail or f"uv exited {result.returncode}",
            )
    _rewrite_installed_launchers(staging)


def _locked_operation(rows: Sequence[LockedRow]) -> str:
    return "locked-" + hashlib.sha256("\n".join(row.text for row in rows).encode()).hexdigest()


def _release_locked_wheels(rows: Sequence[LockedRow]) -> None:
    if storage_admission.enabled():
        storage_admission.release_wheels(_locked_operation(rows))


def _install_locked_bounded(
    uv: Path,
    python: Path,
    staging: Path,
    rows: Sequence[LockedRow],
    *,
    dependencies: Callable[[Path], None] | None = None,
    cache_root: Path | None = None,
) -> None:
    observed = observe_base(python)
    inspected = fetch_locked_wheels(
        rows, staging.parent, observed, cache_root=cache_root, python=python
    )
    with storage_admission.admit(
        _wheel_write_bound(staging.parent, inspected),
        _wheel_archive_write_bound(cache_root or staging.parent / "wheel-cache", inspected),
    ):
        _install_with_uv(
            uv, python, staging, inspected, dependencies=dependencies, cache_root=cache_root
        )


def fetch_locked_wheels(
    rows: Sequence[LockedRow],
    root: Path,
    observed: base_observation.Observation,
    *,
    cache_root: Path | None = None,
    python: Path = Path(sys.executable),
) -> list[_Inspected]:
    """Fetch exact dependency artifacts through the existing bounded storage owner."""
    inspected: list[_Inspected] = []
    operation = _locked_operation(rows)
    for row in rows:
        if not row.url or len(row.hashes) != 1:
            raise EnvironmentRefusal(
                "package_environment_lock_wheels_required",
                "refresh preparation facts from Tensorhub to obtain the exact selected wheels",
            )
        if cache_root is not None:
            from cozy_runtime.internal import _image_wheel_cache

            _require_dependency_cache_root(cache_root)
            seeded = _image_wheel_cache.read(row.hashes[0], observed)
            if (
                seeded is not None
                and (seeded.wheel.distribution, seeded.wheel.version) == (row.name, row.version)
                and _image_wheel_cache.available(seeded, python, cache_root)
            ):
                inspected.append(seeded)
                continue
        filename = unquote(Path(urlsplit(row.url).path).name)
        expected = root / ".stage" / operation / "wheels" / filename
        cached = (
            cache_root / ".stage" / row.hashes[0].removeprefix("sha256:") / filename
            if cache_root is not None
            else None
        )
        if (
            cache_root is not None
            and cached is not None
            and (cached.exists() or cached.is_symlink())
        ):
            _require_dependency_cache_root(cache_root)
            _require_wheel_archive_directory(cached.parent.parent)
            _require_wheel_archive_directory(cached.parent)
            length = _verify_wheel_archive(cached, row.hashes[0])
            path = expected = cached
        elif storage_admission.enabled():
            path, length = storage_admission.fetch_wheel(
                operation, filename, row.hashes[0], row.url
            )
        else:
            # Local package install roots differ. Retain each exact archive only once
            # under the machine cache, independently of the project's full closure.
            if cache_root is not None:
                _require_dependency_cache_root(cache_root)
                expected = cache_root / ".stage" / row.hashes[0].removeprefix("sha256:") / filename
            path, length = _fetch_local_locked_wheel(expected, row)
        if path != expected:
            raise EnvironmentRefusal(
                "package_environment_wheel_invalid", "wheel staging path changed"
            )
        verified = VerifiedFile.record(path, row.hashes[0], length)
        item, verified = _measure_inspected_wheel(
            path, "custom", observed, allow_base_owned=True, verified=verified
        )
        if (item.wheel.distribution, item.wheel.version, verified.digest, verified.length) != (
            row.name,
            row.version,
            row.hashes[0],
            length,
        ):
            raise EnvironmentRefusal(
                "package_environment_wheel_invalid", "wheel differs from exact lock"
            )
        _require_compatible_tags(item.wheel, observed)
        inspected.append(item)
    return inspected


def _fetch_local_locked_wheel(path: Path, row: LockedRow) -> tuple[Path, int]:
    """The local disk owner admits downloads before bytes are written, as on a rental."""
    if path.exists():
        if path.is_symlink() or not path.is_file() or _hash_file(path) != row.hashes[0]:
            raise EnvironmentRefusal("package_environment_wheel_invalid", "cached wheel changed")
        return path, path.stat().st_size

    from cozy_runtime.internal import egress

    _require_public_wheel_url(row.url)
    return egress.fetch_public_wheel(path, row.url, row.hashes[0], MAX_WHEEL_BYTES)


def _require_public_wheel_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.scheme not in {"https", "http"}
        or (parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1"})
    ):
        raise EnvironmentRefusal(
            "package_environment_lock_invalid", "dependency URL must be public and credential-free"
        )


def read_dependency_requirements(raw: bytes) -> LockedRequirements:
    """Private dependency transport selects exact public wheel artifacts before submission."""
    locked = read_locked_requirements(raw)
    if locked.indexes or len(locked.rows) > MAX_PACKAGE_WHEELS:
        raise EnvironmentRefusal(
            "package_environment_lock_invalid", "dependency lock must contain only exact wheel rows"
        )
    for row in locked.rows:
        _require_public_wheel_url(row.url)
        requirement = Requirement(row.text.split(" --", 1)[0])
        if (
            requirement.marker is not None
            or len(row.hashes) != 1
            or not _DIGEST.fullmatch(row.hashes[0])
        ):
            raise EnvironmentRefusal(
                "package_environment_lock_invalid",
                "dependency wheel must have one exact sha256 and no marker",
            )
    return locked


def _locked_content_identity(
    locked_digest: str, interpreter: Mapping[str, str], uv_version: str
) -> dict[str, Any]:
    return {
        "format": LOCKED_CONTENT_FORMAT,
        "interpreter": dict(interpreter),
        "locked_requirements_digest": locked_digest,
        "uv_version": uv_version,
    }


def _locked_receipt(
    environment_digest: str, content_digest: str, locked_digest: str, locked_length: int
) -> dict[str, Any]:
    return {
        "content_digest": content_digest,
        "environment_digest": environment_digest,
        "format": LOCKED_RECEIPT_FORMAT,
        "locked_requirements": {"digest": locked_digest, "length": locked_length},
    }


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


def _interpreter_layout(python: Path) -> dict[str, str]:
    """Measure the selected interpreter's real prefix and import layout."""
    if not python.is_absolute() or not python.is_file():
        raise EnvironmentRefusal("package_environment_python_invalid", str(python))
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json,sys,sysconfig; print(json.dumps({"
            "'base_prefix':sys.base_prefix,'prefix':sys.prefix,'executable':sys.executable,"
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
    fields = {"base_prefix", "prefix", "purelib", "executable"}
    if (
        result.returncode != 0
        or not isinstance(observed, dict)
        or set(observed) != fields
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in observed.values()
        )
    ):
        raise EnvironmentRefusal("package_environment_python_invalid", str(python))
    try:
        selected = Path(observed["executable"]).samefile(python)
    except OSError:
        selected = False
    if not selected:
        raise EnvironmentRefusal("package_environment_python_invalid", str(python))
    return observed


def _require_observed_interpreter(python: Path, observed: base_observation.Observation) -> None:
    from cozy_runtime.internal import python_interpreters

    actual = python_interpreters.probe(python)
    if (actual.abi, actual.version) != (observed.python_abi, observed.python_full_version):
        raise EnvironmentRefusal(
            "package_environment_python_changed",
            f"observed {observed.python_full_version}/{observed.python_abi}; "
            f"actual {actual.version}/{actual.abi}",
        )
    if ".".join(actual.version.split(".")[:2]) not in python_interpreters.supported_minors():
        raise EnvironmentRefusal("package_environment_python_unsupported", actual.version)


def _require_same_interpreter(python: Path, dependency_python: Path) -> None:
    from cozy_runtime.internal import python_interpreters

    selected = python_interpreters.probe(python)
    source = python_interpreters.probe(dependency_python)
    if (selected.abi, selected.version) != (source.abi, source.version):
        raise EnvironmentRefusal(
            "package_environment_dependency_python_mismatch",
            f"selected Python {selected.version}/{selected.abi} cannot copy dependency "
            f"environment {source.version}/{source.abi}",
        )


def is_base_interpreter(python: Path) -> bool:
    """Identify the control interpreter independently of a private package venv."""
    layout = _interpreter_layout(python)
    return layout["prefix"] == layout["base_prefix"]


def open_existing_environment(python: Path, content_digest: str) -> InstalledEnvironment:
    """Open one ordinary local venv without installing or changing it."""

    _require_digest(content_digest, "content_digest")
    if not python.is_absolute() or python.parent.name != "bin" or not python.is_file():
        raise EnvironmentRefusal("package_environment_python_invalid", str(python))
    _require_observed_interpreter(python, observe_base(python))
    generation = python.parent.parent
    if not (generation / "pyvenv.cfg").is_file():
        raise EnvironmentRefusal("package_environment_venv_invalid", str(generation))
    observed = _interpreter_layout(python)
    site = Path(observed["purelib"])
    if (
        observed.get("prefix") != str(generation)
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
    if "dependency_snapshot" in identity:
        from cozy_runtime.internal import _installed_dependencies

        _installed_dependencies.read_manifest(generation, identity["dependency_snapshot"])
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
    from cozy_runtime.internal import python_interpreters

    actual = python_interpreters.probe(environment_python)
    if actual.abi != python_abi or actual.version != identity["interpreter"].get(
        "python_full_version"
    ):
        raise EnvironmentRefusal("package_environment_python_changed", str(environment_python))
    if ".".join(actual.version.split(".")[:2]) not in python_interpreters.supported_minors():
        raise EnvironmentRefusal("package_environment_python_unsupported", actual.version)
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
    if isinstance(value, dict) and value.get("format") == LOCKED_RECEIPT_FORMAT:
        if set(value) != {"content_digest", "environment_digest", "format", "locked_requirements"}:
            raise EnvironmentRefusal("package_environment_receipt_invalid", "wrong fields")
        for field in ("content_digest", "environment_digest"):
            item = value.get(field)
            if not isinstance(item, str):
                raise EnvironmentRefusal(
                    "package_environment_receipt_invalid", f"{field} is not a digest"
                )
            _require_digest(item, field)
        ref = value.get("locked_requirements")
        if (
            not isinstance(ref, dict)
            or set(ref) != {"digest", "length"}
            or not isinstance(ref.get("digest"), str)
            or not isinstance(ref.get("length"), int)
            or isinstance(ref.get("length"), bool)
            or ref["length"] <= 0
        ):
            raise EnvironmentRefusal(
                "package_environment_receipt_invalid", "locked_requirements is not an exact ref"
            )
        _require_digest(str(ref["digest"]), "locked_requirements.digest")
        return value
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
        if wheel.import_roots != tuple(sorted(set(wheel.import_roots))) or not all(
            valid_import_root(root) for root in wheel.import_roots
        ):
            raise EnvironmentRefusal("package_environment_wheel_invalid", wheel.filename)
        _require_compatible_tags(wheel, base)
        for root in wheel.import_roots:
            providers = base.root_providers.get(root)
            if providers is not None and "python-stdlib" in providers:
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
        if grant.image_seed:
            from cozy_runtime.internal import _image_wheel_cache

            seeded = _image_wheel_cache.read(wheel.digest, base)
            if seeded is None or seeded.wheel != wheel or seeded.path != grant.path:
                raise EnvironmentRefusal("package_environment_image_seed_invalid", wheel.digest)
            inspected.append(seeded)
            continue
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
            # a second time after admission already verified it.
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
    import_owners: dict[PurePosixPath, tuple[str, bool]] = {}
    resource_files: set[PurePosixPath] = set()
    resource_directories: set[PurePosixPath] = set()
    python_major, python_minor = _target_python(base)
    site_packages = PurePosixPath("lib", f"python{python_major}.{python_minor}", "site-packages")
    for item in inspected:
        for member in item.members:
            installed = _install_path(member.filename)
            resource = (
                site_packages / installed
                if installed is not None
                else _prefix_resource(member.filename)
            )
            if member.is_dir() or resource is None:
                continue
            if (
                resource in resource_files
                or resource in resource_directories
                or any(parent in resource_files for parent in resource.parents)
            ):
                raise EnvironmentRefusal("package_environment_resource_collision", str(resource))
            resource_files.add(resource)
            resource_directories.update(resource.parents)
        for prefix, concrete in _import_ownership(item, site_packages).items():
            if len(prefix.parts) == 1:
                providers = base.root_providers.get(prefix.name)
                if providers is not None and "python-stdlib" in providers:
                    raise _base_import_collision(item.wheel, prefix.name, providers)
            prior = import_owners.get(prefix)
            if prior is not None and prior[0] != item.wheel.distribution:
                if concrete or prior[1]:
                    raise EnvironmentRefusal(
                        "package_environment_import_root_duplicate",
                        f"module={prefix} distributions={prior[0]},{item.wheel.distribution}",
                    )
            else:
                import_owners[prefix] = (item.wheel.distribution, concrete)
    return tuple(inspected)


def _import_ownership(item: _Inspected, site_packages: PurePosixPath) -> dict[PurePosixPath, bool]:
    """Separate additive PEP 420 directories from modules and regular packages."""
    ownership: dict[PurePosixPath, bool] = {}
    for member in item.members:
        path = _install_path(member.filename)
        if path is None:
            resource = _prefix_resource(member.filename)
            if resource is not None and resource.is_relative_to(site_packages):
                path = resource.relative_to(site_packages)
        if member.is_dir() or path is None or not path.name.endswith((".py", *_NATIVE_SUFFIXES)):
            continue
        name = path.name.partition(".")[0]
        module = path.parent if name == "__init__" else path.with_name(name)
        if not module.parts or not valid_import_root(module.parts[0]):
            continue
        ownership[module] = True
        for parent in module.parents:
            if parent.parts:
                ownership.setdefault(parent, False)
    return ownership


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
                        wheel.owner == "project"
                        and member.compress_size > 0
                        and member.file_size > member.compress_size * MAX_WHEEL_EXPANSION
                    )
                ):
                    raise EnvironmentRefusal(
                        "package_environment_wheel_expansion_cap", wheel.filename
                    )
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise EnvironmentRefusal("package_environment_wheel_symlink", name)
                if member.is_dir():
                    continue
                installed = _install_path(name)
                prefix_resource = _prefix_resource(name)
                if installed is None and ".data/" in name and not prefix_resource:
                    raise EnvironmentRefusal(
                        "package_environment_wheel_install_scheme_forbidden", name
                    )
                suffix = posix.suffix.lower()
                # Prebuilt wheels may carry inert source/header resources. Every
                # installation uses --no-build; these files never trigger a build.
                with archive.open(member) as body:
                    magic = body.read(4)
                native = suffix in _NATIVE_SUFFIXES or magic == b"\x7fELF"
                if native:
                    native_path = installed if installed is not None else prefix_resource
                    assert native_path is not None
                    native_members.append(native_path.as_posix())
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
            for metadata_name in (dist_infos[0], wheel_infos[0]):
                if archive.getinfo(metadata_name).file_size > MAX_METADATA_BYTES:
                    raise EnvironmentRefusal(
                        "package_environment_wheel_metadata_oversized", wheel.filename
                    )
            metadata = archive.read(dist_infos[0])
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
            if root_is_pure not in {"true", "false"}:
                raise EnvironmentRefusal(
                    "package_environment_wheel_metadata_mismatch", "invalid Root-Is-Purelib"
                )
            # FFI libraries can be ABI-neutral and installed into purelib (NVIDIA's
            # wheels use both). Native bytes still require a concrete target platform.
            if native_members and any(tag.platform == "any" for tag in declared_tags):
                raise EnvironmentRefusal(
                    "package_environment_native_wheel_tag_not_concrete",
                    f"{wheel.filename}: native wheels require concrete platform tags",
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
                if providers is not None and "python-stdlib" in providers:
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
    recorded: set[str] = set()
    try:
        with (
            archive.open(record_name) as source,
            io.TextIOWrapper(source, encoding="utf-8", newline="") as text,
        ):
            for row in csv.reader(text):
                if len(row) != 3 or not row[0] or row[0] in recorded or row[0] not in names:
                    raise EnvironmentRefusal(
                        "package_environment_wheel_record_invalid", record_name
                    )
                recorded.add(row[0])
                if row[0] == record_name:
                    if row[1] or row[2]:
                        raise EnvironmentRefusal(
                            "package_environment_wheel_record_invalid", record_name
                        )
                    continue
                if not row[1] or not row[2].isdigit():
                    raise EnvironmentRefusal("package_environment_wheel_record_unhashed", row[0])
                algorithm, separator, encoded = row[1].partition("=")
                expected = archive.getinfo(row[0]).file_size
                if int(row[2]) != expected:
                    raise EnvironmentRefusal("package_environment_wheel_record_mismatch", row[0])
                digest, count = hashlib.sha256(), 0
                with archive.open(row[0]) as payload:
                    while chunk := payload.read(1 << 20):
                        count += len(chunk)
                        if count > expected:
                            raise EnvironmentRefusal(
                                "package_environment_wheel_record_mismatch", row[0]
                            )
                        digest.update(chunk)
                got = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
                if algorithm != "sha256" or not separator or encoded != got or count != expected:
                    raise EnvironmentRefusal("package_environment_wheel_record_mismatch", row[0])
    except (UnicodeError, csv.Error) as exc:
        raise EnvironmentRefusal("package_environment_wheel_record_invalid", record_name) from exc
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
    platforms = {"linux_x86_64"}
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
    platforms = sorted(_target_platforms(base))
    tags = set(cpython_tags((major, minor), abis=[interpreter], platforms=platforms))
    tags.update(compatible_tags((major, minor), interpreter=interpreter, platforms=platforms))
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
            "implementation_version": base.python_full_version,
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "python_full_version": base.python_full_version,
            "python_version": f"{major}.{minor}",
            "sys_platform": "linux",
        }
    )
    return environment


def _verify_python_requirements(
    inspected: Sequence[_Inspected],
    base: base_observation.Observation,
) -> None:
    """Validate the complete selected wheel closure without ambient image providers."""
    from cozy_runtime.internal import package_requirements

    available = {
        item.wheel.distribution: package_requirements._Distribution(
            item.wheel.version, item.requires_python or "", item.requirements
        )
        for item in inspected
    }
    package_requirements.validate_closure(available, set(available), base)


def _install_path(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    first = path.parts[0]
    if first.endswith(".data"):
        if len(path.parts) < 3 or path.parts[1] not in {"purelib", "platlib"}:
            return None
        return PurePosixPath(*path.parts[2:])
    return path


def _prefix_resource(name: str) -> PurePosixPath | None:
    """Map standard wheel data schemes into this CPython venv's prefix."""
    parts = PurePosixPath(name).parts
    if len(parts) < 3 or not parts[0].endswith(".data"):
        return None
    if parts[1] == "data":
        return PurePosixPath(*parts[2:])
    if parts[1] == "scripts":
        return PurePosixPath("bin", *parts[2:])
    if parts[1] == "headers":
        distribution = _name(parts[0].removesuffix(".data").rsplit("-", 1)[0])
        return PurePosixPath(
            "include",
            "site",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            distribution,
            *parts[2:],
        )
    return None


def _relocatable_python_script(body: bytes) -> bytes:
    return b"#!/bin/sh\n'''exec' \"${0%/*}/python\" \"$0\" \"$@\"\n' '''\n" + body


def _verify_prefix_resources(generation: Path, inspected: Sequence[_Inspected]) -> None:
    for item in inspected:
        resources = [
            (member, resource)
            for member in item.members
            if not member.is_dir() and (resource := _prefix_resource(member.filename)) is not None
        ]
        if not resources:
            continue
        with zipfile.ZipFile(item.path) as archive:
            for member, resource in resources:
                target = generation / resource
                try:
                    info = target.lstat()
                    if not stat.S_ISREG(info.st_mode) or target.resolve() != target:
                        raise OSError("resource path or size changed")
                    digest = hashlib.sha256()
                    length = 0
                    with archive.open(member) as source:
                        if resource.parts[0] == "bin" and source.read(8).startswith(b"#!python"):
                            source.seek(0)
                            source.readline()
                            prefix = _relocatable_python_script(b"")
                            digest.update(prefix)
                            length += len(prefix)
                        else:
                            source.seek(0)
                        while chunk := source.read(1 << 20):
                            digest.update(chunk)
                            length += len(chunk)
                    if (
                        info.st_size != length
                        or _hash_file(target) != "sha256:" + digest.hexdigest()
                    ):
                        raise OSError("resource bytes changed")
                except OSError as exc:
                    raise EnvironmentRefusal(
                        "package_environment_installed_resource_changed", str(resource)
                    ) from exc


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


def _wheel_archive_path(cache: Path, wheel: Wheel) -> Path:
    return cache / ".stage" / wheel.digest.removeprefix("sha256:") / wheel.filename


def _require_wheel_archive_directory(directory: Path) -> None:
    # Older local downloads created intermediate parents with the caller's umask.
    # Group-writable parents are safe only beneath an owned 0700 directory.
    if not directory.is_absolute() or directory.resolve() != directory:
        raise EnvironmentRefusal("package_environment_dependency_cache_invalid", str(directory))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = directory.lstat()
    parent = directory.parent.stat()
    private_parent = parent.st_uid == os.geteuid() and stat.S_IMODE(parent.st_mode) == 0o700
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or (info.st_mode & 0o022 and not private_parent)
    ):
        raise EnvironmentRefusal("package_environment_dependency_cache_invalid", str(directory))


def _verify_wheel_archive(path: Path, digest: str, length: int | None = None) -> int:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or path.resolve() != path
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o222
            or (length is not None and info.st_size != length)
            or not 0 < info.st_size <= MAX_WHEEL_BYTES
            or _hash_file(path) != digest
        ):
            raise OSError("cached archive differs from the verified wheel")
    except OSError as exc:
        raise EnvironmentRefusal("package_environment_wheel_invalid", str(path)) from exc
    return info.st_size


def _cache_wheel_archive(cache: Path, item: _Inspected) -> Path:
    """Give uv a stable input path across closures; never modify its cache internals."""
    target = _wheel_archive_path(cache, item.wheel)
    for directory in (cache / ".stage", target.parent):
        _require_wheel_archive_directory(directory)

    if target.exists() or target.is_symlink():
        _verify_wheel_archive(target, item.wheel.digest, item.wheel.length)
        return target
    temporary = target.parent / (".archive-" + uuid.uuid4().hex)
    try:
        source_info = item.path.lstat()
        linked = False
        if (
            stat.S_ISREG(source_info.st_mode)
            and source_info.st_uid == os.geteuid()
            and not source_info.st_mode & 0o222
            and item.path.resolve() == item.path
        ):
            try:
                os.link(item.path, temporary)
                linked = True
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
                    raise
        if not linked:
            with item.path.open("rb") as source, temporary.open("xb") as destination:
                shutil.copyfileobj(source, destination, 4 << 20)
                destination.flush()
                os.fsync(destination.fileno())
            temporary.chmod(0o444)
        if (
            temporary.stat().st_size != item.wheel.length
            or _hash_file(temporary) != item.wheel.digest
        ):
            raise EnvironmentRefusal("package_environment_wheel_invalid", str(item.path))
        # A concurrent exact publisher may win; verify its file before using it.
        with suppress(FileExistsError):
            os.link(temporary, target)
        _verify_wheel_archive(target, item.wheel.digest, item.wheel.length)
        _fsync_dir(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _install_with_uv(
    uv: Path,
    python: Path,
    staging: Path,
    inspected: Sequence[_Inspected],
    supplements: Sequence[str] = (),
    *,
    dependencies: Callable[[Path], None] | None = None,
    cache_root: Path | None = None,
) -> None:
    if cache_root is not None:
        _require_dependency_cache_root(cache_root)
    archive_cache = cache_root or staging.parent / "wheel-cache"
    _require_dependency_cache_root(archive_cache)
    archives = [
        _cache_wheel_archive(archive_cache, item)
        for item in sorted(inspected, key=lambda item: item.wheel.digest)
        if not item.image_seed
    ]
    cache = (cache_root or staging.parent) / "uv-cache"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    commands = [
        [
            str(uv),
            "venv",
            "--no-project",
            "--no-config",
            "--offline",
            "--no-python-downloads",
            "--link-mode",
            "hardlink",
            "--python",
            str(python),
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
            "hardlink",
            "--python",
            str(staging / "bin" / "python"),
            *(str(path) for path in archives),
        ],
    ]
    if not archives:
        commands.pop()
    seeded = [item for item in inspected if item.image_seed]
    if seeded:
        from cozy_runtime.internal import _image_wheel_cache

        requirements = staging.parent / (".seed-requirements-" + uuid.uuid4().hex)
        requirements.write_text("".join(_image_wheel_cache.requirement(item) for item in seeded))
        # One command handles public image payloads; archives/private packages keep
        # their ordinary hardlink cache and are installed into the same isolated venv.
        commands.append(_image_wheel_cache.command(uv, staging / "bin/python", requirements))
    if supplements:
        # Only the fixed built-in local capture can request a numpy supplement.
        # Private package closures are complete and never reach this compatibility lane.
        commands.append(
            [
                str(uv),
                "pip",
                "install",
                "--no-config",
                "--no-deps",
                "--no-build",
                "--link-mode",
                "hardlink",
                "--python",
                str(staging / "bin" / "python"),
                *supplements,
            ]
        )
    for command in commands:
        if "--cache-dir" not in command:
            command.extend(["--cache-dir", str(cache)])
    try:
        _run_install_commands(commands, staging, dependencies)
    finally:
        if seeded:
            requirements.unlink(missing_ok=True)


def _run_install_commands(
    commands: Sequence[Sequence[str]],
    staging: Path,
    dependencies: Callable[[Path], None] | None,
) -> None:
    for command in commands:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            env={
                **package_install_environment(),
                "TMPDIR": str(staging.parent),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        if result.returncode != 0:
            detail = result.stderr.strip()[-2000:] or result.stdout.strip()[-2000:]
            raise EnvironmentRefusal(
                "package_environment_uv_failed",
                detail or f"uv exited {result.returncode}",
            )

    from cozy_runtime.internal import _image_wheel_cache

    _image_wheel_cache.private_metadata(staging)
    _rewrite_installed_launchers(staging)
    # uv must finish before immutable shared dependency inodes enter the venv.
    # Installation cannot overwrite a prior capture through a shared payload.
    if dependencies is not None:
        dependencies(staging)


def _rewrite_installed_launchers(staging: Path) -> None:
    from cozy_runtime.internal import _installed_dependencies

    site = next(staging.glob("lib/python*/site-packages"))
    _installed_dependencies.rewrite_launchers(
        staging, site, sorted(site.glob("*.dist-info")), set(), replace=True
    )


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
        "format": "cozy.runtime.PackageEnvironment/2",
        "interpreter": dict(interpreter),
        "uv_version": uv_version,
        "wheel_digests": sorted(wheel.digest for wheel in wheels),
    }


def _content_matches_receipt(identity: object, receipt: Mapping[str, Any]) -> bool:
    if receipt.get("format") == LOCKED_RECEIPT_FORMAT:
        ref = receipt.get("locked_requirements")
        return (
            isinstance(identity, dict)
            and set(identity) - {"dependency_snapshot"}
            == {"format", "interpreter", "locked_requirements_digest", "uv_version"}
            and identity.get("format") == LOCKED_CONTENT_FORMAT
            and isinstance(identity.get("interpreter"), dict)
            and set(identity["interpreter"])
            == {"libc", "os_arch", "python_abi", "python_full_version"}
            and all(isinstance(v, str) and v for v in identity["interpreter"].values())
            and isinstance(identity.get("uv_version"), str)
            and bool(identity["uv_version"])
            and isinstance(ref, dict)
            and identity.get("locked_requirements_digest") == ref.get("digest")
        )
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
        and set(identity)
        in (
            {"format", "interpreter", "uv_version", "wheel_digests"},
            {"format", "interpreter", "uv_version", "wheel_digests", "dependency_snapshot"},
        )
        and identity.get("format") == "cozy.runtime.PackageEnvironment/2"
        and isinstance(identity.get("interpreter"), dict)
        and set(identity["interpreter"]) == {"libc", "os_arch", "python_abi", "python_full_version"}
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
            info = path.stat()
            mode = info.st_mode
            # uv owns shared payload inodes. Seal private directories/metadata only;
            # chmod on a hardlink would mutate every sibling and the uv cache.
            if stat.S_ISREG(mode) and info.st_nlink > 1:
                continue
            path.chmod(0o555 if stat.S_ISDIR(mode) or mode & 0o111 else 0o444)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        if not item.is_symlink() and item.is_dir():
            item.chmod(item.stat().st_mode | stat.S_IWUSR)
    path.chmod(path.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(path)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _materialize_dependency_snapshot(
    source: Input,
    root: Path,
    python: Path,
    dependency_python: Path,
    base: base_observation.Observation | None,
    *,
    builtin_capture: bool | None = None,
    dependency_cache: Path | None = None,
) -> InstalledEnvironment:
    from cozy_runtime.internal import _installed_dependencies

    observed = base or observe_base(python)
    _require_observed_interpreter(python, observed)
    wheels = _verify_inventory(source, observed, venv_precedence=True)
    inspected = _verify_grants(source, wheels, observed)
    supplements: tuple[str, ...] = ()
    if builtin_capture is not None:
        from cozy_runtime.internal import builtin_operations

        if len(inspected) != 1 or inspected[0].wheel.distribution != builtin_operations.NAME:
            raise EnvironmentRefusal(
                "builtin_carrier", "only the fixed Runtime carrier may capture an SDK"
            )
        builtin_operations.validate_carrier(inspected[0].path, inspected[0].wheel.version)
        if not builtin_capture and "numpy" not in observed.distribution_versions:
            supplements = ("numpy>=1.26",)
    provided = {
        item.wheel.distribution: (item.wheel.version, item.requirements) for item in inspected
    }
    # Creator's authored venv already supplies the selected installed closure.
    # Keep that closure for older/raw callers too: exact_dependencies controls
    # online boot resolution, not whether root extras were flattened. Only the
    # fixed carrier, which has no root extras, prunes an arbitrary SDK roster.
    included = None
    if builtin_capture is not None:
        included = _installed_dependencies.required_distributions(
            dependency_python,
            provided,
            _marker_environment(observed),
            allow_absent=frozenset({"numpy"}),
            current_system_sdk=True,
        )
    uv = _uv_binary()
    if _observed_uv_version(uv) != observed.uv_version:
        raise EnvironmentRefusal("package_environment_uv_changed", observed.uv_version)
    _require_owned_root(root)
    _remove_dead_staging(root)
    contents, receipts = root / "contents", root / "receipts"
    contents.mkdir(parents=True, exist_ok=True, mode=0o711)
    receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    contents.chmod(0o711)
    receipts.chmod(0o700)
    staging = root / f".environment-staging-{os.getpid()}-{uuid.uuid4().hex}"
    manifest = b""

    def copy_dependencies(target: Path) -> None:
        nonlocal manifest
        manifest = _installed_dependencies.snapshot(
            dependency_python,
            target,
            set(provided),
            included=included,
            cache=dependency_cache,
            current_system_sdk=builtin_capture is not None,
        )

    try:
        _install_with_uv(
            uv,
            python,
            staging,
            inspected,
            supplements,
            dependencies=copy_dependencies,
            cache_root=dependency_cache,
        )
        if supplements:
            # Only newly resolved optional numerical files join the existing immutable
            # snapshot. The final verification below covers both the original SDK bytes
            # and the supplement; the resolver cannot replace a captured dependency.
            additional = _installed_dependencies.snapshot(
                staging / "bin/python", staging, set(), included={"numpy"}, inventory_only=True
            )
            original_body = canonical.parse_canonical(manifest)
            extra_body = canonical.parse_canonical(additional)
            original_body["distributions"] += extra_body["distributions"]
            original_body["files"] += extra_body["files"]
            original_body["distributions"].sort(key=lambda row: row["distribution"])
            original_body["files"].sort(key=lambda row: row["path"])
            manifest = canonical.write(original_body)
        target_site = next(staging.glob("lib/python*/site-packages"))
        captured = [
            info
            for info in target_site.glob("*.dist-info")
            if _name(
                str(
                    BytesParser()
                    .parsebytes((info / "METADATA").read_bytes(), headersonly=True)
                    .get("Name", "")
                )
            )
            in {wheel.distribution for wheel in wheels}
        ]
        _installed_dependencies.rewrite_launchers(
            staging, target_site, captured, set(), replace=True
        )
        _installed_dependencies.verify_files(staging, manifest)
        # Re-read the copied metadata after wheel installation: a source edit during
        # capture cannot silently leave an incomplete transitive closure.
        if builtin_capture is not None:
            _installed_dependencies.required_distributions(
                staging / "bin/python",
                provided,
                _marker_environment(observed),
                allow_absent=frozenset({"numpy"}) if builtin_capture else frozenset(),
            )
        actual = observe_base(staging / "bin" / "python")
        if actual.interpreter_identity != observed.interpreter_identity:
            raise EnvironmentRefusal(
                "package_environment_dependency_snapshot_mismatch", "interpreter"
            )
        _verify_inventory(source, actual, venv_precedence=True)
        if not builtin_capture:
            from cozy_runtime.internal import package_requirements

            package_requirements.validate(
                staging
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages",
                python,
                actual,
            )
        snapshot = canonical.parse_canonical(manifest)
        assert isinstance(snapshot, dict)
        for row in snapshot["distributions"]:
            if actual.distribution_versions.get(row["distribution"]) != row["version"]:
                raise EnvironmentRefusal(
                    "package_environment_dependency_snapshot_mismatch", row["distribution"]
                )
        identity = _content_identity(wheels, actual.interpreter_identity, observed.uv_version)
        identity["dependency_snapshot"] = {"digest": _sha(manifest), "length": len(manifest)}
        content_bytes = canonical.write(identity)
        content_digest = _sha(content_bytes)
        receipt_bytes = canonical.write(_receipt(source, wheels, content_digest))
        receipt_digest = _sha(receipt_bytes)
        final = contents / content_digest.removeprefix("sha256:")
        _write_once(staging / _installed_dependencies.MANIFEST_NAME, manifest)
        _write_once(staging / CONTENT_IDENTITY_NAME, content_bytes)
        _make_read_only(staging)
        _fsync_dir(staging)
        reused = False
        try:
            os.rename(staging, final)
            final.chmod(0o555)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _installed_dependencies.read_manifest(final, identity["dependency_snapshot"])
            _remove_tree(staging)
            reused = True
        _fsync_dir(contents)
        _publish_receipt(
            receipts / f"{receipt_digest.removeprefix('sha256:')}.json", receipt_bytes, receipts
        )
        return open_generation(
            root,
            receipt_digest=receipt_digest,
            environment_digest=source.environment_digest,
            python=python,
            reused=reused,
        )
    except OSError as exc:
        raise EnvironmentRefusal("package_environment_dependency_snapshot_io", str(exc)) from exc
    finally:
        if staging.exists():
            _remove_tree(staging)
