"""Ordinary wheel/requirement parsing and interpreter facts for uv installation.

No composite package identities, copied SDK snapshots, or installation receipts live here.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion

from cozy_runtime.internal import base_observation
from cozy_runtime.protocol import weights_limits

MAX_METADATA_BYTES = 4 << 20


MAX_PACKAGE_WHEELS = weights_limits.MAX_LOCAL_PACKAGE_FILES


_NORMALIZE = re.compile(r"[-_.]+")


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

    indexes: tuple[str, ...]
    rows: tuple[LockedRow, ...]


_INDEX_DIRECTIVES = ("--index-url", "--extra-index-url")


def read_locked_requirements(raw: bytes) -> LockedRequirements:
    """Admit one locked-requirements export.

    The only admitted rows are blank/comment lines, https index directives, and exact
    `name==version` requirements carrying at least one `--hash=sha256:`. Anything a resolver
    would have freedom over — a range, a bare name, a path or URL row, an editable — refuses.
    Row order is the exporter's choice; a name may repeat under different markers.
    """

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
        hashes = tuple(option.removeprefix("--hash=") for option in options)
        locked = LockedRow(
            text=row, name=name, version=version, url=requirement.url or "", hashes=hashes
        )
        if locked not in rows:
            rows.append(locked)
    if not rows:
        raise EnvironmentRefusal("package_environment_lock_invalid", "the export pins nothing")
    return LockedRequirements(indexes=tuple(indexes), rows=tuple(rows))


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


def interpreter_layout(python: Path) -> dict[str, str]:
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


def _target_python(base: base_observation.Observation) -> tuple[int, int]:
    match = re.fullmatch(r"cp([0-9])([0-9]{2})", base.python_abi)
    if match is None:
        raise EnvironmentRefusal(
            "package_environment_base_invalid", "Python ABI is not the cpNNN spelling"
        )
    return int(match.group(1)), int(match.group(2))


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


def _name(value: str) -> str:
    return _NORMALIZE.sub("-", value).lower()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
