"""Copy the regular RECORD closure of a local venv without starting its interpreter."""

from __future__ import annotations

import base64
import configparser
import csv
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Mapping, Sequence
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import canonical
from cozy_runtime.internal.package_environment import (
    _DIGEST,
    MAX_METADATA_BYTES,
    MAX_PACKAGE_WHEELS,
    MAX_WHEEL_MEMBERS,
    EnvironmentRefusal,
    _hash_file,
    _name,
    _sha,
    normalized_distribution,
    valid_wheel_version,
)

MANIFEST_NAME = "installed-dependencies.json"
MAX_DEPENDENCY_MANIFEST_BYTES = 64 << 20


def required_distributions(
    source_python: Path,
    provided: Mapping[str, tuple[str, Sequence[Requirement]]],
    environment: dict[str, str],
    *,
    allow_absent: frozenset[str] = frozenset(),
) -> set[str]:
    """Read the declared dependency graph without importing a mutable source venv.

    Exact captured wheels supply their own metadata. The installed SDK may contain
    unrelated packages, including heavyweight optional accelerator dependencies.
    Extras propagate along dependency edges and cycles terminate by visited extras.
    """
    sites = list(source_python.parent.parent.glob("lib/python*/site-packages"))
    if len(sites) != 1 or sites[0].resolve() != sites[0]:
        raise EnvironmentRefusal(
            "package_environment_dependency_source_invalid", str(source_python)
        )
    installed: dict[str, tuple[str, Sequence[Requirement]]] = {}
    requires_python: dict[str, str] = {}
    for info in sorted(sites[0].glob("*.dist-info")):
        metadata = info / "METADATA"
        if (
            info.is_symlink()
            or not metadata.is_file()
            or metadata.is_symlink()
            or metadata.stat().st_size > MAX_METADATA_BYTES
        ):
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
        headers = BytesParser().parsebytes(metadata.read_bytes(), headersonly=True)
        name, version = _name(str(headers.get("Name", ""))), str(headers.get("Version", ""))
        if name in provided:
            continue
        if not normalized_distribution(name) or not valid_wheel_version(version):
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
        if name in installed:
            raise EnvironmentRefusal("package_environment_dependency_distribution_collision", name)
        try:
            installed[name] = (
                version,
                tuple(Requirement(row) for row in headers.get_all("Requires-Dist", [])),
            )
        except InvalidRequirement as exc:
            raise EnvironmentRefusal(
                "package_environment_dependency_record_invalid", str(info)
            ) from exc
        requires_python[name] = str(headers.get("Requires-Python", ""))
    available = {**installed, **provided}
    pending = [Requirement(name) for name in {*provided, "cozy-runtime"}]
    visited: dict[str, set[str]] = {}
    while pending:
        requirement = pending.pop()
        name = _name(requirement.name)
        if name not in available:
            if name in allow_absent:
                continue
            raise EnvironmentRefusal("package_environment_dependency_missing", str(requirement))
        version, dependencies = available[name]
        try:
            if requirement.specifier and not requirement.specifier.contains(
                Version(version), prereleases=True
            ):
                raise EnvironmentRefusal(
                    "package_environment_dependency_base_conflict",
                    f"{requirement}: installed {version}",
                )
            python = requires_python.get(name, "")
            if python and not SpecifierSet(python).contains(
                environment["python_full_version"], prereleases=True
            ):
                raise EnvironmentRefusal(
                    "package_environment_python_incompatible", f"{name}: {python}"
                )
        except (InvalidVersion, InvalidSpecifier) as exc:
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", name) from exc
        extras = {"", *requirement.extras} - visited.get(name, set())
        if not extras:
            continue
        visited.setdefault(name, set()).update(extras)
        for dependency in dependencies:
            if dependency.marker is None or any(
                dependency.marker.evaluate({**environment, "extra": extra}) for extra in extras
            ):
                pending.append(dependency)
    return set(visited) - provided.keys()


def _dependency_record(info: Path) -> list[tuple[str, str, str]]:
    record = info / "RECORD"
    if not record.is_file() or record.is_symlink() or record.stat().st_size > MAX_METADATA_BYTES:
        raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
    try:
        rows = list(csv.reader(record.read_text().splitlines()))
    except (UnicodeError, csv.Error) as exc:
        raise EnvironmentRefusal(
            "package_environment_dependency_record_invalid", str(info)
        ) from exc
    if len(rows) > MAX_WHEEL_MEMBERS or any(len(row) != 3 for row in rows):
        raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
    return [(row[0], row[1], row[2]) for row in rows]


def _dependency_entrypoints(info: Path) -> dict[str, str]:
    path = info / "entry_points.txt"
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise EnvironmentRefusal("package_environment_dependency_entrypoint_invalid", str(path))

    class EntryPoints(configparser.ConfigParser):
        def optionxform(self, optionstr: str) -> str:
            return optionstr

    parser = EntryPoints(interpolation=None)
    try:
        parser.read_string(path.read_text())
    except (UnicodeError, configparser.Error) as exc:
        raise EnvironmentRefusal(
            "package_environment_dependency_entrypoint_invalid", str(path)
        ) from exc
    output: dict[str, str] = {}
    for group in ("console_scripts", "gui_scripts"):
        if group not in parser:
            continue
        for name, target in parser[group].items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*"
                r"(?:\.[A-Za-z_]\w*)*(?:\s*\[[A-Za-z0-9_, .-]+\])?",
                target,
            ):
                raise EnvironmentRefusal(
                    "package_environment_dependency_entrypoint_invalid", str(path)
                )
            if name in output and output[name] != target:
                raise EnvironmentRefusal("package_environment_dependency_entrypoint_invalid", name)
            output[name] = target.split("[", 1)[0].strip()
    return output


def _copy_dependency_file(source: Path, target: Path, expected: str, size: str) -> None:
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise EnvironmentRefusal("package_environment_dependency_member_unsupported", str(source))
    if size and (not size.isdecimal() or int(size) != info.st_size):
        raise EnvironmentRefusal("package_environment_dependency_member_changed", str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise EnvironmentRefusal("package_environment_dependency_member_collision", str(target))
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        before = os.fstat(incoming.fileno())
        try:
            fcntl.ioctl(outgoing.fileno(), 0x40049409, incoming.fileno())  # Linux FICLONE.
        except OSError as exc:
            if exc.errno not in {
                errno.EXDEV,
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.ENOSYS,
            }:
                raise
            shutil.copyfileobj(incoming, outgoing, 1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
        after = os.fstat(incoming.fileno())
    # uv may link/unlink other cache names for this inode while we copy it. That changes
    # ctime and nlink without changing content; the copied bytes still prove their RECORD hash.
    for observed in (info, before):
        if (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_mode,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_mode,
        ):
            raise EnvironmentRefusal("package_environment_dependency_member_changed", str(source))
    _verify_dependency_hash(target, expected)
    target.chmod(0o755 if info.st_mode & 0o111 else 0o644)


def _verify_dependency_hash(target: Path, expected: str) -> None:
    if expected:
        algorithm, separator, encoded = expected.partition("=")
        if not separator or algorithm not in {"sha256", "sha384", "sha512"}:
            raise EnvironmentRefusal("package_environment_dependency_hash_unsupported", str(target))
        digest = hashlib.new(algorithm)
        with target.open("rb") as copied:
            while chunk := copied.read(1 << 20):
                digest.update(chunk)
        if base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode() != encoded:
            raise EnvironmentRefusal("package_environment_dependency_member_changed", str(target))


def snapshot(
    source_python: Path,
    staging: Path,
    excluded: set[str],
    *,
    included: set[str] | None = None,
    inventory_only: bool = False,
) -> bytes:
    # Metadata discovery must never start this interpreter: its editable hooks are mutable.
    prefix = source_python.parent.parent
    if inventory_only and prefix != staging:
        raise EnvironmentRefusal(
            "package_environment_dependency_source_invalid", str(source_python)
        )
    sites = list(prefix.glob("lib/python*/site-packages"))
    if (
        not source_python.is_absolute()
        or source_python.parent.name != "bin"
        or not source_python.is_file()
        or not (prefix / "pyvenv.cfg").is_file()
        or prefix.resolve() != prefix
        or len(sites) != 1
        or sites[0].resolve() != sites[0]
        or not (staging / sites[0].relative_to(prefix)).is_dir()
        or list(sites[0].glob("*.egg-link"))
        or list(sites[0].glob("*.egg-info"))
    ):
        raise EnvironmentRefusal(
            "package_environment_dependency_source_invalid", str(source_python)
        )
    source_site = sites[0]
    target_site = staging / source_site.relative_to(prefix)
    distributions: list[dict[str, Any]] = []
    copied: set[Path] = set()
    names: set[str] = set()
    selected_infos: list[Path] = []
    for info in sorted(source_site.glob("*.dist-info")):
        if info.is_symlink() or not info.is_dir():
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
        metadata = info / "METADATA"
        if (
            not metadata.is_file()
            or metadata.is_symlink()
            or metadata.stat().st_size > MAX_METADATA_BYTES
        ):
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
        headers = BytesParser().parsebytes(metadata.read_bytes(), headersonly=True)
        name, version = _name(str(headers.get("Name", ""))), str(headers.get("Version", ""))
        if not normalized_distribution(name) or not valid_wheel_version(version):
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
        if name in excluded or (included is not None and name not in included):
            continue
        if name in names:
            raise EnvironmentRefusal("package_environment_dependency_distribution_collision", name)
        names.add(name)
        selected_infos.append(target_site / info.name)
        direct = info / "direct_url.json"
        if direct.is_file():
            if direct.is_symlink() or direct.stat().st_size > MAX_METADATA_BYTES:
                raise EnvironmentRefusal(
                    "package_environment_dependency_record_invalid", str(direct)
                )
            try:
                value = json.loads(direct.read_text())
            except (ValueError, UnicodeError) as exc:
                raise EnvironmentRefusal(
                    "package_environment_dependency_record_invalid", str(direct)
                ) from exc
            if not isinstance(value, dict) or not isinstance(value.get("dir_info", {}), dict):
                raise EnvironmentRefusal(
                    "package_environment_dependency_record_invalid", str(direct)
                )
            if value.get("dir_info", {}).get("editable"):
                raise EnvironmentRefusal("package_environment_dependency_editable", name)
        launchers = _dependency_entrypoints(info)
        rows = _dependency_record(info)
        for relative, digest, length in rows:
            member = source_site / relative
            if not relative or "\x00" in relative or PurePosixPath(relative).is_absolute():
                raise EnvironmentRefusal(
                    "package_environment_dependency_member_unsupported", relative
                )
            resolved = member.resolve()
            lexical = Path(os.path.abspath(member))
            if not resolved.is_relative_to(prefix) or resolved != lexical or member.is_symlink():
                raise EnvironmentRefusal(
                    "package_environment_dependency_member_unsupported", relative
                )
            if "__editable__" in member.name:
                raise EnvironmentRefusal("package_environment_dependency_editable", name)
            if member.suffix in {".pth", ".pyc"}:
                continue
            if not member.is_file():
                raise EnvironmentRefusal("package_environment_dependency_member_missing", relative)
            destination = staging / resolved.relative_to(prefix)
            if resolved.parent == prefix / "bin":
                if resolved.name in launchers:
                    continue
                with member.open("rb") as stream:
                    if stream.read(4) != b"\x7fELF":
                        raise EnvironmentRefusal(
                            "package_environment_dependency_launcher_unsupported", relative
                        )
            if inventory_only:
                if length and (not length.isdecimal() or int(length) != member.stat().st_size):
                    raise EnvironmentRefusal(
                        "package_environment_dependency_member_changed", str(member)
                    )
                _verify_dependency_hash(member, digest)
            else:
                _copy_dependency_file(member, destination, digest, length)
            copied.add(destination)
        if metadata not in {source_site / relative for relative, _, _ in rows}:
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(metadata))
        if info / "RECORD" not in {source_site / relative for relative, _, _ in rows}:
            raise EnvironmentRefusal("package_environment_dependency_record_invalid", str(info))
        distributions.append({"distribution": name, "version": version})
        if len(distributions) > MAX_PACKAGE_WHEELS or len(copied) > MAX_WHEEL_MEMBERS:
            raise EnvironmentRefusal(
                "package_environment_dependency_snapshot_oversized", str(source_site)
            )
    rewrite_launchers(staging, target_site, selected_infos, copied, replace=inventory_only)
    manifest = canonical.write(
        {
            "format": "cozy.runtime.InstalledDependencySnapshot/1",
            "distributions": distributions,
            "files": [
                {
                    "path": str(path.relative_to(staging)),
                    "digest": _hash_file(path),
                    "length": path.stat().st_size,
                }
                for path in sorted(copied)
            ],
        }
    )
    if len(manifest) > MAX_DEPENDENCY_MANIFEST_BYTES:
        raise EnvironmentRefusal(
            "package_environment_dependency_snapshot_oversized", str(source_site)
        )
    return manifest


def rewrite_launchers(
    staging: Path,
    target_site: Path,
    infos: Sequence[Path],
    copied: set[Path],
    *,
    replace: bool = False,
) -> None:
    # Rewrite installed launchers without the old environment shebangs. The shell/Python
    # wrapper is relative to its own bin directory, so atomic rename cannot break it.
    for info in infos:
        rows = _dependency_record(info)
        entries = _dependency_entrypoints(info)
        for name, target in entries.items():
            module, attribute = target.split(":", 1)
            body = (
                "#!/bin/sh\n'''exec' \"${0%/*}/python\" \"$0\" \"$@\"\n' '''\n"
                "import sys\nfrom importlib import import_module\n"
                f"entry = import_module({module!r})\n"
                f"for name in {attribute.split('.')!r}:\n    entry = getattr(entry, name)\n"
                "sys.exit(entry())\n"
            )
            path = staging / "bin" / name
            if path.exists() and not replace:
                raise EnvironmentRefusal(
                    "package_environment_dependency_member_collision", str(path)
                )
            path.write_text(body)
            path.chmod(0o755)
            copied.add(path)
        existing = {relative for relative, _, _ in rows}
        for name in entries:
            existing.add(os.path.relpath(staging / "bin" / name, target_site))
        record = info / "RECORD"
        with record.open("w", newline="") as output:
            writer = csv.writer(output)
            for relative in sorted(existing):
                path = (target_site / relative).resolve()
                if not path.is_relative_to(staging) or (not replace and path not in copied):
                    continue
                if path == record:
                    writer.writerow((relative, "", ""))
                else:
                    encoded = (
                        base64.urlsafe_b64encode(
                            bytes.fromhex(_hash_file(path).removeprefix("sha256:"))
                        )
                        .rstrip(b"=")
                        .decode()
                    )
                    writer.writerow((relative, "sha256=" + encoded, path.stat().st_size))


def read_manifest(generation: Path, ref: object) -> dict[str, Any]:
    if (
        not isinstance(ref, dict)
        or set(ref) != {"digest", "length"}
        or not isinstance(ref.get("digest"), str)
        or not _DIGEST.fullmatch(ref["digest"])
        or type(ref.get("length")) is not int
        or not 0 < ref["length"] <= MAX_DEPENDENCY_MANIFEST_BYTES
    ):
        raise EnvironmentRefusal("package_environment_dependency_snapshot_invalid", str(generation))
    path = generation / MANIFEST_NAME
    try:
        if path.is_symlink() or path.stat().st_size != ref["length"]:
            raise EnvironmentRefusal("package_environment_dependency_snapshot_invalid", str(path))
        raw = path.read_bytes()
        if _sha(raw) != ref["digest"]:
            raise EnvironmentRefusal("package_environment_dependency_snapshot_invalid", str(path))
        value = canonical.parse_canonical(raw)
    except (OSError, canonical.CanonicalError) as exc:
        raise EnvironmentRefusal(
            "package_environment_dependency_snapshot_invalid", str(path)
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("format") != "cozy.runtime.InstalledDependencySnapshot/1"
    ):
        raise EnvironmentRefusal("package_environment_dependency_snapshot_invalid", str(path))
    return value


def verify_files(staging: Path, manifest: bytes) -> None:
    value = canonical.parse_canonical(manifest)
    assert isinstance(value, dict)
    for row in value["files"]:
        path = staging / row["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["length"]
            or _hash_file(path) != row["digest"]
        ):
            raise EnvironmentRefusal("package_environment_dependency_member_collision", str(path))
