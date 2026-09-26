"""Ordinary uv package installs with opaque, invocation-stable lifecycle generations."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cozy_runtime.internal.config import package_install_environment
from cozy_runtime.internal.package_environment import EnvironmentRefusal

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
MAX_SOURCE_BYTES = 1 << 30
MAX_SOURCE_FILES = 100_000
IMAGE_UV_CACHE = Path("/opt/cozy/dependency-seed/uv-cache")


@dataclass(frozen=True, slots=True)
class InstalledEnvironment:
    installation_id: str
    generation: Path
    site_packages: Path
    python: Path
    package: str
    release: str
    reused: bool = False


def _directory(root: Path, installation_id: str) -> Path:
    if not root.is_absolute() or not _ID.fullmatch(installation_id):
        raise EnvironmentRefusal("package_installation_invalid", "invalid installation location")
    return root / "installations" / installation_id


def open_installation(root: Path, installation_id: str) -> InstalledEnvironment:
    directory = _directory(root, installation_id)
    try:
        record = json.loads((directory / "installation.json").read_bytes())
        python = Path(record.get("python", directory / "venv" / "bin" / "python"))
        generation = Path(record.get("generation", python.parent.parent))
        sites = ([Path(record["site_packages"])] if "site_packages" in record
                 else list((generation / "lib").glob("python*/site-packages")))
        if len(sites) != 1 or not python.is_file():
            raise ValueError("venv is incomplete")
        return InstalledEnvironment(
            installation_id, generation, sites[0], python,
            record["package"], record["release"], True,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EnvironmentRefusal("package_installation_absent", installation_id) from exc


def _run(command: list[str], *, directory: Path, environment: dict[str, str], lock_fd: int = -1) -> None:
    result = subprocess.run(
        command, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False,
        **({"pass_fds": (lock_fd,)} if os.name != "nt" and lock_fd >= 0 else {}),
    )
    if result.returncode:
        raise EnvironmentRefusal(
            "package_installation_uv_failed", (result.stderr or result.stdout)[-2000:]
        )


def _extract(archive: Path, destination: Path) -> None:
    """Extract regular source members only, within explicit path and expansion bounds."""
    if not archive.is_file() or archive.stat().st_size > MAX_SOURCE_BYTES:
        raise EnvironmentRefusal("package_source_invalid", "source archive exceeds its bound")
    destination.mkdir(mode=0o755)
    total = 0
    count = 0
    seen: set[str] = set()
    with tarfile.open(archive, mode="r:") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute() or not name.parts or any(p in {".", ".."} for p in name.parts)
                or "\\" in member.name or member.name in seen
                or not (member.isfile() or member.isdir())
            ):
                raise EnvironmentRefusal("package_source_invalid", "unsafe source member")
            count += 1
            total += member.size
            if count > MAX_SOURCE_FILES or total > MAX_SOURCE_BYTES:
                raise EnvironmentRefusal("package_source_invalid", "source inventory exceeds bounds")
            seen.add(member.name)
            target = destination.joinpath(*name.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming = source.extractfile(member)
            if incoming is None:
                raise EnvironmentRefusal("package_source_invalid", "missing source bytes")
            with incoming, target.open("xb") as output:
                shutil.copyfileobj(incoming, output)
            target.chmod(0o644 | (member.mode & 0o111))
    if not (destination / "pyproject.toml").is_file() or not (destination / "uv.lock").is_file():
        raise EnvironmentRefusal("package_source_invalid", "source requires pyproject.toml and uv.lock")


def _install(
    root: Path, *, package: str, release: str, python: Path,
    installation_id: str = "", source_archive: Path | None = None,
    requirements: bytes = b"", wheels: tuple[Path, ...] = (),
    cache: Path | None = None, lock_fd: int = -1,
) -> InstalledEnvironment:
    """Create a venv without package fingerprints; existing IDs are operation replays only."""
    identifier = installation_id or "install-" + uuid.uuid4().hex
    directory = _directory(root, identifier)
    if (directory / "installation.json").is_file():
        installed = open_installation(root, identifier)
        if (installed.package, installed.release) != (package, release):
            raise EnvironmentRefusal(
                "package_installation_conflict", "installation ID belongs to another package release"
            )
        return installed
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o711)
    directory.mkdir(exist_ok=True, mode=0o711)
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentRefusal("package_installation_uv_absent", "uv is required")
    environment = package_install_environment()
    environment["UV_PROJECT_ENVIRONMENT"] = str(directory / "venv")
    if IMAGE_UV_CACHE.is_dir():
        # The image already contains uv's expanded wheels and index records.
        # Continue using that normal machine cache for matching and new packages.
        # Symlinks avoid overlayfs copying the baked Torch payload into each venv;
        # uv owns resolution, integrity and cache locking, with no Cozy inventory.
        environment["UV_CACHE_DIR"] = str(IMAGE_UV_CACHE)
        environment["UV_LINK_MODE"] = "symlink"
    elif cache is not None:
        environment["UV_CACHE_DIR"] = str(cache / "uv-cache")
    try:
        if source_archive is not None:
            source = directory / "source"
            if not source.exists():
                staged_source = directory / ".source-staging"
                if staged_source.exists():
                    shutil.rmtree(staged_source)
                _extract(source_archive, staged_source)
                os.replace(staged_source, source)
            _run(
                [uv, "sync", "--frozen", "--no-dev", "--no-python-downloads",
                 "--project", str(source), "--python", str(python)],
                directory=source, environment=environment, lock_fd=lock_fd,
            )
        else:
            if not (directory / "venv" / "bin" / "python").is_file():
                if (directory / "venv").exists():
                    shutil.rmtree(directory / "venv")
                _run(
                    [uv, "venv", "--no-project", "--no-config", "--no-python-downloads",
                     "--python", str(python), str(directory / "venv")],
                    directory=directory, environment=environment, lock_fd=lock_fd,
                )
            command = [uv, "pip", "install", "--no-config", "--python",
                       str(directory / "venv" / "bin" / "python")]
            if requirements:
                locked = directory / "requirements.txt"
                locked.write_bytes(requirements)
                _run(command + ["--require-hashes", "--no-deps", "--requirements", str(locked)],
                     directory=directory, environment=environment, lock_fd=lock_fd)
            if wheels:
                _run(command + ["--no-deps", *[str(wheel) for wheel in wheels]],
                     directory=directory, environment=environment, lock_fd=lock_fd)
            elif not requirements:
                distribution = package.rsplit("/", 1)[-1]
                _run(command + [f"{distribution}=={release}"],
                     directory=directory, environment=environment, lock_fd=lock_fd)
        record = directory / "installation.json"
        temporary = directory / ".installation.json"
        temporary.write_text(json.dumps({"package": package, "release": release}) + "\n")
        os.replace(temporary, record)
    except Exception:
        # Retain unpublished uv state for the next operation replay. The process
        # lock also rides the uv child, so a surviving installer cannot be raced.
        raise
    installed = open_installation(root, identifier)
    return InstalledEnvironment(
        installed.installation_id, installed.generation, installed.site_packages,
        installed.python, installed.package, installed.release,
    )


def open_existing_environment(
    python: Path, installation_id: str, *, package: str = "", release: str = "",
) -> InstalledEnvironment:
    """Adopt a normal same-user uv venv for its invocation lifetime."""
    from cozy_runtime.internal.package_environment import _interpreter_layout

    if not _ID.fullmatch(installation_id) or not python.is_absolute():
        raise EnvironmentRefusal("package_installation_invalid", "invalid local installation")
    generation = python.parent.parent
    if not (generation / "pyvenv.cfg").is_file():
        raise EnvironmentRefusal("package_installation_invalid", "a uv virtual environment is required")
    layout = _interpreter_layout(python)
    site = Path(layout["purelib"])
    if layout["prefix"] != str(generation) or not site.is_relative_to(generation):
        raise EnvironmentRefusal("package_installation_invalid", "Python does not belong to its venv")
    return InstalledEnvironment(installation_id, generation, site, python, package, release, True)


def retain_environment(
    root: Path, python: Path, *, package: str, release: str,
) -> InstalledEnvironment:
    """Retain the Runtime package's existing uv environment without copying its files."""
    identifier = "install-" + uuid.uuid4().hex
    from cozy_runtime.internal.package_environment import _interpreter_layout
    layout = _interpreter_layout(python)
    held = InstalledEnvironment(identifier, Path(layout["prefix"]), Path(layout["purelib"]),
                                python, package, release, True)
    directory = _directory(root, identifier)
    directory.mkdir(parents=True, mode=0o711)
    (directory / "installation.json").write_text(json.dumps({
        "package": package, "release": release, "python": str(python),
        "generation": str(held.generation), "site_packages": str(held.site_packages),
    }) + "\n")
    return held


@contextmanager
def _installation_lock(root: Path, identifier: str):
    directory = _directory(root, identifier)
    locks = directory.parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (locks / (identifier + ".lock")).open("a+b") as held:
        if os.name == "nt":
            import msvcrt
            held.write(b"0")
            held.flush()
            held.seek(0)
            msvcrt.locking(held.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        try:
            yield held.fileno()
        finally:
            if os.name == "nt":
                held.seek(0)
                msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)


def install(
    root: Path, *, package: str, release: str, python: Path,
    installation_id: str = "", source_archive: Path | None = None,
    requirements: bytes = b"", wheels: tuple[Path, ...] = (), cache: Path | None = None,
) -> InstalledEnvironment:
    identifier = installation_id or "install-" + uuid.uuid4().hex
    with _installation_lock(root, identifier) as lock_fd:
        return _install(
            root, package=package, release=release, python=python,
            installation_id=identifier, source_archive=source_archive,
            requirements=requirements, wheels=wheels, cache=cache, lock_fd=lock_fd,
        )
