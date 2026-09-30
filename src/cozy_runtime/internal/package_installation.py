"""Ordinary uv package installs with opaque, invocation-stable lifecycle generations."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tomllib
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NotRequired, TypedDict

import msgspec
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from cozy_runtime.internal import host_paths
from cozy_runtime.internal.config import package_install_environment
from cozy_runtime.internal.package_environment import EnvironmentRefusal

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
MAX_SOURCE_BYTES = 1 << 30
MAX_SOURCE_FILES = 100_000
IMAGE_UV_CACHE = Path("/opt/cozy/dependency-seed/uv-cache")
#: Installation ids of published environments keyed by their content (release, lock, ABI).
PUBLISHED_PREFIX = "release-"
#: An account-index URL: its origin, then the path that names it (account, then page or file).
_ACCOUNT_INDEX = re.compile(rb"https?://[^/\s'\"]+(/v1/index/[A-Za-z0-9._-]+/(?:simple|files)/)")
#: Cozy's SDK. A lock freezes a package's dependencies, not the Runtime its executor runs:
#: the package's declared bounds choose that, and everything else installs as locked.
RUNTIME = "cozy-runtime"
SDK = (RUNTIME, "tensorfs")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class InstallationRecord(TypedDict):
    package: str
    release: str
    python: NotRequired[str]
    generation: NotRequired[str]
    site_packages: NotRequired[str]
    #: The base interpreter, the machine SDK the SDK was chosen against, and the SDK installed.
    interpreter: NotRequired[str]
    machine: NotRequired[dict[str, str]]
    sdk: NotRequired[dict[str, str]]


def machine_sdk() -> dict[str, str]:
    """The SDK this machine's Runtime runs, which an installation's SDK is chosen against."""
    found = {}
    for name in SDK:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            found[name] = importlib.metadata.version(name)
    return found


def machine_wheels(install_root: Path) -> Path | None:
    """Where `cozy machine install` keeps the wheels this machine's Runtime and TensorFS were
    installed from (`<machine root>/opt/cozy/wheels`); None for an install root outside a
    machine layout."""
    layout = host_paths.Layout.of_install_root(install_root)
    return layout.machine_wheels if layout is not None else None


@dataclass(frozen=True, slots=True)
class HubIndex:
    """This machine's own Hub: where a synced project's account-index rows resolve."""

    origin: str
    ca: Path | None = None


@dataclass(frozen=True, slots=True)
class InstalledEnvironment:
    installation_id: str
    generation: Path
    site_packages: Path
    python: Path
    package: str
    release: str
    reused: bool = False
    # Local record incarnation, not package content or a peer compatibility fence.
    incarnation: tuple[int, int, int, int] | None = None


def _directory(root: Path, installation_id: str) -> Path:
    if not root.is_absolute() or not _ID.fullmatch(installation_id):
        raise EnvironmentRefusal("package_installation_invalid", "invalid installation location")
    return root / "installations" / installation_id


def _read_record(directory: Path) -> InstallationRecord:
    return msgspec.json.decode(
        (directory / "installation.json").read_bytes(), type=InstallationRecord
    )


def open_installation(root: Path, installation_id: str) -> InstalledEnvironment:
    directory = _directory(root, installation_id)
    try:
        with (directory / "installation.json").open("rb") as stream:
            record = msgspec.json.decode(stream.read(), type=InstallationRecord)
            info = os.fstat(stream.fileno())
        python = Path(record.get("python", directory / "venv" / "bin" / "python"))
        generation = Path(record.get("generation", python.parent.parent))
        sites = (
            [Path(record["site_packages"])]
            if "site_packages" in record
            else list((generation / "lib").glob("python*/site-packages"))
        )
        if len(sites) != 1 or not python.is_file():
            raise ValueError("venv is incomplete")
        return InstalledEnvironment(
            installation_id,
            generation,
            sites[0],
            python,
            record["package"],
            record["release"],
            True,
            (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)
            if generation.is_relative_to(directory) and _retained(directory)
            else None,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EnvironmentRefusal("package_installation_absent", installation_id) from exc


def _run(
    command: list[str], *, directory: Path, environment: dict[str, str], lock_fd: int = -1
) -> None:
    result = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        pass_fds=(lock_fd,) if sys.platform != "win32" and lock_fd >= 0 else (),
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
                name.is_absolute()
                or not name.parts
                or any(p in {".", ".."} for p in name.parts)
                or "\\" in member.name
                or member.name in seen
                or not (member.isfile() or member.isdir())
            ):
                raise EnvironmentRefusal("package_source_invalid", "unsafe source member")
            count += 1
            total += member.size
            if count > MAX_SOURCE_FILES or total > MAX_SOURCE_BYTES:
                raise EnvironmentRefusal(
                    "package_source_invalid", "source inventory exceeds bounds"
                )
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
        raise EnvironmentRefusal(
            "package_source_invalid", "source requires pyproject.toml and uv.lock"
        )


def _bind_account_index(source: Path, hub: HubIndex | None) -> None:
    """Point a synced project's account-index URLs at this machine's own Hub. A lock names
    that index by its path (account, page, file digest), never by the Hub its author used."""
    for name in ("pyproject.toml", "uv.lock"):
        path = source / name
        raw = path.read_bytes()
        if not _ACCOUNT_INDEX.search(raw):
            continue
        if hub is None:
            raise EnvironmentRefusal(
                "package_account_index_unreachable",
                "the project resolves account-index packages but this machine has no Hub grant",
            )
        path.write_bytes(_ACCOUNT_INDEX.sub(hub.origin.rstrip("/").encode() + rb"\1", raw))
        if hub.ca is not None:
            trusted = source.parent / "hub-ca"
            trusted.mkdir(exist_ok=True)
            shutil.copyfile(hub.ca, trusted / "tensorhub-ca.pem")


def _environment(directory: Path, cache: Path | None) -> dict[str, str]:
    environment = package_install_environment()
    if IMAGE_UV_CACHE.is_dir():
        # The image already contains uv's expanded wheels and index records.
        # Continue using that normal machine cache for matching and new packages.
        # Symlinks avoid overlayfs copying the baked Torch payload into each venv;
        # uv owns resolution, integrity and cache locking, with no Cozy inventory.
        environment["UV_CACHE_DIR"] = str(IMAGE_UV_CACHE)
        environment["UV_LINK_MODE"] = "symlink"
    elif cache is not None:
        environment["UV_CACHE_DIR"] = str(cache / "uv-cache")
    if (directory / "hub-ca").is_dir():
        # uv trusts one certificate set: the platform's, plus the grant's private Hub CA.
        environment["SSL_CERT_DIR"] = (
            f"{ssl.get_default_verify_paths().openssl_capath}:{directory / 'hub-ca'}"
        )
    return environment


def _site(generation: Path) -> Path:
    sites = list((generation / "lib").glob("python*/site-packages"))
    if len(sites) != 1:
        raise EnvironmentRefusal("package_installation_invalid", "venv is incomplete")
    return sites[0]


def _split_runtime(requirements: bytes) -> tuple[bytes, bytes]:
    """A lock without its Runtime rows, and those rows alone; both keep the index lines."""
    exact, runtime = [], []
    for line in requirements.decode().replace("\\\n", " ").splitlines():
        name = _NAME.match(line.strip())
        if name is not None and canonicalize_name(name.group()) == RUNTIME:
            runtime.append(line)
            continue
        exact.append(line)
        if line.strip().startswith("--"):
            runtime.append(line)
    rows = "\n".join(runtime) + "\n" if any(_NAME.match(row.strip()) for row in runtime) else ""
    return ("\n".join(exact) + "\n").encode(), rows.encode()


def _select_sdk(
    site: Path,
    declared: list[str],
    constraints: Path,
    install: Callable[[list[str]], None],
    as_locked: Callable[[], None],
    mine: dict[str, str],
    wheels: Path | None,
) -> None:
    """Install the SDK the requirements admit (the installed distributions' and `declared`):
    this machine's own Runtime and TensorFS (`mine`, found on the index or among `wheels`),
    else the newest published Runtime they admit (TensorFS as installed, where that Runtime
    accepts it), else the lock's own. Every other distribution stays as installed.

    A machine running a local build (a `+` version) never hands a package a different SDK:
    when its own wheels cannot be installed, the installation refuses, naming both."""
    bounds = {name: (SpecifierSet(), frozenset[str]()) for name in SDK}
    pinned: dict[str, str] = {}
    requirements = list(declared)
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = canonicalize_name(distribution.metadata["Name"] or "")
        if name not in bounds:
            pinned[name] = distribution.version
            requirements += distribution.requires or ()
    for raw in requirements:
        requirement = Requirement(raw)
        wanted = canonicalize_name(requirement.name)
        if wanted in bounds and (
            requirement.marker is None or requirement.marker.evaluate({"extra": ""})
        ):
            specifier, extras = bounds[wanted]
            bounds[wanted] = (specifier & requirement.specifier, extras | requirement.extras)
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in pinned.items()))
    named = {
        name: name + (f"[{','.join(sorted(extras))}]" if extras else "")
        for name, (_, extras) in bounds.items()
    }
    ranged = [named[name] + str(specifier) for name, (specifier, _) in bounds.items()]
    preferred = [
        f"{named[name]}=={mine[name]}"
        if name in mine and specifier.contains(mine[name], prereleases=True)
        else named[name] + str(specifier)
        for name, (specifier, _) in bounds.items()
    ]
    links = ["--find-links", str(wheels)] if wheels is not None and wheels.is_dir() else []
    local = {name: version for name, version in mine.items() if "+" in version}
    if local:
        # A local build is what the owner is testing: the package runs it or nothing.
        builds = ", ".join(f"{name} {version}" for name, version in sorted(local.items()))
        try:
            if any(not bounds[name][0].contains(v, prereleases=True) for name, v in local.items()):
                raise EnvironmentRefusal("package_runtime_incompatible", "outside the bounds")
            install([*links, *preferred])
        except EnvironmentRefusal as exc:
            raise EnvironmentRefusal(
                "package_runtime_unavailable",
                f"this machine runs {builds}, a local build; the package admits "
                f"{', '.join(ranged)} and {wheels} does not install it, so no other SDK is "
                f"put in its place ({exc.code})",
            ) from exc
        return
    for rows in dict.fromkeys((tuple(preferred), tuple(ranged))):
        with contextlib.suppress(EnvironmentRefusal):
            install([*links, *rows])
            return
    as_locked()


def _build(
    directory: Path,
    generation: Path,
    python: Path,
    *,
    package: str,
    release: str,
    environment: dict[str, str],
    lock_fd: int,
) -> dict[str, str]:
    """Install an installation's retained inputs into `generation`: the lock exactly, except
    the Runtime, which the package's bounds choose (`_select_sdk`). Returns the SDK installed."""
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentRefusal("package_installation_uv_absent", "uv is required")
    environment = {**environment, "UV_PROJECT_ENVIRONMENT": str(generation)}

    def run(*argv: str, cwd: Path = directory) -> None:
        _run([uv, *argv], directory=cwd, environment=environment, lock_fd=lock_fd)

    source = directory / "source"
    declared: list[str] = []
    pip = ("pip", "install", "--no-config", "--python", str(generation / "bin" / "python"))
    if source.is_dir():
        lock = tomllib.loads((source / "uv.lock").read_text())
        declared = (
            tomllib.loads((source / "pyproject.toml").read_text())
            .get("project", {})
            .get("dependencies", [])
        )
        floating = any(canonicalize_name(row["name"]) == RUNTIME for row in lock.get("package", ()))
        sync = ("sync", "--frozen", "--no-dev", "--no-python-downloads", "--project", str(source))
        run(
            *sync,
            "--python",
            str(python),
            *("--no-install-package", RUNTIME) * floating,
            cwd=source,
        )

        def as_locked() -> None:
            run(*sync, "--python", str(python), "--inexact", cwd=source)

    else:
        if not (generation / "bin" / "python").is_file():
            shutil.rmtree(generation, ignore_errors=True)
            run(
                *("venv", "--no-project", "--no-config", "--no-python-downloads"),
                *("--python", str(python), str(generation)),
            )
        requirements = directory / "requirements.txt"
        exact, runtime = _split_runtime(
            requirements.read_bytes() if requirements.is_file() else b""
        )
        floating = bool(runtime)

        def hashed(name: str, rows: bytes) -> None:
            (directory / name).write_bytes(rows)
            run(*pip, "--require-hashes", "--no-deps", "--requirements", str(directory / name))

        wheels = sorted(str(path) for path in (directory / "wheels").glob("*.whl"))
        if requirements.is_file():
            hashed(".exact-requirements.txt", exact)
        if wheels:
            run(*pip, "--no-deps", *wheels)
        elif not requirements.is_file():
            run(*pip, f"{package.rsplit('/', 1)[-1]}=={release}")

        def as_locked() -> None:
            hashed(".runtime-requirements.txt", runtime)

    site = _site(generation)
    if floating:
        constraints = directory / ".sdk-constraints.txt"
        _select_sdk(
            site,
            declared,
            constraints,
            lambda rows: run(*pip, "--constraints", str(constraints), *rows),
            as_locked,
            machine_sdk(),
            machine_wheels(directory.parent.parent),
        )
    return {
        name: distribution.version
        for distribution in importlib.metadata.distributions(path=[str(site)])
        if (name := canonicalize_name(distribution.metadata["Name"] or "")) in SDK
    }


def _write_record(directory: Path, generation: Path, record: InstallationRecord) -> None:
    """Publish the record naming `generation` as the installation's environment."""
    record["python"] = str(generation / "bin" / "python")
    record["generation"] = str(generation)
    record["site_packages"] = str(_site(generation))
    temporary = directory / ".installation.json"
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, directory / "installation.json")


def _install(
    root: Path,
    *,
    package: str,
    release: str,
    python: Path,
    installation_id: str = "",
    source_archive: Path | None = None,
    requirements: bytes = b"",
    wheels: tuple[Path, ...] = (),
    cache: Path | None = None,
    hub: HubIndex | None = None,
    lock_fd: int = -1,
) -> InstalledEnvironment:
    """Create a venv without package fingerprints; existing IDs are operation replays only.
    Its inputs stay in the installation, so a later machine Runtime rebuilds it (`refresh`)."""
    identifier = installation_id or "install-" + uuid.uuid4().hex
    directory = _directory(root, identifier)
    if (directory / "installation.json").is_file():
        installed = open_installation(root, identifier)
        if (installed.package, installed.release) != (package, release):
            raise EnvironmentRefusal(
                "package_installation_conflict",
                "installation ID belongs to another package release",
            )
        # Held from before a Runtime update: its SDK follows this machine's Runtime.
        return _refresh(root, identifier, cache, lock_fd)
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o711)
    directory.mkdir(exist_ok=True, mode=0o711)
    # A failure retains unpublished uv state for the next operation replay. The process
    # lock also rides the uv child, so a surviving installer cannot be raced.
    if source_archive is not None:
        source = directory / "source"
        if not source.exists():
            staged_source = directory / ".source-staging"
            if staged_source.exists():
                shutil.rmtree(staged_source)
            _extract(source_archive, staged_source)
            os.replace(staged_source, source)
        _bind_account_index(source, hub)
    else:
        if requirements:
            (directory / "requirements.txt").write_bytes(requirements)
        if wheels:
            staged = directory / ".wheels-staging"
            shutil.rmtree(staged, ignore_errors=True)
            shutil.rmtree(directory / "wheels", ignore_errors=True)
            staged.mkdir()
            for wheel in wheels:
                try:
                    os.link(wheel, staged / wheel.name)
                except OSError:
                    shutil.copyfile(wheel, staged / wheel.name)
            os.replace(staged, directory / "wheels")
    generation = directory / "venv"
    sdk = _build(
        directory,
        generation,
        python,
        package=package,
        release=release,
        environment=_environment(directory, cache),
        lock_fd=lock_fd,
    )
    _write_record(
        directory,
        generation,
        {
            "package": package,
            "release": release,
            "interpreter": str(python),
            "machine": machine_sdk(),
            "sdk": sdk,
        },
    )
    installed = open_installation(root, identifier)
    return InstalledEnvironment(
        installed.installation_id,
        installed.generation,
        installed.site_packages,
        installed.python,
        installed.package,
        installed.release,
        incarnation=installed.incarnation,
    )


def _retained(directory: Path) -> bool:
    """Whether an installation keeps the inputs it was built from: a synced source, wheels,
    or a published release's lock."""
    return any((directory / name).exists() for name in ("source", "wheels", "requirements.txt"))


def stale(root: Path, installation_id: str) -> bool:
    """Whether an installation's SDK was chosen against another machine Runtime than this
    one (a Runtime update) and `refresh` rebuilds it."""
    directory = _directory(root, installation_id)
    try:
        record = _read_record(directory)
    except (OSError, msgspec.DecodeError):
        return False
    return record.get("machine") != machine_sdk() and _retained(directory)


def refresh(root: Path, installation_id: str, *, cache: Path | None = None) -> InstalledEnvironment:
    """The installation, rebuilt from its retained inputs as a new generation when its SDK
    was chosen against another machine Runtime (a Runtime update): its ID names a package
    revision, not an executor. Executors of the old generation keep its files."""
    with _installation_lock(root, installation_id) as lock_fd:
        return _refresh(root, installation_id, cache, lock_fd)


def _refresh(
    root: Path, installation_id: str, cache: Path | None, lock_fd: int
) -> InstalledEnvironment:
    directory = _directory(root, installation_id)
    installed = open_installation(root, installation_id)
    record = _read_record(directory)
    machine = machine_sdk()
    if record.get("machine") == machine or not _retained(directory):
        return installed
    key = hashlib.sha256(json.dumps(machine, sort_keys=True).encode()).hexdigest()[:16]
    generation = directory / f"venv-{key}"
    shutil.rmtree(generation, ignore_errors=True)
    sdk = _build(
        directory,
        generation,
        Path(record.get("interpreter", installed.python)),
        package=installed.package,
        release=installed.release,
        environment=_environment(directory, cache),
        lock_fd=lock_fd,
    )
    record["machine"], record["sdk"] = machine, sdk
    _write_record(directory, generation, record)
    for old in directory.glob("venv*"):
        if old not in (generation, installed.generation):
            shutil.rmtree(old, ignore_errors=True)
    return open_installation(root, installation_id)


def open_existing_environment(
    python: Path,
    installation_id: str,
    *,
    package: str = "",
    release: str = "",
) -> InstalledEnvironment:
    """Adopt a normal same-user uv venv for its invocation lifetime."""
    from cozy_runtime.internal.package_environment import interpreter_layout

    if not _ID.fullmatch(installation_id) or not python.is_absolute():
        raise EnvironmentRefusal("package_installation_invalid", "invalid local installation")
    generation = python.parent.parent
    if not (generation / "pyvenv.cfg").is_file():
        raise EnvironmentRefusal(
            "package_installation_invalid", "a uv virtual environment is required"
        )
    layout = interpreter_layout(python)
    site = Path(layout["purelib"])
    if layout["prefix"] != str(generation) or not site.is_relative_to(generation):
        raise EnvironmentRefusal(
            "package_installation_invalid", "Python does not belong to its venv"
        )
    return InstalledEnvironment(installation_id, generation, site, python, package, release, True)


def retain_environment(
    root: Path,
    python: Path,
    *,
    package: str,
    release: str,
) -> InstalledEnvironment:
    """Retain the Runtime package's existing uv environment without copying its files."""
    identifier = "install-" + uuid.uuid4().hex
    from cozy_runtime.internal.package_environment import interpreter_layout

    layout = interpreter_layout(python)
    held = InstalledEnvironment(
        identifier, Path(layout["prefix"]), Path(layout["purelib"]), python, package, release, True
    )
    directory = _directory(root, identifier)
    directory.mkdir(parents=True, mode=0o711)
    (directory / "installation.json").write_text(
        json.dumps(
            {
                "package": package,
                "release": release,
                "python": str(python),
                "generation": str(held.generation),
                "site_packages": str(held.site_packages),
            }
        )
        + "\n"
    )
    return held


@contextmanager
def _installation_lock(root: Path, identifier: str) -> Generator[int, None, None]:
    directory = _directory(root, identifier)
    locks = directory.parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (locks / (identifier + ".lock")).open("a+b") as held:
        if sys.platform == "win32":
            import msvcrt

            held.write(b"0")
            held.flush()
            held.seek(0)
            msvcrt.locking(held.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield held.fileno()
            finally:
                held.seek(0)
                msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            try:
                yield held.fileno()
            finally:
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)


def remove(root: Path, installation_id: str) -> None:
    """Delete one installation directory under its installation lock."""
    directory = _directory(root, installation_id)
    with _installation_lock(root, installation_id):
        shutil.rmtree(directory, ignore_errors=True)
    (directory.parent / ".locks" / (installation_id + ".lock")).unlink(missing_ok=True)


def install(
    root: Path,
    *,
    package: str,
    release: str,
    python: Path,
    installation_id: str = "",
    source_archive: Path | None = None,
    requirements: bytes = b"",
    wheels: tuple[Path, ...] = (),
    cache: Path | None = None,
    hub: HubIndex | None = None,
) -> InstalledEnvironment:
    identifier = installation_id or "install-" + uuid.uuid4().hex
    with _installation_lock(root, identifier) as lock_fd:
        return _install(
            root,
            package=package,
            release=release,
            python=python,
            installation_id=identifier,
            source_archive=source_archive,
            requirements=requirements,
            wheels=wheels,
            cache=cache,
            hub=hub,
            lock_fd=lock_fd,
        )
