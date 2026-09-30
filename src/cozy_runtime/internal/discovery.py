"""The isolation boundary: import exactly ONE configured export, freeze it, and stop.

That is the COMPLETE discovery algorithm (§1.0). Importing a module never registers
anything — registration is a method call on the app object — so there is nothing to
harvest, no package to walk, no dependency to stub and no source to parse. Everything a
package interface says comes from the object package.toml named.

Execution containment is a TIER-INDEPENDENT property of the caller, not of this module:
the trusted build runs `describe` in a disposable container of the already-built release
image with no GPU, no weights, no network and no credentials (tensorhub-build.md).
Discovery offers an explicit builder wall-clock ceiling. Runtime-supervised discovery
disables that ceiling and uses the executor progress/death watcher instead.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import signal
import sys
import threading
import tomllib
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, ModuleType

from cozy_runtime.author import App, ConformanceError, Surface, describe
from cozy_runtime.internal.static_interface import read_application

#: Author top-level code runs inside the boundary. It may import heavy dependencies; it may
#: not EXECUTE them at module scope (§1.0), so a generous ceiling still catches a module
#: that loads weights or talks to a network on import.
IMPORT_SECONDS = 60.0


class DiscoveryError(ConformanceError):
    """A BUILD refusal at the boundary: the package cannot be discovered as configured."""

    default_code = "discovery"


@dataclass(frozen=True, slots=True)
class Discovered:
    """The frozen app plus the two facts identity is made of (§1.0)."""

    app: App
    application: str
    """The configured export path — half of application identity, the release digest being
    the other half. No second author-chosen name exists to drift."""
    project: Path
    module: ModuleType
    surfaces: tuple[Surface, ...]
    bindings: Mapping[str, Mapping[str, object]]


def discover(project: Path, *, seconds: float | None = IMPORT_SECONDS) -> Discovered:
    """Import the one configured export inside this environment and freeze it."""
    project = project.resolve()
    application = read_application(project)
    return _discover(application, project, _bindings(project), seconds=seconds, add=project)


def assert_locked_environment(project: Path) -> str:
    """Refuse unless THIS interpreter is the package's own locked environment (#713).

    Importing package code is running it, and package code runs only where it is already
    allowed to run: the venv its lock describes. The proof is the package's own
    distribution, at pyproject's version, resolving from the running interpreter — a
    publisher's host or a hub never has that.
    """
    path = project / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DiscoveryError(f"{path}: {exc}", code="pyproject") from exc
    meta = data.get("project", {})
    name, version = meta.get("name"), meta.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise DiscoveryError(f"{path} names no [project] name and version", code="pyproject")
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DiscoveryError(
            f"{name} is not installed in {sys.executable}: the conformance oracle imports "
            "package code, which runs only inside the package's own locked environment — "
            "run it from that venv (`uv run --locked cozy-runtime describe --conformance`)",
            code="conformance_environment",
        ) from exc
    if installed != version:
        raise DiscoveryError(
            f"{sys.executable} has {name} {installed}, but {path} declares {version}: not "
            "this tree's locked environment",
            code="conformance_environment",
        )
    return f"{name} {version} in {sys.executable}"


def discover_installed(application: str, *, seconds: float | None = IMPORT_SECONDS) -> Discovered:
    """Freeze one package interface-named export from the installed package environment.

    A materialized rental has an immutable venv, not a copied source tree or package.toml.
    The exact PackageInterface supplies the application identity; importing anything else
    or adding a host path would recreate the source/PYTHONPATH authority the fixed worker rejects.
    """

    return _discover(application, Path("/"), {}, seconds=seconds, add=None)


def discover_distribution(
    distribution: str, *, seconds: float | None = IMPORT_SECONDS
) -> Discovered:
    """Discover one installed wheel through its single standard ``cozy.application`` entry."""

    try:
        installed = importlib.metadata.distribution(distribution)
        entries = tuple(installed.entry_points)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DiscoveryError(
            f"installed distribution {distribution!r} is absent", code="missing_distribution"
        ) from exc
    applications = [entry.value for entry in entries if entry.group == "cozy.application"]
    if len(applications) != 1:
        raise DiscoveryError(
            f"{distribution!r} exposes {len(applications)} cozy.application entries; exactly one "
            "is required",
            code="application_entrypoint_count",
        )
    return _discover(applications[0], Path("/"), {}, seconds=seconds, add=None)


def _discover(
    application: str,
    project: Path,
    bindings: Mapping[str, Mapping[str, object]],
    *,
    seconds: float | None,
    add: Path | None,
) -> Discovered:
    if application.count(":") != 1 or not all(part.strip() for part in application.split(":")):
        raise DiscoveryError(
            f"{application!r} is not one <module>:<attribute> application",
            code="application_path",
        )
    module_name, _, attribute = application.partition(":")
    path = _sys_path(add) if add is not None else _no_path()
    with path, _wall_clock(seconds, application):
        module = _import(module_name, application)
    app = getattr(module, attribute, None)
    if app is None:
        others = sorted(n for n, v in vars(module).items() if isinstance(v, App))
        raise DiscoveryError(
            f"{module_name} exports no {attribute!r}"
            + (f" (it does export {', '.join(others)})" if others else " and no App at all"),
            code="missing_application",
        )
    if not isinstance(app, App):
        raise DiscoveryError(
            f"{application} is a {type(app).__name__}, not an App — the configured export IS "
            "the registry, and nothing else is discovered",
            code="not_an_app",
        )
    surfaces = describe(app)  # duplicate names already refused at registration; this freezes
    return Discovered(app, application, project, module, surfaces, bindings)


def _import(module_name: str, application: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        if missing == module_name:
            raise DiscoveryError(
                f"{application}: module {module_name!r} is not importable in this environment",
                code="missing_module",
            ) from exc
        raise DiscoveryError(
            f"{application}: importing {module_name} needs {missing!r}, which this environment "
            f"does not have — add it to pyproject.toml and re-lock\n\n{traceback.format_exc()}",
            code="missing_dependency",
        ) from exc
    except Exception as exc:
        # A broken submodule fails with the REAL traceback — never a silently dropped
        # function, never a summary of someone else's error (§1.0).
        raise DiscoveryError(
            f"{application}: importing {module_name} raised "
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            code="import_failed",
        ) from exc


def _bindings(project: Path) -> Mapping[str, Mapping[str, object]]:
    """package.toml's binding table. NOT interface content — the fixed-point scan reads it
    to prove no mutable binding leaked into the package interface."""
    from cozy_runtime.internal.bindings import read_table

    return read_table(project / "package.toml")


@contextmanager
def _sys_path(project: Path) -> Iterator[None]:
    added = [str(project / "src"), str(project)]
    sys.path[:0] = added
    try:
        yield
    finally:
        for entry in added:
            if entry in sys.path:
                sys.path.remove(entry)


@contextmanager
def _no_path() -> Iterator[None]:
    """The installed-environment discovery context: intentionally no path mutation."""

    yield


@contextmanager
def _wall_clock(seconds: float | None, application: str) -> Iterator[None]:
    """An optional ceiling on author top-level code. POSIX main thread only; elsewhere the builder's
    container bound is the one that holds, and this is a no-op rather than a false promise."""
    if (
        seconds is None
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def fire(signum: int, frame: FrameType | None) -> None:
        raise DiscoveryError(
            f"{application}: importing the app exceeded {seconds:g}s — heavy dependencies are "
            "imported but never EXECUTED at module scope (§1.0)",
            code="import_timeout",
        )

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
