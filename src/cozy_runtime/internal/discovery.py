"""The isolation boundary: import exactly ONE configured export, freeze it, and stop.

That is the COMPLETE discovery algorithm (§1.0). Importing a module never registers
anything — registration is a method call on the app object — so there is nothing to
harvest, no package to walk, no dependency to stub and no source to parse. Everything a
descriptor says comes from the object package.toml named.

Execution containment is a TIER-INDEPENDENT property of the caller, not of this module:
the trusted build runs `describe` in a disposable container of the already-built release
image with no GPU, no weights, no network and no credentials (tensorhub-build.md). What
this module owns is the bound the runtime needs regardless of tier — a wall-clock ceiling
on author top-level code — and the refusal set around the one import.
"""

from __future__ import annotations

import importlib
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


def read_application(project: Path) -> str:
    """`[application] object` from package.toml, or the zero/ambiguous refusal."""
    path = project / "package.toml"
    if not path.is_file():
        raise DiscoveryError(
            f"{path} does not exist: a package repository has package.toml naming its "
            'one app export ([application] object = "my_package:app")',
            code="no_package_toml",
        )
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise DiscoveryError(f"{path} is not valid TOML: {exc}", code="package_toml") from exc
    application = data.get("application", {})
    obj = application.get("object") if isinstance(application, dict) else None
    if not isinstance(obj, str) or not obj:
        raise DiscoveryError(
            f"{path} names no [application] object — discovery imports exactly ONE configured "
            "export and nothing else, so zero named apps is ambiguous by construction",
            code="zero_application",
        )
    if obj.count(":") != 1 or not all(part.strip() for part in obj.split(":")):
        raise DiscoveryError(
            f'[application] object = "{obj}" is not <module>:<attribute>', code="application_path"
        )
    return obj


def discover(project: Path, *, seconds: float = IMPORT_SECONDS) -> Discovered:
    """Import the one configured export inside this environment and freeze it."""
    project = project.resolve()
    application = read_application(project)
    return _discover(application, project, _bindings(project), seconds=seconds, add=project)


def discover_installed(application: str, *, seconds: float = IMPORT_SECONDS) -> Discovered:
    """Freeze one descriptor-named export from the installed package environment.

    A materialized rental has an immutable venv, not a copied source tree or package.toml.
    The exact PackageDescriptor supplies the application identity; importing anything else
    or adding a host path would recreate the source/PYTHONPATH authority the adapter rejects.
    """

    return _discover(application, Path("/"), {}, seconds=seconds, add=None)


def _discover(
    application: str,
    project: Path,
    bindings: Mapping[str, Mapping[str, object]],
    *,
    seconds: float,
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
    """package.toml's binding table. NOT descriptor content — the fixed-point scan reads it
    to prove no mutable binding leaked into the descriptor."""
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
def _wall_clock(seconds: float, application: str) -> Iterator[None]:
    """A ceiling on author top-level code. POSIX main thread only; elsewhere the builder's
    container bound is the one that holds, and this is a no-op rather than a false promise."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
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
