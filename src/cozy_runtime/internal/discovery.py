"""The executor's import of ONE installed export, to run it: import it, freeze it, stop.

This never describes a package. Its interface is read from source alone
(`static_interface`, #713); the executor imports the release only to serve it, inside the
release's own installed environment. Importing a module never registers anything —
registration is a method call on the app object — so there is nothing to harvest, no package
to walk and no dependency to stub.
"""

from __future__ import annotations

import importlib
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

from cozy_runtime.author import App
from cozy_runtime.author._describe import freeze
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._recipes import DeclaredRecipe
from cozy_runtime.author._signature import Surface
from cozy_runtime.derive import recipes as declared_recipes
from cozy_runtime.internal.static_interface import missing_app


class DiscoveryError(ConformanceError):
    """The installed release cannot be imported as its interface names it."""

    default_code = "discovery"


@dataclass(frozen=True, slots=True)
class Discovered:
    """The frozen app plus the application path its interface names."""

    app: App
    application: str
    module: ModuleType
    surfaces: tuple[Surface, ...]
    #: the declared recipes it serves as `quantize`
    recipes: tuple[DeclaredRecipe, ...] = ()


def discover_installed(application: str, *, recipes: Sequence[DeclaredRecipe] = ()) -> Discovered:
    """The app at `application` in the installed environment, serving the `recipes` its interface
    declares (`quantize`). No sys.path change: the materialized venv is the only authority."""
    if application.count(":") != 1 or not all(part.strip() for part in application.split(":")):
        raise DiscoveryError(
            f"{application!r} is not one <module>:<attribute> application",
            code="application_path",
        )
    module_name, _, attribute = application.partition(":")
    module = _import(module_name, application)
    app = getattr(module, attribute, None)
    if app is None:
        others = sorted(n for n, v in vars(module).items() if isinstance(v, App))
        raise DiscoveryError(
            missing_app(module_name, attribute, others),
            code="missing_application",
        )
    if not isinstance(app, App):
        raise DiscoveryError(
            f"{application} is a {type(app).__name__}, not an App — the configured export IS "
            "the registry, and nothing else is discovered",
            code="not_an_app",
        )
    if recipes:
        declared_recipes.register(app, recipes)
    surfaces = freeze(app)  # duplicate names already refused at registration
    return Discovered(app, application, module, surfaces, tuple(recipes))


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
