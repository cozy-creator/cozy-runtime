"""Source-authored model defaults; names are resolved by the granting host.

A default names a release, `[org/]model@release`, never a lane: the machine picks the lane
from the GPU and the request's bits. A default with no org, or the org `_`, names a first-party
model, the root namespace's."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from cozy_runtime.author._errors import ConformanceError

_SLUG = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?"
_DEFAULT = re.compile(rf"(?:(?:_|{_SLUG})/)?{_SLUG}@(?P<release>[A-Za-z0-9][A-Za-z0-9._+!-]{{0,63}})")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def model_default(value: object) -> str:
    """Validate one `[org/]model@release` without resolving it."""
    match = _DEFAULT.fullmatch(value) if isinstance(value, str) else None
    if match is None or _DIGEST.fullmatch(match["release"]):
        raise ConformanceError(
            f"model default {value!r} must be [org/]model@release: an explicit release, "
            "never a digest, and no lane (the machine picks the lane)",
            code="model_defaults",
        )
    return match.string


def model_defaults(value: object) -> dict[str, str]:
    """Copy a `defaults=` declaration: Model argument -> its default release."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 16:
        raise ConformanceError(
            "defaults must map at most 16 Model arguments to releases", code="model_defaults"
        )
    out = {}
    for parameter, default in value.items():
        if not isinstance(parameter, str) or not parameter.isidentifier():
            raise ConformanceError(
                "model defaults keys must be argument names", code="model_defaults"
            )
        out[parameter] = model_default(default)
    return out


def model_components(value: object) -> dict[str, tuple[str, ...]]:
    """Copy a `components=` declaration: Model argument -> the components the callable runs.
    Absent means all of them."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 16:
        raise ConformanceError(
            "components must map at most 16 Model arguments to component names",
            code="model_components",
        )
    out = {}
    for parameter, names in value.items():
        if not isinstance(parameter, str) or not parameter.isidentifier():
            raise ConformanceError(
                "model components keys must be argument names", code="model_components"
            )
        if (
            not isinstance(names, (list, tuple))
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ConformanceError(
                f"components[{parameter!r}] must be a non-empty sequence of unique names",
                code="model_components",
            )
        out[parameter] = tuple(names)
    return out


def model_phases(value: object) -> tuple[tuple[str, ...], ...]:
    """Normalize literal component groups for both source and imported registration."""
    if not isinstance(value, (tuple, list)):
        raise ConformanceError(
            "phases must be a sequence of component groups", code="model_phases"
        )
    out: list[tuple[str, ...]] = []
    for phase in value:
        names = (phase,) if isinstance(phase, str) else phase
        if (
            not isinstance(names, (tuple, list, set, frozenset))
            or not names
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ConformanceError(
                "each phase must name nonempty component strings", code="model_phases"
            )
        out.append(tuple(sorted(set(names))))
    return tuple(out)



def model_on_demand(value: object) -> tuple[str, ...]:
    """Copy an `on_demand=` declaration: the Model arguments built on first use."""
    if value is None:
        return ()
    if (
        not isinstance(value, (list, tuple))
        or any(not isinstance(name, str) or not name.isidentifier() for name in value)
        or len(set(value)) != len(value)
    ):
        raise ConformanceError(
            "on_demand must be a sequence of unique Model argument names", code="model_on_demand"
        )
    return tuple(value)


#: The longest rule: one small boolean expression, never a program.
MAX_NEEDS_RULE = 1024
#: A component of a Model argument, as a `needs=` key names it: `model.vision`.
_COMPONENT_KEY = re.compile(r"(?P<argument>[A-Za-z_][A-Za-z0-9_]*)\.(?P<component>[A-Za-z0-9_][A-Za-z0-9_-]*)")


def needs_component(key: str) -> tuple[str, str] | None:
    """A `needs=` key's Model argument and component, where it names one (`model.vision`)."""
    match = _COMPONENT_KEY.fullmatch(key)
    return None if match is None else (match["argument"], match["component"])


def model_needs(value: object) -> dict[str, str]:
    """Copy a `needs=` declaration: a Model argument, a component of one
    (`<argument>.<component>`), or a function this one calls, -> the CEL expression over
    `request` that says whether a request needs it."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 16:
        raise ConformanceError(
            "needs must map at most 16 Model arguments or called functions to rules",
            code="model_needs",
        )
    out = {}
    for name, rule in value.items():
        if not isinstance(name, str) or not (name.isidentifier() or needs_component(name)):
            raise ConformanceError(
                "needs keys must be Model argument or function names, or "
                "<argument>.<component>",
                code="model_needs",
            )
        if not isinstance(rule, str) or not rule.strip() or len(rule) > MAX_NEEDS_RULE:
            raise ConformanceError(
                f"needs[{name!r}] must be one CEL expression over `request`, at most "
                f"{MAX_NEEDS_RULE} characters",
                code="model_needs",
            )
        out[name] = rule.strip()
    return out


def check_model_arguments(
    declared: Mapping[str, object], parameters: Sequence[str], *, what: str = "defaults"
) -> None:
    unknown = sorted(set(declared) - set(parameters))
    if unknown:
        raise ConformanceError(
            f"{what} name arguments that are not injected Models: " + ", ".join(unknown),
            code=f"model_{what}",
            fields=unknown,
        )
