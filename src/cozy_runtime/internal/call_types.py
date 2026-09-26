"""Bind generated wire results to the installed callable's public Python types."""

from __future__ import annotations

import contextlib
import typing
from dataclasses import replace
from types import ModuleType
from typing import Any

from cozy_runtime.author import ConformanceError
from cozy_runtime.author._calls import _CallType, _export
from cozy_runtime.author._walker import is_struct
from cozy_runtime.internal import schema


def public_result(module: ModuleType, binding: _CallType) -> _CallType:
    """Names locate a public type; its complete wire shape establishes the join.

    The wire decoder retains its generated type and strict unknown-field policy.
    Only an admitted result is converted to this installed class for the author.
    """
    function = getattr(module, binding.export, None)
    exported = _export(function) if callable(function) else None
    candidates: list[Any] = []
    previous = getattr(module, "__cozy_bindings__", {}).get(binding.export)
    if isinstance(previous, _CallType):
        candidates.append(previous.python_result)
    if exported is not None:
        candidates.append(
            typing.get_type_hints(exported.implementation, include_extras=True).get("return")
        )
    elif callable(function):
        with contextlib.suppress(NameError, TypeError):
            candidates.append(
                typing.get_type_hints(function, globalns=vars(module), include_extras=True).get(
                    "return"
                )
            )
    candidates.append(getattr(module, binding.result.__name__, None))
    expected = schema.render(binding.result, decoded_bounds=True)
    for candidate in candidates:
        if not is_struct(candidate):
            continue
        if (
            schema.render(candidate, decoded_bounds=True) != expected
            or candidate.__struct_config__.array_like
        ):
            raise ConformanceError(
                "installed result differs from the captured interface", code="child_interface"
            )
        return replace(binding, python_result=candidate)
    return binding
