"""Resolve explicit model overrides against one captured callable graph."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from cozy_runtime.internal import package_interface
from cozy_runtime.protocol import worker_pb2 as pb

from .workspace_executions import ExecutionWorkspaceRefusal

type Key = tuple[str, str, str]
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SCALE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _refuse(detail: str) -> ExecutionWorkspaceRefusal:
    return ExecutionWorkspaceRefusal("model_override_invalid", detail)


def normalized(choice: pb.ModelChoice) -> pb.ModelChoice:
    copied = pb.ModelChoice()
    copied.CopyFrom(choice)
    if copied.source:
        if copied.repository or copied.release or copied.lane or copied.HasField("manifest"):
            raise _refuse("a model selects a provider source or a catalog checkpoint")
    elif copied.profiles:
        raise _refuse("source profiles require a provider source")
    if copied.HasField("manifest") and len(copied.manifest.digest) != 32:
        raise _refuse("a model manifest requires a SHA-256 digest")
    if not (copied.repository or copied.source or copied.HasField("manifest") or copied.adapters):
        raise _refuse("a model override must select a checkpoint or an adapter stack")
    for adapter in copied.adapters:
        if any(c in adapter.component + adapter.source_component for c in "=,: \t\n\r"):
            raise _refuse("adapter components must be single names")
        if adapter.source:
            if adapter.model or adapter.release or adapter.lane or adapter.manifest:
                raise _refuse("an adapter selects a provider source or a catalog checkpoint")
        elif not (adapter.model or adapter.manifest) or adapter.profiles:
            raise _refuse("an adapter requires a checkpoint; profiles require a provider source")
        if adapter.manifest and not re.fullmatch(r"sha256:[0-9a-f]{64}", adapter.manifest):
            raise _refuse("an adapter manifest requires a SHA-256 digest")
        adapter.source_component = adapter.source_component or "adapter"
        adapter.scale = adapter.scale or "1"
        if not _SCALE.fullmatch(adapter.scale):
            raise _refuse("adapter strength must be a finite decimal number")
        try:
            finite = math.isfinite(float(adapter.scale))
        except (ValueError, OverflowError):
            finite = False
        if not finite:
            raise _refuse("adapter strength must be finite")
    return copied


def resolve(
    capture: pb.MachineExecutionCapture,
    root_entrypoint: str,
    interfaces: Mapping[str, package_interface.PackageInterface],
) -> dict[Key, pb.ModelChoice]:
    """Return cloned choices keyed by exact callee, callable and model parameter.

    Bare names select only the root. Qualified names cannot broadcast to an
    unrelated callable with the same parameter, including reference generators.
    """
    if not capture.model_choices:
        return {}
    packages: dict[str, str] = {}
    for identity, package in (
        *((row.installation_id, row.package) for row in capture.installed_packages),
        *((row.key, row.package) for row in capture.deferred_installations),
    ):
        if not identity or not package or identity in packages:
            raise _refuse("captured package identities are missing or ambiguous")
        packages[identity] = package
    root = capture.root_installation_id
    if root not in packages:
        raise _refuse("the captured root package is absent")
    allowed = {(root, root_entrypoint)}
    for binding in capture.bindings:
        identity = binding.callee_installation_id or binding.callee_deferred_key
        if identity and binding.entrypoint:
            allowed.add((identity, binding.entrypoint))
    slots: set[Key] = set()
    for identity, name in allowed:
        interface = interfaces.get(identity)
        if identity not in packages or interface is None:
            continue
        declarations = [row for _, row in interface.callables() if row.name == name]
        if len(declarations) != 1:
            continue
        slots.update((identity, name, row.parameter) for row in declarations[0].models)
    result: dict[Key, pb.ModelChoice] = {}
    for original in capture.model_choices:
        selector = original.parameter
        if ".models." not in selector:
            if not _NAME.fullmatch(selector):
                raise _refuse("a bare model parameter must be a Python identifier")
            candidates = [(root, root_entrypoint, selector)]
        else:
            if selector.count(".models.") != 1:
                raise _refuse("a model selector must name one callable model slot")
            prefix, parameter = selector.split(".models.", 1)
            if not _NAME.fullmatch(parameter):
                raise _refuse("a model parameter must be a Python identifier")
            if "/" in prefix:
                package, name = prefix.rsplit("/", 1)
                candidates = [
                    key
                    for key in slots
                    if packages[key[0]] == package and key[1:] == (name, parameter)
                ]
            else:
                candidates = [(root, prefix, parameter)]
        candidates = [key for key in candidates if key in slots]
        if len(candidates) != 1:
            raise _refuse(f"model target {selector!r} is unknown or ambiguous in this capture")
        key = candidates[0]
        if key in result:
            raise _refuse(f"model target {selector!r} repeats an existing selection")
        choice = normalized(original)
        choice.parameter = key[2]
        result[key] = choice
    return result
