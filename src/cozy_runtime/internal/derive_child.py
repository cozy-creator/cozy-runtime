"""The private bounded child for request-time model construction.

The worker sends config bytes through a pipe to the package's own interpreter. The child
constructs on ``meta`` and returns transient tensor rows through a pipe; neither side writes,
names, hashes, or retains a TensorRequirements document.
"""

from __future__ import annotations

import base64
import importlib.util
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from cozy_runtime.internal import canonical, sandbox
from cozy_runtime.internal.exits import Exit

if TYPE_CHECKING:
    from cozy_runtime.author import Config, ModelBinding

RESULT_PREFIX = b"COZY_DERIVE="
MEMORY_CAP_BYTES = 24 << 30
CPU_CAP_SECONDS = 600
WALL_CAP_SECONDS = 900
PIPE_CAP_BYTES = 512 << 20

RequirementRow = tuple[str, str, str | None, tuple[int, ...]]


class DeriveRefusal(Exception):
    """One stable construction refusal carried across the child boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class SlotRequest:
    slot: str
    config: bytes
    assets: Mapping[str, bytes] = field(default_factory=dict)
    tensor_dtypes: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class DeriveRequest:
    """One config per sorted, unique package model-slot path."""

    slots: tuple[SlotRequest, ...]
    project: Path | None = None
    application: str = ""

    def render(self) -> dict[str, Any]:
        package: dict[str, Any] = (
            {"application": self.application}
            if self.application
            else {"project": str(self.project)}
        )
        return {
            "package": package,
            "slots": [
                {
                    "slot": row.slot,
                    "config": row.config,
                    "assets": dict(sorted(row.assets.items())),
                    "tensor_dtypes": None if row.tensor_dtypes is None else dict(row.tensor_dtypes),
                }
                for row in self.slots
            ],
        }


@dataclass(frozen=True, slots=True)
class TensorRequirements:
    """Transient request-time rows. This value is never a file or durable identity."""

    slot: str
    rows: tuple[RequirementRow, ...]
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceSlot:
    """A declared job Model input; its attempt holds bytes without constructing them."""

    slot: str


def read_request(raw: bytes) -> DeriveRequest:
    try:
        value = msgspec.msgpack.decode(raw)
    except msgspec.DecodeError as err:
        raise DeriveRefusal("derive_request_invalid", str(err)) from err
    if not isinstance(value, dict) or set(value) != {"package", "slots"}:
        raise DeriveRefusal("derive_request_invalid", "request field set")
    package = value["package"]
    if not isinstance(package, dict) or len(package) != 1:
        raise DeriveRefusal("derive_request_invalid", "package names a project or an application")
    project: Path | None = None
    application = ""
    if "project" in package:
        spelled = package["project"]
        if not isinstance(spelled, str) or not Path(spelled).is_absolute():
            raise DeriveRefusal("derive_request_invalid", "project is an absolute path")
        project = Path(spelled)
    elif "application" in package:
        application = str(package["application"])
        if application.count(":") != 1 or not all(p.strip() for p in application.split(":")):
            raise DeriveRefusal("derive_request_invalid", "application is <module>:<attribute>")
    else:
        raise DeriveRefusal("derive_request_invalid", "package names a project or an application")
    encoded_rows = value["slots"]
    if not isinstance(encoded_rows, list) or not encoded_rows:
        raise DeriveRefusal("derive_request_invalid", "slots is a non-empty array")
    slots: list[SlotRequest] = []
    for row in encoded_rows:
        if not isinstance(row, dict) or set(row) != {"slot", "config", "assets", "tensor_dtypes"}:
            raise DeriveRefusal("derive_request_invalid", "slot row field set")
        slot = row["slot"]
        config = row["config"]
        encoded_assets = row["assets"]
        if (
            not isinstance(slot, str)
            or not slot
            or not isinstance(config, bytes)
            or not isinstance(encoded_assets, dict)
        ):
            raise DeriveRefusal("derive_request_invalid", f"slot {slot!r}")
        try:
            document = canonical.parse_canonical(config)
        except canonical.CanonicalError as err:
            raise DeriveRefusal("derive_request_invalid", f"slot {slot!r} config: {err}") from err
        if not isinstance(document, dict):
            raise DeriveRefusal("derive_request_invalid", f"slot {slot!r} config is not an object")
        assets: dict[str, bytes] = {}
        for name, value in encoded_assets.items():
            if not isinstance(name, str) or not name or not isinstance(value, bytes):
                raise DeriveRefusal("derive_request_invalid", f"slot {slot!r} asset")
            assets[name] = value
        dtypes = row["tensor_dtypes"]
        if dtypes is not None and (
            not isinstance(dtypes, dict)
            or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in dtypes.items()
            )
        ):
            raise DeriveRefusal("derive_request_invalid", f"slot {slot!r} tensor dtypes")
        slots.append(SlotRequest(slot, config, assets, dtypes))
    names = [row.slot for row in slots]
    if names != sorted(set(names)):
        raise DeriveRefusal("derive_request_invalid", "slots are sorted unique by path")
    return DeriveRequest(tuple(slots), project=project, application=application)


def derive_in(python: Path, request: DeriveRequest) -> tuple[TensorRequirements | SourceSlot, ...]:
    """Construct in the package interpreter and return only transient in-memory rows."""

    child_env = sandbox.child_environment({})
    child_env.update(
        {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}
    )
    try:
        result = subprocess.run(
            [str(python), "-I", "-B", "-m", "cozy_runtime.internal.derive_child"],
            cwd="/",
            env=child_env,
            input=msgspec.msgpack.encode(request.render()),
            capture_output=True,
            timeout=WALL_CAP_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise DeriveRefusal("derive_child_failed", f"{type(err).__name__}: {err}") from err
    encoded = next(
        (
            line.removeprefix(RESULT_PREFIX)
            for line in reversed(result.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        b"",
    )
    if not encoded:
        stderr = result.stderr.decode(errors="replace").strip()
        if "No module named cozy_runtime" in stderr or "No module named 'cozy_runtime'" in stderr:
            raise DeriveRefusal(
                "derive_runtime_absent",
                f"{python}: its cozy-runtime has no private request-time derivation worker; "
                "relock the package to the current runtime",
            )
        raise DeriveRefusal(
            "derive_child_failed",
            stderr[-1200:] or f"child exited {result.returncode} without a result",
        )
    try:
        response = msgspec.msgpack.decode(base64.b64decode(encoded, validate=True))
    except (ValueError, msgspec.DecodeError) as err:
        raise DeriveRefusal("derive_child_invalid", str(err)) from err
    if not isinstance(response, dict):
        raise DeriveRefusal("derive_child_invalid", "child result is not an object")
    if "error" in response:
        raise DeriveRefusal(
            str(response.get("error") or "derive_child_failed"),
            str(response.get("detail") or "refused"),
        )
    if result.returncode != 0:
        raise DeriveRefusal("derive_child_failed", f"child exited {result.returncode}")
    return read_response(response, request)


def read_response(
    response: Mapping[str, Any], request: DeriveRequest
) -> tuple[TensorRequirements | SourceSlot, ...]:
    if set(response) != {"requirements"} or not isinstance(response["requirements"], list):
        raise DeriveRefusal("derive_child_invalid", "requirements field")
    out: list[TensorRequirements | SourceSlot] = []
    for answer in response["requirements"]:
        if isinstance(answer, dict) and set(answer) == {"slot", "source"}:
            if not isinstance(answer["slot"], str) or answer["source"] is not True:
                raise DeriveRefusal("derive_child_invalid", "source slot field value")
            out.append(SourceSlot(answer["slot"]))
            continue
        if not isinstance(answer, dict) or set(answer) != {"slot", "components", "tensors"}:
            raise DeriveRefusal("derive_child_invalid", "requirements row field set")
        components = answer["components"]
        tensors = answer["tensors"]
        if (
            not isinstance(components, list)
            or not components
            or not all(isinstance(component, str) and component for component in components)
            or len(set(components)) != len(components)
            or not isinstance(tensors, list)
        ):
            raise DeriveRefusal("derive_child_invalid", "requirements shape")
        rows: list[RequirementRow] = []
        for tensor in tensors:
            if not isinstance(tensor, dict) or set(tensor) != {
                "component",
                "key",
                "dtype",
                "shape",
            }:
                raise DeriveRefusal("derive_child_invalid", "tensor row field set")
            component, key, dtype, shape = (
                tensor["component"],
                tensor["key"],
                tensor["dtype"],
                tensor["shape"],
            )
            if (
                not isinstance(component, str)
                or component not in components
                or not isinstance(key, str)
                or not key
                or (dtype is not None and not isinstance(dtype, str))
                or not isinstance(shape, list)
                or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape)
            ):
                raise DeriveRefusal("derive_child_invalid", "tensor row value")
            rows.append((component, key, dtype, tuple(shape)))
        out.append(TensorRequirements(str(answer["slot"]), tuple(rows), tuple(components)))
    if [answer.slot for answer in out] != [row.slot for row in request.slots]:
        raise DeriveRefusal("derive_child_invalid", "rows do not answer the requested slots")
    return tuple(out)


def _render(requirements: Sequence[TensorRequirements | SourceSlot]) -> dict[str, Any]:
    return {
        "requirements": [
            {"slot": answer.slot, "source": True}
            if isinstance(answer, SourceSlot)
            else {
                "slot": answer.slot,
                "components": list(answer.components),
                "tensors": [
                    {
                        "component": component,
                        "key": key,
                        "dtype": dtype,
                        "shape": list(shape),
                    }
                    for component, key, dtype, shape in answer.rows
                ],
            }
            for answer in requirements
        ]
    }


def _emit(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(
        RESULT_PREFIX + base64.b64encode(msgspec.msgpack.encode(dict(value))) + b"\n"
    )


def _derive_slot(
    binding: ModelBinding, config: Config, assets: Mapping[str, bytes]
) -> TensorRequirements:
    """Derive the ordered construction once; serving validates its actual destinations.

    Rebuilding here only samples determinism, which belongs in author tests. It
    cannot replace the serving Loader's census/fit or component-agreement checks.
    """

    from cozy_runtime.author import Artifact
    from cozy_runtime.internal.derive import derive

    result = derive(
        binding.model_class(),
        Artifact("derive://request", {}, config, assets=assets, custody="canonical"),
        component_use=binding.components,
    )
    rows = tuple(
        (component, key, dtype, tuple(shape)) for component, key, dtype, shape in result.rows()
    )
    return TensorRequirements(binding.path, rows, result.components)


def _warm_torch() -> None:
    """Warm torch's lazy imports before rlimits, never package code or a GPU."""

    if importlib.util.find_spec("torch") is None:
        return  # Job SourceSlots inspect declarations without constructing Torch models.
    __import__("torch")
    for name in ("torch._dynamo", "torch._refs"):
        try:
            __import__(name)
        except ModuleNotFoundError as err:
            if err.name != name:
                raise


def main() -> int:
    from cozy_runtime.author import Config, ConformanceError
    from cozy_runtime.internal.derive import DeriveError
    from cozy_runtime.internal.discovery import discover, discover_installed

    try:
        raw = sys.stdin.buffer.read(PIPE_CAP_BYTES + 1)
        if len(raw) > PIPE_CAP_BYTES:
            raise DeriveRefusal("derive_request_invalid", f"request exceeds {PIPE_CAP_BYTES} bytes")
        request = read_request(raw)
        _warm_torch()
        sandbox.install(allow_write_prefixes=())
        sandbox.bound(address_bytes=MEMORY_CAP_BYTES, cpu_seconds=CPU_CAP_SECONDS)
        found = (
            discover(request.project)
            if request.project is not None
            else discover_installed(request.application)
        )
        bindings = {
            binding.path: binding
            for surface in found.surfaces
            if not surface.hidden
            for binding in surface.model_bindings
        }
        sources = {
            binding.path
            for surface in found.surfaces
            if not surface.hidden and surface.kind == "job"
            for binding in surface.model_bindings
        }
        requirements: list[TensorRequirements | SourceSlot] = []
        for row in request.slots:
            binding = bindings.get(row.slot)
            if binding is None:
                raise DeriveRefusal(
                    "derive_slot_unknown",
                    f"{row.slot!r} is no model slot of this package "
                    f"(have {', '.join(sorted(bindings)) or 'none'})",
                )
            document = canonical.parse_canonical(row.config)
            if not isinstance(document, dict):
                raise DeriveRefusal("derive_input_invalid", f"{row.slot}: config is not an object")
            if row.slot in sources:
                requirements.append(SourceSlot(row.slot))
            else:
                requirements.append(
                    _derive_slot(
                        binding,
                        Config(document, "config", tensor_dtypes=row.tensor_dtypes),
                        row.assets,
                    )
                )
    except DeriveRefusal as err:
        _emit({"error": err.code, "detail": err.detail})
        return int(Exit.structural)
    except ConformanceError as err:
        _emit({"error": str(getattr(err, "code", "derive_conformance")), "detail": str(err)})
        return int(Exit.structural)
    except (DeriveError, sandbox.FenceViolation) as err:
        name = getattr(err, "construct", None) or getattr(err, "fence", None)
        _emit({"error": str(name or "derive_refused"), "detail": str(err)})
        return int(Exit.structural)
    except Exception as err:
        _emit({"error": "derive_internal", "detail": f"{type(err).__name__}: {err}"})
        return int(Exit.internal)
    _emit(_render(requirements))
    return int(Exit.ok)


if __name__ == "__main__":
    raise SystemExit(main())
