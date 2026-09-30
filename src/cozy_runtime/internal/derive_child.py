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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

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
_Name = Annotated[str, msgspec.Meta(min_length=1)]


class DeriveRefusal(Exception):
    """One stable construction refusal carried across the child boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class SlotRequest(msgspec.Struct, frozen=True):
    """One slot's wire row; both sides of the pipe encode and decode this struct."""

    slot: _Name
    config: bytes
    assets: dict[_Name, bytes] = {}
    tensor_dtypes: dict[_Name, _Name] | None = None
    adapters: bytes = b""


class _Package(msgspec.Struct, frozen=True, omit_defaults=True):
    project: str = ""
    application: str = ""


class _Request(msgspec.Struct, frozen=True):
    package: _Package
    slots: Annotated[tuple[SlotRequest, ...], msgspec.Meta(min_length=1)]


@dataclass(frozen=True, slots=True)
class DeriveRequest:
    """One config per sorted, unique package model-slot path."""

    slots: tuple[SlotRequest, ...]
    project: Path | None = None
    application: str = ""

    def render(self) -> _Request:
        if self.application:
            return _Request(_Package(application=self.application), self.slots)
        return _Request(_Package(project=str(self.project)), self.slots)


class _Tensor(msgspec.Struct, frozen=True):
    component: str
    key: _Name
    dtype: str | None
    shape: tuple[Annotated[int, msgspec.Meta(ge=0)], ...]


class _Answer(msgspec.Struct, frozen=True):
    slot: str
    source: bool = False
    components: tuple[_Name, ...] = ()
    tensors: tuple[_Tensor, ...] = ()


class DeriveResponse(msgspec.Struct, frozen=True):
    """The child's one result line: requirements, or a refusal's code and detail."""

    requirements: tuple[_Answer, ...] = ()
    error: str | None = None
    detail: str | None = None
    model_adapters: tuple[str, ...] = ()


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
        wire = msgspec.msgpack.decode(raw, type=_Request)
    except msgspec.DecodeError as err:
        raise DeriveRefusal("derive_request_invalid", str(err)) from err
    project, application = wire.package.project, wire.package.application
    if bool(project) == bool(application):
        raise DeriveRefusal("derive_request_invalid", "package names a project or an application")
    if project and not Path(project).is_absolute():
        raise DeriveRefusal("derive_request_invalid", "project is an absolute path")
    if application and (
        application.count(":") != 1 or not all(p.strip() for p in application.split(":"))
    ):
        raise DeriveRefusal("derive_request_invalid", "application is <module>:<attribute>")
    for row in wire.slots:
        try:
            document = canonical.parse_canonical(row.config)
        except canonical.CanonicalError as err:
            raise DeriveRefusal(
                "derive_request_invalid", f"slot {row.slot!r} config: {err}"
            ) from err
        if not isinstance(document, dict):
            raise DeriveRefusal(
                "derive_request_invalid", f"slot {row.slot!r} config is not an object"
            )
    names = [row.slot for row in wire.slots]
    if names != sorted(set(names)):
        raise DeriveRefusal("derive_request_invalid", "slots are sorted unique by path")
    return DeriveRequest(
        wire.slots, project=Path(project) if project else None, application=application
    )


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
            input=msgspec.msgpack.encode(request.render(), order="deterministic"),
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
        response = msgspec.msgpack.decode(
            base64.b64decode(encoded, validate=True), type=DeriveResponse
        )
    except ValueError as err:  # binascii.Error or msgspec.DecodeError
        raise DeriveRefusal("derive_child_invalid", str(err)) from err
    _require_adapters(response, request)
    if response.error is not None:
        raise DeriveRefusal(response.error or "derive_child_failed", response.detail or "refused")
    if result.returncode != 0:
        raise DeriveRefusal("derive_child_failed", f"child exited {result.returncode}")
    return read_response(response, request)


def _require_adapters(response: DeriveResponse, request: DeriveRequest) -> None:
    from cozy_runtime.internal.lora_contract import CAPABILITY

    if any(row.adapters for row in request.slots) and CAPABILITY not in response.model_adapters:
        raise DeriveRefusal(
            "adapter_composition_unsupported",
            "the package SDK cannot construct prepared LoRA graphs; update its Runtime dependency",
        )


def read_response(
    response: DeriveResponse, request: DeriveRequest
) -> tuple[TensorRequirements | SourceSlot, ...]:
    _require_adapters(response, request)
    out: list[TensorRequirements | SourceSlot] = []
    for answer in response.requirements:
        if answer.source:
            out.append(SourceSlot(answer.slot))
            continue
        components = answer.components
        if not components or len(set(components)) != len(components):
            raise DeriveRefusal("derive_child_invalid", "requirements shape")
        if any(tensor.component not in components for tensor in answer.tensors):
            raise DeriveRefusal("derive_child_invalid", "tensor row value")
        rows = tuple((t.component, t.key, t.dtype, t.shape) for t in answer.tensors)
        out.append(TensorRequirements(answer.slot, rows, components))
    if [answer.slot for answer in out] != [row.slot for row in request.slots]:
        raise DeriveRefusal("derive_child_invalid", "rows do not answer the requested slots")
    return tuple(out)


def _render(requirements: Sequence[TensorRequirements | SourceSlot]) -> dict[str, object]:
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


def _emit(value: dict[str, object]) -> None:
    from cozy_runtime.internal.lora_contract import CAPABILITY

    value["model_adapters"] = [CAPABILITY]
    sys.stdout.buffer.write(RESULT_PREFIX + base64.b64encode(msgspec.msgpack.encode(value)) + b"\n")


def _derive_slot(
    binding: ModelBinding, config: Config, assets: Mapping[str, bytes], adapters: bytes = b""
) -> TensorRequirements:
    """Derive the ordered construction once; serving validates its actual destinations.

    Rebuilding here only samples determinism, which belongs in author tests. It
    cannot replace the serving Loader's census/fit or component-agreement checks.
    """

    from cozy_runtime.author import Artifact
    from cozy_runtime.internal.derive import derive
    from cozy_runtime.internal.lora_composition import bind

    result = derive(
        binding.model_class(),
        bind(
            Artifact("derive://request", {}, config, assets=assets, custody="canonical"),
            binding.model_class,
            adapters,
        ),
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
    from cozy_runtime.author import CapabilityError, Config, ConformanceError
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
            if row.slot in sources:
                requirements.append(SourceSlot(row.slot))
            else:
                requirements.append(
                    _derive_slot(
                        binding,
                        Config(
                            canonical.parse_canonical(row.config),
                            "config",
                            tensor_dtypes=row.tensor_dtypes,
                        ),
                        row.assets,
                        row.adapters,
                    )
                )
    except DeriveRefusal as err:
        _emit({"error": err.code, "detail": err.detail})
        return int(Exit.structural)
    except (ConformanceError, CapabilityError) as err:
        _emit({"error": err.code, "detail": str(err)})
        return int(Exit.structural)
    except (DeriveError, sandbox.FenceViolation) as err:
        name = err.construct if isinstance(err, DeriveError) else err.fence
        _emit({"error": name or "derive_refused", "detail": str(err)})
        return int(Exit.structural)
    except Exception as err:
        _emit({"error": "derive_internal", "detail": f"{type(err).__name__}: {err}"})
        return int(Exit.internal)
    _emit(_render(requirements))
    return int(Exit.ok)


if __name__ == "__main__":
    raise SystemExit(main())
