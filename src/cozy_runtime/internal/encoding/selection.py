"""cr-006 seam 3 — SELECTION: device identity, admission by rule, and the launch table.

Seams one and two build providers. This one says which of them may serve THIS tensor on THIS
card, by rule (xs-065, owner 2026-10-22): nothing here reads a measurement, a shipped table or
a machine's history, so a card no one has benchmarked selects as a measured one of its row.

`admit` is the whole rule. A verbatim tensor always serves. A decoder serves on CPU and CUDA,
and on MPS unless the tensor stores float8 parts. A native leaf serves on CUDA when the card
clears its device floor and the card's `arch` row has every tensor-core format the leaf
declares (`LeafProvider.formats`): a lane runs natively wherever its tensor cores exist and
decodes only where they do not. Which kernel computes inside the leaf is its dispatch's own
rule (`int8_linear`, `fp8_linear`, `nvfp4_linear`): the family's prebuilt kernel, else a
generic one (cuBLASLt, Triton, torch). `resolution.py` takes one route per tensor.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NoReturn

from cozy_runtime.internal import accel, arch, canonical
from cozy_runtime.internal.encoding.adapters import CoreDequant
from cozy_runtime.internal.encoding.comfy import ComfyDequant
from cozy_runtime.internal.encoding.formats import (
    Encoded,
    Int8ConvRotDequant,
    LeafProvider,
    MicroScaledDequant,
    Nvfp4Dequant,
    Provider,
    RowwiseDequant,
    TensorwiseDequant,
    Verbatim,
    columnwise,
    refuse,
)
from cozy_runtime.internal.encoding.leaves import (
    Int8ConvRotNativeLeaf,
    MicroScaledNativeLeaf,
    Nvfp4NativeLeaf,
    RowwiseNativeLeaf,
)

# `torch: Any` is the torch module, which the check venv does not carry.

# ------------------------------------------------------------------------- the device


@dataclass(frozen=True, slots=True)
class DeviceFacts:
    """The card a plan resolves for: its class (`predicate`), its architecture (`sm`, which
    picks the `arch` row `admit` reads) and the facts a plan and a refusal name it by (#549.9):
    the REAL driver from NVML with torch's CUDA runtime beside it, the selected card's index
    and NVML configuration (power cap and MIG profile), and its OS and architecture.
    """

    kind: str
    name: str
    sm: int
    driver: str
    """The REAL NVIDIA driver version, from NVML. Never `torch.version.cuda`."""
    configuration: str
    index: int = 0
    """The selected card. NVML is queried with `-i <index>`."""
    cuda_runtime: str = ""
    """The CUDA runtime torch was built against — a BUILD fact, carried beside the driver
    rather than in place of it. This is the field that used to be called `driver`."""
    os_arch: str = ""
    """`Linux/x86_64` or `Linux/aarch64`."""

    @property
    def predicate(self) -> str:
        """The device CLASS a provider floor names: `cuda.sm89`, `mps`, `cpu`."""
        return f"cuda.sm{self.sm}" if self.kind == "cuda" else self.kind

    def satisfies(self, predicate: str) -> bool:
        """Does this device meet a provider's declared floor?

        `cuda.sm90+` admits sm90 and up; a bare `cuda.sm89` is exact; `any` admits all.
        The floor lives on the PROVIDER (a fact about an implementation), never on the
        spec (an immutable byte contract that hardware has no business inside).
        """
        if predicate == "any":
            return True
        if predicate in ("cpu", "mps"):
            return self.kind == predicate
        if not predicate.startswith("cuda.sm") or self.kind != "cuda":
            return False
        floor = predicate.removeprefix("cuda.sm")
        if floor.endswith("+"):
            return self.sm >= int(floor[:-1])
        return self.sm == int(floor)

    def tuple_key(self) -> str:
        """The device identity in one string, as a conformance report names it."""
        return (
            f"{self.kind}/{self.name}/dev{self.index}/{self.configuration}/"
            f"driver={self.driver}/cudart={self.cuda_runtime}/{self.os_arch}"
        )

    def document(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "name": self.name,
            "sm": self.sm,
            "driver": self.driver,
            "cuda_runtime": self.cuda_runtime,
            "configuration": self.configuration,
            "index": self.index,
            "os_arch": self.os_arch,
            "predicate": self.predicate,
        }


def measure_device(torch: Any, index: int = 0) -> DeviceFacts:
    """Read the SELECTED device once, as the facts a plan names it by.

    `index` is honoured all the way down (#549.9): torch is asked about that device and NVML
    is queried with `-i index`. Passing 0 and reading card 0 while filling card 3 is the
    defect this signature always looked like it did not have.

    An unreadable knob is SPELLED rather than defaulted.
    """
    kind = accel.host_backend_family()
    os_arch = f"{platform.system()}/{platform.machine()}"
    if not accel.present(torch, kind):
        return DeviceFacts(
            "cpu", "cpu", 0, "", "default", index=0, cuda_runtime="", os_arch=os_arch
        )
    identity = accel.device_identity(torch, kind, index)
    if kind == "mps":
        return DeviceFacts(
            kind=kind,
            name=str(identity["name"]),
            sm=0,
            driver=str(identity["driver_api"]),
            configuration=accel.memory_class(kind),
            os_arch=os_arch,
        )
    return DeviceFacts(
        kind=kind,
        name=str(identity["name"]),
        sm=int(identity["sm"]),
        # THE REAL DRIVER, from NVML, on the SELECTED device — never `torch.version.cuda`,
        # which is the wheel's build-time CUDA runtime and rides beside it as such (#549.9).
        # The CUDA driver API version torch reports is folded in so an unreadable NVML still
        # leaves a driver fact in the key rather than a bare "unreadable".
        driver=f"{_nvml('driver_version', index)}/api{identity['driver_api']}",
        configuration=f"{_nvml('power.limit', index)}W/{_nvml('mig.mode.current', index)}",
        index=int(index),
        cuda_runtime=str(identity["cuda_runtime"]),
        os_arch=os_arch,
    )


def _nvml(query: str, index: int) -> str:
    """One bounded NVML read OF THE SELECTED DEVICE (`-i {index}`, #549.9). `N/A` and a
    failure are both SPELLED."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unreadable"
    return (out.splitlines() or ["unreadable"])[0].strip().replace(" ", "") or "unreadable"


def device_line(device: DeviceFacts) -> str:
    """The device FACT, spelled the same way in every capability refusal. A refusal that
    says only "unsupported" sends a reader to the wrong question."""
    return (
        f"{device.predicate} ({device.name} #{device.index}, sm{device.sm}, "
        f"driver {device.driver}, cuda runtime {device.cuda_runtime or '-'}, "
        f"config {device.configuration}, {device.os_arch or '-'})"
    )


# ------------------------------------------------------------------- the runtime build


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """WHAT RAN a plan: the Runtime release and the torch build (#549.9)."""

    release: str
    torch_build: str

    def document(self) -> dict[str, str]:
        return {"release": self.release, "torch_build": self.torch_build}


def measure_runtime(torch: Any, release: str) -> RuntimeIdentity:
    return RuntimeIdentity(release=release, torch_build=str(getattr(torch, "__version__", "")))


# --------------------------------------------------------------------------- admission

#: The destinations a decoder or a native leaf writes; any other dtype is stored verbatim.
FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def admit(provider: Provider, device: DeviceFacts, output_dtype: str, encoded: Encoded) -> str:
    """ "" when `provider` may serve this tensor into `output_dtype` on `device`, else why not:
    a rule of the provider's declarations and the card's `arch` row, nothing measured."""
    if provider.route == "verbatim":
        return ""
    if output_dtype not in FLOAT_DTYPES:
        return f"{provider.name} writes float destinations and this one is {output_dtype}"
    if not device.satisfies(provider.device_predicate):
        return (
            f"{provider.name} needs {provider.device_predicate} and this device is "
            f"{device.predicate}"
        )
    if isinstance(provider, LeafProvider):
        if device.kind != "cuda":
            return f"{provider.name} runs on CUDA tensor cores and this device is {device.kind}"
        if output_dtype not in provider.out_dtypes:
            wrote = "/".join(sorted(provider.out_dtypes))
            return f"{provider.name} writes {wrote}, not {output_dtype}"
        row = arch.hardware(device.sm)
        lacking = sorted(provider.formats - row.formats())
        if lacking:
            return (
                f"{device.predicate} ({row.name}) has no {'/'.join(lacking)} tensor cores for "
                f"{provider.name}"
            )
        return ""
    if device.kind == "mps" and any(part.dtype.startswith("f8_") for part in encoded.parts):
        return f"{provider.name}: MPS has no float8, and this tensor stores float8 parts"
    return ""


# --------------------------------------------------------------------------- the seams

#: Routes this runtime names but does not build, by scheme: the device class that would serve
#: one, and what owns building it. A refusal reads this to name the alternative.
NAMED_SEAMS: dict[str, tuple[str, str]] = {
    "fp8.tensorwise": (
        "cuda.sm90+",
        "encoded-GEMM fp8 tensor-core execution via torch._scaled_mm; elsewhere the "
        "route is decoded_float, which this runtime serves",
    ),
}


# ------------------------------------------------------------------------ the launch table


def launch_providers() -> dict[str, tuple[Provider, ...]]:
    """Scheme -> its candidate providers, in the order `Selector` prefers them within a route:
    the first one `admit` lets serve a tensor is that route's."""
    out: dict[str, list[Provider]] = {}
    builds: tuple[Callable[[str], Provider], ...] = (
        Verbatim,
        # The decode cores first: Triton kernels on CUDA, where a scheme's own dequant is eager.
        CoreDequant,
        TensorwiseDequant,
        RowwiseDequant,
        columnwise,
        RowwiseNativeLeaf,
        Int8ConvRotDequant,
        Int8ConvRotNativeLeaf,
        Nvfp4Dequant,
        Nvfp4NativeLeaf,
        MicroScaledDequant,
        MicroScaledNativeLeaf,
        ComfyDequant,
    )
    for build in builds:
        for scheme in sorted(build("").schemes):
            out.setdefault(scheme, []).append(build(scheme))
    return {scheme: tuple(candidates) for scheme, candidates in out.items()}


# ------------------------------------------------------------------------------ digests


def implementation_digest(provider: Provider) -> str:
    """The provider's identity: name, route, device floor, oracle, parts, revision, schemes.
    A plan names the implementation it executes by it."""
    return canonical.digest(
        {
            "name": provider.name,
            "route": provider.route,
            "device_predicate": provider.device_predicate,
            "suite": provider.suite,
            "required": sorted(provider.required),
            "optional": sorted(provider.optional),
            "code_revision": provider.code_revision,
            "schemes": sorted(provider.schemes),
        }
    )


# ------------------------------------------------------------------------- the selector


@dataclass(frozen=True, slots=True)
class Selection:
    """What the fill plane needs to move and decode one tensor. Its provider is bound to the
    tensor's params."""

    provider: Provider
    encoded: Encoded
    reason: str = ""
    """Why this provider, in one sentence, carried into the resolved plan."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", replace(self.provider, params=self.encoded.params))  # type: ignore[type-var]


@dataclass(frozen=True, slots=True)
class Refusal:
    """A leaf this card does not serve for a tensor whose bytes and shape admit it (`admit`
    says why). Never a by-design route: a geometry the format rules out is
    `TensorCandidates.geometry_refused`, and a recipe's float layer is stored plain."""

    leaf: str
    reason: str


@dataclass(frozen=True, slots=True)
class TensorCandidates:
    """Every route this tensor CAN take on this device, plus what refused the others."""

    key: str
    encoded: Encoded
    output_dtype: str
    by_route: Mapping[str, Selection] = field(default_factory=dict)
    """route -> the first admitted implementation of it, in launch order."""
    geometry_refused: tuple[tuple[str, str], ...] = ()
    """(provider name, why) for candidates this tensor's own SHAPE ruled out (#549.5)."""
    unqualified: str = ""
    """Why no route serves this tensor on this device, when none does."""
    routes_offered: frozenset[str] = frozenset()
    """Every route this tensor's BYTES and SHAPE admit on some device."""
    refused: tuple[Refusal, ...] = ()
    """The encoded leaves this card does not serve for this tensor, with why."""


class Selector:
    """Per-tensor candidate resolution by scheme, by rule (`admit`)."""

    def __init__(self, providers: Mapping[str, tuple[Provider, ...]], device: DeviceFacts) -> None:
        self.providers = providers
        self.device = device

    def candidates(
        self, key: str, encoded: Encoded, output_dtype: str, shape: Sequence[int]
    ) -> TensorCandidates:
        """Every servable route for one tensor. Filters in order, each with its own remedy:
        scheme known, parts accounted for, shape and params supported, admitted here."""
        providers = self.providers.get(encoded.scheme)
        if not providers:
            self._refuse_unknown(key, encoded)
        among: list[Provider] = []
        mismatched: list[tuple[Provider, list[str], list[str]]] = []
        geometry: list[tuple[str, str]] = []
        params = dict(encoded.params)
        for listed in providers:
            candidate = replace(listed, params=encoded.params)  # type: ignore[type-var]
            missing = sorted(candidate.required - encoded.roles)
            surplus = sorted(encoded.roles - candidate.required - candidate.optional)
            if missing or surplus:
                mismatched.append((candidate, missing, surplus))
                continue
            why = candidate.supports(shape, params)
            if why:
                geometry.append((candidate.name, why))
                continue
            among.append(candidate)
        if not among and mismatched and not geometry:
            raise refuse(
                "role_mismatch",
                f"{key}: the header stores {encoded.scheme} parts {sorted(encoded.roles)} and no "
                "provider accounts for them — "
                + " | ".join(
                    f"{c.name} requires {sorted(c.required)}, tolerates {sorted(c.optional)} "
                    f"(missing {m}, unhandled {s})"
                    for c, m, s in mismatched
                )
                + ".",
                [key],
            )
        if not among:
            raise refuse(
                "geometry_unsupported",
                f"{key}: every implementation of {encoded.scheme} refuses this tensor "
                f"{tuple(shape)} {params} — "
                + " | ".join(f"{name}: {why}" for name, why in geometry),
                [key],
            )
        best: dict[str, Selection] = {}
        refused: list[Refusal] = []
        reasons: list[str] = []
        for provider in among:
            why = admit(provider, self.device, output_dtype, encoded)
            if why:
                reasons.append(why)
                if provider.route == "encoded_gemm":
                    refused.append(Refusal(provider.name, why))
            elif provider.route not in best:
                best[provider.route] = Selection(
                    provider, encoded, self._reason(provider, geometry)
                )
        seam = NAMED_SEAMS.get(encoded.scheme)
        unqualified = (
            ""
            if best
            else f"{encoded.scheme} has no provider on {device_line(self.device)} for output "
            f"dtype {output_dtype}: {' | '.join(reasons)}"
            + (f". The route that WOULD serve it is {seam[0]}: {seam[1]}" if seam else "")
        )
        return TensorCandidates(
            key,
            encoded,
            output_dtype,
            best,
            tuple(geometry),
            unqualified,
            frozenset(provider.route for provider in among),
            () if "encoded_gemm" in best else tuple(refused),
        )

    def _reason(self, provider: Provider, geometry: Sequence[tuple[str, str]]) -> str:
        why = f"{provider.name} [{provider.route}] on {self.device.predicate}"
        if geometry:
            why += "; geometry ruled out " + ", ".join(f"{n} ({w[:80]}…)" for n, w in geometry)
        return why

    def _refuse_unknown(self, key: str, encoded: Encoded) -> NoReturn:
        seam = NAMED_SEAMS.get(encoded.scheme)
        if seam is None:
            raise refuse(
                "unknown_encoding",
                f"{key}: the checkpoint stores scheme {encoded.scheme!r} and this runtime decodes "
                f"none of that name (it decodes {sorted(self.providers)})",
                [key],
            )
        raise refuse(
            "encoding_unqualified",
            f"{key}: this runtime has no provider for {encoded.scheme} on "
            f"{device_line(self.device)}; the route that would serve it is {seam[0]}: {seam[1]}",
            [key],
        )
