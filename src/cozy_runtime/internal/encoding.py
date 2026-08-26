"""cr-006 — the selected-encoding compute plane: which module executes which stored bytes.

cr-005 moves bytes that already ARE the destination's dtype. This module is the other
case: a checkpoint whose per-tensor encoding says the stored bytes are a CODE for the
logical tensor, and something has to turn the one into the other.

Three rules hold the shape, and each is a refusal rather than a convention:

**Selection is by DIGEST, never by name.** The header assigns every tensor an
`EncodingSpec` object digest. That digest — not the alias, not the key, not the dtype —
picks the provider. Aliases (`fp8-scaled-scalar/1`) exist for display and are read from
`tensorfs.seed_digests()` so a digest can never be transcribed wrong into this file.
Two physically different role sets under one alias are two digests and two providers,
which is exactly what the H3 production artifact carries (150 modules with an
`input_scale`, 50 `mlp.fc2` without).

**The decode target is the CONTRACT's dtype, and the contract cannot express an
encoding.** `ModelConstructionContract` names destinations and nothing else — no parts,
no scale_of, no dequantize (contract.py's `FORBIDDEN_VOCABULARY` makes that an import-time
proof). So the decode target is the destination's own dtype, the header's logical dtype
must equal it, and a disagreement is cr-005's `dtype_mismatch` unchanged. Nothing about
quantization reaches the author surface or the contract.

**Capability is fail-closed and device-keyed.** A provider serves an (encoding, device)
pair only when a CapabilityRecord exists for it, and a record cannot be written without
the numerics evidence that minted it (`probe.py`). An unrecognised device is the ABSENCE
of evidence, never permission — v1's `DTYPE_MIN_SM` fell OPEN on an unknown card and
served plausible wrong output.

**The device-honesty split, stated once.** On sm89 (this repo's development card) fp8
E4M3 is a STORAGE dtype: the tensor cores do not consume it, `torch._scaled_mm` needs
sm90. So the launch tier's fp8 route is `decoded_float` — dequantize at fill, compute in
the destination's float dtype, which is what the consumer-card ecosystem does and what
weight-only quantization means. It buys bytes read and load time, NOT VRAM: the resident
weights are float either way. The sm90+ scaled-mm route and the SM100 mxfp8 native route
are NAMED here as seams with their owning milestone; they are not built, and an artifact
that needs one refuses typed naming the device fact rather than silently taking the
float route it was not qualified for.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from cozy_runtime.internal import accel

#: This module's own refusal codes, added to cr-005's matrix. Closed, like that one.
REFUSALS = frozenset(
    {
        "unknown_encoding",  # the header cites a spec digest no provider claims
        "encoding_unqualified",  # a claimed encoding with no capability record for THIS device
        "role_mismatch",  # the delivered parts are not the roles the provider decodes
    }
)


class EncodingRefusal(Exception):
    """A typed encoding refusal. Every one fires at BIND, before a byte moves."""

    def __init__(self, message: str, *, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(fields)


def refuse(code: str, message: str, fields: Sequence[str] = ()) -> EncodingRefusal:
    if code not in REFUSALS:
        raise AssertionError(f"{code} is not a declared encoding refusal")
    return EncodingRefusal(message, code=code, fields=fields)


# ------------------------------------------------------------------------ the registry

#: TensorFS dtype name -> torch dtype name, for the ROLE carriers. cr-005's `TORCH_DTYPES`
#: is the same fact for logical tensors; this module imports that one rather than keeping
#: a second copy — see `torch_dtype()` below.


def aliases() -> dict[str, str]:
    """spec digest -> platform alias, read from the COMPILED registry.

    The digest is identity and the alias is display, so this direction is the safe one:
    a provider registers against a digest it looked up by alias at import, and a registry
    change that moves a digest breaks selection loudly instead of decoding wrong bytes.
    """
    import tensorfs

    return {digest: alias for alias, digest in tensorfs.seed_digests()}


def digests_of(alias: str) -> tuple[str, ...]:
    """Every spec digest the platform registry files under one alias, in registry order."""
    import tensorfs

    return tuple(digest for name, digest in tensorfs.seed_digests() if name == alias)


# --------------------------------------------------------------------------- the parts


@dataclass(frozen=True, slots=True)
class RolePart:
    """One stored role of one encoded tensor, exactly as the header declares it."""

    role: str
    dtype: str
    """The TensorFS carrier dtype name (`f8_e4m3fn`, `f32`)."""
    shape: tuple[int, ...]
    nbytes: int

    @property
    def numel(self) -> int:
        n = 1
        for extent in self.shape:
            n *= extent
        return n


@dataclass(frozen=True, slots=True)
class Encoded:
    """The per-tensor encoding assignment: the digest, and the roles it stores."""

    encoding: str
    """`sha256:...` of the EncodingSpec object. THE selection key."""
    alias: str
    """Display only. Never resolved through."""
    parts: tuple[RolePart, ...]

    @property
    def roles(self) -> frozenset[str]:
        return frozenset(p.role for p in self.parts)

    @property
    def stored_nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts)

    def part(self, role: str) -> RolePart:
        for p in self.parts:
            if p.role == role:
                return p
        raise refuse(
            "role_mismatch",
            f"this tensor stores roles {sorted(self.roles)} and no {role!r}",
            [role],
        )


# ------------------------------------------------------------------------- the device


@dataclass(frozen=True, slots=True)
class DeviceFacts:
    """What a capability record is keyed against.

    `configuration` is the NVML power cap plus MIG profile (§3.2): two "same model" GPUs
    at different power caps are not the same device, and first-writer-wins must never let
    a power-starved pod mint the fleet's coefficients.
    """

    kind: str
    name: str
    sm: int
    driver: str
    configuration: str

    @property
    def predicate(self) -> str:
        """The device CLASS a capability record names: `cuda.sm89`, `cpu`."""
        return "cpu" if self.kind != "cuda" else f"cuda.sm{self.sm}"

    def satisfies(self, predicate: str) -> bool:
        """Does this device meet a provider's declared floor?

        `cuda.sm90+` admits sm90 and up; a bare `cuda.sm89` is exact; `any` admits all.
        The floor lives on the PROVIDER (a fact about an implementation), never on the
        spec (an immutable byte contract that hardware has no business inside).
        """
        if predicate == "any":
            return True
        if predicate == "cpu":
            return self.kind == "cpu"
        if not predicate.startswith("cuda.sm") or self.kind != "cuda":
            return False
        floor = predicate.removeprefix("cuda.sm")
        if floor.endswith("+"):
            return self.sm >= int(floor[:-1])
        return self.sm == int(floor)

    def tuple_key(self) -> str:
        """The probe cache key (§3.2): accelerator model, device configuration, driver."""
        return f"{self.kind}/{self.name}/{self.configuration}/{self.driver}"


def measure_device(torch: Any, index: int = 0) -> DeviceFacts:
    """Read the device once, as facts a probe result may be keyed by.

    Everything here is STABLE for the life of the machine: the power CAP (not the draw —
    a key that moves every read is not a cache key), the MIG profile, the driver's CUDA
    version. An unreadable knob is SPELLED into the key rather than defaulted, so a probe
    minted while NVML was mute cannot be reused as one minted while it answered.
    """
    kind = accel.host_backend_family()
    if not accel.present(torch, kind):
        return DeviceFacts("cpu", "cpu", 0, "", "default")
    identity = accel.device_identity(torch, kind, index)
    return DeviceFacts(
        kind=kind,
        name=str(identity["name"]),
        sm=int(identity["sm"]),
        driver=str(identity["driver"]),
        configuration=f"{_nvml('power.limit')}W/{_nvml('mig.mode.current')}",
    )


def _nvml(query: str) -> str:
    """One bounded NVML read. `N/A` and a failure are both SPELLED, never a number."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unreadable"
    return (out.splitlines() or ["unreadable"])[0].strip().replace(" ", "") or "unreadable"


# ------------------------------------------------------------------------ the providers


class Provider(Protocol):
    """One implementation that turns one encoding's stored roles into one destination.

    Providers present a PLAIN compute-dtype tensor at the leaf boundary and need no adapter
    awareness (cr-006's scope note): the SDK side-branch wraps the leaf, cr-010.
    """

    @property
    def name(self) -> str: ...

    @property
    def route(self) -> str:
        """`verbatim` or `decoded_float` — the delivery route (§3.2)."""
        ...

    @property
    def device_predicate(self) -> str: ...

    @property
    def required(self) -> frozenset[str]:
        """Roles this provider MUST be handed. A header short of one refuses."""
        ...

    @property
    def optional(self) -> frozenset[str]:
        """Roles it tolerates. A header carrying anything else refuses."""
        ...

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class Verbatim:
    """`plain/1`: the stored bytes ARE the logical bytes. There is nothing to decode.

    Entry zero of the registry is a real provider rather than an absent form, so the
    selection path has no "no encoding" branch to get wrong.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.plain.verbatim/1"

    @property
    def route(self) -> str:
        return "verbatim"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"value"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        raise AssertionError("a verbatim role is streamed straight into its destination")


@dataclass(frozen=True, slots=True)
class ScaledScalarDequant:
    """`fp8-scaled-scalar/1` -> float, weight-only: `value = data * weight_scale`.

    That equation is the spec's own, transcribed from the registry entry and from nothing
    else. `input_scale`, where the artifact carries it, is the ACTIVATION scale of a
    native fp8 matmul — it is deliberately NOT applied to the weight, because applying it
    would be a different (and wrong) tensor. On this route activations stay in the
    destination's float dtype, which is strictly more precise than the native path the
    scale was minted for; the two spec variants therefore decode identically and are still
    two providers, because a delivered role set that does not match refuses rather than
    being tolerated.

    Numerically this is exact up to the destination dtype: `data` is an e4m3 value and
    `weight_scale` an f32 scalar, so the product is representable and the only error is
    the one the QUANTIZER already committed. The decode adds no error of its own, which is
    what the probe's numerics self-check measures against an f64 reference.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.fp8-scaled-scalar.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        # A CAST, not a matmul: e4m3 -> float is available wherever torch has the dtype,
        # which is every CUDA card torch builds for and the CPU. The sm90 floor belongs to
        # the NATIVE route (`torch._scaled_mm`), which is a different provider and is not
        # built here — see NAMED_SEAMS.
        return "any"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "weight_scale"})

    @property
    def optional(self) -> frozenset[str]:
        # `input_scale` is the ACTIVATION scale of the native fp8 matmul. The two spec
        # digests under this alias differ by exactly its presence, and WHICH digest a
        # tensor cites is checked by TensorFS against the header's own closure (its fact,
        # its owner) — this provider serves both because the value law is identical.
        return frozenset({"input_scale"})

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        data = parts["data"]
        weight_scale = parts["weight_scale"]
        # The scale is a rank-0 f32. Multiplying in the DESTINATION dtype rather than in
        # f32-then-cast is deliberate: the destination is the compute contract, and an
        # intermediate f32 tensor of the full weight would double this route's scratch.
        destination.copy_(data.view(destination.shape).to(destination.dtype))
        destination.mul_(weight_scale.to(destination.dtype).reshape(()))


#: Encodings this runtime RECOGNISES but does not execute, with the device class that
#: would and the milestone that owns building it. Data, not machinery: there is no stub
#: provider object, no empty package and no dormant enum case — the refusal below reads
#: this table to name the alternative, which is the only consumer it has.
NAMED_SEAMS: dict[str, tuple[str, str]] = {
    "fp8-scaled-scalar/1": (
        "cuda.sm90+",
        "native fp8 tensor-core execution via torch._scaled_mm — cr-008c's native-provider "
        "leg; on sm89 the qualified route is decoded_float, which this runtime serves",
    ),
    "fp8-rowwise/1": (
        "cuda.sm90+",
        "per-output-row fp8; the decoded_float provider is a small generalisation of the "
        "scalar one and lands with its first consuming lane (cr-008c)",
    ),
    "mxfp8/1": (
        "cuda.sm100+",
        "MXFP8 native (tensor cores consume E4M3 elements + E8M0 block scales directly), "
        "or the sm89+ software-scaled arm that applies the 32-block E8M0 scales in the "
        "kernel prologue — both NEW kernel work. The route that needs no new kernel is "
        "DECODED FLOAT: element * 2^(scale-127) per 32-element block into the logical "
        "dtype, the same shape as this build's fp8 provider and not built because no lane "
        "selects mxfp8 (proto-001 measured it 1.06 dB BEHIND per-tensor fp8 for 3.1% more "
        "bytes, throughput ABSENT on SM90 silicon). What never happens is a silent "
        "per-channel reinterpretation of block scales",
    ),
    "nvfp4-w4a4/1": (
        "cuda.sm100+",
        "the v1 SVDQuant/W4A4 native stack, ported as providers with its layout contracts "
        "already registered as encoding rules; sequences with its consuming lane's tier",
    ),
    "svdq-microscale/1": (
        "cuda.sm100+",
        "svdq micro-scale, same port as nvfp4-w4a4/1; its wscale geometry is PROVISIONAL "
        "pending producer-mined confirmation (tfs-008)",
    ),
}


def launch_providers() -> dict[str, Any]:
    """The LAUNCH TIER provider set, keyed by spec digest.

    Read from the compiled registry at call time so a digest is never transcribed. The set
    is deliberately small: `plain/1` plus the encoding proto-001 SELECTED. Everything else
    the registry knows is in `NAMED_SEAMS` and refuses naming its device class.
    """
    out: dict[str, Any] = {}
    for digest in digests_of("plain/1"):
        out[digest] = Verbatim(digest)
    # BOTH role sets one real fp8_scaled artifact carries. They share one value law, so
    # they share one implementation; which digest a given tensor may cite is TensorFS's
    # fact, enforced when the header is parsed against its own encoding closure.
    for digest in digests_of("fp8-scaled-scalar/1"):
        out[digest] = ScaledScalarDequant(digest)
    return out


# ------------------------------------------------------------------- capability records


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """The tfs-013 key, kept whole: an (encoding, implementation, device) qualification.

    A record cannot exist without `evidence_digest`, which is the digest of the probe
    observation that minted it — so "qualified" always means "something ran and was
    measured", never "someone believed".
    """

    encoding_spec_id: str
    implementation_digest: str
    runtime_release: str
    device_predicate: str
    output_dtype: str
    delivery_route: str
    evidence_digest: str


class Capabilities:
    """The fail-closed admission table. No default branch, no "unknown means allowed"."""

    def __init__(self, records: Sequence[CapabilityRecord] = ()) -> None:
        self.records = list(records)

    def add(self, record: CapabilityRecord) -> None:
        self.records.append(record)

    def admit(
        self, encoding: str, device: DeviceFacts, output_dtype: str, alias: str
    ) -> CapabilityRecord:
        for record in self.records:
            if (
                record.encoding_spec_id == encoding
                and record.output_dtype == output_dtype
                and device.satisfies(record.device_predicate)
            ):
                return record
        elsewhere = sorted(
            {r.device_predicate for r in self.records if r.encoding_spec_id == encoding}
        )
        seam = NAMED_SEAMS.get(alias)
        alternative = f" The route that WOULD serve it is {seam[0]}: {seam[1]}." if seam else ""
        raise refuse(
            "encoding_unqualified",
            f"{alias} ({encoding[:23]}…) has no qualified provider on "
            f"{device_line(device)} for output dtype {output_dtype}. Qualified device "
            f"predicates for this encoding: {elsewhere or 'none'}.{alternative} An "
            "unrecognised device is the ABSENCE of evidence, never permission.",
            [encoding],
        )


def device_line(device: DeviceFacts) -> str:
    """The device FACT, spelled the same way in every capability refusal. A refusal that
    says only "unsupported" sends a reader to the wrong question."""
    return (
        f"{device.predicate} ({device.name}, sm{device.sm}, driver cuda {device.driver}, "
        f"config {device.configuration})"
    )


# ------------------------------------------------------------------------- the selector


@dataclass(frozen=True, slots=True)
class Selection:
    """What the fill plane needs to move and decode one tensor."""

    provider: Any
    record: CapabilityRecord
    encoded: Encoded


class Selector:
    """Per-tensor provider selection, by digest, fail-closed, priced at O(1) per key."""

    def __init__(
        self, providers: Mapping[str, Any], capabilities: Capabilities, device: DeviceFacts
    ) -> None:
        self.providers = dict(providers)
        self.capabilities = capabilities
        self.device = device
        self.aliases = aliases()

    def select(self, key: str, encoded: Encoded, output_dtype: str) -> Selection:
        provider = self.providers.get(encoded.encoding)
        if provider is None:
            alias = self.aliases.get(encoded.encoding)
            if alias is None:
                raise refuse(
                    "unknown_encoding",
                    f"{key}: the checkpoint cites encoding {encoded.encoding} and this "
                    "runtime's registry does not contain that spec at all — the pinned "
                    "registry plus reader set IS this runtime's encoding capability, and "
                    "an unregistered encoding refuses before generation rather than being "
                    "guessed from the key or the carrier dtypes",
                    [key],
                )
            seam = NAMED_SEAMS.get(alias)
            if seam is None:
                raise refuse(
                    "unknown_encoding",
                    f"{key}: {alias} is a REGISTERED encoding with no provider in this "
                    "runtime build and no named seam — the registry knows the bytes and "
                    "nothing here executes them",
                    [key],
                )
            # A registered encoding whose provider is a NAMED, UNBUILT seam is a DEVICE
            # CAPABILITY statement, not an unknown one, so it refuses in the capability
            # vocabulary and names the card it is standing on. The alternative reading —
            # "unknown encoding" — would send a reader looking for a registry problem
            # that does not exist.
            raise refuse(
                "encoding_unqualified",
                f"{key}: {alias} is registered and this runtime has no provider for it on "
                f"{device_line(self.device)}. The route that WOULD serve it is {seam[0]}: "
                f"{seam[1]}. Named and not built — never silently served by a route it was "
                "not qualified for.",
                [key],
            )
        missing = sorted(provider.required - encoded.roles)
        surplus = sorted(encoded.roles - provider.required - provider.optional)
        if missing or surplus:
            raise refuse(
                "role_mismatch",
                f"{key}: the header stores roles {sorted(encoded.roles)}; the provider "
                f"selected by this tensor's own digest requires {sorted(provider.required)} "
                f"and tolerates {sorted(provider.optional)} — missing {missing}, "
                f"unhandled {surplus}. A role a decoder cannot account for is a different "
                "physical contract, never a tolerated variant",
                [key],
            )
        record = self.capabilities.admit(
            encoded.encoding, self.device, output_dtype, self.aliases.get(encoded.encoding, "?")
        )
        return Selection(provider, record, encoded)


# ------------------------------------------------------------------------------ digests


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """`sha256:` over sorted-key compact JSON — the one spelling this module uses."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def implementation_digest(provider: Any) -> str:
    """The provider's identity: its name, route, device floor and exact role set."""
    return canonical_digest(
        {
            "name": provider.name,
            "route": provider.route,
            "device_predicate": provider.device_predicate,
            "required": sorted(provider.required),
            "optional": sorted(provider.optional),
        }
    )
