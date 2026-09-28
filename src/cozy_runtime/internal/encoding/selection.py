"""cr-006 seam 3 — SELECTION: device identity, capability records, and the launch table.

Seams one and two build providers. This one decides which of them may run, on THIS machine,
for THIS tensor — and, since #549.1, it stops deciding one line earlier than it used to.

**What changed.** `Capabilities.admit` used to be a second, parallel authority: it ranked
providers per tensor under the deployment's objective, and refused `objective_unranked` the
moment the objective was `latency` and two providers qualified — which is the default
objective, so the default path refused as soon as a second provider existed. Meanwhile
`delivery.py` picked ONE artifact-wide rung from a table that DOES hold latency
measurements. Two choosers, neither able to see the other's numbers.

This module now offers CANDIDATES and a reason, and `resolution.py` makes ONE joint
decision over whole plans. That is the whole of #549.3: latency is measurable where the
measurement lives (per resolved plan, in the fact store), footprint and
`kernel_numeric_error` are measurable per provider (in the records here), and nothing has
to refuse for want of an axis.

**The fidelity axis is now `kernel_numeric_error`** (#549.11). It measures a route's own
GEMM against an f64 reference over the exactly-dequantized weights. It is a KERNEL
diagnostic and it is not model quality: quality is a paired video+audio eval on real
outputs, owned by the eval plane, and a Frobenius norm may never stand in for one.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from cozy_runtime.internal import accel, canonical
from cozy_runtime.internal.encoding.formats import (
    MX_BLOCK,
    Encoded,
    MicroScaledDequant,
    Provider,
    RowwiseDequant,
    ScaledScalarDequant,
    Verbatim,
    aliases,
    refuse,
    registry_digests,
)
from cozy_runtime.internal.encoding.leaves import MicroScaledNativeLeaf, RowwiseNativeLeaf

# `torch: Any` is the torch module, which the check venv does not carry.

# ------------------------------------------------------------------------- the device


@dataclass(frozen=True, slots=True)
class DeviceFacts:
    """What a capability record is keyed against — STRENGTHENED by #549.9.

    Every field is stable for the life of the machine, which is what lets a probe result be
    keyed by it. What the pre-#549 key was MISSING, and why each one matters:

    * `driver` was `torch.version.cuda`, **which is not the driver** — it is the CUDA
      runtime torch was built against. Two pods with identical torch wheels and different
      NVIDIA drivers produced the same key, so a record minted under a driver with a
      miscompiled kernel was reused under one without it. The real driver version now comes
      from NVML and the CUDA runtime is carried BESIDE it, spelled as what it is.
    * `index` and the NVML `configuration` behind it. `_nvml` queried **GPU zero** whatever
      device was selected, so a multi-GPU pod minted every record against card 0's power cap
      and MIG profile. The query is now `-i <index>`.
    * `os_arch`. The same card under a different libc/kernel/arch is a different measurement
      subject, and nothing recorded it.

    `configuration` is the NVML power cap plus MIG profile (§3.2): two "same model" GPUs at
    different power caps are not the same device, and first-writer-wins must never let a
    power-starved pod mint the fleet's coefficients.
    """

    kind: str
    name: str
    sm: int
    driver: str
    """The REAL NVIDIA driver version, from NVML. Never `torch.version.cuda`."""
    configuration: str
    index: int = 0
    """The device this record was measured on. NVML is queried with `-i <index>`."""
    cuda_runtime: str = ""
    """The CUDA runtime torch was built against — a BUILD fact, carried beside the driver
    rather than in place of it. This is the field that used to be called `driver`."""
    os_arch: str = ""
    """`Linux/x86_64`. The measurement subject includes its operating system."""

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
        """The measured device identity carried by qualification evidence."""
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
    """Read the SELECTED device once, as facts a probe result may be keyed by.

    `index` is honoured all the way down (#549.9): torch is asked about that device and NVML
    is queried with `-i index`. Passing 0 and reading card 0 while filling card 3 is the
    defect this signature always looked like it did not have.

    An unreadable knob is SPELLED into the key rather than defaulted, so a probe minted
    while NVML was mute cannot be reused as one minted while it answered.
    """
    kind = accel.host_backend_family()
    os_arch = f"{platform.system()}/{platform.machine()}"
    if not accel.present(torch, kind):
        return DeviceFacts(
            "cpu", "cpu", 0, "", "default", index=0, cuda_runtime="", os_arch=os_arch
        )
    identity = accel.device_identity(torch, kind, index)
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
    """One bounded NVML read OF THE SELECTED DEVICE. `N/A` and a failure are both SPELLED.

    `-i {index}` is #549.9's fix and it is one flag: without it every record on a multi-GPU
    host was keyed by card ZERO's power cap and MIG profile, however many cards the fill
    plane was about to use.
    """
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
    """WHAT WAS RUNNING when a measurement was taken (#549.9).

    A capability record is a claim about an implementation on a device. Both halves have an
    identity and only the device half had one: two runtime builds with different arithmetic
    in `encoding/` produced records that were indistinguishable and therefore interchangeable.

    `source_digest` is over this runtime's own encoding/fill/probe sources, so an edit to a
    decode or a leaf forward invalidates the records that measured the old one. It is
    computed from the installed FILES rather than from a version string, because a version
    string is a promise and a digest is a fact.
    """

    release: str
    source_digest: str
    torch_build: str

    def document(self) -> dict[str, str]:
        return {
            "release": self.release,
            "source_digest": self.source_digest,
            "torch_build": self.torch_build,
        }


#: The modules whose bytes decide what a capability record MEANS. Anything that changes a
#: decode, a leaf forward, a swizzle, an oracle or the admission arithmetic is here; the CLI,
#: the worker and the author surface are not, because they cannot change a measurement.
IDENTITY_SOURCES = (
    "internal/encoding/formats.py",
    "internal/encoding/leaves.py",
    "internal/encoding/selection.py",
    "internal/instruments.py",
    "internal/probe.py",
    "internal/resolution.py",
)


def runtime_source_digest() -> str:
    """A digest over `IDENTITY_SOURCES` as installed. Missing files are SPELLED, not skipped.

    A source file that cannot be read is recorded as absent rather than silently omitted:
    two builds that differ by a deleted module must not share a digest.
    """
    root = Path(__file__).resolve().parent.parent.parent
    body: dict[str, str] = {}
    for rel in IDENTITY_SOURCES:
        path = root / rel
        try:
            body[rel] = canonical.digest({"bytes": path.read_bytes().hex()})
        except OSError:
            body[rel] = "absent"
    return canonical.digest(body)


def measure_runtime(torch: Any, release: str) -> RuntimeIdentity:
    return RuntimeIdentity(
        release=release,
        source_digest=runtime_source_digest(),
        torch_build=str(getattr(torch, "__version__", "unknown")),
    )


# --------------------------------------------------------------------- geometry classes


def geometry_class(shape: Sequence[int]) -> str:
    """The COARSE geometry a measurement was taken at, recorded on every record (#549.9).

    It is EVIDENCE, not a filter. A record says "this was measured on whole-32-block rank-2
    geometry", and `Provider.supports` is what actually gates a tensor — filtering admission
    on this string as well would refuse the decode floor for a ragged tensor, which is the
    one route that handles one.
    """
    if len(shape) != 2:
        return f"rank{len(shape)}"
    return "rank2.block32" if int(shape[1]) % MX_BLOCK == 0 else "rank2.ragged32"


# ------------------------------------------------------------------- capability records


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """The tfs-013 key, kept whole and widened by #549.9: an (encoding, implementation,
    runtime, device, dtype, geometry) qualification.

    A record cannot exist without `evidence_digest`, which is the digest of the probe
    observation that minted it — so "qualified" always means "something ran and was
    measured", never "someone believed".

    The two MEASURED axes below are what turned this from a membership test into an
    ORDERING (#517e). They are stored on the record and not derived at admission because
    they are properties of a probe run: a number the table carries is a number something
    produced, and one nothing produced is spelled -1.0 rather than defaulted to a value
    that would win or lose a comparison it was never entitled to enter.
    """

    encoding_spec_id: str
    implementation_digest: str
    runtime_release: str
    device_predicate: str
    output_dtype: str
    delivery_route: str
    evidence_digest: str
    resident_bits_per_element: float = -1.0
    """FOOTPRINT axis. The bits this route leaves resident per logical element, measured on
    the probe's own tensor: the destination dtype's width on a decode route (the weights are
    float either way), the role bytes on an encoded-GEMM one. -1.0 means NOT MEASURED."""
    kernel_numeric_error: float = -1.0
    """The KERNEL DIAGNOSTIC axis, renamed from `route_deviation_rel`/"fidelity" (#549.11).
    Relative Frobenius of the ROUTE's own GEMM output against an f64 reference over the
    exactly-dequantized weights, on identical activations — the one quantity comparable
    across a decode route and an encoded one, because it prices the whole route rather than
    one of its halves. -1.0 means NOT MEASURED.

    It is NOT model quality and may never be reported as such: quality is a paired
    video+audio eval on real outputs and belongs to the eval plane (#549.6)."""
    leaf_speedup_x: float = -1.0
    """The LEAF SPEED evidence, encoded-GEMM routes only: the float GEMM's wall over the
    leaf's forward at the probe shape, activation quantization inside the timed region
    (`probe._leaf_bench`). Above 1.0 the leaf wins; below it the leaf LOSES to decoding and
    the route is a memory play. -1.0 means NOT MEASURED, which every decode route is.

    It is evidence at ONE shape, not a whole-plan wall, so nothing ranks on it (see
    `resolution.score`); it exists so that a leaf that measured slower is READABLE on every
    record, plan and boot line that serves it, rather than banked and never read."""
    runtime_source_digest: str = ""
    torch_build: str = ""
    os_arch: str = ""
    device_index: int = 0
    device_driver: str = ""
    cuda_runtime: str = ""
    geometry_class: str = ""
    implementation_revision: str = ""

    def axis(self, objective: str) -> float:
        """The ranking key for one objective. Lower is better; -1.0 means UNRANKABLE."""
        if objective == "footprint":
            return self.resident_bits_per_element
        if objective == "kernel_numeric_error":
            return self.kernel_numeric_error
        return -1.0

    def document(self) -> dict[str, str | int | float]:
        return {
            "encoding_spec_id": self.encoding_spec_id,
            "implementation_digest": self.implementation_digest,
            "implementation_revision": self.implementation_revision,
            "runtime_release": self.runtime_release,
            "runtime_source_digest": self.runtime_source_digest,
            "torch_build": self.torch_build,
            "os_arch": self.os_arch,
            "device_predicate": self.device_predicate,
            "device_index": self.device_index,
            "device_driver": self.device_driver,
            "cuda_runtime": self.cuda_runtime,
            "output_dtype": self.output_dtype,
            "geometry_class": self.geometry_class,
            "delivery_route": self.delivery_route,
            "evidence_digest": self.evidence_digest,
            "resident_bits_per_element": self.resident_bits_per_element,
            "kernel_numeric_error": self.kernel_numeric_error,
            "leaf_speedup_x": self.leaf_speedup_x,
        }


#: Which MEASURED record field each PER-TENSOR objective ranks implementations on.
#:
#: `latency` is deliberately absent AND ITS ABSENCE IS NO LONGER A REFUSAL (#549.3). Nothing
#: on a capability record measures a quantity comparable between "a decode paid once at
#: fill" and "a GEMM paid every step" — but the PLAN FACT store does, per resolved plan, and
#: `resolution.py` ranks whole plans there. What used to happen instead was that the default
#: objective refused before the layer holding those measurements ever ran.
OBJECTIVE_AXIS: dict[str, str] = {
    "footprint": "resident_bits_per_element",
    "kernel_numeric_error": "kernel_numeric_error",
}


class Capabilities:
    """The fail-closed admission table. No default branch, no "unknown means allowed"."""

    def __init__(self, records: Sequence[CapabilityRecord] = ()) -> None:
        self.records = list(records)

    def add(self, record: CapabilityRecord) -> None:
        self.records.append(record)

    def qualified(
        self,
        encoding: str,
        device: DeviceFacts,
        output_dtype: str,
        among: Mapping[str, Provider],
    ) -> list[CapabilityRecord]:
        """Every record that admits one of `among` on this device, for this dtype.

        `among` is implementation digest -> provider, for exactly the candidates whose
        declared role set the header satisfies AND whose geometry preflight passed. A record
        for an implementation this build does not carry admits nothing, and neither does a
        provider whose declared device floor this card misses — the two filters are separate
        because they answer different questions ("was this measured here" and "does this
        implementation claim this card"), and collapsing them would let a stale record select
        a provider that is gone.
        """
        return [
            record
            for record in self.records
            if record.encoding_spec_id == encoding
            and record.output_dtype == output_dtype
            and device.satisfies(record.device_predicate)
            and record.implementation_digest in among
            and device.satisfies(among[record.implementation_digest].device_predicate)
        ]

    def rank(
        self, records: Sequence[CapabilityRecord], among: Mapping[str, Provider], objective: str
    ) -> list[CapabilityRecord]:
        """Order candidates best-first under one objective. NEVER refuses (#549.3).

        An objective with a measured axis orders on it, lower better, the implementation NAME
        breaking an exact tie so the answer is a function of the measurements and not of dict
        iteration order. An objective with NO per-record axis — `latency` — orders by name
        alone and says nothing: the plan level ranks routes, and a stable order here just
        keeps this function total.

        A candidate whose axis is UNMEASURED (-1.0) sorts LAST rather than first: -1.0 is
        smaller than every real measurement, and "nobody measured it" must never win a
        comparison it was not entitled to enter.
        """
        axis = OBJECTIVE_AXIS.get(objective)

        def key(record: CapabilityRecord) -> tuple[int, float, str]:
            name = among[record.implementation_digest].name
            if axis is None:
                return (0, 0.0, name)
            value = record.axis(objective)
            return (1, 0.0, name) if value < 0.0 else (0, value, name)

        return sorted(records, key=key)

    def admit(
        self,
        encoding: str,
        device: DeviceFacts,
        output_dtype: str,
        alias: str,
        *,
        objective: str,
        among: Mapping[str, Provider],
    ) -> CapabilityRecord:
        """The best qualified record for this (encoding, device, dtype). One answer.

        Kept as the single-answer door for callers that legitimately want one — the fill
        plane, once a plan has already been resolved. Plan resolution uses `qualified` plus
        `rank` so it can build alternative plans and price them WHOLE.
        """
        records = self.qualified(encoding, device, output_dtype, among)
        if not records:
            raise refuse(
                "encoding_unqualified",
                self._why(encoding, device, output_dtype, alias, among),
                [encoding],
            )
        return self.rank(records, among, objective)[0]

    def _why(
        self,
        encoding: str,
        device: DeviceFacts,
        output_dtype: str,
        alias: str,
        among: Mapping[str, Provider],
    ) -> str:
        elsewhere = sorted(
            {r.device_predicate for r in self.records if r.encoding_spec_id == encoding}
        )
        seam = NAMED_SEAMS.get(alias)
        alternative = f" The route that WOULD serve it is {seam[0]}: {seam[1]}." if seam else ""
        # WHICH implementations exist and what each one's floor is. "No qualified provider"
        # is the same words whether nothing was built or nothing was measured, and those are
        # different problems with different remedies.
        built = " | ".join(
            f"{p.name} [{p.route}] floor {p.device_predicate}"
            f"{'' if device.satisfies(p.device_predicate) else ' — this card misses that floor'}"
            for p in sorted(among.values(), key=lambda p: p.name)
        )
        return (
            f"{alias} ({encoding[:23]}…) has no qualified provider on "
            f"{device_line(device)} for output dtype {output_dtype}. Qualified device "
            f"predicates for this encoding: {elsewhere or 'none'}. Implementations this "
            f"build carries: {built or 'none'}.{alternative} An unrecognised device is the "
            "ABSENCE of evidence, never permission."
        )


# --------------------------------------------------------------------------- the seams

#: Encodings this runtime RECOGNISES but does not execute, with the device class that
#: would and the milestone that owns building it. Data, not machinery: there is no stub
#: provider object, no empty package and no dormant enum case — the refusal reads this
#: table to name the alternative, which is the only consumer it has.
NAMED_SEAMS: dict[str, tuple[str, str]] = {
    "fp8-scaled-scalar/1": (
        "cuda.sm90+",
        "encoded-GEMM fp8 tensor-core execution via torch._scaled_mm — cr-008c's "
        "encoded-provider leg; on sm89 the qualified route is decoded_float, which this "
        "runtime serves",
    ),
    # `fp8-rowwise/1` and `mxfp8/1` LEFT THIS TABLE when their encoded-GEMM leaves were
    # built. A seam entry means "this runtime recognises the bytes and executes none of
    # them"; both now have TWO providers each, a decode floor and an encoded leaf, and the
    # honest answer for a device that serves neither is the candidate list the refusal
    # prints. sm100 (B200) stays UNMEASURED: no card was rented, so no record and no floor.
    "nvfp4-w4a4/1": (
        "cuda.sm100+",
        "the v1 SVDQuant/W4A4 native stack, ported as providers with its layout contracts "
        "already registered as encoding rules; sequences with its consuming lane's tier. "
        "PAUSED behind #549's plan-identity gates — no new provider lands until they do",
    ),
    "svdq-microscale/1": (
        "cuda.sm100+",
        "svdq micro-scale, same port as nvfp4-w4a4/1; its wscale geometry is PROVISIONAL "
        "pending producer-mined confirmation (tfs-008)",
    ),
}


# ------------------------------------------------------------------------ the launch table


class SpecUnreviewed(Exception):
    """A provider in this build declares a spec digest the compiled registry does not carry.

    Loud at LAUNCH, by construction (#549.4). The alternative — silently dropping the
    provider — is how a registry bump turns into an `encoding_unqualified` on a production
    artifact three deploys later, with nothing pointing at the bump.
    """


def launch_providers() -> dict[str, tuple[Provider, ...]]:
    """The LAUNCH TIER provider set: spec digest -> the ORDERED CANDIDATES for that digest.

    **Built from `reviewed_specs`, not from aliases** (#549.4). Each provider names the exact
    digests it was reviewed against; this function inverts that into digest -> providers and
    checks every declared digest against the compiled registry. Two consequences, both the
    point:

    * a NEW digest the registry files under an existing alias acquires NO PROVIDER. Under
      the alias-keyed spelling it inherited the sibling's decoder — a semantically different
      `mxfp8/1` would have been decoded by `cozy.mxfp8.dequant/1` on the strength of its
      display name.
    * a digest that LEAVES the registry raises `SpecUnreviewed` here, at launch, naming the
      provider and the digest, instead of quietly unregistering a route.

    The order within a digest is REGISTRATION order and it decides NOTHING. Selection among
    candidates is `resolution.py`'s joint plan decision over records that exist only where a
    probe measured.
    """
    known = registry_digests()
    out: dict[str, list[Provider]] = {}
    declared: list[tuple[str, str]] = []
    builds: tuple[Callable[[str], Provider], ...] = (
        Verbatim,
        ScaledScalarDequant,
        RowwiseDequant,
        RowwiseNativeLeaf,
        MicroScaledDequant,
        MicroScaledNativeLeaf,
    )
    for build in builds:
        # The provider is instantiated once per digest it reviewed, which is what makes
        # `provider.encoding` a fact rather than a lookup.
        probe = build("")
        for digest in sorted(probe.reviewed_specs):
            declared.append((probe.name, digest))
            if digest not in known:
                continue
            out.setdefault(digest, []).append(build(digest))
    missing = [(name, digest) for name, digest in declared if digest not in known]
    if missing:
        raise SpecUnreviewed(
            "this build declares spec digests the compiled registry does not carry: "
            + " | ".join(f"{name} -> {digest}" for name, digest in sorted(missing))
            + ". A provider's reviewed digest set is its claim about which physical byte "
            "contracts it was read against; a registry that moved one has changed what "
            "those bytes mean, and inferring the replacement by ALIAS is exactly the "
            "substitution #549.4 forbids"
        )
    return {digest: tuple(candidates) for digest, candidates in out.items()}


# ------------------------------------------------------------------------------ digests


def implementation_digest(provider: Provider) -> str:
    """The provider's identity: name, route, device floor, oracle, role set, revision, specs.

    `suite` joins the key because it is part of what a record MEANS: two implementations
    checked by different oracles are not interchangeable, and a record whose evidence came
    from one must not admit the other.

    `code_revision` and `reviewed_specs` joined it with #549.9/#549.4: an implementation
    whose arithmetic changed, or whose reviewed contract set changed, is a different
    implementation, and records that measured the old one must not admit the new one.
    """
    return canonical.digest(
        {
            "name": provider.name,
            "route": provider.route,
            "device_predicate": provider.device_predicate,
            "suite": provider.suite,
            "required": sorted(provider.required),
            "optional": sorted(provider.optional),
            "code_revision": provider.code_revision,
            "reviewed_specs": sorted(provider.reviewed_specs),
        }
    )


# ------------------------------------------------------------------------- the selector


@dataclass(frozen=True, slots=True)
class Selection:
    """What the fill plane needs to move and decode one tensor."""

    provider: Provider
    record: CapabilityRecord
    encoded: Encoded
    reason: str = ""
    """WHY this provider, in one sentence, carried into the resolved plan. A plan that says
    `decoded_float` where a leaf exists has to be able to say why — #549.5's ragged-K case is
    exactly that, and a plan whose label is honest but whose reason is missing is half a
    fix."""


@dataclass(frozen=True, slots=True)
class TensorCandidates:
    """Every route this tensor CAN take on this device, plus what refused the others.

    This is the object #549.1 needed and the old `select`-only surface could not provide:
    plan resolution has to build the encoded plan AND the decode-floor plan before pricing
    either, so it needs the alternatives rather than one winner.
    """

    key: str
    encoded: Encoded
    output_dtype: str
    by_route: Mapping[str, Selection] = field(default_factory=dict)
    """route -> its best qualified implementation, already ranked under the objective."""
    geometry_refused: tuple[tuple[str, str], ...] = ()
    """(provider name, why) for candidates this tensor's own SHAPE ruled out (#549.5)."""
    unqualified: str = ""
    """The capability refusal, verbatim, if NO route qualified."""
    routes_offered: frozenset[str] = frozenset()
    """Every route this tensor's BYTES and SHAPE admit on some device, qualified here or
    not. `by_route` minus this is what this card did not qualify."""


class Selector:
    """Per-tensor candidate resolution: by digest, fail-closed, O(1) per key.

    `objective` is the DEPLOYMENT's stated axis, carried down from the resolved plan. It is a
    constructor argument rather than a per-call one because delivery is decided ONCE per
    generation: a per-tensor objective would let one component's leaves and another's
    disagree about what this deployment is for.
    """

    def __init__(
        self,
        providers: Mapping[str, tuple[Provider, ...]],
        capabilities: Capabilities,
        device: DeviceFacts,
        objective: str = "kernel_numeric_error",
    ) -> None:
        #: digest -> ORDERED CANDIDATES.
        self.providers = providers
        self.capabilities = capabilities
        self.device = device
        self.objective = objective
        self.aliases = aliases()

    def candidates(
        self, key: str, encoded: Encoded, output_dtype: str, shape: Sequence[int]
    ) -> TensorCandidates:
        """Every servable route for one tensor, with the geometry preflight already applied.

        Order of the four filters, and why it is this order: REGISTRY (is this digest known
        at all), ROLES (is this provider a provider OF THESE BYTES), GEOMETRY (can it consume
        THIS SHAPE — #549.5, pure, before any allocation), then CAPABILITY (was it measured
        here). Each answers a different question and each has a different remedy, so
        collapsing any two would print the wrong sentence.
        """
        providers = self.providers.get(encoded.encoding)
        if not providers:
            self._refuse_unknown(key, encoded)
        among: dict[str, Provider] = {}
        mismatched: list[tuple[Provider, list[str], list[str]]] = []
        geometry: list[tuple[str, str]] = []
        for candidate in providers:
            missing = sorted(candidate.required - encoded.roles)
            surplus = sorted(encoded.roles - candidate.required - candidate.optional)
            if missing or surplus:
                mismatched.append((candidate, missing, surplus))
                continue
            why = candidate.supports(shape)
            if why:
                geometry.append((candidate.name, why))
                continue
            among[implementation_digest(candidate)] = candidate
        if not among and mismatched and not geometry:
            raise refuse(
                "role_mismatch",
                f"{key}: the header stores roles {sorted(encoded.roles)} and no provider "
                f"reviewed for this tensor's own digest accounts for them — "
                + " | ".join(
                    f"{c.name} requires {sorted(c.required)}, tolerates {sorted(c.optional)} "
                    f"(missing {m}, unhandled {s})"
                    for c, m, s in mismatched
                )
                + ". A role a decoder cannot account for is a different physical contract, "
                "never a tolerated variant",
                [key],
            )
        if not among:
            raise refuse(
                "geometry_unsupported",
                f"{key}: every implementation of {self.aliases.get(encoded.encoding, '?')} "
                f"refuses this tensor's geometry {tuple(shape)} — "
                + " | ".join(f"{name}: {why}" for name, why in geometry)
                + ". The device is not the problem and neither are the bytes: nothing here "
                "consumes this shape, and a reshape would be a different model",
                [key],
            )
        records = self.capabilities.qualified(encoded.encoding, self.device, output_dtype, among)
        offered = frozenset(provider.route for provider in among.values())
        if not records:
            return TensorCandidates(
                key,
                encoded,
                output_dtype,
                {},
                tuple(geometry),
                self.capabilities._why(
                    encoded.encoding,
                    self.device,
                    output_dtype,
                    self.aliases.get(encoded.encoding, "?"),
                    among,
                ),
                offered,
            )
        best: dict[str, Selection] = {}
        for record in self.capabilities.rank(records, among, self.objective):
            provider = among[record.implementation_digest]
            if provider.route in best:
                continue
            reason = self._reason(provider, record, among, geometry)
            best[provider.route] = Selection(provider, record, encoded, reason)
        return TensorCandidates(key, encoded, output_dtype, best, tuple(geometry), "", offered)

    def _reason(
        self,
        provider: Provider,
        record: CapabilityRecord,
        among: Mapping[str, Provider],
        geometry: Sequence[tuple[str, str]],
    ) -> str:
        peers = sorted(p.name for p in among.values() if p.name != provider.name)
        why = (
            f"{provider.name} [{provider.route}], measured on {record.device_predicate}: "
            f"{record.resident_bits_per_element:.3f} resident b/elt, "
            f"kernel_numeric_error {record.kernel_numeric_error:.6f}"
        )
        if geometry:
            why += "; geometry ruled out " + ", ".join(f"{n} ({w[:80]}…)" for n, w in geometry)
        elif peers:
            why += f"; also qualified: {', '.join(peers)}"
        return why

    def _refuse_unknown(self, key: str, encoded: Encoded) -> NoReturn:
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
            # #549.4's ARM. The alias is registered, this build's providers reviewed OTHER
            # digests under it, and nothing here has read THESE bytes. Under the alias-keyed
            # launch table this tensor would have been handed the sibling digest's decoder.
            raise refuse(
                "spec_unreviewed",
                f"{key}: {alias} is a REGISTERED alias and this runtime reviewed no provider "
                f"against spec {encoded.encoding[:30]}… in particular. An alias is a display "
                "label the registry may file several physically different specs under — "
                "`nvfp4-w4a4/1` files four — so a provider claims DIGESTS, and a digest it "
                "never read acquires no decoder from a name it happens to share",
                [key],
            )
        # A registered encoding whose provider is a NAMED, UNBUILT seam is a DEVICE
        # CAPABILITY statement, not an unknown one.
        raise refuse(
            "encoding_unqualified",
            f"{key}: {alias} is registered and this runtime has no provider for it on "
            f"{device_line(self.device)}. The route that WOULD serve it is {seam[0]}: "
            f"{seam[1]}. Named and not built — never silently served by a route it was "
            "not qualified for.",
            [key],
        )

    def select(
        self, key: str, encoded: Encoded, output_dtype: str, shape: Sequence[int] = ()
    ) -> Selection:
        """ONE answer, for a caller that has already had a plan resolved for it.

        `shape` defaults to the encoded `data` role's own shape when a caller has not got the
        logical row to hand — every geometry law in this build is about the payload's two
        outer dimensions, and the data role carries them.
        """
        if not shape:
            data = next((p for p in encoded.parts if p.role in ("data", "value")), None)
            shape = data.shape if data else ()
        found = self.candidates(key, encoded, output_dtype, shape)
        if not found.by_route:
            raise refuse("encoding_unqualified", found.unqualified or f"{key}: unqualified", [key])
        order = self.capabilities.rank(
            [s.record for s in found.by_route.values()],
            {implementation_digest(s.provider): s.provider for s in found.by_route.values()},
            self.objective,
        )
        first = order[0]
        for selection in found.by_route.values():
            if selection.record is first:
                return selection
        return next(iter(found.by_route.values()))
