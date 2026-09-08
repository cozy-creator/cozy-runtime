"""The runtime picks the fastest attention kernel this device and this image support (cr-124).

ATTENTION IS WEIGHTLESS. No checkpoint stores an attention kernel, so no checkpoint can name
one, and binding a kernel to a lane pointer was a category error (decision #707 as amended
2026-09-08). The only genuine weight-to-kernel coupling is in the GEMMs — fp8-rowwise weights need a
rowwise-fp8 GEMM, mxfp8 needs block scales — and that coupling is already expressed by the
checkpoint's own weight encoding plus `resolution.py`'s encoded-leaf routing. It needs no
contract, and neither does this.

THE RULE IS STATIC AND VERSIONED. `CANDIDATES` is an ordered table whose order is MEASURED on
the device it claims, never asserted from a vendor's benchmark. Selection walks it and takes
the first entry this device admits, this image ships, and the attention site's own dtype and
head dimension fit. Nothing is measured at run time and nothing falls back per call, so what
ran is a pure function of (device, image digest, checkpoint) — which is why one BOOT line is
the whole record and no per-attempt attestation is kept.

A NUMERICS-CHANGING KERNEL IS NOT AUTOMATICALLY FASTER, and speed is not the only thing a
rank asserts. 8-bit attention has now been measured (h3a-013, H100 SXM, H3's own geometry):
SageAttention2 sm90 is 1.12x rank 1, and flash-attn3's OWN fp8-e4m3 path — the same baked `.so`
this image already ships — is 1.31x. Neither is ranked, and the reason is not the speed:

  An EXACT kernel needs no quality evidence to be promoted. flash-attn3 became rank 1 over
  cuDNN on a speed measurement alone because the two agree to max abs 1.2e-7, FOUR orders
  BELOW bf16's own 2.3e-3 representation floor — the same computation, reordered. The
  generated video still moved (26.5 dB PSNR, 0.873 SSIM against the paired control), because a
  30-step sampler amplifies any perturbation; that is a property of the sampler, not a defect
  of either kernel, and it is provable that the arithmetic did not change.

  An INEXACT kernel cannot make that argument. Both 8-bit arms sit at 3.7-4.0e-2, ~16x the
  bf16 floor. Nothing we own can currently separate "fp8 degrades quality" from "fp8 changes
  the sample": h3a-001's gate is unbuilt, cozy-eval has no ratified conditions, and h3a-020 §5
  MEASURED that `cozy-eval compare` refuses the paired video form for an attention-backend
  change and that paired audio distance cannot separate them either. A promotion would change
  every customer's output on an image upgrade, silently, with no instrument that could tell us
  whether it got worse. And it would not even reach the target: fp8 takes the 5 s cell to
  2.68 c/s against a 2.5 target, still 7% short, and the 14 s cell to 4.75, 1.9x short. So it
  buys an unverifiable quality risk for something that misses anyway.

  fp8 therefore belongs in `PINNABLE` — available for measurement, chosen deliberately — and
  the pin is the instrument that produces the evidence a promotion would need. Note also that
  it cannot be spelled here yet: diffusers 0.40.0's `_flash_3_hub` backend hard-codes
  `q_descale=k_descale=v_descale=None` and constrains q/k/v to bf16/fp16, so FA3-fp8 is not an
  `AttentionBackendName` at all. h3a-013 reached it by calling the baked kernel directly.
  Adding it is a new dispatch path that quantises per call, not a row in this table.

AN IMPORT IS NOT A PROOF THAT A KERNEL COMPUTES. h3a-013 measured the image's baked
`sageattention` sm90 kernel returning UNINITIALIZED MEMORY on an H100 — no exception, the right
shape and dtype, rel L2 311, at 14.5x the card's fp8 peak because no kernel launched at all
(upstream `sm90_compile.py` bound a `register_fake` meta stub over the `custom_op` of the same
name). The image gate proved only that it imported and that its extension file existed; both
were true. So a kernel is now PROVEN, once, at prepare, against exact fp32 attention on the
device it will actually run on, and one that disagrees beyond its DECLARED budget is dropped
from auto-selection and refused to a pin. This does not weaken the purity claim above: the
verdict is a deterministic function of (device, image), computed once, never per call.

The implausible RATE — 14.5x a card's theoretical peak — is the symptom that made the defect
visible, and it is deliberately NOT the check. It would need a per-device peak table we do not
maintain and would say nothing about a kernel that is merely wrong at a plausible speed. The
numeric check catches the same failure with three orders of magnitude of margin (rel L2 311
against a 0.2 budget) and no device-specific constant.

AND THE CHECK ABOVE CANNOT RUN IF THE PROCESS IS ALREADY DEAD. A kernel with no device image
for the card does not raise: FlashAttention-3's `flash_fwd_launch_template.h` prints
`no kernel image is available for execution on the device` and calls `exit(1)`. h3a-022 measured
exactly that on an sm_120 RTX PRO 6000 with the snapshot THIS image bakes, and there is no
Python exception at any point — `_computes` never gets to run, `available`'s broad `except`
never sees anything, and the pod dies at prepare with no typed refusal. A C++ abort is not
recoverable, so it has to be UNREACHABLE. Admission is therefore bounded ABOVE as well as
below, and a device outside a kernel's range is refused by name before the kernel is ever
called.

THE UPPER BOUND IS READ FROM THE BINARY, NOT DECLARED BESIDE IT. `kernels` writes a
`metadata.json` next to every built variant whose `backend.archs` is the artifact's own
statement of what it carries — the baked flash-attn3 snapshot says `["8.0", "9.0a"]`, and
`9.0a` is architecture-CONDITIONAL, so it has an image for sm_90 and for nothing else. A static
`max_sm` beside it would be a second copy of that list, and a second copy drifts: this table
already said "sm90 and up" while the binary carried sm_90a, which is the whole defect. So a
BAKED candidate declares no ceiling at all and is bounded by the metadata of the variant that
was actually loaded; a candidate with no such artifact — torch's own backends, and a
distribution whose diffusers backend names its architecture in its own name — carries a static
`max_sm` with the reason it is not a guess. A baked artifact that declares no archs is a
REFUSAL, not a pass: a check that reads nothing must not look like a check that read.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from cozy_runtime.internal.anima_optimization import DIFFUSERS_VERSION
from cozy_runtime.internal.encoding import DeviceFacts

try:
    from kernels import get_local_kernel
except ImportError as _absent_kernels:  # pragma: no cover - exercised by the absent arm
    KERNELS_ABSENT = f"does not import ({_absent_kernels})"
else:
    KERNELS_ABSENT = ""

#: Where the worker image bakes `kernels` Hub snapshots (tensorhub's cu130 Dockerfile):
#: `<distribution>/build/<variant>/` holds one built variant and its metadata. Nothing here is
#: fetched — a pod's executor has no network for it.
IMAGE_KERNELS_DIR = Path("/opt/cozy/kernels")


class AttentionRefusal(RuntimeError):
    """A typed refusal on the attention axis. Only a PIN can raise one; auto never does."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class Candidate:
    """One attention kernel the runtime knows how to reach, and what it needs to run."""

    #: The stable public name. This is the execution-path override's vocabulary (cr-125) and
    #: the word the boot line prints; it is NOT diffusers' backend spelling, which is an
    #: implementation detail of the version we pin.
    name: str
    #: diffusers' backend member value at `DIFFUSERS_VERSION`.
    backend: str
    #: The lowest CUDA compute capability, as `DeviceFacts.sm` spells it (90 = sm90). This is a
    #: FUNCTIONAL floor and it is allowed to be stricter than the binary: the baked flash-attn3
    #: snapshot carries an sm_80 cubin, but FlashAttention-3's kernels are Hopper's and h3a-012
    #: measured it on sm90, so sm90 is where the entry starts.
    min_sm: int
    #: Activation dtypes the kernel takes. Empty = any (torch's own dispatch).
    dtypes: frozenset[str]
    #: The largest per-head dimension the kernel serves. 0 = unbounded.
    max_head_dim: int
    #: The highest CUDA compute capability this candidate's binary has an image for; 0 = not
    #: bounded here. A BAKED candidate MUST leave this 0 — its ceiling is read from the
    #: artifact's own `metadata.json` (`_baked_archs`), because a static copy of a binary's
    #: arch list is the drift that killed a Blackwell pod. 0 on a NON-baked candidate is a
    #: CLAIM, not an omission: torch's in-tree backends are bounded by torch, they raise a
    #: Python exception when they have no plan for a card (h3a-022 measured cuDNN doing exactly
    #: that on sm_100 and computing fine on sm_120), and the floor may never be bounded at all
    #: because a device it refused would hold no attention kernel whatsoever.
    max_sm: int = 0
    #: A `kernels` Hub snapshot the image BAKES, by distribution directory name; empty for a
    #: kernel that is installed as an ordinary distribution or is torch's own.
    baked: str = ""
    #: An installed distribution this kernel needs, by project name; empty when torch's own.
    distribution: str = ""
    #: The relative-L2 disagreement with exact fp32 attention this kernel is ALLOWED at
    #: `PROBE`, at which point it stops being attention. Every entry declares one, because the
    #: budget is the statement of what the kernel is: an exact kernel may not drift past bf16's
    #: own floor, and an 8-bit one is permitted its quantisation and nothing more.
    rel_l2_max: float = 0.0
    #: Why this entry sits where it sits. A rank is a MEASUREMENT or it is not a rank.
    why: str = ""


#: RANKED, FASTEST FIRST — and every rank cites the run that put it there. A kernel with no
#: measurement ON THIS DEVICE is not ranked, because a rank we cannot cite is a guess.
#: Reordering this table changes what a pod computes, so it moves with the runtime version
#: and with nothing else.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3",
        backend="_flash_3_hub",
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=256,
        # NO `max_sm`. This is the one entry whose ceiling is a BINARY's, and the binary says
        # it: the baked snapshot's `metadata.json` declares `["8.0", "9.0a"]`, so the ceiling
        # is sm_90 and it is read at resolution rather than restated here. Restating it is
        # what broke: this row said sm90-and-up while the `.so` carried `sm_90a`, and h3a-022
        # measured the consequence — `exit(1)` from C++ on an sm_120 card, before any check
        # this module owns could run.
        baked="flash-attn3",
        rel_l2_max=2e-2,
        why=(
            "MEASURED on an H100 SXM (h3a-012, one DiT forward at S=106,932): 22.84 s of "
            "attention against cuDNN's 28.45 s — 718 vs 576 TF/s, 29.02 vs 34.70 s per "
            "forward. On the SERVED path (th-186 run 410, FL2VA 5 s / 30 steps) 159.6 s "
            "execution against a paired 181.2 s control: -14.3 % s/step. Same math to "
            "kernel-precision noise: max abs 1.2e-7, rel L2 2.8e-10, audio bit-identical"
        ),
    ),
    Candidate(
        name="sdpa",
        backend="native",
        min_sm=0,
        dtypes=frozenset(),
        max_head_dim=0,
        # UNBOUNDED IN BOTH DIRECTIONS, and that is the one bound this module may not add. A
        # ceiling here would leave a card with no attention kernel at all, and there is
        # nothing to protect it from: torch's chooser owns its own fallback down to the math
        # backend, and its failures are Python exceptions the numeric self-check already
        # catches (h3a-022 measured torch's own dispatch computing correctly on sm_120 at
        # 365.0 TF/s).
        max_sm=0,
        rel_l2_max=2e-2,
        why=(
            "torch's own `scaled_dot_product_attention` dispatch, and the FLOOR: it takes "
            "every dtype, head dimension and mask, so a site no ranked kernel admits still "
            "runs. It is not a fallback nobody measured — on sm90 torch's C++ chooser already "
            "selects cuDNN (h3a-012: 34.70 s/forward, the arm every H3 number is quoted "
            "against), which is why pinning cuDNN explicitly buys nothing"
        ),
    ),
)

#: ADMITTED AND NAMEABLE, BUT NOT RANKED. Two different reasons, both the same rule: a rank is
#: a measurement. Name one with the execution-path override (cr-125) to measure it; a
#: measurement that beats the table promotes it into `CANDIDATES` in a runtime release.
PINNABLE: tuple[Candidate, ...] = (
    Candidate(
        name="cudnn",
        backend="_native_cudnn",
        min_sm=80,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=128,
        # NO CEILING, and it is a claim rather than an oversight. This is torch's in-tree
        # cuDNN frontend, not a binary of ours: it is built for whatever the image's cuDNN
        # supports, and when it has no plan for a card it RAISES — v1's B200 screen got
        # `cuDNN Frontend error: No valid execution plans built` on sm_100, which auto drops
        # and a pin surfaces typed. h3a-022 then measured cuDNN 9.20 serving sm_120 at
        # 373.6 TF/s, the fastest exact arm on that card, so a ceiling would be wrong as well
        # as unnecessary.
        max_sm=0,
        rel_l2_max=2e-2,
        why=(
            "MEASURED SLOWER than the floor it duplicates (h3a-012): 34.99 s against 34.70 s, "
            "because diffusers' wrapper adds three `.contiguous()` around the same cuDNN "
            "kernel torch already picks. Nameable so the reference arm can be pinned "
            "explicitly, never chosen"
        ),
    ),
    Candidate(
        name="sageattention",
        backend="_sage_qk_int8_pv_fp8_cuda_sm90",
        min_sm=90,
        dtypes=frozenset({"bfloat16", "float16"}),
        max_head_dim=128,
        # sm90 AND ONLY sm90, and this ceiling cannot drift because the architecture is IN THE
        # BACKEND NAME: `_sage_qk_int8_pv_fp8_cuda_sm90` reaches
        # `sageattention._qattn_sm90`, which the image builds as
        # `arch=compute_90a,code=sm_90a` — architecture-conditional, so it has an image for
        # sm_90 and for nothing above or below it. h3a-022 measured
        # `cudaErrorNoKernelImageForDevice` from this extension on sm_120. A future image that
        # bakes an sm_120 SageAttention adds a NEW row for its own sm_120 entry point; it does
        # not widen this one.
        max_sm=90,
        distribution="sageattention",
        rel_l2_max=2e-1,
        why=(
            "8-bit attention: Q/K int8 per-thread, V fp8-e4m3 per-channel, fp32+fp32 PV. It "
            "CHANGES THE NUMBERS, which the owner admits (2026-09-08). NOW TIMED (h3a-013, "
            "H100 SXM at H3's geometry): 1.12x flash-attn3, against flash-attn3's own fp8 "
            "path at 1.31x — so it is not even the best 8-bit arm, and 1.12x does not justify "
            "a dependency. It also SHIPPED DEAD: worker images before th-192 baked upstream "
            "eb615cf6, whose `sm90_compile.py` bound a `register_fake` meta stub over the "
            "custom op, so every sm90 call returned its uninitialized output buffer at rel L2 "
            "311 with no error. The self-check below refuses exactly that, on any image"
        ),
    ),
)

#: Every kernel the override may name, by public name.
BY_NAME: dict[str, Candidate] = {c.name: c for c in (*CANDIDATES, *PINNABLE)}


@dataclass(frozen=True, slots=True)
class Ready:
    """One candidate PROVEN present in this environment, with diffusers' member for it."""

    candidate: Candidate
    member: Any
    location: str = ""
    """A baked kernel's variant directory; empty otherwise."""

    @property
    def name(self) -> str:
        return self.candidate.name


def _registry() -> Any:
    """diffusers' attention backend registry at the reviewed version, or a typed absence."""
    try:
        installed = metadata.version("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise AttentionRefusal(
            "attention_kernel_absent", "diffusers is not installed; attention backends are its"
        ) from exc
    if installed != DIFFUSERS_VERSION:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"the reviewed attention backends are diffusers=={DIFFUSERS_VERSION}; this "
            f"environment installs {installed}",
        )
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    return _AttentionBackendRegistry


def _member(backend: str) -> Any:
    from diffusers.models.attention_dispatch import AttentionBackendName

    return AttentionBackendName(backend)


def _capability(declared: str) -> tuple[int, int, str] | None:
    """One architecture as an artifact spells it — `"9.0a"`, `"8.0"`, `"sm_120"` — or None.

    None is "this runtime cannot read what the artifact declared", and every caller treats it
    as NOT covered, because the alternative is calling a kernel we could not check.
    """
    entry = declared.strip().lower().removeprefix("sm_").removeprefix("compute_")
    suffix = entry[-1] if entry[-1:] in ("a", "f") else ""
    digits = entry.removesuffix(suffix) if suffix else entry
    major, _, minor = digits.partition(".")
    if not minor:
        major, minor = digits[:-1], digits[-1:]
    if not (major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor), suffix


def _covers(declared: str, sm: int) -> bool:
    """Does device code built for `declared` have an image for a device of capability `sm`?

    NVIDIA's binary-compatibility rule and nothing more: a cubin built for X.y runs on X.z
    where z >= y, and NEVER across a major generation — so sm_90a code has no image on sm_100
    or sm_120 however close the numbers look. An ACCELERATED target (the `a` suffix) is
    architecture-CONDITIONAL — Hopper's `wgmma` and Blackwell's `tcgen05` are only reachable
    through it — and runs on exactly the architecture it was built for, so it does not even
    carry forward to a higher minor. A FAMILY target (`f`, CUDA 13) does.
    """
    parsed = _capability(declared)
    if parsed is None:
        return False
    major, minor, suffix = parsed
    if (major, minor) == (sm // 10, sm % 10):
        return True
    return suffix != "a" and major == sm // 10 and minor <= sm % 10


def _declared_archs(variant: Path) -> list[str]:
    """The architectures a `kernels` build variant says it carries, from its own metadata.

    `build/<variant>/metadata.json` is written by kernel-builder beside the `.so` it describes,
    so this is the artifact's statement about itself and not a second copy of it. An empty
    answer is a refusal at the call site, never an admission.
    """
    try:
        document = json.loads((variant / "metadata.json").read_text())
        archs = document["backend"]["archs"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [str(arch) for arch in archs] if isinstance(archs, list) else []


def _baked_archs(candidate: Candidate) -> dict[str, list[str]]:
    """Every baked variant of this candidate, and the architectures each one declares."""
    builds = IMAGE_KERNELS_DIR / candidate.baked / "build" if candidate.baked else None
    if builds is None or not builds.is_dir():
        return {}
    return {
        variant.name: _declared_archs(variant)
        for variant in sorted(builds.iterdir())
        if (variant / "metadata.json").is_file()
    }


def _admit_arch(candidate: Candidate, archs: dict[str, list[str]], device: DeviceFacts) -> None:
    """Refuse a card that none of these baked variants has device code for.

    Silence is NOT admission: a variant present but declaring no architecture is refused, the
    same as one declaring the wrong ones — a check that read nothing must not look like a
    check that passed. The one quiet case is no baked snapshot at all, which is `_load_baked`'s
    `attention_kernel_absent` to report and not this function's business.
    """
    if device.kind != "cuda" or not archs:
        return
    if any(_covers(arch, device.sm) for declared in archs.values() for arch in declared):
        return
    raise AttentionRefusal(
        "attention_kernel_unsupported",
        f"{candidate.name} is baked as {archs} and none of those builds carries device code "
        f"for {device.name} (sm{device.sm}). This is a REFUSAL rather than an attempt because "
        "the attempt is not survivable: FlashAttention-3's launcher prints `no kernel image is "
        "available for execution on the device` and exits the process from C++, so neither the "
        "numeric self-check nor auto-selection's own fallback would ever run (h3a-022 measured "
        "it on an sm_120 RTX PRO 6000 with this exact snapshot). Only a worker image baking a "
        "build that carries this architecture changes the answer",
    )


def _load_baked(candidate: Candidate, device: DeviceFacts) -> tuple[Any, str]:
    """Load the image's baked snapshot into diffusers' Hub slot so nothing is fetched.

    diffusers resolves a Hub kernel over the network on first use; filling `kernel_fn` and the
    wrapped forward/backward slots makes `_maybe_download_kernel_for_backend` a no-op, which
    is the only reason a pod without egress can run FlashAttention-3 at all.

    `_resolve` has already refused a card no baked variant claims. This repeats the check
    against the variant `kernels` ACTUALLY resolved, which is the authoritative one — the
    resolution is by torch ABI and CUDA version, not by architecture, so an image baking two
    variants could pass the first check on one of them and load the other. Loading is safe on
    any card (a `.so` full of Hopper cubins still dlopens on Blackwell); CALLING it is not, so
    the check sits after the load and before diffusers' registry is touched, and a refused
    kernel leaves no half-filled slot behind.
    """
    if KERNELS_ABSENT:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} loads through the `kernels` distribution, which {KERNELS_ABSENT}",
        )
    location = IMAGE_KERNELS_DIR / candidate.baked
    builds = location / "build"
    variants = (
        sorted(p for p in builds.iterdir() if (p / "metadata.json").is_file())
        if (builds.is_dir())
        else []
    )
    if not variants:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} is not baked under {builds}; the worker image bakes it there "
            "and this executor has no network to fetch it",
        )
    try:
        module = get_local_kernel(location)
    except Exception as exc:
        # A snapshot this torch cannot load is an ABSENT kernel, not a crash: the image
        # baked a variant built for another ABI, and the floor still serves.
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} does not load from {location}: {type(exc).__name__}: {exc}",
        ) from exc
    loaded = Path(str(getattr(module, "__file__", ""))).resolve().parent
    _admit_arch(candidate, {loaded.name: _declared_archs(loaded)}, device)
    from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY

    member = _member(candidate.backend)
    if member not in _HUB_KERNELS_REGISTRY:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"diffusers=={DIFFUSERS_VERSION} does not resolve {candidate.backend!r} through "
            "its Hub-kernel registry, so a baked snapshot cannot be placed in its slot",
        )
    config = _HUB_KERNELS_REGISTRY[member]
    config.kernel_fn = _lookup(module, config.function_attr, candidate.name)
    for attr, slot in (
        ("wrapped_forward_attr", "wrapped_forward_fn"),
        ("wrapped_backward_attr", "wrapped_backward_fn"),
    ):
        dotted = getattr(config, attr)
        if dotted:
            setattr(config, slot, _lookup(module, dotted, candidate.name))
    return member, str(loaded)


def _lookup(module: Any, dotted: str, name: str) -> Any:
    entry: Any = module
    for part in dotted.split("."):
        entry = getattr(entry, part, None)
        if entry is None:
            raise AttentionRefusal("attention_kernel_absent", f"{name} does not expose {dotted}")
    return entry


#: The self-check's shape: (batch, sequence, heads, head dimension). Small enough that the
#: whole proof costs under a millisecond of a prepare that already takes minutes, and large
#: enough to cross a real kernel's tiling — 256 rows is two 128-row blocks, and 128 is the head
#: dimension every entry in both tables serves.
PROBE: tuple[int, int, int, int] = (1, 256, 4, 128)


def _computes(entry: Ready, device: DeviceFacts) -> None:
    """PROVE the kernel computes attention on THIS device, or refuse it by name.

    Runs one forward through the same `dispatch_attention_fn` a pinned processor dispatches
    through — not a hand-rolled call to the kernel — and compares it against exact fp32
    attention over the kernel's own inputs, so bf16's input rounding is not charged to it.

    Non-finite output fails BEFORE the ratio, because a NaN comparison is False and would
    otherwise report a pass. That is the whole reason this is written as a guard rather than a
    one-line assert: the h3a-023 hazard note records live NaN reports for this kernel family on
    Hopper, and the check that misses NaN is worse than no check.
    """
    if device.kind != "cuda":
        # There is no device to prove anything on, and the only entry a CPU executor admits is
        # torch's own eager dispatch. Silence here is the honest answer, not a skipped gate.
        return
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    batch, sequence, heads, dim = PROBE
    dtype = (
        torch.bfloat16
        if not entry.candidate.dtypes or "bfloat16" in entry.candidate.dtypes
        else torch.float16
    )
    seed = torch.Generator(device="cuda").manual_seed(0x5A6E)
    shape = (batch, sequence, heads, dim)
    query, key, value = (
        (torch.randn(shape, generator=seed, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
        for _ in range(3)
    )
    with torch.no_grad():
        got = dispatch_attention_fn(query, key, value, backend=entry.member)
    if isinstance(got, tuple):
        got = got[0]

    exact = torch.nn.functional.scaled_dot_product_attention(
        *(tensor.transpose(1, 2).float() for tensor in (query, key, value))
    ).transpose(1, 2)
    if tuple(got.shape) != tuple(exact.shape):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned {tuple(got.shape)} where attention over {shape} is "
            f"{tuple(exact.shape)}",
        )
    got = got.float()
    if not bool(torch.isfinite(got).all()):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned non-finite values on {device.name} (sm{device.sm}); it is "
            "not serving attention and a NaN would pass every ratio test",
        )
    disagreement = float((got - exact).norm() / exact.norm())
    if disagreement > entry.candidate.rel_l2_max:
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} disagrees with exact fp32 attention at rel L2 {disagreement:.4g} on "
            f"{device.name} (sm{device.sm}), past its declared budget of "
            f"{entry.candidate.rel_l2_max:.4g}. A kernel that returns its uninitialized output "
            "buffer raises nothing and has the right shape (h3a-013), so this is the only "
            "thing that can tell the difference",
        )


def _resolve(candidate: Candidate, device: DeviceFacts) -> Ready:
    """One candidate, proven present here AND proven to compute, or a typed refusal.

    Present is not enough: an import and an extension file were both true of a kernel that
    returned uninitialized memory (h3a-013), so resolution ends with `_computes`.

    IT BEGINS WITH THE ARCHITECTURE, before diffusers or the loader is asked anything. The
    artifact's own `metadata.json` needs neither, and a card it has no image for must be
    refused wherever this runs — a dev box, a CI runner, a pod — not only where the rest of
    the stack happens to be installed.
    """
    _admit_arch(candidate, _baked_archs(candidate), device)
    registry = _registry()
    member = _member(candidate.backend)
    if member not in registry._backends:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"diffusers=={DIFFUSERS_VERSION} registers no backend {candidate.backend!r}",
        )
    if candidate.baked:
        loaded_member, location = _load_baked(candidate, device)
        ready = Ready(candidate, loaded_member, location)
        _computes(ready, device)
        return ready
    if candidate.distribution:
        try:
            installed = metadata.version(candidate.distribution)
        except metadata.PackageNotFoundError as exc:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.name} needs the {candidate.distribution} distribution, which "
                "this image does not install",
            ) from exc
        try:
            import_module(candidate.distribution.replace("-", "_"))
        except Exception as exc:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.distribution}=={installed} does not import: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
    ready = Ready(candidate, member)
    _computes(ready, device)
    return ready


def _device_admits(candidate: Candidate, device: DeviceFacts) -> str:
    """Empty when this DEVICE is inside the candidate's DECLARED range, else why it is not.

    Both ends, and the ceiling is the one this module was missing: an open-ended range is a
    claim that the kernel serves every architecture ever built, which is false of every entry
    that is a compiled binary rather than a dispatcher. A candidate whose ceiling belongs to a
    baked artifact declares none here and is bounded by `_admit_arch` at resolution instead.
    """
    if candidate.min_sm and (device.kind != "cuda" or device.sm < candidate.min_sm):
        return (
            f"{candidate.name} needs sm{candidate.min_sm} or better; this executor holds "
            f"{device.kind} sm{device.sm}"
        )
    if candidate.max_sm and device.kind == "cuda" and device.sm > candidate.max_sm:
        serves = (
            f"sm{candidate.max_sm} only"
            if candidate.min_sm == candidate.max_sm
            else f"sm{candidate.min_sm} through sm{candidate.max_sm}"
        )
        return (
            f"{candidate.name} serves {serves}; this executor holds sm{device.sm}, and the "
            "binary behind it carries no device code for that architecture. Calling it anyway "
            "returns `cudaErrorNoKernelImageForDevice` or kills the process from C++ rather "
            "than raising something this runtime could recover from (h3a-022)"
        )
    return ""


def for_device(device: DeviceFacts) -> list[Candidate]:
    """The ranked table filtered by DEVICE alone — no import, no image. The pure half.

    Pure means it can only apply the bounds the TABLE declares. A baked kernel's real ceiling
    is in its artifact, so it is applied one step later, at resolution — which is why this
    list can still name a kernel that `available` then drops.
    """
    return [candidate for candidate in CANDIDATES if not _device_admits(candidate, device)]


def available(device: DeviceFacts) -> list[Ready]:
    """The RANKED kernels this device and this image can actually run, fastest first.

    Never raises: the floor takes everything, so auto-selection has no refusal arm. A ranked
    kernel this image does not ship is simply not in the answer, and an environment with no
    diffusers at all offers nothing — which is right, because it holds no attention site.
    """
    ready: list[Ready] = []
    for candidate in for_device(device):
        try:
            ready.append(_resolve(candidate, device))
        except Exception:
            # BROAD ON PURPOSE. Auto-selection has no refusal arm — the floor takes
            # everything — so a kernel that fails to resolve for ANY reason must drop out of
            # the ranking rather than fail the prepare that was only trying to go faster. A
            # PIN is the opposite and keeps its typed refusals (`pinned`). A kernel that
            # RESOLVES BUT DOES NOT COMPUTE now leaves by this same door, which is the point:
            # auto-selection silently taking a dead kernel is what would have shipped
            # corrupted video (h3a-013).
            continue
    return ready


def pinned(name: str, device: DeviceFacts) -> list[Ready]:
    """The single kernel the execution-path override names, or a typed refusal (cr-125).

    A pin does not fall back. That is the whole point: a run that silently served another
    kernel would make the A/B it was asked for meaningless.
    """
    candidate = BY_NAME.get(name)
    if candidate is None:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"{name!r} is not an attention kernel this runtime knows; it knows {sorted(BY_NAME)}",
        )
    unsupported = _device_admits(candidate, device)
    if unsupported:
        # THE REFUSAL IS THE DISCOVERABILITY. A developer pinning a kernel this card cannot
        # run is told, in the same sentence, what this image and this card actually offer —
        # which is what `image_kernels` is for, and it lists what COMPUTES rather than what
        # imports (cr-127) so the answer never points at another trap.
        raise AttentionRefusal(
            "attention_kernel_unsupported",
            f"{unsupported}. This image runs {image_kernels(device)} on {device.name}",
        )
    return [_resolve(candidate, device)]


# ------------------------------------------------------------------------------ the pin


def _sites(roots: Mapping[str, Any]) -> Iterator[tuple[str, Any, Any]]:
    """(component, attention module, its processor) for every diffusers attention site.

    The walk follows public attributes and mapping values three levels deep and stops at torch
    module trees, where `modules()` finds the attention classes diffusers' own
    `set_attention_backend` looks for. A component's label is the name the fill plane gave it.
    """
    try:
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_processor import Attention, MochiAttention
    except ImportError:
        # No diffusers, no diffusers attention site. Nothing to pin, and not a refusal.
        return

    classes = (Attention, MochiAttention, AttentionModuleMixin)
    seen: set[int] = set()
    queue: deque[tuple[str, Any, int]] = deque(
        (name, root, 0) for name, root in sorted(roots.items())
    )
    while queue:
        label, value, depth = queue.popleft()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if callable(getattr(value, "modules", None)) and callable(
            getattr(value, "parameters", None)
        ):
            for module in value.modules():
                processor = getattr(module, "processor", None)
                if isinstance(module, classes) and processor is not None:
                    yield label, module, processor
            continue
        if depth >= 3:
            continue
        if isinstance(value, Mapping):
            for key, item in value.items():
                queue.append((str(key), item, depth + 1))
        elif hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                if not name.startswith("_"):
                    queue.append((name, item, depth + 1))


def _head_dim(module: Any) -> int:
    """This site's per-head dimension, or 0 when the module does not say.

    A site whose head dimension cannot be read is treated as unknown, which admits only the
    unbounded floor — the runtime does not guess a shape into a kernel that would fault on it.
    """
    declared = getattr(module, "head_dim", None)
    if isinstance(declared, int) and declared > 0:
        return declared
    heads = getattr(module, "heads", None)
    inner = getattr(module, "inner_dim", None)
    if inner is None:
        to_q = getattr(module, "to_q", None)
        inner = getattr(to_q, "out_features", None)
    if isinstance(heads, int) and heads > 0 and isinstance(inner, int) and inner > 0:
        return inner // heads
    return 0


def _dtype(module: Any) -> str:
    for parameter in module.parameters():
        return str(parameter.dtype).removeprefix("torch.")
    return ""


def _admits(candidate: Candidate, dtype: str, head_dim: int) -> bool:
    if candidate.dtypes and dtype not in candidate.dtypes:
        return False
    return not (candidate.max_head_dim and (head_dim == 0 or head_dim > candidate.max_head_dim))


@dataclass(frozen=True, slots=True)
class Applied:
    """What the runtime pinned, per component and per kernel. The BOOT record, and all of it."""

    #: component -> kernel name -> attention sites pinned to it.
    hosts: dict[str, dict[str, int]]
    #: kernel name -> the baked variant directory it loaded from, when it loaded from one.
    locations: dict[str, str]
    #: the ranked kernels this image and device offered, fastest first.
    offered: tuple[str, ...]
    #: set when the caller PINNED one through the execution-path override (cr-125).
    pin: str = ""

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kernels in self.hosts.values():
            for name, sites in kernels.items():
                out[name] = out.get(name, 0) + sites
        return out

    def document(self) -> dict[str, Any]:
        return {
            "kernels": self.totals(),
            "hosts": {name: dict(kernels) for name, kernels in self.hosts.items()},
            "offered": list(self.offered),
            "locations": dict(self.locations),
            "pin": self.pin,
            # RENDERED HERE, not by the worker: one renderer for the one line, in the module
            # that knows what it means.
            "line": self.line(),
        }

    def line(self) -> str:
        """The ONE line a pod prints at boot. It is the whole per-run attention record."""
        return line(self.totals(), self.hosts, list(self.offered), self.pin)


def line(
    totals: Mapping[str, int],
    hosts: Mapping[str, Mapping[str, int]],
    offered: Sequence[str],
    pin: str,
) -> str:
    """The boot line, from the facts alone — so a GROUP prepare renders the same sentence from
    its merged facts as a single one does from its own `Applied` (cr-124)."""
    if not totals:
        return "NONE — this construction holds no diffusers attention site"
    chosen = ", ".join(f"{name} x{sites}" for name, sites in sorted(totals.items()))
    where = "; ".join(
        f"{component}: {'+'.join(sorted(kernels))}" for component, kernels in sorted(hosts.items())
    )
    how = f"PINNED {pin}" if pin else f"auto over {list(offered)}"
    return f"{chosen} ({how}) — {where}"


def install(ready: list[Ready], roots: Mapping[str, Any], pin: str = "") -> Applied:
    """Pin the fastest admitted kernel on every attention site of the constructed components.

    Per SITE, not per model: `set_attention_backend` would also move diffusers' PROCESS-GLOBAL
    active backend, which is what carried a DiT's kernel into an fp16 VAE decode once (cr-122).
    Setting `processor._attention_backend` is the same pin without the propagation.

    With a PIN (cr-125) `ready` holds exactly one kernel, and a site it does not admit is a
    typed refusal rather than a quiet demotion.
    """
    hosts: dict[str, dict[str, int]] = {}
    locations: dict[str, str] = {}
    for component, module, processor in _sites(roots):
        dtype, head_dim = _dtype(module), _head_dim(module)
        for entry in ready:
            if not _admits(entry.candidate, dtype, head_dim):
                continue
            processor._attention_backend = entry.member
            hosts.setdefault(component, {})
            hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
            if entry.location:
                locations[entry.name] = entry.location
            break
        else:
            raise AttentionRefusal(
                "attention_kernel_unsupported",
                f"{ready[0].name if ready else 'no attention kernel'} does not serve "
                f"{component}'s attention: "
                f"{dtype or 'unknown'} activations at head dimension "
                f"{head_dim or 'unknown'}. A pinned kernel is never quietly replaced",
            )
    return Applied(hosts, locations, tuple(entry.name for entry in ready), pin)


def select(device: DeviceFacts, roots: Mapping[str, Any], pin: str = "") -> Applied:
    """The whole decision, in one call: what this device and image offer, pinned on the sites."""
    return install(pinned(pin, device) if pin else available(device), roots, pin)


def image_kernels(device: DeviceFacts) -> str:
    """What this image and device can actually RUN, for a refusal message (cr-125).

    Device-filtered first, so a card the entry does not claim is never probed; and it lists
    what computes, not what imports — naming a kernel here that returns garbage would send a
    developer straight at the trap the refusal was meant to steer them away from.

    The filter is the FULL declared range, not just the floor. On a card above a kernel's
    ceiling that is the difference between an answer and a dead process: probing means calling,
    and calling a kernel with no image for the device exits from C++ (h3a-022).
    """
    shipped: list[str] = []
    for candidate in (*CANDIDATES, *PINNABLE):
        if _device_admits(candidate, device):
            continue
        try:
            _resolve(candidate, device)
        except Exception:
            continue
        shipped.append(candidate.name)
    return ", ".join(shipped) or "none"


def baked_variants() -> dict[str, Any]:
    """The baked snapshots under `IMAGE_KERNELS_DIR`, for the boot record and for triage."""
    out: dict[str, Any] = {}
    for candidate in (*CANDIDATES, *PINNABLE):
        builds = IMAGE_KERNELS_DIR / candidate.baked / "build" if candidate.baked else None
        if builds is None or not builds.is_dir():
            continue
        for variant in sorted(builds.iterdir()):
            metadata_file = variant / "metadata.json"
            if metadata_file.is_file():
                out[candidate.name] = {
                    "variant": variant.name,
                    **json.loads(metadata_file.read_text()),
                }
    return out
