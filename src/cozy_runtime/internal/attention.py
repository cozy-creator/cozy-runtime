"""Select and install attention backends for constructed diffusers modules.

Every kernel is a Runtime-local forward (`attention_upstream`, `attention_sol`) that
`attention_ulysses` registers with Diffusers, so one wrapper serves every degree. Selection
checks device architecture, the kernel's presence in this environment or the image's kernel
site, a numerical probe of the served function at a production sequence length, site
dtype/head dimension and context-parallel support. Explicit pins refuse unsupported sites.
Automatic selection preserves attention activation precision; quantizing those activations
requires an explicit backend choice.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib import import_module, metadata
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from cozy_runtime.author._attention import AttentionContext
from cozy_runtime.author._attention_scope import _EXPECTED
from cozy_runtime.author._numeric import within, worst_of
from cozy_runtime.internal import attention_sol, attention_ulysses, kernel_site
from cozy_runtime.internal import attention_upstream as upstream
from cozy_runtime.internal.attention_ulysses import Local
from cozy_runtime.internal.encoding import DeviceFacts


class AttentionRefusal(RuntimeError):
    """An unsupported explicit selection or unavailable attention implementation."""

    def __init__(self, code: str, detail: str, measured: Mapping[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.code = code
        #: the probe's measurement behind a numeric refusal
        self.measured = measured


@dataclass(frozen=True, slots=True)
class Candidate:
    """One attention kernel the runtime knows how to reach, and what it needs to run."""

    name: str
    backend: str
    local: Local
    min_sm: int  # CUDA capability, e.g. 90 for sm90; zero also admits CPU.
    dtypes: frozenset[str]  # Empty admits all dtypes.
    max_head_dim: int  # Zero is unbounded.
    max_sm: int = 0  # Snapshot binaries use their artifact's architecture metadata instead.
    snapshot: str = ""  # A `kernels` build the image bakes beside its kernel site.
    distribution: str = ""
    jit: bool = False  # Source kernels use the declared device range, not binary metadata.
    rel_l2_max: float = 0.0  # Numerical probe threshold, not an output-quality guarantee.
    why: str = ""
    sms: frozenset[int] = frozenset()
    head_dims: frozenset[int] = frozenset()
    import_name: str = ""
    #: SMs where `rel_l2_max` is provisional: no probe has measured it on that hardware yet.
    uncalibrated: frozenset[int] = frozenset()


_HALF = frozenset({"bfloat16", "float16"})

CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3",
        backend="_cozy_flash_attn3",
        local=upstream.FA3,
        min_sm=90,
        dtypes=_HALF,
        max_head_dim=256,
        snapshot="flash-attn3",
        distribution="kernels",
        rel_l2_max=1e-2,
        why=(
            "Quality-preserving default. H100 H3 N109104: FA3 BF16 about 506 ms, "
            "FA4 about 501 ms, cuDNN about 616 ms; no consistent FA4 gain; all three "
            "0.0017 rel L2 to FP32 (h3a-044). "
            "This is a curated H100 ranking, not a universal hardware winner."
        ),
    ),
    Candidate(
        name="sdpa",
        backend="_cozy_sdpa",
        local=upstream.SDPA,
        min_sm=0,
        dtypes=frozenset(),
        max_head_dim=0,
        rel_l2_max=1e-2,
        why="PyTorch scaled_dot_product_attention; supports all site dtypes and head dimensions.",
    ),
)

PINNABLE: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3-fp8",
        backend="_cozy_flash_attn3_fp8",
        local=upstream.FA3_FP8,
        min_sm=90,
        dtypes=_HALF,
        max_head_dim=128,
        snapshot="flash-attn3",
        distribution="kernels",
        rel_l2_max=6e-2,
        why=(
            "Explicit experiment, numerically out of its budget at H3 length: rel L2 0.31 "
            "to FP32 at 109k rows versus BF16 0.0017 (h3a-044; long-sequence FP8 PV "
            "accumulation). 781.3 s versus BF16 907.1 s for 15 s video."
        ),
    ),
    Candidate(
        name="flash-attn4-fp8",
        backend="_cozy_flash_attn4_fp8",
        local=upstream.FA4_FP8,
        min_sm=100,
        max_sm=103,
        sms=frozenset({100, 103}),
        dtypes=frozenset({"bfloat16"}),
        max_head_dim=128,
        distribution="flash-attn-4",
        import_name="flash_attn.cute",
        jit=True,
        rel_l2_max=6e-2,
        uncalibrated=frozenset({100, 103}),
        why=(
            "Explicit experiment: FA4's SM100-only e4m3 forward with per-head descales. "
            "Hopper's 0.31 (h3a-044) is its FP8 MMA accumulator; Blackwell accumulates in "
            "FP32, so the dequantised-input control (0.027) is the expectation. Unmeasured."
        ),
    ),
    Candidate(
        name="flash-attn4",
        backend="_cozy_flash_attn4",
        local=upstream.FA4,
        min_sm=90,
        dtypes=_HALF,
        max_head_dim=256,
        max_sm=110,
        distribution="flash-attn-4",
        import_name="flash_attn.cute",
        jit=True,
        rel_l2_max=1e-2,
        why=(
            "Unranked: matched H100 BF16 kernels showed no consistent FA4 gain over FA3. "
            "Evidence: outputs/h3-attention-oracle-20260914/"
            "torch214-cudnn924/qualification/REPORT.md."
        ),
    ),
    Candidate(
        name="cudnn",
        backend="_cozy_cudnn",
        local=upstream.CUDNN,
        min_sm=80,
        dtypes=_HALF,
        max_head_dim=128,
        rel_l2_max=1e-2,
        why="SDPA restricted to cuDNN; on H100 it is the kernel `sdpa` dispatches (h3a-012).",
    ),
    Candidate(
        name="sageattention",
        backend="_cozy_sage_int8_fp8",
        local=upstream.SAGE,
        min_sm=89,
        dtypes=_HALF,
        max_head_dim=128,
        max_sm=120,
        sms=frozenset({89, 90, 120}),
        distribution="sageattention",
        rel_l2_max=1e-1,
        uncalibrated=frozenset({89, 120}),
        why=(
            "Explicit experiment: INT8 Q/K + FP8 P/V, 0.0096-0.066 rel L2 on H3 Q/K/V "
            "(h3a-044); 801.2 s versus BF16 907.1 s for 15 s video; matched clips differ. "
            "SM89/SM120 run upstream's SM89 kernel: 0.038 rel L2 standalone on SM120 (h3a-022)."
        ),
    ),
    Candidate(
        name="sageattention-fp16pv",
        backend="_cozy_sage_fp16pv",
        local=upstream.SAGE_FP16PV,
        min_sm=90,
        dtypes=_HALF,
        max_head_dim=128,
        max_sm=90,
        distribution="sageattention",
        rel_l2_max=3e-2,
        why=(
            "Explicit experiment: INT8 Q/K + FP16 P/V (FP32 accumulation), upstream's SM80 "
            "kernel built sm_90a and not upstream-qualified on SM90. H100 Gaussian 3.55e-3 rel "
            "L2 versus FP8 P/V 3.75e-2 and BF16 2.33e-3, at 0.63x FA3 speed (h3a-013). The "
            "budget refuses an FP8-P/V-class answer."
        ),
    ),
    Candidate(
        name="kitchen-int8",
        backend="_cozy_kitchen_int8",
        local=upstream.KITCHEN,
        min_sm=90,
        max_sm=121,
        sms=frozenset(major * 10 + minor for major, minor in upstream.KITCHEN_SMS),
        dtypes=_HALF,
        max_head_dim=256,
        distribution="comfy-kitchen",
        rel_l2_max=5e-2,
        uncalibrated=frozenset({100, 103, 120, 121}),
        why="Explicit dense INT8 experiment; 0.014 rel L2 to FA3 on H3 Q/K/V (h3a-060).",
    ),
    Candidate(
        name="flashinfer-bf16-fp8",
        backend="_cozy_flashinfer_bf16_fp8",
        local=upstream.FLASHINFER,
        min_sm=100,
        max_sm=103,
        sms=frozenset({100, 103}),
        dtypes=frozenset({"bfloat16"}),
        max_head_dim=128,
        head_dims=frozenset({128}),
        distribution="flashinfer-python",
        import_name="flashinfer",
        jit=True,
        rel_l2_max=2e-1,
        uncalibrated=frozenset({100, 103}),
        why="Explicit BF16 Q/K + FP8 P/V experiment; H3 qualification pending (h3a-060).",
    ),
    Candidate(
        name="sageattention3-fp4",
        backend="_cozy_sage3_fp4",
        local=upstream.SAGE3,
        min_sm=120,
        max_sm=120,
        sms=frozenset({120}),
        dtypes=_HALF,
        max_head_dim=128,
        head_dims=frozenset({64, 128}),
        distribution="sageattn3",
        rel_l2_max=2e-1,
        uncalibrated=frozenset({120}),
        why=(
            "Explicit FP4 experiment, consumer Blackwell only upstream. Standalone on an RTX "
            "PRO 6000 at H3 geometry: 0.145 rel L2 to FP32, 704 TF/s, 0.98x H100 FA3 BF16 "
            "(h3a-022)."
        ),
    ),
    Candidate(
        name="sol-attn",
        backend="_cozy_sol_bf16",
        local=attention_sol.SOL,
        min_sm=90,
        max_sm=90,
        sms=frozenset({90}),
        dtypes=frozenset({"bfloat16"}),
        max_head_dim=128,
        head_dims=frozenset({128}),
        distribution="sol-attn",
        rel_l2_max=2e-2,  # Native all-keys arithmetic control, not sparse approximation error.
        why="Explicit sparse experiment on its CuTe SM90 route; eager model layout.",
    ),
)

BY_NAME: dict[str, Candidate] = {c.name: c for c in (*CANDIDATES, *PINNABLE)}


@dataclass(frozen=True, slots=True)
class Ready:
    """One candidate PROVEN present in this environment, with diffusers' member for it."""

    candidate: Candidate
    member: Any
    location: str = ""
    """The selected local snapshot variant directory; empty for ordinary modules."""
    admission: dict[str, Any] | None = None
    """Retained arithmetic/contract checks; never a model-quality qualification."""
    artifact: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.candidate.name


def _registry() -> Any:
    """Diffusers' attention backend registry, or a typed absence.

    Any Diffusers exposing the per-site dispatch registry qualifies; each backend is
    still proven by registration and a numerical smoke test before it serves.
    """
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    except ImportError as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"this environment's diffusers exposes no attention backend registry: {exc}",
        ) from exc
    return _AttentionBackendRegistry


def _member(candidate: Candidate) -> Any:
    """The Diffusers member serving this candidate, registered on first use."""
    return attention_ulysses.member(candidate.backend, candidate.local)


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
    """Does device code built for `declared` have an image for a device of capability `sm`?"""
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


def _artifact(distribution: str) -> dict[str, Any]:
    """Identify installed bytes using the wheel inventory already retained by the closure."""
    installed = metadata.distribution(distribution)
    record = installed.read_text("RECORD")
    if not record:
        raise AttentionRefusal(
            "attention_kernel_absent", f"{distribution} has no installed wheel RECORD"
        )
    result: dict[str, Any] = {
        "distribution": installed.metadata["Name"],
        "version": installed.version,
        "record_digest": "sha256:" + hashlib.sha256(record.encode()).hexdigest(),
    }
    direct = installed.read_text("direct_url.json")
    if direct:
        try:
            origin = json.loads(direct)
            archive = origin.get("archive_info", {})
            digest = archive.get("hashes", {}).get("sha256")
            if not digest and str(archive.get("hash", "")).startswith("sha256="):
                digest = archive["hash"].removeprefix("sha256=")
            if not digest:
                # uv can retain a hash-qualified wheel URL with empty archive_info.
                digest = parse_qs(urlsplit(origin.get("url", "")).fragment).get("sha256", [""])[0]
            if (
                isinstance(digest, str)
                and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest.lower())
            ):
                result["wheel_digest"] = "sha256:" + digest.lower()
        except (ValueError, TypeError, AttributeError):
            pass
    urls = installed.metadata.get_all("Project-URL") or []
    for url in urls:
        label, _, value = url.partition(",")
        if label.strip().lower() == "source":
            result["source"] = value.strip()
    return result


def _snapshot_archs(candidate: Candidate) -> dict[str, list[str]]:
    """Every variant of the image snapshot, and the architectures each one declares."""
    builds = kernel_site.snapshot(candidate.snapshot) / "build" if candidate.snapshot else None
    if builds is None or not builds.is_dir():
        return {}
    return {
        variant.name: _declared_archs(variant)
        for variant in sorted(builds.iterdir())
        if (variant / "metadata.json").is_file()
    }


def _site_archs(candidate: Candidate) -> dict[str, list[str]]:
    """What the bake gate proved the kernel site's build of this candidate carries, only when
    the distribution this process imports IS the site's: a package's own copy says nothing
    here. A source (`jit`) kernel carries no device code to declare."""
    declared = kernel_site.arches(candidate.name)
    if candidate.jit or not candidate.distribution or not declared:
        return {}
    try:
        location = Path(str(metadata.distribution(candidate.distribution).locate_file("")))
    except metadata.PackageNotFoundError:
        return {}
    site = Path(kernel_site.current().path)
    return {"kernel site": declared} if location.resolve() == site.resolve() else {}


def _baked_archs(candidate: Candidate) -> dict[str, list[str]]:
    return _snapshot_archs(candidate) or _site_archs(candidate)


def _admit_arch(candidate: Candidate, archs: dict[str, list[str]], device: DeviceFacts) -> None:
    """Refuse a card that none of these baked builds has device code for."""
    if candidate.jit:
        # NOTHING TO READ, so nothing is checked here and the static `max_sm` on the row is
        # the whole ceiling. A source snapshot declares no architectures because it carries
        # no device code, and treating that silence as the same refusal an empty
        # `metadata.json` earns from a COMPILED variant would refuse the kernel on every
        # card it serves.
        return
    if device.kind != "cuda" or not archs:
        return
    if any(_covers(arch, device.sm) for declared in archs.values() for arch in declared):
        return
    raise AttentionRefusal(
        "attention_kernel_unsupported",
        f"{candidate.name} declares {archs} and none of those builds carries device code "
        f"for {device.name} (sm{device.sm}). This is a REFUSAL rather than an attempt because "
        "the attempt is not survivable: FlashAttention-3's launcher prints `no kernel image is "
        "available for execution on the device` and exits the process from C++, so neither the "
        "numeric self-check nor auto-selection's own fallback would ever run (h3a-022 measured "
        "it on an sm_120 RTX PRO 6000 with this exact snapshot). A worker image or "
        "installed dependency wheel carrying this architecture changes the answer",
    )


def _load_snapshot(candidate: Candidate, loader: Any, device: DeviceFacts) -> str:
    """Load the image's `kernels` snapshot locally (never fetched) and bind its forward."""
    location = kernel_site.snapshot(candidate.snapshot)
    builds = location / "build"
    if not builds.is_dir() or not any((p / "metadata.json").is_file() for p in builds.iterdir()):
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} has no snapshot under {builds}; this image does not bake it",
        )
    try:
        module = loader.get_local_kernel(location)
    except Exception as exc:
        # A snapshot this torch cannot load is an ABSENT kernel, not a crash: the image
        # baked a variant built for another ABI, and the floor still serves.
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} does not load from {location}: {type(exc).__name__}: {exc}",
        ) from exc
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    # kernels supports both build/<variant>/__init__.py and a nested package.
    loaded = next(
        (parent for parent in module_path.parents if parent.parent == builds.resolve()), None
    )
    if loaded is None or not (loaded / "metadata.json").is_file():
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} resolved outside its selected snapshot: {module_path}; "
            f"expected a variant under {builds}",
        )
    _admit_arch(candidate, {loaded.name: _declared_archs(loaded)}, device)
    if getattr(module, "flash_attn_func", None) is None:
        raise AttentionRefusal(
            "attention_kernel_absent", f"{candidate.name} does not expose flash_attn_func"
        )
    upstream.bind_fa3(module, loaded.name)
    return str(loaded)


#: The probe: one H3 rank's post-exchange document at 10 s (a measured packed length, odd so
#: the last block is partial), because error grows with length: FA3 FP8 measured 0.10 rel L2
#: at 16k rows and 0.31 at 109k where a 256-row probe passed it (h3a-044). Every probed
#: kernel reduces within a head, so two heads keep it ~150 MB on any card. The checked rows
#: are held to exact float64 attention over every key.
PROBE: tuple[int, int, int, int] = (1, 73_439, 2, 128)
PROBE_ROWS = 64
#: Rows checked on every probe besides the samples, because a kernel that mishandles one
#: region (a partial tail tile, a tile edge) passes a sample. Every kernel tiles query rows in
#: multiples of 16, so both rows at each multiple of 16 are edge rows of every tile any kernel
#: uses, and the first and last 256 rows hold the first tile and the partial tail tile of any
#: tile height up to 256. Each group is held to the budget alone, so a fault cannot average out.
PROBE_TILE_GRAIN = 16
PROBE_END_ROWS = 256
#: Probe Q/K rows are RMS-normalised and scaled so logits have this standard deviation: the
#: sharp attention QK-normalised video DiTs serve (effective keys ~30 of 73k). N(0, 1) inputs
#: give logit std 1, near-uniform attention where per-head FP8 rounding of K averages out:
#: FA3 FP8 passed its budget there on H100 and garbled H3 video (run 1250). At std 4 its
#: dequantised inputs alone measure 0.11 rel L2 against 0.032 for Kitchen INT8 and 0.0018
#: for BF16 kernels (h3a-044 real H3 deep blocks: 0.18 for FP8 inputs, 0.066 for Sage).
PROBE_LOGIT_STD = 4.0
#: Exact rows per float64 pass: 256 x 73,439 keys is ~150 MB of weights.
_EXACT_ROWS_PER_PASS = 256
#: Torch's quadratic-memory math fallback cannot hold the production length and does not
#: accumulate across keys, so it is probed here; so is Sol's policy contract.
PROBE_SHORT: tuple[int, int, int, int] = (1, 256, 4, 128)
#: A fresh problem per probe, so a kernel handing back a stale buffer that happens to hold the
#: previous probe's correct answer cannot pass this one.
_PROBE_SEEDS = count(0x5A6E)


def probe(
    entry: Ready, device: DeviceFacts, shape: tuple[int, int, int, int] | None = None
) -> dict[str, Any]:
    """PROVE the served function computes attention on THIS device, or refuse it by name.

    `shape` overrides the probe geometry (default `PROBE` on CUDA and `PROBE_SHORT` on CPU,
    whose kernel does not tile; Sol's default is `PROBE_SHORT` plus one row), so a caller can
    hold a kernel to a served per-rank shape."""
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    dtype = (
        torch.bfloat16
        if not entry.candidate.dtypes or "bfloat16" in entry.candidate.dtypes
        else torch.float16
    )
    cuda = device.kind == "cuda"
    target = torch.device("cuda", device.index) if cuda else torch.device("cpu")
    current = torch.cuda.device(target) if cuda else nullcontext()
    seed = torch.Generator(device=target).manual_seed(next(_PROBE_SEEDS))
    if entry.name == "sol-attn":
        if shape is None:
            batch, sequence, heads, dim = PROBE_SHORT
            shape = (batch, sequence + 1, heads, dim)  # Sol's partial final block as well.
        query, key, value = (
            (torch.randn(shape, generator=seed, device=target) * 0.5).to(dtype) for _ in range(3)
        )
        with torch.no_grad(), current:
            return _probe_sol(entry, device, query, key, value)
    query, key, value = probe_inputs(shape or (PROBE if cuda else PROBE_SHORT), seed, target, dtype)
    options = {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": False,
        "scale": None,
        "enable_gqa": False,
    }
    with torch.no_grad(), current:
        impl = entry.candidate.local.impl(query, key, value, options)
        if impl == "math":
            query, key, value = (t[:, : PROBE_SHORT[1]].contiguous() for t in (query, key, value))
        got = dispatch_attention_fn(query, key, value, backend=entry.member)
        got = _validate_probe_contract(entry, device, got, query)
        by_rows, floor, checked = _disagreement(got, query, key, value, seed)
    worst, disagreement = worst_of(by_rows)
    measured = {
        "shape": list(query.shape),
        "checked_rows": checked,
        "rel_l2_by_rows": by_rows,
        "rel_l2_to_exact": disagreement,
        "rounding_floor": floor,
        "rel_l2_max": entry.candidate.rel_l2_max,
        **({"budget": "uncalibrated"} if device.sm in entry.candidate.uncalibrated else {}),
        "impl": impl,
        "logit_std": PROBE_LOGIT_STD,
    }
    where = f"{worst} rows of {checked} checked over {query.shape[1]} keys"
    _gate(entry, device, disagreement, where, measured)
    return measured


def probe_inputs(
    shape: tuple[int, int, int, int], seed: Any, device: Any, dtype: Any
) -> tuple[Any, Any, Any]:
    """Probe Q/K/V: Gaussian, with every Q/K row at RMS `sqrt(PROBE_LOGIT_STD)` (QK norm)."""
    import torch

    query, key, value = (torch.randn(shape, generator=seed, device=device) for _ in range(3))
    gain = PROBE_LOGIT_STD**0.5
    query, key = (t * (gain / t.pow(2).mean(-1, keepdim=True).sqrt()) for t in (query, key))
    return query.to(dtype), key.to(dtype), value.to(dtype)


def _probe_rows(length: int, seed: Any, device: Any) -> dict[str, Any]:
    """The probe's row groups: evenly spaced and seeded random samples, the first and last
    `PROBE_END_ROWS` rows, and both rows at every `PROBE_TILE_GRAIN` edge."""
    import torch

    ends = min(PROBE_END_ROWS, length)
    edges = torch.arange(PROBE_TILE_GRAIN, length, PROBE_TILE_GRAIN, device=device)
    sampled = torch.cat(
        (
            torch.linspace(0, length - 1, PROBE_ROWS, device=device).long(),
            torch.randint(length, (PROBE_ROWS,), generator=seed, device=device),
        )
    )
    return {
        "sampled": sampled,
        "first": torch.arange(ends, device=device),
        "last": torch.arange(length - ends, length, device=device),
        "tile_edges": torch.cat((edges - 1, edges)),
    }


def _disagreement(
    got: Any, query: Any, key: Any, value: Any, seed: Any
) -> tuple[dict[str, float], float, int]:
    """Rel L2 of `got` to exact attention per row group, the BF16/FP16 rounding floor of the
    exact rows, and how many distinct rows were checked."""
    import torch

    groups = _probe_rows(query.shape[1], seed, query.device)
    checked = torch.zeros(query.shape[1], dtype=torch.bool, device=query.device)
    for members in groups.values():
        checked[members] = True
    rows = checked.nonzero().flatten()
    exact = _exact_rows(query, key, value, rows)
    error = got[:, rows].double() - exact
    by_rows: dict[str, float] = {}
    for name, members in groups.items():
        mask = torch.isin(rows, members)
        by_rows[name] = float(error[:, mask].norm() / exact[:, mask].norm())
    floor = float((exact.to(query.dtype).double() - exact).norm() / exact.norm())
    return by_rows, floor, len(rows)


def _exact_rows(query: Any, key: Any, value: Any, rows: Any) -> Any:
    """Exact float64 attention for `rows` of the query over every key, one head and at most
    `_EXACT_ROWS_PER_PASS` rows at a time."""
    import torch

    out = torch.empty(
        (query.shape[0], len(rows), query.shape[2], query.shape[3]),
        dtype=torch.float64,
        device=query.device,
    )
    for head in range(query.shape[2]):
        k, v = key[:, :, head].double(), value[:, :, head].double()
        for start in range(0, len(rows), _EXACT_ROWS_PER_PASS):
            chunk = rows[start : start + _EXACT_ROWS_PER_PASS]
            q = query[:, chunk, head].double()
            weights = (q @ k.transpose(1, 2) * q.shape[-1] ** -0.5).softmax(dim=-1)
            out[:, start : start + len(chunk), head] = weights @ v
    return out


def _gate(
    entry: Ready,
    device: DeviceFacts,
    disagreement: float,
    where: str,
    measured: Mapping[str, Any] | None = None,
) -> None:
    """Hold a finite result to its declared budget. Past rel L2 1 the output carries no
    attention at all (a kernel returning its uninitialized buffer raises nothing and has the
    right shape, h3a-013); below it, the kernel runs and is out of budget, said so by name."""
    budget = entry.candidate.rel_l2_max
    if within(disagreement, budget):
        return
    code = (
        "attention_kernel_out_of_budget"
        if math.isfinite(disagreement) and disagreement < 1
        else "attention_kernel_wrong"
    )
    raise AttentionRefusal(
        code,
        f"{entry.name} disagrees with exact attention at rel L2 {disagreement:.4g} on "
        f"{device.name} (sm{device.sm}) ({where}), past its declared budget of {budget:.4g}"
        + ("" if code == "attention_kernel_wrong" else "; it runs, numerically out of budget"),
        measured,
    )


def _probe_sol(
    entry: Ready, device: DeviceFacts, query: Any, key: Any, value: Any
) -> dict[str, Any]:
    """Separate native arithmetic from the wrapper's approximate sparse policy."""
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope

    inputs = (query, key, value)
    originals = tuple(t.clone() for t in inputs)
    sequence, prefix, padding = query.shape[1], 33, 15
    q_ref, k_ref, v_ref = (t.transpose(1, 2).float() for t in inputs)
    exact = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref).transpose(1, 2)
    # Public sink semantics keep all KV blocks exact inside the actual native
    # kernel. No Runtime dense-prefix overwrite can make this control pass.
    native = _validate_probe_contract(
        entry, device, attention_sol._native(query, key, value, None, sequence), query
    )
    arithmetic = float((native.float() - exact).norm() / exact.norm())
    measured = {
        "shape": list(query.shape),
        "rel_l2_by_rows": {"all_keys": arithmetic},
        "rel_l2_max": entry.candidate.rel_l2_max,
    }
    _gate(entry, device, arithmetic, "all-keys", measured)
    if any(
        not torch.equal(t.view(torch.uint8), old.view(torch.uint8))
        for t, old in zip(inputs, originals, strict=True)
    ):
        raise AttentionRefusal(
            "attention_kernel_wrong", "Sol all-keys control modified its inputs", measured
        )

    padded = tuple(
        torch.cat((t, torch.full_like(t[:, :padding], float("nan"))), dim=1) for t in inputs
    )
    before = tuple(t.view(torch.uint8).clone() for t in padded)
    probe: Any = torch.nn.Module()
    probe.forward = lambda q, k, v: dispatch_attention_fn(q, k, v, backend=entry.member)
    attention_sol.install_site(probe, "probe", "attention")
    try:
        with (
            attention_scope(AttentionLayout(sequence, prefix, 0)),
            attention_sol.observing() as counts,
        ):
            sparse = probe(*padded)
            repeated = probe(*padded)
    finally:
        attention_sol.remove_site(probe, "probe", "attention")
    _validate_probe_contract(entry, device, sparse, padded[0])
    expected_prefix = attention_sol._dense(query[:, :prefix], key, value, None)
    if (
        not torch.equal(sparse[:, :prefix], expected_prefix)
        or torch.count_nonzero(sparse[:, sequence:]).item() != 0
        or not torch.equal(sparse, repeated)
        or any(
            not torch.equal(t.view(torch.uint8), old) for t, old in zip(padded, before, strict=True)
        )
        or counts != dict(sparse=2, dense_step=0, dense_path=0, dense_prefix=2)
    ):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            "Sol sparse policy violated protected-prefix, padding, repeat "
            "or input-ownership contracts",
            measured,
        )
    return {
        **measured,
        "arithmetic_control": "native_all_keys_selected",
        "arithmetic_rel_l2_to_fp32": arithmetic,
        "arithmetic_rel_l2_max": entry.candidate.rel_l2_max,
        "sparse_policy_rel_l2_to_fp32": float(
            (sparse[:, :sequence].float() - exact).norm() / exact.norm()
        ),
        "sparse_error_meaning": "approximation_only_not_model_quality",
        "protected_prefix": prefix,
        "padding": padding,
        "repeat_equal": True,
        "inputs_unchanged": True,
        "calls": counts,
    }


def _validate_probe_contract(entry: Ready, device: DeviceFacts, got: Any, query: Any) -> Any:
    """Geometry and finite output are required even when approximation changes the answer;
    non-finite fails before any ratio, which a NaN would pass."""
    import torch

    if isinstance(got, tuple):
        got = got[0]
    if not isinstance(got, torch.Tensor) or got.dtype != query.dtype or got.device != query.device:
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} must return {query.dtype} on {query.device}, matching its input",
        )
    if tuple(got.shape) != tuple(query.shape):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned {tuple(got.shape)}; attention over "
            f"{tuple(query.shape)} requires {tuple(query.shape)}",
        )
    if not bool(torch.isfinite(got).all()):
        raise AttentionRefusal(
            "attention_kernel_wrong",
            f"{entry.name} returned non-finite values on {device.name} (sm{device.sm}); it is "
            "not serving attention and a NaN would pass every ratio test",
        )
    return got


def _import(candidate: Candidate) -> Any:
    """The candidate's upstream module, from this environment or the image's kernel site."""
    try:
        installed = metadata.version(candidate.distribution)
    except metadata.PackageNotFoundError as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name} needs the {candidate.distribution} distribution, which neither "
            f"this package environment nor the image installs ({kernel_site.current().describe()})",
        ) from exc
    try:
        return import_module(candidate.import_name or candidate.distribution.replace("-", "_"))
    except Exception as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.distribution}=={installed} does not import: {type(exc).__name__}: {exc}",
        ) from exc


def _snapshot_artifact(location: str) -> dict[str, Any]:
    body = (Path(location) / "metadata.json").read_bytes()
    declared = json.loads(body)
    return {
        "snapshot": Path(location).name,
        "id": str(declared.get("id", "")),
        "metadata_digest": "sha256:" + hashlib.sha256(body).hexdigest(),
    }


#: Each kernel's probe verdict, proven once per (kernel, device, kernel site, registered
#: implementation): all fixed for the process's life. The startup warm (`warm`) fills it
#: before the first request, so construction and a request's pin check read it instead of
#: probing again. Presence refusals are cheap and asked again.
_PROVEN: dict[tuple[Any, ...], Ready | AttentionRefusal] = {}
_VERDICTS = frozenset({"attention_kernel_wrong", "attention_kernel_out_of_budget"})


def _proof_key(candidate: Candidate, device: DeviceFacts) -> tuple[Any, ...] | None:
    try:
        served = _registry()._backends.get(_member(candidate))
    except Exception:
        return None  # nothing is registered to prove; `_prove` refuses with the reason
    return (candidate, device, kernel_site.current(), served)


def _resolve(candidate: Candidate, device: DeviceFacts) -> Ready:
    """One candidate, proven present here AND proven to compute, or a typed refusal."""
    key = _proof_key(candidate, device)
    if key is None:
        return _prove(candidate, device)
    known = _PROVEN.get(key)
    if isinstance(known, AttentionRefusal):
        raise known
    if known is None:
        try:
            known = _PROVEN[key] = _prove(candidate, device)
        except AttentionRefusal as exc:
            if exc.code in _VERDICTS:
                _PROVEN[key] = exc
            raise
    return known


def warm(device: DeviceFacts, preferred: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Prove, before any construction exists, what this device's first construction will
    select: the model's preference walk up to its first kernel that serves, then the ranked
    table `available` resolves. One row per kernel: its verdict and what proving it cost."""
    rows: list[dict[str, Any]] = []

    def one(candidate: Candidate) -> bool:
        started = time.perf_counter()
        held = _proof_key(candidate, device) in _PROVEN
        try:
            _resolve(candidate, device)
            status = "ready"
        except Exception as exc:
            status = str(getattr(exc, "code", type(exc).__name__))
        rows.append(
            {
                "kernel": candidate.name,
                "status": status,
                "ms": round((time.perf_counter() - started) * 1000, 1),
                **({"held": True} if held else {}),
            }
        )
        return status == "ready"

    with measuring("warm"):
        for name in preferred:
            candidate = BY_NAME.get(name)
            if candidate is not None and not _device_admits(candidate, device) and one(candidate):
                break
        done = {row["kernel"] for row in rows}
        for candidate in for_device(device):
            if candidate.name not in done:
                one(candidate)
    return rows


def _expected(live_tokens: int, module: Any) -> None:
    """A model's early length statement reaches Sol when a Sol site serves `module`."""
    if "sol-attn" not in observed({"": module}).totals():
        return
    import torch

    device = next((p.device for p in module.parameters() if p.device.type != "meta"), None)
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    if device is not None:
        attention_sol.expect(live_tokens, device)


_EXPECTED.append(_expected)


def _prove(candidate: Candidate, device: DeviceFacts) -> Ready:
    _admit_arch(candidate, _baked_archs(candidate), device)
    registry = _registry()
    module = _import(candidate) if candidate.distribution else None
    location, artifact = "", None
    if candidate.snapshot:
        location = _load_snapshot(candidate, module, device)
        artifact = _snapshot_artifact(location)
    elif candidate.distribution:
        artifact = _artifact(candidate.distribution)
    if candidate.name == "sol-attn":
        _bind_sol_dense(device, registry)
    ready = Ready(candidate, _member(candidate), location, artifact=artifact)
    try:
        admission = probe(ready, device)
    except Exception as exc:
        code = getattr(exc, "code", type(exc).__name__)
        _note(candidate, device, code, getattr(exc, "measured", None))
        raise
    _note(candidate, device, "ready", admission)
    return replace(ready, admission=admission)


#: The phase proving a kernel now: the startup `warm`, a construction's selection
#: (`prepare`) or a request's pin (`request`). Verdicts are kept, so it is when each was MEASURED.
_SOURCE: ContextVar[str] = ContextVar("attention_probe_source", default="prepare")
#: kernel -> this process's latest probe of it. One process serves one device.
_MEASURED: dict[str, dict[str, Any]] = {}


@contextmanager
def measuring(source: str) -> Iterator[None]:
    token = _SOURCE.set(source)
    try:
        yield
    finally:
        _SOURCE.reset(token)


def _note(
    candidate: Candidate, device: DeviceFacts, status: str, measured: Mapping[str, Any] | None
) -> None:
    row = measured or {}
    _MEASURED[candidate.name] = {
        "kernel": candidate.name,
        "status": status,
        "source": _SOURCE.get(),
        "shape": list(row.get("shape") or ()),
        "rel_l2_by_rows": dict(row.get("rel_l2_by_rows") or {}),
        "budget": candidate.rel_l2_max,
        "budget_state": "uncalibrated" if device.sm in candidate.uncalibrated else "calibrated",
    }


def evidence(served: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    """This process's probes as per-rank execution evidence: `probes` for the kernels it
    `served`, `skipped` for every kernel its probe refused. Empty lists are absent."""
    names = set(served)
    probes = [dict(_MEASURED[name]) for name in sorted(names) if name in _MEASURED]
    skipped = [
        dict(row)
        for name, row in sorted(_MEASURED.items())
        if name not in names and row["status"] != "ready"
    ]
    return {**({"probes": probes} if probes else {}), **({"skipped": skipped} if skipped else {})}


def _bind_sol_dense(device: DeviceFacts, registry: Any) -> None:
    """Sol serves only its CuTe SM90 route (upstream silently falls back to Triton without
    CuTe) and recomputes dense rows with the first ranked dense kernel on this rank."""
    if device.kind == "cuda":
        route = attention_sol.backend(device.index)
        if route != "cute_sm90":
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"sol-attn selects its {route!r} route on {device.name}; only cute_sm90 is "
                "served, which needs cutlass.cute and cuda.bindings from the kernel site",
            )
    if attention_sol.dense_identity():
        return
    dense = next(
        (entry for entry in available(device) if _admits(entry.candidate, "bfloat16", 128)),
        None,
    )
    if dense is None:
        raise AttentionRefusal("attention_kernel_absent", "Sol has no qualified dense reference")
    attention_sol.bind_dense(registry._backends[dense.member], dense.name + ":" + dense.location)


def _device_admits(candidate: Candidate, device: DeviceFacts) -> str:
    """Empty when this DEVICE is inside the candidate's DECLARED range, else why it is not."""
    if candidate.sms and device.sm not in candidate.sms:
        return f"{candidate.name} serves SM {sorted(candidate.sms)}, not sm{device.sm}"
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


def available(device: DeviceFacts, skipped: dict[str, str] | None = None) -> list[Ready]:
    """The ranked kernels this device and environment can actually run, fastest first.

    Never raises: the floor takes everything, so auto-selection has no refusal arm. A ranked
    kernel this image does not ship is not in the answer; why is written into `skipped`.
    An environment with no diffusers at all offers nothing — it holds no attention site.
    """
    ready: list[Ready] = []
    for candidate in for_device(device):
        try:
            ready.append(_resolve(candidate, device))
        except Exception as exc:
            if skipped is not None:
                skipped[candidate.name] = f"{type(exc).__name__}: {exc}"[:300]
            # BROAD ON PURPOSE. Auto-selection has no refusal arm — the floor takes
            # everything — so a kernel that fails to resolve for ANY reason must drop out of
            # the ranking rather than fail the prepare that was only trying to go faster. A
            # PIN is the opposite and keeps its typed refusals (`pinned`). A kernel that
            # RESOLVES BUT DOES NOT COMPUTE now leaves by this same door, which is the point:
            # auto-selection silently taking a dead kernel is what would have shipped
            # corrupted video (h3a-013).
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
            f"{unsupported}. This environment runs {image_kernels(device)} on {device.name}",
        )
    return [_resolve(candidate, device)]


# ------------------------------------------------------------------------------ the pin


def _trees(roots: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield each constructed module tree, preserving component labels."""
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
            yield label, value
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


def _attention_records(roots: Mapping[str, Any]) -> Iterator[tuple[str, str, Any, Any]]:
    """All Diffusers attention processors, including those with no backend dispatch."""
    try:
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_processor import Attention, MochiAttention
    except ImportError:
        return
    classes = (Attention, MochiAttention, AttentionModuleMixin)
    seen: set[int] = set()
    for component, tree in _trees(roots):
        for path, module in tree.named_modules():
            processor = getattr(module, "processor", None)
            if id(module) not in seen and isinstance(module, classes) and processor is not None:
                seen.add(id(module))
                yield component, path, module, processor


def _site_records(roots: Mapping[str, Any]) -> Iterator[tuple[str, str, Any, Any]]:
    """Only processors implementing Diffusers' per-site backend selection contract."""
    for record in _attention_records(roots):
        if hasattr(record[3], "_attention_backend"):
            yield record


def _unmanaged_reason(component: str, path: str, processor: Any) -> str:
    return (
        f"{component}/{path} uses {type(processor).__name__}, which has no per-site "
        "attention backend selection"
    )


def _sites(roots: Mapping[str, Any]) -> Iterator[tuple[str, Any, Any]]:
    for component, _path, module, processor in _site_records(roots):
        yield component, module, processor


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


def _projection_dtype(projection: Any) -> str:
    """Encoded output contract, or an ordinary floating Linear's result dtype."""
    import torch

    declared = getattr(projection, "out_dtype", None)
    if declared is None and type(projection) is torch.nn.Linear:
        declared = projection.weight.dtype
    name = str(declared).removeprefix("torch.")
    return name if name in {"float16", "bfloat16", "float32", "float64"} else ""


def _dtype(module: Any) -> str:
    """Infer attention activation precision from the active Q/K/V projection contracts.

    Never inspect unrelated norm parameters or encoded weight buffers. Unknown or
    disagreeing projection contracts admit only an unbounded dtype backend (SDPA).
    Floating Linear inference assumes no external autocast changes its output dtype.
    """
    q, k, v = (getattr(module, name, None) for name in ("to_q", "to_k", "to_v"))
    packed = getattr(module, "to_qkv", None)
    fused = getattr(module, "fused_projections", None)
    projections: tuple[Any, ...]
    if packed is not None and (fused is True or (fused is None and q is None)):
        projections = (packed,)
    elif q is not None and getattr(module, "to_kv", None) is not None and fused is True:
        projections = (q, module.to_kv)
    elif q is not None and k is not None and v is not None:
        projections = (q, k, v)
    else:
        return ""
    dtypes = {_projection_dtype(projection) for projection in projections}
    return next(iter(dtypes)) if len(dtypes) == 1 and "" not in dtypes else ""


def _admits(candidate: Candidate, dtype: str, head_dim: int) -> bool:
    if candidate.head_dims and head_dim not in candidate.head_dims:
        return False
    if candidate.dtypes and dtype not in candidate.dtypes:
        return False
    return not (candidate.max_head_dim and (head_dim == 0 or head_dim > candidate.max_head_dim))


def _context_parallel(member: Any) -> bool:
    """Does diffusers' registry mark this backend as one it can shard? Read at the pin and
    never restated on a row: the backend's own registration is the authority, and a static
    copy of it would be the drift `_admit_arch` exists to refuse."""
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    return bool(_AttentionBackendRegistry._is_context_parallel_available(member))


@dataclass(frozen=True, slots=True)
class Applied:
    """What the runtime pinned, per component and per kernel. The BOOT record, and all of it."""

    #: component -> kernel name -> attention sites pinned to it.
    hosts: dict[str, dict[str, int]]
    #: kernel name -> the baked variant directory it loaded from, when it loaded from one.
    locations: dict[str, str]
    #: the ranked kernels this environment and device offered, fastest first.
    offered: tuple[str, ...]
    #: set when the caller PINNED one through the execution-path override (cr-125).
    pin: str = ""
    admission: dict[str, dict[str, Any]] | None = None
    artifacts: dict[str, dict[str, Any]] | None = None
    #: ranked kernels auto-selection could not use here, and why
    skipped: dict[str, str] | None = None

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
            **({"admission": self.admission} if self.admission else {}),
            **({"artifacts": self.artifacts} if self.artifacts else {}),
            **({"skipped": self.skipped} if self.skipped else {}),
            # RENDERED HERE, not by the worker: one renderer for the one line, in the module
            # that knows what it means.
            "line": self.line(),
        }

    def line(self) -> str:
        """The ONE line a pod prints at boot. It is the whole per-run attention record."""
        skipped = f"; skipped {sorted(self.skipped)}" if self.skipped else ""
        return line(self.totals(), self.hosts, list(self.offered), self.pin) + skipped


def line(
    totals: Mapping[str, int],
    hosts: Mapping[str, Mapping[str, int]],
    offered: Sequence[str],
    pin: str,
) -> str:
    """The boot line, from the facts alone — so a GROUP prepare renders the same sentence from
    its merged facts as a single one does from its own `Applied` (cr-124)."""
    if not totals:
        return "NONE — this construction holds no backend-selectable diffusers attention site"
    chosen = ", ".join(f"{name} x{sites}" for name, sites in sorted(totals.items()))
    where = "; ".join(
        f"{component}: {'+'.join(sorted(kernels))}" for component, kernels in sorted(hosts.items())
    )
    how = f"PINNED {pin}" if pin else f"auto over {list(offered)}"
    return f"{chosen} ({how}) — {where}"


def merge(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """A GROUP prepare's one attention fact, from its models' own `Applied.document()`s."""
    hosts: dict[str, dict[str, int]] = {}
    kernels: dict[str, int] = {}
    offered: list[str] = []
    locations: dict[str, str] = {}
    pins: set[str] = set()
    admission: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for parameter, row in documents.items():
        for component, chosen in (row.get("hosts") or {}).items():
            hosts[f"{parameter}/{component}"] = dict(chosen)
        for name, sites in (row.get("kernels") or {}).items():
            kernels[name] = kernels.get(name, 0) + int(sites)
        locations.update(row.get("locations") or {})
        for name in row.get("offered") or ():
            if name not in offered:
                offered.append(name)
        if row.get("pin"):
            pins.add(str(row["pin"]))
        if row.get("admission"):
            admission[parameter] = row["admission"]
        if row.get("artifacts"):
            artifacts[parameter] = row["artifacts"]
        skipped.update(row.get("skipped") or {})
    if not kernels:
        return None
    pin = "+".join(sorted(pins))
    return {
        "kernels": kernels,
        "hosts": hosts,
        "offered": offered,
        "locations": locations,
        "pin": pin,
        **({"admission": admission} if admission else {}),
        **({"artifacts": artifacts} if artifacts else {}),
        **({"skipped": skipped} if skipped else {}),
        "line": line(kernels, hosts, offered, pin)
        + (f"; skipped {sorted(skipped)}" if skipped else ""),
    }


def _pin_parts(pin: str) -> tuple[str, str]:
    if "=" not in pin:
        return "", pin
    component, name = pin.split("=", 1)
    if not component or not name or "=" in name or any(ch.isspace() for ch in pin):
        raise AttentionRefusal(
            "attention_kernel_unknown", f"invalid attention override {pin!r}; use component=backend"
        )
    return component, name


def for_model(pin: str, parameters: Sequence[str]) -> str:
    """Translate an aggregate model/component pin to one construction's local scope."""
    scope, backend = _pin_parts(pin)
    if "/" not in scope:
        return pin
    model, component = scope.split("/", 1)
    return f"{component}={backend}" if model in parameters else ""


def _matches(component: str, scope: str) -> bool:
    if not scope or component == scope:
        return True
    model, qualified, name = scope.partition("/")
    if not qualified:
        return component.rsplit("/", 1)[-1] == scope
    owners, owned, component_name = component.partition("/")
    return bool(owned and model in owners.split("+") and name == component_name)


def _ineligible(
    candidate: Candidate, device: DeviceFacts, dtype: str, head_dim: int, degree: int
) -> str:
    """Empty when this site, device and environment admit CANDIDATE, else why not. Static:
    resolution and the numerical probe come after."""
    unsupported = _device_admits(candidate, device)
    if unsupported:
        return unsupported
    if not _admits(candidate, dtype, head_dim):
        return (
            f"{candidate.name} does not serve {dtype or 'unknown'} activations at head "
            f"dimension {head_dim or 'unknown'}"
        )
    try:
        _admit_arch(candidate, _baked_archs(candidate), device)
        if candidate.distribution:
            metadata.version(candidate.distribution)
        member = _member(candidate)
        if member not in _registry()._backends:
            return f"{candidate.name} is not registered with Diffusers"
        if degree > 1 and not _context_parallel(member):
            return f"{candidate.name} cannot shard sequence-parallel degree {degree}"
    except metadata.PackageNotFoundError:
        return f"{candidate.name} needs the {candidate.distribution} distribution, not installed"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:300]
    return ""


def _eligible(
    candidate: Candidate, device: DeviceFacts, dtype: str, head_dim: int, degree: int
) -> bool:
    return not _ineligible(candidate, device, dtype, head_dim, degree)


def _preferred(
    order: Sequence[str],
    device: DeviceFacts,
    dtype: str,
    head_dim: int,
    degree: int,
    resolved: dict[str, Ready],
    refused: dict[str, str],
) -> Ready | None:
    """The first kernel of a model's preference that serves this site on this rank, each
    one held to the same capability checks and numerical probe as auto-selection. A kernel
    that fails either is recorded in `refused` and the walk continues."""
    for name in order:
        candidate = BY_NAME.get(name)
        if candidate is None:
            raise AttentionRefusal(
                "attention_kernel_unknown",
                f"model attention preference names {name!r}; this runtime knows {sorted(BY_NAME)}",
            )
        if name in refused:
            continue
        entry = resolved.get(name)
        if entry is not None:
            if (
                not _device_admits(candidate, device)
                and _admits(candidate, dtype, head_dim)
                and (degree == 1 or _context_parallel(entry.member))
            ):
                return entry
            continue
        reason = _ineligible(candidate, device, dtype, head_dim, degree)
        if (
            reason
            and not _device_admits(candidate, device)
            and not _admits(candidate, dtype, head_dim)
        ):
            # This site's shape only; another site may still admit the kernel.
            refused.setdefault(f"{name} at {dtype or 'unknown'}/{head_dim or 'unknown'}", reason)
            continue
        if reason:
            refused[name] = reason
            continue
        try:
            resolved[name] = _resolve(candidate, device)
        except Exception as exc:
            # As in `available`: a preferred kernel that does not resolve or compute here
            # leaves the walk rather than failing a prepare that has a later preference.
            refused[name] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        return resolved[name]
    return None


def install(
    ready: list[Ready],
    roots: Mapping[str, Any],
    pin: str = "",
    degree: int = 1,
    *,
    device: DeviceFacts | None = None,
    choose: Callable[[AttentionContext], str | Sequence[str] | None] | None = None,
    skipped: dict[str, str] | None = None,
) -> Applied:
    """Validate a complete per-site selection before changing any processor.

    `choose` returning a name is an explicit choice and refuses when unavailable; returning
    a sequence is a preference, walked per site under auto-selection's own checks, whose
    refused kernels are written into `skipped`. A pin always overrides either.
    """
    scope, name = _pin_parts(pin)
    hosts: dict[str, dict[str, int]] = {}
    locations: dict[str, str] = {}
    assignments: list[tuple[str, str, Any, Any, Ready]] = []
    resolved = {entry.name: entry for entry in ready}
    refused: dict[str, str] = dict(skipped or {})
    matched = False
    eligible_sites: dict[tuple[str, int], tuple[str, ...]] = {}
    for component, path, module, processor in _attention_records(roots):
        dtype, head_dim = _dtype(module), _head_dim(module)
        forced = bool(name and _matches(component, scope))
        matched |= forced
        if not hasattr(processor, "_attention_backend"):
            reason = _unmanaged_reason(component, path, processor)
            if forced:
                raise AttentionRefusal("attention_kernel_unsupported", reason)
            if choose is not None:
                assert device is not None
                selection = choose(
                    AttentionContext(
                        component,
                        path,
                        device.kind,
                        device.name,
                        device.sm,
                        dtype,
                        head_dim,
                        degree,
                        (),
                        "",
                        reason,
                    )
                )
                # A preference cannot be served here and says so by staying unchanged; an
                # explicit choice refuses.
                if selection is not None and not isinstance(selection, (tuple, list)):
                    raise AttentionRefusal("attention_kernel_unsupported", reason)
            continue

        offered = [
            entry
            for entry in ready
            if (entry.name == name if forced else not pin or entry.candidate in CANDIDATES)
            and _admits(entry.candidate, dtype, head_dim)
            and (degree == 1 or _context_parallel(entry.member))
        ]
        if not offered:
            detail = _site_refusal(
                name if forced else "no available attention kernel", component, dtype, head_dim
            )
            if degree > 1:
                detail = AttentionRefusal(
                    detail.code,
                    f"{detail}; sequence-parallel degree {degree} requires context parallelism",
                )
            raise detail
        entry = offered[0]
        if choose is not None and not forced:
            assert device is not None
            key = (dtype, head_dim)
            if key not in eligible_sites:
                eligible_sites[key] = tuple(
                    c.name
                    for c in BY_NAME.values()
                    if _eligible(c, device, dtype, head_dim, degree)
                )
            eligible = eligible_sites[key]
            context = AttentionContext(
                component,
                path,
                device.kind,
                device.name,
                device.sm,
                dtype,
                head_dim,
                degree,
                eligible,
                entry.name,
                entry.candidate.why,
            )
            selection = choose(context)
            if isinstance(selection, (tuple, list)):
                entry = (
                    _preferred(selection, device, dtype, head_dim, degree, resolved, refused)
                    or entry
                )
            elif selection is not None:
                if not isinstance(selection, str) or selection not in eligible:
                    raise AttentionRefusal(
                        "attention_kernel_unsupported",
                        f"model chose {selection!r} for {component}/{path}; "
                        f"eligible choices are {eligible}",
                    )
                if selection not in resolved:
                    resolved[selection] = pinned(selection, device)[0]
                entry = resolved[selection]
        assignments.append((component, path, module, processor, entry))
        hosts.setdefault(component, {})
        hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
        if entry.location:
            locations[entry.name] = entry.location
    if name and not matched:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{pin!r} matches no attention component"
        )
    _install_sites(
        [
            (entry, component, path, module)
            for component, path, module, _processor, entry in assignments
        ]
    )
    for _component, _path, _module, processor, entry in assignments:
        processor._attention_backend = entry.member
    if skipped is not None:
        skipped.update(refused)
    proofs = {name: entry.admission for name, entry in resolved.items() if entry.admission}
    artifacts = {name: entry.artifact for name, entry in resolved.items() if entry.artifact}
    return Applied(hosts, locations, tuple(resolved), pin, proofs or None, artifacts or None)


def _install_sites(records: Sequence[tuple[Ready, str, str, Any]]) -> None:
    """Install metadata hooks atomically, preserving hooks owned by preparation."""
    added: list[tuple[str, str, Any]] = []
    try:
        for entry, component, path, module in records:
            if entry.name != "sol-attn":
                continue
            local = component.rsplit("/", 1)[-1]
            if getattr(module, "_cozy_sol_site", None) is None:
                added.append((local, path, module))
            attention_sol.install_site(module, local, path)
    except BaseException:
        for component, path, module in reversed(added):
            if getattr(module, "_cozy_sol_site", None) is not None:
                attention_sol.remove_site(module, component, path)
        raise


def observed(roots: Mapping[str, Any], pin: str = "") -> Applied:
    """Read actual processor choices; never substitute the prepared recommendation."""
    names = {candidate.backend: candidate.name for candidate in BY_NAME.values()}
    hosts: dict[str, dict[str, int]] = {}
    for component, _module, processor in _sites(roots):
        backend = getattr(processor, "_attention_backend", None)
        value = getattr(backend, "value", str(backend))
        name = names.get(value, value)
        hosts.setdefault(component, {})
        hosts[component][name] = hosts[component].get(name, 0) + 1
    artifacts: dict[str, dict[str, Any]] = {}
    for name in {name for kernels in hosts.values() for name in kernels}:
        candidate = BY_NAME.get(name)
        if candidate is None or not candidate.distribution or candidate.snapshot:
            continue
        with suppress(metadata.PackageNotFoundError):
            artifacts[name] = _artifact(candidate.distribution)
    return Applied(hosts, {}, (), pin, artifacts=artifacts or None)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Prepared processor choices, restored without resolution or policy re-execution."""

    choices: tuple[tuple[Any, Any], ...]

    def restore(self) -> None:
        for processor, member in self.choices:
            processor._attention_backend = member


def snapshot(roots: Mapping[str, Any]) -> Snapshot:
    return Snapshot(
        tuple((processor, processor._attention_backend) for _, _, processor in _sites(roots))
    )


@contextmanager
def override(
    defaults: Snapshot, roots: Mapping[str, Any], swap: Swap | None, pin: str = ""
) -> Iterator[Applied]:
    """Observe one temporary request selection and restore it even when execution raises."""
    defaults.restore()
    temporary_sites = (
        [
            (component.rsplit("/", 1)[-1], path, module)
            for component, path, module in swap.modules
            if getattr(module, "_cozy_sol_site", None) is None
        ]
        if swap is not None and swap.entry.name == "sol-attn"
        else []
    )
    try:
        applied = None
        if swap is not None:
            applied = swap.apply()
        facts = observed(roots, pin)
        yield (
            replace(
                facts,
                locations=applied.locations,
                admission=applied.admission,
                artifacts={**(facts.artifacts or {}), **(applied.artifacts or {})} or None,
            )
            if applied is not None
            else facts
        )
    finally:
        try:
            for component, path, module in temporary_sites:
                if getattr(module, "_cozy_sol_site", None) is not None:
                    attention_sol.remove_site(module, component, path)
        finally:
            defaults.restore()


def _site_refusal(name: str, component: str, dtype: str, head_dim: int) -> AttentionRefusal:
    return AttentionRefusal(
        "attention_kernel_unsupported",
        f"{name} does not serve {component}'s attention: {dtype or 'unknown'} activations "
        f"at head dimension {head_dim or 'unknown'}. A pinned kernel is never quietly replaced",
    )


def _baked_in(roots: Mapping[str, Any]) -> str:
    for label, tree in _trees(roots):
        for module in tree.modules():
            if hasattr(module, "_orig_mod"):
                return f"{label} is an OptimizedModule wrapping {type(module._orig_mod).__name__}"
            if getattr(module, "_compiled_call_impl", None) is not None:
                return f"{label} holds {type(module).__name__} compiled in place"
    return ""


@dataclass(frozen=True, slots=True)
class Swap:
    """A validated per-request attention backend change, applied atomically."""

    entry: Ready
    sites: tuple[tuple[str, Any], ...]
    pin: str = ""
    modules: tuple[tuple[str, str, Any], ...] = ()

    def apply(self) -> Applied:
        hosts: dict[str, dict[str, int]] = {}
        _install_sites(
            [(self.entry, component, path, module) for component, path, module in self.modules]
        )
        for component, processor in self.sites:
            processor._attention_backend = self.entry.member
            hosts.setdefault(component, {})
            hosts[component][self.entry.name] = hosts[component].get(self.entry.name, 0) + 1
        locations = {self.entry.name: self.entry.location} if self.entry.location else {}
        proofs = {self.entry.name: self.entry.admission} if self.entry.admission else None
        artifacts = {self.entry.name: self.entry.artifact} if self.entry.artifact else None
        return Applied(
            hosts, locations, (self.entry.name,), self.pin or self.entry.name, proofs, artifacts
        )


def _is_selected(defaults: Snapshot, roots: Mapping[str, Any], pin: str) -> bool:
    """A repeated prepared pin is a no-op only at every site in its nonempty scope."""
    scope, name = _pin_parts(pin)
    candidate = BY_NAME.get(name)
    if candidate is None:
        return False
    prepared = {id(processor): member for processor, member in defaults.choices}
    records = [row for row in _attention_records(roots) if _matches(row[0], scope)]
    return bool(records) and all(
        getattr(processor, "_attention_backend", None) == candidate.backend
        and prepared.get(id(processor)) == candidate.backend
        for _, _, _, processor in records
    )


def admit(pin: str, degree: int) -> None:
    """Refuse a pin no construction here could serve before anything is constructed."""
    _, name = _pin_parts(pin)
    if not name:
        return
    candidate = BY_NAME.get(name)
    if candidate is None:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"unknown attention kernel {name!r}; this runtime knows {sorted(BY_NAME)}",
        )
    if degree > 1:
        _registry()
        if not _context_parallel(_member(candidate)):
            raise AttentionRefusal(
                "attention_kernel_unsupported",
                f"{name} does not support context parallelism, which this sequence-parallel "
                f"degree {degree} construction requires",
            )


def check(device: DeviceFacts, roots: Mapping[str, Any], name: str, degree: int = 1) -> Candidate:
    """Refuse a request pin these sites, this device or this group degree cannot serve."""
    admit(name, degree)
    scope, backend = _pin_parts(name)
    candidate = BY_NAME[backend]
    selected = {component: root for component, root in roots.items() if _matches(component, scope)}
    blocked = _baked_in(selected)
    if blocked:
        raise AttentionRefusal(
            "attention_kernel_frozen",
            f"{name} cannot be pinned per request: {blocked}. "
            "Prepare a new generation with this choice in the model attention policy "
            "before compilation",
        )
    unsupported = _device_admits(candidate, device)
    if unsupported:
        raise AttentionRefusal("attention_kernel_unsupported", unsupported)
    sites = 0
    for component, path, module, processor in _attention_records(selected):
        if not hasattr(processor, "_attention_backend"):
            raise AttentionRefusal(
                "attention_kernel_unsupported", _unmanaged_reason(component, path, processor)
            )
        dtype, head_dim = _dtype(module), _head_dim(module)
        if not _admits(candidate, dtype, head_dim):
            raise _site_refusal(candidate.name, component, dtype, head_dim)
        sites += 1
    if not sites:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{name} matches no diffusers attention site"
        )
    return candidate


def plan(device: DeviceFacts, roots: Mapping[str, Any], name: str, degree: int = 1) -> Swap:
    candidate = check(device, roots, name, degree)
    scope, _ = _pin_parts(name)
    entry = pinned(candidate.name, device)[0]
    records = [
        (component, path, module, processor)
        for component, path, module, processor in _site_records(roots)
        if _matches(component, scope)
    ]
    return Swap(
        entry,
        tuple((c, p) for c, _, _, p in records),
        name,
        tuple((c, path, m) for c, path, m, _ in records),
    )


def select(
    device: DeviceFacts,
    roots: Mapping[str, Any],
    pin: str = "",
    degree: int = 1,
    *,
    choose: Callable[[AttentionContext], str | Sequence[str] | None] | None = None,
) -> Applied:
    """Select once at preparation; optional model policy and request pins are validated."""
    scope, name = _pin_parts(pin)
    skipped: dict[str, str] = {}
    ready = pinned(name, device) if name and not scope else available(device, skipped)
    if scope:
        ready = [entry for entry in ready if entry.name != name] + pinned(name, device)
    applied = install(ready, roots, pin, degree, device=device, choose=choose, skipped=skipped)
    return replace(applied, skipped=skipped or None)


def capabilities(device: DeviceFacts) -> dict[str, Any]:
    """This rank's attention capability document: its kernel site and, for every kernel this
    runtime knows, whether it serves here — version, artifact and numeric probe when it does,
    the typed refusal when it does not. Computed per rank; ranks may differ."""
    kernels: dict[str, dict[str, Any]] = {}
    for candidate in (*CANDIDATES, *PINNABLE):
        declared = {
            "context_parallel": candidate.local.head_local,
            "dtypes": sorted(candidate.dtypes),
            "max_head_dim": candidate.max_head_dim,
            "rel_l2_max": candidate.rel_l2_max,
            **({"budget": "uncalibrated"} if device.sm in candidate.uncalibrated else {}),
        }
        unsupported = _device_admits(candidate, device)
        if unsupported:
            kernels[candidate.name] = {
                **declared,
                "status": "attention_kernel_unsupported",
                "detail": unsupported,
            }
            continue
        try:
            ready = _resolve(candidate, device)
        except Exception as exc:
            kernels[candidate.name] = {
                **declared,
                "status": getattr(exc, "code", "attention_kernel_absent"),
                "detail": f"{type(exc).__name__}: {exc}"[:600],
            }
            continue
        kernels[candidate.name] = {
            **declared,
            "status": "ready",
            **({"artifact": ready.artifact} if ready.artifact else {}),
            **({"location": ready.location} if ready.location else {}),
            **({"admission": ready.admission} if ready.admission else {}),
        }
    return {
        "device": {"kind": device.kind, "name": device.name, "sm": device.sm},
        "kernel_site": kernel_site.current().document(),
        "kernels": kernels,
    }


def image_kernels(device: DeviceFacts) -> str:
    """What this environment and device can run, for a refusal message (cr-125)."""
    ready = [
        name for name, row in capabilities(device)["kernels"].items() if row["status"] == "ready"
    ]
    return ", ".join(ready) or "none"


def implementations(seen: Mapping[str, set[str]]) -> str:
    """The implementation line for what served in an `attention_ulysses.observing` scope."""
    names = {candidate.backend: candidate.name for candidate in BY_NAME.values()}
    return attention_ulysses.describe(dict(seen), names)
