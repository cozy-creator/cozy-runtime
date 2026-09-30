"""Select and install attention backends for constructed diffusers modules.

Every kernel is a Runtime-local forward (`attention_upstream`, `attention_sol`) that
`attention_ulysses` registers with Diffusers, so one wrapper serves every degree. A kernel
that is not part of torch is compiled on this machine for this card (`kernel_sources`,
`kernel_compile`) and serves once its artifact is published: selection checks the device's
architecture, the site's dtype, head dimension and context-parallel support, and that the
kernel is compiled, imports and launches. A kernel still compiling is skipped with its
progress, and the chain falls through to the next ready one; SDPA is always ready. Explicit
pins refuse an unsupported or unready kernel. Automatic selection preserves attention
activation precision; quantizing those activations requires an explicit backend choice.
"""

from __future__ import annotations

import site
import sys
import time
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
from typing import Any

import msgspec

from cozy_runtime.author._attention import AttentionContext
from cozy_runtime.author._attention_scope import _EXPECTED
from cozy_runtime.internal import (
    attention_sol,
    attention_ulysses,
    execution_evidence,
    kernel_cache,
    kernel_compile,
    kernel_sources,
)
from cozy_runtime.internal import attention_upstream as upstream
from cozy_runtime.internal.attention_ulysses import Local
from cozy_runtime.internal.encoding import DeviceFacts

# Tensors, torch modules and Diffusers' registry, backend members and processors are `Any`:
# neither torch nor diffusers is installed in the check venv.


class AttentionRefusal(RuntimeError):
    """An unsupported explicit selection or unavailable attention implementation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class Candidate:
    """One attention kernel the runtime knows how to reach, and what it needs to run."""

    name: str
    backend: str
    local: Local
    min_sm: int  # CUDA capability, e.g. 90 for sm90; zero also admits CPU.
    dtypes: frozenset[str]  # Empty admits all dtypes.
    max_head_dim: int  # Zero is unbounded.
    max_sm: int = 0
    #: the `kernel_sources` recipe compiled for this card; empty is part of torch
    artifact: str = ""
    #: the module the kernel is imported from; empty is torch itself
    import_name: str = ""
    why: str = ""
    sms: frozenset[int] = frozenset()
    head_dims: frozenset[int] = frozenset()


_HALF = frozenset({"bfloat16", "float16"})

CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3",
        backend="_cozy_flash_attn3",
        local=upstream.FA3,
        min_sm=90,
        max_sm=90,  # upstream's hopper/ builds sm_90a only; elsewhere it exits from C++
        dtypes=_HALF,
        max_head_dim=256,
        artifact="flash-attn3",
        import_name="flash_attn_interface",
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
        why="PyTorch scaled_dot_product_attention; supports all site dtypes and head dimensions.",
    ),
)

PINNABLE: tuple[Candidate, ...] = (
    Candidate(
        name="flash-attn3-fp8",
        backend="_cozy_flash_attn3_fp8",
        local=upstream.FA3_FP8,
        min_sm=90,
        max_sm=90,
        dtypes=_HALF,
        max_head_dim=128,
        artifact="flash-attn3",
        import_name="flash_attn_interface",
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
        artifact=kernel_sources.PYTHON,
        import_name="flash_attn.cute",
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
        artifact=kernel_sources.PYTHON,
        import_name="flash_attn.cute",
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
        artifact="sageattention",
        import_name="sageattention",
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
        artifact="sageattention",
        import_name="sageattention",
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
        artifact="comfy-kitchen",
        import_name="comfy_kitchen",
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
        artifact=kernel_sources.PYTHON,
        import_name="flashinfer",
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
        artifact="sageattn3",
        import_name="sageattn3",
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
        min_sm=89,
        max_sm=120,
        sms=frozenset(major * 10 + minor for major, minor in attention_sol.ROUTES),
        dtypes=frozenset({"bfloat16"}),
        max_head_dim=128,
        head_dims=frozenset({128}),
        artifact=kernel_sources.PYTHON,
        import_name="sol_attn",
        why="Explicit sparse experiment on its CuTe routes; eager model layout.",
    ),
)

BY_NAME: dict[str, Candidate] = {c.name: c for c in (*CANDIDATES, *PINNABLE)}


@dataclass(frozen=True, slots=True)
class Ready:
    """One candidate compiled, imported and launched on this device, with Diffusers' member."""

    candidate: Candidate
    member: Any
    #: The compiled artifact serving it: flat string fields, as an observation row carries them.
    artifact: dict[str, str] | None = None

    @property
    def name(self) -> str:
        return self.candidate.name


def _registry() -> Any:
    """Diffusers' attention backend registry, or a typed absence."""
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


#: kernel -> its artifact's latest state on this rank: what a run's evidence reports.
_STATES: dict[str, kernel_compile.State] = {}
#: Kernels ready on a device stay ready for the process's life; one whose first launch
#: raised stays refused for it.
_READY: dict[tuple[str, DeviceFacts], Ready] = {}
_BROKEN: dict[tuple[str, DeviceFacts], AttentionRefusal] = {}


def _job(candidate: Candidate, device: DeviceFacts) -> kernel_compile.Job:
    try:
        return kernel_sources.job(candidate.artifact, device.sm)
    except kernel_sources.Absent as exc:
        _STATES[candidate.name] = kernel_compile.State(
            candidate.artifact, "absent", detail=str(exc)
        )
        raise AttentionRefusal("attention_kernel_absent", f"{candidate.name}: {exc}") from exc


def _compiled(candidate: Candidate, device: DeviceFacts) -> dict[str, str] | None:
    """Place the candidate's compiled artifact on the import path, or refuse with its state:
    `compiling` with its progress, `failed` for this boot, `absent`. Submitting is the whole
    of what happens to an unready one; nothing here waits. A package environment that
    carries the module itself serves it, as its pinned version always has."""
    if util.find_spec(candidate.import_name.split(".")[0]) is not None:
        return None
    store = kernel_cache.configured()
    if store is None:
        raise AttentionRefusal(
            "attention_kernel_absent", f"{candidate.name}: no machine kernel store to compile into"
        )
    job = _job(candidate, device)
    state = _STATES[candidate.name] = kernel_compile.submit(store, job)
    if state.state != "ready":
        raise AttentionRefusal(
            f"attention_kernel_{state.state}", f"{candidate.name}: {state.line()}"
        )
    site.addsitedir(str(Path(state.path) / kernel_cache.SITE))
    return {"artifact": job.key.kernel, "key": job.key.digest[:16], "compile_ms": str(state.ms)}


def _import(candidate: Candidate) -> ModuleType:
    try:
        return import_module(candidate.import_name)
    except Exception as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name}: {candidate.import_name} does not import: "
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _launch(ready: Ready, device: DeviceFacts) -> None:
    """One small call through a compiled kernel: one whose first launch raises is not served.
    Not an accuracy check. Torch's own backends need none; Sol launches only inside its
    model's layout, and a sparse step whose object is not ready runs dense."""
    if not ready.candidate.artifact or ready.name == "sol-attn" or device.kind != "cuda":
        return
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    dims = ready.candidate.head_dims or frozenset({128})
    dim = 128 if 128 in dims else min(dims)
    dtype = (
        torch.float16
        if "bfloat16" not in (ready.candidate.dtypes or {"bfloat16"})
        else (torch.bfloat16)
    )
    target = torch.device("cuda", device.index)
    try:
        with torch.no_grad(), torch.cuda.device(target):
            q, k, v = (torch.randn(1, 256, 2, dim, device=target, dtype=dtype) for _ in range(3))
            out = dispatch_attention_fn(q, k, v, backend=ready.member)
            out = out[0] if isinstance(out, tuple) else out
            torch.cuda.synchronize(target)
    except Exception as exc:
        raise AttentionRefusal(
            "attention_kernel_launch_failed",
            f"{ready.name} did not launch on {device.name}: {type(exc).__name__}: {exc}"[:600],
        ) from exc
    if tuple(out.shape) != tuple(q.shape):
        raise AttentionRefusal(
            "attention_kernel_launch_failed", f"{ready.name} returned {tuple(out.shape)}"
        )


def _resolve(candidate: Candidate, device: DeviceFacts) -> Ready:
    """One candidate compiled, imported and launched on this device, or a typed refusal."""
    known = _READY.get((candidate.name, device))
    if known is not None:
        return known
    if (candidate.name, device) in _BROKEN:
        raise _BROKEN[(candidate.name, device)]
    _registry()
    artifact = _compiled(candidate, device) if candidate.artifact else None
    if candidate.import_name:
        module = _import(candidate)
        if candidate.local in (upstream.FA3, upstream.FA3_FP8):
            upstream.bind_fa3(module, (artifact or {}).get("key", "environment"))
    if candidate.name == "sol-attn":
        _sol_route(device)
        if device.kind == "cuda":
            attention_sol.submit_card(device.index)
    ready = Ready(candidate, _member(candidate), artifact)
    try:
        _launch(ready, device)
    except AttentionRefusal as exc:
        _BROKEN[(candidate.name, device)] = exc
        _STATES[candidate.name] = kernel_compile.State(candidate.name, "failed", detail=str(exc))
        raise
    _READY[(candidate.name, device)] = ready
    if candidate.artifact:
        _STATES.setdefault(candidate.name, kernel_compile.State(candidate.artifact, "ready"))
    return ready


def expect(device: DeviceFacts, preferred: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Start compiling, before any construction exists, every kernel this device's first
    construction could select: the model's preference and the ranked table. Returns at once,
    one state row per kernel; the compiles overlap the weights' download."""
    rows: list[dict[str, Any]] = []
    names = [*preferred, *(c.name for c in for_device(device))]
    for name in dict.fromkeys(names):
        candidate = BY_NAME.get(name)
        if candidate is None:
            continue
        unsupported = _device_admits(candidate, device)
        if unsupported:
            rows.append({**_unsupported(name, unsupported).document(), "status": "unsupported"})
            continue
        started = time.perf_counter()
        why = ""
        try:
            _resolve(candidate, device)
            status = "ready"
        except Exception as exc:
            status = exc.code if isinstance(exc, AttentionRefusal) else type(exc).__name__
            why = str(exc)[:300]
        state = _STATES.get(name)
        rows.append(
            {
                **({"detail": why} if why else {}),
                **(state.document() if state is not None else {}),
                "kernel": name,
                "status": status,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    return rows


def pending(applied: Applied, device: DeviceFacts) -> list[str]:
    """Kernels this selection skipped while they compiled that are now ready: a request start
    re-selects to take them."""
    store = kernel_cache.configured()
    names = [name for name, why in (applied.skipped or {}).items() if "compiling" in why]
    if store is None or not names:
        return []
    ready = []
    for name in names:
        candidate = BY_NAME.get(name)
        if candidate is None or not candidate.artifact:
            continue
        with suppress(AttentionRefusal, kernel_sources.Absent):
            if kernel_compile.status(store, _job(candidate, device)).state == "ready":
                ready.append(name)
    return ready


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


def _unsupported(name: str, why: str) -> kernel_compile.State:
    state = _STATES[name] = kernel_compile.State(name, "unsupported", detail=why)
    return state


def evidence(served: Iterable[str]) -> list[dict[str, Any]]:
    """This rank's kernels for a run's execution record: every kernel it served, and every
    kernel of its chains it did not serve and why (compiling, failed, absent, unsupported)."""
    names = set(served)
    if not names:
        return []  # no attention site ran here
    states = {name: kernel_compile.State(name, "ready") for name in names}
    states.update(_STATES)
    return [
        {**state.document(), "kernel": name, "served": name in names}
        for name, state in sorted(states.items())
    ]


def _sol_route(device: DeviceFacts) -> None:
    """Sol serves only its CuTe routes; upstream silently falls back to Triton without CuTe."""
    if device.kind == "cuda":
        route = attention_sol.backend(device.index)
        if not route.startswith("cute_"):
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"sol-attn selects its {route!r} route on {device.name}; only its CuTe routes "
                "are served, which need cutlass.cute and cuda.bindings",
            )


def _sol_dense(
    order: Sequence[str],
    device: DeviceFacts,
    resolved: dict[str, Ready],
    refused: dict[str, str],
) -> Ready:
    """The kernel Sol recomputes dense rows with (its dense steps, protected prefix and dense
    paths): the first one after Sol in the model's preference that serves BF16 at head
    dimension 128 on this rank, else the first ranked one. It runs locally after the Ulysses
    exchange, so it needs no context parallelism. A preferred kernel still compiling is
    recorded in `refused`, so `pending` re-selects when it is ready (run 1565: SDPA served
    Sol's dense steps at 35 s where SageAttention takes about 24 s)."""
    for name in order:
        candidate = BY_NAME.get(name)
        if (
            candidate is None
            or name == "sol-attn"
            or _device_admits(candidate, device)
            or not _admits(candidate, "bfloat16", 128)
        ):
            continue
        if name not in resolved:
            try:
                resolved[name] = _resolve(candidate, device)
            except Exception as exc:
                refused.setdefault(name, f"{type(exc).__name__}: {exc}"[:300])
                continue
        return resolved[name]
    dense = next(
        (entry for entry in available(device) if _admits(entry.candidate, "bfloat16", 128)),
        None,
    )
    if dense is None:
        raise AttentionRefusal("attention_kernel_absent", "Sol has no ready dense reference")
    return dense


def _bind_sol_dense(dense: Ready) -> None:
    attention_sol.bind_dense(_registry()._backends[dense.member], dense.name)


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
    """The ranked table filtered by DEVICE alone: no import, no compile. `available` then
    drops a kernel that is not compiled, does not import or does not launch."""
    return [candidate for candidate in CANDIDATES if not _device_admits(candidate, device)]


def available(device: DeviceFacts, skipped: dict[str, str] | None = None) -> list[Ready]:
    """The ranked kernels this device and environment can actually run, fastest first.

    Never raises: the floor takes everything, so auto-selection has no refusal arm. A ranked
    kernel still compiling, failed or absent here is not in the answer; why is written into
    `skipped`. An environment with no diffusers at all offers nothing: it holds no site.
    """
    ready: list[Ready] = []
    for candidate in for_device(device):
        try:
            ready.append(_resolve(candidate, device))
        except Exception as exc:
            if skipped is not None:
                skipped[candidate.name] = f"{type(exc).__name__}: {exc}"[:300]
            # BROAD ON PURPOSE. Auto-selection has no refusal arm (the floor takes
            # everything), so a kernel that fails to resolve for ANY reason drops out of the
            # ranking rather than failing a prepare that was only trying to go faster. A PIN
            # is the opposite and keeps its typed refusals (`pinned`).
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
        # THE REFUSAL IS THE DISCOVERABILITY: the same sentence says what this card serves.
        raise AttentionRefusal(
            "attention_kernel_unsupported",
            f"{unsupported}. This environment runs {image_kernels(device)} on {device.name}",
        )
    return [_resolve(candidate, device)]


# ------------------------------------------------------------------------------ the pin


def _trees(roots: Mapping[str, object]) -> Iterator[tuple[str, Any]]:
    """Yield each constructed module tree, preserving component labels."""
    seen: set[int] = set()
    queue: deque[tuple[str, object, int]] = deque(
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


def _attention_records(roots: Mapping[str, object]) -> Iterator[tuple[str, str, Any, Any]]:
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


class ClassicSdpa:
    """Diffusers' `AttnProcessor2_0`, the same math, through the per-site backend contract.

    Classic `Attention` sites (SDXL's UNet, AutoencoderKL) call torch SDPA directly, so no
    kernel could be selected, pinned or observed on them. `adopt` gives each one this
    processor. Auto selection keeps SDPA there, bit-identical to before; a pin or a model's
    attention policy may choose another kernel.
    """

    _attention_backend: Any = None
    _parallel_config: Any = None

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        temb: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        shape = hidden_states.shape
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, length = context.shape[:2]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, length, batch)
            attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query, key, value = (x.view(batch, -1, attn.heads, head_dim) for x in (query, key, value))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.flatten(2).to(query.dtype)))
        if len(shape) == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(shape)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def adopt(roots: Mapping[str, object]) -> None:
    """Give every classic `AttnProcessor2_0` site its own `ClassicSdpa`. Idempotent."""
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
    except ImportError:
        return
    for _component, _path, module, processor in list(_attention_records(roots)):
        if type(processor) is AttnProcessor2_0:
            module.set_processor(ClassicSdpa())


def _site_records(roots: Mapping[str, object]) -> Iterator[tuple[str, str, Any, Any]]:
    """Only processors implementing Diffusers' per-site backend selection contract."""
    for record in _attention_records(roots):
        if hasattr(record[3], "_attention_backend"):
            yield record


def _unmanaged_reason(component: str, path: str, processor: Any) -> str:
    return (
        f"{component}/{path} uses {type(processor).__name__}, which has no per-site "
        "attention backend selection"
    )


def _sites(roots: Mapping[str, object]) -> Iterator[tuple[str, Any, Any]]:
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

    peft_linear = getattr(sys.modules.get("peft.tuners.lora.layer"), "Linear", None)
    if peft_linear is not None and type(projection) is peft_linear:
        # PEFT returns the base result's dtype, even with differently typed factors.
        # Inspect the current base: encoded fill can replace it after construction.
        projection = projection.get_base_layer()
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
    #: the ranked kernels this environment and device offered, fastest first.
    offered: tuple[str, ...]
    #: set when the caller PINNED one through the execution-path override (cr-125).
    pin: str = ""
    artifacts: dict[str, dict[str, str]] | None = None
    #: ranked kernels auto-selection could not use here, and why (a compiling one's progress)
    skipped: dict[str, str] | None = None

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kernels in self.hosts.values():
            for name, sites in kernels.items():
                out[name] = out.get(name, 0) + sites
        return out

    def document(self) -> execution_evidence.Boot:
        return _boot(
            self.totals(),
            {name: dict(kernels) for name, kernels in self.hosts.items()},
            list(self.offered),
            self.pin,
            self.artifacts or {},
            self.skipped or {},
        )

    def line(self) -> str:
        """The ONE line a pod prints at boot. It is the whole per-run attention record."""
        return line(self.totals(), self.hosts, list(self.offered), self.pin, self.skipped or {})


def line(
    totals: Mapping[str, int],
    hosts: Mapping[str, Mapping[str, int]],
    offered: Sequence[str],
    pin: str,
    skipped: Collection[str] = (),
) -> str:
    """The boot line, from the facts alone — so a GROUP prepare renders the same sentence from
    its merged facts as a single one does from its own `Applied` (cr-124)."""
    ignored = f"; skipped {sorted(skipped)}" if skipped else ""
    if not totals:
        none = "NONE — this construction holds no backend-selectable diffusers attention site"
        return none + ignored
    chosen = ", ".join(f"{name} x{sites}" for name, sites in sorted(totals.items()))
    where = "; ".join(
        f"{component}: {'+'.join(sorted(kernels))}" for component, kernels in sorted(hosts.items())
    )
    how = f"PINNED {pin}" if pin else f"auto over {list(offered)}"
    return f"{chosen} ({how}) — {where}{ignored}"


def _boot(
    kernels: dict[str, int],
    hosts: dict[str, dict[str, int]],
    offered: list[str],
    pin: str,
    artifacts: Mapping[str, object],
    skipped: dict[str, str],
) -> execution_evidence.Boot:
    # RENDERED HERE, not by the worker: one renderer for the one line, in the module that
    # knows what it means.
    boot = execution_evidence.Boot(
        kernels=kernels,
        hosts=hosts,
        offered=offered,
        pin=pin,
        line=line(kernels, hosts, offered, pin, skipped),
    )
    if artifacts:
        boot["artifacts"] = dict(artifacts)
    if skipped:
        boot["skipped"] = skipped
    return boot


def merge(documents: Mapping[str, object]) -> execution_evidence.Boot | None:
    """A GROUP prepare's one attention fact, from its models' own `Applied.document()`s,
    each decoded here once; a model holding no attention site sends none."""
    hosts: dict[str, dict[str, int]] = {}
    kernels: dict[str, int] = {}
    offered: list[str] = []
    pins: set[str] = set()
    artifacts: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for parameter, document in documents.items():
        if not document:
            continue
        try:
            row = msgspec.convert(document, execution_evidence.Boot, strict=True)
        except msgspec.ValidationError as exc:
            raise AttentionRefusal(
                "attention_fact_malformed", f"{parameter}'s attention fact: {exc}"
            ) from exc
        for component, chosen in row["hosts"].items():
            hosts[f"{parameter}/{component}"] = chosen
        for name, sites in row["kernels"].items():
            kernels[name] = kernels.get(name, 0) + sites
        offered += [name for name in row["offered"] if name not in offered]
        if row["pin"]:
            pins.add(row["pin"])
        if "artifacts" in row:
            artifacts[parameter] = row["artifacts"]
        if "skipped" in row:
            skipped.update(row["skipped"])
    if not kernels:
        return None
    pin = "+".join(sorted(pins))
    return _boot(kernels, hosts, offered, pin, artifacts, skipped)


def _pin_parts(pin: str) -> tuple[str, str]:
    if "=" not in pin:
        return "", pin
    component, name = pin.split("=", 1)
    if not component or not name or "=" in name or any(ch.isspace() for ch in pin):
        raise AttentionRefusal(
            "attention_kernel_unknown", f"invalid attention override {pin!r}; use component=backend"
        )
    return component, name


def reaches(roots: Mapping[str, object], pin: str) -> bool:
    """Whether any attention site here falls in the pin's scope."""
    scope, _ = _pin_parts(pin)
    return any(_matches(component, scope) for component, _, _, _ in _attention_records(roots))


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
    whether it is compiled yet is resolution's question."""
    unsupported = _device_admits(candidate, device)
    if unsupported:
        return unsupported
    if not _admits(candidate, dtype, head_dim):
        return (
            f"{candidate.name} does not serve {dtype or 'unknown'} activations at head "
            f"dimension {head_dim or 'unknown'}"
        )
    try:
        member = _member(candidate)
        if member not in _registry()._backends:
            return f"{candidate.name} is not registered with Diffusers"
        if degree > 1 and not _context_parallel(member):
            return f"{candidate.name} cannot shard sequence-parallel degree {degree}"
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
            # A package written for a newer Runtime may prefer a kernel this one lacks.
            refused[name] = "unknown to this runtime"
            continue
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
            if _device_admits(candidate, device):
                _unsupported(name, reason)
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
    roots: Mapping[str, object],
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
    assignments: list[tuple[str, str, Any, Any, Ready]] = []
    resolved = {entry.name: entry for entry in ready}
    refused: dict[str, str] = dict(skipped or {})
    matched = False
    eligible_sites: dict[tuple[str, int], tuple[str, ...]] = {}
    sol_order: tuple[str, ...] | None = None
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
        if not forced and isinstance(processor, ClassicSdpa):
            offered = [entry for entry in offered if entry.name == "sdpa"] or offered
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
                if entry.name == "sol-attn" and sol_order is None:
                    sol_order = tuple(selection[list(selection).index("sol-attn") + 1 :])
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
    if name and not matched:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{pin!r} matches no attention component"
        )
    if device is not None and any(entry.name == "sol-attn" for *_, entry in assignments):
        _bind_sol_dense(_sol_dense(sol_order or (), device, resolved, refused))
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
    artifacts = {name: entry.artifact for name, entry in resolved.items() if entry.artifact}
    return Applied(hosts, tuple(resolved), pin, artifacts or None)


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


def observed(roots: Mapping[str, object], pin: str = "") -> Applied:
    """Read actual processor choices; never substitute the prepared recommendation."""
    names = {candidate.backend: candidate.name for candidate in BY_NAME.values()}
    hosts: dict[str, dict[str, int]] = {}
    for component, _module, processor in _sites(roots):
        backend = getattr(processor, "_attention_backend", None)
        value = getattr(backend, "value", str(backend))
        name = names.get(value, value)
        hosts.setdefault(component, {})
        hosts[component][name] = hosts[component].get(name, 0) + 1
    served = {name for kernels in hosts.values() for name in kernels}
    artifacts = {
        ready.name: ready.artifact
        for ready in _READY.values()
        if ready.name in served and ready.artifact
    }
    return Applied(hosts, (), pin, artifacts=artifacts or None)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Prepared processor choices, restored without resolution or policy re-execution."""

    choices: tuple[tuple[Any, Any], ...]

    def restore(self) -> None:
        for processor, member in self.choices:
            processor._attention_backend = member


def snapshot(roots: Mapping[str, object]) -> Snapshot:
    return Snapshot(
        tuple((processor, processor._attention_backend) for _, _, processor in _sites(roots))
    )


@contextmanager
def override(
    defaults: Snapshot, roots: Mapping[str, object], swap: Swap | None, pin: str = ""
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


def _baked_in(roots: Mapping[str, object]) -> str:
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
        artifacts = {self.entry.name: self.entry.artifact} if self.entry.artifact else None
        return Applied(hosts, (self.entry.name,), self.pin or self.entry.name, artifacts)


def _is_selected(defaults: Snapshot, roots: Mapping[str, object], pin: str) -> bool:
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


def check(
    device: DeviceFacts, roots: Mapping[str, object], name: str, degree: int = 1
) -> Candidate:
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


def plan(device: DeviceFacts, roots: Mapping[str, object], name: str, degree: int = 1) -> Swap:
    candidate = check(device, roots, name, degree)
    scope, _ = _pin_parts(name)
    entry = pinned(candidate.name, device)[0]
    if entry.name == "sol-attn" and not attention_sol.dense_identity():
        _bind_sol_dense(_sol_dense((), device, {}, {}))
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
    roots: Mapping[str, object],
    pin: str = "",
    degree: int = 1,
    *,
    choose: Callable[[AttentionContext], str | Sequence[str] | None] | None = None,
) -> Applied:
    """Select once at preparation; optional model policy and request pins are validated."""
    adopt(roots)
    scope, name = _pin_parts(pin)
    skipped: dict[str, str] = {}
    ready = pinned(name, device) if name and not scope else available(device, skipped)
    if scope:
        ready = [entry for entry in ready if entry.name != name] + pinned(name, device)
    applied = install(ready, roots, pin, degree, device=device, choose=choose, skipped=skipped)
    return replace(applied, skipped=skipped or None)


def image_kernels(device: DeviceFacts) -> str:
    """What this device serves now, for a refusal message (cr-125): every kernel already
    ready here, and each one still compiling with its progress. Starts no compile."""
    ready = sorted(name for name, held in _READY if held == device)
    busy = [f"{n} {s.line()}" for n, s in sorted(_STATES.items()) if s.state == "compiling"]
    return ", ".join(ready + busy) or "none"


def implementations(seen: Mapping[str, set[str]]) -> str:
    """The implementation line for what served in an `attention_ulysses.observing` scope."""
    names = {candidate.backend: candidate.name for candidate in BY_NAME.values()}
    return attention_ulysses.describe(dict(seen), names)
