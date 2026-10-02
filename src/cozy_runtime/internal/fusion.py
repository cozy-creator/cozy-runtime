"""Fused glue kernels (h3a-015): the elementwise chains a DiT block runs between its GEMMs.

Model-agnostic: every kernel is one pass over rows of a `[rows, width]` bf16 activation and
knows nothing about the module that calls it. `fusion_install.py` owns the map from
diffusers modules to these calls. Shapes are the ones the eager composition already
produces, so nothing here moves a parameter, a buffer or a state-dict key.

Numerics are the EAGER composition's, not "better": every intermediate that eager rounds
to bf16 is rounded here too (`_rb`, in the integer domain so the compiler cannot fold the
round-trip away), and the fp8 epilogue is the leaf's own quantizer (`leaves.py`) run on the
same registers. Measured on sm89 against torch 2.13 eager (`tests/test_fusion.py`): the
modulate, gate-add, SwiGLU and RoPE chains are bit-identical; the RMSNorm itself differs
from torch's fused kernel by one bf16 ulp on ~4e-6 of elements (fp32 reduction order of
the row variance), and those elements are the whole distance between fused and eager.

The `@triton.jit` functions below are the kernel SOURCES. They compile on the machine that runs
them, at worker boot, in the compile manager's niced background builder (`build`), into
Triton's own keyed cache (`TRITON_CACHE_DIR`, inside the machine's kernel store); the store's
entry for `job(arch)` marks that cache filled. `load` serves a READY build or raises typed:
`fusion_kernels_compiling`, `fusion_kernels_failed`, `fusion_kernels_absent` (no store). A
request never compiles. The key is Triton's own hash of every entry (sources, signature,
architecture, options, Triton) plus its launcher module's and the cache location, so a LoRA,
a component swap or a step count cannot invalidate it.

Adapted from NVlabs/Sana `sol-engine` `models/minimax_h3/{GB200,GB10}/fusions.py`
(Apache-2.0): the per-row program shape, the table-row gather inside the kernel, the
partner-channel RoPE and the integer-domain bf16 rounding are theirs.

This module is EXECUTOR-ONLY: torch and Triton are imported at the top, guarded, and a
missing Triton is a typed `FusionUnavailable` at first use, never a silent fallback. The
torch-free runtime package never imports it (`checks/architecture.py::torch-free-import`).
"""

from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import kernel_cache, kernel_compile
from cozy_runtime.internal.accel import readable
from cozy_runtime.internal.encoding.leaves import PreQuantized

try:
    import torch
except ImportError as _absent:  # pragma: no cover - the executor always has torch
    TORCH_ABSENT = f"torch does not import ({_absent})"
else:
    TORCH_ABSENT = ""

try:
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import driver as _cuda_driver
    from triton.compiler import ASTSource
    from triton.compiler.compiler import (
        CompiledKernel,
        get_cache_invalidating_env_vars,
        make_backend,
    )
    from triton.language.extra import libdevice
    from triton.runtime.build import _get_cache_manager
    from triton.runtime.cache import get_cache_key, get_cache_manager
except ImportError as _absent:  # pragma: no cover - exercised by the typed refusal arm
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

WARP_SIZE = 32

_ROW_WARPS = 16
_ROPE_HEADS = 8
_ROPE_WARPS = 4
_SWIGLU_BLOCK = 1024
_SWIGLU_WARPS = 8

#: The width profiles compiled. The constexprs in a spec are the served widths, so a model
#: with other widths needs its own row here.
PROFILES: dict[str, dict[str, int]] = {
    "minimax-h3": {"hidden": 5376, "ffn": 14336, "head_dim": 128, "rotary": 96},
}


class FusionUnavailable(RuntimeError):
    """Triton or this device's compiled kernels are not ready. Typed, never a fallback."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _require_triton() -> None:
    if TRITON_ABSENT:
        raise FusionUnavailable("fusion_triton_absent", TRITON_ABSENT)


def _require_torch() -> None:
    if TORCH_ABSENT:
        raise FusionUnavailable("fusion_triton_absent", TORCH_ABSENT)


@dataclass(frozen=True, slots=True)
class Spec:
    """One compiled entry: a kernel with its constexpr signature and launch options."""

    kernel: str
    signature: tuple[tuple[str, str], ...]
    constexprs: tuple[tuple[str, int], ...]
    aligned: tuple[str, ...]
    num_warps: int

    @property
    def key(self) -> str:
        return f"{self.kernel}[" + ",".join(f"{k}={v}" for k, v in self.constexprs) + "]"


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def specs(*, hidden: int, ffn: int, head_dim: int, rotary: int) -> tuple[Spec, ...]:
    """Every entry one width profile needs. Row counts never enter a spec: they size grids."""
    block = _next_pow2(hidden)
    norm_pointers = ("x_ptr", "branch_ptr", "gate_ptr", "hidden_ptr", "out_ptr", "scale_out_ptr")
    tables = ("weight_ptr", "scale_ptr", "shift_ptr", "index_ptr")
    out: list[Spec] = []
    for residual in (0, 1):
        for fp8 in (0, 1):
            out.append(
                Spec(
                    "norm_mod",
                    (
                        ("x_ptr", "*bf16"),
                        ("branch_ptr", "*bf16"),
                        ("gate_ptr", "*bf16"),
                        ("hidden_ptr", "*bf16"),
                        ("out_ptr", "*fp8e4nv" if fp8 else "*bf16"),
                        ("scale_out_ptr", "*fp32" if fp8 else "*bf16"),
                        ("weight_ptr", "*bf16"),
                        ("scale_ptr", "*bf16"),
                        ("shift_ptr", "*bf16"),
                        ("index_ptr", "*i64"),
                        ("n_cols", "i32"),
                        ("n_index", "i32"),
                        ("stride_x", "i32"),
                        ("stride_table", "i32"),
                        ("eps", "fp32"),
                        ("RESIDUAL", "constexpr"),
                        ("FP8", "constexpr"),
                        ("BLOCK", "constexpr"),
                    ),
                    (("RESIDUAL", residual), ("FP8", fp8), ("BLOCK", block)),
                    (*norm_pointers, *tables, "n_cols", "stride_x", "stride_table"),
                    _ROW_WARPS,
                )
            )
    out.append(
        Spec(
            "gate_add",
            (
                ("x_ptr", "*bf16"),
                ("branch_ptr", "*bf16"),
                ("gate_ptr", "*bf16"),
                ("out_ptr", "*bf16"),
                ("index_ptr", "*i64"),
                ("n_cols", "i32"),
                ("n_index", "i32"),
                ("stride_x", "i32"),
                ("stride_table", "i32"),
                ("BLOCK", "constexpr"),
            ),
            (("BLOCK", block),),
            (
                "x_ptr",
                "branch_ptr",
                "gate_ptr",
                "out_ptr",
                "index_ptr",
                "n_cols",
                "stride_x",
                "stride_table",
            ),
            _ROW_WARPS,
        )
    )
    for fp8 in (0, 1):
        out.append(
            Spec(
                "swiglu",
                (
                    ("x_ptr", "*bf16"),
                    ("out_ptr", "*fp8e4nv" if fp8 else "*bf16"),
                    ("scale_out_ptr", "*fp32" if fp8 else "*bf16"),
                    ("half", "i32"),
                    ("stride_x", "i32"),
                    ("stride_out", "i32"),
                    ("FP8", "constexpr"),
                    ("BLOCK", "constexpr"),
                ),
                (("FP8", fp8), ("BLOCK", _SWIGLU_BLOCK)),
                ("x_ptr", "out_ptr", "scale_out_ptr", "half", "stride_x", "stride_out"),
                _SWIGLU_WARPS,
            )
        )
    out.append(
        Spec(
            "rope_norm",
            (
                ("x_ptr", "*bf16"),
                ("out_ptr", "*bf16"),
                ("weight_ptr", "*bf16"),
                ("cos_ptr", "*fp32"),
                ("sin_ptr", "*fp32"),
                ("seq", "i32"),
                ("heads", "i32"),
                ("eps", "fp32"),
                ("HEAD_DIM", "constexpr"),
                ("ROTARY", "constexpr"),
                ("HEADS", "constexpr"),
            ),
            (("HEAD_DIM", head_dim), ("ROTARY", rotary), ("HEADS", _ROPE_HEADS)),
            ("x_ptr", "out_ptr", "weight_ptr", "cos_ptr", "sin_ptr"),
            _ROPE_WARPS,
        )
    )
    del ffn  # the SwiGLU row is chunked; its width sizes the grid, never the binary
    return tuple(out)


# ------------------------------------------------------------------- kernel sources

if not TRITON_ABSENT:

    @triton.jit
    def _rb(x):  # type: ignore[no-untyped-def]
        # fp32 -> bf16 -> fp32 round-to-nearest-even in the integer domain: the compiler
        # folds `x.to(bf16).to(f32)` to identity and would keep fp32 precision eager lacks.
        bits = x.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        return (bits & -65536).to(tl.float32, bitcast=True)

    @triton.jit
    def _e4m3(value, scale):  # type: ignore[no-untyped-def]
        # Exactly `leaves._triton_rowwise_kernel`'s cast: IEEE divide, clamp, RTNE.
        return tl.clamp(tl.div_rn(value, scale), -448.0, 448.0).to(
            tl.float8e4nv, fp_downcast_rounding="rtne"
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def _norm_mod_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        branch_ptr,
        gate_ptr,
        hidden_ptr,
        out_ptr,
        scale_out_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        n_cols,
        n_index,
        stride_x,
        stride_table,
        eps,
        RESIDUAL: tl.constexpr,
        FP8: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * stride_x + cols
        # One table row per SEQUENCE row; a batch axis wraps (eager broadcasts the index).
        table = tl.load(index_ptr + row % n_index).to(tl.int64) * stride_table + cols
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if RESIDUAL:
            branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(gate_ptr + table, mask=mask, other=0.0).to(tl.float32)
            x = _rb(x + _rb(gate * branch))
            tl.store(hidden_ptr + offset, x.to(tl.bfloat16), mask=mask)
        variance = tl.sum(x * x, axis=0) / n_cols
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = _rb(x * tl.math.rsqrt(variance + eps) * weight)
        scale = tl.load(scale_ptr + table, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table, mask=mask, other=0.0).to(tl.float32)
        out = _rb(_rb(normed * _rb(1.0 + scale)) + shift)
        if FP8:
            amax = tl.max(tl.abs(out), axis=0)
            row_scale = tl.maximum(amax / 448.0, 1e-12)
            tl.store(scale_out_ptr + row, row_scale)
            tl.store(out_ptr + offset, _e4m3(out, row_scale), mask=mask)
        else:
            tl.store(out_ptr + offset, out.to(tl.bfloat16), mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _gate_add_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        branch_ptr,
        gate_ptr,
        out_ptr,
        index_ptr,
        n_cols,
        n_index,
        stride_x,
        stride_table,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * stride_x + cols
        table = tl.load(index_ptr + row % n_index).to(tl.int64) * stride_table + cols
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offset, _rb(x + _rb(gate * branch)).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _silu_product(x_ptr, base, cols, mask, half):  # type: ignore[no-untyped-def]
        value = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + base + half + cols, mask=mask, other=0.0).to(tl.float32)
        # torch's silu: `x / (1 + exp(-x))` in fp32 with IEEE divide and an exact exp.
        activated = _rb(tl.div_rn(gate, 1.0 + libdevice.exp(-gate)))
        return _rb(value * activated)

    @triton.jit  # type: ignore[untyped-decorator]
    def _swiglu_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        out_ptr,
        scale_out_ptr,
        half,
        stride_x,
        stride_out,
        FP8: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One row per program, chunked: a 14336-wide product held twice would spill. The
        # fp8 epilogue needs the row amax first, so it recomputes the chunk on a second
        # sweep from a row that is by then L2-resident.
        row = tl.program_id(0).to(tl.int64)
        base = row * stride_x
        offsets = tl.arange(0, BLOCK)
        if FP8:
            amax = tl.zeros((BLOCK,), dtype=tl.float32)
            for start in range(0, half, BLOCK):
                cols = start + offsets
                amax = tl.maximum(amax, tl.abs(_silu_product(x_ptr, base, cols, cols < half, half)))
            row_scale = tl.maximum(tl.max(amax, axis=0) / 448.0, 1e-12)
            tl.store(scale_out_ptr + row, row_scale)
            for start in range(0, half, BLOCK):
                cols = start + offsets
                product = _silu_product(x_ptr, base, cols, cols < half, half)
                tl.store(
                    out_ptr + row * stride_out + cols, _e4m3(product, row_scale), mask=cols < half
                )
        else:
            for start in range(0, half, BLOCK):
                cols = start + offsets
                product = _silu_product(x_ptr, base, cols, cols < half, half)
                tl.store(
                    out_ptr + row * stride_out + cols, product.to(tl.bfloat16), mask=cols < half
                )

    @triton.jit  # type: ignore[untyped-decorator]
    def _rope_norm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        out_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        seq,
        heads,
        eps,
        HEAD_DIM: tl.constexpr,
        ROTARY: tl.constexpr,
        HEADS: tl.constexpr,
    ):
        # One program per (row, HEADS heads): the RMS is per head, the rotation pairs
        # channel j with j +/- ROTARY/2 inside the rotary span, and cos/sin are per
        # sequence position (a batch axis wraps).
        row = tl.program_id(0).to(tl.int64)
        head0 = tl.program_id(1) * HEADS
        token = row % seq
        head = head0 + tl.arange(0, HEADS)[:, None]
        chan = tl.arange(0, HEAD_DIM)[None, :]
        mask = head < heads
        base = (row * heads + head) * HEAD_DIM
        x = tl.load(x_ptr + base + chan, mask=mask, other=0.0).to(tl.float32)
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1)[:, None] / HEAD_DIM + eps)
        weight = tl.load(weight_ptr + chan).to(tl.float32)
        normed = _rb(x * inv * weight)
        half = ROTARY // 2
        rotating = chan < ROTARY
        low = chan < half
        partner = tl.where(low, chan + half, chan - half)
        paired = tl.load(x_ptr + base + partner, mask=mask & rotating, other=0.0).to(tl.float32)
        paired_weight = tl.load(weight_ptr + partner, mask=rotating, other=0.0).to(tl.float32)
        rotated = _rb(paired * inv * paired_weight)
        rotated = tl.where(low, -rotated, rotated)
        # Eager casts the fp32 tables to bf16 BEFORE multiplying.
        cos = _rb(tl.load(cos_ptr + token * ROTARY + chan, mask=rotating, other=1.0))
        sin = _rb(tl.load(sin_ptr + token * ROTARY + chan, mask=rotating, other=0.0))
        turned = _rb(_rb(normed * cos) + _rb(rotated * sin))
        out = tl.where(rotating, turned, normed)
        tl.store(out_ptr + base + chan, out.to(tl.bfloat16), mask=mask)

    _KERNELS: dict[str, Any] = {
        "norm_mod": _norm_mod_kernel,
        "gate_add": _gate_add_kernel,
        "swiglu": _swiglu_kernel,
        "rope_norm": _rope_norm_kernel,
    }
    _SOURCES: tuple[Any, ...] = (
        _rb,
        _e4m3,
        _silu_product,
        _norm_mod_kernel,
        _gate_add_kernel,
        _swiglu_kernel,
        _rope_norm_kernel,
    )
else:  # pragma: no cover
    _KERNELS = {}
    _SOURCES = ()


def source_digest() -> str:
    """The kernel sources' identity, as the fusion plan's identity records it."""
    _require_triton()
    text = "\n".join(inspect.getsource(fn.fn) for fn in _SOURCES)
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------ compile


def _all_specs() -> tuple[Spec, ...]:
    return tuple(spec for profile in PROFILES.values() for spec in specs(**profile))


def ast_source(spec: Spec) -> Any:
    """The exact Triton `ASTSource` a spec compiles from: builder and loader share it."""
    _require_triton()
    fn = _KERNELS[spec.kernel]
    attrs = {(fn.arg_names.index(name),): [["tt.divisibility", 16]] for name in spec.aligned}
    return ASTSource(fn, dict(spec.signature), constexprs=dict(spec.constexprs), attrs=attrs)


def _backend(arch: int) -> Any:
    """One compiler backend per architecture: constructing one runs `ptxas --version`."""
    backends: dict[int, Any] = getattr(_backend, "_cached", {})
    if arch not in backends:
        backends[arch] = make_backend(GPUTarget("cuda", arch, WARP_SIZE))
        _backend._cached = backends  # type: ignore[attr-defined]
    return backends[arch]


def cache_key(spec: Spec, arch: int) -> str:
    """Triton's own cache hash for (sources, signature, architecture, options, Triton)."""
    backend = _backend(arch)
    options = backend.parse_options({"num_warps": spec.num_warps})
    key = get_cache_key(ast_source(spec), backend, options, get_cache_invalidating_env_vars())
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _launcher_key() -> str:
    """Triton's cache key for its launcher module (`CudaUtils`) on this host, its own way."""
    source = (Path(_cuda_driver.__file__).parent / "driver.c").read_bytes()
    options = {
        "language": ["c"],
        "library_dirs": _cuda_driver.library_dirs(),
        "include_dirs": _cuda_driver.include_dirs,
        "libraries": _cuda_driver.libraries,
        "ccflags": None,
    }
    return str(_get_cache_manager(source, config=options).key)


def job(arch: int) -> kernel_compile.Job:
    """Every profile's kernels for `arch`, compiled into this process's Triton cache."""
    _require_triton()
    entries = {spec.key: cache_key(spec, arch) for spec in _all_specs()}
    inputs = {
        "arch": arch,
        "triton": triton.__version__,
        "entries": entries,
        "launcher": _launcher_key(),
        "cache": triton.knobs.cache.dir,
    }
    key = kernel_cache.Key.of("fusion", inputs)
    return kernel_compile.Job("fusion", key, {"arch": arch, "entries": entries})


def build(
    spec: dict[str, Any], staging: Path, report: Callable[[float], None]
) -> dict[str, object]:
    """The compile manager's builder (`BUILDERS["fusion"]`): every entry into Triton's own
    cache, then Triton's launcher module, so a launch compiles nothing. `staging` stays empty:
    the published entry marks the cache filled."""
    _require_triton()
    arch, entries, todo = int(spec["arch"]), spec["entries"], _all_specs()
    for done, one in enumerate(todo, 1):
        compiled = triton.compile(
            ast_source(one),
            target=GPUTarget("cuda", arch, WARP_SIZE),
            options={"num_warps": one.num_warps},
        )
        if compiled.hash != entries[one.key]:
            raise kernel_compile.CompileFailed(
                "compile_key_mismatch",
                f"{one.key} compiled under {compiled.hash[:12]}, keyed {entries[one.key][:12]}",
            )
        report(done / (len(todo) + 1))
    _cuda_driver.CudaUtils()
    return {"entries": len(todo)}


# ------------------------------------------------------------------------- loader


@dataclass(frozen=True, slots=True)
class _Launch:
    """A compiled kernel's launch. Its tensor operands are read by pointer, so each has real
    bytes behind it first (`accel.readable`): a norm scale read from a sibling module may be
    a weight the plane has not mapped."""

    kernel: Mapping[tuple[int, ...], Callable[..., object]]

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        launch = self.kernel[grid]

        def run(*args: object) -> None:
            at = [i for i, value in enumerate(args) if isinstance(value, torch.Tensor)]
            with readable(*(args[i] for i in at)) as real:
                operands = list(args)
                for i, value in zip(at, real, strict=True):
                    operands[i] = value
                launch(*operands)

        return run


@dataclass(frozen=True, slots=True)
class _Loaded:
    arch: int
    compiled: dict[str, Any]

    def launch(self, kernel: str, **constexprs: int) -> _Launch:
        key = f"{kernel}[" + ",".join(f"{k}={v}" for k, v in constexprs.items()) + "]"
        try:
            return _Launch(self.compiled[key])
        except KeyError:
            raise FusionUnavailable(
                "fusion_kernels_absent",
                f"{key} is not compiled for sm{self.arch}: no width profile covers these widths",
            ) from None


#: Loaded kernels by device index.
_LOADED: dict[int, _Loaded] = {}


def _store() -> kernel_cache.Store:
    store = kernel_cache.configured()
    if store is None:
        raise FusionUnavailable(
            "fusion_kernels_absent", "no machine kernel store: this executor has no rooted worker"
        )
    return store


def _job(device: Any) -> tuple[kernel_cache.Store, kernel_compile.Job]:
    _require_torch()
    major, minor = torch.cuda.get_device_capability(device)
    return _store(), job(major * 10 + minor)


def expect(arch: int) -> str:
    """Submit `arch`'s compile without waiting, before any device exists: the state's line.
    Keyed in the process that will load it (its Triton and cache directory), so an executor
    submits its own at start."""
    try:
        return kernel_compile.submit(_store(), job(arch)).line()
    except FusionUnavailable as exc:
        return f"{exc.code}: {exc}"


def load(device: Any, *, wait: bool = False) -> _Loaded:
    """This device's compiled kernels, loaded onto it from Triton's cache; never a compile.

    Not ready is typed. `wait` is construction under `fusion="require"`, never a request: it
    first waits on the live build of the key. Idempotent per (process, device)."""
    _require_triton()
    _require_torch()
    index = torch.device(device).index or 0
    if index in _LOADED:
        return _LOADED[index]
    store, found = _job(index)
    state = kernel_compile.submit(store, found)
    if wait and state.state == "compiling":
        with store.building(found.key):
            pass
        state = kernel_compile.status(store, found)
    arch = int(found.spec["arch"])
    if state.state != "ready":
        # After the wait, anything but ready is a build that ended without publishing.
        failed = wait or state.state == "failed"
        code = "fusion_kernels_failed" if failed else "fusion_kernels_compiling"
        raise FusionUnavailable(code, f"sm{arch} fused kernels: {state.line()}")
    compiled: dict[str, Any] = {}
    for spec in _all_specs():
        digest = found.spec["entries"][spec.key]
        name = f"{_KERNELS[spec.kernel].__name__}.json"
        group = get_cache_manager(digest).get_group(name) or {}
        if name not in group:
            raise FusionUnavailable(
                "fusion_kernels_corrupt",
                f"{spec.key}: the store marks sm{arch} built, Triton's cache lacks {digest[:12]}",
            )
        kernel = CompiledKernel(ast_source(spec), group, digest)
        try:
            kernel._init_handles()
        except Exception as exc:
            raise FusionUnavailable(
                "fusion_kernels_corrupt", f"{spec.key} does not load on sm{arch}: {exc}"
            ) from exc
        compiled[spec.key] = kernel
    loaded = _LOADED[index] = _Loaded(arch, compiled)
    return loaded


# ---------------------------------------------------------------------- wrappers


def _rows(x: Any, width: int) -> Any:
    flat = x.reshape(-1, width)
    return flat if flat.stride(-1) == 1 else flat.contiguous()


def _table(t: Any) -> Any:
    # `chunk(6, dim=-1)` views have a 6*hidden row stride and a unit column stride; that is
    # addressable as `row * stride(0) + col`, so no copy — anything else is made contiguous.
    return t if t.stride(-1) == 1 else t.contiguous()


def _require(x: Any) -> _Loaded:
    if x.dtype is not torch.bfloat16 or not x.is_cuda:
        raise FusionUnavailable(
            "fusion_block_shape",
            f"fused glue serves bfloat16 CUDA activations, got {x.dtype} on {x.device}",
        )
    return load(x.device)


def _launch_norm_mod(
    x: Any,
    branch: Any,
    gate: Any,
    weight: Any,
    scale: Any,
    shift: Any,
    index: Any,
    eps: float,
    fp8: bool,
) -> tuple[Any, Any, Any]:
    loaded = _require(x)
    width = int(x.shape[-1])
    rows = _rows(x, width)
    n = int(rows.shape[0])
    hidden = torch.empty_like(rows) if branch is not None else rows
    branch_rows = _rows(branch, width) if branch is not None else rows
    gate = _table(gate) if gate is not None else scale
    scale, shift = _table(scale), _table(shift)
    if fp8:
        out = torch.empty(n, width, dtype=torch.float8_e4m3fn, device=x.device)
        row_scale = torch.empty(n, 1, dtype=torch.float32, device=x.device)
    else:
        out = torch.empty_like(rows)
        row_scale = out
    kernel = loaded.launch(
        "norm_mod", RESIDUAL=int(branch is not None), FP8=int(fp8), BLOCK=_next_pow2(width)
    )
    kernel[(n, 1, 1)](
        rows,
        branch_rows,
        gate,
        hidden,
        out,
        row_scale,
        weight,
        scale,
        shift,
        index,
        width,
        index.numel(),
        rows.stride(0),
        scale.stride(0),
        float(eps),
        int(branch is not None),
        int(fp8),
        _next_pow2(width),
    )
    return hidden, out, row_scale


def rmsnorm_modulate(
    x: Any, weight: Any, scale: Any, shift: Any, index: Any, eps: float, *, fp8: bool = False
) -> Any:
    """`rmsnorm(x) * (1 + scale[index]) + shift[index]`, one read and one write.

    With `fp8`, the write is the rowwise leaf's operand (E4M3 payload + per-row scale)
    instead of the bf16 tensor the leaf would have read back to quantize.
    """
    _, out, row_scale = _launch_norm_mod(x, None, None, weight, scale, shift, index, eps, fp8)
    if fp8:
        return PreQuantized(out, row_scale, tuple(x.shape))
    return out.view_as(x)


def gate_add_rmsnorm_modulate(
    residual: Any,
    gate: Any,
    branch: Any,
    index: Any,
    weight: Any,
    scale: Any,
    shift: Any,
    eps: float,
    *,
    fp8: bool = False,
) -> tuple[Any, Any]:
    """`hidden = residual + gate[index] * branch`, then `rmsnorm_modulate(hidden)`.

    Returns `(hidden, normed)`: the new residual stream and the next GEMM's input, from
    one pass over the registers that already hold the row.
    """
    hidden, out, row_scale = _launch_norm_mod(
        residual, branch, gate, weight, scale, shift, index, eps, fp8
    )
    normed = PreQuantized(out, row_scale, tuple(residual.shape)) if fp8 else out.view_as(residual)
    return hidden.view_as(residual), normed


def gate_add(residual: Any, gate: Any, branch: Any, index: Any) -> Any:
    """`residual + gate[index] * branch`, one pass (eager: a gather, a mul and an add)."""
    loaded = _require(residual)
    width = int(residual.shape[-1])
    rows, branch_rows = _rows(residual, width), _rows(branch, width)
    gate = _table(gate)
    out = torch.empty_like(rows)
    block = _next_pow2(width)
    loaded.launch("gate_add", BLOCK=block)[(int(rows.shape[0]), 1, 1)](
        rows,
        branch_rows,
        gate,
        out,
        index,
        width,
        index.numel(),
        rows.stride(0),
        gate.stride(0),
        block,
    )
    return out.view_as(residual)


def swiglu(x: Any, *, fp8: bool = False) -> Any:
    """`value * silu(gate)` over a `[..., 2F]` projection: one read of 2F, one write of F."""
    loaded = _require(x)
    width = int(x.shape[-1])
    half = width // 2
    rows = _rows(x, width)
    n = int(rows.shape[0])
    if fp8:
        out = torch.empty(n, half, dtype=torch.float8_e4m3fn, device=x.device)
        row_scale = torch.empty(n, 1, dtype=torch.float32, device=x.device)
    else:
        out = torch.empty(n, half, dtype=x.dtype, device=x.device)
        row_scale = out
    loaded.launch("swiglu", FP8=int(fp8), BLOCK=_SWIGLU_BLOCK)[(n, 1, 1)](
        rows, out, row_scale, half, rows.stride(0), half, int(fp8), _SWIGLU_BLOCK
    )
    if fp8:
        return PreQuantized(out, row_scale, (*x.shape[:-1], half))
    return out.view(*x.shape[:-1], half)


def rmsnorm_rope(x: Any, weight: Any, cos: Any, sin: Any, eps: float) -> Any:
    """Per-head RMSNorm then partial rotary on `(batch, seq, heads, head_dim)`.

    `cos`/`sin` are `(seq, rotary_dim)`; channels past `rotary_dim` pass through.
    """
    loaded = _require(x)
    batch, seq, heads, head_dim = (int(d) for d in x.shape)
    rotary = int(cos.shape[-1])
    flat = x.reshape(batch * seq, heads * head_dim)
    flat = flat if flat.stride(-1) == 1 else flat.contiguous()
    out = torch.empty_like(flat)
    grid = (batch * seq, -(-heads // _ROPE_HEADS), 1)
    loaded.launch("rope_norm", HEAD_DIM=head_dim, ROTARY=rotary, HEADS=_ROPE_HEADS)[grid](
        flat,
        out,
        weight,
        cos.contiguous(),
        sin.contiguous(),
        seq,
        heads,
        float(eps),
        head_dim,
        rotary,
        _ROPE_HEADS,
    )
    return out.view(batch, seq, heads, head_dim)


def warm(
    device: Any,
    *,
    hidden: int,
    ffn: int,
    heads: int,
    head_dim: int,
    rotary: int,
    fp8: bool,
) -> float:
    """Load and launch every entry the served widths need, on two rows; nothing compiles.
    Returns the milliseconds the loads and first launches took."""
    started = time.perf_counter()
    load(device)
    rows, table = 2, 3
    x = torch.randn(1, rows, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    tables = torch.randn(table, 6 * hidden, device=device, dtype=torch.bfloat16).chunk(6, dim=-1)
    index = torch.arange(rows, device=device) % table
    for quant in {False, fp8}:
        rmsnorm_modulate(x, weight, tables[1], tables[0], index, 1e-5, fp8=quant)
        gate_add_rmsnorm_modulate(
            x, tables[2], x, index, weight, tables[4], tables[3], 1e-5, fp8=quant
        )
        swiglu(torch.randn(1, rows, 2 * ffn, device=device, dtype=torch.bfloat16), fp8=quant)
    gate_add(x, tables[5], x, index)
    q = torch.randn(1, rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    angles = torch.randn(rows, rotary, device=device)
    rmsnorm_rope(q, torch.ones(head_dim, device=device, dtype=torch.bfloat16), angles, angles, 1e-5)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000
