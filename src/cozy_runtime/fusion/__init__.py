"""Fused DiT blocks: a package declares its block's layout, Runtime owns the kernels and the
substitution.

A sequential adaLN block (two norms) runs, per token row:

    normed = norm1(x) * (1 + scale_msa) [+ shift_msa]
    x      = x + act(gate_msa) * attention(normed)       # q/k: per-head RMSNorm, then RoPE
    normed = norm2(x) * (1 + scale_mlp) [+ shift_mlp]
    x      = x + act(gate_mlp) * down(silu(gate(normed)) * up(normed))

A parallel block (one norm, as Flux's single-stream blocks) reads one modulated norm twice:

    normed = norm(x) * (1 + scale_msa) [+ shift_msa]
    x      = x + act(gate_msa) * (attention(normed) + down(silu(gate(normed)) * up(normed)))

Without `scale` the norms are plain and the gates are layer scales, one row for every token (a
ViT block, `x + attention(norm1(x)) * scale1`). Rows are bf16, or fp16 under autocast over an
fp32 residual stream (`rows`).

Eager, each line is several kernels; fused, one row kernel per line and one per q/k carry it,
and every GEMM stays the model's own module. Declare the layout at module scope, so its kernels
compile on the executor's GPUs at boot, and install it in `Model.execution_plan`:

    BLOCK = fusion.DiTBlock(MyBlock, width=4096, head_dim=128, modulation=rows, ...)

    def execution_plan(self, plan):
        return fusion.install(plan, BLOCK, self.transformer)

A construction never waits for the compile: until it is ready the eager blocks serve, and the
fused ones from the next request on. A block whose structure, widths or dtype differ from the
declaration serves eager (reported).

A VAE's channel-first RMS norm (Wan's `RMS_norm` and the Qwen-Image copies of it) is one kernel,
and with the SiLU after it where the package names that pair (`ChannelNorm`, `install_norms`).
A video VAE decoding a lone image frame skips its causal convolutions' zero taps and padded
copies, and upsamples in its own dtype (`LoneFrame`, `install_lone_frame`).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from cozy_runtime.author.kernels import (
    KernelsUnavailable,
    Launch,
    Plan,
    TritonKernels,
    int8_rows,
    quantize_once,
)
from cozy_runtime.fusion import _kernels

try:
    import torch
except ImportError:  # pragma: no cover - the base package is torch-free
    torch = None

try:
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry, dispatch_attention_fn
except ImportError:  # pragma: no cover - only a parallel block composes Diffusers' attention
    _AttentionBackendRegistry = dispatch_attention_fn = None


@dataclass(frozen=True, slots=True)
class Modulation:
    """One block call's adaLN tables: `[rows, width]` views (any row stride) and each token's
    row, `index` (int64 over `tokens` or `batch * tokens` rows; it wraps). A parallel block
    reads only the `_msa` tables; a block without `scale` only the gates."""

    index: Any
    scale_msa: Any = None
    gate_msa: Any = None
    scale_mlp: Any = None
    gate_mlp: Any = None
    shift_msa: Any = None
    shift_mlp: Any = None


@dataclass(frozen=True, slots=True)
class Rope:
    """The block's rotary argument. Its q/k norms apply it, so a sequential block's attention
    receives None and must skip RoPE (it still applies the norms). `interleaved`: pairs (2i,
    2i+1), a complex `[seq, rotary/2]` table or a `(cos, sin)` pair; else halves, a `(cos, sin)`
    `[seq, rotary]` pair. `rotary`: rotated channels per head, 0 for all. `rounding`: how the
    model rounds q/k to the row dtype. `"sum"`: the norm, then `x * cos + rotated * sin` once,
    fp32 tables; `"product"`: the norm, the tables, each product and their sum, as eager rotary
    in the row dtype computes it (`cos.to(x.dtype)`); `"fp32"`: once, at the end (Flux)."""

    argument: str
    interleaved: bool
    rotary: int = 0
    rounding: Literal["sum", "product", "fp32"] = "sum"


@dataclass(frozen=True, slots=True)
class GatedMLP:
    """`down(silu(gate(x)) * up(x))`: submodule names, the linears relative to `module`. Without
    `up`, `gate`'s output holds the gate half, then the up half (one fused linear), or with
    `value_first` the up half first (Diffusers' `SwiGLU`: `hidden, gate = proj(x).chunk(2)`)."""

    module: str
    gate: str
    down: str
    up: str | None = None
    value_first: bool = False


@dataclass(frozen=True)
class DiTBlock:
    """A package's single-stream adaLN block, as Runtime fuses it.

    `modulation(block, arguments)` returns the call's `Modulation` from the block forward's
    bound arguments. `norms` are the pre-attention and pre-MLP norms, `"layer"` (no affine)
    or `"rms"` (weight); one norm makes the block parallel. `rms_rounding`: Diffusers'
    RMSNorm rounds the normalized row, then its product with the weight (`"twice"`);
    `torch.nn.RMSNorm` rounds once. A sequential block's `attention` is called with
    `arguments` (keywords taken from the block's) after the normed rows; a parallel block's is
    composed by Runtime from the module's `to_q`/`to_k`/`to_v`/`to_out` and the kernel Runtime
    selected for it (Diffusers' attention contract), so its processor is not called.
    With `compose`, a sequential block's attention is composed that way too: its processor's
    casts and copies around the q/k norms are skipped. `qk_norms` (relative to the attention)
    are per-head RMSNorms, with a weight unless not `qk_affine`.
    `gate`: the activation applied to the gates. `scale`: the norms are modulated by
    `1 + scale` [`+ shift`]. `weights`: the norm and q/k norm weights' dtype, the rows' wide one or
    `"fp32"` (kept in fp32, Diffusers' `_keep_in_fp32_modules`); `centered`: they are stored
    zero-centered, `1 + weight` multiplies (Krea 2's, Gemma's). `rows`: `"bf16"`, each step
    rounding as eager bf16 does, or
    `"fp16"`, a block under fp16 autocast over an fp32 residual stream (fp32 norm weights and
    gates; the residual's `x + branch * gate` in fp32; a norm rounds once to fp16).
    """

    block: type[Any]
    width: int
    head_dim: int
    modulation: Callable[[Any, Mapping[str, Any]], Modulation]
    norms: tuple[str, ...]
    norm: Literal["layer", "rms"]
    attention: str
    arguments: tuple[str, ...]
    mlp: GatedMLP
    rope: Rope | None = None
    qk_norms: tuple[str, str] = ("norm_q", "norm_k")
    gate: Literal["identity", "tanh"] = "identity"
    scale: bool = True
    shift: bool = False
    rows: Literal["bf16", "fp16"] = "bf16"
    weights: Literal["wide", "fp32"] = "wide"
    centered: bool = False
    rms_rounding: Literal["twice", "once"] = "twice"
    qk_affine: bool = True
    compose: bool = False
    hidden_states: str = "hidden_states"
    entries: _kernels.Entries = field(init=False, repr=False, compare=False)
    kernels: TritonKernels = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.width % 16 or self.head_dim & (self.head_dim - 1) or self.width % self.head_dim:
            raise ValueError(f"{self.block.__name__}: width {self.width} / head {self.head_dim}")
        if len(self.norms) not in (1, 2):
            raise ValueError(f"{self.block.__name__}: one norm (parallel) or two, not {self.norms}")
        rotary = self.rope.rotary or self.head_dim if self.rope else None
        if rotary is not None and (rotary > self.head_dim or rotary % 2):
            raise ValueError(f"{self.block.__name__}: rotary {rotary} of {self.head_dim}")
        if (self.shift and not self.scale) or (self.mlp.value_first and self.mlp.up):
            raise ValueError(f"{self.block.__name__}: shift without scale, or value_first with up")
        f16 = self.rows == "fp16"
        rounding = self.rope.rounding if self.rope else "sum"
        entries = _kernels.entries(
            self.width,
            self.head_dim,
            rotary,
            norm=_kernels.LAYER if self.norm == "layer" else _kernels.RMS,
            weight=self.norm == "rms",
            offset=int(self.centered),
            weights="*fp32" if self.weights == "fp32" else "*fp32" if f16 else "*bf16",
            once=self.rms_rounding == "once",
            modulate=(1 + self.shift) if self.scale else 0,
            tanh=self.gate == "tanh",
            f16=f16,
            interleaved=bool(self.rope and self.rope.interleaved),
            round=_kernels.FP32 if rounding == "fp32" else _kernels.FP16 if f16 else _kernels.BF16,
            product=rounding == "product",
            qk_weight=self.qk_affine,
            parallel=self.parallel,
            composed=self.composed,
        )
        keys = "\n".join(e.key for e in entries.all())
        digest = hashlib.sha256(keys.encode()).hexdigest()[:12]
        name = f"fused.{self.block.__name__.lower()[:44]}.{digest}"
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self, "kernels", TritonKernels(name, entries.all(), origin=_declaring_module())
        )

    @property
    def parallel(self) -> bool:
        return len(self.norms) == 1

    @property
    def composed(self) -> bool:
        """Runtime composes the attention from its projections and the selected kernel."""
        return self.parallel or self.compose

    @property
    def rotary(self) -> int:
        return self.rope.rotary or self.head_dim if self.rope else 0

    @property
    def dtype(self) -> Any:
        """The rows' dtype."""
        return torch.float16 if self.rows == "fp16" else torch.bfloat16

    @property
    def wide(self) -> Any:
        """The residual stream's and the tables' dtype."""
        return torch.float32 if self.rows == "fp16" else torch.bfloat16

    @property
    def weight_dtype(self) -> Any:
        """The norm and q/k norm weights' dtype."""
        return torch.float32 if self.weights == "fp32" else self.wide

    def contract(self) -> dict[str, object]:
        """What a fused block's numerics depend on, beyond its kernels' sources."""
        return {
            "block": f"{self.block.__module__}:{self.block.__qualname__}",
            "modulation": f"{self.modulation.__module__}:{self.modulation.__qualname__}",
            "width": self.width,
            "head_dim": self.head_dim,
            "norm": self.norm,
            "parallel": self.parallel,
            "gate": self.gate,
            "scale": self.scale,
            "shift": self.shift,
            "rows": self.rows,
            "weights": self.weights,
            "centered": self.centered,
            "rms_rounding": self.rms_rounding,
            "qk_affine": self.qk_affine,
            "composed": self.composed,
            "fused_up": self.mlp.up is None,
            "value_first": self.mlp.value_first,
            "rope": None
            if self.rope is None
            else [self.rope.interleaved, self.rope.rotary, self.rope.rounding],
            "arguments": list(self.arguments),
        }


def _declaring_module() -> str:
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back  # type: ignore[assignment]
    return str(frame.f_globals.get("__name__", "")) if frame is not None else ""


class _Mismatch(Exception):
    """A block that is not the declared layout: eager serves it."""


class _Switch:
    """A layout's launcher once its kernels are loaded. Until then (None) the installed
    blocks serve eager: Runtime wraps each block's forward after the plan, so the fused one
    is in place from the start and only the launcher arrives later."""

    __slots__ = ("launch",)

    def __init__(self) -> None:
        self.launch: Launch | None = None


@dataclass(frozen=True, slots=True)
class _Site:
    """One installed block: its declaration, launcher and the modules it calls. `qkv` and `out`:
    a composed attention's projections; `inputs`: the MLP's gate [and up]. A norm's weight is
    read at each call: the bound tensor while the block runs."""

    layout: DiTBlock
    switch: _Switch
    signature: inspect.Signature
    norms: tuple[Any, ...]
    eps: tuple[float, ...]
    attention: Any
    qk: tuple[Any, ...]
    qkv: tuple[Any, ...]
    inputs: tuple[Any, ...]
    down: Any
    out: Any = None


def _identity(layout: DiTBlock) -> str:
    document = {"kernels": layout.kernels.identity, "layout": layout.contract()}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(b"cozy.fusion.dit-block\0" + encoded).hexdigest()


def _hooked(module: Any) -> bool:
    return bool(module._forward_hooks or module._forward_pre_hooks)


def _eps(module: Any, path: str) -> float:
    eps = getattr(module, "eps", None)
    if not isinstance(eps, (int, float)):
        raise _Mismatch(f"{path}: no numeric eps")
    return float(eps)


def _norm(module: Any, layout: DiTBlock, path: str) -> tuple[Any, float]:
    if layout.norm == "layer":
        if not (
            isinstance(module, torch.nn.LayerNorm)
            and module.weight is None
            and module.bias is None
            and tuple(module.normalized_shape) == (layout.width,)
        ):
            raise _Mismatch(f"{path}: not a {layout.width}-wide LayerNorm without affine")
        return None, float(module.eps)
    weight = getattr(module, "weight", None)
    if (
        weight is None
        or tuple(weight.shape) != (layout.width,)
        or weight.dtype is not layout.weight_dtype
        or getattr(module, "bias", None)
    ):
        raise _Mismatch(
            f"{path}: not a {layout.width}-wide {layout.weight_dtype} RMSNorm with a weight"
        )
    return weight, _eps(module, path)


def _projections(attention: Any, layout: DiTBlock, path: str) -> tuple[Any, ...]:
    """A parallel block's `(to_q, to_k, to_v, to_out)`, from a Diffusers attention module."""
    out = getattr(attention, "to_out", None)
    out = out[0] if isinstance(out, torch.nn.ModuleList) else out
    found = (*(getattr(attention, n, None) for n in ("to_q", "to_k", "to_v")), out)
    heads = getattr(attention, "heads", None)
    if (
        dispatch_attention_fn is None
        or not all(isinstance(m, torch.nn.Module) for m in found)
        or not isinstance(heads, int)
        or heads * layout.head_dim != layout.width
        or getattr(attention, "fused_projections", False)
        or not hasattr(getattr(attention, "processor", None), "_attention_backend")
    ):
        raise _Mismatch(f"{path}: not a Diffusers attention with to_q, to_k, to_v and to_out")
    return found


def _site(block: Any, layout: DiTBlock, switch: _Switch, path: str) -> _Site:
    if type(block) is not layout.block or "forward" in vars(block):
        raise _Mismatch(f"{path}: not a {layout.block.__qualname__}")
    names = (layout.mlp.gate, *((layout.mlp.up,) if layout.mlp.up else ()), layout.mlp.down)
    try:
        norms = tuple(block.get_submodule(name) for name in layout.norms)
        attention: Any = block.get_submodule(layout.attention)
        mlp: Any = block.get_submodule(layout.mlp.module)
        *inputs, down = (mlp.get_submodule(n) for n in names)
        fused_qk = layout.rope or layout.composed
        qk = tuple(attention.get_submodule(n) for n in layout.qk_norms) if fused_qk else ()
    except AttributeError as exc:
        raise _Mismatch(f"{path}: {exc}") from exc
    # The fused forward calls neither the norms nor the MLP container: their hooks would not
    # run. Nor, composed, the attention module and its q/k norms.
    bypassed = [*norms, mlp, *(m for m in mlp.children() if m not in (*inputs, down))]
    projections: tuple[Any, ...] = ()
    if layout.composed:
        projections = _projections(attention, layout, f"{path}.{layout.attention}")
        called = (*projections, *qk)
        bypassed += [attention, *qk, *(m for m in attention.children() if m not in called)]
    if any(_hooked(m) for m in bypassed):
        raise _Mismatch(f"{path}: a hook on a module the fused forward does not call")
    found = [_norm(m, layout, f"{path}.{n}") for m, n in zip(norms, layout.norms, strict=True)]
    for norm, name in zip(qk, layout.qk_norms, strict=True):
        weight = getattr(norm, "weight", None)
        affine = (
            weight is not None
            and tuple(weight.shape) == (layout.head_dim,)
            and weight.dtype is layout.weight_dtype
        )
        if affine != layout.qk_affine or "forward" in vars(norm):
            raise _Mismatch(f"{path}.{name}: not the {layout.head_dim}-wide RMSNorm declared")
        _eps(norm, f"{path}.{name}")
    signature = inspect.signature(layout.block.forward)
    missing = {layout.hidden_states, *layout.arguments} - set(signature.parameters)
    passed = signature.parameters if layout.parallel else layout.arguments
    if missing or (layout.rope and layout.rope.argument not in passed):
        raise _Mismatch(f"{path}: the forward lacks {sorted(missing) or [layout.rope]}")
    return _Site(
        layout,
        switch,
        signature,
        norms,
        tuple(e for _, e in found),
        attention,
        qk,
        projections[:3],
        tuple(inputs),
        down,
        projections[3] if projections else None,
    )


def _cuda(dtype: Any, *tensors: Any) -> bool:
    return all(t is None or (t.dtype is dtype and t.is_cuda) for t in tensors)


def _fusable(layout: DiTBlock, x: Any) -> bool:
    """The residual stream the kernels take: bf16, or under fp16 autocast fp16 or fp32."""
    if layout.rows == "bf16":
        return _cuda(torch.bfloat16, x)
    return (
        x.is_cuda
        and x.dtype in (torch.float16, torch.float32)
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") is torch.float16
    )


def _block_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    site: _Site = self._cozy_dit
    layout, launch = site.layout, site.switch.launch
    if launch is None:
        return self._cozy_eager(*args, **kwargs)
    bound = site.signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    x = arguments[layout.hidden_states]
    if not _fusable(layout, x):
        return self._cozy_eager(*args, **kwargs)
    rows = layout.modulation(self, arguments)
    tables = (rows.scale_msa, rows.gate_msa, rows.scale_mlp, rows.gate_mlp)
    if not _cuda(layout.wide, *tables, rows.shift_msa, rows.shift_mlp):
        return self._cozy_eager(*args, **kwargs)
    _, normed = _kernels.norm_modulate(
        launch,
        layout.entries.first,
        x,
        site.norms[0].weight,
        rows.scale_msa,
        rows.shift_msa,
        rows.index,
        site.eps[0],
    )
    run = _parallel if layout.parallel else _sequential
    return run(site, launch, x, normed, rows, arguments)


def _inputs(linears: Sequence[Any], normed: Any) -> list[Any]:
    """Every linear that reads the block's normed rows, on those rows: an encoded one quantizes
    them once between them."""
    with quantize_once(normed):
        return [linear(normed) for linear in linears]


def _mlp(launch: Launch, site: _Site, hidden: Sequence[Any]) -> Any:
    """`down(silu(gate) * up)`. An int8 `down` gets the product from its own quantization pass
    when gate and up are two linears' rows: the same rows, never written in bf16."""
    mlp = site.layout.mlp
    if mlp.up is None:
        (both,) = hidden
        half = int(both.shape[-1]) // 2
        hidden = (both[..., :half], both[..., half:])
        hidden = hidden[::-1] if mlp.value_first else hidden
    else:
        rows = int8_rows(hidden[0], (site.down,), activation="swiglu", up=hidden[1])
        if rows is not None:
            return site.down(rows)
    return site.down(_kernels.swiglu(launch, site.layout.entries.swiglu, *hidden))


def _call_attention(launch: Launch, site: _Site, normed: Any, arguments: Any) -> Any:
    """The block's attention module on the normed rows, its q/k norms applying RoPE."""
    layout = site.layout
    call = {name: arguments[name] for name in layout.arguments}
    if layout.rope is not None and call[layout.rope.argument] is not None:
        tables = _kernels.rope_tables(
            call[layout.rope.argument], interleaved=layout.rope.interleaved, rotary=layout.rotary
        )
        call[layout.rope.argument] = None
        for norm in site.qk:
            norm._cozy_rope = (launch, layout.entries.rope, tables, layout)
    try:
        with quantize_once(normed), _selected(site.attention):
            return site.attention(normed, **call)
    finally:
        for norm in site.qk:
            norm._cozy_rope = None


@contextmanager
def _selected(attention: Any) -> Iterator[None]:
    """The attention's own dispatches that name no backend run the kernel Runtime selected for it,
    not Diffusers' process default: Qwen-Image 2.1's prefill attends each prefix segment and the
    target image that way (`backend=None`), so its first step attended in bf16 (xs-134)."""
    backend = getattr(getattr(attention, "processor", None), "_attention_backend", None)
    if backend is None or _AttentionBackendRegistry is None:
        yield
        return
    default = _AttentionBackendRegistry._active_backend
    _AttentionBackendRegistry._active_backend = backend
    try:
        yield
    finally:
        _AttentionBackendRegistry._active_backend = default


def _sequential(
    site: _Site, launch: Launch, x: Any, normed: Any, rows: Modulation, arguments: Any
) -> Any:
    layout, entries = site.layout, site.layout.entries
    if layout.composed:
        attended = _attend(launch, site, *_inputs(site.qkv, normed), arguments)
    else:
        attended = _call_attention(launch, site, normed, arguments)
    assert entries.second is not None
    x, normed = _kernels.norm_modulate(
        launch,
        entries.second,
        x,
        site.norms[1].weight,
        rows.scale_mlp,
        rows.shift_mlp,
        rows.index,
        site.eps[1],
        branch=attended,
        gate=rows.gate_msa,
    )
    mlp = _mlp(launch, site, _inputs(site.inputs, normed))
    x, _ = _kernels.norm_modulate(
        launch,
        entries.residual,
        x,
        None,
        None,
        None,
        rows.index,
        0.0,
        branch=mlp,
        gate=rows.gate_mlp,
    )
    return x


def _parallel(
    site: _Site, launch: Launch, x: Any, normed: Any, rows: Modulation, arguments: Any
) -> Any:
    q, k, v, *hidden = _inputs((*site.qkv, *site.inputs), normed)
    x, _ = _kernels.norm_modulate(
        launch,
        site.layout.entries.residual,
        x,
        None,
        None,
        None,
        rows.index,
        0.0,
        branch=_attend(launch, site, q, k, v, arguments),
        other=_mlp(launch, site, hidden),
        gate=rows.gate_msa,
    )
    return x


def _attend(launch: Launch, site: _Site, q: Any, k: Any, v: Any, arguments: Any) -> Any:
    """A composed attention: q/k norms and RoPE fused, the kernel Runtime selected, `to_out`."""
    layout, entries = site.layout, site.layout.entries
    heads = (site.attention.heads, layout.head_dim)
    q, k, v = (t.unflatten(-1, heads) for t in (q, k, v))
    if layout.rope is None or arguments[layout.rope.argument] is None:
        q, k = (norm(t) for norm, t in zip(site.qk, (q, k), strict=True))
    else:
        assert entries.rope is not None
        tables = _kernels.rope_tables(
            arguments[layout.rope.argument],
            interleaved=layout.rope.interleaved,
            rotary=layout.rotary,
        )
        q, k = (
            _kernels.head_norm_rope(
                launch, entries.rope, t, norm.weight, norm.eps, tables, layout.dtype
            )
            for norm, t in zip(site.qk, (q, k), strict=True)
        )
    processor = site.attention.processor
    attended = dispatch_attention_fn(
        q,
        k,
        v,
        backend=processor._attention_backend,
        parallel_config=processor._parallel_config,
    )
    return site.out(attended.flatten(2))


def _head_norm_forward(self: Any, hidden_states: Any) -> Any:
    rope = self._cozy_rope
    # fp16 rows: the fp16 projection, or a processor's fp32 upcast of it.
    if rope is None or hidden_states.dim() != 4 or not _fusable(rope[3], hidden_states):
        return self._cozy_eager(hidden_states)
    launch, entry, tables, layout = rope
    return _kernels.head_norm_rope(
        launch, entry, hidden_states, self.weight, self.eps, tables, layout.dtype
    )


_FUSED: dict[tuple[type, str], type] = {}


def _fused(cls: type, kind: str) -> type:
    """The installed subclass of `cls`: same parameters, buffers and state-dict keys. Its eager
    forward is `cls.forward`, taken here, so a subclass made after it (graphs' region class) still
    reaches it; `super(type(self), self)` there resolves to the fused forward again and recurses.
    A class fused already, or a subclass of one, is returned as is: fusing twice changes nothing."""
    if getattr(cls, "_cozy_fused", None) == kind:
        return cls
    if (cls, kind) not in _FUSED:
        body: dict[str, Any] = {
            "__module__": __name__,
            "_cozy_fused": kind,
            "_cozy_eager": cls.forward,
        }
        if kind == "block":
            body["forward"] = _block_forward
        elif kind == "channel":
            body["forward"] = _channel_norm_forward
        elif kind == "identity":
            body["forward"] = _identity_forward
        elif kind == "lone-conv":
            body["forward"] = _lone_conv_forward
        elif kind == "nearest":
            body["forward"] = torch.nn.Upsample.forward
        else:
            body |= {"forward": _head_norm_forward, "_cozy_rope": None}
        _FUSED[(cls, kind)] = type(f"Fused{cls.__name__}", (cls,), body)
    return _FUSED[(cls, kind)]


def _census(module: Any) -> tuple[object, ...]:
    return tuple(
        (name, tuple(value.shape), str(value.dtype), id(value))
        for name, value in (
            *module.named_parameters(remove_duplicate=False),
            *module.named_buffers(remove_duplicate=False),
        )
    )


def _report(plan: Plan, layout: DiTBlock, reason: str = "", blocks: int = 0) -> None:
    """Under the layout's kernel set name, as Runtime reports that set's failed build."""
    document: dict[str, object] = {"block": layout.block.__qualname__, "applied": not reason}
    document |= {"reason": reason} if reason else {"blocks": blocks}
    plan.report(layout.kernels.name, document)
    print(f"[cozy] {layout.kernels.name}: {json.dumps(document, sort_keys=True)}", file=sys.stderr)


def install(plan: Plan, layout: DiTBlock, *roots: Any) -> str:
    """Fuse every `layout.block` under `roots` on `plan.device`: the plan's identity, or ""
    when the eager blocks serve (no CUDA sm80+, kernels absent or failed, or a block that is
    not the declared layout; reported). Never waits for the kernels' compile."""
    device = torch.device(plan.device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        _report(plan, layout, f"fused blocks serve CUDA sm80+, not {device}")
        return ""
    blocks: list[tuple[str, Any]] = [
        (f"{index}.{path}", module)
        for index, root in enumerate(roots)
        for path, module in root.named_modules()
        if type(module) is layout.block
    ]
    if not blocks:
        # Installed already (the blocks are fused, a graph region's subclass included): a no-op.
        if any(
            getattr(type(m), "_cozy_fused", None) == "block"
            and getattr(vars(m).get("_cozy_dit"), "layout", None) == layout
            for root in roots
            for m in root.modules()
        ):
            return _identity(layout)
        _report(plan, layout, f"no {layout.block.__qualname__}")
        return ""
    switch = _Switch()
    try:
        sites = [(block, _site(block, layout, switch, path)) for path, block in blocks]
    except _Mismatch as exc:
        _report(plan, layout, str(exc))
        return ""
    before = [_census(root) for root in roots]
    # An attention module calls its q/k norms; a composed one launches them.
    heads = [norm for _, site in sites for norm in site.qk] if not layout.composed else []
    eager = [(m, type(m)) for m in (*(block for block, _ in sites), *heads)]
    for norm in heads:
        norm.__class__ = _fused(type(norm), "head")
    for block, site in sites:
        block.__class__ = _fused(type(block), "block")
        block._cozy_dit = site
    if [_census(root) for root in roots] != before:
        raise RuntimeError("the fused substitution moved a parameter or buffer")

    def activate(launch: Launch) -> None:
        switch.launch = launch
        _report(plan, layout, blocks=len(blocks))

    try:
        if plan.when_ready is None:
            activate(plan.load(layout.kernels))
        elif not plan.when_ready(layout.kernels, activate):
            _report(plan, layout, "compiling; fused from the next request on")
    except KernelsUnavailable as exc:
        if exc.code == "kernels_corrupt":
            raise
        for module, cls in eager:
            module.__class__ = cls
        _report(plan, layout, f"{exc.code}: {exc}")
        return ""
    return _identity(layout)


@dataclass(frozen=True, slots=True)
class ChannelNorm:
    """A VAE's channel-first RMS norm, `F.normalize(x, dim=1) * scale * gamma [+ bias]` over
    `[batch, channels, ...]`, which Diffusers computes in fp32 and rounds back (Wan's `RMS_norm`,
    `QwenImageRMS_norm`, `QwenImage21RMS_norm`): one read for the squares, one for the writes,
    the same roundings. `silu` names, per block class, the norms its `nonlinearity` (a SiLU)
    follows; every call of that nonlinearity must follow one of them. Those norms apply the SiLU
    in the same kernel and the nonlinearity becomes the identity. Activations are `dtype`."""

    norm: type
    silu: tuple[tuple[type, tuple[str, ...]], ...] = ()
    nonlinearity: str = "nonlinearity"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    bias: bool = False
    entries: tuple[Any, ...] = field(init=False, repr=False, compare=False)
    kernels: TritonKernels = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = _kernels.channel_entries(self.dtype, self.bias, bool(self.silu))
        name = f"fused.{self.norm.__name__.lower()[:44]}.{self.dtype}"
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "kernels", TritonKernels(name, entries, origin=_declaring_module()))

    @property
    def torch_dtype(self) -> Any:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]


@dataclass(slots=True)
class _NormSite:
    layout: ChannelNorm
    switch: _Switch
    silu: bool
    scale: float
    eps: float


def _channel_norm_forward(self: Any, x: Any) -> Any:
    site: _NormSite = vars(self)["_cozy_norm"]
    launch, layout = site.switch.launch, site.layout
    bias = self.bias if isinstance(self.bias, torch.Tensor) else None
    if (
        launch is None
        or x.dtype is not layout.torch_dtype
        or x.dim() < 2
        or not _cuda(layout.torch_dtype, x, self.gamma, *(() if bias is None else (bias,)))
    ):
        y = self._cozy_eager(x)
        return torch.nn.functional.silu(y) if site.silu else y
    entry = layout.entries[int(site.silu)]
    return _kernels.channel_norm(launch, entry, x, self.gamma, bias, site.scale, site.eps)


def _identity_forward(self: Any, x: Any) -> Any:
    return x


def _norm_site(norm: Any, layout: ChannelNorm, switch: _Switch, silu: bool, path: str) -> _NormSite:
    channels = int(norm.gamma.shape[0])
    if not getattr(norm, "channel_first", False) or int(norm.gamma.numel()) != channels:
        raise _Mismatch(f"{path}: not a channel-first norm with one weight per channel")
    if float(getattr(norm, "scale", 0.0)) != channels**0.5:
        raise _Mismatch(f"{path}: scale is not sqrt({channels})")
    if isinstance(norm.bias, torch.Tensor) != layout.bias:
        raise _Mismatch(f"{path}: {'no ' if layout.bias else ''}bias, not as declared")
    if not isinstance(norm.bias, torch.Tensor) and norm.bias != 0.0:
        raise _Mismatch(f"{path}: a constant bias {norm.bias}")
    return _NormSite(layout, switch, silu, channels**0.5, 1e-12)


def install_norms(plan: Plan, layout: ChannelNorm, *roots: Any) -> str:
    """Fuse every `layout.norm` under `roots` on `plan.device` [with the SiLU after it]: the
    plan's identity, or "" when the eager norms serve (no CUDA sm80+, kernels absent or failed,
    or a module that is not the declaration; reported). Never waits for the kernels' compile."""
    device = torch.device(plan.device)
    document: dict[str, object] = {"norm": layout.norm.__qualname__}

    def report(reason: str = "", **counts: int) -> str:
        outcome = {**document, "applied": not reason, **({"reason": reason} if reason else counts)}
        plan.report(layout.kernels.name, outcome)
        print(f"[cozy] {layout.kernels.name}: {json.dumps(outcome, sort_keys=True)}", file=sys.stderr)
        return ""

    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        return report(f"fused norms serve CUDA sm80+, not {device}")
    norms = [
        (f"{index}.{path}", module)
        for index, root in enumerate(roots)
        for path, module in root.named_modules()
        if type(module) is layout.norm
    ]
    if not norms:
        return report(f"no {layout.norm.__qualname__}")
    fused_silu: set[int] = set()
    activations: dict[int, Any] = {}
    for index, root in enumerate(roots):
        for path, block in root.named_modules():
            names = next((n for cls, n in layout.silu if type(block) is cls), None)
            if names is None:
                continue
            activation = getattr(block, layout.nonlinearity, None)
            if type(activation) is not torch.nn.SiLU:
                return report(f"{index}.{path}.{layout.nonlinearity}: not a SiLU")
            for name in names:
                norm = getattr(block, name, None)
                if type(norm) is not layout.norm:
                    return report(f"{index}.{path}.{name}: not a {layout.norm.__qualname__}")
                fused_silu.add(id(norm))
            activations[id(activation)] = activation
    parents = Counter(
        id(child) for root in roots for module in root.modules() for child in module.children()
    )
    if any(parents[key] != 1 for key in activations):
        return report("a declared nonlinearity is called by another module too")
    switch = _Switch()
    try:
        sites = [
            (norm, _norm_site(norm, layout, switch, id(norm) in fused_silu, path))
            for path, norm in norms
        ]
    except _Mismatch as exc:
        return report(str(exc))
    before = [_census(root) for root in roots]
    eager = [(m, type(m)) for m in (*(n for n, _ in sites), *activations.values())]
    for norm, site in sites:
        norm.__class__ = _fused(type(norm), "channel")
        norm._cozy_norm = site
    for activation in activations.values():
        activation.__class__ = _fused(type(activation), "identity")
    if [_census(root) for root in roots] != before:
        raise RuntimeError("the fused substitution moved a parameter or buffer")
    counts = {"norms": len(sites), "with_silu": len(fused_silu)}

    def activate(launch: Launch) -> None:
        switch.launch = launch
        report(**counts)

    try:
        if plan.when_ready is None:
            activate(plan.load(layout.kernels))
        elif not plan.when_ready(layout.kernels, activate):
            report("compiling; fused from the next request on")
    except KernelsUnavailable as exc:
        if exc.code == "kernels_corrupt":
            raise
        for module, cls in eager:
            module.__class__ = cls
        return report(f"{exc.code}: {exc}")
    encoded = json.dumps(
        {"kernels": layout.kernels.identity, "norm": layout.norm.__qualname__,
         "silu": [[cls.__qualname__, list(names)] for cls, names in layout.silu],
         "dtype": layout.dtype, "bias": layout.bias},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(b"cozy.fusion.channel-norm\0" + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LoneFrame:
    """A video VAE decoding one image frame, `[batch, channels, 1, height, width]` with no frame
    cache (Diffusers' Wan and Qwen-Image VAEs). `conv`, its causal convolution (a Conv3d, or the
    Conv2d that specializes it to images, each padding by copy as `_padding` says), runs as one
    2-D convolution: on its last temporal tap, the only one a lone frame meets (the others meet
    the causal padding's zeros), with the spatial zero padding inside the convolution, so no
    padded copy is made. `upsample`, its nearest upsampler, runs in the activations' dtype:
    Diffusers' goes through fp32, and nearest only copies values, so the bits are the same."""

    conv: type
    upsample: type | None = None


def _lone_conv_forward(self: Any, x: Any, cache_x: Any = None) -> Any:
    if cache_x is not None or x.dim() != 5 or x.shape[2] != 1:
        return self._cozy_eager(x, cache_x)
    weight = self.weight if self.weight.dim() == 4 else self.weight[:, :, -1]
    y = torch.nn.functional.conv2d(
        x[:, :, 0],
        weight,
        self.bias,
        self.stride[-2:],
        self._cozy_lone,
        self.dilation[-2:],
        self.groups,
    )
    return y.unsqueeze(2)


def _lone_padding(conv: Any, path: str) -> tuple[int, int]:
    """`conv`'s spatial zero padding (rows, columns) when a lone frame meets its last tap only."""
    pad = tuple(getattr(conv, "_padding", ()))
    if getattr(conv, "padding_mode", None) != "zeros":
        raise _Mismatch(f"{path}: not a zero-padded convolution")
    if isinstance(conv, torch.nn.Conv3d):
        frames = conv.kernel_size[0]
        if (
            tuple(conv.padding) != (0, 0, 0)
            or len(pad) != 6
            or pad[4:] != (frames - 1, 0)
            or conv.dilation[0] != 1
        ):
            raise _Mismatch(f"{path}: not a causal convolution padding {frames - 1} frames before")
    elif not isinstance(conv, torch.nn.Conv2d) or tuple(conv.padding) != (0, 0) or len(pad) != 4:
        raise _Mismatch(f"{path}: not a causal convolution")
    if pad[0] != pad[1] or pad[2] != pad[3]:
        raise _Mismatch(f"{path}: asymmetric spatial padding {pad[:4]}")
    return (int(pad[2]), int(pad[0]))


def install_lone_frame(plan: Plan, layout: LoneFrame, *roots: Any) -> str:
    """`layout` on every `conv` and `upsample` under `roots`: the plan's identity, or "" when the
    eager modules serve (a module that is not the declaration; reported). No kernels, so CPU
    and GPU alike."""
    name = f"lone-frame.{layout.conv.__name__.lower()}"

    def report(reason: str = "", **counts: int) -> str:
        outcome = {
            "conv": layout.conv.__qualname__,
            "applied": not reason,
            **({"reason": reason} if reason else counts),
        }
        plan.report(name, outcome)
        print(f"[cozy] {name}: {json.dumps(outcome, sort_keys=True)}", file=sys.stderr)
        return ""

    found = [
        (f"{index}.{path}", module)
        for index, root in enumerate(roots)
        for path, module in root.named_modules()
    ]
    convs = [(path, m) for path, m in found if type(m) is layout.conv]
    upsamples = [(path, m) for path, m in found if type(m) is layout.upsample]
    if not convs:
        return report(f"no {layout.conv.__qualname__}")
    try:
        padding = [_lone_padding(conv, path) for path, conv in convs]
        for path, upsample in upsamples:
            if not isinstance(upsample, torch.nn.Upsample) or upsample.mode not in (
                "nearest",
                "nearest-exact",
            ):
                raise _Mismatch(f"{path}: not a nearest upsampler")
    except _Mismatch as exc:
        return report(str(exc))
    before = [_census(root) for root in roots]
    for (_, conv), pad in zip(convs, padding, strict=True):
        conv.__class__ = _fused(type(conv), "lone-conv")
        conv._cozy_lone = pad
    for _, upsample in upsamples:
        upsample.__class__ = _fused(type(upsample), "nearest")
    if [_census(root) for root in roots] != before:
        raise RuntimeError("the lone-frame substitution moved a parameter or buffer")
    report(convs=len(convs), upsamples=len(upsamples))
    upsample = getattr(layout.upsample, "__qualname__", None)
    encoded = json.dumps(
        {"conv": layout.conv.__qualname__, "upsample": upsample},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(b"cozy.fusion.lone-frame\0" + encoded).hexdigest()


__all__ = [
    "ChannelNorm",
    "DiTBlock",
    "GatedMLP",
    "LoneFrame",
    "Modulation",
    "Rope",
    "install",
    "install_lone_frame",
    "install_norms",
]
