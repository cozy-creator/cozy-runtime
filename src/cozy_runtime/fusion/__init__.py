"""Fused DiT blocks: a package declares its block's layout, Runtime owns the kernels and the
substitution.

A single-stream adaLN block runs, per token row:

    normed = norm1(x) * (1 + scale_msa) [+ shift_msa]
    x      = x + act(gate_msa) * attention(normed)       # q/k: per-head RMSNorm, then RoPE
    normed = norm2(x) * (1 + scale_mlp) [+ shift_mlp]
    x      = x + act(gate_mlp) * down(silu(gate(normed)) * up(normed))

Eager, each line is several kernels; fused, four row kernels and one per q/k carry it, and
every GEMM stays the model's own module. Declare the layout at module scope, so its kernels
compile on the executor's GPUs at boot, and install it in `Model.execution_plan`:

    BLOCK = fusion.DiTBlock(MyBlock, width=4096, head_dim=128, modulation=rows, ...)

    def execution_plan(self, plan):
        return fusion.install(plan, BLOCK, self.transformer)

A construction never waits for the compile: until it is ready the eager blocks serve, and the
fused ones from the next request on. A block whose structure, widths or dtype differ from the
declaration serves eager (reported).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from cozy_runtime.author.kernels import KernelsUnavailable, Launch, Plan, TritonKernels
from cozy_runtime.fusion import _kernels

try:
    import torch
except ImportError:  # pragma: no cover - the base package is torch-free
    torch = None


@dataclass(frozen=True, slots=True)
class Modulation:
    """One block call's adaLN tables: `[rows, width]` views (any row stride) and each token's
    row, `index` (int64 over `tokens` or `batch * tokens` rows; it wraps)."""

    index: Any
    scale_msa: Any
    gate_msa: Any
    scale_mlp: Any
    gate_mlp: Any
    shift_msa: Any = None
    shift_mlp: Any = None


@dataclass(frozen=True, slots=True)
class Rope:
    """The block's rotary argument. Its attention applies the q/k norms whatever it receives
    and skips RoPE when it is None, so the fused norms apply it. `interleaved`: pairs (2i,
    2i+1), a complex `[seq, rotary/2]` table; else halves, a `(cos, sin)` `[seq, rotary]` pair.
    `rotary`: rotated channels per head, 0 for all."""

    argument: str
    interleaved: bool
    rotary: int = 0


@dataclass(frozen=True, slots=True)
class GatedMLP:
    """`down(silu(gate(x)) * up(x))`: submodule names, the linears relative to `module`."""

    module: str
    gate: str
    up: str
    down: str


@dataclass(frozen=True)
class DiTBlock:
    """A package's single-stream adaLN block, as Runtime fuses it.

    `modulation(block, arguments)` returns the call's `Modulation` from the block forward's
    bound arguments. `norms` are the pre-attention and pre-MLP norms, `"layer"` (no affine)
    or `"rms"` (weight); `attention` is called with `arguments` (keywords taken from the
    block's) after the normed rows; `qk_norms` (relative to it) are per-head RMSNorms with a
    weight. `gate`: the activation applied to both gates.
    """

    block: type[Any]
    width: int
    head_dim: int
    modulation: Callable[[Any, Mapping[str, Any]], Modulation]
    norms: tuple[str, str]
    norm: Literal["layer", "rms"]
    attention: str
    arguments: tuple[str, ...]
    mlp: GatedMLP
    rope: Rope | None = None
    qk_norms: tuple[str, str] = ("norm_q", "norm_k")
    gate: Literal["identity", "tanh"] = "identity"
    shift: bool = False
    hidden_states: str = "hidden_states"
    entries: _kernels.Entries = field(init=False, repr=False, compare=False)
    kernels: TritonKernels = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.width % 16 or self.head_dim & (self.head_dim - 1) or self.width % self.head_dim:
            raise ValueError(f"{self.block.__name__}: width {self.width} / head {self.head_dim}")
        rotary = self.rope.rotary or self.head_dim if self.rope else None
        if rotary is not None and (rotary > self.head_dim or rotary % 2):
            raise ValueError(f"{self.block.__name__}: rotary {rotary} of {self.head_dim}")
        entries = _kernels.entries(
            self.width,
            self.head_dim,
            rotary,
            norm=_kernels.LAYER if self.norm == "layer" else _kernels.RMS,
            weight=self.norm == "rms",
            shift=self.shift,
            tanh=self.gate == "tanh",
            interleaved=bool(self.rope and self.rope.interleaved),
        )
        keys = "\n".join(e.key for e in entries.all())
        name = f"cozy-dit-{hashlib.sha256(keys.encode()).hexdigest()[:16]}"
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self, "kernels", TritonKernels(name, entries.all(), origin=_declaring_module())
        )

    def contract(self) -> dict[str, object]:
        """What a fused block's numerics depend on, beyond its kernels' sources."""
        return {
            "block": f"{self.block.__module__}:{self.block.__qualname__}",
            "modulation": f"{self.modulation.__module__}:{self.modulation.__qualname__}",
            "width": self.width,
            "head_dim": self.head_dim,
            "norm": self.norm,
            "gate": self.gate,
            "shift": self.shift,
            "rope": None if self.rope is None else [self.rope.interleaved, self.rope.rotary],
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
    """One installed block: its declaration, launcher and the modules it calls."""

    layout: DiTBlock
    switch: _Switch
    signature: inspect.Signature
    norm_weights: tuple[Any, Any]
    eps: tuple[float, float]
    attention: Any
    qk: tuple[Any, ...]
    gate: Any
    up: Any
    down: Any


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
    if weight is None or tuple(weight.shape) != (layout.width,) or getattr(module, "bias", None):
        raise _Mismatch(f"{path}: not a {layout.width}-wide RMSNorm with a weight")
    return weight, _eps(module, path)


def _site(block: Any, layout: DiTBlock, switch: _Switch, path: str) -> _Site:
    if type(block) is not layout.block or "forward" in vars(block):
        raise _Mismatch(f"{path}: not a {layout.block.__qualname__}")
    try:
        norms = tuple(block.get_submodule(name) for name in layout.norms)
        attention: Any = block.get_submodule(layout.attention)
        mlp: Any = block.get_submodule(layout.mlp.module)
        gate, up, down = (
            mlp.get_submodule(n) for n in (layout.mlp.gate, layout.mlp.up, layout.mlp.down)
        )
        qk = tuple(attention.get_submodule(n) for n in layout.qk_norms) if layout.rope else ()
    except AttributeError as exc:
        raise _Mismatch(f"{path}: {exc}") from exc
    # The fused forward calls neither the norms nor the MLP container: their hooks would not run.
    bypassed = (*norms, mlp, *(m for m in mlp.children() if m not in (gate, up, down)))
    if any(_hooked(m) for m in bypassed):
        raise _Mismatch(f"{path}: a hook on a module the fused forward does not call")
    (w1, e1), (w2, e2) = (
        _norm(m, layout, f"{path}.{n}") for m, n in zip(norms, layout.norms, strict=True)
    )
    for norm, name in zip(qk, layout.qk_norms, strict=True):
        weight = getattr(norm, "weight", None)
        if weight is None or tuple(weight.shape) != (layout.head_dim,) or "forward" in vars(norm):
            raise _Mismatch(f"{path}.{name}: not a {layout.head_dim}-wide RMSNorm with a weight")
        _eps(norm, f"{path}.{name}")
    signature = inspect.signature(layout.block.forward)
    missing = {layout.hidden_states, *layout.arguments} - set(signature.parameters)
    if missing or (layout.rope and layout.rope.argument not in layout.arguments):
        raise _Mismatch(f"{path}: the forward lacks {sorted(missing) or [layout.rope]}")
    return _Site(
        layout,
        switch,
        signature,
        (w1, w2),
        (e1, e2),
        attention,
        qk,
        gate,
        up,
        down,
    )


def _bf16_cuda(*tensors: Any) -> bool:
    return all(t is None or (t.dtype is torch.bfloat16 and t.is_cuda) for t in tensors)


def _block_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    site: _Site = self._cozy_dit
    layout, launch = site.layout, site.switch.launch
    bound = site.signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    x = arguments[layout.hidden_states]
    if launch is None or not _bf16_cuda(x):
        return super(type(self), self).forward(*args, **kwargs)
    rows = layout.modulation(self, arguments)
    if not _bf16_cuda(
        rows.scale_msa, rows.gate_msa, rows.scale_mlp, rows.gate_mlp, rows.shift_msa, rows.shift_mlp
    ):
        return super(type(self), self).forward(*args, **kwargs)
    entries = layout.entries
    _, normed = _kernels.norm_modulate(
        launch,
        entries.first,
        x,
        site.norm_weights[0],
        rows.scale_msa,
        rows.shift_msa,
        rows.index,
        site.eps[0],
    )
    call = {name: arguments[name] for name in layout.arguments}
    if layout.rope is not None and call[layout.rope.argument] is not None:
        tables = _kernels.rope_tables(
            call[layout.rope.argument], interleaved=layout.rope.interleaved
        )
        call[layout.rope.argument] = None
        for norm in site.qk:
            norm._cozy_rope = (launch, entries.rope, tables)
    try:
        attended = site.attention(normed, **call)
    finally:
        for norm in site.qk:
            norm._cozy_rope = None
    x, normed = _kernels.norm_modulate(
        launch,
        entries.second,
        x,
        site.norm_weights[1],
        rows.scale_mlp,
        rows.shift_mlp,
        rows.index,
        site.eps[1],
        branch=attended,
        gate=rows.gate_msa,
    )
    mlp = site.down(_kernels.swiglu(launch, site.gate(normed), site.up(normed)))
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


def _head_norm_forward(self: Any, hidden_states: Any) -> Any:
    rope = self._cozy_rope
    if rope is None or hidden_states.dim() != 4 or not _bf16_cuda(hidden_states):
        return super(type(self), self).forward(hidden_states)
    launch, entry, tables = rope
    return _kernels.head_norm_rope(launch, entry, hidden_states, self.weight, self.eps, tables)


_FUSED: dict[tuple[type, str], type] = {}


def _fused(cls: type, kind: str) -> type:
    """The installed subclass of `cls`: same parameters, buffers and state-dict keys. A module
    shared by two blocks is fused once."""
    if cls in _FUSED.values():
        return cls
    if (cls, kind) not in _FUSED:
        body: dict[str, Any] = {"__module__": __name__}
        if kind == "block":
            body["forward"] = _block_forward
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


def _report(document: dict[str, object]) -> None:
    print(f"[cozy] fused DiT blocks: {json.dumps(document, sort_keys=True)}", file=sys.stderr)


def install(plan: Plan, layout: DiTBlock, *roots: Any) -> str:
    """Fuse every `layout.block` under `roots` on `plan.device`: the plan's identity, or ""
    when the eager blocks serve (no CUDA sm80+, kernels absent or failed, or a block that is
    not the declared layout; reported). Never waits for the kernels' compile."""
    device = torch.device(plan.device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        _report({"applied": False, "reason": f"fused blocks serve CUDA sm80+, not {device}"})
        return ""
    blocks: list[tuple[str, Any]] = [
        (f"{index}.{path}", module)
        for index, root in enumerate(roots)
        for path, module in root.named_modules()
        if type(module) is layout.block
    ]
    if not blocks:
        _report({"applied": False, "reason": f"no {layout.block.__qualname__}"})
        return ""
    switch = _Switch()
    try:
        sites = [(block, _site(block, layout, switch, path)) for path, block in blocks]
    except _Mismatch as exc:
        _report({"applied": False, "reason": str(exc)})
        return ""
    before = [_census(root) for root in roots]
    eager = [(m, type(m)) for block, site in sites for m in (block, *site.qk)]
    for block, site in sites:
        for norm in site.qk:
            norm.__class__ = _fused(type(norm), "head")
        block.__class__ = _fused(type(block), "block")
        block._cozy_dit = site
    if [_census(root) for root in roots] != before:
        raise RuntimeError("the fused substitution moved a parameter or buffer")

    def activate(launch: Launch) -> None:
        switch.launch = launch
        _report({"applied": True, "blocks": len(blocks), "kernels": layout.kernels.name})

    try:
        if plan.when_ready is None:
            activate(plan.load(layout.kernels))
        elif not plan.when_ready(layout.kernels, activate):
            _report({"applied": False, "reason": "compiling; fused from the next request on"})
    except KernelsUnavailable as exc:
        if exc.code == "kernels_corrupt":
            raise
        for module, cls in eager:
            module.__class__ = cls
        _report({"applied": False, "reason": f"{exc.code}: {exc}"})
        return ""
    return _identity(layout)


__all__ = ["DiTBlock", "GatedMLP", "Modulation", "Rope", "install"]
