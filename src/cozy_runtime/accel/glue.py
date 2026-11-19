"""Fused row kernels for a transformer block whose forward a package writes itself: each elementwise
chain the block runs between its GEMMs, one pass over `[rows, width]` bf16 rows. They are `fusion`'s
kernels; a block that `fusion.DiTBlock` can declare has Runtime write its forward instead.

Declare a `Glue` at module scope, so its sets compile on the executor's GPUs at boot and a release
ships them prebuilt, and load the set the card builds in `Model.execution_plan`:

    GLUE = glue.Glue("my-dit-glue", width=3072, ffn=8192, head_dim=128, rotary=128)

    def execution_plan(self, plan):
        launch = plan.load(GLUE.kernels(plan.device))
        ...  # the fused forwards call glue.norm_modulate(launch, GLUE, x, ...)

The declaration states the model's rounding, and each kernel rounds wherever eager bf16 rounds, so a
fused chain equals eager up to the row variance's reduction order. With `fp8=True` a norm or SwiGLU
writes the rowwise fp8 leaf's operand (`PreQuantized`: E4M3 and a per-row scale) from the registers
holding the row, in place of bf16 rows the leaf would read back to quantize. Those entries need fp8
(sm89+): `Glue.kernels` names the set a card builds.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from types import FrameType
from typing import TYPE_CHECKING, Literal, overload

from cozy_runtime import rng
from cozy_runtime.accel.fusion import _kernels
from cozy_runtime.accel.kernels import Launch, PreQuantized, TritonKernels

if TYPE_CHECKING:
    import torch
    from torch import Tensor
else:
    try:
        import torch
    except ImportError:  # pragma: no cover - the base package is torch-free
        torch = None


def _declaring_module() -> str:
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
    return str(frame.f_globals.get("__name__", "")) if frame is not None else ""


@dataclass(frozen=True)
class Glue:
    """A block's row kernels at its widths. Its norms are RMSNorms with a weight, modulated by `1 +
    scale` [`+ shift`] read at each token's table row; `rms_rounding`: `"twice"` (Diffusers'
    RMSNorm) or `"once"` (`torch.nn.RMSNorm`). Its q/k norms are per-head RMSNorms with a weight,
    then RoPE on the first `rotary` channels (0: all): split-half or `interleaved`, rounded once
    (`"sum"`) or at each product, the tables cast to bf16 first (`"product"`, eager rotary in bf16).
    `ffn` is the MLP's inner width, at which `checked` tries SwiGLU."""

    name: str
    width: int
    ffn: int
    head_dim: int
    rotary: int = 0
    shift: bool = False
    rms_rounding: Literal["twice", "once"] = "twice"
    rope_rounding: Literal["sum", "product"] = "sum"
    interleaved: bool = False
    absent: str = field(init=False, repr=False, compare=False)
    """Why no kernel serves here (no Triton), or ""."""
    entries: _kernels.Entries = field(init=False, repr=False, compare=False)
    fp8_entries: _kernels.Fp8Entries = field(init=False, repr=False, compare=False)
    bf16: TritonKernels = field(init=False, repr=False, compare=False)
    """Every sm80+ card's set."""
    fp8: TritonKernels = field(init=False, repr=False, compare=False)
    """With the fp8 epilogues, for cards with fp8 (sm89+): Triton has no E4M3 below, and one entry
    that does not build refuses its whole set."""

    def __post_init__(self) -> None:
        if self.width % 16 or self.ffn % 16 or self.head_dim & (self.head_dim - 1):
            raise ValueError(
                f"{self.name}: width {self.width}, ffn {self.ffn}, head {self.head_dim}"
            )
        if self.rotary > self.head_dim or self.rotary % 2:
            raise ValueError(f"{self.name}: rotary {self.rotary} of {self.head_dim}")
        object.__setattr__(self, "absent", _kernels.TRITON_ABSENT)
        origin = _declaring_module()
        if self.absent:
            object.__setattr__(self, "bf16", TritonKernels(self.name, (), origin=origin))
            object.__setattr__(self, "fp8", TritonKernels(f"{self.name}-fp8", (), origin=origin))
            return
        once = self.rms_rounding == "once"
        entries = _kernels.entries(
            self.width,
            self.head_dim,
            self.rotary or self.head_dim,
            norm=_kernels.RMS,
            weight=True,
            offset=0,
            weights="*bf16",
            once=once,
            modulate=1 + self.shift,
            tanh=False,
            f16=False,
            interleaved=self.interleaved,
            round=_kernels.BF16,
            product=self.rope_rounding == "product",
            qk_weight=True,
            parallel=False,
            composed=False,
        )
        fp8 = _kernels.fp8_entries(
            self.width,
            norm=_kernels.RMS,
            weight=True,
            offset=0,
            once=once,
            modulate=1 + self.shift,
            tanh=False,
        )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "fp8_entries", fp8)
        object.__setattr__(self, "bf16", TritonKernels(self.name, entries.all(), origin=origin))
        object.__setattr__(
            self,
            "fp8",
            TritonKernels(f"{self.name}-fp8", (*entries.all(), *fp8.all()), origin=origin),
        )

    def kernels(self, device: torch.device) -> TritonKernels:
        """The set `device` builds."""
        return self.fp8 if torch.cuda.get_device_capability(device) >= (8, 9) else self.bf16


def _bf16(x: Tensor) -> None:
    if x.dtype is not torch.bfloat16 or not x.is_cuda:
        raise ValueError(f"the glue's rows are bfloat16 on CUDA, not {x.dtype} on {x.device}")


@overload
def norm_modulate(
    launch: Launch,
    glue: Glue,
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    index: Tensor,
    eps: float,
    *,
    fp8: Literal[False] = False,
) -> Tensor: ...
@overload
def norm_modulate(
    launch: Launch,
    glue: Glue,
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    index: Tensor,
    eps: float,
    *,
    fp8: Literal[True],
) -> PreQuantized: ...
@overload
def norm_modulate(
    launch: Launch,
    glue: Glue,
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    index: Tensor,
    eps: float,
    *,
    fp8: bool,
) -> Tensor | PreQuantized: ...
def norm_modulate(
    launch: Launch,
    glue: Glue,
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    index: Tensor,
    eps: float,
    *,
    fp8: bool = False,
) -> Tensor | PreQuantized:
    """`rmsnorm(x) * (1 + scale[index]) [+ shift[index]]`: one read, one write. Tables are `[rows,
    width]` views (any row stride); `index` (int64) gives each token's row, wrapping over a batch.
    """
    _bf16(x)
    shift = shift if glue.shift else None
    if fp8:
        _, payload, scales = _kernels.norm_modulate_fp8(
            launch, glue.fp8_entries.first, x, weight, scale, shift, index, eps
        )
        return PreQuantized(payload, scales, tuple(x.shape))
    _, out = _kernels.norm_modulate(launch, glue.entries.first, x, weight, scale, shift, index, eps)
    assert out is not None  # a norm's entry
    return out


@overload
def gate_add_norm_modulate(
    launch: Launch,
    glue: Glue,
    residual: Tensor,
    gate: Tensor,
    branch: Tensor,
    index: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    eps: float,
    *,
    fp8: Literal[False] = False,
    into: Tensor | None = None,
) -> tuple[Tensor, Tensor]: ...
@overload
def gate_add_norm_modulate(
    launch: Launch,
    glue: Glue,
    residual: Tensor,
    gate: Tensor,
    branch: Tensor,
    index: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    eps: float,
    *,
    fp8: Literal[True],
    into: Tensor | None = None,
) -> tuple[Tensor, PreQuantized]: ...
@overload
def gate_add_norm_modulate(
    launch: Launch,
    glue: Glue,
    residual: Tensor,
    gate: Tensor,
    branch: Tensor,
    index: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    eps: float,
    *,
    fp8: bool,
    into: Tensor | None = None,
) -> tuple[Tensor, Tensor | PreQuantized]: ...
def gate_add_norm_modulate(
    launch: Launch,
    glue: Glue,
    residual: Tensor,
    gate: Tensor,
    branch: Tensor,
    index: Tensor,
    weight: Tensor,
    scale: Tensor,
    shift: Tensor | None,
    eps: float,
    *,
    fp8: bool = False,
    into: Tensor | None = None,
) -> tuple[Tensor, Tensor | PreQuantized]:
    """`hidden = residual + gate[index] * branch`, then `norm_modulate(hidden)`, from the registers
    that hold the row: `(hidden, normed)`. `hidden` is written `into` where given (contiguous and
    16-byte aligned; `branch` or `residual` itself may be it)."""
    _bf16(residual)
    shift = shift if glue.shift else None
    if fp8:
        hidden, payload, scales = _kernels.norm_modulate_fp8(
            launch, glue.fp8_entries.second, residual, weight, scale, shift, index, eps,
            branch=branch, gate=gate, into=into,
        )  # fmt: skip
        return hidden, PreQuantized(payload, scales, tuple(residual.shape))
    assert glue.entries.second is not None  # a sequential layout's
    hidden, out = _kernels.norm_modulate(
        launch,
        glue.entries.second,
        residual,
        weight,
        scale,
        shift,
        index,
        eps,
        branch=branch,
        gate=gate,
        into=into,
    )
    assert out is not None  # a norm's entry
    return hidden, out


def gate_add(
    launch: Launch,
    glue: Glue,
    residual: Tensor,
    gate: Tensor,
    branch: Tensor,
    index: Tensor,
    *,
    into: Tensor | None = None,
) -> Tensor:
    """`residual + gate[index] * branch` in one pass (eager: a gather, a multiply and an add),
    written `into` where given (`residual` or `branch` itself may be it)."""
    _bf16(residual)
    hidden, _ = _kernels.norm_modulate(
        launch,
        glue.entries.residual,
        residual,
        None,
        None,
        None,
        index,
        0.0,
        branch=branch,
        gate=gate,
        into=into,
    )
    return hidden


@overload
def swiglu(
    launch: Launch, glue: Glue, gate: Tensor, up: Tensor, *, fp8: Literal[False] = False
) -> Tensor: ...
@overload
def swiglu(
    launch: Launch, glue: Glue, gate: Tensor, up: Tensor, *, fp8: Literal[True]
) -> PreQuantized: ...
@overload
def swiglu(
    launch: Launch, glue: Glue, gate: Tensor, up: Tensor, *, fp8: bool
) -> Tensor | PreQuantized: ...
def swiglu(
    launch: Launch, glue: Glue, gate: Tensor, up: Tensor, *, fp8: bool = False
) -> Tensor | PreQuantized:
    """`silu(gate) * up`, rows of equal shape and any row stride: halves of one fused projection are
    views (Diffusers' `SwiGLU` is `proj(x)`'s [up | gate])."""
    _bf16(gate)
    if fp8:
        payload, scales = _kernels.swiglu_fp8(launch, glue.fp8_entries.swiglu, gate, up)
        return PreQuantized(payload, scales, tuple(gate.shape))
    return _kernels.swiglu(launch, glue.entries.swiglu, gate, up)


def head_norm_rope(
    launch: Launch, glue: Glue, x: Tensor, weight: Tensor, eps: float, cos: Tensor, sin: Tensor
) -> Tensor:
    """Per-head RMSNorm, then RoPE on `[batch, seq, heads, head_dim]` `x`, in place: each program
    reads its heads whole before it writes them, so `x` (a projection's own output) never exists
    twice. `cos` and `sin` are `(seq, rotary)` (interleaved: or `(seq, rotary / 2)`); channels past
    `rotary` pass through."""
    _bf16(x)
    x = x if x.is_contiguous() else x.contiguous()
    assert glue.entries.rope is not None  # every glue has its rope
    rotary = glue.rotary or glue.head_dim
    tables = _kernels.rope_tables((cos, sin), interleaved=glue.interleaved, rotary=rotary)
    return _kernels.head_norm_rope(
        launch, glue.entries.rope, x, weight, eps, tables, torch.bfloat16, into=x
    )


def warm(launch: Launch, glue: Glue, device: torch.device, *, heads: int, fp8: bool) -> float:
    """Launch every entry the set holds, on two rows; nothing compiles. The milliseconds the first
    launches took to enqueue."""
    started = time.perf_counter()
    rows, table = 2, 3
    x = torch.ones(1, rows, glue.width, device=device, dtype=torch.bfloat16)
    weight = torch.ones(glue.width, device=device, dtype=torch.bfloat16)
    tables = torch.ones(table, 6 * glue.width, device=device, dtype=torch.bfloat16).chunk(6, dim=-1)
    index = torch.arange(rows, device=device) % table
    ffn = torch.ones(1, rows, 2 * glue.ffn, device=device, dtype=torch.bfloat16)
    for quant in {False, fp8}:
        norm_modulate(launch, glue, x, weight, tables[1], tables[0], index, 1e-5, fp8=quant)
        gate_add_norm_modulate(
            launch, glue, x, tables[2], x, index, weight, tables[4], tables[3], 1e-5, fp8=quant
        )
        swiglu(launch, glue, ffn[..., glue.ffn :], ffn[..., : glue.ffn], fp8=quant)
    gate_add(launch, glue, x, tables[5], x, index)
    q = torch.ones(1, rows, heads, glue.head_dim, device=device, dtype=torch.bfloat16)
    angles = torch.zeros(rows, glue.rotary or glue.head_dim, device=device)
    head_norm_rope(
        launch,
        glue,
        q,
        torch.ones(glue.head_dim, device=device, dtype=torch.bfloat16),
        1e-5,
        angles,
        angles,
    )
    return (time.perf_counter() - started) * 1000


def _off(got: Tensor, want: Tensor) -> float:
    """`got`'s relative L2 error against the exact `want`, inf where it is not finite."""
    got = got.double()
    if not bool(torch.isfinite(got).all()):
        return float("inf")
    return float((got - want).norm() / want.norm().clamp(min=1e-30))


def checked(launch: Launch, glue: Glue, device: torch.device, *, heads: int, fp8: bool) -> str:
    """Each entry against its arithmetic in float64 on this card: what disagrees, or "". Rounding to
    bf16 loses 1e-3 and an E4M3 row 5e-2; a misaddressed tile or a wrong dtype loses O(1).
    Split-half RoPE only."""
    generator = rng.generator(20261104)

    def random(*shape: int, by: float = 1.0) -> Tensor:
        return (rng.noise(shape, generator, device=device) * by).to(torch.bfloat16)

    rows, table, eps, width, inner = 3, 5, 1e-5, glue.width, glue.ffn
    x, branch = random(1, rows, width), random(1, rows, width)
    weight = random(width, by=0.1)
    gate, scale, shift = (random(table, width, by=0.5) for _ in range(3))
    index = torch.arange(rows, device=device) % table
    wide = {
        "gate": gate[index].double(),
        "scale": scale[index].double(),
        "shift": shift[index].double(),
    }
    added = x.double() + wide["gate"] * branch.double()

    def norm(value: Tensor) -> Tensor:
        rms = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps) * weight.double()
        return rms * (1 + wide["scale"]) + (wide["shift"] if glue.shift else 0)

    ffn = random(1, rows, 2 * inner)
    up, gated = ffn[..., :inner], ffn[..., inner:]
    product = up.double() * torch.nn.functional.silu(gated.double())
    errors: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        errors["norm"] = (
            _off(
                norm_modulate(launch, glue, x, weight, scale, shift, index, eps), norm(x.double())
            ),
            1e-2,
        )
        hidden, out = gate_add_norm_modulate(
            launch, glue, x, gate, branch, index, weight, scale, shift, eps
        )
        errors["gate add"] = _off(hidden, added), 1e-2
        errors["gate add norm"] = _off(out, norm(added)), 1e-2
        errors["residual"] = _off(gate_add(launch, glue, x, gate, branch, index), added), 1e-2
        errors["swiglu"] = _off(swiglu(launch, glue, gated, up), product), 1e-2
        if fp8:
            quantized = (
                (
                    norm_modulate(launch, glue, x, weight, scale, shift, index, eps, fp8=True),
                    "norm",
                ),
                (swiglu(launch, glue, gated, up, fp8=True), "swiglu"),
                (
                    gate_add_norm_modulate(
                        launch, glue, x, gate, branch, index, weight, scale, shift, eps, fp8=True
                    )[1],
                    "gate add norm",
                ),
            )
            wants = {"norm": norm(x.double()), "swiglu": product, "gate add norm": norm(added)}
            for operand, name in quantized:
                decoded = operand.payload.double() * operand.scale.double()
                errors[f"fp8 {name}"] = _off(decoded.view_as(wants[name]), wants[name]), 0.1
        if not glue.interleaved:
            rotary = glue.rotary or glue.head_dim
            q = random(1, rows, heads, glue.head_dim)
            angles = (
                torch.rand(rows, rotary, generator=generator, dtype=torch.float32).to(device) * 6.0
            )
            cos, sin = torch.cos(angles), torch.sin(angles)
            head_weight = random(glue.head_dim, by=0.1)
            rms = (
                q.double()
                * torch.rsqrt(q.double().pow(2).mean(-1, keepdim=True) + eps)
                * head_weight.double()
            )
            half = rotary // 2
            low, high = rms[..., :half], rms[..., half:rotary]
            c, s = cos.double()[None, :, None], sin.double()[None, :, None]
            turned = torch.cat(
                [
                    low * c[..., :half] - high * s[..., :half],
                    high * c[..., half:] + low * s[..., half:],
                    rms[..., rotary:],
                ],
                dim=-1,
            )
            got = head_norm_rope(launch, glue, q.clone(), head_weight, eps, cos, sin)
            errors["rope"] = _off(got, turned), 1e-2
    wrong = {name: f"{got:.2e}" for name, (got, bar) in errors.items() if not got <= bar}
    return f"{wrong} of {len(errors)} entries off their arithmetic" if wrong else ""


__all__ = [
    "Glue",
    "checked",
    "gate_add",
    "gate_add_norm_modulate",
    "head_norm_rope",
    "norm_modulate",
    "swiglu",
    "warm",
]
