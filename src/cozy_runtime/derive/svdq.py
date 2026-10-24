"""SVDQuant's producer (xs-066): `svdq.nvfp4` from a model's original 16-bit weights and its own
activations.

`Calibration` hooks the linears a plan encodes as `svdq.nvfp4` while the package runs real prompts
through its model, and keeps each one's per-channel `|x|` max and a uniform reservoir of input rows.
`fit` then encodes one weight `W` `[N, K]` (on the GPU when there is one):
- smoothing `s = |x|max^alpha / |W|max^(1-alpha)` per input channel, `alpha` chosen per linear by
  the mean per-token error of the whole scheme on the reservoir (`ALPHAS`, 0 meaning none);
- a rank-`RANK` SVD of `W · s`: `up = U S`, `down = V^T / s`, the branch the outliers live in;
- the residual `W · s - up · down · s` in nvfp4 (`encode_nvfp4`, its min-SSE block scales);
- `input_scale` = `MARGIN` * the calibrated `|x / s|` max / 2688: the activation's second level.
The logical weight is `residual / s + up · down`, which TensorFS decodes.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import random
import time
import zlib
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

from cozy_runtime import graphs
from cozy_runtime.author._context import device as placed_device
from cozy_runtime.internal import svdq_linear
from cozy_runtime.internal.encoding.formats import nvfp4_values

from .microscale import NVFP4_BLOCK, Encoded, encode_nvfp4

#: Smoothing exponents tried per linear; 0 is no smoothing.
ALPHAS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
#: The branch's rank (Nunchaku's and the xs-064 study's).
RANK = 32
#: Calibration rows kept per linear, sampled uniformly over every call.
ROWS = 512
#: Rows sampled from each call before the reservoir draw.
PER_CALL = 32
#: Weight rows a fit's nvfp4 round trips take at a time: what bounds its temporaries.
BLOCK = 2048
#: The column multiple the kernels need (`svdq_linear`'s K step).
K_ALIGN = svdq_linear.K_ALIGN
#: `input_scale`'s headroom over the calibrated `|x / s|` max: a request's activations may exceed
#: the calibration's before a block scale saturates at 448.
MARGIN = 2.0


class Stats(NamedTuple):
    """One linear's calibration: per-channel `|x|` max `[K]` and sampled input rows `[n, K]`."""

    xmax: Any
    rows: Any


class Calibration:
    """While open, the forward pre-hooks on every `nn.Linear` of `components` whose
    `component.<module>.weight` is in `keys` record its inputs (`Stats`) without a device sync:
    the per-channel `|x|` max on the device, and `PER_CALL` sampled rows of each call into a
    uniform reservoir of `ROWS` in pinned host memory. Runtime's CUDA graphs are bypassed while
    it is open, so every call runs the hooks."""

    def __init__(
        self, torch: ModuleType, components: Mapping[str, Any], keys: Iterable[str]
    ) -> None:
        self.torch = torch
        wanted = set(keys)
        self.modules: dict[str, Any] = {}
        for component, root in components.items():
            for name, module in root.named_modules():
                key = f"{component}.{name}.weight"
                if key in wanted and isinstance(module, torch.nn.Linear):
                    self.modules[key] = module
        missing = sorted(wanted - set(self.modules))
        if missing:
            raise ValueError(f"calibration finds no linear for {missing[:3]} ({len(missing)})")
        self._xmax: dict[str, Any] = {}
        self._rows: dict[str, Any] = {}
        self._seen: dict[str, int] = {}
        self._handles: list[Any] = []
        self._bypass: Any = None
        self._slots = random.Random(0)
        self._picks: dict[Any, Any] = {}

    def __enter__(self) -> Calibration:
        self._bypass = graphs.BYPASS.set(True)
        for key, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(self._hook(key)))
        return self

    def __exit__(self, kind: object, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        graphs.BYPASS.reset(self._bypass)
        if placed_device().type == "cuda":
            self.torch.cuda.synchronize()  # the reservoir's row copies have landed
        missing = sorted(set(self.modules) - set(self._seen))
        if kind is None and missing:
            raise ValueError(
                f"calibration saw no input for {missing[:3]} ({len(missing)}): the prompts never "
                "reached those linears"
            )

    def _hook(self, key: str) -> Any:
        torch = self.torch

        def record(module: Any, args: tuple[Any, ...]) -> None:
            with torch.no_grad():
                flat = args[0].detach().reshape(-1, args[0].shape[-1])
                amax = flat.abs().amax(0).float()
                held = self._xmax.get(key)
                self._xmax[key] = amax if held is None else torch.maximum(held, amax)
                picks = self._picks.get(flat.device)
                if picks is None:
                    picks = self._picks[flat.device] = torch.Generator(flat.device).manual_seed(0)
                take = min(PER_CALL, flat.shape[0])
                pick = torch.randint(flat.shape[0], (take,), generator=picks, device=flat.device)
                rows = flat[pick].to(torch.bfloat16)
                if key not in self._rows:
                    self._rows[key] = torch.empty(
                        ROWS, flat.shape[1], dtype=torch.bfloat16, pin_memory=flat.is_cuda
                    )
                reservoir, seen = self._rows[key], self._seen.get(key, 0)
                for row in rows:
                    seen += 1
                    slot = seen - 1 if seen <= ROWS else self._slots.randrange(seen)
                    if slot < ROWS:
                        reservoir[slot].copy_(row, non_blocking=True)
                self._seen[key] = seen

        return record

    def stats(self, component: str, key: str) -> Stats:
        name = f"{component}.{key}"
        if name not in self._seen:
            raise ValueError(f"calibration saw no input for {name}: run the model first")
        return Stats(self._xmax[name].cpu(), self._rows[name][: min(self._seen[name], ROWS)])


class Fitted(NamedTuple):
    """One weight encoded: the residual's codes, each companion's raw bytes in `companions`
    order, the alpha that chose them, the residual's amax and nvfp4 round-trip error (what the
    encoder's bound holds), the logical weight's squared error and energy, and the mean per-token
    output error on the reservoir of the scheme and of plain nvfp4 W4A4."""

    payload: bytes
    companions: tuple[bytes, ...]
    alpha: float
    amax: float
    max_abs_err: float
    sse: float
    energy: float
    saturated: int
    error: float
    nvfp4_error: float
    #: Seconds the residual's nvfp4 encode took (on the host).
    encode_s: float


def companions(rows: int, cols: int) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """The parts after `weight`, `(role, dtype, shape)`, in the order `fit` writes them."""
    return (
        ("scale", "F8_E4M3", (rows, cols // NVFP4_BLOCK)),
        ("scale_2", "F32", ()),
        ("pre_quant_scale", "F32", (cols,)),
        ("lowrank_down", "BF16", (RANK, cols)),
        ("lowrank_up", "BF16", (rows, RANK)),
        ("input_scale", "F32", ()),
    )


def _bf16(torch: ModuleType, t: Any) -> Any:
    return t.to(torch.bfloat16).float()


def fit(torch: ModuleType, key: str, weight: np.ndarray, stats: Stats, device: Any) -> Fitted:
    """`svdq.nvfp4` for one weight: the alpha whose whole scheme best reproduces the reservoir's
    outputs, token by token; then the stored parts. A weight's temporaries are bounded by
    `BLOCK` rows of it at a time beside the weight and its residual."""
    w = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32)).to(device)
    rows = stats.rows.to(device).float()
    if len(rows) == 0:
        raise ValueError(f"{key}: calibration kept no rows")
    xmax = stats.xmax.to(device).float().clamp(min=1e-5)
    ref = rows @ w.t()

    def off(y: Any) -> float:
        return float(((y - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-30)).mean())

    def scale(t: Any) -> Any:
        return (t.abs().amax() / svdq_linear.LEVELS_MAX).clamp(min=1e-30)

    plain = off(_through(torch, svdq_linear.fake(torch, _bf16(torch, rows), scale(rows)), w))
    wmax = w.abs().amax(0).clamp(min=1e-5)
    best: tuple[float, float, Any, Any, Any] | None = None
    for alpha in ALPHAS:
        s = torch.ones_like(xmax) if alpha == 0 else (xmax**alpha / wmax ** (1 - alpha)).clamp(1e-5)
        s = s / s.mean()
        u, sv, vt = _top(torch, w * s, RANK + 16, zlib.crc32(key.encode()))
        up, down = _bf16(torch, u[:, :RANK] * sv[:RANK]), _bf16(torch, vt[:RANK] / s)
        residual = w * s - up @ (down * s)
        gx = ((xmax / s).amax() * MARGIN / svdq_linear.LEVELS_MAX).clamp(min=1e-30)
        y = _through(torch, svdq_linear.fake(torch, _bf16(torch, rows / s), gx), residual)
        err = off(y + _bf16(torch, rows @ down.t()) @ up.t())
        if best is None or err < best[0]:
            best = (err, alpha, s, up, down)
        del residual, y
    assert best is not None
    _, alpha, s, up, down = best
    residual = w * s - up @ (down * s)
    started = time.perf_counter()
    encoded = encode_nvfp4(key, residual.cpu().numpy())
    encode_s = time.perf_counter() - started
    gx = (xmax / s).amax() * MARGIN / svdq_linear.LEVELS_MAX
    p = 1.0 / s
    stored = _decode(torch, encoded, residual.shape, device)
    y = svdq_linear.fake(torch, _bf16(torch, rows * p), gx) @ stored.t()
    served = off(y + _bf16(torch, rows @ down.t()) @ up.t())
    amax, round_trip = float(residual.abs().max()), float((stored - residual).abs().max())
    del residual
    error = stored.mul_(p).add_(up @ down).sub_(w)
    scales, second = encoded.companions

    def raw(t: Any, dtype: Any) -> bytes:
        return bytes(t.to(dtype).reshape(-1).contiguous().cpu().view(torch.uint8).numpy().tobytes())

    return Fitted(
        encoded.payload,
        (
            scales.raw,
            second.raw,
            raw(p, torch.float32),
            raw(down, torch.bfloat16),
            raw(up, torch.bfloat16),
            raw(gx.reshape(()), torch.float32),
        ),
        alpha,
        amax,
        round_trip,
        float(error.square().sum(dtype=torch.float64)),
        float(w.square().sum(dtype=torch.float64)),
        encoded.saturated,
        served,
        plain,
        encode_s,
    )


def _through(torch: ModuleType, xq: Any, weight: Any) -> Any:
    """`xq @ fake(weight)ᵀ`: `weight` through nvfp4 at its own second level, `BLOCK` rows at a
    time."""
    g = (weight.abs().amax() / svdq_linear.LEVELS_MAX).clamp(min=1e-30)
    return torch.cat(
        [xq @ svdq_linear.fake(torch, part, g).t() for part in weight.split(BLOCK)], dim=1
    )


def _top(torch: ModuleType, a: Any, q: int, seed: int) -> tuple[Any, Any, Any]:
    """`a`'s top `q` singular triplets `(U, S, Vᵀ)` by randomized subspace iteration from a
    `seed`ed start: the same weight fits to the same bytes on every run."""
    start = torch.Generator(device=a.device).manual_seed(seed)
    y = a @ torch.randn(a.shape[1], q, generator=start, device=a.device)
    for _ in range(4):
        y = a @ (a.t() @ torch.linalg.qr(y).Q)
    basis = torch.linalg.qr(y).Q
    u, sv, vt = torch.linalg.svd(basis.t() @ a, full_matrices=False)
    return basis @ u, sv, vt


def _decode(torch: ModuleType, encoded: Encoded, shape: Any, device: Any) -> Any:
    """`encode_nvfp4`'s payload back to f32, by the TensorFS law."""
    rows, cols = int(shape[0]), int(shape[1])
    weight = torch.frombuffer(bytearray(encoded.payload), dtype=torch.uint8).view(rows, cols // 2)
    scale, second = encoded.companions
    block = torch.frombuffer(bytearray(scale.raw), dtype=torch.uint8).view(torch.float8_e4m3fn)
    g = torch.frombuffer(bytearray(second.raw), dtype=torch.float32)
    return nvfp4_values(
        torch, weight.to(device), block.view(rows, cols // NVFP4_BLOCK).to(device), g.to(device)
    )
