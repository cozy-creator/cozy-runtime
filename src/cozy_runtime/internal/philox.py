"""Counter-based noise (`rng.counter_noise`, `rng.counter_uniform`): Philox4x32-10 of the counter
(block, step, stream, kind) under the seed, four 32-bit words per block of four elements, then
Box-Muller (Cephes log, sine and cosine) or 24-bit uniforms in float32, each operation rounded
once, cast to the output's dtype.

On CUDA one Triton kernel (`KERNELS`, a Runtime set: shipped prebuilt, built at machine start
otherwise) computes each element where it stores it, so a draw holds only its output. It
compiles with no contraction (`fp_fusion=False`), the IEEE square root (`sqrt_rn`) and no unary
minus (Triton's is `0 - x`, +0 for +0), and serves once its probe matches `law` bit for bit on
the card (`kernel_ready.serve`). Until then, and on a CPU, meta tensors, another dtype or another
device, `law` computes the same values in torch ops, about 30 bytes per element at its peak.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from cozy_runtime.accel.kernels import Launch, TritonEntry, TritonKernels
from cozy_runtime.internal import kernel_ready, verify

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"triton does not import ({_absent})"
    if TYPE_CHECKING:
        raise
else:
    TRITON_ABSENT = ""

#: Philox4x32 multipliers and key increments (Salmon et al. 2011, Random123)
_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF
#: The kernel's output dtypes, by its `OUT` constexpr
_OUT = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
#: Counter blocks (four elements each) per program
_BLOCK = 256


def int64(value: int) -> int:
    """`value` modulo 2**64 as an int64 tensor holds it: a seed past 2**63 wraps."""
    return (value + 2**63) % 2**64 - 2**63


def draw(
    shape: Sequence[int],
    seed: int | Tensor,
    step: int | Tensor,
    stream: int,
    kind: int,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> Tensor:
    """Normal (`kind` 0) or uniform (1) noise of `shape` in `dtype`: the kernel's where it
    serves, else `law`'s, the same values."""
    given = next((v.device for v in (step, seed) if isinstance(v, Tensor)), "cpu")
    where = torch.device(given if device is None else device)
    if where.type == "cuda" and cast("int | None", where.index) is None:  # torch types it int
        where = torch.device("cuda", torch.cuda.current_device())
    launch = _served(where, dtype, seed, step)
    if launch is None:
        return law(shape, seed, step, stream, kind, where).to(dtype)
    return _launched(launch, shape, seed, step, stream, kind, dtype, where)


def law(
    shape: Sequence[int], seed: Any, step: Any, stream: int, kind: int, device: torch.device
) -> Tensor:
    """The draw in torch ops, in float32: what the kernel computes, operation for operation."""
    words = _words(shape, seed, step, stream, kind, device)
    if kind:
        values = (words.reshape(-1) >> 8).float() * 5.960464477539063e-08
    else:
        radius = (_log((words[:, 0::2] + 1).float() * 2.3283064365386963e-10) * -2.0).sqrt()
        cos, sin = _turn(words[:, 1::2])
        values = torch.stack((radius * cos, radius * sin), -1).reshape(-1)
    return values[: math.prod(shape)].reshape(tuple(shape))


def _words(
    shape: Sequence[int], seed: Any, step: Any, stream: int, kind: int, device: torch.device
) -> Tensor:
    """Philox4x32-10 under the key `seed`: the four 32-bit words (int64) of each block of four
    elements of `shape`, one row per block."""
    blocks = (math.prod(shape) + 3) // 4
    c = [torch.arange(blocks, device=device), step & _MASK, stream & _MASK, kind]
    k0, k1 = seed & _MASK, (seed >> 32) & _MASK
    for r in range(10):
        if r:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
        hi0, lo0 = _mulhilo(c[0], _M0)
        hi1, lo1 = _mulhilo(c[2], _M1)
        c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0]
    return torch.stack(c, -1)


def _mulhilo(a: Any, m: int) -> tuple[Any, Any]:
    """The high and low 32-bit words of `a * m`, a 32-bit word by a 32-bit constant, by 16-bit
    halves of `m` so int64 never overflows."""
    low = a * (m & 0xFFFF)
    mid = a * (m >> 16) + (low >> 16)
    return mid >> 16, ((mid & 0xFFFF) << 16) | (low & 0xFFFF)


def _log(x: Tensor) -> Tensor:
    """Natural log of float32 `x` > 0 (Cephes logf)."""
    m, e = torch.frexp(x)
    low = m < 0.7071067690849304
    x = torch.where(low, m + m, m) - 1.0
    e = (e - low.int()).float()
    z = x * x
    y = x * 7.0376836292e-2 + -1.1514610310e-1
    y = y * x + 1.1676998740e-1
    y = y * x + -1.2420140846e-1
    y = y * x + 1.4249322787e-1
    y = y * x + -1.6668057665e-1
    y = y * x + 2.0000714765e-1
    y = y * x + -2.4999993993e-1
    y = y * x + 3.3333331174e-1
    return x + (y * x * z + e * -2.12194440e-4 - z * 0.5) + e * 0.693359375


def _turn(w: Tensor) -> tuple[Tensor, Tensor]:
    """Cosine and sine of a uniform angle set by the 32-bit word `w`: its top two bits pick a
    quarter turn, the rest an angle in [-pi/4, pi/4) (Cephes sinf and cosf)."""
    x = ((w & 0x3FFFFFFF) - 0x20000000).float() * 1.4629180792671596e-09
    z = x * x
    s = ((z * -1.9515295891e-4 + 8.3321608736e-3) * z - 1.6666654611e-1) * z * x + x
    c = ((z * 2.443315711809948e-5 - 1.388731625493765e-3) * z + 4.166664568298827e-2) * z * z
    c = c - z * 0.5 + 1.0
    quarter, half = ((w >> 30) & 1) != 0, (w >> 31) != 0
    cos, sin = torch.where(quarter, s * -1.0, c), torch.where(quarter, c, s)
    return torch.where(half, cos * -1.0, cos), torch.where(half, sin * -1.0, sin)


if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]
    def _words_kernel(seed, step, stream, block, KIND: tl.constexpr):  # type: ignore[no-untyped-def]
        """`_words` of `block`, the program's counter blocks."""
        key = tl.load(seed).to(tl.uint64, bitcast=True)
        k0, k1 = (key & 0xFFFFFFFF).to(tl.uint32), (key >> 32).to(tl.uint32)
        c0, c1 = block.to(tl.uint32), tl.load(step).to(tl.uint32)
        c2, c3 = stream.to(tl.uint32), tl.full(block.shape, KIND, tl.uint32)
        for r in tl.static_range(10):
            if r > 0:
                k0 = tl.add(k0, 0x9E3779B9, sanitize_overflow=False)
                k1 = tl.add(k1, 0xBB67AE85, sanitize_overflow=False)
            hi0, lo0 = tl.umulhi(c0, 0xD2511F53), tl.mul(c0, 0xD2511F53, sanitize_overflow=False)
            hi1, lo1 = tl.umulhi(c2, 0xCD9E8D57), tl.mul(c2, 0xCD9E8D57, sanitize_overflow=False)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        return c0, c1, c2, c3

    @triton.jit
    def _log_kernel(x):  # type: ignore[no-untyped-def]
        """`_log`, frexp by its bits (x is normal)."""
        bits = x.to(tl.int32, bitcast=True)
        m = ((bits & 0x007FFFFF) | 0x3F000000).to(tl.float32, bitcast=True)
        low = m < 0.7071067690849304
        x = tl.where(low, m + m, m) - 1.0
        e = ((bits >> 23) - 126 - low.to(tl.int32)).to(tl.float32)
        z = x * x
        y = x * 7.0376836292e-2 + -1.1514610310e-1
        y = y * x + 1.1676998740e-1
        y = y * x + -1.2420140846e-1
        y = y * x + 1.4249322787e-1
        y = y * x + -1.6668057665e-1
        y = y * x + 2.0000714765e-1
        y = y * x + -2.4999993993e-1
        y = y * x + 3.3333331174e-1
        return x + (y * x * z + e * -2.12194440e-4 - z * 0.5) + e * 0.693359375

    @triton.jit
    def _turn_kernel(w):  # type: ignore[no-untyped-def]
        """`_turn`."""
        x = ((w & 0x3FFFFFFF).to(tl.int32) - 0x20000000).to(tl.float32) * 1.4629180792671596e-09
        z = x * x
        s = ((z * -1.9515295891e-4 + 8.3321608736e-3) * z - 1.6666654611e-1) * z * x + x
        c = ((z * 2.443315711809948e-5 - 1.388731625493765e-3) * z + 4.166664568298827e-2) * z * z
        c = c - z * 0.5 + 1.0
        quarter, half = ((w >> 30) & 1) != 0, (w >> 31) != 0
        cos, sin = tl.where(quarter, s * -1.0, c), tl.where(quarter, c, s)
        return tl.where(half, cos * -1.0, cos), tl.where(half, sin * -1.0, sin)

    @triton.jit  # type: ignore[untyped-decorator]
    def _kernel(  # type: ignore[no-untyped-def]
        out, seed, step, stream, n, KIND: tl.constexpr, OUT: tl.constexpr, BLOCK: tl.constexpr
    ):
        """`law` for `BLOCK` counter blocks, stored as `out`'s dtype (`OUT` keys the entry)."""
        program = tl.program_id(0).to(tl.int64)
        w0, w1, w2, w3 = _words_kernel(
            seed, step, stream, program * BLOCK + tl.arange(0, BLOCK), KIND
        )
        if KIND == 0:
            u = (tl.join(w0, w2).to(tl.uint64) + 1).to(tl.float32) * 2.3283064365386963e-10
            radius = tl.sqrt_rn(_log_kernel(u) * -2.0)
            cos, sin = _turn_kernel(tl.join(w1, w3))
            values = tl.join(radius * cos, radius * sin)
        else:
            words = tl.join(tl.join(w0, w2), tl.join(w1, w3))  # join's axis is the minor one
            values = (words >> 8).to(tl.float32) * 5.960464477539063e-08
        at = program * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)
        tl.store(out + at, tl.reshape(values, [4 * BLOCK]).to(out.dtype.element_ty), mask=at < n)


def _entries() -> tuple[TritonEntry, ...]:
    if TRITON_ABSENT:
        return ()
    names = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
    return tuple(
        TritonEntry(
            _kernel,
            {"out": f"*{names[dtype]}", "seed": "*i64", "step": "*i64", "stream": "i64",
             "n": "i64", "KIND": "constexpr", "OUT": "constexpr", "BLOCK": "constexpr"},
            {"KIND": kind, "OUT": code, "BLOCK": _BLOCK},
            ("out",),
            fp_fusion=False,
        )
        for kind, (dtype, code) in itertools.product((0, 1), _OUT.items())
    )  # fmt: skip


#: Built with the Runtime's own sets (`machine_kernels.triton_sets`): a request never compiles.
KERNELS = TritonKernels("cozy-counter-noise", _entries())


def _served(device: torch.device, dtype: torch.dtype, *values: int | Tensor) -> Launch | None:
    """This card's kernel once it serves (`kernel_ready.serve`), else None: `law` serves."""
    if device.type != "cuda" or dtype not in _OUT or TRITON_ABSENT or not kernel_ready.TRITON:
        return None
    for value in values:
        if isinstance(value, Tensor) and (value.device != device or value.dtype != torch.int64):
            return None
    capturing = torch.cuda.is_current_stream_capturing()
    return kernel_ready.serve(_name(device), _opener(device), capturing)


def _launched(
    launch: Launch,
    shape: Sequence[int],
    seed: int | Tensor,
    step: int | Tensor,
    stream: int,
    kind: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    out = torch.empty(tuple(shape), dtype=dtype, device=device)
    if n := out.numel():
        seed, step = (
            v if isinstance(v, Tensor) else torch.full((1,), int64(v), device=device)
            for v in (seed, step)
        )
        grid = (triton.cdiv(n, 4 * _BLOCK),)
        with torch.cuda.device(device):
            launch(_kernel, grid, out, seed, step, stream & _MASK, n, kind, _OUT[dtype], _BLOCK)
    return out


def _name(device: torch.device) -> str:
    return f"{KERNELS.name} {device}"


def _opener(device: torch.device) -> kernel_ready.Opener[Launch]:
    return partial(kernel_ready.TRITON[0], KERNELS, partial(probe, device), device=device)


def prime(index: int) -> None:
    """Opens cuda:`index`'s kernel at device init (`kernel_ready.prime`), beside the weight fill,
    so the first draw serves it."""
    if not TRITON_ABSENT and kernel_ready.TRITON:
        device = torch.device("cuda", index)
        kernel_ready.prime(_name(device), _opener(device))


def probe(device: torch.device, launch: Launch) -> verify.Verdict:
    """The kernel against `law` on this card, bit for bit: both kinds, every dtype, one and many
    programs with a ragged end, host and tensor seeds and steps, seeds past 2**63."""
    tensor = partial(torch.full, (1,), device=device)
    keys: tuple[tuple[int | Tensor, int | Tensor, int], ...] = (
        (0, 0, 0),
        (2**64 - 1, 2**32 + 7, 2**32 - 1),
        (tensor(int64(0x9E3779B97F4A7C15)), tensor(9), 3),
    )
    draws = itertools.product((0, 1), _OUT, (1, 7, 12 * _BLOCK + 5, 1 << 16), keys)
    for kind, dtype, n, (seed, step, stream) in draws:
        got = _launched(launch, (n,), seed, step, stream, kind, dtype, device).view(torch.uint8)
        want = law((n,), seed, step, stream, kind, device).to(dtype).view(torch.uint8)
        if not torch.equal(got, want):
            at = int((got != want).nonzero()[0]) // (torch.finfo(dtype).bits // 8)
            return verify.Verdict(False, f"kind {kind} {dtype} n={n}: element {at} differs")
    return verify.Verdict(True, "bit-identical to the torch law")
