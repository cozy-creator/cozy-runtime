"""Every scheme the decode cores (`decode_cores.py`) serve: one `Adapter` row each, with its params,
its parts' dtype and shape rules (cozytensors/1 spec §6) and its decode as core calls.
`encoding/adapters.py` serves each row as a decode provider and `encoding/formats.part_rule`
checks a tensor's parts by the row. Outside the `encoding` package, so Triton's interpreter
imports it.

Parts are as the fill plane hands them over: runtime dtype names, and an `F4` part as its packed
bytes, `u8 [..., C/2]`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime.internal import decode_cores as dc
from cozy_runtime.internal.decode_cores import (
    HAS_TRITON,
    ROW,
    Cores,
    Packing,
    blocks_of,
    round_into,
    unpack,
)

if TYPE_CHECKING:
    from torch import Generator, Tensor
    from torch import device as Device
    from torch import dtype as Dtype

try:
    import triton
    import triton.language as tl
except ImportError:  # the eager reference still runs
    triton = None

TORCH = {
    "f32": "float32",
    "bf16": "bfloat16",
    "f16": "float16",
    "f8_e4m3fn": "float8_e4m3fn",
    "f8_e5m2": "float8_e5m2",
    "f8_e8m0": "float8_e8m0fnu",
    "u8": "uint8",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}

Shape = tuple[str, tuple[int, ...] | int]  # ("exact", (extents...)) or ("count", n)
Rule = tuple[object, ...] | Callable[[object], bool]


@dataclass(frozen=True)
class View:
    """What a part rule sees: the stored view (`S` under transposed storage), the params and the
    parts' (dtype, shape)."""

    shape: tuple[int, ...]
    params: Mapping[str, object]
    parts: Mapping[str, tuple[str, tuple[int, ...]]]
    flat: bool = False

    @property
    def cols(self) -> int:
        return math.prod(self.shape) if self.flat else (self.shape[-1] if self.shape else 0)

    @property
    def rows(self) -> int:
        return 1 if self.flat else (math.prod(self.shape) // self.cols if self.cols else 0)

    @property
    def lead(self) -> tuple[int, ...]:
        return self.shape[:-1]

    @property
    def height(self) -> int:
        """N: the rows of each matrix of a stack `[..., N, C]`."""
        return self.shape[-2]

    @property
    def matrices(self) -> int:
        return math.prod(self.shape[:-2])


@dataclass(frozen=True)
class Part:
    role: str
    #: runtime dtype names: the part's own dtype is authoritative
    dtypes: tuple[str, ...]
    shape: Callable[[View], Shape]
    required: bool = True


Decode = Callable[
    [Cores, Mapping[str, "Tensor"], Mapping[str, object], tuple[int, ...], "Tensor"], None
]
Sample = Callable[[ModuleType, View, "Tensor"], dict[str, "Tensor"]]


def _no_rule(view: View) -> str:
    return ""


def _no_fix(torch: ModuleType, view: View, parts: dict[str, Tensor]) -> None:
    return None


@dataclass(frozen=True)
class Adapter:
    """One scheme onto the cores."""

    scheme: str
    #: every param the decode reads, with its allowed values or a predicate
    params: Mapping[str, Rule]
    parts: tuple[Part, ...]
    #: (cores, parts, params, stored shape, destination of that shape): the core calls
    decode: Decode
    #: the params the probe qualifies the provider at
    probe_params: Mapping[str, object]
    #: any shape as one row (else rank 2 or more as [R, C])
    flat: bool = False
    #: conditions across parts: "" or why not
    rule: Callable[[View], str] = _no_rule
    #: makes random parts valid where values matter (tables, indices, shapes)
    fix: Callable[[ModuleType, View, dict[str, Tensor]], None] = _no_fix

    @property
    def required(self) -> frozenset[str]:
        return frozenset(p.role for p in self.parts if p.required)

    @property
    def optional(self) -> frozenset[str]:
        return frozenset(p.role for p in self.parts if not p.required)


# ---------------------------------------------------------------------------- the checks


def _allowed(rule: Rule, value: object) -> bool:
    if callable(rule):
        return bool(rule(value))
    return any(type(value) is type(a) and value == a for a in rule)


def params_rule(a: Adapter, params: Mapping[str, object]) -> str:
    """ "" when `params` are the row's: none unknown, every one present, each allowed."""
    unknown = sorted(set(params) - set(a.params))
    if unknown:
        return f"{a.scheme}: unknown params {unknown}"
    for key, rule in a.params.items():
        if key not in params:
            return f"{a.scheme}: param {key} is missing"
        if not _allowed(rule, params[key]):
            return f"{a.scheme}: param {key}={params[key]!r}"
    return ""


def stored_shape(params: Mapping[str, object], shape: Sequence[int]) -> tuple[int, ...]:
    """`S`: the logical shape, its last two axes swapped under `transposed: true` (spec §6)."""
    shape = tuple(int(x) for x in shape)
    if params.get("transposed") is True and len(shape) >= 2:
        return (*shape[:-2], shape[-1], shape[-2])
    return shape


def geometry_rule(a: Adapter, shape: Sequence[int], params: Mapping[str, object]) -> str:
    if not a.flat and len(shape) < 2:
        return f"{a.scheme} views rank 2 or more, not {tuple(shape)}"
    return ""


def parts_rule(
    a: Adapter,
    shape: Sequence[int],
    params: Mapping[str, object],
    parts: Mapping[str, tuple[str, tuple[int, ...]]],
) -> str:
    """ "" when the parts are the row's: roles accounted, dtypes and shapes by its rules."""
    why = params_rule(a, params) or geometry_rule(a, shape, params)
    if why:
        return why
    roles = {p.role for p in a.parts}
    if set(parts) - roles or a.required - set(parts):
        return f"{a.scheme}: parts {sorted(parts)}, requires {sorted(a.required)}"
    view = View(stored_shape(params, shape), params, parts, a.flat)
    for p in a.parts:
        if p.role not in parts:
            continue
        dtype, got = parts[p.role]
        names = p.dtypes
        try:
            kind, want = p.shape(view)
        except ValueError as e:
            return f"{a.scheme}: {e}"
        fits = tuple(got) == want if isinstance(want, tuple) else math.prod(got) == want
        if dtype not in names or not fits:
            return f"{a.scheme}: part {p.role} is {dtype} {list(got)}, wants {names} {kind} {want}"
    return a.rule(view)


def run(
    a: Adapter, cores: Cores, parts: Mapping[str, Tensor], params: Mapping[str, object], out: Tensor
) -> None:
    """Decode into `out` (the logical tensor): through the stored tensor `S` and back under
    transposed storage."""
    shape = stored_shape(params, out.shape)
    if not (params.get("transposed") is True and len(shape) >= 2):
        a.decode(cores, parts, params, shape, out)
        return
    stored = cores.torch.empty(shape, dtype=out.dtype, device=out.device)
    a.decode(cores, parts, params, shape, stored)
    out.copy_(stored.transpose(-1, -2))


def _random(
    torch: ModuleType, dtype: str, extents: tuple[int, ...], generator: Generator
) -> Tensor:
    """Random elements of `dtype`: every pattern of an integer part; finite, modest values of a
    float one (the probe scores a GEMM over the decoded weight, which a NaN or infinity spoils)."""
    n = math.prod(extents)
    if dtype in ("f32", "bf16", "f16"):
        values: Tensor = torch.randn(n, generator=generator) * 0.05
        return values.to(getattr(torch, TORCH[dtype])).reshape(extents)
    if dtype == "f8_e8m0":  # 2^-8 .. 2^1
        codes: Tensor = torch.randint(119, 129, (n,), generator=generator, dtype=torch.int32)
        return codes.to(torch.uint8).view(torch.float8_e8m0fnu).reshape(extents)
    width = getattr(torch, TORCH[dtype]).itemsize
    raw: Tensor = torch.randint(0, 256, (n * width,), generator=generator, dtype=torch.int32)
    if dtype == "f8_e4m3fn":  # exponents up to 7: every value below 2, no NaN
        raw = raw & 0xBF
    return raw.to(torch.uint8).view(getattr(torch, TORCH[dtype])).reshape(extents)


def sample(
    torch: ModuleType, a: Adapter, shape: Sequence[int], generator: Generator
) -> dict[str, Tensor]:
    """Random stored parts of a tensor of `shape` at the row's probe params, then the row's `fix`
    for the parts whose values the decode checks or that hold floats in their bytes."""
    params = a.probe_params
    view = View(stored_shape(params, shape), params, {}, a.flat)
    out: dict[str, Tensor] = {}
    for p in a.parts:
        if not p.required:
            continue
        dtype = p.dtypes[0]
        _, want = p.shape(view)
        extents = want if isinstance(want, tuple) else (want,)
        out[p.role] = _random(torch, dtype, extents, generator)
    a.fix(torch, view, out)
    return out


# --------------------------------------------------------------------------- shared math


def _f32(torch: ModuleType, t: Tensor) -> Tensor:
    """A part's elements as f32, exactly (E8M0 by its bits: `b << 23`, 0 the subnormal 2^-127,
    255 NaN)."""
    if t.dtype == torch.float8_e8m0fnu:
        raw = t.view(torch.uint8).reshape(-1).to(torch.int32)
        bits: Tensor = torch.where(raw == 0, torch.full_like(raw, 1 << 22), raw << 23)
        bits = torch.where(raw == 255, torch.full_like(bits, 0x7FC00000), bits)
        return bits.view(torch.float32)
    return t.reshape(-1).to(torch.float32)


def _codes(torch: ModuleType, dtype: str, device: Device) -> Tensor:
    """Every code of an 8-bit float dtype, as f32: the table its bytes index."""
    codes: Tensor = (
        torch.arange(256, dtype=torch.uint8)
        .view(getattr(torch, dtype))
        .to(torch.float32)
        .to(device)
    )
    return codes


def _table(torch: ModuleType, values: Sequence[float], device: Device) -> Tensor:
    table: Tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return table


def _u8(torch: ModuleType, t: Tensor) -> Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _count(n: int) -> Callable[[View], Shape]:
    return lambda v: ("count", n)


def _logical(v: View) -> Shape:
    return "exact", v.shape


def _f4(v: View) -> Shape:
    """An `F4` part of the stored shape, as its packed bytes."""
    if v.shape[-1] % 2:
        raise ValueError(f"F4 {list(v.shape)} is not whole bytes")
    return "exact", (*v.lead, v.shape[-1] // 2)


def _one_or_per_matrix(role: str) -> Callable[[View], Shape]:
    def shape(v: View) -> Shape:
        got = v.parts.get(role)
        return ("count", 1) if got is None or math.prod(got[1]) == 1 else ("count", v.matrices)

    return shape


def _intp(params: Mapping[str, object], key: str) -> int:
    """A param the row checked is an integer."""
    value = params[key]
    assert isinstance(value, int), key
    return value


def _pair(params: Mapping[str, object], key: str) -> tuple[int, int]:
    """A param the row checked is two integers."""
    value = params[key]
    assert isinstance(value, (list, tuple)) and len(value) == 2, key
    return int(value[0]), int(value[1])


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
SCALE_LAYOUTS = ("row_major", "cublas_128x4")


# ------------------------------------------------------------------------------- nvfp4


def _row_blocks(size: int, layout: str, per_matrix: bool) -> Callable[[View], Shape]:
    """Row blocks of `size`: `[lead..., W]` row-major; cuBLAS tiles per matrix of a stack."""

    def shape(v: View) -> Shape:
        w = -(-v.cols // size)
        if layout == "cublas_128x4" or v.params.get("scale_layout") == "cublas_128x4":
            stack = v.shape[:-2] if per_matrix else ()
            n = v.height if per_matrix else v.rows
            return "exact", (*stack, 128 * -(-n // 128), 4 * -(-w // 4))
        return "exact", (*v.lead, w)

    return shape


def _nvfp4_rule(v: View) -> str:
    return "" if v.cols % 16 == 0 else f"nvfp4: {v.cols} columns are not whole blocks of 16"


def _nvfp4(
    c: Cores,
    parts: Mapping[str, Tensor],
    params: Mapping[str, object],
    shape: tuple[int, ...],
    out: Tensor,
) -> None:
    torch = c.torch
    n, cols, m = shape[-2], shape[-1], math.prod(shape[:-2])
    words, scale = _u8(torch, parts["weight"]), _f32(torch, parts["scale"])
    second = _f32(torch, parts["scale_2"])
    layout, divide = str(params["scale_layout"]), params["global_scale"] == "divide"
    pack = Packing(4, 8, order=str(params["nibble_order"]))
    e2m1, post = _table(torch, E2M1, out.device), parts.get("pre_quant_scale")
    o = out.reshape(-1)
    if m == 1 or (layout == "row_major" and second.numel() == 1):
        c.table(
            words,
            pack,
            m * n,
            cols,
            e2m1,
            scale,
            blocks_of(16, layout),
            o,
            scale_2=second[:1],
            divide=divide,
            post=post,
        )
        return
    w, s, e = n * cols // 2, scale.numel() // m, n * cols
    for k in range(m):
        g = second[k : k + 1] if second.numel() > 1 else second[:1]
        c.table(
            words[k * w : (k + 1) * w],
            pack,
            n,
            cols,
            e2m1,
            scale[k * s : (k + 1) * s],
            blocks_of(16, layout),
            o[k * e : (k + 1) * e],
            scale_2=g,
            divide=divide,
            post=post,
        )


NVFP4 = Adapter(
    "nvfp4",
    {
        "block_size": (16,),
        "global_scale": ("multiply", "divide"),
        "nibble_order": ("low_first", "high_first"),
        "scale_layout": SCALE_LAYOUTS,
    },
    (
        Part("weight", ("u8",), _f4),
        Part("scale", ("f8_e4m3fn",), _row_blocks(16, "", True)),
        Part("scale_2", ("f32",), _one_or_per_matrix("scale_2")),
        Part("input_scale", ("f32",), _one_or_per_matrix("input_scale"), required=False),
        Part(
            "pre_quant_scale", ("f32", "bf16", "f16"), lambda v: ("count", v.cols), required=False
        ),
    ),
    _nvfp4,
    {
        "block_size": 16,
        "global_scale": "divide",
        "nibble_order": "high_first",
        "scale_layout": "cublas_128x4",
    },
    rule=_nvfp4_rule,
)

# --------------------------------------------------------------------------- svdq.nvfp4


def _svdq_rule(v: View) -> str:
    """nvfp4's blocks, one matrix, and the two low-rank factors of one rank."""
    if len(v.shape) != 2:
        return f"svdq.nvfp4 is one matrix, not {list(v.shape)}"
    down, up = v.parts.get("lowrank_down"), v.parts.get("lowrank_up")
    if down is not None and up is not None and down[1][0] != up[1][-1]:
        return (
            f"svdq.nvfp4: lowrank_down {list(down[1])} and lowrank_up {list(up[1])} differ in rank"
        )
    return _nvfp4_rule(v)


def _rank(v: View, role: str, axis: int) -> int:
    """A factor's rank, as stored; 32 (the producer's) for a part not yet drawn (`sample`)."""
    got = v.parts.get(role)
    if got is None:
        return 32
    if len(got[1]) != 2 or got[1][axis] < 1:
        raise ValueError(f"{role} is not a rank-2 factor")
    return int(got[1][axis])


def _svdq(
    c: Cores,
    parts: Mapping[str, Tensor],
    params: Mapping[str, object],
    shape: tuple[int, ...],
    out: Tensor,
) -> None:
    """nvfp4's value `((T[q] x s) x g)`, `x c_i` (`channel_scale`), `x p_j` (`pre_quant_scale`),
    then `+ up[i, r] x down[r, j]` in r order, in f32, rounded once."""
    torch = c.torch
    n, cols = shape
    value = torch.empty(n * cols, dtype=torch.float32, device=out.device)
    c.table(
        _u8(torch, parts["weight"]),
        Packing(4, 8, order=str(params["nibble_order"])),
        n,
        cols,
        _table(torch, E2M1, out.device),
        _f32(torch, parts["scale"]),
        blocks_of(16, str(params["scale_layout"])),
        value,
        scale_2=_f32(torch, parts["scale_2"])[:1],
        divide=params["global_scale"] == "divide",
    )
    value = value.view(n, cols)
    if "channel_scale" in parts:
        value = value * _f32(torch, parts["channel_scale"]).view(n, 1)
    value = value * _f32(torch, parts["pre_quant_scale"]).view(1, cols)
    up = parts["lowrank_up"].to(torch.float32).reshape(n, -1)
    down = parts["lowrank_down"].to(torch.float32).reshape(up.shape[1], cols)
    term = torch.empty_like(value)
    for r in range(up.shape[1]):  # each product rounded to f32, then added: the law's order
        value.add_(torch.mul(up[:, r : r + 1], down[r : r + 1, :], out=term))
    round_into(torch, value, out.reshape(-1))


SVDQ_NVFP4 = Adapter(
    "svdq.nvfp4",
    NVFP4.params,
    (
        *NVFP4.parts[:4],
        Part("pre_quant_scale", ("f32", "bf16", "f16"), lambda v: ("count", v.cols)),
        Part("channel_scale", ("f32", "f8_e4m3fn"), lambda v: ("count", v.rows), required=False),
        Part(
            "lowrank_down",
            ("bf16", "f16", "f32"),
            lambda v: ("exact", (_rank(v, "lowrank_down", 0), v.cols)),
        ),
        Part(
            "lowrank_up",
            ("bf16", "f16", "f32"),
            lambda v: ("exact", (v.rows, _rank(v, "lowrank_up", 1))),
        ),
    ),
    _svdq,
    {**NVFP4.probe_params, "scale_layout": "row_major"},
    rule=_svdq_rule,
)

# ------------------------------------------------------------------------------- mxfp8


def _per_matrix_table(
    c: Cores,
    words: Tensor,
    pack: Packing,
    shape: tuple[int, ...],
    values: Tensor,
    scale: Tensor,
    size: int,
    layout: str,
    out: Tensor,
) -> None:
    """`T[q] x s` on row blocks of `size`: one call, or one per matrix of a stack when cuBLAS
    tiles each matrix on its own (spec §6)."""
    n, cols, m = shape[-2], shape[-1], math.prod(shape[:-2])
    if m == 1 or layout == "row_major":
        c.table(words, pack, m * n, cols, values, scale, blocks_of(size, layout), out.reshape(-1))
        return
    w, s, e = words.numel() // m, scale.numel() // m, n * cols
    o = out.reshape(-1)
    for k in range(m):
        c.table(
            words[k * w : (k + 1) * w],
            pack,
            n,
            cols,
            values,
            scale[k * s : (k + 1) * s],
            blocks_of(size, layout),
            o[k * e : (k + 1) * e],
        )


def _mxfp8(
    c: Cores,
    parts: Mapping[str, Tensor],
    params: Mapping[str, object],
    shape: tuple[int, ...],
    out: Tensor,
) -> None:
    torch = c.torch
    weight = parts["weight"]
    values = _codes(torch, str(weight.dtype).removeprefix("torch."), out.device)
    _per_matrix_table(
        c,
        _u8(torch, weight),
        Packing(8, 8),
        shape,
        values,
        _f32(torch, parts["scale"]),
        32,
        str(params["scale_layout"]),
        out,
    )


MXFP8 = Adapter(
    "mxfp8",
    {"block_size": (32,), "scale_layout": SCALE_LAYOUTS},
    (
        Part("weight", ("f8_e4m3fn", "f8_e5m2"), _logical),
        Part("scale", ("f8_e8m0",), _row_blocks(32, "", True)),
    ),
    _mxfp8,
    {"block_size": 32, "scale_layout": "cublas_128x4"},
)

# ----------------------------------------------------------------- fp8.blockwise, mxfp4


def _block_size(v: object) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(type(n) is int and n > 0 for n in v)


def _tiles(v: View) -> Shape:
    br, bc = _pair(v.params, "block_size")
    return "exact", (*v.shape[:-2], -(-v.height // br), -(-v.cols // bc))


def _fp8_blockwise(
    c: Cores,
    parts: Mapping[str, Tensor],
    params: Mapping[str, object],
    shape: tuple[int, ...],
    out: Tensor,
) -> None:
    torch = c.torch
    br, bc = _pair(params, "block_size")
    n, cols, m = shape[-2], shape[-1], math.prod(shape[:-2])
    tiles = _f32(torch, parts["scale"]).reshape(m, -(-n // br), -(-cols // bc))
    rows = tiles.repeat_interleave(br, dim=1)[:, :n].reshape(m * n, -1)  # each row its tile row
    c.table(
        _u8(torch, parts["weight"]),
        Packing(8, 8),
        m * n,
        cols,
        _codes(torch, "float8_e4m3fn", out.device),
        rows,
        blocks_of(bc),
        out,
    )


FP8_BLOCKWISE = Adapter(
    "fp8.blockwise",
    {"block_size": _block_size},
    (
        Part("weight", ("f8_e4m3fn",), _logical),
        Part("scale", ("f8_e8m0", "f32", "bf16", "f16"), _tiles),
    ),
    _fp8_blockwise,
    {"block_size": [128, 128]},
)


def _mxfp4(
    c: Cores,
    parts: Mapping[str, Tensor],
    params: Mapping[str, object],
    shape: tuple[int, ...],
    out: Tensor,
) -> None:
    torch = c.torch
    cols = shape[-1]
    c.table(
        _u8(torch, parts["weight"]),
        Packing(4, 8),
        math.prod(shape) // cols,
        cols,
        _table(torch, E2M1, out.device),
        _f32(torch, parts["scale"]),
        blocks_of(32),
        out,
    )


MXFP4 = Adapter(
    "mxfp4",
    {"block_size": (32,), "transposed": (False, True)},
    (
        Part("weight", ("u8",), _f4),
        Part("scale", ("f8_e8m0",), _row_blocks(32, "row_major", False)),
    ),
    _mxfp4,
    {"block_size": 32, "transposed": True},
)

# ------------------------------------------------------------------------------- gguf.*

#: ggml type: (elements, bytes) per struct
GGUF = {
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "mxfp4": (32, 17),
    "nvfp4": (64, 36),
    "q1_0": (128, 18),
    "q2_0": (64, 18),
}
#: ggml's kvalues for FP4: E2M1 doubled (spec Appendix A.2)
KVALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

if HAS_TRITON:

    @triton.jit
    def _u32(base, at, mask):  # type: ignore[no-untyped-def]
        b0 = dc.u8_at(base, at, mask)
        b1 = dc.u8_at(base, at + 1, mask)
        b2 = dc.u8_at(base, at + 2, mask)
        b3 = dc.u8_at(base, at + 3, mask)
        return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.uint32, bitcast=True)

    @triton.jit
    def _pow2(x):  # type: ignore[no-untyped-def]
        """2^x exactly, x in -149..127."""
        bits = tl.where(
            x >= -126, (x + 127) << 23, 1 << (tl.minimum(tl.maximum(x, -149), -127) + 149)
        )
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _kvalue(q):  # type: ignore[no-untyped-def]
        """ggml's kvalues_fp4[q]: E2M1 doubled; code 8 is +0."""
        m = q & 7
        mag = tl.where(m < 4, m, (2 + (m & 1)) << tl.maximum((m >> 1) - 1, 0))
        return tl.where(q >= 8, -mag, mag).to(tl.float32)

    @triton.jit
    def _nibble(base, at, n, el, mask):  # type: ignore[no-untyped-def]
        """q4_0's order: element j of 2n in byte at + j % n, low nibble first half."""
        byte = dc.u8_at(base, at + el % n, mask)
        return tl.where(el < n, byte & 15, byte >> 4)

    @triton.jit
    def _q4_0(base, el, mask):  # type: ignore[no-untyped-def]
        return (_nibble(base, 2, 16, el, mask) - 8).to(tl.float32) * dc.f16_at(base, 0, mask)

    @triton.jit
    def _q4_1(base, el, mask):  # type: ignore[no-untyped-def]
        q = _nibble(base, 4, 16, el, mask).to(tl.float32)
        return q * dc.f16_at(base, 0, mask) + dc.f16_at(base, 2, mask)

    @triton.jit
    def _q5_0(base, el, mask):  # type: ignore[no-untyped-def]
        high = ((_u32(base, 2, mask) >> el.to(tl.uint32)) & 1).to(tl.int32)
        q = _nibble(base, 6, 16, el, mask) + 16 * high
        return (q - 16).to(tl.float32) * dc.f16_at(base, 0, mask)

    @triton.jit
    def _q5_1(base, el, mask):  # type: ignore[no-untyped-def]
        high = ((_u32(base, 4, mask) >> el.to(tl.uint32)) & 1).to(tl.int32)
        q = (_nibble(base, 8, 16, el, mask) + 16 * high).to(tl.float32)
        return q * dc.f16_at(base, 0, mask) + dc.f16_at(base, 2, mask)

    @triton.jit
    def _q8_0(base, el, mask):  # type: ignore[no-untyped-def]
        return dc.i8_at(base, 2 + el, mask).to(tl.float32) * dc.f16_at(base, 0, mask)

    @triton.jit
    def _low2(base, at, el, mask):  # type: ignore[no-untyped-def]
        """q2_k/q3_k: element t's 2-bit code in qs at `at`."""
        h = el // 128
        j = (el % 128) // 32
        half = (el % 32) // 16
        return (dc.u8_at(base, at + 32 * h + 16 * half + el % 16, mask) >> (2 * j)) & 3

    @triton.jit
    def _q2_k(base, el, mask):  # type: ignore[no-untyped-def]
        q = _low2(base, 16, el, mask).to(tl.float32)
        sc = dc.u8_at(base, el // 16, mask)
        dl = dc.f16_at(base, 80, mask) * (sc & 15).to(tl.float32)
        ml = dc.f16_at(base, 82, mask) * (sc >> 4).to(tl.float32)
        return dl * q - ml

    @triton.jit
    def _q3_k(base, el, mask):  # type: ignore[no-untyped-def]
        q = _low2(base, 32, el, mask)
        at = 16 * ((el % 32) // 16) + el % 16
        high = (dc.u8_at(base, at, mask) >> (el // 32)) & 1
        b = el // 16
        raw = dc.u8_at(base, 96 + b % 8, mask)
        lo = tl.where(b < 8, raw & 15, raw >> 4)
        sc = (lo | (((dc.u8_at(base, 104 + b % 4, mask) >> (2 * (b // 4))) & 3) << 4)) - 32
        dl = dc.f16_at(base, 108, mask) * sc.to(tl.float32)
        return dl * (q - 4 + 4 * high).to(tl.float32)

    @triton.jit  # type: ignore[untyped-decorator]
    def _k_quant(base, el, mask, QS: tl.constexpr, QH: tl.constexpr):  # type: ignore[no-untyped-def]
        """q4_k (QH 0) and q5_k: 8 sub-blocks of 32, 6-bit scale and min (get_scale_min_k4)."""
        a = el // 32
        ai = a % 4
        x = dc.u8_at(base, 4 + ai, mask)
        y = dc.u8_at(base, 8 + ai, mask)
        z = dc.u8_at(base, 12 + ai, mask)
        c = tl.where(a < 4, x & 63, (z & 15) | ((x >> 6) << 4))
        e = tl.where(a < 4, y & 63, (z >> 4) | ((y >> 6) << 4))
        byte = dc.u8_at(base, QS + 32 * (a // 2) + el % 32, mask)
        q = tl.where(a % 2 == 0, byte & 15, byte >> 4)
        if QH:
            q = q + 16 * ((dc.u8_at(base, 16 + el % 32, mask) >> a) & 1)
        dl = dc.f16_at(base, 0, mask) * c.to(tl.float32)
        ml = dc.f16_at(base, 2, mask) * e.to(tl.float32)
        return dl * q.to(tl.float32) - ml

    @triton.jit
    def _q4_k(base, el, mask):  # type: ignore[no-untyped-def]
        return _k_quant(base, el, mask, 16, 0)

    @triton.jit
    def _q5_k(base, el, mask):  # type: ignore[no-untyped-def]
        return _k_quant(base, el, mask, 48, 1)

    @triton.jit
    def _q6_k(base, el, mask):  # type: ignore[no-untyped-def]
        h = el // 128
        j = (el % 128) // 32
        lane = el % 32
        byte = dc.u8_at(base, 64 * h + 32 * (j % 2) + lane, mask)
        low = tl.where(j < 2, byte & 15, byte >> 4)
        q = (low | (((dc.u8_at(base, 128 + 32 * h + lane, mask) >> (2 * j)) & 3) << 4)) - 32
        sc = dc.i8_at(base, 192 + 8 * h + lane // 16 + 2 * j, mask).to(tl.float32)
        return dc.f16_at(base, 208, mask) * sc * q.to(tl.float32)

    @triton.jit
    def _mxfp4_struct(base, el, mask):  # type: ignore[no-untyped-def]
        d = _pow2(dc.u8_at(base, 0, mask) - 128)
        return _kvalue(_nibble(base, 1, 16, el, mask)) * d

    @triton.jit
    def _ue4m3(b):  # type: ignore[no-untyped-def]
        """ggml's UE4M3: no sign, no NaN, 0x7F zero."""
        e = (b >> 3) & 15
        m = (b & 7).to(tl.float32)
        v = tl.where(e == 0, m * _pow2(e * 0 - 9), (m + 8.0) * _pow2(e - 10))
        return tl.where(b == 0x7F, 0.0, v)

    @triton.jit
    def _nvfp4_struct(base, el, mask):  # type: ignore[no-untyped-def]
        k = el // 16
        d = _ue4m3(dc.u8_at(base, k, mask)) * 0.5
        return _kvalue(_nibble(base, 4 + 8 * k, 8, el % 16, mask)) * d

    @triton.jit
    def _q1_0(base, el, mask):  # type: ignore[no-untyped-def]
        bit = (dc.u8_at(base, 2 + el // 8, mask) >> (el % 8)) & 1
        d = dc.f16_at(base, 0, mask)
        return tl.where(bit == 1, d, -d)

    @triton.jit
    def _q2_0(base, el, mask):  # type: ignore[no-untyped-def]
        q = (dc.u8_at(base, 2 + el // 4, mask) >> (2 * (el % 4))) & 3
        return (q - 1).to(tl.float32) * dc.f16_at(base, 0, mask)

    GGUF_KERNELS: dict[str, object] = {
        "q4_0": _q4_0,
        "q4_1": _q4_1,
        "q5_0": _q5_0,
        "q5_1": _q5_1,
        "q8_0": _q8_0,
        "q2_k": _q2_k,
        "q3_k": _q3_k,
        "q4_k": _q4_k,
        "q5_k": _q5_k,
        "q6_k": _q6_k,
        "mxfp4": _mxfp4_struct,
        "nvfp4": _nvfp4_struct,
        "q1_0": _q1_0,
        "q2_0": _q2_0,
    }
else:
    GGUF_KERNELS = {}


def _gguf_eager(name: str) -> Callable[[ModuleType, Tensor], Tensor]:
    """ggml's dequantize_row for `name`, every struct of `[count, size]` u8 at once, f32 in ggml's
    order of operations."""

    def decode(torch: ModuleType, s: Tensor) -> Tensor:
        f32: Dtype = torch.float32
        i: Tensor = s.to(torch.int32)

        def f16(at: int) -> Tensor:
            return s[:, at : at + 2].contiguous().view(torch.float16).to(f32)

        def u32(at: int) -> Tensor:
            return s[:, at : at + 4].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF

        def nibbles(at: int, n: int) -> Tensor:
            qs = i[:, at : at + n]
            joined: Tensor = torch.cat((qs & 15, qs >> 4), dim=1)
            return joined

        def kvalue(q: Tensor) -> Tensor:
            table: Tensor = torch.tensor(KVALUES, dtype=f32, device=s.device)
            return table[q]

        def pow2(x: Tensor) -> Tensor:
            bits: Tensor = torch.where(x >= -126, (x + 127) << 23, 1 << (x.clamp(-149, -127) + 149))
            return bits.to(torch.int32).view(f32)

        t: Tensor = torch.arange(GGUF[name][0], device=s.device)
        if name in ("q4_0", "q4_1", "q5_0", "q5_1"):
            at, base = {"q4_0": (2, 0), "q4_1": (4, 0), "q5_0": (6, 2), "q5_1": (8, 4)}[name]
            q: Tensor = nibbles(at, 16)
            if name in ("q5_0", "q5_1"):
                q = q + 16 * ((u32(base)[:, :1] >> t) & 1).to(torch.int32)
            if name in ("q4_0", "q5_0"):
                return (q - (8 if name == "q4_0" else 16)).to(f32) * f16(0)
            return q.to(f32) * f16(0) + f16(2)
        if name == "q8_0":
            return s[:, 2:34].contiguous().view(torch.int8).to(f32) * f16(0)
        if name in ("q2_k", "q3_k"):
            h, j, half, lane = t // 128, (t % 128) // 32, (t % 32) // 16, t % 16
            at = 16 if name == "q2_k" else 32
            q = (i[:, at + 32 * h + 16 * half + lane] >> (2 * j)) & 3
            b = t // 16
            if name == "q2_k":
                sc = i[:, b]
                dl = f16(80) * (sc & 15).to(f32)
                ml = f16(82) * (sc >> 4).to(f32)
                return dl * q.to(f32) - ml
            high = (i[:, 16 * half + lane] >> (t // 32)) & 1
            raw = i[:, 96 + b % 8]
            lo: Tensor = torch.where(b < 8, raw & 15, raw >> 4)
            sc = (lo | (((i[:, 104 + b % 4] >> (2 * (b // 4))) & 3) << 4)) - 32
            return (f16(108) * sc.to(f32)) * (q - 4 + 4 * high).to(f32)
        if name in ("q4_k", "q5_k"):
            a = t // 32
            ai = a % 4
            x, y, z = i[:, 4 + ai], i[:, 8 + ai], i[:, 12 + ai]
            c: Tensor = torch.where(a < 4, x & 63, (z & 15) | ((x >> 6) << 4))
            e: Tensor = torch.where(a < 4, y & 63, (z >> 4) | ((y >> 6) << 4))
            byte = i[:, (16 if name == "q4_k" else 48) + 32 * (a // 2) + t % 32]
            q = torch.where(a % 2 == 0, byte & 15, byte >> 4)
            if name == "q5_k":
                q = q + 16 * ((i[:, 16 + t % 32] >> a) & 1)
            return (f16(0) * c.to(f32)) * q.to(f32) - f16(2) * e.to(f32)
        if name == "q6_k":
            h, j, lane = t // 128, (t % 128) // 32, t % 32
            byte = i[:, 64 * h + 32 * (j % 2) + lane]
            low: Tensor = torch.where(j < 2, byte & 15, byte >> 4)
            q = (low | (((i[:, 128 + 32 * h + lane] >> (2 * j)) & 3) << 4)) - 32
            sc = s[:, 192:208].contiguous().view(torch.int8).to(f32)[:, 8 * h + lane // 16 + 2 * j]
            return f16(208) * sc * q.to(f32)
        if name == "mxfp4":
            return kvalue(nibbles(1, 16)) * pow2(i[:, :1] - 128)
        if name == "nvfp4":
            b = i[:, 0:4]
            e, m = (b >> 3) & 15, (b & 7).to(f32)
            d: Tensor = torch.where(
                e == 0, m * pow2(torch.full_like(e, -9)), (m + 8.0) * pow2(e - 10)
            )
            d = torch.where(b == 0x7F, torch.zeros_like(d), d) * 0.5
            q = torch.cat([nibbles(4 + 8 * k, 8) for k in range(4)], dim=1)
            return kvalue(q) * d.repeat_interleave(16, dim=1)
        if name == "q1_0":
            bit = (i[:, 2 + t // 8] >> (t % 8)) & 1
            d = f16(0)
            selected: Tensor = torch.where(bit == 1, d, -d)
            return selected
        q = (i[:, 2 + t // 4] >> (2 * (t % 4))) & 3  # q2_0
        return (q - 1).to(f32) * f16(0)

    return decode


def _gguf_blocks(name: str) -> Callable[[View], Shape]:
    n, size = GGUF[name]

    def shape(v: View) -> Shape:
        numel = math.prod(v.shape)
        if numel % n:
            raise ValueError(f"{list(v.shape)} is not whole blocks of {n}")
        if v.shape and v.shape[-1] % n == 0:
            return "exact", (*v.shape[:-1], v.shape[-1] // n, size)
        return "exact", (numel // n, size)

    return shape


#: Each struct's F16 fields (byte offsets): finite in a sample.
GGUF_F16 = {
    "q4_0": (0,),
    "q4_1": (0, 2),
    "q5_0": (0,),
    "q5_1": (0, 2),
    "q8_0": (0,),
    "q2_k": (80, 82),
    "q3_k": (108,),
    "q4_k": (0, 2),
    "q5_k": (0, 2),
    "q6_k": (208,),
    "mxfp4": (),
    "nvfp4": (),
    "q1_0": (0,),
    "q2_0": (0,),
}


def _gguf_fix(name: str) -> Callable[[ModuleType, View, dict[str, Tensor]], None]:
    def fix(torch: ModuleType, v: View, parts: dict[str, Tensor]) -> None:
        size = GGUF[name][1]
        structs = parts["blocks"].reshape(-1, size)
        for at in GGUF_F16[name]:
            d = (torch.randn(structs.shape[0]) * 0.05).to(torch.float16)
            structs[:, at : at + 2] = d.view(torch.uint8).reshape(-1, 2)
        if name == "mxfp4":  # its E8M0 exponent, near 1
            structs[:, 0] = torch.randint(119, 129, (structs.shape[0],)).to(torch.uint8)

    return fix


def _gguf(name: str) -> Adapter:
    n, size = GGUF[name]

    def decode(
        c: Cores,
        parts: Mapping[str, Tensor],
        params: Mapping[str, object],
        shape: tuple[int, ...],
        out: Tensor,
    ) -> None:
        c.blocks(
            _u8(c.torch, parts["blocks"]),
            n,
            size,
            out,
            kernel_fn=GGUF_KERNELS.get(name),
            eager_fn=_gguf_eager(name),
            post=parts.get("scale"),
        )

    parts: tuple[Part, ...] = (Part("blocks", ("u8",), _gguf_blocks(name)),)
    if name == "nvfp4":
        parts += (
            Part("scale", ("f32",), _count(1), required=False),
            Part("input_scale", ("f32",), _count(1), required=False),
        )
    return Adapter(
        f"gguf.{name}",
        {"block_size": (n,)},
        parts,
        decode,
        {"block_size": n},
        flat=True,
        fix=_gguf_fix(name),
    )


# -------------------------------------------------------------------------------- bnb.*

NF4 = (
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.091050036,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
)
FP4 = (
    0.0,
    0.0052083335,
    0.6666667,
    1.0,
    0.33333334,
    0.5,
    0.16666667,
    0.25,
    -0.0,
    -0.0052083335,
    -0.6666667,
    -1.0,
    -0.33333334,
    -0.5,
    -0.16666667,
    -0.25,
)
BNB_BLOCKS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
NESTED = ("nested_absmax", "nested_quant_map", "nested_offset")


def _bnb_blocks(v: View) -> int:
    return -(-v.cols // _intp(v.params, "block_size"))


def _bnb_rule(v: View) -> str:
    double = v.params["double_quant"] is True
    if (v.parts["absmax"][0] == "u8") != double:
        return "bnb: absmax is U8 codes exactly when double_quant"
    if {r for r in NESTED if r in v.parts} != (set(NESTED) if double else set()):
        return "bnb: the nested parts are present exactly when double_quant"
    return ""


def _bnb(table: tuple[float, ...]) -> Decode:
    def decode(
        c: Cores,
        parts: Mapping[str, Tensor],
        params: Mapping[str, object],
        shape: tuple[int, ...],
        out: Tensor,
    ) -> None:
        torch = c.torch
        values = _table(torch, table, out.device)
        if not torch.equal(parts["quant_map"].reshape(-1).to(torch.float32), values):
            raise ValueError("bnb: quant_map is not the scheme's table")
        absmax = parts["absmax"]
        if params["double_quant"]:
            codes = _u8(torch, absmax).to(torch.int64)
            group = torch.arange(codes.numel(), device=codes.device) // 256
            nested = (
                _f32(torch, parts["nested_quant_map"])[codes]
                * _f32(torch, parts["nested_absmax"])[group]
            )
            scale = nested + _f32(torch, parts["nested_offset"])[0]
        else:
            scale = _f32(torch, absmax)
        n = math.prod(shape)
        c.table(
            _u8(torch, parts["weight"]),
            Packing(4, 8, order="high_first"),
            1,
            n,
            values,
            scale,
            blocks_of(_intp(params, "block_size")),
            out,
        )

    return decode


def _bnb_fix(table: tuple[float, ...]) -> Callable[[ModuleType, View, dict[str, Tensor]], None]:
    def fix(torch: ModuleType, v: View, parts: dict[str, Tensor]) -> None:
        parts["quant_map"] = torch.tensor(table, dtype=torch.float32)

    return fix


def _bnb_four_bit(name: str, table: tuple[float, ...]) -> Adapter:
    return Adapter(
        name,
        {"block_size": BNB_BLOCKS, "double_quant": (False, True)},
        (
            Part("weight", ("u8",), lambda v: ("exact", (-(-v.cols // 2), 1))),
            Part("absmax", ("f32", "u8"), lambda v: ("exact", (_bnb_blocks(v),))),
            Part("quant_map", ("f32",), lambda v: ("exact", (16,))),
            Part(
                "nested_absmax",
                ("f32",),
                lambda v: ("exact", (-(-_bnb_blocks(v) // 256),)),
                required=False,
            ),
            Part("nested_quant_map", ("f32",), lambda v: ("exact", (256,)), required=False),
            Part("nested_offset", ("f32",), _count(1), required=False),
        ),
        _bnb(table),
        {"block_size": 64, "double_quant": False},
        flat=True,
        rule=_bnb_rule,
        fix=_bnb_fix(table),
    )


#: The f32 nearest 1/127, bits 0x3C010204: bnb.int8 multiplies by it.
R127 = 0.007874015718698502


def _bnb_int8(
    c: Cores,
    parts: Mapping[str, Tensor],
    params: Mapping[str, object],
    shape: tuple[int, ...],
    out: Tensor,
) -> None:
    torch = c.torch
    if "weight_format" in parts and int(_u8(torch, parts["weight_format"])[0]) != 0:
        raise ValueError("bnb.int8: weight_format is not 0 (row)")
    c.affine(
        _u8(torch, parts["weight"]),
        Packing(8, 8, signed=True),
        shape[0],
        shape[1],
        parts["SCB"],
        ROW,
        out,
        post=torch.tensor([R127], dtype=torch.float32, device=out.device),
    )


BNB_INT8 = Adapter(
    "bnb.int8",
    {},
    (
        Part("weight", ("i8",), _logical),
        Part("SCB", ("f32",), lambda v: ("count", v.rows)),
        Part("weight_format", ("u8",), _count(1), required=False),
    ),
    _bnb_int8,
    {},
    rule=lambda v: "" if len(v.shape) == 2 else f"bnb.int8: shape {list(v.shape)} is not [N, K]",
)

# ---------------------------------------------------------------------- gptq, awq, ct

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _group_size(v: object) -> bool:
    return type(v) is int and (v == -1 or 1 <= v <= 2**31)


def _group(v: View) -> int:
    g = _intp(v.params, "group_size")
    return max(v.cols, 1) if g == -1 else g


def _groups(v: View) -> int:
    return -(-v.cols // _group(v))


def _gptq(bits: int) -> Adapter:
    codes = Packing(bits, 32, transposed=True, axis=0, fill="dense")
    zeros = Packing(bits, 32, fill="dense")

    def decode(
        c: Cores,
        parts: Mapping[str, Tensor],
        params: Mapping[str, object],
        shape: tuple[int, ...],
        out: Tensor,
    ) -> None:
        torch = c.torch
        v = View(shape, params, {})
        g, w, rows, cols = _group(v), _groups(v), v.rows, v.cols
        z = (unpack(torch, parts["qzeros"], zeros, w, rows) + _intp(params, "zero_offset")) % (
            1 << bits
        )
        if params["symmetric"] and bool((z != 1 << (bits - 1)).any()):
            raise ValueError(f"gptq: symmetric, and a zero is not {1 << (bits - 1)}")
        g_idx = parts["g_idx"].reshape(-1).to(torch.int64)
        if not params["act_order"] and not torch.equal(
            g_idx, torch.arange(cols, device=g_idx.device) // g
        ):
            raise ValueError("gptq: act_order is false, and g_idx is not each column's j // g")
        if cols and (int(g_idx.min()) < 0 or int(g_idx.max()) >= w):
            raise ValueError(f"gptq: g_idx outside {w} groups")
        c.affine(
            parts["qweight"],
            codes,
            rows,
            cols,
            parts["scales"],
            blocks_of(g, "transposed"),
            out,
            zero=z.reshape(-1).to(torch.float32),
            g_idx=g_idx.to(torch.int32),
        )

    def fix(torch: ModuleType, v: View, parts: dict[str, Tensor]) -> None:
        parts["g_idx"] = (torch.arange(v.cols) // _group(v)).to(torch.int32)

    return Adapter(
        f"gptq.int{bits}",
        {
            "group_size": _group_size,
            "symmetric": (True, False),
            "act_order": (True, False),
            "zero_offset": (0, 1),
        },
        (
            Part("qweight", ("i32",), lambda v: ("exact", (codes.words(v.cols), v.rows))),
            Part("qzeros", ("i32",), lambda v: ("exact", (_groups(v), zeros.words(v.rows)))),
            Part("scales", ("f16", "bf16"), lambda v: ("exact", (_groups(v), v.rows))),
            Part("g_idx", ("i32",), lambda v: ("exact", (v.cols,))),
        ),
        decode,
        {"group_size": 128, "symmetric": False, "act_order": False, "zero_offset": 1},
        fix=fix,
    )


AWQ_CODES = Packing(4, 32, transposed=True, axis=1, order=AWQ_ORDER)
AWQ_ZEROS = Packing(4, 32, order=AWQ_ORDER)


def _awq(
    c: Cores,
    parts: Mapping[str, Tensor],
    params: Mapping[str, object],
    shape: tuple[int, ...],
    out: Tensor,
) -> None:
    torch = c.torch
    v = View(shape, params, {})
    z = unpack(torch, parts["qzeros"], AWQ_ZEROS, _groups(v), v.rows).reshape(-1).to(torch.float32)
    c.affine(
        parts["qweight"],
        AWQ_CODES,
        v.rows,
        v.cols,
        parts["scales"],
        blocks_of(_group(v), "transposed"),
        out,
        zero=z,
    )


AWQ_INT4 = Adapter(
    "awq.int4",
    {"group_size": _group_size},
    (
        Part("qweight", ("i32",), lambda v: ("exact", (v.cols, AWQ_CODES.words(v.rows)))),
        Part("qzeros", ("i32",), lambda v: ("exact", (_groups(v), AWQ_ZEROS.words(v.rows)))),
        Part("scales", ("f16", "bf16"), lambda v: ("exact", (_groups(v), v.rows))),
    ),
    _awq,
    {"group_size": 128},
)


def _ct_packing(bits: int, params: Mapping[str, object], axis: int) -> Packing:
    return Packing(bits, 32, axis=axis, fill=str(params["packing"]), offset=1 << (bits - 1))


def _ct_rule(v: View) -> str:
    if v.params["strategy"] == "channel" and v.params["group_size"] != v.cols:
        return f"ct: channel, and group_size {v.params['group_size']} is not {v.cols}"
    if (v.params["symmetric"] is True) == ("weight_zero_point" in v.parts):
        return "ct: weight_zero_point is present exactly when not symmetric"
    return ""


def _ct(bits: int) -> Adapter:
    def groups(v: View) -> int:
        return -(-v.cols // _intp(v.params, "group_size"))

    def decode(
        c: Cores,
        parts: Mapping[str, Tensor],
        params: Mapping[str, object],
        shape: tuple[int, ...],
        out: Tensor,
    ) -> None:
        torch = c.torch
        v = View(shape, params, {})
        if parts["weight_shape"].reshape(-1).tolist() != list(shape):
            raise ValueError(f"ct: weight_shape is not {list(shape)}")
        g, w = _intp(params, "group_size"), groups(v)
        index = None
        if "weight_g_idx" in parts:
            idx = parts["weight_g_idx"].reshape(-1).to(torch.int64)
            if not bool((idx == -1).any()):
                want = torch.full((w,), g, device=idx.device)
                if w:
                    want[-1] = v.cols - (w - 1) * g
                if (idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= w)) or not torch.equal(
                    torch.bincount(idx, minlength=w), want
                ):
                    raise ValueError("ct: weight_g_idx does not give each group its g columns")
                index = idx.to(torch.int32)
        zero = None
        if "weight_zero_point" in parts:
            n, m, p = v.height, v.matrices, _ct_packing(bits, params, 0)
            stack = parts["weight_zero_point"].reshape(m, -1)
            zero = torch.cat([unpack(torch, stack[k], p, n, w).reshape(-1) for k in range(m)])
            zero = zero.to(torch.float32)
        c.affine(
            parts["weight_packed"],
            _ct_packing(bits, params, 1),
            v.rows,
            v.cols,
            parts["weight_scale"],
            blocks_of(g),
            out,
            zero=zero,
            g_idx=index,
        )

    def fix(torch: ModuleType, v: View, parts: dict[str, Tensor]) -> None:
        parts["weight_shape"] = torch.tensor(v.shape, dtype=torch.int32)

    return Adapter(
        f"ct.int{bits}",
        {
            "group_size": lambda x: type(x) is int and 1 <= x <= 2**31,
            "strategy": ("group", "channel"),
            "symmetric": (True, False),
            "packing": ("word", "dense"),
        },
        (
            Part(
                "weight_packed",
                ("i32",),
                lambda v: ("exact", (*v.lead, _ct_packing(bits, v.params, 1).words(v.cols))),
            ),
            Part("weight_scale", ("f32", "bf16", "f16"), lambda v: ("exact", (*v.lead, groups(v)))),
            Part(
                "weight_zero_point",
                ("i32",),
                lambda v: (
                    "exact",
                    (*v.shape[:-2], _ct_packing(bits, v.params, 0).words(v.height), groups(v)),
                ),
                required=False,
            ),
            Part("weight_shape", ("i32", "i64"), lambda v: ("exact", (len(v.shape),))),
            Part("weight_g_idx", ("i32",), lambda v: ("exact", (v.cols,)), required=False),
        ),
        decode,
        {"group_size": 128, "strategy": "group", "symmetric": True, "packing": "dense"},
        rule=_ct_rule,
        fix=fix,
    )


#: Every scheme the cores decode, one row each.
ADAPTERS: tuple[Adapter, ...] = (
    NVFP4,
    SVDQ_NVFP4,
    MXFP8,
    FP8_BLOCKWISE,
    MXFP4,
    *(_gguf(name) for name in GGUF),
    *(_gptq(bits) for bits in (2, 3, 4, 8)),
    AWQ_INT4,
    *(_ct(bits) for bits in range(1, 9)),
    _bnb_four_bit("bnb.nf4", NF4),
    _bnb_four_bit("bnb.fp4", FP4),
    BNB_INT8,
)
BY_SCHEME = {a.scheme: a for a in ADAPTERS}
