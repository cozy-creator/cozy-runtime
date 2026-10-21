"""The decode cores (`internal/decode_cores.py`) against the cozytensors/1 vectors, on every path:
the eager reference here, the Triton kernels on CPU under Triton's interpreter (the launcher test
runs this file again with `TRITON_INTERPRET=1`), and the kernels on a CUDA device when there is
one.

- `encoding/vectors/cores.json`: the cores' own cases (affine and table over the producers'
  packings, and gguf-py's q4_0/q8_0 structs), bit for bit.
- `encoding/vectors/cases/*.json`: every scheme row of `decode_schemes.ADAPTERS`, through
  `decode_schemes.run`, and the schemes xs-045's own decoders serve (fp8 and int8 tensorwise and
  rowwise, mxfp8) through one decode function each, bit for bit; every refusal case refused by
  its row's part rules or its decode.

This file imports only the cores and the rows, so the interpreter can load it.
"""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import decode_cores as cores
from cozy_runtime.internal.decode_cores import HAS_TRITON, Cores, Grid, Packing, blocks_of
from cozy_runtime.internal.decode_schemes import BY_SCHEME, parts_rule, run

try:
    import triton
    import triton.language as tl
except ImportError:  # the eager reference still runs
    triton = None

torch = pytest.importorskip("torch")

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
CUDA = torch.cuda.is_available() and not INTERPRETED and HAS_TRITON
VECTORS = Path(cores.__file__).parent / "encoding" / "vectors"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _paths() -> list[tuple[bool, str]]:
    """(kernel, device) for every path this process runs."""
    if INTERPRETED:
        return [(True, "cpu")]
    return [(False, "cpu")] + ([(True, "cuda")] if CUDA else [])


PATHS = pytest.mark.parametrize(("kernel", "device"), _paths())

# ------------------------------------------------------------- the schemes on the cores

Decode = Callable[[Cores, Mapping[str, Any], Mapping[str, object], tuple[int, ...], Any], None]


def view(shape: tuple[int, ...], flat: bool) -> tuple[int, int]:
    numel = math.prod(shape)
    if flat:
        return 1, numel
    return (numel // shape[-1] if shape[-1] else 0), shape[-1]


def element_codes(dtype: Any, device: Any) -> Any:
    """Every code of an 8-bit dtype as f32: the table its bytes index."""
    return torch.arange(256, dtype=torch.uint8).view(dtype).to(torch.float32).to(device)


def e8m0(scale: Any) -> Any:
    """E8M0 bytes as f32 by their bits: `b << 23`, 0 the subnormal 2^-127, 255 NaN."""
    raw = scale.view(torch.uint8).to(torch.int32)
    bits = torch.where(raw == 0, torch.full_like(raw, 1 << 22), raw << 23)
    return torch.where(raw == 255, torch.full_like(bits, 0x7FC00000), bits).view(torch.float32)


def fp8(
    grid: Grid,
    c: Cores,
    parts: Mapping[str, Any],
    params: Mapping[str, object],
    shape: Any,
    out: Any,
) -> None:
    flat = grid == cores.TENSOR  # tensorwise: one scale for the tensor
    rows, cols = view(shape, flat)
    weight = parts["weight"]
    c.table(
        weight.view(torch.uint8),
        Packing(8, 8),
        rows,
        cols,
        element_codes(weight.dtype, weight.device),
        parts["scale"],
        grid,
        out,
    )


def int8(
    grid: Grid,
    c: Cores,
    parts: Mapping[str, Any],
    params: Mapping[str, object],
    shape: Any,
    out: Any,
) -> None:
    flat = grid == cores.TENSOR  # tensorwise: one scale for the tensor
    rows, cols = view(shape, flat)
    c.affine(
        parts["weight"].view(torch.uint8),
        Packing(8, 8, signed=True),
        rows,
        cols,
        parts["scale"],
        grid,
        out,
    )


def mxfp8(
    c: Cores, parts: Mapping[str, Any], params: Mapping[str, object], shape: Any, out: Any
) -> None:
    rows, cols = view(shape, False)
    weight = parts["weight"]
    c.table(
        weight.view(torch.uint8),
        Packing(8, 8),
        rows,
        cols,
        element_codes(weight.dtype, weight.device),
        e8m0(parts["scale"]),
        blocks_of(32, str(params["scale_layout"])),
        out,
    )


DECODES: dict[str, Decode] = {
    "fp8.tensorwise": partial(fp8, cores.TENSOR),
    "fp8.rowwise": partial(fp8, cores.ROW),
    "fp8.columnwise": partial(fp8, cores.COL),
    "int8.tensorwise": partial(int8, cores.TENSOR),
    "int8.rowwise": partial(int8, cores.ROW),
    "int8.columnwise": partial(int8, cores.COL),
    "mxfp8": mxfp8,
}

# ----------------------------------------------------------------------- the vectors

DTYPES = {
    "F32": "float32",
    "BF16": "bfloat16",
    "F16": "float16",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
}


def raw(text: str) -> Any:
    return torch.frombuffer(bytearray(base64.b64decode(text)), dtype=torch.uint8)


def part(p: Mapping[str, Any], device: str) -> Any:
    """A vector part as the fill plane hands it over: `F4` as its packed bytes `[..., C/2]`."""
    flat = raw(p["bits"])
    if p["dtype"] == "F4":
        return flat.reshape(*p["shape"][:-1], p["shape"][-1] // 2).to(device)
    return flat.view(getattr(torch, DTYPES[p["dtype"]])).reshape(p["shape"]).to(device)


def scheme_cases(refusals: bool = False) -> list[dict[str, Any]]:
    """The cases of every scheme a row or a decode function above serves."""
    out = []
    for path in sorted((VECTORS / "cases").glob("*.json")):
        for c in json.loads(path.read_text())["cases"]:
            scheme = c.get("encoding", {}).get("scheme")
            if (scheme in BY_SCHEME or (scheme in DECODES and not refusals)) and (
                ("refuse" in c) == refusals
            ):
                out.append(c)
    return out


def same(got: Any, expect: str) -> bool:
    """Bit for bit, the spec's canonical NaN included."""
    want = raw(expect)
    return bool(torch.equal(got.cpu().reshape(-1).view(torch.uint8), want))


def decode(c: Cores, case: Mapping[str, Any], device: str) -> Any:
    parts = {p["role"]: part(p, device) for p in case["parts"]}
    out = torch.empty(*case["shape"], dtype=getattr(torch, DTYPES[case["dtype"]]), device=device)
    params, scheme = case["encoding"]["params"], case["encoding"]["scheme"]
    if scheme in BY_SCHEME:
        run(BY_SCHEME[scheme], c, parts, params, out)
    else:
        DECODES[scheme](c, parts, params, tuple(case["shape"]), out)
    return out


@PATHS
def test_every_scheme_the_cores_express_decodes_its_vectors(kernel: bool, device: str) -> None:
    c = Cores(torch, kernel)
    cases = scheme_cases()
    seen = set()
    for case in cases:
        assert same(decode(c, case, device), case["expect"]), case["key"]
        seen.add(case["encoding"]["scheme"])
    assert len(cases) >= 200 and len(seen) >= 30, (len(cases), sorted(seen))


#: The fill plane hands an `F4` part over as its packed bytes, so a `U8` part of the same bytes
#: reads alike: only TensorFS's own decoder, which sees the header's dtype, refuses these.
UNSEEN = {("mxfp4", "weight-u8")}


def rolepart(p: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    """A part's (runtime dtype, shape), as the fill plane lists it."""
    if p["dtype"] == "F4":
        return "u8", (*p["shape"][:-1], p["shape"][-1] // 2)
    names = {"F8_E4M3": "f8_e4m3fn", "F8_E5M2": "f8_e5m2", "F8_E8M0": "f8_e8m0"}
    return names.get(p["dtype"], p["dtype"].lower()), tuple(p["shape"])


def test_every_refusal_case_is_refused() -> None:
    """By the row's rules (params, parts, shapes) before a byte moves, or by its decode, which
    checks the values the spec constrains (tables, indices, zeros, the stored shape)."""
    if INTERPRETED:
        pytest.skip("the eager path's")
    cases = scheme_cases(refusals=True)
    for case in cases:
        scheme, params = case["encoding"]["scheme"], case["encoding"]["params"]
        if (scheme, case["key"]) in UNSEEN:
            continue
        roles = {p["role"]: rolepart(p) for p in case["parts"]}
        if case["dtype"] not in ("F32", "BF16", "F16") or parts_rule(
            BY_SCHEME[scheme], case["shape"], params, roles
        ):
            continue
        with pytest.raises((ValueError, cores.CodeOutsideTable)):
            decode(Cores(torch, False), {**case, "dtype": "F32"}, "cpu")
    assert len(cases) >= 70, len(cases)


# ------------------------------------------------------------------- the core cases


def eager_gguf(kind: str) -> Callable[[Any, Any], Any]:
    """gguf's block_q4_0 {f16 d; u8 qs[16]}: element j in qs[j]'s low nibble, j + 16 in its high
    one, (q - 8) x d; block_q8_0 {f16 d; i8 qs[32]}: q x d."""

    def decode(torch: Any, structs: Any) -> Any:
        d = structs[:, :2].contiguous().view(torch.float16).to(torch.float32)
        if kind == "q4_0":
            qs = structs[:, 2:18].to(torch.int32)
            return (torch.cat((qs & 15, qs >> 4), dim=1).to(torch.float32) - 8.0) * d
        return structs[:, 2:].contiguous().view(torch.int8).to(torch.float32) * d

    return decode


KERNELS: dict[str, Any] = {}
if HAS_TRITON:

    @triton.jit
    def q4_0(base, el, mask):  # type: ignore[no-untyped-def]
        byte = cores.u8_at(base, 2 + el % 16, mask)
        q = tl.where(el < 16, byte & 15, byte >> 4)
        return (q.to(tl.float32) - 8.0) * cores.f16_at(base, 0, mask)

    @triton.jit
    def q8_0(base, el, mask):  # type: ignore[no-untyped-def]
        return cores.i8_at(base, 2 + el, mask).to(tl.float32) * cores.f16_at(base, 0, mask)

    KERNELS = {"q4_0": q4_0, "q8_0": q8_0}


def core_case(c: Cores, case: Mapping[str, Any], device: str) -> Any:
    rows, cols = case["rows"], case["cols"]
    out = torch.empty(rows * cols, dtype=torch.float32, device=device)

    def f32(key: str) -> Any:
        return raw(case[key]).view(torch.float32).to(device)

    if case["core"] == "blocks":
        c.blocks(
            raw(case["blocks"]).to(device),
            case["elements"],
            case["size"],
            out,
            kernel_fn=KERNELS.get(case["type"]),
            eager_fn=eager_gguf(case["type"]),
        )
        return out
    p = dict(case["packing"])
    if isinstance(p.get("order"), list):
        p["order"] = tuple(p["order"])
    g = case["grid"]
    grid = Grid(g["kind"], g.get("size", 1), g.get("layout", "row_major"))
    g_idx = torch.tensor(g["index"], dtype=torch.int32, device=device) if "index" in g else None
    words = raw(case["words"]).to(device)
    if case["core"] == "table":
        c.table(words, Packing(**p), rows, cols, f32("table"), f32("scale"), grid, out, g_idx=g_idx)
    else:
        c.affine(
            words,
            Packing(**p),
            rows,
            cols,
            f32("scale"),
            grid,
            out,
            zero=f32("zero") if "zero" in case else None,
            bias=f32("bias") if "bias" in case else None,
            g_idx=g_idx,
        )
    return out


@PATHS
def test_the_cores_decode_every_core_case(kernel: bool, device: str) -> None:
    c = Cores(torch, kernel)
    cases = json.loads((VECTORS / "cores.json").read_text())["cases"]
    for case in cases:
        assert same(core_case(c, case, device), case["expect"]), case["name"]
    assert len(cases) == 16


@PATHS
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_round_l_is_nearest_even_with_the_canonical_nan(
    kernel: bool, device: str, dtype: str
) -> None:
    """Every rounding boundary of the destination: f32 patterns at, beside and between its
    neighbours, signed zeros, subnormals, overflow and NaNs of any sign and payload."""
    generator = torch.Generator().manual_seed(7)
    bits = torch.randint(0, 1 << 31, (4096,), generator=generator, dtype=torch.int64)
    bits = torch.cat(
        (
            bits,
            bits | (1 << 31),
            torch.tensor(
                [
                    0x7FC00001,
                    0xFFC00000,
                    0x7F800001,
                    0x477FF000,
                    0x477FEFFF,
                    0x7F7FFFFF,
                    0x00000001,
                    0x80000000,
                    0x3F808000,
                    0x3F818000,
                ]
            ),
        )
    )
    values = bits.to(torch.int32).view(torch.float32).to(device)
    n = values.numel()
    out = torch.empty(n, dtype=getattr(torch, dtype), device=device)
    # q = 1 times a scale per column: each value exactly, then round_L.
    ones = torch.ones(n, dtype=torch.uint8, device=device)
    Cores(torch, kernel).affine(ones, Packing(8, 8), 1, n, values, cores.COL, out)
    want = values.cpu().to(getattr(torch, dtype)).view(torch.int16).clone()
    want[values.cpu().isnan()] = 0x7FC0 if dtype == "bfloat16" else 0x7E00
    assert torch.equal(out.cpu().view(torch.int16), want)


@PATHS
def test_a_code_outside_its_table_refuses(kernel: bool, device: str) -> None:
    words = torch.tensor([0x10, 0x32], dtype=torch.uint8, device=device)
    out = torch.empty(4, device=device)
    with pytest.raises(cores.CodeOutsideTable):
        Cores(torch, kernel).table(
            words,
            Packing(4, 8),
            1,
            4,
            torch.ones(3, device=device),
            torch.ones(1, device=device),
            cores.TENSOR,
            out,
        )


def test_the_kernels_hold_under_tritons_interpreter() -> None:
    if INTERPRETED:
        pytest.skip("the launcher itself")
    pytest.importorskip("triton")
    env = {
        **os.environ,
        "TRITON_INTERPRET": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", __file__]
    run = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 0 and " passed" in run.stdout, run.stdout[-4000:] + run.stderr
