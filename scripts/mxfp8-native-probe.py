#!/usr/bin/env python3
"""The BLACKWELL-NATIVE MXFP8 compute probe. POD-SIDE ONLY — never runs on the dev box.

`scripts/mxfp8-pod.py` qualified the DECODED-FLOAT route (#511a): mxfp8 bytes are
dequantized into the destination's own float dtype at fill, and the matmul is an ordinary
bf16 matmul. That route buys bytes-read and load-time and, as #511's open owner question
records, it does NOT buy VRAM — the resident DiT is float either way.

The other route is the one MXFP8 was designed for: Blackwell tensor cores consume E4M3
elements plus E8M0 per-32-block scales DIRECTLY, so the weights stay at ~8.25 bits
RESIDENT and the scales are applied inside the MMA. That is the `cuda.sm100+` door
NAMED_SEAMS has held open since 864220f. This script is what walks up to the door on real
silicon and reports which side of it we are on.

Three questions, in this order, and the first one can end the run:

  matrix    WHICH (library, version, arch) combinations expose a block-scaled MX GEMM at
            all, with every refusal's text recorded VERBATIM. A "no" here is a finding,
            not a failure — #511's honest answer is the deliverable either way.
  numerics  what the native GEMM's arithmetic actually is, against the ratified
            dequant-to-bf16 reference on the SAME bytes: the full 65,536-pair format
            domain forced through the native kernel by an identity operand, then block 0
            of the real H3 curve DiT.
  cost      what it BUYS — native vs bf16 vs dequant-then-bf16 at the real DiT block's
            GEMM shapes, and the resident VRAM for one block's weights both ways.

**The activation half, stated up front.** A native MX GEMM takes BOTH operands in mxfp8.
The ratified lane is weight-only: activations stay bf16 and only the weight was ever
quantized. So the native route is not a faster spelling of the same arithmetic — it is
w8a8-mx, and it pays an activation quantization the ratified lane does not. Every numeric
below therefore reports the native path against BOTH references and separates the
kernel's own error from the quantization's, rather than quoting one number that hides
which half it came from.

    python3 mxfp8-native-probe.py all
"""

from __future__ import annotations

import dataclasses
import json
import math
import struct
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/root/mxnative")
REPO = "Comfy-Org/MiniMax-H3"
REV = "4cc1d817b6184899b41293954329f576cb5ae86b"
MEMBER = "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{MEMBER}"

BLOCK = "blocks.0."
#: The four keys the quantize-mxfp8 job's inclusion map covers in one block, and the only
#: GEMMs a native route would ever execute here.
QUANTIZED = (
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)
MX_BLOCK = 32
E4M3_MAX = 448.0
MIB = 1 << 20

PASS = 0
FAIL = 0
REPORT: dict[str, Any] = {"format": "cozy.runtime.mxfp8_native/1"}


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"\n          {detail}" if detail else ""))
    return ok


def verbatim(exc: BaseException) -> str:
    """An error's text EXACTLY as the library wrote it. A paraphrased refusal is worthless
    as a support-matrix entry: the whole value of this run is that someone else can grep
    the vendor's source for this string."""
    return f"{type(exc).__module__}.{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- the device


def facts() -> dict[str, Any]:
    import torch

    head("facts — the card, the toolkit, the libraries")
    out: dict[str, Any] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        out |= {
            "device": str(props.name),
            "sm": major * 10 + minor,
            "arch": f"sm{major}{minor}",
            "vram_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
        }
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    out["nvidia_smi"] = smi.stdout.strip()
    for name in ("float8_e4m3fn", "float8_e8m0fnu", "float4_e2m1fn_x2"):
        out[f"dtype.{name}"] = hasattr(torch, name)
    out["has_scaled_mm"] = hasattr(torch, "_scaled_mm")
    out["has_scaled_mm_v2"] = hasattr(torch, "_scaled_mm_v2")
    try:
        out["blas"] = str(torch.backends.cuda.preferred_blas_library())
    except Exception as exc:
        out["blas"] = verbatim(exc)
    for op in ("_scaled_mm", "_scaled_mm_v2"):
        try:
            schemas = torch._C._jit_get_schemas_for_operator(f"aten::{op}")
            out[f"schema.{op}"] = [str(s) for s in schemas]
        except Exception as exc:
            out[f"schema.{op}"] = [verbatim(exc)]
    for key, value in out.items():
        if not key.startswith("schema."):
            print(f"  {key:28} {value}")
    for op in ("_scaled_mm", "_scaled_mm_v2"):
        for line in out[f"schema.{op}"]:
            print(f"  schema  {line}")
    REPORT["facts"] = out
    return out


# ------------------------------------------------------------------- the scale swizzle


def to_blocked(torch: Any, scales: Any) -> Any:
    """[rows, cols] E8M0 bytes -> cuBLASLt's 128x4 blocked ("swizzled") scale layout.

    The block-scaled MMA does not read a row-major scale grid: it reads 128x4 tiles, each
    stored as (32, 4, 4) with the inner two transposed. Transcribed from the layout the
    kernel documents; if the runtime accepts a plain row-major grid instead, the matrix leg
    below records THAT as the observed fact rather than assuming this one is required.
    """
    rows, cols = int(scales.shape[0]), int(scales.shape[1])
    row_blocks = -(-rows // 128)
    col_blocks = -(-cols // 4)
    padded = torch.zeros(row_blocks * 128, col_blocks * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def mx_quantize(torch: Any, weight: Any) -> tuple[Any, Any, int]:
    """The producing job's quantizer (`cozy_jobs/quantize.py`), transcribed unchanged.

    `e = floor(log2(amax)) - 8`, one E8M0 byte per 32 elements along the last axis, the
    payload clamped to e4m3's 448. The native route and the ratified dequant route must be
    fed by the SAME quantizer or the comparison measures two producers, not two consumers.
    """
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    work = weight.to(torch.float32)
    blocks = work.view(rows, cols // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0, torch.floor(torch.log2(amax)) - 8.0, torch.full_like(amax, -127.0)
    )
    byte = (exponent + 127.0).clamp(0.0, 254.0)
    scale = torch.pow(2.0, byte - 127.0)
    quant = (blocks / scale.unsqueeze(-1)).view(rows, cols)
    saturated = int((quant.abs() > E4M3_MAX).sum())
    return quant.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), byte.to(torch.uint8), saturated


def reference_dequant(torch: Any, data: Any, byte: Any, dtype: Any) -> Any:
    """THE RATIFIED REFERENCE — `cozy.mxfp8.dequant/1` itself, imported, not re-implemented.

    #511 qualified that provider bit-exactly over the whole format domain on CUDA. Every
    delta this script reports is measured against it, so it must be the same object the
    runtime serves and not a second transcription that could drift from it.
    """
    from cozy_runtime.internal.encoding import MicroScaledDequant

    provider = MicroScaledDequant("sha256:reference")
    destination = torch.empty(data.shape, dtype=dtype, device=data.device)
    provider.decode(torch, {"data": data, "scale": byte}, destination)
    return destination


# ------------------------------------------------------------------- the support matrix


def _attempt(name: str, note: str, fn: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"attempt": name, "note": note}
    try:
        value = fn()
    except BaseException as exc:
        row |= {"ok": False, "error": verbatim(exc), "where": traceback.format_exc().strip()[-400:]}
    else:
        row |= {"ok": True, "result": value}
    flag = "ok  " if row["ok"] else "REFUSED"
    print(f"  {flag:8} {name}")
    print(f"           {row.get('error', row.get('result'))}")
    return row


def matrix() -> list[dict[str, Any]]:
    """Every spelling of a block-scaled MX GEMM this stack could plausibly expose, tried.

    The point is not to find one that works; it is to leave a (library, version, arch)
    table whose refusals are quoted well enough that a reader can tell a MISSING FEATURE
    from a WRONG CALL — the two failures that look identical in a one-line "unsupported".
    """
    import torch

    head("matrix — which (library, version, arch) exposes a block-scaled MX GEMM")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    m = n = k = 256
    a8 = torch.randint(0, 240, (m, k), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    b8 = torch.randint(0, 240, (n, k), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    bt = b8.t()  # [K, N], column-major, which is the layout _scaled_mm demands of B
    ones = torch.ones((), dtype=torch.float32, device=device)
    sa_u8 = torch.full((m, k // MX_BLOCK), 127, dtype=torch.uint8, device=device)
    sb_u8 = torch.full((n, k // MX_BLOCK), 127, dtype=torch.uint8, device=device)

    def e8m0(x: Any) -> Any:
        return x.view(torch.float8_e8m0fnu) if hasattr(torch, "float8_e8m0fnu") else x

    def described(out: Any) -> str:
        return f"{tuple(out.shape)} {out.dtype} finite={bool(torch.isfinite(out).all())}"

    rows.append(
        _attempt(
            "torch._scaled_mm per-tensor fp8 (the sm90 route)",
            "does this card do fp8 tensor-core GEMM AT ALL, block scales aside",
            lambda: described(
                torch._scaled_mm(a8, bt, scale_a=ones, scale_b=ones, out_dtype=torch.bfloat16)
            ),
        )
    )
    rows.append(
        _attempt(
            "torch._scaled_mm MX, e8m0 scales, BLOCKED (swizzled) layout",
            "the documented Blackwell spelling: 128x4-tiled E8M0 grids",
            lambda: described(
                torch._scaled_mm(
                    a8,
                    bt,
                    scale_a=e8m0(to_blocked(torch, sa_u8)),
                    scale_b=e8m0(to_blocked(torch, sb_u8)),
                    out_dtype=torch.bfloat16,
                )
            ),
        )
    )
    rows.append(
        _attempt(
            "torch._scaled_mm MX, e8m0 scales, PLAIN row-major grids",
            "the same call with unswizzled [M, K/32] / [N, K/32] scales",
            lambda: described(
                torch._scaled_mm(
                    a8,
                    bt,
                    scale_a=e8m0(sa_u8),
                    scale_b=e8m0(sb_u8),
                    out_dtype=torch.bfloat16,
                )
            ),
        )
    )
    rows.append(
        _attempt(
            "torch._scaled_mm MX, UINT8 scales, blocked layout",
            "separates 'dtype not accepted' from 'block scaling not supported'",
            lambda: described(
                torch._scaled_mm(
                    a8,
                    bt,
                    scale_a=to_blocked(torch, sa_u8),
                    scale_b=to_blocked(torch, sb_u8),
                    out_dtype=torch.bfloat16,
                )
            ),
        )
    )
    if hasattr(torch, "_scaled_mm_v2"):
        rows.append(
            _attempt(
                "torch._scaled_mm_v2 MX",
                "the typed-recipe surface",
                lambda: _v2(torch, a8, bt, sa_u8, sb_u8),
            )
        )
    else:
        rows.append(
            {
                "attempt": "torch._scaled_mm_v2 MX",
                "note": "the typed-recipe surface",
                "ok": False,
                "error": f"absent from torch {torch.__version__}",
            }
        )
        print(f"  {'ABSENT':8} torch._scaled_mm_v2 (torch {torch.__version__})")
    rows.append(
        _attempt("torchao mx_formats", "the reference MX implementation", lambda: _torchao(torch))
    )
    rows.append(_attempt("transformer_engine MXFP8BlockScaling", "TE's own recipe gate", _te))
    REPORT["matrix"] = rows
    served = [r["attempt"] for r in rows if r["ok"]]
    check(
        "at least one library exposes a block-scaled MX GEMM on this arch",
        bool(served),
        f"{len(served)} of {len(rows)} spellings work: {served}"
        if served
        else "NONE — every spelling refused; the refusals above ARE the deliverable",
    )
    return rows


def _v2(torch: Any, a8: Any, bt: Any, sa: Any, sb: Any) -> str:
    """`_scaled_mm_v2` takes the scaling recipe as an enum rather than inferring it from the
    scale tensors' shapes, so the same MX call has to be spelled a second way."""
    recipes = getattr(torch, "ScalingType", None)
    mx = getattr(recipes, "BlockWise1x32", None) if recipes is not None else None
    if mx is None:
        return "torch.ScalingType.BlockWise1x32 absent — no MX recipe to name"
    out = torch._scaled_mm_v2(
        a8,
        bt,
        [to_blocked(torch, sa).view(torch.float8_e8m0fnu)],
        [mx],
        [to_blocked(torch, sb).view(torch.float8_e8m0fnu)],
        [mx],
        out_dtype=torch.bfloat16,
    )
    return f"{tuple(out.shape)} {out.dtype}"


def _torchao(torch: Any) -> str:
    import torchao  # type: ignore[import-not-found]
    from torchao.prototype.mx_formats.mx_tensor import MXTensor  # type: ignore[import-not-found]

    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    mx_x = MXTensor.to_mx(x, torch.float8_e4m3fn, MX_BLOCK)
    mx_w = MXTensor.to_mx(w, torch.float8_e4m3fn, MX_BLOCK)
    out = torch.mm(mx_x, mx_w.t())
    return f"torchao {torchao.__version__}: {tuple(out.shape)} {out.dtype}"


def _te() -> str:
    """TransformerEngine's MXFP8 gate is the specific hazard #469a's Blackwell premise
    rests on: the recipe has historically demanded compute capability 10.0 exactly and
    refused 12.x. Its own support function returns the reason as a STRING, so this asks it
    rather than inferring the answer from an exception."""
    import transformer_engine  # type: ignore[import-not-found]
    import transformer_engine.pytorch as te  # type: ignore[import-not-found]
    from transformer_engine.common import recipe  # type: ignore[import-not-found]

    version = getattr(transformer_engine, "__version__", "unknown")
    reasons = []
    for name in ("check_mxfp8_support", "check_fp8_support"):
        fn = getattr(te.fp8.FP8GlobalStateManager, name, None) or getattr(te.fp8, name, None)
        reasons.append(f"{name}={fn() if fn else 'absent'}")
    mx = recipe.MXFP8BlockScaling()
    return f"transformer_engine {version}: {'; '.join(reasons)}; recipe={type(mx).__name__}"


# ---------------------------------------------------------------------------- numerics


def _native(torch: Any, a8: Any, sa: Any, b8: Any, sb: Any, out_dtype: Any) -> Any:
    """One native MX GEMM: A [M,K] e4m3 x B [N,K] e4m3 -> [M,N], scales E8M0 per 32.

    Both scale grids go through the blocked layout because that is what the matrix leg
    observed the kernel to require; if it observed otherwise this function is what changes.
    """
    return torch._scaled_mm(
        a8,
        b8.t(),
        scale_a=to_blocked(torch, sa).view(torch.float8_e8m0fnu),
        scale_b=to_blocked(torch, sb).view(torch.float8_e8m0fnu),
        out_dtype=out_dtype,
    )


#: The native kernel's shape law for 1x32 block scaling, MEASURED (torch 2.9.1's own
#: refusal text): each scale grid must carry EXACTLY `rows * cols / 32` elements and be
#: contiguous. That is an element COUNT, not a layout — see `layout()` for what it does not
#: check. A blocked grid therefore only fits without padding when the row count is a
#: multiple of 128 and the block count is a multiple of 4, which is why the domain sweep
#: below is laid out as 512 x 128 rather than the fill probe's 256 x 256.
DOMAIN_ROWS, DOMAIN_K = 512, 128


#: The three code points a GEMM structurally cannot carry through this sweep. `0 * NaN` is
#: NaN in IEEE, so a single NaN anywhere in a row — an E4M3 NaN element, or ANY element
#: under an E8M0-NaN block scale — poisons every dot product that row participates in, and
#: the identity operand's 127 exact zeros do not save it. That is not a defect of the native
#: route (`nan_arm` below shows the ratified dequant-then-matmul lane poisons identically);
#: it is what a matmul IS. So the sweep substitutes them, EXCLUDES them from the comparison
#: by name, and the propagation question is asked separately where it can be answered.
E4M3_NAN = (0x7F, 0xFF)
E8M0_NAN = 0xFF
SUBSTITUTE = 0x00


def _domain_operands(torch: Any, device: Any) -> tuple[Any, Any, Any, Any]:
    """All 65,536 (element byte, scale byte) pairs laid out as one [512, 128] MX operand.

    Blocks are numbered `g = row * 4 + block`, 0..2047; block `g` carries scale byte `g // 8`
    and element bytes `(g % 8) * 32 ..+32`. Eight consecutive blocks therefore sweep all 256
    element patterns under one scale byte, and 256 scale bytes sweep the rest — the exact
    cross product #511's fill probe covers, in the shape the tensor core will accept. The
    766 NaN-bearing pairs carry `SUBSTITUTE` instead and are excluded by `_domain_oracle`.
    """
    payload = bytearray()
    for row in range(DOMAIN_ROWS):
        for block in range(DOMAIN_K // MX_BLOCK):
            base = ((row * (DOMAIN_K // MX_BLOCK) + block) % 8) * 32
            payload += bytes(SUBSTITUTE if b in E4M3_NAN else b for b in range(base, base + 32))
    a8 = (
        torch.frombuffer(payload, dtype=torch.uint8)
        .reshape(DOMAIN_ROWS, DOMAIN_K)
        .clone()
        .to(device)
        .view(torch.float8_e4m3fn)
    )
    blocks = DOMAIN_ROWS * DOMAIN_K // MX_BLOCK
    grid = torch.arange(blocks, device=device) // 8
    sa = (
        torch.where(grid == E8M0_NAN, torch.full_like(grid, 127), grid)
        .to(torch.uint8)
        .reshape(DOMAIN_ROWS, DOMAIN_K // MX_BLOCK)
    )
    # B is an exact identity at scale byte 127 (2**0), so C[m, n] is the DEQUANTIZED element
    # A[m, n] and nothing else: 127 of the 128 products in each dot product are exact zeros.
    identity = torch.eye(DOMAIN_K, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    sb = torch.full((DOMAIN_K, DOMAIN_K // MX_BLOCK), 127, dtype=torch.uint8, device=device)
    return a8, sa, identity, sb


def _domain_oracle(row: int, col: int) -> tuple[int, int]:
    """(element byte, scale byte) at output position (row, col), or (-1, -1) if excluded."""
    g = row * (DOMAIN_K // MX_BLOCK) + col // MX_BLOCK
    element, scale = (g % 8) * 32 + col % MX_BLOCK, g // 8
    if element in E4M3_NAN or scale == E8M0_NAN:
        return -1, -1
    return element, scale


def domain() -> None:
    """The full 65,536-pair (e4m3 element x e8m0 scale) domain, FORCED THROUGH the native
    kernel by an identity operand.

    #511's probe decoded that domain with a cast and a multiply. A GEMM cannot be asked the
    same question directly — but with B an exact identity at scale byte 127 (2**0), the
    native kernel's output IS the dequantized element, and the same oracle applies unchanged.
    What it measures that the fill probe cannot: whether the block scale the TENSOR CORE
    applies is the same number `cozy.mxfp8.dequant/1` computes, over every code point
    including E8M0's NaN and the products that leave f32's range.
    """
    import torch

    from cozy_runtime.internal.probe import e4m3_value, e8m0_value

    head("domain — all 65,536 (element, scale) pairs through the NATIVE kernel")
    device = torch.device("cuda")
    a8, sa, identity, sb = _domain_operands(torch, device)
    rows, cols = DOMAIN_ROWS, DOMAIN_K

    got = _native(torch, a8, sa, identity, sb, torch.float32).double().cpu().reshape(-1).tolist()
    ref_list = reference_dequant(torch, a8, sa, torch.float32).double().cpu().reshape(-1).tolist()

    elements = [e4m3_value(b) for b in range(256)]
    scales = [e8m0_value(b) for b in range(256)]
    bands = ("normal", "subnormal", "overflow", "underflow", "nan")
    counts = dict.fromkeys(bands, 0)
    wrong = dict.fromkeys(bands, 0)
    worst_rel = worst_abs = 0.0
    first_bad = ""
    disagreements = 0
    info = torch.finfo(torch.float32)
    finite_max, smallest_normal = float(info.max), float(info.tiny)
    ulp_sub = smallest_normal * float(info.eps)
    rel_bound, abs_bound = 2.0 * float(info.eps), 2.0 * ulp_sub

    excluded = 0
    for r in range(rows):
        for c in range(cols):
            element_byte, scale_byte = _domain_oracle(r, c)
            if element_byte < 0:
                excluded += 1
                continue
            want = elements[element_byte] * scales[scale_byte]
            have = got[r * cols + c]
            mine = ref_list[r * cols + c]
            if not (
                (math.isnan(have) and math.isnan(mine))
                or have == mine
                or (math.isinf(have) and math.isinf(mine) and (have > 0) == (mine > 0))
            ):
                disagreements += 1
            where = f"element {element_byte:#04x} x scale {scale_byte:#04x}"
            if math.isnan(want):
                counts["nan"] += 1
                if not math.isnan(have):
                    wrong["nan"] += 1
                    first_bad = first_bad or f"{where}: {have} != nan"
            elif abs(want) > finite_max:
                counts["overflow"] += 1
                if not math.isinf(have) or (have > 0) != (want > 0):
                    wrong["overflow"] += 1
                    first_bad = first_bad or f"{where}: {have} != inf"
            elif want != 0.0 and abs(want) < ulp_sub / 2.0:
                counts["underflow"] += 1
                if have != 0.0:
                    wrong["underflow"] += 1
                    first_bad = first_bad or f"{where}: {have} != 0"
            elif abs(want) < smallest_normal:
                counts["subnormal"] += 1
                error = abs(have - want)
                worst_abs = max(worst_abs, error)
                if error > abs_bound:
                    wrong["subnormal"] += 1
                    first_bad = first_bad or f"{where}: {have} != {want} (abs {error:.3e})"
            else:
                counts["normal"] += 1
                error = abs(have - want)
                worst_abs = max(worst_abs, error)
                worst_rel = max(worst_rel, error / abs(want))
                if error / abs(want) > rel_bound:
                    wrong["normal"] += 1
                    relative = error / abs(want)
                    first_bad = first_bad or f"{where}: {have} != {want} (rel {relative:.3e})"

    measured = rows * cols - excluded
    REPORT["domain"] = {
        "pairs_measured": measured,
        "pairs_excluded_nan": excluded,
        "counts": counts,
        "wrong": wrong,
        "max_rel_error": worst_rel,
        "max_abs_error": worst_abs,
        "tolerance": rel_bound,
        "disagreements_vs_dequant_reference": disagreements,
        "first_wrong": first_bad,
    }
    check(
        f"the native kernel decodes all {measured} non-NaN pairs within the format's bound",
        not any(wrong.values()),
        f"domain {counts}; wrong {wrong}; max rel {worst_rel:.3e} <= {rel_bound:.3e}; "
        f"{excluded} NaN-bearing pairs excluded (see nan_arm); {first_bad}",
    )
    check(
        "the TENSOR CORE's block scale agrees with `cozy.mxfp8.dequant/1` pair for pair",
        disagreements == 0,
        f"{disagreements} of {measured} pairs differ from the ratified provider's decode",
    )


def nan_arm() -> None:
    """The 766 pairs the sweep had to exclude, asked where a GEMM CAN answer them.

    Two questions, and only the second is about the encoding:

    1. Does a poisoned value poison the whole dot product? Yes, on BOTH routes, because
       `0 * NaN` is NaN — so this is not a native-route hazard, it is a matmul fact, and the
       arm exists to show the two lanes behave identically rather than to grade one.
    2. **What does the tensor core think E8M0 byte 0xFF MEANS?** #511c ruled that the conform
       corpus is normative over the registry's prose, which reads `2**128` where the
       reference decoder returns NaN. That ruling was about software. Here the same question
       is put to the hardware, and the answer decides whether a native provider could ever be
       bit-compatible with `cozy.mxfp8.dequant/1` on a poisoned block.
    """
    import torch

    head("nan_arm — NaN propagation, and what the tensor core thinks E8M0 0xFF means")
    device = torch.device("cuda")
    k, n = 128, 128
    identity = torch.eye(k, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    sb = torch.full((n, k // MX_BLOCK), 127, dtype=torch.uint8, device=device)

    def one_row(element: int, scale: int) -> tuple[float, float]:
        """A single [128, 128] row: `element` at column 0 of a block scaled by `scale`,
        zeros elsewhere. Returned as (native output, ratified dequant-then-matmul output)
        at column 0, so the two lanes are read at the same place."""
        data = torch.zeros(128, k, dtype=torch.uint8, device=device)
        data[:, 0] = element
        a8 = data.view(torch.float8_e4m3fn)
        sa = torch.full((128, k // MX_BLOCK), 127, dtype=torch.uint8, device=device)
        sa[:, 0] = scale
        native = _native(torch, a8, sa, identity, sb, torch.float32)
        lane = reference_dequant(torch, a8, sa, torch.bfloat16).float() @ reference_dequant(
            torch, identity, sb, torch.bfloat16
        ).float().t()
        return float(native[0, 0]), float(lane[0, 0])

    banked: dict[str, Any] = {}
    for label, element, scale in (
        ("e4m3 NaN element (0x7f)", 0x7F, 127),
        ("e8m0 NaN block scale (0xff)", 0x38, E8M0_NAN),  # 0x38 is e4m3 1.0
        ("clean control (1.0 x 2**0)", 0x38, 127),
    ):
        native, lane = one_row(element, scale)
        banked[label] = {"native": native, "dequant_lane": lane}
        print(f"  {label:30} native {native!r:24} dequant lane {lane!r}")
    check(
        "a poisoned element poisons BOTH lanes identically — a matmul fact, not an encoding one",
        math.isnan(banked["e4m3 NaN element (0x7f)"]["native"])
        and math.isnan(banked["e4m3 NaN element (0x7f)"]["dequant_lane"]),
        "0 * NaN is NaN, so the identity operand's exact zeros do not contain the poison "
        "on either route",
    )
    native_ff = banked["e8m0 NaN block scale (0xff)"]["native"]
    REPORT["nan_arm"] = banked | {
        "e8m0_0xff_hardware_reading": "nan"
        if math.isnan(native_ff)
        else ("inf" if math.isinf(native_ff) else repr(native_ff))
    }
    check(
        "the tensor core reads E8M0 0xFF the way the conform corpus does (NaN, not 2**128)",
        math.isnan(native_ff),
        f"hardware returned {native_ff!r}; #511c ruled the reference normative over the "
        "registry prose that reads 2**128, and this is that ruling checked against silicon",
    )


def layout() -> None:
    """The hazard the matrix leg exposed: torch checks the scale grid's ELEMENT COUNT, not
    its LAYOUT.

    A row-major `[rows, cols/32]` grid and the 128x4-tiled grid the tensor core actually
    reads have the same number of elements and are both contiguous, so both are ACCEPTED.
    Only one of them is the right number in the right place. That is the silent-wrong-output
    class the encoding plane exists to prevent, so it is measured here and named in
    NAMED_SEAMS rather than left for a future implementer to discover in a render.
    """
    import torch

    head("layout — blocked vs row-major scale grids, both accepted, only one correct")
    device = torch.device("cuda")
    a8, sa, identity, sb = _domain_operands(torch, device)
    want = reference_dequant(torch, a8, sa, torch.float32)
    banked: dict[str, Any] = {}

    def blocked(grid: Any) -> Any:
        return to_blocked(torch, grid).view(torch.float8_e8m0fnu)

    def row_major(grid: Any) -> Any:
        return grid.reshape(-1).contiguous().view(torch.float8_e8m0fnu)

    layouts: tuple[tuple[str, Any], ...] = (("blocked_128x4", blocked), ("row_major", row_major))
    for name, grid in layouts:
        try:
            got = torch._scaled_mm(
                a8, identity.t(), scale_a=grid(sa), scale_b=grid(sb), out_dtype=torch.float32
            )
        except BaseException as exc:
            banked[name] = {"accepted": False, "error": verbatim(exc)}
            print(f"  {name:16} REFUSED  {verbatim(exc)[:120]}")
            continue
        # The comparison is on the FINITE positions only: the sweep's high scale bytes
        # overflow f32, and how the kernel spells an overflow is `domain`'s question, not
        # this one. What this one asks is whether the right scale reached the right block.
        finite = torch.isfinite(got) & torch.isfinite(want)
        matched = int((got[finite] == want[finite]).sum())
        total = int(finite.sum())
        banked[name] = {
            "accepted": True,
            "finite_positions": total,
            "bit_identical": matched,
            "wrong": total - matched,
        }
        print(
            f"  {name:16} accepted; {matched}/{total} finite positions bit-identical to the "
            f"ratified decode"
        )
    REPORT["layout"] = banked
    right = [n for n, v in banked.items() if v.get("wrong") == 0]
    check(
        "exactly ONE accepted scale layout is the correct one",
        len(right) == 1,
        f"correct: {right}; accepted: {[n for n, v in banked.items() if v['accepted']]} — a "
        "wrong-layout grid passes the runtime's shape check and returns wrong numbers",
    )


# ------------------------------------------------------------------------- real weights


def _get(start: int, end: int) -> bytes:
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request, timeout=900) as response:
        if response.status != 206:
            raise SystemExit(f"expected a 206 partial response, got {response.status}")
        return bytes(response.read())


def fetch() -> None:
    """Block 0 of the real H3 curve DiT by ONE HTTP range — 735 MiB, not 40.2 GB.

    Weights never transit the dev box and the 40.2 GB tree is never pulled: a safetensors
    header is a byte map, so one block is addressable without the file (the same trick
    02469c9 used, against the same pinned revision, so the two legs share a subject).
    """
    import torch

    head("fetch — block 0 of the real H3 curve DiT, by HTTP range over the pinned revision")
    ROOT.mkdir(parents=True, exist_ok=True)
    length = struct.unpack("<Q", _get(0, 8))[0]
    header = json.loads(_get(8, 8 + length))
    base = 8 + length
    keys = sorted(k for k in header if k.startswith(BLOCK) and "adaln" not in k)
    lo = min(header[k]["data_offsets"][0] for k in keys)
    hi = max(header[k]["data_offsets"][1] for k in keys)
    started = time.time()
    span = _get(base + lo, base + hi)
    took = time.time() - started
    print(f"  source     {MEMBER} @ {REV[:12]}, {len(header)} keys")
    print(f"  fetched    {len(span) / MIB:.1f} MiB in {took:.1f}s")
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    out = {}
    for key in keys:
        a, b = header[key]["data_offsets"]
        out[key[len(BLOCK) :]] = (
            torch.frombuffer(bytearray(span[a - lo : b - lo]), dtype=dtypes[header[key]["dtype"]])
            .reshape(tuple(int(n) for n in header[key]["shape"]))
            .clone()
        )
    torch.save({k: v for k, v in out.items() if k in QUANTIZED}, ROOT / "block0.pt")
    print(f"  kept       {len(QUANTIZED)} GEMM weights: {[tuple(out[k].shape) for k in QUANTIZED]}")


def _arms(torch: Any, weight: Any, x: Any) -> dict[str, Any]:
    """One GEMM, four ways, on identical bytes.

    * `bf16`     — the unquantized truth: x_bf16 @ W_bf16.
    * `dequant`  — THE RATIFIED LANE: x_bf16 @ dequant(W_mx). Weight-only.
    * `native`   — the Blackwell route: mx(x) x mx(W) on the tensor cores. w8a8-mx.
    * `exact`    — the native route's arithmetic done in f32 by an ordinary matmul, from the
                   same quantized operands. It isolates what the KERNEL costs (accumulation
                   order, internal precision) from what the QUANTIZATION costs, which a
                   single native-vs-reference number cannot separate.
    """
    w_mx, w_scale, _ = mx_quantize(torch, weight)
    x_mx, x_scale, _ = mx_quantize(torch, x)
    w_deq = reference_dequant(torch, w_mx, w_scale, torch.bfloat16)
    x_deq = reference_dequant(torch, x_mx, x_scale, torch.float32)
    return {
        "bf16": (x @ weight.t()).float(),
        "dequant": (x @ w_deq.t()).float(),
        "native": _native(torch, x_mx, x_scale, w_mx, w_scale, torch.bfloat16).float(),
        "exact": x_deq @ reference_dequant(torch, w_mx, w_scale, torch.float32).t(),
    }


def _delta(torch: Any, got: Any, want: Any) -> dict[str, float]:
    d = float((got - want).norm() / want.norm())
    return {
        "relative_frobenius": d,
        "snr_db": -20.0 * math.log10(max(d, 1e-12)),
        "max_abs": float((got - want).abs().max()),
        "max_rel": float(((got - want).abs() / want.abs().clamp(min=1e-6)).max()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1))
        ),
    }


def weights() -> None:
    import torch

    head("weights — the native GEMM against the ratified reference, on real H3 weights")
    block = torch.load(ROOT / "block0.pt")
    torch.manual_seed(0x469A)
    banked: dict[str, Any] = {}
    for key in QUANTIZED:
        weight = block[key].cuda()
        rows, cols = int(weight.shape[0]), int(weight.shape[1])
        x = (torch.randn(512, cols, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
        arms = _arms(torch, weight, x)
        banked[key] = {
            "shape": [rows, cols],
            "native_vs_dequant": _delta(torch, arms["native"], arms["dequant"]),
            "native_vs_bf16": _delta(torch, arms["native"], arms["bf16"]),
            "dequant_vs_bf16": _delta(torch, arms["dequant"], arms["bf16"]),
            "native_vs_exact": _delta(torch, arms["native"], arms["exact"]),
        }
        d = banked[key]
        print(
            f"  {key:24} [{rows}, {cols}]\n"
            f"    native vs DEQUANT (the ratified lane)  "
            f"rel {d['native_vs_dequant']['relative_frobenius']:.6f}  "
            f"SNR {d['native_vs_dequant']['snr_db']:.3f} dB\n"
            f"    native vs BF16    (unquantized truth)  "
            f"rel {d['native_vs_bf16']['relative_frobenius']:.6f}  "
            f"SNR {d['native_vs_bf16']['snr_db']:.3f} dB\n"
            f"    dequant vs BF16   (what #511 banked)   "
            f"rel {d['dequant_vs_bf16']['relative_frobenius']:.6f}  "
            f"SNR {d['dequant_vs_bf16']['snr_db']:.3f} dB\n"
            f"    native vs EXACT   (the KERNEL alone)   "
            f"rel {d['native_vs_exact']['relative_frobenius']:.6f}  "
            f"SNR {d['native_vs_exact']['snr_db']:.3f} dB"
        )
    REPORT["weights"] = banked
    # NO PASS/FAIL THRESHOLD, for #511's reason: what an encoding is ALLOWED to cost is
    # job-009's ruling through the banked noise floors, not a constant invented here. What
    # IS assertable without inventing anything is the ORDERING these four arms imply.
    check(
        "the kernel's own error is far below the quantization's",
        all(
            v["native_vs_exact"]["relative_frobenius"] < v["native_vs_bf16"]["relative_frobenius"]
            for v in banked.values()
        ),
        "the native route's delta is dominated by w8a8-mx quantization, not by the "
        "tensor core's accumulation order",
    )
    check(
        "the native route is measurably LOSSIER than the ratified weight-only lane",
        all(
            v["native_vs_bf16"]["relative_frobenius"] > v["dequant_vs_bf16"]["relative_frobenius"]
            for v in banked.values()
        ),
        "expected and reported, not hidden: native MX quantizes the ACTIVATIONS too, "
        "which the decoded-float lane never does",
    )


# ------------------------------------------------------------------------- speed & vram


def _time(fn: Any, torch: Any, reps: int = 50) -> float:
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / reps / 1e6


def _shape_row(torch: Any, key: str, weight: Any, tokens: int) -> dict[str, Any]:
    """One (weight, token count) pair, five arms, warm.

    Every operand is a PARAMETER rather than a closed-over loop variable, so each timed
    callable can only see the shape it was built for — a benchmark that accidentally times
    the previous iteration's tensors is the classic way to publish a number nobody can
    reproduce.
    """
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    x = (torch.randn(tokens, cols, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
    w_mx, w_scale, _ = mx_quantize(torch, weight)
    x_mx, x_scale, _ = mx_quantize(torch, x)
    w_blocked = to_blocked(torch, w_scale).view(torch.float8_e8m0fnu)
    x_blocked = to_blocked(torch, x_scale).view(torch.float8_e8m0fnu)
    w_t = w_mx.t()
    w_deq = reference_dequant(torch, w_mx, w_scale, torch.bfloat16)

    def native_gemm() -> Any:
        return torch._scaled_mm(
            x_mx, w_t, scale_a=x_blocked, scale_b=w_blocked, out_dtype=torch.bfloat16
        )

    def native_full() -> Any:
        quant, scale, _ = mx_quantize(torch, x)
        return torch._scaled_mm(
            quant,
            w_t,
            scale_a=to_blocked(torch, scale).view(torch.float8_e8m0fnu),
            scale_b=w_blocked,
            out_dtype=torch.bfloat16,
        )

    def dequant_per_step() -> Any:
        return x @ reference_dequant(torch, w_mx, w_scale, torch.bfloat16).t()

    row: dict[str, Any] = {
        "key": key,
        "shape": [tokens, rows, cols],
        "activation_quant_ms": _time(lambda: mx_quantize(torch, x), torch),
        "activation_swizzle_ms": _time(lambda: to_blocked(torch, x_scale), torch),
        "bf16_ms": _time(lambda: x @ weight.t(), torch),
        "dequant_resident_bf16_ms": _time(lambda: x @ w_deq.t(), torch),
        "dequant_per_step_ms": _time(dequant_per_step, torch),
        "native_gemm_ms": _time(native_gemm, torch),
        "native_with_activation_quant_ms": _time(native_full, torch),
    }
    flops = 2.0 * tokens * rows * cols
    for name in list(row):
        if name.endswith("_ms"):
            row[name.removesuffix("_ms") + "_tflops"] = round(flops / (row[name] / 1e3) / 1e12, 2)
    return row


def speed() -> None:
    """Native vs bf16 vs dequant-then-bf16 at the real DiT block's GEMM shapes.

    The dequant arm is timed BOTH ways on purpose. Steady-state serving dequantizes once at
    fill and then runs bf16 forever, so `bf16` IS the ratified lane's per-step cost and the
    fused number is only what a re-decode-per-step design would pay. Quoting the fused
    number as the lane's cost would flatter the native route by a factor nobody pays.
    """
    import torch

    head("speed — the real DiT block GEMM shapes, warm")
    block = torch.load(ROOT / "block0.pt")
    banked = []
    for key in QUANTIZED:
        for tokens in (512, 4096):
            row = _shape_row(torch, key, block[key].cuda(), tokens)
            banked.append(row)
            rows, cols = row["shape"][1], row["shape"][2]
            print(
                f"  {key:22} M={tokens:<6} [{rows}x{cols}]\n"
                f"    activation quantize      {row['activation_quant_ms']:8.3f} ms  "
                f"(+{row['activation_swizzle_ms']:.3f} ms swizzle) — UNFUSED eager reference\n"
                f"    bf16                     {row['bf16_ms']:8.3f} ms  "
                f"{row['bf16_tflops']:7.2f} TFLOP/s\n"
                f"    dequant lane (resident)  {row['dequant_resident_bf16_ms']:8.3f} ms  "
                f"{row['dequant_resident_bf16_tflops']:7.2f} TFLOP/s\n"
                f"    dequant per step         {row['dequant_per_step_ms']:8.3f} ms  "
                f"{row['dequant_per_step_tflops']:7.2f} TFLOP/s\n"
                f"    native MX (gemm only)    {row['native_gemm_ms']:8.3f} ms  "
                f"{row['native_gemm_tflops']:7.2f} TFLOP/s\n"
                f"    native MX (+act quant)   {row['native_with_activation_quant_ms']:8.3f} ms  "
                f"{row['native_with_activation_quant_tflops']:7.2f} TFLOP/s"
            )
    REPORT["speed"] = banked
    wins = sum(1 for r in banked if r["native_gemm_ms"] < r["bf16_ms"])
    check(
        "the native MX GEMM is faster than the bf16 GEMM it would replace",
        wins == len(banked),
        f"{wins} of {len(banked)} shape/token combinations, "
        f"{min(r['bf16_ms'] / r['native_gemm_ms'] for r in banked):.2f}x to "
        f"{max(r['bf16_ms'] / r['native_gemm_ms'] for r in banked):.2f}x",
    )
    print(
        "  note       the `+act quant` arm carries THIS SCRIPT's eager quantizer (log2, pow,\n"
        "             clamp, cast, swizzle — five full-width passes over the activation), not a\n"
        "             fused kernel. It is reported because it is what was RUN, and it is the\n"
        "             number that decides whether a serving lane wins: the GEMM alone is not a\n"
        "             deliverable if reaching it costs more than it saves"
    )


def vram() -> None:
    """The claim the decoded-float lane structurally cannot make, measured.

    #511's open owner question is exactly this: decoded-float buys bytes-read, NOT VRAM,
    because the resident weights are float either way. The native route holds e4m3 payload
    plus one E8M0 byte per 32 elements — 8.25 bits — resident THROUGH the matmul. So this
    counts allocator bytes for the four GEMM weights of one real block, both ways, and
    multiplies by 50 blocks for the DiT-scale number.
    """
    import torch

    head("vram — resident bytes for one block's GEMM weights, native vs dequant")
    block = torch.load(ROOT / "block0.pt")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated()
    bf16 = [block[k].cuda() for k in QUANTIZED]
    resident_bf16 = torch.cuda.memory_allocated() - before
    quantized = [mx_quantize(torch, w) for w in bf16]
    del bf16
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # The SERVING residency, not a best case: the payload plus the ONE scale grid the kernel
    # reads. The row-major grid the quantizer produced is not held — swizzling is a load-time
    # step, so keeping both would price a copy no serving worker has any reason to keep.
    before = torch.cuda.memory_allocated()
    held = [
        (d.clone(), to_blocked(torch, s).view(torch.float8_e8m0fnu)) for d, s, _ in quantized
    ]
    resident_native = torch.cuda.memory_allocated() - before
    del quantized
    elements = sum(int(d.numel()) for d, _ in held)
    REPORT["vram"] = {
        "resident_bf16_bytes": resident_bf16,
        "resident_native_bytes": resident_native,
        "elements": elements,
        "bits_per_element_native": resident_native * 8.0 / elements,
        "blocks_in_h3_curve_dit": 50,
        "dit_gemm_bf16_gib": resident_bf16 * 50 / (1 << 30),
        "dit_gemm_native_gib": resident_native * 50 / (1 << 30),
    }
    v = REPORT["vram"]
    print(
        f"  one block  bf16 {resident_bf16 / MIB:.1f} MiB   native {resident_native / MIB:.1f} MiB"
        f"   ({resident_native / resident_bf16:.3f}x, "
        f"{v['bits_per_element_native']:.3f} bits/element)\n"
        f"  x50 blocks bf16 {v['dit_gemm_bf16_gib']:.2f} GiB   native "
        f"{v['dit_gemm_native_gib']:.2f} GiB   — the GEMM weights of the whole curve DiT, "
        f"{v['dit_gemm_bf16_gib'] - v['dit_gemm_native_gib']:.2f} GiB saved"
    )
    check(
        "the native route holds the weights at ~8.25 bits RESIDENT, which the "
        "decoded-float lane cannot",
        resident_native < resident_bf16 * 0.6,
        f"{v['bits_per_element_native']:.3f} bits/element resident against bf16's 16.0 — "
        "this is #511's open owner question answered: decoded-float buys bytes-read and "
        "load-time, the native route buys the RESIDENCY",
    )
    del held


# -------------------------------------------------------------------------- the red arm


def redarm() -> None:
    """A Hopper-class record must NOT admit the native provider.

    The plant is the one #511e's gate makes cheap: mint a record whose device predicate
    names the arch this run measured, then ask the admission table for it while standing on
    sm89 and sm90. A capability table that says yes to a card the suite never ran on is the
    exact v1 defect (`DTYPE_MIN_SM` fell OPEN on an unknown card) this plane exists to
    prevent, so the arm is observed RED before anything is claimed green.
    """
    import torch

    from cozy_runtime.internal.encoding import (
        Capabilities,
        CapabilityRecord,
        DeviceFacts,
        EncodingRefusal,
    )

    head("redarm — a Hopper-class device must not admit the native record")
    major, minor = torch.cuda.get_device_capability(0)
    measured = f"cuda.sm{major * 10 + minor}"
    caps = Capabilities(
        [
            CapabilityRecord(
                encoding_spec_id="sha256:mxfp8-native-probe",
                implementation_digest="sha256:native",
                runtime_release="cozy/mxfp8-native-probe",
                device_predicate=measured,
                output_dtype="bfloat16",
                delivery_route="native_block_scaled",
                evidence_digest="sha256:this-run",
            )
        ]
    )
    for sm, name in ((89, "L40S"), (90, "H100")):
        device = DeviceFacts("cuda", name, sm, "13.0", "default")
        try:
            caps.admit("sha256:mxfp8-native-probe", device, "bfloat16", "mxfp8/1")
        except EncodingRefusal as exc:
            check(
                f"sm{sm} ({name}) REFUSES the {measured} native record",
                exc.code == "encoding_unqualified",
                f"[{exc.code}] {str(exc)[:150]}",
            )
        else:
            check(f"sm{sm} ({name}) REFUSES the {measured} native record", False, "it ADMITTED it")
    name = str(REPORT.get("facts", {}).get("device", "?"))
    device = DeviceFacts("cuda", name, major * 10 + minor, "13.0", "default")
    try:
        caps.admit("sha256:mxfp8-native-probe", device, "bfloat16", "mxfp8/1")
    except EncodingRefusal as exc:
        check("the measured arch ADMITS its own record", False, str(exc)[:150])
    else:
        check("the measured arch ADMITS its own record", True, measured)


def dequant_lane() -> None:
    """The RATIFIED lane, qualified on this card — alongside the native measurement, never
    replacing it.

    #511 minted `cozy.mxfp8.dequant/1`'s records on sm89 and cpu, and a capability table
    that has never seen sm120 must refuse sm120: an unrecognised device is the ABSENCE of
    evidence, never permission. So the same `accel.qualification_gate` -> `probe.qualify`
    path the fill plane uses runs here, on Blackwell, and the record it mints is the reason
    a 5090 may serve this artifact through the built route TODAY — independently of whether
    the native route is ever built.
    """
    import torch

    from cozy_runtime.internal.encoding import MicroScaledDequant, measure_device
    from cozy_runtime.internal.probe import qualify

    head("dequant_lane — the RATIFIED decoded-float provider, qualified on this card")
    device = measure_device(torch)
    provider = MicroScaledDequant("sha256:mxfp8-spec-under-test")
    caps, evidence = qualify(
        torch,
        {provider.encoding: provider},
        device,
        ["bfloat16", "float32", "float16"],
        release="cozy/mxfp8-native-probe",
        cache=None,
    )
    by_dtype = {o["output_dtype"]: o["evidence"] for o in evidence["observations"]}
    print(f"  device     {device.predicate} ({device.name}, driver cuda {device.driver})")
    for name in ("bfloat16", "float32"):
        n = by_dtype[name]["numerics"]
        check(
            f"{name}: all {n['pairs']} pairs decode correctly through the FILL route on "
            f"{device.predicate}",
            by_dtype[name]["passed"],
            f"max rel {n['max_rel_error']:.3e} <= {n['tolerance']:.3e}; "
            f"{by_dtype[name]['micro_bench']['gb_per_s']} GB/s decode",
        )
    check(
        "RED here too: float16 fails on Blackwell and mints no record",
        not by_dtype["float16"]["passed"],
        f"{by_dtype['float16']['numerics']['wrong']['normal']} normal pairs wrong — "
        "#511b's measured refusal is a property of the FORMAT, not of sm89",
    )
    REPORT["dequant_lane"] = {
        "device_predicate": device.predicate,
        "records": [dataclasses.asdict(r) for r in caps.records],
        "observations": evidence["observations"],
    }
    print(f"  minted     {len(caps.records)} capability record(s) on {device.predicate}")


def save() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "native-report.json").write_text(json.dumps(REPORT, indent=1, default=str))
    print(f"\n--- native-report.json ({(ROOT / 'native-report.json').stat().st_size} B)")


LEGS = {
    "facts": facts,
    "matrix": matrix,
    "layout": layout,
    "domain": domain,
    "nan_arm": nan_arm,
    "fetch": fetch,
    "weights": weights,
    "speed": speed,
    "vram": vram,
    "dequant_lane": dequant_lane,
    "redarm": redarm,
    "save": save,
}


def main(argv: list[str]) -> int:
    names = argv or ["all"]
    if names == ["all"]:
        names = list(LEGS)
    for name in names:
        LEGS[name]()
    print(f"\n{PASS} ok, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
