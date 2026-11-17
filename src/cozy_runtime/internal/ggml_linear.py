"""llama.cpp's quantized matmuls for the `ggml.*` lanes (xs-213): MMVQ for a few rows, few-row MMQ and MMQ
above, as ggml's own MUL_MAT rule picks among them per card, type and row count (`_kernels._ggml`, vendored from
llama.cpp b12466). The weight stays its stored blocks; ggml quantizes the f32 activation to q8_1 and multiplies on
dp4a or the int8 tensor cores, writing f32.

It serves a card the module carries a kernel for once its probe agrees with the exact decode there
(`kernel_ready.verified`); elsewhere, and where a row is not whole 512-column chunks (ggml pads the last row of a
weight to them and reads that far), the weight decodes to a float one and torch multiplies.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

from functools import partial
from types import ModuleType

from cozy_runtime._kernels import ABSENT, ARCHS, NO_GGML
from cozy_runtime.internal import int8_linear, kernel_ready, prebuilt, verify
from cozy_runtime.internal.encoding.formats import Dtype, Tensor, params_of

OP = "llama.cpp-ggml"
#: ggml's type ids (`ggml.h`) of the block encodings it runs.
TYPES = {
    "ggml.q8_0": 8,
    "ggml.q5_0": 6,
    "ggml.q5_1": 7,
    "ggml.q4_k": 12,
    "ggml.q5_k": 13,
    "ggml.q6_k": 14,
}
#: ggml pads a quantized weight's last row to this many columns and its kernels read that far
#: (`MATRIX_ROW_PADDING`): a weight serves only with whole ones.
ROW_PADDING = 512
#: What a probe's output may differ from the exact decode's GEMM by: q8_1 activations, f32 sums.
BUDGET = 2e-2


def capability(torch: ModuleType, index: int) -> int:
    major, minor = torch.cuda.get_device_capability(index)
    return int(major) * 10 + int(minor)


def carried(sm: int) -> str:
    """Why the installed wheel has no ggml matmul for this card, or ""."""
    if ABSENT or NO_GGML:
        return ABSENT or NO_GGML
    if not any(target.rstrip("af") == f"sm_{sm}" for target in ARCHS["_ggml"]["cubin"]):
        return f"the wheel's _ggml carries no sm_{sm} kernel ({' '.join(ARCHS['_ggml']['cubin'])})"
    return ""


def native(torch: ModuleType, index: int) -> ModuleType | None:
    """The module on this card once its probe passes here, off any request thread; None until then, and for
    good where it fails or the wheel has no kernel for the card."""
    if carried(capability(torch, index)):
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    return kernel_ready.serve(f"{OP} cuda:{index}", partial(_verified, torch, index), capturing)


def prime(torch: ModuleType, index: int) -> None:
    """Opens this card's ggml matmuls at device init (`kernel_ready.prime`)."""
    if not carried(capability(torch, index)):
        kernel_ready.prime(f"{OP} cuda:{index}", partial(_verified, torch, index))


def _verified(torch: ModuleType, index: int, wait: bool) -> ModuleType:
    from cozy_runtime._kernels import _ggml

    sm = capability(torch, index)
    return kernel_ready.verified(
        _ggml, OP, prebuilt.artifact("_ggml"), sm, partial(_probe, torch, index)
    )


def serving(index: int) -> str:
    """What this card's ggml leaves serve now: a run's `gemm.selected`."""
    return OP if kernel_ready.ready(f"{OP} cuda:{index}") is not None else "decode"


def why(index: int) -> str:
    failed = kernel_ready.failed(f"{OP} cuda:{index}")
    return "" if failed is None else str(failed)


def kernel(torch: ModuleType, index: int, scheme: str, k: int, n: int, m: int) -> str:
    """The kernel ggml's rule takes for `m` rows of a [n, k] `scheme` weight here: mmvq, mmq_few or mmq."""
    from cozy_runtime._kernels import _ggml

    with torch.cuda.device(index):
        return str(_ggml.KERNELS[_ggml.kernel(index, TYPES[scheme], k, n, m)])


def matmul(
    ops: ModuleType, torch: ModuleType, x: Tensor, blocks: Tensor, scheme: str, n: int
) -> Tensor:
    """f32 `x[..., K] @ W.T` for `n` rows of `scheme` blocks; raises what the module said."""
    k = int(x.shape[-1])
    rows = x.reshape(-1, k).to(torch.float32).contiguous()
    out = torch.empty(rows.shape[0], n, dtype=torch.float32, device=x.device)
    index = x.get_device()

    def allocate(nbytes: int) -> Tensor:
        held: Tensor = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
        return held

    code = ops.matmul(
        index, TYPES[scheme], blocks.data_ptr(), rows.data_ptr(), out.data_ptr(), k, n,
        int(rows.shape[0]), allocate, torch.cuda.current_stream(x.device).cuda_stream,
    )  # fmt: skip
    if code != 0:
        said = ops.error() if code == ops.UNKNOWN else ops.error_name(code)
        raise RuntimeError(f"ggml matmul ({scheme}, {k}x{n}, {rows.shape[0]} rows): {said}")
    shaped: Tensor = out.view(*x.shape[:-1], n)
    return shaped


def linear(
    torch: ModuleType,
    x: Tensor,
    blocks: Tensor,
    scheme: str,
    n: int,
    bias: Tensor | None,
    out_dtype: Dtype,
    decode: object,
) -> Tensor:
    """`x @ W.T + bias` for one ggml weight: its blocks on this card's kernels, else the weight decoded by
    `decode()` (a float [n, k]) and multiplied by torch."""
    index = x.get_device() if x.is_cuda else -1
    ops = native(torch, index) if index >= 0 and int(x.shape[-1]) % ROW_PADDING == 0 else None
    if ops is not None:
        try:
            y = matmul(ops, torch, x, blocks, scheme, n)
            if bias is not None:
                y = y + bias.to(torch.float32)
            return y.to(out_dtype)
        except Exception as exc:
            int8_linear.failed(OP, index, exc)
    weight: Tensor = decode()  # type: ignore[operator]
    out: Tensor = torch.nn.functional.linear(x.to(weight.dtype), weight, bias)
    return out.to(out_dtype)


def _probe(torch: ModuleType, index: int, ops: ModuleType) -> verify.Verdict:
    """Each type at the rows each kernel serves (1, 6 and 48), against f64 over the exact decode."""
    from cozy_runtime.internal.decode_schemes import BY_SCHEME
    from cozy_runtime.internal.encoding.adapters import CoreDequant

    device = torch.device("cuda", index)
    generator = torch.Generator(device="cpu").manual_seed(0x6611)
    n, k = 256, 1024
    errors: dict[str, float] = {}
    for scheme in TYPES:
        weight = torch.randn(n, k, generator=generator) * 0.05
        blocks = encode(torch, scheme, weight).to(device)
        exact = torch.empty(n, k, dtype=torch.float32, device=device)
        core = CoreDequant(scheme, params_of(BY_SCHEME[scheme].probe_params))
        core.decode(torch, {"blocks": blocks}, exact)
        for m in (1, 6, 48):
            x = (torch.randn(m, k, generator=generator) * 0.5).to(device)
            got = matmul(ops, torch, x, blocks, scheme, n)
            want = x.double() @ exact.double().t()
            errors[f"{scheme} m={m}"] = verify.error(torch, got, want)
    return verify.within(errors, BUDGET)


def encode(torch: ModuleType, scheme: str, weight: Tensor) -> Tensor:
    """`weight`'s blocks by TensorFS's ggml-ref/1, the producer that writes the lanes."""
    from tensorfs import raw

    rows, cols = (int(s) for s in weight.shape)
    encoded = raw.quantize(
        scheme, weight.float().contiguous().numpy().tobytes(), dtype="F32", shape=[rows, cols],
        strategy="ggml-ref/1", params={"block_size": BLOCK[scheme]},
    )  # fmt: skip
    data = bytearray(encoded.parts["blocks"].tobytes())
    blocks: Tensor = torch.frombuffer(data, dtype=torch.uint8).reshape(
        rows, cols // BLOCK[scheme], -1
    )
    return blocks


#: Elements per block of each type.
BLOCK = {
    "ggml.q8_0": 32,
    "ggml.q5_0": 32,
    "ggml.q5_1": 32,
    "ggml.q4_k": 256,
    "ggml.q5_k": 256,
    "ggml.q6_k": 256,
}
