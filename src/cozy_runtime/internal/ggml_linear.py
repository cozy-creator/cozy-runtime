"""llama.cpp's quantized matmuls for the `ggml.*` lanes (xs-213): MMVQ for a few rows, few-row
MMQ and MMQ above, as ggml's own MUL_MAT rule picks among them per card, type and row count
(`_kernels._ggml`, vendored from llama.cpp b12466). The weight stays its stored blocks; the
activation (bf16, f16 or f32) is quantized to q8_1, and the kernels multiply on dp4a or the int8
tensor cores, writing f32.

It serves a card the module carries a kernel for once its probe agrees with the exact decode there
(`kernel_ready.verified`), for a weight whose rows are whole 512-column chunks (ggml pads the last
row of a weight to them and reads that far), contiguous and 16-byte aligned. Elsewhere the weight
decodes a band of rows at a time and torch multiplies.

A call's own memory beyond its input and output is at most `WORKING`: its rows go through in
pieces (each piece's q8_1 activation, an f32 copy of other dtypes, and its f32 product), or its
weight's bands decode one at a time. That is below the floor a site plans a cut for
(`execution.catch`), so a site plans a ggml linear as it plans any other.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

from functools import partial
from types import ModuleType

from cozy_runtime._kernels import ABSENT, ARCHS, NO_GGML
from cozy_runtime.internal import int8_linear, kernel_ready, prebuilt, verify
from cozy_runtime.internal.encoding import ggml_types
from cozy_runtime.internal.encoding.formats import Dtype, Tensor, params_of

OP = "llama.cpp-ggml"
#: ggml's type ids (`ggml.h`) of the block encodings it runs.
TYPES = ggml_types.WRITTEN
#: ggml pads a quantized weight's last row to this many columns and its kernels read that far
#: (`MATRIX_ROW_PADDING`): a weight serves only with whole ones.
ROW_PADDING = 512
#: What a probe's output may differ from the exact decode's GEMM by: q8_1 activations, f32 sums.
BUDGET = 2e-2
#: A call's memory beyond its input and output, at most (module doc).
WORKING = 32 << 20


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
    """The module on this card once its probe passes here, off any request thread; None until
    then, and for
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


def kernel(
    torch: ModuleType, index: int, scheme: str, k: int, n: int, m: int, blocks: Tensor
) -> str:
    """The kernel ggml's rule takes for `m` rows of this [n, k] `scheme` weight here: mmvq,
    mmq_few or mmq."""
    from cozy_runtime._kernels import _ggml

    with torch.cuda.device(index):
        return str(_ggml.KERNELS[_ggml.kernel(index, TYPES[scheme], k, n, m, blocks.data_ptr())])


def unserved(blocks: Tensor, k: int) -> str:
    """Why ggml's kernels cannot read this weight as it is held, or ""."""
    if k % ROW_PADDING:
        return (
            f"ggml's kernels read rows in {ROW_PADDING}-column chunks; this weight's are {k} long"
        )
    if not blocks.is_contiguous():
        return "ggml's kernels read a weight's blocks contiguous, and this one is strided"
    if blocks.data_ptr() % 16:
        return "ggml's kernels read a weight 16-byte aligned, and this one is not"
    return ""


def matmul(
    ops: ModuleType, torch: ModuleType, x: Tensor, blocks: Tensor, scheme: str, n: int,
    out: Tensor | None = None,
) -> Tensor:
    """f32 `x[..., K] @ W.T` for `n` rows of `scheme` blocks, into `out` (contiguous f32
    `[rows, n]`) where given; raises what the module said. A bf16 or f16 `x` is quantized to q8_1
    as it is; another dtype goes through f32."""
    k = int(x.shape[-1])
    name = str(x.dtype).removeprefix("torch.")
    if name not in ops.ACTIVATIONS:
        x, name = x.to(torch.float32), "float32"
    rows = x.reshape(-1, k).contiguous()
    if out is None:
        out = torch.empty(rows.shape[0], n, dtype=torch.float32, device=x.device)
    index = x.get_device()

    def allocate(nbytes: int) -> Tensor:
        held: Tensor = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
        return held

    code = ops.matmul(
        index, TYPES[scheme], blocks.data_ptr(), rows.data_ptr(), ops.ACTIVATIONS.index(name),
        out.data_ptr(), k, n, int(rows.shape[0]), allocate,
        torch.cuda.current_stream(x.device).cuda_stream,
    )  # fmt: skip
    if code != 0:
        said = ops.error() if code == ops.UNKNOWN else ops.error_name(code)
        raise RuntimeError(f"ggml matmul ({scheme}, {k}x{n}, {rows.shape[0]} rows): {said}")
    shaped: Tensor = out.view(*x.shape[:-1], n)
    return shaped


def ids_unserved(torch: ModuleType, index: int) -> str:
    """Why this card's ggml kernels can't run a mixture's routed products (`matmul_ids`) now, or "":
    the wheel's `_ggml` has no MUL_MAT_ID, or its probe hasn't passed here (`VERIFYING` while it runs)."""
    if absent := carried(capability(torch, index)):
        return absent
    ops = native(torch, index)
    if ops is None:
        return why(index) or VERIFYING
    if not hasattr(ops, "matmul_ids"):
        return "this wheel's _ggml has no MUL_MAT_ID (matmul_ids)"
    return ""


#: `ids_unserved` while this card's ggml probe runs: it serves from the next request.
VERIFYING = "the ggml kernels are being verified on this card"


def matmul_ids(
    ops: ModuleType, torch: ModuleType, x: Tensor, table: Tensor, ids: Tensor, scheme: str, n: int,
    stride: int = 0,
) -> Tensor:
    """f32 `[T * k, n]`: each (token t, slot j) pair's product of its row of `x` (f32, `[T, K]`, a row a
    token, or `[T * k, K]`, a row a pair) by the `[n, K]` `scheme` blocks of expert `ids[t, j]` (int32),
    read at `table[expert]` (int64 addresses on the device), or with `stride` (bytes) at `table`'s address
    plus `expert * stride` (one stack of experts, llama.cpp's layout): llama.cpp's MUL_MAT_ID by MMVQ (`ops`,
    this card's `native`), a few tokens a launch. Raises what the module said; `ids_unserved` first."""
    tokens, k = (int(s) for s in ids.shape)
    rows, width = (int(s) for s in x.shape)
    if rows not in (tokens, tokens * k):
        raise ValueError(f"{rows} rows of x for {tokens} tokens of {k} experts")
    out: Tensor = torch.empty(tokens * k, n, dtype=torch.float32, device=x.device)

    def allocate(nbytes: int) -> Tensor:
        held: Tensor = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
        return held

    code = ops.matmul_ids(
        x.get_device(), TYPES[scheme], table.data_ptr(), stride, x.data_ptr(), rows // tokens, ids.data_ptr(),
        out.data_ptr(), width, n, tokens, k, allocate, torch.cuda.current_stream(x.device).cuda_stream,
    )  # fmt: skip
    if code != 0:
        said = ops.error() if code == ops.UNKNOWN else ops.error_name(code)
        raise RuntimeError(f"ggml MUL_MAT_ID ({scheme}, {width}x{n}, {tokens} tokens of {k}): {said}")
    return out


def products(torch: ModuleType, x: Tensor, weights: list[tuple[Tensor, str, int]]) -> list[Tensor] | None:
    """Each weight's f32 product `x[..., K] @ W.T` of one 16-bit `x`, from one q8_1 quantize of it,
    where ggml's rule takes MMVQ for every one of them on this card (a few rows): the quantize
    sibling linears of one input would each launch, once. `weights` are (blocks, scheme, n). None
    where they can't share it (another kernel, another dtype, a module without it), for the caller
    to call each linear."""
    if not x.is_cuda:
        return None
    index = x.get_device()
    ops = native(torch, index)
    name = str(x.dtype).removeprefix("torch.")
    if ops is None or not hasattr(ops, "quantize") or name not in ("bfloat16", "float16"):
        return None
    k = int(x.shape[-1])
    rows = x.reshape(-1, k).contiguous()
    m = int(rows.shape[0])
    for blocks, scheme, n in weights:
        if unserved(blocks, k) or kernel(torch, index, scheme, k, n, m, blocks) != "mmvq":
            return None
    stream = torch.cuda.current_stream(x.device).cuda_stream
    q8_1: Tensor = torch.empty(int(ops.quantized_bytes(k, m)), dtype=torch.uint8, device=x.device)
    code = ops.quantize(index, rows.data_ptr(), ops.ACTIVATIONS.index(name), q8_1.data_ptr(), k, m, stream)
    if code != 0:
        raise RuntimeError(f"ggml quantize ({k} columns, {m} rows): {ops.error_name(code)}")

    def allocate(nbytes: int) -> Tensor:
        held: Tensor = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
        return held

    made: list[Tensor] = []
    for blocks, scheme, n in weights:
        out: Tensor = torch.empty(m, n, dtype=torch.float32, device=x.device)
        code = ops.matmul_quantized(
            index, TYPES[scheme], blocks.data_ptr(), q8_1.data_ptr(), out.data_ptr(), k, n, m, allocate, stream
        )
        if code != 0:
            said = ops.error() if code == ops.UNKNOWN else ops.error_name(code)
            raise RuntimeError(f"ggml matmul of a shared q8_1 ({scheme}, {k}x{n}, {m} rows): {said}")
        made.append(out.view(*x.shape[:-1], n))
    return made


def linear(
    torch: ModuleType,
    x: Tensor,
    blocks: Tensor,
    scheme: str,
    n: int,
    bias: Tensor | None,
    out_dtype: Dtype,
) -> Tensor:
    """`x @ W.T + bias` for one ggml weight: its blocks on this card's kernels, a piece of rows at a
    time, else decoded a band at a time and multiplied by torch (module doc)."""
    k = int(x.shape[-1])
    rows = x.reshape(-1, k)
    m = int(rows.shape[0])
    out: Tensor = torch.empty(m, n, dtype=out_dtype, device=x.device)
    index = x.get_device() if x.is_cuda else -1
    ops = native(torch, index) if index >= 0 else None
    if ops is not None:
        try:
            if why := unserved(blocks, k):
                raise ValueError(why)
            step = max(1, WORKING // (4 * k + 2 * k + 4 * n))
            for at in range(0, m, step):
                if out_dtype == torch.float32:  # the product as the kernel writes it: no copy
                    y = matmul(ops, torch, rows[at : at + step], blocks, scheme, n, out[at : at + step])
                else:
                    y = matmul(ops, torch, rows[at : at + step], blocks, scheme, n)
                if bias is not None:
                    y += bias.to(torch.float32)
                if out_dtype != torch.float32:
                    out[at : at + step].copy_(y)
                del y
            return out.view(*x.shape[:-1], n)
        except Exception as exc:
            int8_linear.failed(OP, index, exc)
    band = max(1, WORKING // (rows.element_size() * k + out.element_size() * m))
    for at in range(0, n, band):
        weight = decoded(torch, scheme, blocks[at : at + band], rows.dtype)
        part = bias[at : at + band].to(rows.dtype) if bias is not None else None
        out[:, at : at + band] = torch.nn.functional.linear(rows, weight, part)
        del weight
    return out.view(*x.shape[:-1], n)


def decoded(torch: ModuleType, scheme: str, blocks: Tensor, dtype: Dtype) -> Tensor:
    """Rows of a weight as floats [rows, K], by the decode cores."""
    from cozy_runtime.internal.decode_schemes import BY_SCHEME
    from cozy_runtime.internal.encoding.adapters import CoreDequant

    out: Tensor = torch.empty(
        int(blocks.shape[0]), int(blocks.shape[1]) * ggml_types.block(scheme), dtype=dtype,
        device=blocks.device,
    )  # fmt: skip
    CoreDequant(scheme, params_of(BY_SCHEME[scheme].probe_params)).decode(
        torch, {"blocks": blocks}, out
    )
    return out


def _probe(torch: ModuleType, index: int, ops: ModuleType) -> verify.Verdict:
    """Each type at the rows each kernel serves (1, 6 and 48), against f64 over the exact decode."""
    from cozy_runtime.internal.decode_schemes import BY_SCHEME
    from cozy_runtime.internal.encoding.adapters import CoreDequant

    device = torch.device("cuda", index)
    generator = torch.Generator(device="cpu").manual_seed(0x6611)
    n, k = 256, 1024
    errors: dict[str, float] = {}
    #: a failing MUL_MAT_ID case's output rows that are not finite
    unfinite: dict[str, str] = {}
    for scheme in TYPES:
        weight = torch.randn(n, k, generator=generator) * 0.05
        blocks = encode(torch, scheme, weight).to(device)
        exact = torch.empty(n, k, dtype=torch.float32, device=device)
        core = CoreDequant(scheme, params_of(BY_SCHEME[scheme].probe_params))
        core.decode(torch, {"blocks": blocks}, exact)
        for m in (1, 6, 48):
            x = (torch.randn(m, k, generator=generator) * 0.5).to(device)
            for dtype in (torch.float32, torch.bfloat16):
                rows = x.to(dtype)
                got = matmul(ops, torch, rows, blocks, scheme, n)
                want = rows.double() @ exact.double().t()
                errors[f"{scheme} m={m} {str(dtype).removeprefix('torch.')}"] = verify.error(
                    torch, got, want
                )
        if hasattr(ops, "matmul_ids"):
            errors.update(_probe_ids(torch, ops, index, generator, scheme, blocks, exact, unfinite))
    found = verify.within(errors, BUDGET)
    failing = [f"{case} {errors[case]:.2g}{unfinite.get(case, '')}" for case in errors if errors[case] > BUDGET]
    return found if found.ok else verify.Verdict(False, f"{found.detail}; failing: {'; '.join(failing[:16])}")


def _probe_ids(
    torch: ModuleType, ops: ModuleType, index: int, generator: object, scheme: str, blocks: Tensor,
    exact: Tensor, unfinite: dict[str, str],
) -> dict[str, float]:
    """MUL_MAT_ID over a table of three experts (`blocks` and two more of `scheme`) at 1, 3 and 9 tokens
    (one token's kernel, several tokens', two launches) of 2 slots, a row a token and a row a pair, against
    each pair's f64 product over the exact decode; and the same over the three as one stack (llama.cpp's
    own layout, `stacked`). A case whose output is not finite names its rows in `unfinite`."""
    from cozy_runtime.internal.decode_schemes import BY_SCHEME
    from cozy_runtime.internal.encoding.adapters import CoreDequant

    device = torch.device("cuda", index)
    n, k = (int(s) for s in exact.shape)
    held, decoded = [blocks], [exact]
    core = CoreDequant(scheme, params_of(BY_SCHEME[scheme].probe_params))
    for _ in range(2):
        more = encode(torch, scheme, torch.randn(n, k, generator=generator) * 0.05).to(device)
        out = torch.empty(n, k, dtype=torch.float32, device=device)
        core.decode(torch, {"blocks": more}, out)
        held.append(more)
        decoded.append(out)
    table = torch.tensor([b.data_ptr() for b in held], dtype=torch.int64, device=device)
    stack = torch.stack(held).contiguous()
    errors: dict[str, float] = {}
    for tokens in (1, 3, 9):
        ids = torch.randint(0, len(held), (tokens, 2), generator=generator, dtype=torch.int32).to(device)
        for per in (1, 2):
            x = (torch.randn(tokens * per, k, generator=generator) * 0.5).to(device)
            rows = x.double().repeat_interleave(2 // per, dim=0)
            weights = torch.stack([decoded[int(e)] for e in ids.reshape(-1).tolist()]).double()
            want = torch.einsum("pk,pnk->pn", rows, weights)
            for layout, base, stride in (("", table, 0), (" stacked", stack, int(stack[0].numel()))):
                got = matmul_ids(ops, torch, x, base, ids, scheme, n, stride)
                case = f"{scheme} ids t={tokens} rows={per}{layout}"
                errors[case] = verify.error(torch, got, want)
                bad = (~torch.isfinite(got)).any(-1).nonzero().reshape(-1).tolist()
                if bad:
                    unfinite[case] = f" (rows {bad} of {int(got.shape[0])} not finite)"
    return errors


def encode(torch: ModuleType, scheme: str, weight: Tensor) -> Tensor:
    """`weight`'s blocks by TensorFS's ggml-ref/1, the producer that writes the lanes."""
    from tensorfs import raw

    rows, cols = (int(s) for s in weight.shape)
    encoded = raw.quantize(
        scheme, weight.float().contiguous().numpy().tobytes(), dtype="F32", shape=[rows, cols],
        strategy="ggml-ref/1", params={"block_size": ggml_types.block(scheme)},
    )  # fmt: skip
    data = bytearray(encoded.parts["blocks"].tobytes())
    blocks: Tensor = torch.frombuffer(data, dtype=torch.uint8).reshape(
        rows, cols // ggml_types.block(scheme), -1
    )
    return blocks
