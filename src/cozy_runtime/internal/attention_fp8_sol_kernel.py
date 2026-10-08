# SPDX-License-Identifier: Apache-2.0
"""Experimental range-bounded FP8/F16 exact tiles within upstream Sol routing.

Derived from TeleFuser f21457a92e92f4a7570f0cb6b445ee334f1072b0,
telefuser/kernel/sol_attn/triton_ref/fwd.py, itself derived from NVlabs/Sana.
See native/NOTICE and native/LICENSE for provenance and licensing.
Changes: no autotuning; preserve original BF16 routing/summary operands; use
amax16 E4M3 exact operands and FP16 tile MMA; widen into FP32 online state.
"""

import triton
import triton.language as tl


@triton.jit
def quantize_blocks(source, quantized, scales, T, H: tl.constexpr, D: tl.constexpr):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = block * 64 + tl.arange(0, 64)
    dims = tl.arange(0, D)
    offsets = ((batch * T + rows[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    values = tl.load(source + offsets, mask=rows[:, None] < T, other=0.0).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = tl.maximum(amax / 16.0, 1.1754943508222875e-38)
    codes = tl.minimum(tl.maximum(values / scale, -16.0), 16.0).to(tl.float8e4nv)
    tl.store(quantized + offsets, codes, mask=rows[:, None] < T)
    tl.store(scales + (batch * tl.cdiv(T, 64) + block) * H + head, scale)


@triton.jit
def forward_kernel(
    q_ptr,
    q_route_ptr,
    k_ptr,
    v_ptr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    kc_ptr,
    vc_ptr,
    threshold_ptr,
    o_ptr,
    scale,
    T,
    NPAD,
    sink_start_block,
    sink_end_block,
    HAS_SINK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(
        tl.arange(0, GROUP_SIZE),
        GROUP_SIZE,
    )
    token_offsets = tl.max_contiguous(
        tl.arange(0, BLOCK_SIZE),
        BLOCK_SIZE,
    )
    dims = tl.arange(0, D)
    value_dims = v_tile * BV + tl.arange(0, BV)
    q_tokens = q_block * BLOCK_SIZE + token_offsets
    q_valid = q_tokens < T
    q_offsets = ((batch * T + q_tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
    q_scale = tl.load(q_scale_ptr + (batch * NT + q_block) * H + head)
    q_route = tl.load(q_route_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK_SIZE, T - q_block * BLOCK_SIZE).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    route_threshold = tl.load(threshold_ptr + (batch * NT + q_block) * H + head)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc_offsets = ((batch * NPAD + block_indices[:, None]) * H + head) * D + dims[None, :]
        vc_offsets = ((batch * NPAD + block_indices[:, None]) * H + head) * D + value_dims[None, :]
        kc = tl.load(kc_ptr + kc_offsets, mask=valid[:, None], other=0.0)
        vc = tl.load(vc_ptr + vc_offsets, mask=valid[:, None], other=0.0)
        scores = tl.dot(q_route, kc.T).to(tl.float32) * scale_log2
        exact = (tl.sum(scores, axis=0) / q_len > route_threshold) | (
            tl.abs(q_block - block_indices) <= 1
        )
        if HAS_SINK:
            exact = exact | ((block_indices >= sink_start_block) & (block_indices < sink_end_block))
        exact = exact & valid

        approximate = valid & ~exact
        has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0
        approximate_scores = tl.where(
            approximate[None, :],
            scores,
            -float("inf"),
        )
        safe_scores = tl.where(has_approximate, approximate_scores, 0.0)
        candidate_max = tl.maximum(row_max, tl.max(safe_scores, axis=1))
        new_max = tl.where(has_approximate, candidate_max, row_max)
        alpha = tl.math.exp2(tl.where(has_approximate, row_max - new_max, 0.0))
        probability = tl.math.exp2(safe_scores - tl.where(has_approximate, new_max, 0.0)[:, None])
        probability = tl.where(
            has_approximate & approximate[None, :],
            probability,
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            probability.to(vc.dtype),
            vc,
        )
        lengths = tl.minimum(
            BLOCK_SIZE,
            tl.maximum(0, T - block_indices * BLOCK_SIZE),
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            probability * lengths[None, :],
            axis=1,
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        num_exact = tl.sum(exact.to(tl.int32), axis=0)
        for _ in range(num_exact):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset,
                GROUP_SIZE,
                exact_offsets,
            )
            kv_tokens = block * BLOCK_SIZE + token_offsets
            kv_valid = kv_tokens < T
            k_offsets = ((batch * T + kv_tokens[:, None]).to(tl.int64) * H + head) * D + dims[
                None, :
            ]
            k = tl.load(
                k_ptr + k_offsets,
                mask=kv_valid[:, None],
                other=0.0,
            )
            k_scale = tl.load(k_scale_ptr + (batch * NT + block) * H + head)
            # Each FP16 matrix accumulator covers D=128 products bounded by16^2.
            exact_scores = tl.dot(q, k.T, out_dtype=tl.float16).to(tl.float32)
            exact_scores *= q_scale * k_scale * scale_log2
            exact_scores += tl.where(
                kv_valid[None, :],
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(
                exact_probability,
                axis=1,
            )
            v_offsets = ((batch * T + kv_tokens[:, None]).to(tl.int64) * H + head) * D + value_dims[
                None, :
            ]
            v = tl.load(
                v_ptr + v_offsets,
                mask=kv_valid[:, None],
                other=0.0,
            )
            v_scale = tl.load(v_scale_ptr + (batch * NT + block) * H + head)
            probability_scale = tl.maximum(tl.max(exact_probability, axis=1), 1.0e-6) / 16.0
            probability_fp8 = tl.minimum(exact_probability / probability_scale[:, None], 16.0).to(
                tl.float8e4nv
            )
            # FP16 covers one N64 product; the running online-softmax state stays FP32.
            partial = tl.dot(probability_fp8, v, out_dtype=tl.float16).to(tl.float32)
            output = output * alpha[:, None] + partial * (probability_scale[:, None] * v_scale)
            row_max = new_max

    output_offsets = ((batch * T + q_tokens[:, None]).to(tl.int64) * H + head) * D + value_dims[
        None, :
    ]
    tl.store(
        o_ptr + output_offsets,
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=q_valid[:, None],
    )
