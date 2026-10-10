# Sage3 preprocessing across head partitions: candidate design

Status: source-backed design only. No candidate GPU kernel, inference run, serving default, installed package, or rental was changed. The existing `49486473` head-probe package remains frozen. A future experiment needs a separate captured package and an available authorized Blackwell lease; do not keep or rent a card while implementing this design.

## Evidence and its boundary

Ordinary CLI run **5218** on an RTX 5090, Runtime 0.23.2, PyTorch 2.14.0/CUDA 13.0 compared fixed BF16 Q/K/V with full `B=1,N=109225,H=56,D=128` and contiguous head groups. Sage3 full-length groups of 14 and 8 heads differed from the full call in 41 final BF16 elements, with maximum absolute difference `3.0517578125e-5`. Groups of 28 and 7 heads, all small cases, and all Kitchen INT8 controls were exact. Full and grouped repeat controls were exact and finite.

Before FP4 quantization, the separately repeated upstream preprocessing API already differed: 14-head groups changed 88 Q-centered elements and 10 correction elements; 8-head groups changed 175 Q-centered, 56 K-centered, and 11 correction elements. V was unchanged. The probe did not retain Q/K means or independently replace only the correction term. This is a concrete preprocessing difference for fixed operands, not proof of the first divergence in an actual H3 video or of its quality impact. It also does not prove that FP4 quantization and attention become exact once preprocessing is fixed.

The original report is `h3-dense-quality-recheck-20261009/diagnostics/head/outputs/5218-report.json`, SHA256 `f375146c78cc21eced52558c80f70a8d69e73dc638bda2ceab7e72e971571d72`; the concise analysis is `native/parallel-result-identity/rtx5090-head-5218-summary.json` in the multi-GPU campaign outputs.

## Exact arithmetic to preserve

The retained [SageAttention API at d1a57a](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell/sageattn3/api.py#L75) was compared byte-for-byte with the immutable upstream URL. SHA256: `9aec83f8d7165e3b563ba4fc91f6ca5e09b07ef3445b6db5d1105912e35dfb90`.

Its global-mean path performs these operations in order:

1. Compute the K mean over original live tokens, then subtract it in place.
2. Pad Q, centered K, and V to a multiple of 128 and make them contiguous.
3. Compute Q's mean over the padded sequence, then subtract it.
4. Multiply Q's mean by centered K; the BF16 matmul result is then converted to FP32 for `delta_s`.

Consequently, for this case K uses denominator **109225**, Q uses **109312**, padded Q is **minus its mean**, and padded K/V remain zero. BF16 rounding at means, centered outputs, and the correction before its FP32 conversion must remain explicit. Switching to a live-token Q mean, per-block means, or an unrounded FP32 correction would change the recipe.

[PyTorch 2.14's CUDA mean](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/cuda/ReduceMomentKernel.cu#L41) accumulates BF16 inputs in FP32 and projects to the output type. Its [reduction configuration](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/cuda/Reduce.cuh#L1039) depends on strides, output count, device occupancy, and potentially multiple blocks per output. [MeanOps](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/SharedReduceOps.h#L152) supplies the sum and reciprocal factor. Changing H changes the number of mean outputs, so a changed reduction tree is a source-backed mechanism. The actual launch configuration was not captured; a specific CUDA kernel/CTA choice is not proven by the current report.

## Smallest useful control

Before implementing a fused replacement, expose the unmodified upstream intermediates `km`, `qm`, centered Q/K, BF16 correction, and FP32 correction in a separate diagnostic. Compare full versus grouped calls at each boundary.

As a small-code control, compute each mean through a contiguous token-last copy, then restore only the small mean tensor's axes. This makes the reduction dimension contiguous without Python loops over heads. It is a hypothesis to test, not a determinism guarantee: PyTorch still chooses its reduction configuration. Merely reshaping or transposing a view does not change the underlying strided loads or establish a fixed reduction tree.

This control adds full Q/K copies. At 56 heads it adds approximately **6.27 GB of logical read/write traffic** per call, plus temporary storage. It is useful for localization but should not become the serving default without timing and peak-memory evidence. Avoid expanding either input to FP32 solely to use an eager elementwise correction: that adds large intermediates and changes rounding unless deliberately restored.

## Proposed serving candidate, if localization supports it

Keep the existing FP4 quantizers and attention kernel unchanged. Replace only preprocessing behind an explicit experimental selection. Limit the initial implementation to the measured H3 case: batch one, BF16, head dimension 128, equal Q/K/V head counts, global means, unmasked attention. Preserve input ownership; do not mutate shared Q/K/V or monkeypatch a global upstream API.

Use fixed token/channel tiles and a fixed reduction tree independent of H, rank, grid size, or SM occupancy:

- **Partial means:** one program owns a token tile and channel tile of one head. A conservative initial design is 256 tokens by 32 channels with a fixed warp count. The head index changes only pointer offsets. Produce FP32 partial sums in fixed token order; Q's absent padded values are zeros, and K's mask ends at the original length. No atomics or cross-head reduction. Q/K may share one launch with separate program roles; no Python head loop.
- **Finish means:** a second fixed tree combines the token partials for each head/channel. Apply FP32 reciprocals for the correct Q/K denominators, then round means to BF16. Tile count depends on sequence length only. At this shape, both means' FP32 partial storage is approximately 23.35 MiB for the proposed 256-token tiles. This is an initial resource estimate, not a tuned launch choice.
- **Center/pad and correction:** a fixed token tile loads the original inputs and BF16 means, rounds centered K to BF16 before using it, and writes contiguous centered Q/K and padded V. Compute the 128-channel correction with a fixed FP32 multiply/reduction tree, round its result to BF16, then store FP32. Preserve padded Q as `-qm`; padded K/V/correction stay zero. A fused program can remove the current K clone/centering/padding passes and Q padding/centering passes, creating a plausible way to pay for deterministic reduction without extra full copies.

The correction requires its own control: even with identical means and K, batched matmul may select different arithmetic across head counts. A fixed 128-channel reduction removes that selection dependence, but need not round identically to cuBLAS. Preserve the existing BF16 rounding point and separately measure its differences from upstream. Do not claim unchanged numerical quality merely because the output dtype matches.

Do not autotune these arithmetic trees on H. Fixed tiling gives a reason to expect partition-invariant operations; generated-code differences, memory behavior, and actual output identity still require GPU evidence. The candidate may intentionally differ from the old upstream result while becoming consistent across partitions. Matching every old head-dependent result simultaneously is impossible where those results already disagree.

## Bounded next GPU experiment

Prepare and review the new local diagnostic package entirely on CPU first. Use the ordinary default-home CLI only after human quality review or on the next explicitly assigned idle Blackwell. No new rental is authorized by this design.

One request, with an **eight-minute actual invocation budget**, should prioritize:

1. Fixed seed 581204, N=129 and N=109225, H=56 against groups of 28/14/8/7. Retain full/group same-configuration repeat controls and actual strides, alignment, software/source hashes, SM properties, and matmul reduction flags without changing global settings.
2. Upstream intermediates and the token-last mean control. Then test fixed means with upstream correction, followed by fixed means plus fixed correction. This separates mean and correction changes. If resources or budget are inadequate, report a named untested arm; do not silently shrink the full case.
3. Require byte identity across partitions for each candidate intermediate, FP4 payload/scale, and final output, with all outputs finite. Report candidate-versus-upstream differences separately. Preserve rounding/casting, padded-tail, and exact-input controls. A second fixed seed is useful only after the first complete matrix fits the same budget.
4. Benchmark warmed upstream and candidate preprocessing and complete attention separately, alternating their order for repeated CUDA-event samples. Exclude compilation, metric scans, and probe copies from those intervals; record peak memory and ordinary request duration separately. Reject a default change if complete attention becomes measurably slower or memory pressure increases. Kernel event measurements alone do not establish full-video parity.

Only a passing diagnostic justifies a subsequent matched full-video experiment with normal CLI inputs, runtime counters, phase timings, peak memory, and the user's visual quality gate. No claim of full 1/4/8-GPU video determinism, no-slowdown behavior, or a production-ready implementation is made here.
