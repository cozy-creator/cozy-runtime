# Sage3 fixed preprocessing experiment

This isolated research package compares the pinned upstream global-mean Sage3 path with fixed token reductions and correction. It does not register a serving kernel, modify an installed Runtime, or change an attention default. The existing head probe and forty-video quality study remain unchanged.

The motivating RTX 5090 experiment found different Sage3 results for identical full-length operands partitioned into different head counts. Differences were already visible in preprocessing. This package investigates that boundary; it does not prove actual multi-GPU video determinism.

## Comparisons

One ordinary CLI request uses seed 581204, 56 heads, head dimension 128, and sequence lengths 129 and 109225. Each full call is compared with contiguous groups representing degrees 2, 4, 7 and 8. It compares full/group repeats, original input bytes, means, centered/padded Q/K/V, BF16 and FP32 correction, public FP4 payloads/scales, and final output.

Four variants isolate the changes:

| Variant | Mean reduction | Correction |
| --- | --- | --- |
| Upstream replay | Pinned upstream expressions | Upstream BF16 matmul, then FP32 |
| Token-last control | Contiguous token-last copy, then PyTorch mean | Upstream correction |
| Fixed means control | Fixed token tiles/tree | Upstream correction |
| Fixed candidate | Fixed token tiles/tree | Fixed 128-channel tree, BF16 rounding, then FP32 |

The replay must match the actual pinned upstream preprocessing API and the public Runtime attention callable. The fixed candidate's instrumented record must match its lean timed callable. A mismatch refuses the diagnostic rather than presenting the control as equivalent.

The candidate uses 256-token × 32-channel partial sums, a second fixed reduction tree, then a fused 16-token × 128-channel centering/padding/correction program. Head count only changes grid/addressing. Input stride is not specialized, including alignment specialization. No atomics, head-dependent autotuning, global precision flag changes, or global monkeypatches are used.

K's mean divides by the original live length. Q's mean divides by the padded length. Means and centered tensors round to BF16. Padded Q is minus its mean; padded K/V/correction are zero. The fixed correction can differ numerically from upstream cuBLAS despite retaining its BF16 output rounding. Candidate-versus-upstream differences are reported separately from partition identity; visual quality remains a later gate.

## Timing and memory

The primary speed comparison is the **actual public Runtime Sage3 callable** versus the lean fixed candidate, with unchanged upstream FP4 quantizers and attention. Three warmed CUDA-event samples alternate order. Preprocessing and complete attention have separate intervals; compilation, input creation, copies made by the probe, and metrics are excluded.

The instrumented replay, token-last path and fixed-means-only control are diagnostic timing rows. In particular, the fixed-means control computes a fixed correction before replacing it with upstream matmul; it is not an optimized means-only implementation. Do not use those extra-work controls as the public performance baseline.

The full case requires at least 28 GiB of free/reusable device memory and refuses insufficient headroom. That is a conservative estimate, not a measured peak. The request retains actual peak allocation. Only one full reference and one comparison record are held; head groups are compared sequentially without concatenating a full grouped output. No input dimensions are reduced after a failure.

## CPU checks

The tests execute the actual Triton kernels through its CPU interpreter with CPU tensors. They verify partition identity, original input preservation, padding, denominators, and BF16 rounding. Offline compilation builds SM120 code for both shapes without loading it or opening a CUDA context. These are CPU/source controls, not GPU qualification.

```sh
TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' python test_cpu.py -v
python compile_cpu.py --out /path/to/offline-compile.json
```

## Normal CLI qualification

The package is pinned to public Runtime 0.23.4, TensorFS 0.6.1, Torch 2.14.0 and Triton 3.8.0, with source/native hash gates in `cohort.json`. It initially admits SM120 only. Prepare the capture before renting:

```sh
cozy package lock
cozy run ./research/h3-sage3-fixed-preprocess/verify --describe
```

After the owner assigns an exclusive RTX 5090 lease, submit once through the ordinary default-home CLI:

```sh
cozy run ./research/h3-sage3-fixed-preprocess/verify \
  --input=/path/to/input.json --rental=assigned-machine \
  --idempotency-key=one-stable-key --out=/path/to/owned-output --json
```

Input: `{"seed":581204,"size":"all"}`. The owner must enforce an eight-minute actual invocation budget and cancel through the ordinary CLI if exceeded. Retain the original accepted identity; do not resubmit on uncertain observation. No new rental is authorized by this document.

A candidate advances only if actual intermediates, quantized payload/scales, final outputs and repeat controls are finite and partition-identical, and complete attention does not become slower or materially more memory hungry. Any subsequent full-video test requires a separate matched run and human quality review. No GPU result is claimed yet.
