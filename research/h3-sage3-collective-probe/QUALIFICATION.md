# Fixed-means integration boundary

Run5247 used public Runtime0.23.5/TensorFS0.6.1 and two RTX5090 GPUs through the ordinary
Cozy CLI. Its report SHA256 is
`fece57aaa0a6958e711d7403ccd8393bf8ff82c4bc43c05d0d8ce36931b9a1bf`.
The research-local fixed-means source is byte-identical to the proposed serving module
`internal/attention_sage3_fixed.py` (SHA256 `251bf1bb87d24275d0c01dcc63a3d76352c550436c6594cc72f830db3ff1c8d8`).
These hashes are evidence identities, not serving admission requirements.

## What passed

- Actual two-rank NCCL Ulysses at B1/N109225/H56/D128, with unequal token shards54613/54612
  and28heads per rank: original and fixed kernels both matched their full-head reference
  exactly, including intermediate tensors, output, repeat, and input-immutability checks.
- Separate two-pass sensitivity control on the same two GPUs, using14local heads per call:
  original centering differed in88Q elements,10correction elements and41output elements;
  fixed means produced zero differences at every recorded stage. This is not four-rank
  transport qualification. Transport and input checks passed before interpreting arithmetic.
- Maximum allocation19.421GiB per GPU. Three warm max-rank wall-time samples averaged
  343.060ms original versus339.844ms fixed for actual two-rank exchange plus attention.
  Full-head reference averaged395.160ms versus387.033ms. No regression was observed in
  these samples; this is not a video speed claim or statistical performance guarantee.

The full packed-input fixed output differs from the original full output in171of782924800
BF16 values, maximum absolute delta0.00006103515625, relative L2 error0.0000014063171102.
The same full-reference measurement was replicated on each rank and is counted once.
Unlike the earlier contiguous fixture, packed-input output is not byte-identical to the
original algorithm. This numerical result is not a perceptual quality approval.

## Minimal serving integration

The draft already exposes explicit `sageattention3-fp4-fixed-means` and
`kitchen-sol-producer-fp4-fixed-means-shared-qkv` selectors. The dense selector is head-local;
the existing producer companion remains single-GPU until the separately reviewed CP
producer implementation extends that explicit recipe. Add no additional fixed-means
materialized preset merely to duplicate that integration. Approved H3 defaults, dense/sparse
schedules, protected rows, and model weights remain unchanged.

The kernel accepts B1/BF16/D128 equal noncausal Q/K/V on SM120. It uses fixed per-head
token reduction trees, BF16 mean/centering boundaries, live-K versus padded-Q denominators,
zero K/V padding, and int64 source offsets for full packed inputs. It preserves the upstream
BF16 matmul correction and the public FP4 quantizers/attention. No global math settings,
source-hash gate, or package-version gate are installed in serving code.

Remaining gates before any default proposal:

1. Integrate the explicit CP producer recipe on the current source base without changing
   existing automatic eligibility; run its real CPU policy/collective checks.
2. Qualify the combined opt-in through normal CLI, preserving the actual selector and
   rank receipts. The dense-only research copy does not prove the full selector/processor
   combination or an entire H3 denoising pipeline.
3. Localize any remaining pipeline differences before making broader determinism claims.
   The retained upstream BF16 correction uses cuBLAS, so measured14/28/56head equality is
   not a mathematical guarantee for every shape/device/library. No full-video matrix is
   authorized by this bounded correctness study.

The package return fix in version0.0.2 only carries the report as an owned CPU uint8 tensor
through Diffusers' actual CP output hook. Run5245 failed at that hook and supplies no
numerical pass evidence. Run5247 successfully returned the report.
