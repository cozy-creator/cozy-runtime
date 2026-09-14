# Prepared LoRA merge arithmetic

`internal/lora_merge.py` is an isolated preparation reference. No serving path,
worker, cache entry or default selects it. `merge_rowwise_fp8` receives canonical
A `[rank, in]`, B `[out, rank]`, independent strengths and optional alpha. It
computes decoded base plus each ordered `strength * (alpha / rank) * (B @ A)`
in FP32, then emits new E4M3 data and FP32 row-scale tiles. Missing alpha means
rank, so the default coefficient is the supplied strength.

Only the exact reviewed rowwise spec digests are accepted. Rank-one `[out]`
and keepdim `[out, 1]` scale shapes remain distinct. The native leaf interprets
the multiplier as `data.float() * scale.reshape(out, 1)`. A raw H3 scalar-scaled
header is not rowwise merely because its materialized leaf can be normalized
to that representation. The caller must perform a legitimate source-format
conversion before using this helper. Plain FP16/BF16 factors promote losslessly;
FP32 factors retain all bits. Encoded FP8 factor lanes require ordinary
TensorFS decoding before preparation and are not accepted as plain FP8 arrays.

The reference explicitly accumulates rank outer products in FP32, then applies
each adapter coefficient once. It does not normalize strengths, multiply A and
B independently by the same strength, reorder adapters or coalesce duplicates.
It changes no global TF32 settings. This rank loop is a correctness control,
not a claimed fastest implementation: compare it with highest-mode FP32 GEMM
on real H100 tensors before production selection. Cross-device byte identity
is not established by the CPU tests.

## Explicit scale policies

There is no default scale policy. `preserve` keeps the original quantization
grid and refuses if merged values exceed its E4M3 range; it never silently
clips overflow. `grow` preserves each positive original scale unless it must
increase to fit the merged values. `recalibrate` recomputes every row's grid.
Both preserve and grow require positive original scales. Output statistics
include how many row scales changed.

To isolate the codec from the LoRA arithmetic, the probe supplies zero-valued
factors with a nonzero coefficient. This deliberately takes the decode/encode
path; an actual zero-strength request still returns the original-part graft.
The independent control shows that recalibration alone can alter base weights,
while preserving the old grid keeps the tested zero-update codes unchanged.
A separate overflow control refuses preservation at 449 with original unit
scale, then proves grow-only expands enough to represent the value.

## Ownership and memory

The caller owns immutable input parts and a native TensorFS transaction. It
consumes tiles and commits only after the iterator succeeds. The helper neither
opens a database nor invents cache aliases. Existing native execution must key
the prepared result by exact base parts/encoding, ordered factor identities,
strengths/alpha, explicit scale policy and computation identity, including `ALGORITHM` and the captured
implementation. Mutable release/lane names resolve before that identity is made.
Keep compact LoRA checkpoints authoritative; cache the completed merged base
variant, not an expanded full-size delta for every adapter.

All-zero coefficients return `None` after shape, scalar and finite-factor/scale
validation, so the caller grafts original base parts with no re-encoding. The
unchanged base payload keeps its existing validity responsibility. During an
active merge, invalid decoded base rows or FP32 overflow can refuse iteration;
partial output must never become a committed checkpoint.

The scratch calculation reserves `4096 + 4 * max_rank * in_features` bytes plus
`20 * in_features + 4 * max_rank + 64` bytes per emitted row. It reduces row count
to fit the requested budget and refuses if even one row cannot fit. This covers
decoded/merged FP32 data, delta/product buffers, one promoted A and B tile,
absolute/reduction/mask intermediates, FP8 codes and FP64 error sums. It never
allocates a full decoded weight or full output weight. Caller-owned inputs,
retained output tiles and allocator bookkeeping/cache are separate. The GPU
allocation peak still needs measurement before integrating this estimate with
Runtime's admission model.

Output error statistics compare reconstructed encoded weights with the merged
FP32 target. They are not quality scores. A small error relative to total base
weights can still obscure a small LoRA update, so the GPU comparison must also
measure error relative to the update and the actual linear outputs.

## CPU evidence and remaining proof

Eight focused cases use independent dense algebra and an enumerated E4M3FN value
table: orientation/alpha/scales, mixed-factor strengths, FP32 preservation, row
tails and chunk byte invariance, ordered cancellation, zero/finiteness and
shape/spec/budget/overflow refusal. Input tensors remain unchanged. They run
in the existing CPU Torch CI job, alongside normal type/fence checks.

`research/lora-merge/cpu_real_factors.py` additionally reads verified native
factor objects from the imported spatial and Wushu LoRAs, preserving their FP16
and FP32 source bytes. It samples five output rows of the actual 7168-by-5376
projection, using a generated rowwise base, and compares the result with an
independent FP64 dense reference. This validates arithmetic with real factors;
it is not a full H3 model or quality proof. Reproduce with the task's retained
adapter headers and local TensorFS store:

```sh
uv run --extra derive python research/lora-merge/cpu_real_factors.py \
  --evidence /home/fidika/cozy_v2/outputs/h3-general-lora-20260914 \
  --store /home/fidika/.tensorfs --scale-policy preserve
```

On the generated-base fixture, preserving original scales reduces total-weight
relative L2 from 0.02715 to 0.004646, with error/update norm reducing from 5.50
to 0.94. Grow-only is identical because this fixture needs no growth. Its
original code maxima are only 2.0, 1.25, 0.40625, 1.25 and 2.0, so changing the
grid causes much of the larger error. These fixture results do not establish
which policy best preserves actual H3 outputs.

The GPU microbenchmark, cached-variant lifecycle and ordinary generated-video
comparison remain separate acceptance steps under tracker h3a-053. No image
or Runtime release is needed to review this arithmetic candidate.
