# MiniMax H3 Turbo bake-once design checkpoint

## Current boundary

The released H3 Turbo lane is a base DiT plus a separate PDD-8 adapter. The runtime
installs permanent consumers while constructing an AdaLN-pruned base, then arms the
selected overlay from each request's `attention_kwargs`. Each matching linear forward
currently executes the adapter's down and up BF16 projections; this is the steady-state
cost we want to remove. Turbo tables and output heads are also selected per evaluation,
so merging the six linear families alone does not produce a Turbo model.

The base lane is normally FP8-pruned. An in-place `weight += up @ down` would either
silently replace encoded storage with a floating tensor or require an unreviewed
dequantize/requantize path. The runtime therefore must refuse an encoded/custom linear
until its provider publishes that capability. There is no production bake-once claim in
this checkpoint.

## Prototype in this branch

`bake_lora_linear` in `src/cozy_runtime/models/minimax_h3/turbo.py` composes one adapter
site into a new ordinary floating-point `torch.nn.Linear`. It computes the delta once in
FP32, casts to the original floating dtype, copies the bias, and retains neither source
module nor adapter factors. It refuses non-`nn.Linear` modules and encoded dtypes. The
focused CPU tests prove that the resulting forward does not call
`LoRAFactors.accumulate`, that the source remains byte-identical, and that two requests'
composed modules do not share mutable parameters.

This prototype intentionally does not attach to `OfficialH3Pipeline`, remove existing
hooks, merge AdaLN tables, or replace the selected output heads. Its arithmetic is not
yet a bit-exact proof of the current per-forward BF16 rounding contract.

## Proposed production contract

1. Preparation constructs an immutable base artifact and a composer capability record.
   The record names ordinary compatibility metadata (base lane, adapter family/rank,
   storage/compute dtype, table/head schedule, and composer version), never an exact
   model fingerprint as the compatibility mechanism.
2. A composer either returns a private prepared Turbo root or returns a typed refusal. It
   must not mutate the cached pristine base. A cached default Turbo root is safe to reuse
   only when it is immutable and the request selects that exact prepared capability.
3. Arbitrary adapter sets are composed into separate roots. A request selector chooses a
   root; it never arms mutable hooks or changes another request's root.
4. The composer must cover all six linear families plus the already producer-tabled
   `adaln_proj.linear` family and the evaluation-specific video/audio heads. A partial
   merge is refused rather than silently falling back to per-step hooks.
5. FP8 support requires an explicit provider operation that dequantizes, applies the
   delta with the released rounding policy, requantizes, and reports the resulting storage
   metadata. Until then, FP8 H3 Turbo remains the existing overlay path.
6. Qualification compares composed and overlay outputs across every Turbo evaluation,
   CP partition/tail shape, base-only request after Turbo, Turbo request after base-only,
   two concurrent request roots, and process restart. It must measure zero adapter GEMMs
   after preparation and preserve model-quality/audio gates.

## Tracker issue draft

Filed as [tracker #243](https://github.com/cozy-creator/tracker/issues/243); the draft below
is kept beside the source prototype so the design boundary remains reviewable with the PR.

**Title:** H3 Turbo: add capability-gated bake-once composer with request-isolated roots

**Problem:** The current PDD-8 overlay installs permanent hooks and performs two adapter
GEMMs for every matching linear on every denoise evaluation. The base lane is FP8-pruned,
and Turbo also substitutes AdaLN tables and output heads, so a generic in-place LoRA merge
would corrupt encoded storage or leave a partially dynamic graph.

**Deliverable:** Add a preparation-time composer capability for a complete H3 Turbo root.
The composer must produce an immutable private root, leave the pristine base untouched,
and return a typed refusal for unsupported storage/providers. The default Turbo root may
be cached and reused; arbitrary adapter sets must receive isolated roots. Compatibility is
expressed through ordinary model/composer metadata, not exact SHA/fingerprint gates.

**Acceptance evidence:**

- no LoRA hook or adapter GEMM occurs after preparation;
- all six linear families, tabled AdaLN family, and both output heads match the overlay
  path at every scheduled evaluation and CP tail shape within the reviewed tolerance;
- base-only request remains pristine after Turbo request A;
- Turbo requests A and B cannot mutate or observe each other's roots, including concurrent
  execution and cache reuse;
- encoded FP8/custom modules refuse until provider dequantize/requantize metadata and
  rounding are available;
- ordinary H3 CLI, media, audio, and release gates remain separate and are not claimed by
  this source-only prototype.

**Dependencies:** Provider-owned FP8 composition capability; producer metadata for tables,
heads, and schedule; generic overlay parser/compatibility metadata work (Creator #718).
