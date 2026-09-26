# H3 Turbo bake-once capability boundary

H3 Turbo remains a base AdaLN-pruned DiT plus a PDD-8 adapter. The default path is still
the specialized overlay: `LoRAFactors.accumulate` runs its released BF16 down/up GEMMs for
each matching forward. This change adds an explicit opt-in `bake_once` contract before the
request state is touched.

`OfficialH3Pipeline.turbo_composer_capability(trunk)` inspects the filled DiT. Encoded
providers replace projections with role-buffer modules that have no floating `.weight`; the
capability reports their paths and returns `turbo_bake_unsupported` rather than silently
dequantizing or mutating FP8 storage. An all-floating synthetic root is also refused until
the complete graph composer covers PDD AdaLN tables, evaluation-specific output heads, and
an isolated root per adapter set. The existing specialized path is unchanged when
`bake_once=False`.

The model method accepts `bake_once=True` so a caller can request a strict no-fallback bake.
It calls the capability before overlay selection or `attention_kwargs` mutation. A refusal
therefore cannot leave request state armed or accidentally fall through to per-step adapter
GEMMs. The current FP8/PDD-8 lane intentionally remains overlay-only.

Completing the composer requires provider-owned dequantize/compose/requantize semantics,
numerical equivalence across CP tail shapes, static table/head selection, and request-root
cache isolation. No GPU, quality, or release qualification is claimed here.
