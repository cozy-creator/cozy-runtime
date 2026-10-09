# Restore the approved H3 fusion and LoRA bake paths

The approved RTX 5090 attention matrix used Runtime wheel
`4460767713804fb09722222af816d8663862e6f51c6211d65d33472db8d2c03a`.
Its timings include the fusion compatibility fix and LoRA bake optimizations from
PR1250 and PR1248. Runtime0.21.6 still lacks both changes; promoting attention alone
would not reproduce that implementation.

This port copies exactly two production modules from that approved wheel:

- `fusion_install.py` replaces source-text hashes with the callable and module
  topology contract. Diffusers0.41's harmless SwiGLU annotation no longer disables
  fusion. Unsupported semantics, including skipped hooks and instance forwards,
  are refused before any substitution. Applied/fallback state remains visible.
- `lora_bake.py` uses native E4M3 decoding with the existing signed saturation
  policy for reserved codes, reuses the FP32 down factor and encoding table within
  each bake, and skips diagnostic arithmetic when no statistics are requested.
  It retains no new tensors between bakes.

The fusion kernels themselves are unchanged. The experimental alternative FP8
GEMM, attention selection, residency planner, dependency declarations, and public
version are outside this port.

## Qualification

CPU checks cover actual Diffusers0.40/0.41 contracts, semantic refusal before
mutation, all256 E4M3 codes including signed zero/reserved values, byte-equivalent
bakes with and without statistics, and per-call temporary lifetimes. Current
weight-plane integration must use TensorFS0.6, not the old benchmark test
environment's TensorFS0.4.

The original ordinary-CLI GPU gates5011 and5012 and full video matrix establish
the earlier0.19 implementation. They are provenance, not execution qualification
for a new0.21 wheel. Before promotion, run the updated fusion fixture's BF16,
encoded, inactive-hook, baked-hook, and unbaked-hook cases on the new candidate;
repeat the LoRA gate's all-code and byte-equivalence checks. Then use the current
H3 package through the ordinary CLI without an attention override for matched
15-second Turbo and regular requests. Keep the original prompts, seeds, dimensions,
steps, base weights, and PDD adapter.

The parent benchmark owner controls GPU runs and deployment. This branch does not
update a live machine or change the H3 default by itself.
