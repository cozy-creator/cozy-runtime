# Opt-in H3 CP producer integration — preparation

The raw Kitchen producer fixture (run5249, public Runtime0.23.5/TensorFS0.6.1, twoRTX5090s) produced exact full56/head28 outputs, bootstrap/previous-step statistics, transport and repeats. It did not execute H3 projections, module hooks or the real model's split rotary presentation.

This isolated task will compose CP support into the existing explicit fixed-mean producer recipe from PR1290. Current published attention defaults and single-GPU producer behavior stay unchanged.

Required boundaries: keep full H3 rotary tables per request and component BEFORE the Diffusers CP split hook, without a per-block all-gather; project raw local QKV through the original modules and hooks; use actual Ulysses to obtain global rows/local heads; invoke existing Kitchen sol_attn_chunked with global4096-row chunks, original RMS/RoPE/prefix/margin and per-rank prior statistics; inverse output then original projection. Clear request state and preserve replay input statistics.

CPU hook-order/group-lifecycle tests and a bounded ordinary-CLI integration probe must cover the real projection boundary, including encoded operands and LoRA. Raw-QKV identity alone is not a full-model or speed qualification. No defaults are switched by this preparation, and no GPU request is authorized here.

The implemented opt-in name is `kitchen-sol-producer-fp4-fixed-means-shared-qkv`; no additional selector is introduced. Its single-GPU and multi-GPU paths use the same existing Kitchen producer. The published default producer recipes remain context-parallel-ineligible and the H3 preference lists are unchanged.

Full rotary ownership uses an opaque component token and a Diffusers ModelHook registered before the CP split hook. It retains references only during an active sparse opt-in call; ordinary PyTorch forward hooks would see already-split output and are not used. The real two-process Gloo check verifies full129-row tables against local65/64-row views, same storage, two components, nesting and next-request cleanup.

Follower state is owned by the existing Attention selection hold for the whole attempt, selected from actual held recipes even with an empty CLI pin. Each mirrored RunCommand borrows that state. Restore, new selection, refusal and matching-construction unload clear it. The new recipe retains the original per-site statistics inputs for same-step retries; this behavior is confined to the opt-in recipe and does not silently merge the older public producer replay proposal.

Validation includes actual follower RunCommands at steps4/5, later-call OOM replay, fresh next-attempt step6, actual RankGroup/follower-process restoration after body failure/cancellation, and source typing. Local projection assembly is `_project_qkv`: original modules/encoded operands/LoRA hooks, bounded chunks copied into one local packed buffer, then Q/K/V views for exchange. The separate GPU projection and lifecycle fixtures remain required before a serving or performance qualification.
