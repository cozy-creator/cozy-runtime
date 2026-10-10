# Opt-in H3 CP producer integration — preparation

The raw Kitchen producer fixture (run5249, public Runtime0.23.5/TensorFS0.6.1, twoRTX5090s) produced exact full56/head28 outputs, bootstrap/previous-step statistics, transport and repeats. It did not execute H3 projections, module hooks or the real model's split rotary presentation.

This isolated task will compose CP support into the existing explicit fixed-mean producer recipe from PR1290. Current published attention defaults and single-GPU producer behavior stay unchanged.

Required boundaries: keep full H3 rotary tables per request and component BEFORE the Diffusers CP split hook, without a per-block all-gather; project raw local QKV through the original modules and hooks; use actual Ulysses to obtain global rows/local heads; invoke existing Kitchen sol_attn_chunked with global4096-row chunks, original RMS/RoPE/prefix/margin and per-rank prior statistics; inverse output then original projection. Clear request state and preserve replay input statistics.

CPU hook-order/group-lifecycle tests and a bounded ordinary-CLI integration probe must cover the real projection boundary, including encoded operands and LoRA. Raw-QKV identity alone is not a full-model or speed qualification. No defaults are switched by this preparation, and no GPU request is authorized here.
