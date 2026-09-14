# Upstream Sol-Attn integration

This adapter calls the upstream `sol_attn.sol_attn` package. It does not implement
a sparse kernel. The first supported interface is BF16, batch-one NHD, head
width128, unmasked noncausal inference on SM90. Context parallelism, compiled
execution, dropout, GQA and LSE requests refuse explicitly.

A model opens `attention_scope(AttentionLayout(...))` around each DiT step. It
supplies the actual live document length, conditioning-prefix length, step index,
dense initial-step count and dense module prefixes. Nothing is inferred from
QKV values. Runtime installs exception-safe site hooks only for Sol selection.
A registered model can thus keep refiners and selected early layers dense.

The v1 sparse call uses tau1.0, exact threshold statistics, one KV split and a
prefix sink. The protected query prefix is recomputed with a resolved dense
reference. Explicit trailing padding is excluded and receives zero output.
`exact` describes threshold calculation, not exact full attention.

The selector binds the dense callable and records its identity before warmup;
no per-forward kernel search or global backend switch occurs. Missing model
layout or site metadata is a refusal. H3's builtin adapter supplies the NVIDIA
policy through this explicit scope; a generic request cannot invent an unknown
conditioning prefix.

CPU tests cover immutable/nested scopes, real module-hook cleanup, genuine dense
attention with poisoned trailing padding, refiner document isolation and invalid
requests. The separate native CUDA test checks protected-query outputs and input
bytes. CPU passes and an imported package are not GPU or quality qualification.
Preprocessing, required contiguous copies, metadata and dense-prefix work must be
included in H3 benchmarks. This backend stays an explicit approximate choice
until suitable full-video quality evidence is accepted.
