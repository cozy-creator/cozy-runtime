# Upstream Sol-Attn integration

This adapter calls the upstream `sol_attn.sol_attn` package. It does not implement
a sparse kernel. The first supported interface is BF16, batch-one NHD, head
width128, unmasked noncausal inference on SM90. Any Ulysses degree that divides
the heads is supported. Ring attention, compiled execution, dropout,
GQA and LSE requests refuse explicitly.

A model opens `attention_scope(AttentionLayout(...))` around each DiT step. It
supplies the actual live document length, conditioning-prefix length, step index,
dense initial-step count and dense module prefixes. Nothing is inferred from
QKV values. Runtime installs exception-safe site hooks only for Sol selection.
A registered model can thus keep refiners and selected early layers dense.

Mirrored component calls carry this closed layout to each follower and restore
its context afterward. Diffusers' unequal-length Ulysses exchange assembles the
complete token sequence on each head shard before the same sparse/dense policy
runs. The prefix and trailing padding always describe the global document;
they are never reinterpreted as GPU-local token offsets. Output exchange restores
each GPU's original local sequence length. The upstream Sol kernel is unchanged.

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
requests. Real Gloo tests at degrees 2, 4, 7 and 8 cover unequal token shards and the
component-call layout transport. A separate native helper runs actual NCCL and
Sol, compares head partitions against the full-head call, and uses upstream
preparation plus a native all-keys control to prove genuine sparse work. This
primitive passed on two H100s; four-GPU CUDA and full H3 video quality remain
separate qualification steps. CPU passes and an imported package are not GPU
or quality qualification.
Preprocessing, required contiguous copies, metadata and dense-prefix work must be
included in H3 benchmarks. This backend stays an explicit approximate choice
until suitable full-video quality evidence is accepted.

## Eight-step H3 Turbo policy

The Turbo model sampling methods accept the request-local keyword
`sol_dense_steps`. For example, a Python workflow calls:

```python
base_model.sample_fl2va_turbo(
    state,
    turbo_lora=turbo_model,
    on_step=on_step,
    cancel=cancel,
    checks=checks,
    sol_dense_steps=4,
)
```

The Ref2VA method accepts the same keyword. This only sets the existing model
layout policy; select `sol-attn` separately through the normal attention override.
The Turbo default is 4: four dense evaluations followed by four with sparse video
attention. The user reviewed one matched 15-second fight scene and considered both
3/5 and 4/4 comparable in quality, preferring 4/4 as closer to the dense original.
That supports this Turbo policy choice, not a broad quality-equivalence claim or
an automatic switch to Sol. Explicit 3, 8 and 10 remain supported. Standard
30-step generation retains ten initial dense evaluations; warmup stays dense
independently. Values must be nonnegative integers, excluding booleans.

The count names initial transformer evaluations: eight evaluations use a
nine-point sigma grid, and evaluation indices are 0 through 7. A value of 3
keeps evaluations 1–3 dense and permits sparse video attention in 4–8; a value
of 4 keeps 1–4 dense and permits sparsity in 5–8. Every evaluation still protects
text, reference media, generated audio, the two refiner blocks and the first two
main transformer blocks. There is one DiT forward per evaluation, with no CFG
pass multiplier.

For H3's 50 main blocks and two refiner blocks, expected successful wrapper call
counts per GPU (excluding construction/warmup) are:

| Initial dense evaluations | sparse | dense_prefix | dense_step | dense_path |
|---|---:|---:|---:|---:|
| 3 | 240 | 240 | 144 | 32 |
| 4 (Turbo default) | 192 | 192 | 192 | 32 |
| 8 or 10 | 0 | 0 | 384 | 32 |

`dense_prefix` is additional work within each sparse call, not a second complete
DiT attention. Report each GPU separately when evaluating context parallelism.

This starts from the zero-based dense-step/layer gates in the
[qualified NVIDIA H3 integration](https://github.com/NVlabs/Sana/blob/8e0db4fa562d727ea28b8d63015c196db7d97cae/models/minimax_h3/GB200/sol_attn_h3.py).
The 3/4-step choices are our Turbo policies, not upstream quality claims. CPU
tests run both upstream H3 denoisers with reduced widths and the full layer count,
verify the policy reaches every attention call, preserve dense CPU output at zero
tolerance across policy values, and check warmup and failure cleanup. Additional
prompts and environment/GPU-count qualification remain separate evidence.

The ordinary migrated H3 workflow calls `sample_fl2va_turbo` or
`sample_ref2va_turbo` without a `sol_dense_steps` argument. Once it uses this SDK,
it inherits 4 directly; no package-specific sampler or policy branch is required.
Diagnostic helpers that explicitly pass 3/4/8/10 keep their chosen policy. A
workflow still defining package-local model classes must adopt the existing
Runtime model migration first; updating its SDK dependency alone does not replace
those classes.
