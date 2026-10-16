Runtime 0.25.0 is the xs-040 cohort: one request kind, weights fetched as they are needed, and TensorD 0.6.0 on TensorFS 0.10.0.

- One registration (xs-040): `app.job`, the job services and pause/resume are gone. Every function is an `app.entrypoint`; a writer declares `weights=` and reads its sources as `Weights[M]`; `accelerator=False` runs without a GPU. Interfaces have one `entrypoints` list. Packages published on older Runtimes must be republished.
- The fetch planner asks TensorFS for each component's objects in the order its stages use them, from Start on, so imports, executor boot and construction run while weights download; unused components are never fetched. On-demand slots (`app.entrypoint(on_demand=...)`) fetch at first use.
- Graphs by default: placed components record their calls once and replay them; LM decode runs as replayed CUDA graphs, with room checks and a Transformers fallback on a tight pool.
- Host work per op and per block near zero (SDXL −26% on H100); spilled tensors keep 256-byte alignment.
- Attention: Sol on B200/B300 with FA4's block-sparse FP8 forward; an int8 model may take kitchen-int8 attention like its projections (sm86/89/120/121); the native fp8-rowwise op.
- TensorD 0.6.0: one execution path for every request, a memoized consumer keys checkpoints by manifest, the datacenter model cache. TensorFS 0.10.0: the Store's one Fetcher and the volume tier.
