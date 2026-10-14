Runtime 0.23.36 bundles TensorD 0.5.29 (TensorFS native v0.8.0).

- Packages bring their own kernels, execution plans and attention producers (#1371, `docs/package-extensions.md`): `cozy_runtime.author.kernels` (`TritonKernels`/`TritonEntry` compiled per sealed GPU before `Plan.load` returns, `PreQuantized`, `readable`, plane-safe `rms_rope_split_half`), `Model.execution_plan(plan)` (enters the constructed digest, stage memo and prepare facts), `AttentionProducer` + `register_attention`, and the generic `kitchen-sol-fp4-fixed-means` backend.
- The Runtime's own H3 glue, producer and `fusion=` consent are unchanged: minimax-h3 ≤1.35.x and anima 0.5.0 run as on 0.23.35. minimax-h3 1.36.0 and anima 0.5.1 require this release.
- TensorD 0.5.29: a waiting root's blocker ignores its own claim, so it logs one `machine.gpu.wait` entry per blocker change and its wait line keeps naming the run it waits behind.
