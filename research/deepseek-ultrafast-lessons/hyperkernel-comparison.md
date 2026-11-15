# Hyperkernel, Baseten, and a native Cozy execution path

Source inspection, 2026-11-15. Hyperkernel snapshot: [`4076a822d2a99ea91554920aa3a45742eb7de4fd`](https://github.com/Hyperstitioncc/Hyperkernel/tree/4076a822d2a99ea91554920aa3a45742eb7de4fd). Baseten snapshot: [`76766a97308bfd1658fb055e835d0f92de737dc3`](https://github.com/basetenlabs/deepseek-ultrafast/tree/76766a97308bfd1658fb055e835d0f92de737dc3). No GPU execution of either engine was performed.

## What changes in the execution model

The [Hyperstition post](https://x.com/hyperstition_cc/status/2118432349352595528) describes a generator for model-specific megakernels. Its public repository is a Rust host, `hyperserve`, plus compiled B300 kernel bundles. The source repository has common loading, memory, GPU, scheduling, caching and API modules: [library modules](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/src/lib.rs), [bundle contract](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/src/bundle.rs), [dependencies](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/Cargo.toml).

For its LLM path, the Rust host [schedules request/cache commands](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/src/engine/mod.rs#L148-L159) and [pushes them to a GPU-visible ring](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/src/gpu/feed.rs#L132-L154). Its common [loader](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/src/gpu/load.rs#L446-L495) launches compiled entries with cooperative/cluster/dependent options. This is a resident-program interface, rather than a Rust host dispatching every tensor operator. The loader also supports multiple entries and auxiliary launches, so “one kernel launch” should be read for the specific model/program being described, not every action of the whole application.

Baseten's [model code](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/crates/engine/src/model.rs#L3956-L4044) captures a graph containing drafting, verification and acceptance, and its [server loop](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/crates/server/src/engine.rs#L333) replays that graph. It uses multiple specialized kernels with selective fusion and PDL. A graph submission containing many kernels and one persistent kernel executing an internal schedule are different mechanisms.

| Aspect | Baseten Ultrafast | Hyperkernel public release |
|---|---|---|
| Host | Rust engine and server | Rust shared serving host with model-specific modules |
| GPU execution | Captured multi-kernel pipeline, selected fusion/persistence | Precompiled megakernel bundles; LLM host communicates through command/result rings |
| Editable GPU implementation | CUDA C++/PTX source included | Cubins and SASS listings; generator and editable kernel sources are not in the inspected release |
| Current target | DeepSeek, SM100/SM103 and supported TP configurations | B300 bundles for the listed models, shapes and concurrency presets |
| Shared software | Workspace crates plus external dependencies | Shared Rust library/modules, generic bundle interface, CUDA driver binding and external crates |

The release's [notices](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/THIRD_PARTY_NOTICES.md) explicitly describe the bundles as binaries compiled from its own sources. That does not disclose the generator's implementation language or prove it avoids DSLs. The [README](https://github.com/Hyperstitioncc/Hyperkernel/blob/4076a822d2a99ea91554920aa3a45742eb7de4fd/README.md) names B300 requirements and limits such as precompiled protein lengths; it also uses Python/PyTorch for some checkpoint-conversion steps. Runtime independence is separate from authoring and conversion independence.

## Interpretation for Cozy

There is a credible direction toward a specialized native serving artifact whose repeated computation no longer needs Python or PyTorch dispatch. Source inspection supports that possibility; it does not establish that every model/workload should become a separate hand-maintained runtime.

Three decisions are independent:

1. **Authoring language:** model definitions and kernels can be written or generated in Python, a GPU DSL, CUDA C++, or another frontend.
2. **Deployment artifact:** the output can be precompiled GPU binaries plus native host code and tensor/layout metadata. TensorFS already compiles [Triton into cubins embedded in Rust](https://github.com/cozy-creator/cozy-creator/blob/b7fb810bdfdfb5a9f8a0d55d3d0c13c42fea4cc9/crates/tensorfs-py/kernels/build.py). No serving-time DSL interpreter follows from DSL source.
3. **Execution schedule:** choose ordinary kernels, graph replay, or a persistent region/program according to workload and measured benefit. Native deployment does not require a whole-model megakernel.

A future Cozy native model executor could therefore load model-specific compiled stages while reusing TensorFS for weight bytes, Varena for memory and lifetime, and the shared execution system for cancellation, outputs and scheduling. Model-specific layout/scheduling belongs in its package/artifact; common infrastructure remains shared even if statically linked into an executable. This is an architectural option to evaluate, not a newly authorized rewrite or a claim that the current package interfaces already admit arbitrary resident programs.

For H3, one persistent whole-model kernel must still handle large activations, changing layouts, precision semantics, multi-GPU exchanges and operation on cards where weights/working memory do not all fit. One launch does not imply all model data stays on chip. A repeated block or stage is a reasonable first scope if a measured baseline reveals meaningful launch, synchronization or intermediate-memory cost. Keep model output and memory conditions fixed while comparing it with the current fused-kernel/graph implementation.

The performance claims in the post use several different models and baselines. There is no controlled Hyperkernel-versus-Baseten comparison in the material inspected. In particular, Hyperkernel's published-vLLM comparisons cannot be divided into Baseten's Tetris rates to establish a winner.

The user subsequently pointed to [the announcement's chart](https://x.com/hyperstition_cc/status/2118432349352595528/photo/1), which was downloaded from X's image CDN and visually inspected. It **does supply absolute rates**: DeepSeek V4.1 Flash, AgentX, TP4 on 4×B300, tokens/s/user labeled P90. Hyperkernel: 1,129.6 (one user), 1,059.6 (two), 948.4 (four), 734.5 (eight). Its NVIDIA/vLLM comparisons mix the Oct 20 specialized path (826.5 at one user, 571.4 at eight) with Oct 2 standard decode (497.5 at four); the two-user Hyperkernel bar compares with one-user vLLM.

Baseten's 1,740 on 4×GB300 divided by 1,129.6 is 1.540×; 2,060 on 8×B300 divided by 1,129.6 is 1.824×. These are arithmetic ratios between published headlines, not controlled speedups: GPU configuration, AgentX versus the coding demo, and percentile/timing aggregation differ. The earlier inability to locate an absolute Hyperkernel rate was a missed chart, not absence of a published figure. Baseten's README does not supply a matched concurrency curve.
