# DeepSeek Ultrafast lessons for Cozy Creator

Research date: 2026-11-15. Status: source investigation in progress; no new GPU measurements.

Owner: Codex `/root`. Purpose: evaluate model/hardware specialization, fused CUDA kernels, CUDA graphs, and transferable H3 improvements. Branch: `codex/deepseek-h3-kernel-research-20261115`. Runtime base: `af2a4753cb70598b75405942901e4f8f3b24eda6`. Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/deepseek-h3-kernel-research-20261115`.

Source snapshots:

- [DeepSeek Ultrafast](https://github.com/basetenlabs/deepseek-ultrafast/tree/76766a97308bfd1658fb055e835d0f92de737dc3)
- Runtime `af2a4753cb70598b75405942901e4f8f3b24eda6`
- Packages `e4b179bbd099f681658dc7ef5a059f484969cf94`
- Creator `b7fb810bdfdfb5a9f8a0d55d3d0c13c42fea4cc9`
- Tracker `4666c6808461d1463b1e191fa62fa9117a5b14b8`

## Questions

1. Is the upstream implementation a whole-model megakernel or a graph of specialized kernels?
2. Which advantages follow from the algorithm, fusion, layouts, GPU instructions, and host language?
3. Which techniques apply to H3 diffusion, long video sequences, and memory-constrained cards?
4. What does Cozy already implement, and which missing pieces merit controlled measurements?

## Initial evidence

The upstream [server engine](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/crates/server/src/engine.rs) captures and replays a speculative decoding graph. The [kernel build](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/crates/kernels/build.rs) compiles multiple CUDA translation units to target-specific cubins. These establish specialization and graph replay, not an end-to-end single-kernel design or a measured advantage over GPU DSLs.

The completed note will separate observed code, published measurements, hypotheses, and proposed experiments. It will preserve current model quality, numerical settings, GPU degree, and the full 15-second H3 workload when defining performance comparisons.
