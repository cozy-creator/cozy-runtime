# Varena and TensorFS without PyTorch

Checked 2026-11-15. Source: Creator RC `5e80a4ba05ea5b966f9f0b6d29b578e7a394f477`. Binary inspection and the native smoke test used the existing **TensorFS 0.25.0** installation in the H3 QKV task's `.venv-typecheck`; this was not a rebuild of the current RC's 0.25.1 core.

## Dependency boundary

| Component | PyTorch dependency | Evidence |
|---|---|---|
| Rust Varena core / `libvarena.so` | None in the inspected native dependency declarations; talks to CUDA driver directly | [Cargo manifest](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/varena/Cargo.toml), [CUDA loading](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/varena/src/cuda.rs#L141) |
| Native Rust TensorFS | No Torch dependency; `cuda` enables Varena | [Cargo manifest](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/tensorfs/Cargo.toml) |
| TensorFS Python extension `_ext` | Python/PyO3 adapter, distinct from the Torch shim | [Manifest](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/tensorfs-py/Cargo.toml) |
| `_varena_torch-<version>.so` | Explicitly links `torch_cuda`, `c10_cuda`, `c10` and CUDA runtime | [Build flags](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/tensorfs-py/shim/build.py#L36-L43) |
| `tensorfs.varena.torch` | Imports Torch and loads the shim for that exact Torch version | [Allocator binding](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/tensorfs-py/python/tensorfs/varena/torch.py#L121-L168) |

Thus a native Rust model executor can bypass both Python adapters and the Torch shim. It need not replace Varena's memory core. The [native ABI](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/varena/include/varena.h#L108-L124) already exposes region use/pointers, stream-ordered working allocations and completion. Kernel launches remain the caller's responsibility.

Varena intentionally builds as a **shared Rust dylib**, not an rlib. Its manifest says every embedding must share one core to avoid duplicate process registries. “No PyTorch” therefore does not imply “no shared library.” TensorFS and a native H3 executor should resolve the same Varena core.

## Binary and execution checks

`readelf -d` on the installed `libvarena.so` lists Rust `libstd`, `libdl`, `libgcc_s`, pthread, libc and the loader; it lists no `libtorch` or `libc10`. The separate Torch shim lists `libtorch_cuda.so`, `libc10_cuda.so`, `libc10.so` and `libcudart.so.13`. The Python `_ext` links Varena but not Torch directly.

A tiny standalone C program loaded `libvarena.so`, called `vr_v1_abi_version`, and inspected its own process mappings. Result:

```text
abi_version=1 torch_libraries=0 python_libraries=0
```

The existing [native C/CUDA example](https://github.com/cozy-creator/cozy-creator/blob/5e80a4ba05ea5b966f9f0b6d29b578e7a394f477/crates/varena/examples/native_engine.c) was then compiled unchanged against that core and the inspected header. On the local RTX 4070 Laptop, it opened a Varena GPU/weight region, entered phases, acquired a Use, allocated scratch, launched a CUDA kernel on an explicit stream, checked completion, and released ownership. Its two passes returned:

```json
{"native_engine":"pass","passes":2,"values":[2,4,6,8]}
```

The native executable links only Varena and libc directly, loading the CUDA driver itself. No Python or Torch invocation occurs in this example. No compute process remained after it exited. Evidence is retained under `~/cozy_v2/outputs/varena-native-audit-20261115/`: `receipt.json`, `native-gpu-receipt.json`, the unmodified example/header, and the small loader probe.

This qualifies the dependency separation and a tiny native CUDA allocation/use path. It does not qualify a new H3 engine, all Varena behavior, graph replay, streaming pressure, or the current RC native binary.

## Consequences for a native H3 implementation

The native executor must explicitly perform responsibilities currently supplied by Torch and its adapter: track tensor shapes/strides/dtypes, allocate outputs and scratch, retain weight Uses through consumers, order streams, prepare/replay graph memory, and retire resources after completion. Raw pointers alone do not retain residency. Rust RAII can wrap the native handles, but its Drop behavior must preserve asynchronous completion semantics.

Baseten's GPU code is **CUDA C++ with inline PTX**, compiled by `nvcc` into cubins; its Rust code is the host orchestration. [Actual build](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/crates/kernels/build.rs#L37-L56). A Triton/CuTe-generated cubin can be deployed through the same general native execution pattern; source language and serving dependencies are separate decisions.

For Cozy, the proposed computational path can be Rust H3 → native TensorFS/Varena → CUDA driver and compiled kernels, with no Python Runtime SDK, Torch or libtorch in that path. A native job adapter still has to honor Cozy's inputs, cancellation, progress and outputs. This proposal does not claim the complete H3 port or that adapter already exists. Using Rust `tch`/LibTorch bindings alone would retain PyTorch and would not meet this specific goal.
