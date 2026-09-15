# TensorFS Torch and PEFT qualification

The current integration combines Runtime #523/#528 with master `a7976db2`.
It consumes public TensorFS 0.3.44 and the exact wire57 adapter contract from
worker-protocol `b6e56391`; no storage schema or additional wire fields changed.

Stored rowwise FP8 uses `tensorfs.from_parts` and an ordinary `nn.Linear.weight`
Parameter. Codes and scales remain the physical storage; Runtime retains the
activation quantizer and separate bias addition. When TorchAO is installed, the
provider registry registers the standard TorchAO candidate instead of the legacy
module leaf. Its real native oracle must pass on the current CUDA SM89+ device.
A stored FP8 checkpoint alone never establishes a native execution route.

Before PEFT wraps a component, Runtime retains its original base parameter slots.
This now covers plain BF16 components as well as encoded ones, so eviction and
restaging preserve request-owned factors and canonical checkpoint names. Adapter
components using block paging refuse before wrapping; whole-component staged
residency is qualified. Physical storage accounting uses `physical_tensors`.

## Local evidence

The owned environment uses Torch 2.13.0+cu130, TensorFS 0.3.44, PyPI TorchAO 0.18.0,
PEFT 0.20.0, Accelerate 1.15.0, Transformers 5.16.1 and Diffusers 0.40.0. TensorFS
0.3.44 intentionally refuses TorchAO `0.18.0+cu130`; do not substitute that wheel.
No shared CLI, daemon, rental or worker image was changed.

On an RTX 4070 Laptop (SM89):

- 61 tensor, PEFT, fusion, custody and binding tests pass, including the actual
  CUDA projection equivalence test.
- Eight streamed/native-residency tests pass. Real TensorFS stores fill plain
  BF16 and both rowwise scale layouts; active PEFT survives three evict/restage
  cycles with identical outputs, factor identity and physical base accounting.
- Active PEFT dispatch records `aten::_scaled_mm` with two E4M3 operands. CUDA
  traces contain the SM89 CUTLASS E4M3 tensor-core base GEMM and BF16 adapter GEMMs.
- The existing two-component native lifecycle proof passes with standard weights,
  repeated identical allocation, admission refusal and poisoned-generation checks.
  Its measured staging peak equals the admitted 3,153,920 bytes.
- Full mypy (481 source files), Ruff (486 files), and all 20 architecture fences
  pass. Focused authored-adapter, custody and registry checks also pass; GPU-only
  warm checks skip in the separate base-only checking environment.

Logs, dependency inventory, source identity and native profiler traces are retained
in `outputs/h3a080-peft-finish-20260915` in the workspace.

## Full H3 gate

These small native projections and streamed components do not establish H3 video
quality, memory at full sequence geometry or H100 behavior. Before promotion,
qualify the exact reviewed image cohort on H100, then use ordinary `cozy` with
default `~/.cozy` for the prepared 15-second fight comparison: baseline, Spatial
0.3, and Wushu 0.8; seed 7381 and 30 steps stay fixed. Existing FP16 Spatial and
FP32 Wushu storage lanes require an explicit preparation cast to BF16 factors.

Require actual `encoded_gemm`/`torchao.fp8-rowwise.native/1` for every eligible
base projection and native FP8 dispatch evidence while adapters are active.
Keep the selected BF16 Sol attention policy identical across comparison arms.
Refuse unexplained floating fallback; inspect completed video and audio, exact
adapter echo/call counts, timing, peak memory and spend before calling it qualified.
