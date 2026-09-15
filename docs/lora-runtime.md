# Generic model-slot LoRA execution

The serving wire binds an ordered adapter stack to a model slot: base component,
ordinary checkpoint/component, and finite decimal strength. Creator uses ordinary
checkpoint references; authored calls use the existing retained-artifact
`with_adapters` wrapper. Model packages do not implement LoRA loading or math.
The source converter owns ecosystem key normalization.

Canonical tensors are `<module>.lora_A.weight` `[rank,input]`,
`<module>.lora_B.weight` `[output,rank]`, and optional scalar `<module>.alpha`
(default rank). Runtime validates every tensor against the selected base snapshot
and target geometry, reads verified TensorFS data, checks finite values and owns
independent request copies. Zero strength validates the source but allocates no
device factors. Nonempty configs/assets refuse until their semantics are supported.

Native PEFT owns projection, scaling and accumulation. Supported targets are
ordinary Linear modules, including real TorchAO quantized weights. Factors use
the base's logical floating dtype; a one-time preparation cast reports degraded
mode. Prepared BF16 lanes avoid that cast. Base weights remain immutable; no
rounded merge/unmerge is used. Construction identity includes target geometry and
implementation identity, while ordered checkpoints and strengths identify requests.

Header inspection prices factor residency and source/validation memory. Actual
projection inputs determine full-sequence temporary admission before forward;
there is no fixed 4,096-row scratch estimate. CUDA checks include both physically
available and authorized free bytes. Host checks include effective cgroup limits.
Standard quantized binding preserves the previous activation quantizer, fast
accumulation and separate bias addition while delegating GEMM to TorchAO.

Base residency tracks original logical parameter slots across PEFT nesting.
Eviction/reload touches base-owned parameters, while request factors remain live
until cleanup. Physical storage census and H3 finite checks flatten quantized
parts without expanding a dense weight. Current standard bindings support whole
component residency for plain and encoded weights. Adapter components that use
block paging refuse before wrapping.

Existing semantic hooks stay outside native PEFT and retain their handles. H3
fusion gives active or unowned adapters the original floating input. Request
gates reject alternate PEFT selection arguments, changed strengths, escaped or
expired contexts and concurrent use. Failed teardown preserves gates and poisons
the binding. A failure inside native PEFT may retain operands in a traceback;
cleanup propagates poison so the executor generation is retired. A script error
after successful inference can still clean up and reuse a healthy model.

CPU tests cover actual PEFT, TensorFS custody and caller paths. Isolated H100
projection results are recorded under `research/standard-weight-interop`; they
do not establish full-video quality, every tensor shape or cross-GPU byte identity.
The standard provider is registered when TorchAO is installed and admitted by its
current-device native oracle. Streamed CUDA fill/restaging and active PEFT FP8
dispatch now pass on SM89; see `qualification/tensorfs-torch-consumer.md`. H100 and
ordinary 15-second baseline/Spatial/Wushu video comparisons remain promotion gates.

This source is coordinated with refreshed wire57, Creator adapter transport and
Tensorhub nested model closure. No partial protocol deployment or PyPI release
is implied. The experimental wire55 qualification branch is never a production
merge base. Current local-installation reuse capability and newer native custody
behavior remain part of the refreshed protocol and Runtime.

## Declared execution dependencies

Base-only `[model-execution]` does not require or import PEFT/TorchAO. The platform
or image selects `[lora]` when adapters are requested. Its qualified release cohort
is PEFT 0.20.0, TorchAO 0.18.0, Accelerate 1.15.0 and Transformers 5.16.1, with the
worker-owned Torch family satisfying `>=2.13,<3`. No CUDA wheel/index is forced.
This cohort and the historical Transformers4 corpus are explicitly separate
extras; selecting incompatible extras refuses dependency resolution.

A fresh CPU environment installed the built Runtime wheel with `[model-execution]`
and proved empty adapter selection without PEFT/TorchAO. Adding `[lora]` resolved
the released dependencies through package metadata and passed 71 caller/PEFT tests
with one GPU skip, preserving CPU Torch and without installing Ruff. This CPU
closure proof does not replace native H100 qualification of the released libraries.
