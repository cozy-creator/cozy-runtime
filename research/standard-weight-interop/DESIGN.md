# Community-first quantized weights and LoRA

Use ordinary `nn.Linear.weight` holding a real upstream TorchAO `Float8Tensor`. Keep TensorFS responsible for encoded bytes and their leases. Make Runtime bind those bytes to the standard parameter object, then use PEFT's existing LoRA layer. There is no measured evidence here that the custom parameter-free `EncodedLinear` representation is faster enough to justify sacrificing that interface.

For the current rowwise FP8 format, this does **not require changing checkpoint bytes or inventing a new file format**. Existing E4M3 payload, FP32 row scales, logical shape/dtype and encoding semantics map to TorchAO's qdata, scale and block size. The scale reshape is a view. Backend choice, accumulation mode, activation quantization and Python class names belong to Runtime, not the persistent container. Other layouts need individual compatibility checks; this proof does not certify every MXFP8/FP4 layout.

The CPU prototype uses the actual full-size spatial and Wushu factors, a generated encoded base and unmodified pinned upstream libraries. It demonstrates:

- `nn.Linear.weight` is genuinely a TorchAO tensor, with unchanged FP8 payload and scale storage aliases; no fake dense weight exists.
- PEFT automatically selects `TorchaoLoraLinear`, holds BF16 adapter parameters and switches spatial→Wushu→spatial exactly.
- A 7168×5376 weight occupies38,563,840 physical bytes, versus77,070,336 logical BF16 bytes. Current Runtime's direct storage-pointer census refuses the wrapper. Upstream tensor flattening plus Runtime's existing distinct-storage counter reports the correct physical bytes and deduplicates aliases.
- The **direct weight path** supports base-only `.to(device="meta")` and standard `load_state_dict({"weight": Float8Tensor(...)}, assign=True)`. Adapter objects/bytes survive, the logical `weight` key survives, and restored CPU output is exact.
- TorchAO's existing `unwrap_tensor_subclass` parametrization also exposes plain physical parameters and supports `.to_empty(meta)` plus restaging. It is an alternative, not the default recommendation: it changes state-dict keys and makes a blind dtype cast alter FP8 payload and FP32 scale types.

Prefer the direct parameter representation and a small physical-tensor binding layer. Preserve the **logical checkpoint slots** established before PEFT changes module nesting. Associate each logical slot with its actual base module/parameter handle and encoded physical parts; do not re-infer checkpoint identity from a later whole-model `state_dict()` that includes `base_layer` and request-owned LoRA parameters.

Ownership must also be explicit. Component eviction operates on base-owned slots, while request-owned adapter parameters remain until that request finishes. Whole-tree `.to_empty(meta)` was tested and does move PEFT's adapter parameters to meta: the existing recursive release rule cannot be reused blindly. Preserve tied/shared storage once, retain TensorFS leases while aliases exist, and never resize borrowed or still-shared storage to zero. Standard PyTorch assignment can adopt storage, but does not invent a safe ownership policy for it.

The persistent format should carry data semantics: names, shape/dtype, exact encoding/scale convention, physical roles, integrity and any already-supported alias relationships. No `TorchAO`/`PEFT` class name, generated parameter path or process pointer needs to enter it. If a future layout's packing or scale semantics cannot be expressed by the existing encoding description, add that missing data contract; do not redesign the container merely because Runtime used the wrong module representation.

Two execution gates remain before replacing the serving path:

1. **Activation contract and quantizer fidelity.** The current `PreQuantized` dataclass is not a Tensor. Its existing payload/scales can become a Float8Tensor without copying or re-quantizing, but a BF16 PEFT adapter cannot use those FP8 activations as its original input. Keep ordinary BF16 input available for LoRA; the base must quantize it once or consume an explicitly compatible pre-quantized value. Never dequantize a lossy activation and pretend it is the original. TorchAO's default scale arithmetic/zero handling differs from the qualified Cozy recipe, and its configured dynamic path rejects already-quantized inputs. Qualify a declared recipe or upstream fix; do not silently change it. A community baseline may opt out of the custom fused shortcut while that integration is measured.
2. **Full-sequence memory and real performance.** Upstream PEFT removes the bespoke4096-row launch loop, but can keep full output temporaries. The prototype's109,104-token,7168-output BF16 example needs a conservative additional3,135,212,544-byte bound beyond input/base output. Admission must price the actual largest affected projection, factors and base workspace. Then compare the existing W8A8 base against TorchAO with identical encoded operands, followed by15-second baseline/spatial/Wushu runs, repeats and switches.

The CPU forward used here is explicitly TorchAO's BF16-activation **weight-only dequantization path**, not FP8 CUDA GEMM or a speed result. It must not become an accidental production fallback. Unchanged encoded operands can exercise the upstream W8A8 path on H100 next. Fusion is disabled without a real re-quantization recipe, and rounded `unmerge` is never a recovery strategy: a prepared merged variant must be separately qualified and recovered from immutable base bytes.

This is a design/prototype stage of h3a-065. It does not close the PEFT serving implementation or GPU qualification work.

The first serving-side prerequisite now has a regression fix:
`internal.forensics.distinct_storage_bytes` uses TorchAO's upstream flattening
utility for wrapper tensors. Real CPU tests cover direct Float8Tensor parameters,
shared payload/scale views, unchanged logical keys and meta eviction. Ordinary
tensor accounting still needs neither a TorchAO import nor a TorchAO dependency.
This fixes physical census only; it does not authorize whole-tree eviction or
select a new GEMM provider.

The next candidate is now implemented in `encoding/torchao_rowwise.py`, with
`BaseWeights` integrated into streamed fill, eviction and restaging for that
provider. It binds the weight in place on the existing `nn.Linear`; original
hooks and parameter names remain valid. A receiver-bound forward performs the
existing activation quantization, upstream TorchAO `F.linear` without fused bias,
then the original separate bias addition. This small binding is necessary for
the first comparison because upstream's ordinary BF16 bias epilogue changes the
rounding order. Its CPU forward explicitly refuses a weight-only fallback.

The provider is deliberately **not in `launch_providers` yet**. Qualification
must run the actual CUDA test, then the native streaming lifecycle and ordinary
request/clip comparisons. Do not infer GPU proof from CPU tests. The current
CPU tests cover all638 logical slots of an actual50-block upstream H3 meta
construction, real PEFT base eviction/reload and exact CPU weight-only output,
retained external storage aliases, tied Parameter objects, late request-owned
Parameters, hook handles and deepcopy. Distinct shared views and components
mixing this standard binding with legacy encoded replacement modules explicitly
refuse. They are not silently normalized.

The permanent request-level PEFT change is still pending the coordinated modern
LoRA integration foundation. The existing wire55 qualification cohort already
has that foundation and can receive a bounded math/lifetime delta for testing;
that cohort is never a permanent feature branch or production protocol upgrade.
