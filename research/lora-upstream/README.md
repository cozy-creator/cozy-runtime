# LoRA upstream reuse investigation

Owner: `/root/lora_upstream_reuse`. Base Runtime commit: `b30c56c2`.

This investigation uses current, unmodified upstream code to decide which parts
of Runtime's experimental LoRA path should be replaced. It makes no GPU
performance or video-quality claim from CPU results.

| Project | Examined commit |
|---|---|
| PEFT | `e99fdd275075068d093f8996a59bc0eb865f71a2` |
| Diffusers | `1f7be81f9bc84e7f2ab1ae91d37c7236c4c15691` |
| ComfyUI | `b00584967778871f329c2d31190a5e9b767b8366` |
| TorchAO | `9e5ea7fde7132df0c09b2acb12c2defad90aecb9` |

The executable proof uses the real published spatial and Wushu LoRA factors at
their full 7168-by-5376 projection geometry. Its base weights and activations are
generated fixtures; retained factors are hash-checked. It exercises upstream
PEFT loading, dtype selection, residual execution, switching, and fusion. It
also attempts supported custom-module injection against Runtime's actual encoded
leaf. No fake `.weight` property or replacement FP8 computation is introduced.

## Decision

Reuse PEFT's adapter loading, selection, residual layers and supported preparation
math. Do not grow another custom LoRA arithmetic framework. The existing FP32
residual is a qualification reference, not the desired inference implementation.
For model/codec combinations where fusion passes quality qualification, prepare a
merged variant once, retain it through TensorFS and execute the ordinary base
kernel. Use the original immutable base to switch variants; never undo a rounded
merge by subtraction.

Our FP8 representation is a real integration boundary, but not a reason to own
LoRA matrix multiplication. A documented PEFT custom dispatch mapping presently
fails on a parameter-free encoded leaf. Direct construction of PEFT's existing
`Linear` wrapper works without copying or decoding the base. TorchAO's existing
`Float8Tensor` can also represent the same physical buffers in an ordinary
`nn.Linear`, and PEFT then selects its existing `TorchaoLoraLinear` automatically.
The latter is a worthwhile deletion candidate, not a qualified replacement: the
CPU proof also exposes incompatible quantization defaults, storage census and
pre-quantized-input conventions.

No serving defaults, shared installations, image selection or GPU hardware were
changed by this investigation. The expensive custom implementation lives on the
separate experimental LoRA branch; this research branch starts from clean master.

## What upstream already owns

| Area | Existing implementation | Consequence for Cozy |
|---|---|---|
| Residual inference | PEFT `Linear.forward`: base forward, two adapter `nn.Linear` calls, scaling/addition and output dtype restoration | Reuse the layer; avoid the custom 4096-row arithmetic and scratch loops where memory admission permits |
| Inference dtype | `get_peft_model(..., autocast_adapter_dtype=False)` and explicit factor dtype at load | FP32 is PEFT's default promotion too, but it is optional; upstream inference does not require FP32 |
| Multiple adapters | PEFT `add_adapter`, `set_adapter`, state loading; Diffusers `set_adapters(..., adapter_weights=...)` | Reuse adapter execution state; retain Cozy's external named-checkpoint and request-isolation contract |
| Same-slot replacement | PEFT `hotswap_adapter_from_state_dict`, `prepare_model_for_compiled_hotswap`; Diffusers `load_lora_weights(..., hotswap=True)` | Existing in-place adapter replacement avoids growing the module tree or recompiling compatible graphs; it is not a persistent merged-variant cache |
| Dense fusion | Diffusers `fuse_lora` delegates to PEFT `merge`; `merge_and_unload` removes the wrapper | No reason to reimplement ordinary LoRA dense-product arithmetic |
| Quantized fusion | PEFT `TorchaoLoraLinear.merge` dequantizes, adds the delta, calls the recovered quantization recipe | Quantization is still required. The library does not promise exact residual equivalence or reversible rounded fusion |
| ComfyUI resident models | `ModelPatcher.patch_weight_to_device` applies patches once during load, then uses the patched weight | This is the desired preparation/inference separation for a qualified variant |
| ComfyUI recovery | `unpatch_model` restores retained original weights | Matches immutable-base recovery; preferable to subtractive unmerge |
| ComfyUI constrained memory | `LowVramPatch` applies patches while a weight is staged; current `LoRAAdapter.h` also supports residual bypass | ComfyUI does not guarantee all configurations avoid inference-time work |

Sources are pinned rather than inferred from current documentation examples:
[PEFT residual/merge](https://github.com/huggingface/peft/blob/e99fdd275075068d093f8996a59bc0eb865f71a2/src/peft/tuners/lora/layer.py#L952-L1114),
[PEFT adapter dtype](https://github.com/huggingface/peft/blob/e99fdd275075068d093f8996a59bc0eb865f71a2/src/peft/tuners/tuners_utils.py#L2686),
[PEFT hot swapping](https://github.com/huggingface/peft/blob/e99fdd275075068d093f8996a59bc0eb865f71a2/src/peft/utils/hotswap.py#L319),
[Diffusers loading](https://github.com/huggingface/diffusers/blob/1f7be81f9bc84e7f2ab1ae91d37c7236c4c15691/src/diffusers/loaders/peft.py#L79),
[Diffusers fusion](https://github.com/huggingface/diffusers/blob/1f7be81f9bc84e7f2ab1ae91d37c7236c4c15691/src/diffusers/loaders/peft.py#L646),
[TorchAO PEFT merge](https://github.com/huggingface/peft/blob/e99fdd275075068d093f8996a59bc0eb865f71a2/src/peft/tuners/lora/torchao.py#L30),
[ComfyUI patch/load/recovery](https://github.com/Comfy-Org/ComfyUI/blob/b00584967778871f329c2d31190a5e9b767b8366/comfy/model_patcher.py#L899-L1172),
[ComfyUI bypass](https://github.com/Comfy-Org/ComfyUI/blob/b00584967778871f329c2d31190a5e9b767b8366/comfy/weight_adapter/lora.py#L289).

ComfyUI's quantized setter dequantizes and recalculates scales when patching. It
does not add raw FP8 codes or eliminate codec error. Its restoration backup is
the reusable design here; importing its whole model manager into Runtime would
duplicate scheduling, residency and eviction ownership. This probe imports real
ComfyUI for comparison; it copies no ComfyUI source into Runtime.
[Quantized setter](https://github.com/Comfy-Org/ComfyUI/blob/b00584967778871f329c2d31190a5e9b767b8366/comfy/ops.py#L1439-L1457).

## Executed proof

`probe.py` ran with CUDA hidden and two CPU threads. It imports the pinned PEFT,
ComfyUI and TorchAO implementations, verifies the retained canonical factor bytes,
then uses full projection geometry (7168 outputs, 5376 inputs; ranks 16 and 32).
The 64-token activations and base weights are generated. Neither the actual H3
FP8 CUDA forward nor full-video generation is claimed by this proof.

| Observation | Actual result |
|---|---|
| PEFT default / explicit no-autocast factors | FP32 / BF16 |
| BF16 spatial delta vs FP32 delta, strength 0.3 | 0.38191% relative L2 |
| BF16 Wushu delta vs FP32 delta, strength 0.8 | 0.30959% relative L2 |
| ComfyUI residual bypass vs PEFT BF16 delta | Exactly equal, both adapters |
| PEFT spatial → Wushu → spatial selection | Exactly equal restored output |
| PEFT dense BF16 fusion vs its residual | 0.22478% output relative L2 |
| PEFT merge/unmerge restores original BF16 weight bytes | **No** |
| ComfyUI patch/unpatch restores original BF16 weight bytes | **Yes** |
| PEFT default injection into actual Cozy encoded leaf | Refuses unsupported module |
| Documented PEFT custom mapping to existing `Linear` | Fails `StopIteration` selecting the leaf device |
| Direct upstream `Linear` around actual encoded leaf | Works; identical base object and payload storage |
| Existing TorchAO `Float8Tensor` around the same payload/scales | Both physical storage aliases retained |
| PEFT standard injection into that TorchAO `nn.Linear` | Selects upstream `TorchaoLoraLinear`; BF16 adapter factors |
| Generic merge into Cozy encoded leaf | Refuses absent `.weight` |
| TorchAO merge with no quantization recipe | Refuses missing `get_apply_tensor_subclass` |

These are arithmetic and API results, not estimated image quality. The prior
actual-H3 FP8 merge probe measured rounding error comparable to the entire LoRA
update: weight error / update norm 1.0388 with a grow-only grid. That evidence
prevents declaring either PEFT or ComfyUI fusion equivalent just because a
floating-point sum is mathematically valid. See retained
`outputs/h3-lora-profile-20260914/MERGE-REPORT.md`.

The custom-injection failure is small and specific:
`LoraModel._replace_module` looks for known weight attributes, then uses
`next(child.parameters())`. Our encoded leaf has `data`/`scale` buffers and no
Parameters. A supported buffer-device fallback upstream would remove that
impedance mismatch; no fake dense `.weight` should be added. An all-buffer toy
model also needs the top-level device census addressed, whereas real H3 has
ordinary normalization Parameters. The proof includes a real LayerNorm for that
reason. [Exact injector](https://github.com/huggingface/peft/blob/e99fdd275075068d093f8996a59bc0eb865f71a2/src/peft/tuners/lora/model.py#L364-L391).

## TorchAO is promising, but not a drop-in representation

The `Float8Tensor(qdata, scale, ..., dtype=torch.bfloat16)` constructor adopts
existing physical FP8 bytes without decoding or re-quantizing them. Its standard
tensor flattening exposes both `qdata` and `scale`. PEFT recognizes the resulting
real quantized weight through its existing dispatcher, with no custom LoRA class.
[Constructor](https://github.com/pytorch/ao/blob/9e5ea7fde7132df0c09b2acb12c2defad90aecb9/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L84-L166),
[flattening](https://github.com/pytorch/ao/blob/9e5ea7fde7132df0c09b2acb12c2defad90aecb9/torchao/utils.py#L924-L966).

The same real CPU probe also establishes adoption gates:

- **Physical census:** `named_parameters()` exposes one logical BF16 `weight`
  (77,070,336 nominal bytes); `named_buffers()` exposes neither physical part.
  Actual flattened bytes are 38,563,840. Runtime must flatten and account physical
  storage, preserve alias ownership, and test evacuation/restaging; simple dtype
  accounting is incorrect. The module path also becomes `...base_layer` after
  PEFT injection, so authored target names and restaging must use stable bindings.
- **Recursive dtype conversion:** the current Cozy leaf's comment attributes
  safety to using buffers instead of Parameters. The actual
  `EncodedLinear.to(dtype=torch.bfloat16)` casts both its FP8 payload and FP32
  scales to BF16. Buffers do not provide that protection. On the current TorchAO
  source/Torch pairing, requesting FP16 through ordinary module `.to` raises
  `NotImplementedError` and leaves the existing physical storage intact. Neither
  behavior is an already-qualified general conversion boundary.
- **Quantization semantics:** current TorchAO's Torch fallback divides the BF16
  amax before casting the scale to FP32 and has no default positive floor. Its
  zero row produced scale zero and nonfinite FP8 codes. Cozy casts amax to FP32
  before division and floors scales at 1e-12. On ordinary random rows, 6,105 of
  333,312 FP8 codes differed. These are CPU paths; no assertion is made about
  MSLK's CUDA quantizer. Explicit source checks and GPU parity are required.
- **Kernel selection:** TorchAO AUTO can choose MSLK; the current Cozy base uses
  `torch._scaled_mm(..., use_fast_accum=True)`. TorchAO exposes TORCH selection
  and `Float8MMConfig(use_fast_accum=True)`, but preprocessing/output details must
  still be compared using actual retained H3 inputs.
- **Pre-fused inputs:** Cozy's `PreQuantized` input is not a Tensor, and ordinary
  upstream `nn.Linear` rejects it. TorchAO can carry pre-quantized activations in
  its own tensor type with dynamic activation quantization disabled; blindly
  handing an already-quantized input to the default dynamic path is rejected.
  Existing fusion hooks and their removal handles must survive any leaf change.
- **Codec metadata:** PEFT recovers re-quantization from
  `hf_quantizer.quantization_config.get_apply_tensor_subclass`. Adoption must
  provide a real codec recipe or keep merge disabled. Never synthesize metadata
  merely to make a refusal disappear.

[TorchAO scale rule](https://github.com/pytorch/ao/blob/9e5ea7fde7132df0c09b2acb12c2defad90aecb9/torchao/quantization/quant_primitives.py#L2207-L2246),
[quantizer/backend selection](https://github.com/pytorch/ao/blob/9e5ea7fde7132df0c09b2acb12c2defad90aecb9/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L168-L258),
[activation and GEMM dispatch](https://github.com/pytorch/ao/blob/9e5ea7fde7132df0c09b2acb12c2defad90aecb9/torchao/quantization/quantize_/workflows/float8/float8_tensor.py#L337-L459).

## Smallest integration and deletion plan

1. Keep checkpoint resolution, canonical key/shape/finite validation, adapter
   strengths/order, immutable base identity, request isolation and memory grants
   in Cozy. These are platform responsibilities rather than adapter arithmetic.
2. Qualify PEFT's existing residual layer against the unchanged current encoded
   base first, using explicit BF16 factors as a candidate and FP32 as the
   reference. Runtime can install the existing wrapper at its already-owned
   encoded-leaf boundary; a tiny upstream device fallback is another route. Both
   avoid rewriting low-rank math. Preserve stable target mappings/hook handles.
3. Measure full-sequence memory before replacing fixed tiling. At 109,104 tokens,
   one FP32 [tokens,7168] output is 2.913 GiB, and an FP32 [tokens,5376] input cast
   is 2.185 GiB. PEFT removes dozens of tiled launch sequences but can have several
   such temporaries live. BF16 reduces this pressure, but its quality is unproven.
   Do not trade the measured launch overhead for an uncharged OOM.
4. Compare the normal 15-second fight request for both adapters/strengths,
   baseline, repeats and switching; record preparation vs execution separately.
   No speedup is claimed until this runs through ordinary Cozy execution.
5. For accepted merge/codec combinations, use upstream GEMM/merge in preparation
   and retain the resulting immutable encoded variant in TensorFS. Variant
   identity includes base, ordered adapters/strengths, codec/scale policy and
   computation revision. Reuse a matching variant, otherwise reload the pristine
   base. Delete the inference residual hook only for these qualified variants.
6. Evaluate replacing custom encoded wrappers with TorchAO only after the storage,
   quantizer, hook and PreQuantized gates above. This can delete additional custom
   module/device glue; it should not expand into a new generic quantization layer.

The custom rank-ordered merge implementation is not the production direction:
the prior actual-GPU probe produced identical final FP8 codes/scales with an
ordinary FP32 GEMM, and the latter was faster. Preserve a reference if needed for
diagnosis; avoid shipping two competing preparation algorithms.

## Reproduction and retained evidence

The workstation run used an isolated venv, pinned source checkouts and existing
local Torch/TensorFS wheels. No dev tools or dependency changes were sent to a
worker. Current PEFT, Diffusers and TorchAO were installed/imported from the
commits above; ComfyUI was imported from its unchanged checkout. The environment
and source fingerprints are retained beside `cpu-proof.json`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$runtime_worktree/src" \
  "$proof_python" research/lora-upstream/probe.py \
  --factor-dir "$retained_factors" \
  --comfy-source "$comfy_checkout" \
  --torchao-source "$torchao_checkout" \
  --output "$proof_output"
```

The input directory contains `cpu-real-factors.json` plus `spatial-A.bin`,
`spatial-B.bin`, `wushu-A.bin`, `wushu-B.bin`. Every factor is hash checked before
execution. Workstation evidence lives at
`/home/fidika/cozy_v2/outputs/h3-lora-upstream-20260914/`; the original inputs at
`/home/fidika/cozy_v2/outputs/h3-general-lora-20260914/prepared-merge/`.
