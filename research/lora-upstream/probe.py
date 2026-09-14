"""Exercise actual upstream LoRA implementations; CPU arithmetic is not a GPU benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--comfy-source", type=Path, required=True)
    parser.add_argument("--torchao-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("Run with CUDA_VISIBLE_DEVICES='' to keep this proof on CPU")
    sys.path.insert(0, str(args.torchao_source))

    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.tuners.lora.layer import Linear

    from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE
    from cozy_runtime.internal.encoding.leaves import (
        PreQuantized,
        RowwiseNativeLeaf,
        quantize_activation_rowwise,
    )

    sys.path.insert(0, str(args.comfy_source))
    # The upstream CLI's supported CPU mode prevents its device manager selecting CUDA.
    sys.argv = ["upstream-lora-proof", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    from comfy.model_patcher import ModelPatcher
    from comfy.weight_adapter.lora import LoRAAdapter

    torch.set_num_threads(2)
    torch.manual_seed(7381)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    in_features, out_features, rows = 5376, 7168, 64
    x = torch.randn(rows, in_features, dtype=torch.bfloat16)
    base_weight = (torch.randn(out_features, in_features) * 0.01).to(torch.bfloat16)
    metadata = json.loads((args.factor_dir / "cpu-real-factors.json").read_text())
    factors = {}
    for adapter in metadata["factors"]:
        loaded = []
        for letter, part in zip(("A", "B"), adapter["parts"], strict=True):
            path = args.factor_dir / f"{adapter['label']}-{letter}.bin"
            content = bytearray(path.read_bytes())
            actual = hashlib.sha256(content).hexdigest()
            assert actual == part["sha256"].removeprefix("sha256:"), path
            dtype = {"f16": torch.float16, "f32": torch.float32}[part["dtype"]]
            loaded.append(torch.frombuffer(content, dtype=dtype).clone().reshape(part["shape"]))
        factors[adapter["label"]] = loaded

    def configuration(label):
        rank = factors[label][0].shape[0]
        return LoraConfig(r=rank, lora_alpha=rank, target_modules=["0"], inference_mode=True)

    def load_adapter(model, label, dtype):
        state = {
            f"base_model.model.0.lora_{letter}.weight": tensor.to(dtype)
            for letter, tensor in zip(("A", "B"), factors[label], strict=True)
        }
        incompatible = set_peft_model_state_dict(model, state, adapter_name=label)
        assert not incompatible.unexpected_keys, incompatible
        assert not any(f".{label}.weight" in key for key in incompatible.missing_keys), incompatible

    def dense_model(autocast_adapter_dtype):
        base = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
        base.weight.data.copy_(base_weight)
        return get_peft_model(
            torch.nn.Sequential(base),
            configuration("spatial"),
            adapter_name="spatial",
            autocast_adapter_dtype=autocast_adapter_dtype,
        ).eval()

    def metrics(value, reference):
        error = (value.double() - reference.double()).norm()
        return {
            "relative_l2": float(error / reference.double().norm()),
            "max_abs": float((value.float() - reference.float()).abs().max()),
        }

    result = {
        "scope": (
            "CPU: real full-size adapter factors; generated base and activations; "
            "no GPU/video claim"
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "peft", "diffusers", "transformers", "accelerate")
        },
        "shape": {"input": list(x.shape), "weight": list(base_weight.shape)},
        "factor_sources": metadata["factors"],
        "adapters": {},
    }

    with torch.inference_mode():
        promoted = dense_model(True)
        inference = dense_model(False)
        result["peft_default_dtype"] = str(
            promoted.base_model.model[0].lora_A["spatial"].weight.dtype
        )
        result["peft_autocast_false_dtype"] = str(
            inference.base_model.model[0].lora_A["spatial"].weight.dtype
        )
        assert result["peft_default_dtype"] == "torch.float32"
        assert result["peft_autocast_false_dtype"] == "torch.bfloat16"

        for label, strength in (("spatial", 0.3), ("wushu", 0.8)):
            if label != "spatial":
                promoted.add_adapter(label, configuration(label), autocast_adapter_dtype=True)
                inference.add_adapter(label, configuration(label), autocast_adapter_dtype=False)
            outputs = {}
            deltas = {}
            for mode, model, dtype in (
                ("fp32", promoted, torch.float32),
                ("bf16", inference, torch.bfloat16),
            ):
                load_adapter(model, label, dtype)
                model.set_adapter(label)
                layer = model.base_model.model[0]
                layer.set_scale(label, strength)
                outputs[mode] = model(x)
                deltas[mode] = layer.lora_B[label](layer.lora_A[label](x.to(dtype))) * strength
            # Use ComfyUI's actual loaded adapter and bypass API, not a copied formula.
            a, b = factors[label]
            comfy = LoRAAdapter.load(
                "projection",
                {
                    "projection.lora_A.weight": a,
                    "projection.lora_B.weight": b,
                },
                alpha=None,
                dora_scale=None,
            )
            assert comfy is not None
            comfy.multiplier = strength
            comfy_delta = comfy.h(x, outputs["bf16"])
            assert torch.equal(comfy_delta, deltas["bf16"])
            result["adapters"][label] = {
                "rank": int(a.shape[0]),
                "strength": strength,
                "bf16_delta_vs_fp32": metrics(deltas["bf16"], deltas["fp32"]),
                "bf16_result_vs_fp32": metrics(outputs["bf16"], outputs["fp32"]),
                "comfy_bypass_equals_peft_bf16_delta": True,
            }

        promoted.set_adapter("spatial")
        spatial_once = promoted(x)
        promoted.set_adapter("wushu")
        promoted(x)
        promoted.set_adapter("spatial")
        result["peft_switch_back_exact"] = torch.equal(spatial_once, promoted(x))
        assert result["peft_switch_back_exact"]

        # PEFT fusion removes adapter work. It does not promise unchanged rounded outputs.
        layer = promoted.base_model.model[0]
        pristine = layer.get_base_layer().weight.detach().clone()
        layer.merge(safe_merge=True, adapter_names=["spatial"])
        fused = promoted(x)
        result["peft_bf16_fusion_vs_residual"] = metrics(fused, spatial_once)
        layer.unmerge()
        result["peft_bf16_unmerge_restores_weight_bytes"] = torch.equal(
            layer.get_base_layer().weight.view(torch.uint8), pristine.view(torch.uint8)
        )
        # Always recover the immutable original, regardless of subtraction accuracy.
        layer.get_base_layer().weight.copy_(pristine)

        # ComfyUI restores its retained original, rather than subtracting a rounded delta.
        comfy_base = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
        )
        comfy_base[0].weight.copy_(base_weight)
        patcher = ModelPatcher(comfy_base, torch.device("cpu"), torch.device("cpu"))
        a, b = factors["spatial"]
        adapter = LoRAAdapter.load(
            "projection",
            {
                "projection.lora_A.weight": a,
                "projection.lora_B.weight": b,
            },
            alpha=None,
            dora_scale=None,
        )
        assert adapter is not None
        assert patcher.add_patches({"0.weight": adapter}, strength_patch=0.3) == ["0.weight"]
        patcher.patch_weight_to_device("0.weight", device_to=torch.device("cpu"))
        result["comfy_patch_changed_weight"] = not torch.equal(comfy_base[0].weight, base_weight)
        patcher.unpatch_model()
        result["comfy_unpatch_restores_weight_bytes"] = torch.equal(
            comfy_base[0].weight.view(torch.uint8), base_weight.view(torch.uint8)
        )
        assert result["comfy_patch_changed_weight"]
        assert result["comfy_unpatch_restores_weight_bytes"]

        # Construct the REAL Cozy leaf; do not invent a dense .weight facade or CPU GEMM.
        payload = base_weight.to(torch.float8_e4m3fn)
        scales = torch.ones(out_features, dtype=torch.float32)
        leaf = RowwiseNativeLeaf(SPEC_ROWWISE).leaf(
            torch,
            {"data": payload, "scale": scales},
            torch.nn.Linear(in_features, out_features, bias=False, device="meta"),
            torch.bfloat16,
        )
        assert not hasattr(leaf, "weight")
        # Real H3 also has ordinary normalization Parameters. A buffers-only toy model
        # fails PEFT's top-level device census before it reaches the adapter dispatcher.
        encoded_model = torch.nn.Sequential(leaf, torch.nn.LayerNorm(out_features))
        config = configuration("spatial")
        try:
            get_peft_model(encoded_model, config, adapter_name="spatial")
        except ValueError as error:
            result["encoded_default_injection_refusal"] = str(error)
        else:
            raise AssertionError("Expected default dispatch to refuse the unsupported encoded leaf")

        config._register_custom_module({type(leaf): Linear})
        try:
            get_peft_model(encoded_model, config, adapter_name="spatial")
        except StopIteration:
            result["encoded_custom_injection_refusal"] = (
                "StopIteration: LoraModel._replace_module uses next(child.parameters()) "
                "when the leaf has no weight/qweight/W_q/in_proj_weight"
            )
        else:
            raise AssertionError("Expected the current upstream buffer-only placement gap")
        # The reusable upstream layer constructor itself works; the model injector's
        # device selection is the gap. This does not bypass or fake the base forward.
        actual = Linear(leaf, "spatial", config=config, r=16, lora_alpha=16)
        actual.lora_A["spatial"].weight.copy_(factors["spatial"][0].float())
        actual.lora_B["spatial"].weight.copy_(factors["spatial"][1].float())
        result["encoded_direct_upstream_wrapper"] = {
            "wrapper": f"{type(actual).__module__}.{type(actual).__name__}",
            "base_identity_preserved": actual.get_base_layer() is leaf,
            "payload_alias_preserved": actual.get_base_layer().data.data_ptr()
            == payload.data_ptr(),
            "has_fake_weight": hasattr(leaf, "weight"),
        }
        assert actual.get_base_layer() is leaf
        assert actual.get_base_layer().data.data_ptr() == payload.data_ptr()
        try:
            actual.merge(safe_merge=True)
        except AttributeError as error:
            result["encoded_generic_merge_refusal"] = str(error)
        else:
            raise AssertionError("Generic PEFT fusion must not silently reinterpret encoded bytes")
        result["encoded_forward"] = "not run: native W8A8 forward requires CUDA"

        # A genuine upstream quantized Tensor gives nn.Linear its ordinary .weight
        # while retaining encoded storage. This is not a materialized dense facade.
        from torchao.float8.inference import Float8MMConfig
        from torchao.quantization import PerRow
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
            QuantizeTensorToFloat8Kwargs,
        )

        quantized = Float8Tensor(
            payload,
            scales.reshape(-1, 1),
            block_size=[1, in_features],
            dtype=torch.bfloat16,
            mm_config=Float8MMConfig(use_fast_accum=True),
            act_quant_kwargs=QuantizeTensorToFloat8Kwargs(granularity=PerRow()),
        )
        standard_leaf = torch.nn.Linear(in_features, out_features, bias=False, device="meta")
        standard_leaf.weight = torch.nn.Parameter(quantized, requires_grad=False)
        standard = get_peft_model(
            torch.nn.Sequential(standard_leaf),
            configuration("spatial"),
            adapter_name="spatial",
            autocast_adapter_dtype=False,
        )
        load_adapter(standard, "spatial", torch.bfloat16)
        standard_wrapper = standard.base_model.model[0]
        final_weight = standard_wrapper.get_base_layer().weight
        result["torchao_encoded_injection"] = {
            "wrapper": f"{type(standard_wrapper).__module__}.{type(standard_wrapper).__name__}",
            "weight_type": f"{type(final_weight).__module__}.{type(final_weight).__name__}",
            "payload_alias_preserved": final_weight.qdata.data_ptr() == payload.data_ptr(),
            "scale_alias_preserved": final_weight.scale.data_ptr() == scales.data_ptr(),
            "factor_dtype": str(standard_wrapper.lora_A["spatial"].weight.dtype),
            "forward": "not run: no CUDA kernel qualification in CPU proof",
        }
        assert final_weight.qdata.data_ptr() == payload.data_ptr()
        assert final_weight.scale.data_ptr() == scales.data_ptr()
        tensor_names, attributes = final_weight.__tensor_flatten__()
        result["torchao_storage_census"] = {
            "named_base_parameters": [name for name, _ in standard_leaf.named_parameters()],
            "named_base_buffers": [name for name, _ in standard_leaf.named_buffers()],
            "logical_weight_bytes": final_weight.numel() * final_weight.element_size(),
            "physical_weight_bytes": sum(
                getattr(final_weight, name).nbytes for name in tensor_names
            ),
            "flatten_tensor_names": tensor_names,
            "flatten_attribute_names": list(attributes),
            "base_path_after_peft": "base_model.model.0.base_layer",
        }
        dtype_leaf = RowwiseNativeLeaf(SPEC_ROWWISE).leaf(
            torch,
            {"data": payload.clone(), "scale": scales.clone()},
            torch.nn.Linear(in_features, out_features, bias=False, device="meta"),
            torch.bfloat16,
        )
        dtype_leaf.to(dtype=torch.bfloat16)
        try:
            standard_leaf.to(dtype=torch.float16)
        except NotImplementedError as error:
            result["torchao_to_fp16_refusal"] = str(error)
        result["recursive_dtype_conversion"] = {
            "cozy_after_to_bf16": {
                "payload_dtype": str(dtype_leaf.data.dtype),
                "scale_dtype": str(dtype_leaf.scale.dtype),
            },
            "torchao_after_to_fp16": {
                "logical_dtype": str(standard_leaf.weight.dtype),
                "payload_dtype": str(standard_leaf.weight.qdata.dtype),
                "scale_dtype": str(standard_leaf.weight.scale.dtype),
                "payload_alias_preserved": standard_leaf.weight.qdata.data_ptr()
                == payload.data_ptr(),
                "scale_alias_preserved": standard_leaf.weight.scale.data_ptr() == scales.data_ptr(),
            },
        }
        quantizer_input = x.clone()
        quantizer_input[0].zero_()
        quantizer_input[1].fill_(1e-15)
        cozy_q, cozy_scale = quantize_activation_rowwise(torch, quantizer_input)
        upstream_q = Float8Tensor.from_hp(quantizer_input, granularity=PerRow())
        result["torchao_default_activation_quantizer"] = {
            "zero_row_scale": float(upstream_q.scale[0].item()),
            "zero_row_finite": bool(upstream_q.qdata[0].float().isfinite().all()),
            "cozy_zero_row_scale": float(cozy_scale[0].item()),
            "ordinary_scale_bytes_equal": torch.equal(
                upstream_q.scale[2:].view(torch.uint8), cozy_scale[2:].view(torch.uint8)
            ),
            "ordinary_changed_codes": int(
                (upstream_q.qdata[2:].view(torch.uint8) != cozy_q[2:].view(torch.uint8)).sum()
            ),
            "scope": "TorchAO Torch CPU quantizer versus Cozy eager CPU floor; no GPU kernel claim",
        }
        try:
            standard_leaf(PreQuantized(payload=cozy_q, scale=cozy_scale, shape=tuple(x.shape)))
        except (TypeError, AttributeError) as error:
            result["torchao_cozy_prequantized_refusal"] = str(error)
        else:
            raise AssertionError(
                "The existing PreQuantized dataclass must not silently re-quantize"
            )
        try:
            standard_wrapper.merge(safe_merge=True)
        except ValueError as error:
            result["torchao_merge_without_codec_metadata_refusal"] = str(error)
        else:
            raise AssertionError("Expected PEFT to require the requantization recipe")

    result["cuda_initialized"] = torch.cuda.is_initialized()
    assert not result["cuda_initialized"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
