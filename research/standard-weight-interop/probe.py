"""CPU standard-weight/PEFT lifecycle prototype, using unmodified upstream code.

The CPU numerical path is explicitly BF16 activation + dequantized-weight matmul,
not FP8 CUDA GEMM emulation or a performance claim. Encoded buffers/factors remain
independently inspectable. W8A8 uses the upstream Tensor API but requires a GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
from types import SimpleNamespace
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("This prototype requires CUDA_VISIBLE_DEVICES=''")

    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
    from torchao.utils import UnwrapTensorSubclass, get_model_size_in_bytes, unwrap_tensor_subclass

    from cozy_runtime.author._loader import census
    from cozy_runtime.internal.encoding.leaves import PreQuantized
    from cozy_runtime.internal.forensics import distinct_storage_bytes

    torch.set_num_threads(2)
    torch.manual_seed(7381)
    torch.set_float32_matmul_precision("highest")
    in_features, out_features = 5376, 7168
    payload = torch.randint(-100, 101, (out_features, in_features)).to(torch.float8_e4m3fn)
    scale = torch.linspace(1e-4, 1e-2, out_features, dtype=torch.float32).reshape(-1, 1)
    original_payload = payload.clone()
    original_scale = scale.clone()
    x = torch.randn(8, in_features, dtype=torch.bfloat16)
    result: dict[str, Any] = {
        "scope": "CPU ownership/interface proof; real adapter factors; generated FP8 base; no CUDA GEMM or quality/speed claim",
        "versions": {
            n: metadata.version(n)
            for n in ("torch", "torchao", "peft", "diffusers", "transformers", "tensorfs")
        },
    }

    def encoded(q: Any, s: Any, *, activation_quantization: Any = None) -> Any:
        return Float8Tensor(
            q,
            s,
            block_size=[1, in_features],
            dtype=torch.bfloat16,
            mm_config=Float8MMConfig(use_fast_accum=True),
            act_quant_kwargs=activation_quantization,
            kernel_preference=KernelPreference.TORCH,
        )

    def leaf(q: Any, s: Any) -> Any:
        module = torch.nn.Linear(in_features, out_features, bias=False, device="meta")
        module.weight = torch.nn.Parameter(encoded(q, s), requires_grad=False)
        return module

    base = leaf(payload, scale)
    assert base.weight.qdata.data_ptr() == payload.data_ptr()
    assert base.weight.scale.data_ptr() == scale.data_ptr()
    result["adoption"] = {
        "module": type(base).__name__,
        "weight_type": type(base.weight).__name__,
        "logical_dtype": str(base.weight.dtype),
        "payload_dtype": str(base.weight.qdata.dtype),
        "scale_dtype": str(base.weight.scale.dtype),
        "payload_alias": True,
        "scale_alias": True,
        "logical_bytes": base.weight.numel() * base.weight.element_size(),
        "physical_bytes": payload.nbytes + scale.nbytes,
        "torchao_physical_census": get_model_size_in_bytes(torch.nn.Sequential(base)),
    }
    flattened = UnwrapTensorSubclass().right_inverse(base.weight)
    assert distinct_storage_bytes(flattened) == payload.nbytes + scale.nbytes
    assert distinct_storage_bytes([*flattened, *flattened]) == payload.nbytes + scale.nbytes
    result["direct_weight_physical_census_via_upstream_flattening"] = {
        "bytes": distinct_storage_bytes(flattened),
        "aliases_count_once": True,
        "logical_weight_key_unchanged": list(base.state_dict()) == ["weight"],
    }
    result["logical_census_before_wrapping"] = [
        row.key for row in census(SimpleNamespace(components={"dit": base})).destinations
    ]
    try:
        result["runtime_direct_storage_census"] = distinct_storage_bytes(base.parameters())
    except Exception as exc:
        result["runtime_direct_storage_census_refusal"] = f"{type(exc).__name__}: {exc}"

    factors = {}
    for row in json.loads((args.factor_dir / "cpu-real-factors.json").read_text())["factors"]:
        parts = []
        for letter, info in zip(("A", "B"), row["parts"], strict=True):
            data = bytearray((args.factor_dir / f"{row['label']}-{letter}.bin").read_bytes())
            assert hashlib.sha256(data).hexdigest() == info["sha256"].removeprefix("sha256:")
            dtype = {"f16": torch.float16, "f32": torch.float32}[info["dtype"]]
            parts.append(torch.frombuffer(data, dtype=dtype).clone().reshape(info["shape"]))
        factors[row["label"]] = parts

    def config(label: str) -> Any:
        rank = factors[label][0].shape[0]
        return LoraConfig(r=rank, lora_alpha=rank, target_modules=["0"], inference_mode=True)

    with torch.no_grad():
        model = get_peft_model(
            torch.nn.Sequential(base),
            config("spatial"),
            adapter_name="spatial",
            autocast_adapter_dtype=False,
        ).eval()
        model.add_adapter("wushu", config("wushu"), autocast_adapter_dtype=False)
        layer = model.base_model.model[0]
        assert type(layer).__name__ == "TorchaoLoraLinear"
        for label, strength in (("spatial", 0.3), ("wushu", 0.8)):
            state = {
                f"base_model.model.0.lora_{letter}.weight": value.to(torch.bfloat16)
                for letter, value in zip(("A", "B"), factors[label], strict=True)
            }
            loaded = set_peft_model_state_dict(model, state, adapter_name=label)
            assert not loaded.unexpected_keys
            assert not any(f".{label}.weight" in name for name in loaded.missing_keys)
            layer.set_scale(label, strength)
        result["peft"] = {
            "wrapper": type(layer).__name__,
            "base_object_preserved": layer.base_layer is base,
            "factor_dtypes": {n: str(m.weight.dtype) for n, m in layer.lora_A.items()},
        }

        # Direct standard Tensor-subclass path: preserve the logical weight key.
        # Upstream device movement works even though to_empty is unsupported.
        model.set_adapter("spatial")
        direct_before = model(x).clone()
        direct_adapters = {
            n: (p.data_ptr(), p.clone()) for n, p in model.named_parameters() if ".lora_" in n
        }
        base.to(device="meta")
        assert base.weight.device.type == base.weight.qdata.device.type == "meta"
        restored = base.load_state_dict({"weight": encoded(payload, scale)}, assign=True)
        assert not restored.missing_keys and not restored.unexpected_keys
        assert all(
            p.data_ptr() == direct_adapters[n][0] and torch.equal(p, direct_adapters[n][1])
            for n, p in model.named_parameters()
            if n in direct_adapters
        )
        assert torch.equal(model(x), direct_before)
        result["direct_weight_evict_reload"] = {
            "method": "base.to(meta), base.load_state_dict(weight=Float8Tensor, assign=True)",
            "logical_key": list(base.state_dict()),
            "all_roles_reached_meta": True,
            "adapter_identity_and_values_preserved": True,
            "restored_output_exact": True,
            "payload_alias_preserved": base.weight.qdata.data_ptr() == payload.data_ptr(),
            "scale_alias_preserved": base.weight.scale.data_ptr() == scale.data_ptr(),
        }

        # Public upstream export/compile helper exposes plain physical Parameters
        # while keeping nn.Linear.weight a real Float8Tensor.
        unwrap_tensor_subclass(model)
        base = layer.get_base_layer()
        assert isinstance(base, torch.nn.Linear)
        assert isinstance(base.weight, Float8Tensor)
        assert base.weight.qdata.data_ptr() == payload.data_ptr()
        assert base.weight.scale.data_ptr() == scale.data_ptr()
        physical = dict(base.named_parameters())
        assert set(physical) == {
            "parametrizations.weight.original0",
            "parametrizations.weight.original1",
        }
        assert distinct_storage_bytes(physical.values()) == payload.nbytes + scale.nbytes
        result["logical_census_after_parametrization"] = [
            row.key for row in census(SimpleNamespace(components={"dit": base})).destinations
        ]
        result["whole_peft_census_includes_request_parameters"] = [
            row.key
            for row in census(SimpleNamespace(components={"dit": model})).destinations
            if "lora_" in row.key
        ]
        result["parametrized_storage"] = {
            "module": type(base).__name__,
            "weight_type": type(base.weight).__name__,
            "parameters": {
                n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in physical.items()
            },
            "runtime_census": distinct_storage_bytes(physical.values()),
            "same_storage_counted_twice": distinct_storage_bytes(
                [*physical.values(), *physical.values()]
            ),
        }
        outputs = {}
        for label in ("spatial", "wushu", "spatial"):
            model.set_adapter(label)
            output = model(x)
            if label in outputs:
                assert torch.equal(output, outputs[label])
            outputs[label] = output.clone()
        result["cpu_switch_back_exact"] = True

        adapter_parameters = {
            n: p for n, p in model.named_parameters() if ".lora_A." in n or ".lora_B." in n
        }
        before_adapters = {n: (p.data_ptr(), p.clone()) for n, p in adapter_parameters.items()}
        canonical_roles = {name: t.clone() for name, t in base.state_dict().items()}
        base.to_empty(device="meta", recurse=True)
        assert all(p.device.type == "meta" for p in base.parameters())
        assert all(
            p.data_ptr() == before_adapters[n][0] and torch.equal(p, before_adapters[n][1])
            for n, p in adapter_parameters.items()
        )
        reloaded = base.load_state_dict(canonical_roles, assign=True)
        assert not reloaded.missing_keys and not reloaded.unexpected_keys
        assert torch.equal(base.weight.qdata.view(torch.uint8), original_payload.view(torch.uint8))
        assert torch.equal(base.weight.scale, original_scale)
        assert torch.equal(model(x), outputs["spatial"])
        result["base_only_evict_reload"] = {
            "all_base_parameters_were_meta": True,
            "adapter_objects_and_values_preserved": True,
            "output_exact_after_reload": True,
            "canonical_source_bytes_unchanged": True,
        }

        # This is the current recursive-release shape's negative control: it also
        # removes the request-owned adapter tensors from their resident device.
        whole = copy.deepcopy(model)
        whole.to_empty(device="meta", recurse=True)
        result["whole_tree_to_empty_moves_adapters"] = all(
            p.device.type == "meta" for n, p in whole.named_parameters() if ".lora_" in n
        )
        assert result["whole_tree_to_empty_moves_adapters"]

        # Preserve the distinction between tensor-subclass and plain-parameter
        # traversal. Generic dtype conversion is not a safe codec operation.
        unwrapped_dtype = copy.deepcopy(base)
        unwrapped_dtype.to(dtype=torch.bfloat16)
        result["recursive_dtype_cast_after_unwrap"] = {
            n: str(p.dtype) for n, p in unwrapped_dtype.named_parameters()
        }
        direct_dtype = leaf(original_payload.clone(), original_scale.clone())
        try:
            direct_dtype.to(dtype=torch.float16)
        except Exception as exc:
            result["direct_subclass_dtype_refusal"] = f"{type(exc).__name__}: {exc}"
        result["direct_subclass_dtype_after_attempt"] = {
            "logical": str(direct_dtype.weight.dtype),
            "payload": str(direct_dtype.weight.qdata.dtype),
            "scale": str(direct_dtype.weight.scale.dtype),
        }
        direct_move = leaf(original_payload.clone(), original_scale.clone())
        try:
            direct_move.to(device="meta")
            result["direct_subclass_to_meta"] = {
                "logical_device": str(direct_move.weight.device),
                "payload_device": str(direct_move.weight.qdata.device),
                "scale_device": str(direct_move.weight.scale.device),
            }
        except Exception as exc:
            result["direct_subclass_to_meta_refusal"] = f"{type(exc).__name__}: {exc}"
        direct_empty = leaf(original_payload.clone(), original_scale.clone())
        try:
            direct_empty.to_empty(device="meta", recurse=True)
            result["direct_subclass_to_empty"] = {
                "logical_device": str(direct_empty.weight.device),
                "payload_device": str(direct_empty.weight.qdata.device),
            }
        except Exception as exc:
            result["direct_subclass_to_empty_refusal"] = f"{type(exc).__name__}: {exc}"

        # Existing pre-quantized payload is adopted, not run through a second
        # quantizer. CPU _scaled_mm is deliberately not emulated here.
        from cozy_runtime.internal.encoding.leaves import quantize_activation_rowwise

        xq, xs = quantize_activation_rowwise(torch, x)
        wrapped_x = Float8Tensor(
            xq,
            xs.reshape(-1, 1),
            block_size=[1, in_features],
            dtype=x.dtype,
            mm_config=Float8MMConfig(use_fast_accum=True),
            kernel_preference=KernelPreference.TORCH,
        )
        assert wrapped_x.qdata.data_ptr() == xq.data_ptr()
        assert wrapped_x.scale.data_ptr() == xs.data_ptr()
        result["prequantized_activation_adoption"] = {
            "payload_alias": True,
            "scale_alias": True,
            "base_act_quant_kwargs": base.weight.act_quant_kwargs,
            "native_forward": "not run: W8A8 scaled_mm requires CUDA",
        }
        try:
            layer.lora_A["spatial"](wrapped_x)
        except Exception as exc:
            result["peft_bf16_adapter_cannot_consume_fp8_activation"] = (
                f"{type(exc).__name__}: {exc}"
            )
        else:
            result["peft_bf16_adapter_fp8_activation"] = (
                "accepted on CPU; requires numerical review"
            )
        try:
            base(PreQuantized(payload=xq, scale=xs, shape=tuple(x.shape)))
        except (TypeError, AttributeError) as exc:
            result["old_prequantized_dataclass_refusal"] = f"{type(exc).__name__}: {exc}"
        try:
            layer.merge(safe_merge=True)
        except ValueError as exc:
            result["merge_without_recipe_refusal"] = str(exc)
        else:
            raise AssertionError("No codec recipe was supplied; merge must not silently quantize")

    result["bf16_full_sequence_memory_bound_example"] = {
        "rows": 109104,
        "in_features": in_features,
        "out_features": out_features,
        "rank": 32,
        "factor_bytes": (in_features + out_features) * 32 * 2,
        "additional_bytes_beyond_input_and_base_output": 109104 * (2 * out_features + 32) * 2,
        "scope": "Conservative arithmetic bound for this projection, not a GPU peak measurement",
    }
    assert torch.equal(payload.view(torch.uint8), original_payload.view(torch.uint8))
    assert torch.equal(scale, original_scale)
    assert not torch.cuda.is_initialized()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
