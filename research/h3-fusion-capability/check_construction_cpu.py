"""Exercise every GPU-fixture constructor at its actual widths, without CUDA."""

from __future__ import annotations

import gc
import inspect
import json
from importlib.metadata import version

import torch
from h3_fusion_capability import compile_architecture, construct_case

from cozy_runtime.internal import fusion, fusion_install
from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf
from cozy_runtime.models.minimax_h3.turbo import _LoRAHook

print(
    json.dumps(
        {
            "runtime": version("cozy-runtime"),
            "runtime_path": fusion_install.__file__,
            "diffusers": version("diffusers"),
            "torch": torch.__version__,
            "cuda_visible": torch.cuda.is_available(),
            "constructor_signature": str(inspect.signature(RowwiseNativeLeaf)),
            "leaf_signature": str(inspect.signature(RowwiseNativeLeaf.leaf)),
        }
    ),
    flush=True,
)
assert not torch.cuda.is_available()
assert version("cozy-runtime") == "0.19.0+dev.fusioncap.165e4d7b"
fusion_install._require_capabilities()
assert type(compile_architecture((12, 0))) is int
inspect.signature(fusion.expect).bind(compile_architecture((12, 0)))

with torch.inference_mode():
    for mode in ("bf16", "encoded", "encoded_inactive", "encoded_baked", "encoded_unbaked"):
        case = construct_case(mode, torch.device("cpu"))
        assert tuple(case.hidden.shape) == (1, 129, 5376)
        assert tuple(case.temb.shape) == (2, 128)
        assert tuple(case.rotary[0].shape) == (129, 96)
        fusion_install._validate_transformer(case.model, "fl2va_dit")
        fusion_install._tensor_census(case.model)
        inspect.signature(case.block.forward).bind(
            case.hidden, case.temb, case.indices, case.rotary
        )
        tables = case.block.adaln_proj(case.temb)
        assert len(tables) == 6 and all(tuple(t.shape) == (6, 5376) for t in tables)
        encoded = hooks = 0
        for parent, name in case.slots:
            leaf = getattr(parent, name)
            inspect.signature(leaf.forward).bind(
                torch.empty(1, 129, leaf.in_features, dtype=torch.bfloat16)
            )
            if mode != "bf16":
                assert leaf.encoding == "fp8-rowwise/1"
                assert leaf.provider == "cozy.fp8-rowwise.native-leaf/1"
                assert leaf.out_dtype == torch.bfloat16
                assert leaf.data.dtype == torch.float8_e4m3fn
                assert tuple(leaf.data.shape) == (leaf.out_features, leaf.in_features)
                assert leaf.scale.numel() == leaf.out_features
                encoded += 1
            values = list(leaf._forward_hooks.values())
            if mode.startswith("encoded_"):
                assert len(values) == 1 and type(values[0]) is _LoRAHook
                assert values[0]._cozy_baked == (mode == "encoded_baked")
                assert (values[0].owner._arming is None) == (mode == "encoded_inactive")
                hooks += 1
            else:
                assert not values
        assert encoded == (0 if mode == "bf16" else 5)
        assert hooks == (5 if mode.startswith("encoded_") else 0)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "actual_hidden_width": 5376,
                    "actual_ffn_width": 14336,
                    "constructed_encoded_leaves": encoded,
                    "actual_lora_hooks": hooks,
                    "all_helper_arguments_valid": True,
                    "topology_valid": True,
                    "full_width_constructor_pass": True,
                }
            ),
            flush=True,
        )
        del case, tables, leaf, parent, values
        gc.collect()
print(json.dumps({"all_five_passed": True, "gpu_forward_executed": False}), flush=True)
