"""Host-only check of the exact production semantic expression; no native plan query."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn

from cozy_runtime.internal import prebuilt
from cozy_runtime.internal.encoding.leaves import Int8ConvRotNativeLeaf
from cozy_runtime.internal.execution import phase
from cozy_runtime.internal.execution.native_linear_split import _Dispatch, _layout
from cozy_runtime.internal.execution.native_refiner_ffn import _Geometry
from cozy_runtime.internal.execution.native_residual_ffn import _Projection


@dataclass(frozen=True)
class LibraryMetadata:
    """Type-correct metadata only, explicitly not a selected GPU library."""

    identity: str = "metadata-only:unqueried-native-artifact"
    family: str = "mma.sync"


def projection(width: int, channels: int) -> _Projection:
    original = nn.Linear(width, channels, bias=False, device="meta", dtype=torch.bfloat16)
    leaf = Int8ConvRotNativeLeaf().leaf(
        torch, {"weight": torch.zeros(channels, width, dtype=torch.int8),
                "scale": torch.ones(channels, dtype=torch.float32)}, original, torch.bfloat16
    ).eval()
    return _Projection(leaf, leaf.data, leaf.scale, None)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = Path(__file__).parents[2] / "src/cozy_runtime/internal/execution/native_refiner_ffn.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "split_ffn")
    expressions = {
        name: compile(ast.Expression(next(
            node.value for node in function.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        )), str(source), "eval") for name in ("semantic", "key")
    }
    # These values exercise native result types only. Neither kind/config/workspace
    # numbers nor artifact identity are claimed to be supported or queried on a GPU.
    dispatch = _Dispatch(17, 0, 7, 4096)
    geometry = _Geometry(17, dispatch, dispatch, dispatch, dispatch, 17 * 512, 17, 4096)
    choice = geometry.identity, geometry.allocations(17, 256, 512, 2)
    up, down = projection(256, 1024), projection(512, 256)
    assert type(up.module.provider) is str and type(up.module.encoding) is str
    assert str(prebuilt.Prebuilt.__annotations__["identity"]) == "str"
    env = {
        "phase": phase, "hashlib": hashlib, "torch": torch, "_layout": _layout,
        "identities": (("metadata-only:source", "metadata-only:representation"),),
        "x": torch.zeros(1, 17, 256, dtype=torch.bfloat16), "up": up, "down": down,
        "library": LibraryMetadata(), "choices": [choice],
        "geometries": {geometry.identity: geometry},
    }

    def encode(values):
        semantic = eval(expressions["semantic"], values)
        if semantic is None:
            return None
        return eval(expressions["key"], {**values, "semantic": semantic})

    key = encode(env)
    assert key is not None and key.startswith("runtime.split.refiner_ffn/1:")
    changed_geometry = replace(geometry, up=replace(dispatch, config=dispatch.config + 1))
    mutations = {
        "selected_dispatch_geometry": {
            "geometries": {changed_geometry.identity: changed_geometry},
            "choices": [(changed_geometry.identity, choice[1])],
        },
        "input_layout": {"x": env["x"][:, ::2]},
        "library_identity": {"library": replace(LibraryMetadata(), identity="metadata-only:other")},
        "allocation_menu": {"choices": [(choice[0], (*choice[1], 512))]},
    }
    changes = {name: encode({**env, **mutation}) for name, mutation in mutations.items()}
    original = up.module.provider
    up.module.provider = original + ":semantic-mutation"
    changes["provider_identity"] = encode(env)
    up.module.provider = original
    assert all(value is not None and value != key for value in changes.values())
    assert encode(env) == key
    # Freeze the complete menu, not only its outer collection. Global strictness stays.
    assert encode({**env, "choices": [(choice[0], list(choice[1]))]}) is None
    assert encode({**env, "geometries": {geometry.identity: asdict(geometry)}}) is None
    assert phase.configuration({"mutable": 1}) is None
    assert phase.configuration(["mutable"]) is None
    assert not torch.cuda.is_initialized()
    assert "libcuda.so" not in Path("/proc/self/maps").read_text()
    print(json.dumps({
        "passed": True, "scope": "exact production semantic expression, host metadata only",
        "native_plan_queried": False, "gpu_plan_qualification": False,
        "real_cpu_tensor_layouts_and_native_leaf_provider_types": True,
        "dispatch_values": "explicit metadata-only values; no supported GPU plan claimed",
        "key": key, "mutation_keys": changes, "global_mutable_metadata_rejection": True,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cuda_initialized": False, "libcuda_mapped": False,
    }, indent=2))


if __name__ == "__main__":
    main()
