"""Native multipart custody, meta census and actual CPU encoded-GEMM adapter arithmetic.

This does not exercise the weight plane's device binding.
"""

from pathlib import Path

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Source, Target, Tensor
from test_lora_composition import compose, fixture, header

from cozy_runtime.author import Artifact, Config, ModelRegistry
from cozy_runtime.author._loader import census
from cozy_runtime.internal import lora_composition
from cozy_runtime.internal.encoding import SPEC_ROWWISE
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of
from native_weights import NativeExecution


def test_native_encoded_base_keeps_parts_and_matches_cpu_peft_arithmetic(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from test_lora_peft import TinyModel

    from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, quantize_activation_rowwise

    class EncodedTinyModel(TinyModel, encoded_leaves="accept"):
        pass

    store, pristine, first, second = fixture(tmp_path)
    original_header = header(store, pristine)
    payload = torch.tensor([[8, 16, 24], [16, 20, 24]]).to(torch.float8_e4m3fn)
    scales = torch.tensor([0.125, 0.25], dtype=torch.float32)
    declaration = Derivation(
        {"base": Source(pristine.manifest.digest, pristine.manifest.length)},
        {
            "transformer": Target(
                "base",
                "transformer",
                ("proj.weight",),
                {
                    "proj.weight": Tensor(
                        "f32",
                        (2, 3),
                        SPEC_ROWWISE,
                        {"data": Part("f8_e4m3fn", (2, 3)), "scale": Part("f32", (2,))},
                    )
                },
            )
        },
        {},
        tuple(("transformer", name) for name in ("proj.weight", "proj.bias", "untouched.weight")),
    )
    with (
        NativeExecution(
            store, tmp_path, "encode", {"base": pristine}, {"model": 1 << 20}
        ) as execution,
        execution.client.open_output("model", declaration) as writer,
    ):
        writer.add_part("transformer", "proj.weight", "data", payload.view(torch.uint8).numpy())
        writer.add_part("transformer", "proj.weight", "scale", scales.view(torch.uint8).numpy())
        encoded = execution.client.adopt_model(writer.commit())

    encoded_header = header(store, encoded)
    selected, graph = compose(store, tmp_path, "adapt", encoded, [(first, 0.5), (second, -0.25)])
    before = tensorfs.parse_header(encoded_header)["components"]["transformer"]["proj.weight"]
    after = tensorfs.parse_header(header(store, selected))["components"]["transformer"]
    assert after["proj.base_layer.weight"] == before
    assert set(before["parts"]) == {"data", "scale"}
    assert header(store, encoded) == encoded_header
    assert header(store, pristine) == original_header
    checkpoint = Checkpoint(store.root, selected.manifest.digest)
    rows = checkpoint.rows("transformer")
    artifact = lora_composition.bind(
        Artifact(selected.manifest.digest, tensor_schema_of(rows), Config({})),
        EncodedTinyModel,
        tensorfs.parse_header(header(store, selected))["configs"][lora_composition.GRAPH_CONFIG],
    )
    registry = ModelRegistry(release="encoded-cpu-proof", substrate=lambda: torch.device("meta"))
    model = registry.acquire("run.models.model", EncodedTinyModel, artifact, mode="derive")
    walked = census(model.pipe)
    requirements = [
        tensorfs.TensorRequirement(
            component=row.component,
            key=row.key.removeprefix(row.component + "."),
            logical_dtype=row.spec.dtype,
            shape=list(row.spec.shape),
        )
        for row in walked.destinations
    ]
    assert tensorfs.fit(
        requirements, checkpoint.header_bytes, custody="canonical", encoded_leaves=True
    )["ok"]
    assert {row.key for row in walked.destinations} == {row.key for row in rows}
    assert all(parameter.is_meta for parameter in model.pipe.components["transformer"].parameters())
    assert model._cozy_adapters == graph.adapters

    # Native source reads supply the exact encoded physical role buffers. The leaf is
    # constructed directly for CPU arithmetic; accelerator fill remains a separate gate.
    root = model.pipe.components["transformer"]
    root.to_empty(device="cpu")
    parts = {"data": torch.empty_like(payload), "scale": torch.empty_like(scales)}
    with (
        NativeExecution(store, tmp_path, "readback", {"model": selected}, {}) as execution,
        execution.client.source(selected.manifest.digest) as source,
        torch.no_grad(),
    ):
        for row in rows:
            if row.name == "proj.base_layer.weight":
                for role, destination in parts.items():
                    source.read_part_into(
                        "transformer", row.name, role, 0, destination.view(torch.uint8).numpy()
                    )
            else:
                destination = root.state_dict()[row.name]
                source.read_part_into(
                    "transformer",
                    row.name,
                    "value",
                    0,
                    destination.reshape(-1).view(torch.uint8).numpy(),
                )
    assert torch.equal(parts["data"].view(torch.uint8), payload.view(torch.uint8))
    assert torch.equal(parts["scale"], scales)
    native = RowwiseNativeLeaf("fp8-rowwise/1").leaf(
        torch, parts, root.proj.base_layer, torch.float32
    )
    assert native.roles()["data"] is parts["data"]
    assert native.roles()["scale"] is parts["scale"]
    assert not tuple(native.parameters())
    root.proj.base_layer = native
    x = torch.tensor([[1.25, -2, 0.5], [0, 1, 2]])
    quantized, activation_scale = quantize_activation_rowwise(torch, x)
    expected = (quantized.float() * activation_scale) @ (payload.float() * scales[:, None]).t()
    expected += torch.tensor([1.0, -1.0])
    for a, b, scale in (
        (torch.tensor([[1.0, 0.0, 2.0]]), torch.tensor([[3.0], [4.0]]), 1.0),
        (
            torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float16),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16),
            -0.25,
        ),
    ):
        expected += (
            torch.nn.functional.linear(torch.nn.functional.linear(x.to(a.dtype), a), b) * scale
        )
    torch.testing.assert_close(root.proj(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(root.proj(x), expected, rtol=0, atol=0)
    assert torch.equal(native.data.view(torch.uint8), payload.view(torch.uint8))
    assert torch.equal(native.scale, scales)
    assert header(store, encoded) == encoded_header
    assert not torch.cuda.is_initialized()
    registry.unload_all()
