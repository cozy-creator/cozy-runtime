"""Actual PEFT arithmetic and the native prepared graph's construction census on CPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from peft.tuners.lora.layer import Linear as PeftLinear

import pytest
import tensorfs

from cozy_runtime.author import (
    AdapterCompatibility,
    AdapterRef,
    Artifact,
    Config,
    Loader,
    Model,
    ModelRegistry,
    uses_components,
)
from cozy_runtime.author._loader import census
from cozy_runtime.internal import lora_composition
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of
from cozy_runtime.internal.lora_contract import FORMAT, Graph, Linear
from native_weights import NativeExecution
from test_lora_composition import compose, fixture, header


torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from testdata.consumer_lora import armed, install, overlay_for  # noqa: E402


class Pipeline:
    def __init__(self, config: Config) -> None:
        root = torch.nn.Module()
        # The factory keeps seeing the pristine checkpoint's dtype names.
        assert config.tensor_dtype("transformer.proj.weight", default="f32") == "f32"
        root.proj = torch.nn.Linear(3, 2)
        root.untouched = torch.nn.Linear(1, 1, bias=False)
        self.components = {"transformer": root}


class TinyModel(Model[Pipeline]):
    __adapter_compatibility__ = (AdapterCompatibility("lora", "", ("transformer",)),)
    pipe: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=Pipeline)

    @uses_components("transformer")
    def component(self) -> object:
        return self.pipe.components["transformer"]


@pytest.mark.parametrize("reverse", [False, True])
def test_native_peft_graph_matches_reference_and_keeps_ordinary_destinations(
    tmp_path: Path, reverse: bool
) -> None:
    store, base, first, second = fixture(tmp_path)
    selected = [(first, 0.5), (second, -0.25)]
    if reverse:
        selected.reverse()
    result, graph = compose(store, tmp_path, "prepared", base, selected)
    checkpoint = Checkpoint.from_store(store, result.manifest.digest)
    rows = checkpoint.rows("transformer")
    document = tensorfs.parse_header(header(store, result))
    artifact = lora_composition.bind(
        Artifact(result.manifest.digest, tensor_schema_of(rows), Config({})),
        TinyModel,
        document["configs"][lora_composition.GRAPH_CONFIG],
    )
    registry = ModelRegistry(release="cpu-proof", substrate=lambda: torch.device("meta"))
    model = registry.acquire("run.models.model", TinyModel, artifact, mode="derive")
    members = model.pipe.components
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
        requirements, checkpoint.header_bytes, custody="canonical", encoded_leaves=False
    )["ok"]
    assert {row.key for row in walked.destinations} == {row.key for row in rows}
    assert all(value.is_meta for value in members["transformer"].parameters())
    assert model._cozy_adapters == graph.adapters
    assert sum(row.spec.nbytes for row in walked.destinations) == sum(row.nbytes for row in rows)

    # Real native source leases fill exactly the CPU test's already-censused destinations.
    # CUDA streaming fill/admission remains a separate integration qualification.
    members["transformer"].to_empty(device="cpu")
    with NativeExecution(store, tmp_path, "numerics", {"model": result}, {}) as execution:
        with execution.client.source(result.manifest.digest) as source:
            with torch.no_grad():
                destinations = members["transformer"].state_dict()
                for row in rows:
                    destination = destinations[row.name]
                    source.read_part_into(
                        "transformer",
                        row.name,
                        "value",
                        0,
                        destination.reshape(-1).view(torch.uint8).numpy(),
                    )
    component = members["transformer"]
    x = torch.tensor([[1.25, -2, 0.5], [0, 1, 2]])
    base_weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = torch.tensor([1.0, -1.0])
    expected = torch.nn.functional.linear(x, base_weight, bias)
    inputs = [
        (torch.tensor([[1.0, 0.0, 2.0]]), torch.tensor([[3.0], [4.0]]), 1.0),
        (
            torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float16),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16),
            -0.25,
        ),
    ]
    if reverse:
        inputs.reverse()
    for a, b, scaling in inputs:
        expected = (
            expected
            + torch.nn.functional.linear(torch.nn.functional.linear(x.to(a.dtype), a), b) * scaling
        )
    actual = component.proj(x)
    assert torch.equal(actual, expected)
    assert torch.equal(component.proj.base_layer.weight, base_weight)
    assert torch.equal(component.proj.base_layer.bias, bias)
    assert torch.equal(actual, component.proj(x))
    assert not torch.cuda.is_initialized()
    registry.unload_all()


def test_graph_cannot_bypass_a_models_adapter_opt_in(tmp_path: Path) -> None:
    store, base, first, _second = fixture(tmp_path)
    result, _graph = compose(store, tmp_path, "prepared", base, [(first, 0.5)])
    checkpoint = Checkpoint.from_store(store, result.manifest.digest)
    with pytest.raises(Exception, match="adapter_incompatible"):
        lora_composition.bind(
            Artifact(
                result.manifest.digest, tensor_schema_of(checkpoint.rows("transformer")), Config({})
            ),
            Model,
            tensorfs.parse_header(header(store, result))["configs"][lora_composition.GRAPH_CONFIG],
        )


def test_extra_peft_preserves_a_models_consumer_lora_residual() -> None:
    """A request LoRA's PEFT layer wraps a linear that a model's consumer hook adapts: the hook
    still adds inside `base_layer`, PEFT adds its own term, and neither changes the base."""
    torch.manual_seed(419)
    root = torch.nn.Sequential(torch.nn.Linear(16, 16))
    install(root, ["0"])
    overlay = overlay_for(root, ["0"], rank=1, scale=1.0)
    with torch.no_grad():
        for parameter in overlay.parameters():
            parameter.fill_(0.125)
    original = {key: value.clone() for key, value in root[0].state_dict().items()}
    graph = Graph(
        FORMAT,
        (Linear("dit", "0", "adapter_0", 1, 1.0, 0.5, "f32"),),
        (AdapterRef("sha256:" + "12" * 32, 0.5, "lora", "dit", "adapter"),),
    )
    lora_composition.composer(graph)(SimpleNamespace(components={"dit": root}))
    layer = root[0]
    with torch.no_grad():
        layer.lora_A["adapter_0"].weight.fill_(0.25)
        layer.lora_B["adapter_0"].weight.fill_(-0.5)
    x = torch.randn(7, 16)
    plain = torch.nn.functional.linear(x, original["weight"], original["bias"])
    factors = overlay.factors("0")

    def extra(layer: PeftLinear) -> Tensor:
        return layer.lora_B["adapter_0"](layer.lora_A["adapter_0"](x)) * 0.5

    with torch.no_grad():
        assert torch.equal(layer(x), plain + extra(layer))
        with armed(root, overlay):
            adapted = plain + torch.nn.functional.linear(
                torch.nn.functional.linear(x, factors.lora_down), factors.lora_up
            )
            assert torch.equal(layer(x), adapted + extra(layer))
        assert torch.equal(layer(x), plain + extra(layer))
        for key, before in original.items():
            assert torch.equal(layer.base_layer.state_dict()[key], before)
    assert not torch.cuda.is_initialized()
