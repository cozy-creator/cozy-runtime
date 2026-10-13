"""Constructor initialization is not a load()-time weight transform (run 5387).

timm initializes in plain Python (`uniform_`, `erfinv_`, `mul_`, `add_`, `clamp_`), so each op
after the first reads the destination: before the fill inside the factory call that
registered it, and checkpoint values once that call returns, including the weight plane's own
tensors the fill installed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import Loader, Model, ModelRegistry, uses_components  # noqa: E402
from cozy_runtime.author._errors import classify  # noqa: E402
from cozy_runtime.author._loader import Artifact, Config  # noqa: E402
from cozy_runtime.author._model import component_use  # noqa: E402
from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.derive import (  # noqa: E402
    ConstructionFault,
    Observations,
    derive,
    meta_substrate,
    serving_substrate,
)
from cozy_runtime.internal.fill import tensor_schema_of  # noqa: E402
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights  # noqa: E402
from test_weight_plane import store_component  # noqa: E402

pytestmark = pytest.mark.skipif(not plane.available(), reason="a TensorFS with the weight plane")

HIDDEN = 64
ARTIFACT = Artifact("sha256:" + "0" * 64, {}, Config({}))


def timm_trunc_normal_(tensor: Any, std: float = 0.02, a: float = -2.0, b: float = 2.0) -> Any:
    """`timm.layers.weight_init.trunc_normal_`, op for op."""
    low, high = ((1 + math.erf(bound / std / math.sqrt(2))) / 2 for bound in (a, b))
    with torch.no_grad():
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2))
        tensor.add_(0.0)
        tensor.clamp_(min=a, max=b)
    return tensor


class Backbone(torch.nn.Linear):  # type: ignore[name-defined,misc]  # optional real Torch
    def __init__(self) -> None:
        super().__init__(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        timm_trunc_normal_(self.weight)


class Graph:
    def __init__(self, config: Config) -> None:
        self.components = {"net": Backbone()}


class Initialized(Model[Graph]):
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=Graph)

    @uses_components("net")
    def run(self, x: Any) -> Any:
        return self.graph.components["net"](x)


class Transformed(Initialized):
    """A load()-time transform of a filled destination."""

    def load(self, loader: Loader) -> None:
        super().load(loader)
        with torch.no_grad():
            self.graph.components["net"].weight.copy_(torch.ones(HIDDEN, HIDDEN))


class Reinitialized(Initialized):
    """A later factory call re-initializing the first construction's filled destination."""

    def load(self, loader: Loader) -> None:
        super().load(loader)
        weight = self.graph.components["net"].weight
        loader.construct(Graph, factory=lambda c: (timm_trunc_normal_(weight), Graph(c))[1])


def serve(tmp_path: Path, cls: type[Initialized] = Initialized) -> tuple[Initialized, Any, Any]:
    """The executor's prepare on a CPU executor."""
    value = torch.randn(HIDDEN, HIDDEN, generator=torch.Generator().manual_seed(5)) / HIDDEN**0.5
    checkpoint = store_component(tmp_path, "net", {"weight": value})
    rows = checkpoint.rows("net")
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=1 << 30)
    backend = PlaneBackend.for_script(
        {"net": checkpoint}, rows, weights=weights, construction="init", release="p/1"
    )
    backend.expect({"net": [row.key for row in rows]})
    registry = ModelRegistry(
        release="p/1", backend=backend, substrate=lambda: serving_substrate("cpu", Observations())
    )
    artifact = Artifact(checkpoint.manifest_id, tensor_schema_of(rows), Config({}))
    try:
        model = registry.acquire("run.models.model", cls, artifact)
    except Exception:
        backend.close()
        raise
    residency = WeightResidency(
        weights, backend.components, component_use(cls), refine=backend.refine
    )
    object.__setattr__(model, "_cozy_residency", residency)
    return model, backend, value.half()


def test_constructor_initialization_derives_and_serves_the_checkpoint(tmp_path: Path) -> None:
    derived = derive(Initialized(), ARTIFACT)
    assert [(t.key, t.shape) for t in derived.logical_tensors] == [("net.weight", (HIDDEN, HIDDEN))]

    model, backend, weight = serve(tmp_path)
    x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
    with torch.inference_mode():
        assert torch.equal(model.run(x), torch.nn.functional.linear(x, weight))
    backend.close()


@pytest.mark.parametrize("substrate", [meta_substrate, serving_substrate])
@pytest.mark.parametrize("cls", [Transformed, Reinitialized])
def test_a_write_after_its_construction_returns_stays_refused(
    substrate: Any, cls: type[Initialized]
) -> None:
    registry = ModelRegistry(release="p/1", substrate=lambda: substrate("cpu", Observations()))
    with pytest.raises(ConstructionFault, match="in-place weight transform") as refused:
        registry.acquire("run.models.model", cls, ARTIFACT, mode="derive")
    assert classify(refused.value).code == "construction_fault"


@pytest.mark.parametrize("cls", [Transformed, Reinitialized])
def test_a_write_to_a_weight_plane_tensor_is_refused(
    tmp_path: Path, cls: type[Initialized]
) -> None:
    with pytest.raises(ConstructionFault, match="in-place weight transform") as refused:
        serve(tmp_path, cls)
    assert classify(refused.value).code == "construction_fault"
