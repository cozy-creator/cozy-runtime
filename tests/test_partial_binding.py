"""A binding that selects some of a construction's components leaves the others on meta.

Real: a TensorFS store, `PlaneBackend` and `WeightResidency` on a CPU executor, and the
production construction path (`ModelRegistry.acquire` -> `Loader.construct`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import (  # noqa: E402
    CapabilityError,
    Loader,
    Model,
    ModelRegistry,
    uses_components,
)
from cozy_runtime.author._loader import Artifact, Config, ModelFitRefused  # noqa: E402
from cozy_runtime.author._model import component_use  # noqa: E402
from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.derive import Observations, serving_substrate  # noqa: E402
from cozy_runtime.internal.fill import tensor_schema_of  # noqa: E402
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights  # noqa: E402
from test_weight_plane import store_component  # noqa: E402

pytestmark = pytest.mark.skipif(not plane.available(), reason="a TensorFS with the weight plane")

HIDDEN = 64


class Pair:
    """Builds both components whatever the binding selects, as H3's pipeline builds both DiTs."""

    def __init__(self, config: Config) -> None:
        self.components = {
            name: torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
            for name in ("first", "second")
        }


class PairModel(Model[Pair]):
    pipe: Pair

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pair, factory=Pair)

    @uses_components("first")
    def first(self, x: Any) -> Any:
        return self.pipe.components["first"](x)

    @uses_components("second")
    def second(self, x: Any) -> Any:
        return self.pipe.components["second"](x)


def bind(tmp_path: Path, component: str) -> tuple[PairModel, PlaneBackend, Any]:
    """The executor's prepare on a CPU executor, binding `component` alone."""
    value = torch.randn(HIDDEN, HIDDEN, generator=torch.Generator().manual_seed(5)) / HIDDEN**0.5
    checkpoint = store_component(tmp_path, component, {"weight": value})
    rows = checkpoint.rows(component)
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=1 << 30)
    backend = PlaneBackend.for_script(
        {component: checkpoint}, rows, weights=weights, construction="partial", release="p/1"
    )
    backend.expect({component: [row.key for row in rows]})
    seen = Observations()
    registry = ModelRegistry(
        release="p/1", backend=backend, substrate=lambda: serving_substrate("cpu", seen)
    )
    artifact = Artifact(checkpoint.manifest_id, tensor_schema_of(rows), Config({}))
    try:
        model = registry.acquire("run.models.model", PairModel, artifact)
    except Exception:
        backend.close()  # as the executor's prepare does
        raise
    residency = WeightResidency(
        weights, backend.components, component_use(PairModel), refine=backend.refine
    )
    object.__setattr__(model, "_cozy_residency", residency)
    return model, backend, value.half()


def test_an_unselected_component_stays_on_meta_and_refuses_its_scope(tmp_path: Path) -> None:
    model, backend, weight = bind(tmp_path, "first")
    record = model._cozy_record
    assert record is not None
    assert (record.components, record.skipped) == (("first",), ("second",))
    assert set(backend.components) == {"first"}
    assert model.pipe.components["second"].weight.device.type == "meta"

    x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
    with torch.inference_mode():
        assert torch.equal(model.first(x), torch.nn.functional.linear(x, weight))
        with pytest.raises(CapabilityError, match="second") as refused:
            model.second(x)
    assert refused.value.code == "component_unbound"
    assert refused.value.fields == ("second",)
    # the refusal leaves no scope open: the bound component still serves
    with torch.inference_mode():
        assert torch.equal(model.first(x), torch.nn.functional.linear(x, weight))
    backend.close()


def test_a_binding_that_selects_nothing_built_refuses_the_construction(tmp_path: Path) -> None:
    with pytest.raises(ModelFitRefused) as refused:
        bind(tmp_path, "other")
    assert refused.value.fit["code"] == "destination_absent"
    assert "first, second" in refused.value.message
