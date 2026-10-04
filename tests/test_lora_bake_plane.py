"""A prepared LoRA graph on the weight plane: a GPU executor bakes each LoRA'd linear into its
resident bytes when they are placed; a CPU executor and the per-step strategy compute the
update every forward. The checkpoint's stored bytes never change."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tensorfs

from cozy_runtime.author import Artifact, Config, Loader
from cozy_runtime.internal import accel, lora_composition, plane
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights
from test_lora_composition import compose, fixture, header
from test_lora_peft import TinyModel

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

DEVICES = ["cpu"] + (["cuda"] if accel.present(torch, "cuda") and plane.available() else [])


class Pipeline:
    def __init__(self, config: Config) -> None:
        with torch.device("meta"):
            root = torch.nn.Module()
            root.proj = torch.nn.Linear(3, 2)
            root.untouched = torch.nn.Linear(1, 1, bias=False)
        self.components = {"transformer": root}


def expected(x: Any) -> Any:
    """The fixture's base projection with its two adapters (strengths 0.5 and -0.25)."""
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=x.device)
    out = torch.nn.functional.linear(x, weight, torch.tensor([1.0, -1.0], device=x.device))
    terms = (
        ([[1.0, 0.0, 2.0]], [[3.0], [4.0]], 0.5 * 2.0 / 1),
        ([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]], -0.25 * 2.0 / 2),
    )
    for a, b, scale in terms:
        down, up = torch.tensor(a, device=x.device), torch.tensor(b, device=x.device)
        out = out + x @ down.t() @ up.t() * scale
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_a_gpu_executor_bakes_a_prepared_graph_and_per_step_serves_it_unbaked(
    tmp_path: Path, device: str
) -> None:
    store, base, first, second = fixture(tmp_path)
    result, _ = compose(store, tmp_path, "baked", base, [(first, 0.5), (second, -0.25)])
    checkpoint = Checkpoint(store.root, result.manifest.digest)
    rows = checkpoint.rows("transformer")
    configs = tensorfs.parse_header(header(store, result))["configs"]
    artifact = lora_composition.bind(
        Artifact(result.manifest.digest, tensor_schema_of(rows), Config({})),
        TinyModel,
        configs[lora_composition.GRAPH_CONFIG],
    )
    weights = Weights(
        torch, torch.device(device, 0) if device == "cuda" else torch.device(device), device
    )
    weights.set_budget(-1, pinned=64 << 20)
    backend = PlaneBackend.for_script(
        {"transformer": checkpoint}, rows, weights=weights, construction="baked", release="baked/1"
    )
    try:
        backend.expect({"transformer": tuple(row.key for row in rows)})
        model = Loader(artifact, owner=SimpleNamespace(), backend=backend).construct(
            Pipeline, factory=Pipeline
        )
        root: Any = model.components["transformer"]
        (region,) = [r for r in backend.components["transformer"].regions if r.bakes]
        assert [len(bake.factors) for bake in region.bakes] == [2]
        assert region.bake is (device == "cuda")
        x = torch.tensor([[1.25, -2.0, 0.5], [0.0, 1.0, 2.0]], device=device)
        residency = WeightResidency(weights, backend.components, {"run": ("transformer",)})

        def served() -> Any:
            residency.admit("run", ("transformer",))
            try:
                with torch.no_grad():
                    return root.proj(x)
            finally:
                residency.release("run", ("transformer",))

        # f32 weights and small integers: baked and per-step sums are both exact here
        assert torch.equal(served(), expected(x))
        assert root.proj._cozy_baked is (device == "cuda")
        if device == "cuda":
            assert (region.bake, region.baked) == (True, True)
            weights.set_lora("per_step")  # drops the baked bytes; they refill unmodified
            assert (region.bake, region.baked) == (False, False)
            assert torch.equal(served(), expected(x))
            assert root.proj._cozy_baked is False
            weights.set_lora("bake")
            assert torch.equal(served(), expected(x))
            assert (region.bake, region.baked) == (True, True)
    finally:
        backend.close()
