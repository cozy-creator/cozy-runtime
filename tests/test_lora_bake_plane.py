"""A prepared LoRA graph on the weight plane: a GPU executor bakes each LoRA'd linear into its
bytes when they are placed (the default); a CPU executor and a request's per-step mode compute
the update every forward. The checkpoint's stored bytes never change."""

from __future__ import annotations

import math
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
    checkpoint = Checkpoint.from_store(store, result.manifest.digest)
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

        def served() -> tuple[Any, bool, bool]:
            """The output, and whether the layer skipped its update over baked bytes, read
            while the stage holds the weights (a release may evict them)."""
            residency.admit("run", ("transformer",))
            try:
                with torch.no_grad():
                    return root.proj(x), root.proj._cozy_baked, region.baked
            finally:
                residency.release("run", ("transformer",))

        # f32 weights and small integers: baked and per-step sums are both exact here
        cuda = device == "cuda"
        before = weights.facts().lora_baked
        out, skipped, baked = served()
        assert torch.equal(out, expected(x)) and (skipped, baked) == (cuda, cuda)
        facts = weights.facts()  # the invoke reply says what was baked
        assert facts.lora_baked - before == (1 if cuda else 0)
        assert (facts.lora_bake_ms > 0) is cuda or before > 0
        if cuda:
            weights.set_lora("per_step")  # drops the baked bytes; they refill unmodified
            assert (region.bake, region.baked) == (False, False)
            out, skipped, baked = served()
            assert torch.equal(out, expected(x)) and (skipped, baked) == (False, False)
            weights.set_lora("bake")
            out, skipped, baked = served()
            assert torch.equal(out, expected(x)) and (skipped, baked) == (True, True)
    finally:
        backend.close()


WIDTH, RANK = 512, 16


def _encoded_fixture(tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    """An fp8-rowwise base projection and one LoRA whose delta is far below E4M3's step."""
    import numpy as np
    from tensorfs.derived import Config as NativeConfig
    from tensorfs.derived import Derivation, Part, Target, Tensor

    from cozy_runtime.derive.microscale import encode_fp8_rowwise
    from cozy_runtime.internal.encoding import SPEC_PLAIN, SPEC_ROWWISE
    from native_weights import NativeExecution
    from test_lora_composition import Value, model

    rng = np.random.default_rng(26)
    weight = rng.normal(0, 0.05, (WIDTH, WIDTH)).astype(np.float32)
    encoded = encode_fp8_rowwise("proj", weight)
    store = tensorfs.Store.init(tmp_path / "store")
    tensors = {
        "proj.weight": Tensor(
            "f32",
            (WIDTH, WIDTH),
            SPEC_ROWWISE,
            {"data": Part("f8_e4m3fn", (WIDTH, WIDTH)), "scale": Part("f32", (WIDTH,))},
        ),
        "proj.bias": Tensor("f32", (WIDTH,), SPEC_PLAIN, {"value": Part("f32", (WIDTH,))}),
    }
    declaration = Derivation(
        {},
        {"transformer": Target(add=tensors)},
        {"model": NativeConfig("add")},
        tuple(("transformer", key) for key in tensors),
        files={"LICENSE": b"Fixture weights\n"},
    )
    with (
        NativeExecution(store, tmp_path, "encoded", {}, {"model": 1 << 22}) as execution,
        execution.client.open_output("model", declaration) as writer,
    ):
        writer.add_part("transformer", "proj.weight", "data", encoded.payload)
        writer.add_part("transformer", "proj.weight", "scale", encoded.companions[0].raw)
        writer.add_part("transformer", "proj.bias", "value", bytes(4 * WIDTH))
        writer.add_config("model", b"{}")
        base = execution.client.adopt_model(writer.commit())
    down = rng.normal(0, 0.02, (RANK, WIDTH)).astype(np.float32)
    up = rng.normal(0, 0.02, (WIDTH, RANK)).astype(np.float32)
    adapter = model(
        store,
        tmp_path,
        "adapter",
        "adapter",
        {
            "proj.lora_A.weight": Value((RANK, WIDTH), tuple(down.flatten().tolist())),
            "proj.lora_B.weight": Value((WIDTH, RANK), tuple(up.flatten().tolist())),
        },
    )
    return store, base, adapter, (torch.from_numpy(down), torch.from_numpy(up))


@pytest.mark.skipif("cuda" not in DEVICES, reason="the weight plane on a CUDA card")
def test_a_baked_fp8_projection_keeps_the_loras_effect(tmp_path: Path) -> None:
    from cozy_runtime.author import AdapterCompatibility, Model, uses_components

    class EncodedPipeline:
        def __init__(self, config: Config) -> None:
            with torch.device("meta"):
                root = torch.nn.Module()
                root.proj = torch.nn.Linear(WIDTH, WIDTH)
            self.components = {"transformer": root}

    class EncodedModel(Model[EncodedPipeline], encoded_leaves="accept"):
        __adapter_compatibility__ = (
            AdapterCompatibility("lora", "", ("transformer",), -math.inf, math.inf),
        )
        pipe: EncodedPipeline

        def load(self, loader: Loader) -> None:
            self.pipe = loader.construct(EncodedPipeline, factory=EncodedPipeline)

        @uses_components("transformer")
        def component(self) -> object:
            return self.pipe.components["transformer"]

    store, base, adapter, (down, up) = _encoded_fixture(tmp_path)
    strength = 3.0
    result, _ = compose(store, tmp_path, "encoded-view", base, [(adapter, strength)])
    checkpoint = Checkpoint.from_store(store, result.manifest.digest)
    rows = checkpoint.rows("transformer")
    configs = tensorfs.parse_header(header(store, result))["configs"]
    artifact = lora_composition.bind(
        Artifact(result.manifest.digest, tensor_schema_of(rows), Config({})),
        EncodedModel,
        configs[lora_composition.GRAPH_CONFIG],
    )
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=64 << 20)
    backend = PlaneBackend.for_script(
        {"transformer": checkpoint}, rows, weights=weights, construction="fp8", release="fp8/1"
    )
    try:
        backend.expect({"transformer": tuple(row.key for row in rows)})
        owner = type("Owner", (), {"__encoded_leaves__": "accept"})()
        built = Loader(artifact, owner=owner, backend=backend).construct(
            EncodedPipeline, factory=EncodedPipeline
        )
        root: Any = built.components["transformer"]
        (region,) = [r for r in backend.components["transformer"].regions if r.bakes]
        x = torch.randn(64, WIDTH, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        residency = WeightResidency(weights, backend.components, {"run": ("transformer",)})

        def served() -> tuple[Any, bool, bool]:
            residency.admit("run", ("transformer",))
            try:
                with torch.no_grad():
                    return root.proj(x).float(), root.proj._cozy_baked, region.baked
            finally:
                residency.release("run", ("transformer",))

        baked, skipped, held = served()
        assert (skipped, held) == (True, True)
        weights.set_lora("per_step")
        per_step, skipped, held = served()
        assert (skipped, held) == (False, False)
        update = (x @ down.cuda().t() @ up.cuda().t()) * strength
        none = per_step - update
        effect = (baked - none).flatten() @ update.flatten() / update.flatten().square().sum()
        assert 0.8 < float(effect) < 1.2, float(effect)
    finally:
        backend.close()
