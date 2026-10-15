"""Baked LoRA bytes are a variant of a layout the machine keeps beside its stored bytes
(`weights.attach/1`): a replacement executor maps them, copies nothing to the GPU and bakes
nothing. Real planes, the executor's own exchange, and the custody stand-in of
`test_device_tier`. A prepared PEFT graph, and a model's consumer overlay through its hooks."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

import tensorfs  # noqa: E402

from cozy_runtime.author import Artifact, Config, Loader  # noqa: E402
from cozy_runtime.internal import accel, lora_composition  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of  # noqa: E402
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights  # noqa: E402
from test_device_tier import Custody, executor_with  # noqa: E402
from test_lora_bake_plane import Pipeline, expected  # noqa: E402
from test_lora_composition import Value, compose, fixture, model  # noqa: E402
from test_lora_consumer_bake import _dit, _inputs, _overlay  # noqa: E402
from test_lora_peft import TinyModel  # noqa: E402
from testdata.consumer_lora import armed  # noqa: E402

pytestmark = pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card")


def construct(weights: Weights, checkpoint: Checkpoint, name: str, pipeline: Any) -> Any:
    """One artifact's component on `weights`, as a load does: (backend, module)."""
    rows = checkpoint.rows(name)
    backend = PlaneBackend.for_script(
        {name: checkpoint}, rows, weights=weights, construction=name, release="variant/1"
    )
    backend.expect({name: tuple(row.key for row in rows)})
    artifact = Artifact(checkpoint.manifest_id, tensor_schema_of(rows), Config({}))
    if pipeline is Pipeline:
        graph = checkpoint.header["configs"][lora_composition.GRAPH_CONFIG]
        artifact = lora_composition.bind(artifact, TinyModel, graph)
    built = Loader(artifact, owner=SimpleNamespace(), backend=backend).construct(
        pipeline, factory=pipeline
    )
    return backend, built.components[name]


def test_a_replacement_attaches_a_baked_graph_and_never_bakes(tmp_path: Path) -> None:
    """The first executor bakes its LoRA'd projection and offers the region under its variant;
    a replacement maps it, bakes nothing and computes the same. Its bytes equal an executor's
    own bake, byte for byte. Revoked, it bakes its own. A machine that keeps no variants is
    never offered baked bytes."""
    store, base, first, second = fixture(tmp_path)
    result, _ = compose(store, tmp_path, "baked", base, [(first, 0.5), (second, -0.25)])
    checkpoint = Checkpoint.from_store(store, result.manifest.digest)
    x = torch.tensor([[1.25, -2.0, 0.5], [0.0, 1.0, 2.0]], device="cuda")
    backends: list[Any] = []

    def served(weights: Weights) -> Any:
        """Construct on `weights` and return a stage runner: (output, the projection's device
        bytes while the stage holds them, whether its update skipped)."""
        backend, root = construct(weights, checkpoint, "transformer", Pipeline)
        backends.append(backend)
        (region,) = [r for r in backend.components["transformer"].regions if r.bakes]
        residency = WeightResidency(weights, backend.components, {"run": ("transformer",)})

        def run() -> tuple[Any, Any, bool]:
            residency.admit("run", ("transformer",))
            try:
                with torch.no_grad():
                    out = root.proj(x)
                    held = root.proj.weight.detach().clone().view(torch.uint8)
                    return out, held, bool(root.proj._cozy_baked)
            finally:
                residency.release("run", ("transformer",))

        return SimpleNamespace(region=region, run=run, attached=region.attached)

    custody = Custody()
    older = Custody(variants=False)
    try:
        filler, _ = executor_with(custody, tmp_path / "a")
        if not filler.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        a = served(filler)
        before = filler.facts()
        out, baked, skipped = a.run()
        assert torch.equal(out, expected(x)) and skipped and a.attached == ""
        assert filler.facts().lora_baked - before.lora_baked == 1 and filler.variants
        variant, index = a.region.variant, a.region.index
        layout = backends[0].components["transformer"].ws.digest
        assert variant.startswith("sha256:") and custody.regions(layout, variant) == [index]
        assert index not in custody.regions(layout, ""), "a holding of its own"

        replacement, _ = executor_with(custody, tmp_path / "b")
        b = served(replacement)
        assert b.attached == variant, "its region arrived baked"
        before = replacement.facts()
        out, mapped, skipped = b.run()
        assert torch.equal(out, expected(x)) and skipped and torch.equal(mapped, baked)
        facts = replacement.facts()
        assert facts.h2d_bytes == 0, "attached weights are never copied"
        assert facts.lora_baked == before.lora_baked
        assert facts.lora_attached - before.lora_attached == 1

        # Revoked: the next stage fills the stored bytes and bakes its own, the same bytes.
        assert replacement.revoke(layout) > 0
        out, own, skipped = b.run()
        assert torch.equal(out, expected(x)) and skipped and torch.equal(own, baked)
        assert replacement.facts().lora_baked - before.lora_baked == 1

        # A machine that keeps no variants (an older one) is offered stored bytes only.
        solo, _ = executor_with(older, tmp_path / "c")
        c = served(solo)
        out, _, skipped = c.run()
        assert torch.equal(out, expected(x)) and skipped and not solo.variants
        assert set(older.held) <= {(layout, "")} and index not in older.regions(layout, "")
    finally:
        for backend in backends:
            backend.close()
        custody.close()
        older.close()


def _stored(store: Any, root: Path, name: str, module: Any) -> Checkpoint:
    tensors = {
        key: Value(tuple(value.shape), tuple(value.float().flatten().tolist()))
        for key, value in module.state_dict().items()
    }
    return Checkpoint.from_store(
        store, model(store, root, name, name, tensors, b"{}").manifest.digest
    )


def test_a_replacement_attaches_blocks_baked_with_a_consumer_overlay(tmp_path: Path) -> None:
    """A turbo overlay's shape: a weight set of its own whose factors the DiT's hooks bake
    into the DiT's blocks. Those blocks are named by the overlay's layout and scale, so the
    replacement maps them baked; a base forward refills them with the stored bytes."""
    torch.manual_seed(26)
    reference, overlay = _dit(), _overlay("cpu")
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, 0.1)
    store = tensorfs.Store.init(tmp_path / "store")
    dit_at = _stored(store, tmp_path, "fl2va_dit", reference)
    lora_at = _stored(store, tmp_path, "fl2va_overlay", overlay)
    reference, overlay = reference.to("cuda"), overlay.to("cuda")
    inputs = _inputs("cuda")
    backends: list[Any] = []

    class Dit:
        def __init__(self, config: Config) -> None:
            self.components = {"fl2va_dit": _dit()}

    class Lora:
        def __init__(self, config: Config) -> None:
            self.components = {"fl2va_overlay": _overlay("cpu")}

    def forward(model: Any, adapter: Any) -> list[Any]:
        with torch.no_grad(), nullcontext() if adapter is None else armed(model, adapter):
            return [t.float() for t in model(**inputs)]

    def served(weights: Weights) -> Any:
        dit_backend, dit = construct(weights, dit_at, "fl2va_dit", Dit)
        lora_backend, lora = construct(weights, lora_at, "fl2va_overlay", Lora)
        backends.extend([dit_backend, lora_backend])
        names = ("fl2va_dit", "fl2va_overlay")
        components = {**dit_backend.components, **lora_backend.components}
        residency = WeightResidency(weights, components, {"run": names})
        blocks = [r for r in components["fl2va_dit"].regions if r.bakes]

        def run(turbo: bool) -> list[Any]:
            residency.admit("run", names)
            try:
                return forward(dit, lora if turbo else None)
            finally:
                residency.release("run", names)

        return SimpleNamespace(run=run, blocks=blocks, layout=components["fl2va_dit"].ws.digest)

    turbo_ref = forward(reference, overlay)
    base_ref = forward(reference, None)
    custody = Custody()
    try:
        filler, _ = executor_with(custody, tmp_path / "a")
        if not filler.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        a = served(filler)
        filler.variants = False  # as when sharing began after its load: its first offer tells
        before = filler.facts()
        turbo = a.run(True)
        for got, want in zip(turbo, turbo_ref, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        weights_baked = filler.facts().lora_baked - before.lora_baked
        (variant,) = {region.variant for region in a.blocks}
        assert weights_baked == 18 and variant.startswith("sha256:")
        assert custody.regions(a.layout, variant) == [region.index for region in a.blocks]

        replacement, _ = executor_with(custody, tmp_path / "b")
        b = served(replacement)
        assert {region.attached for region in b.blocks} == {variant}
        before = replacement.facts()
        for got, want in zip(b.run(True), turbo, strict=True):
            assert torch.equal(got, want)
        facts = replacement.facts()
        assert facts.h2d_bytes == 0, "attached weights are never copied"
        assert facts.lora_baked == before.lora_baked
        assert facts.lora_attached - before.lora_attached == weights_baked

        # A base forward does not want the overlay's delta: its blocks fill privately again.
        for got, want in zip(b.run(False), base_ref, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        assert replacement.facts().h2d_bytes > 0
        for got, want in zip(b.run(True), turbo, strict=True):
            assert torch.equal(got, want)
        assert replacement.facts().lora_baked - before.lora_baked == weights_baked
    finally:
        for backend in backends:
            backend.close()
        custody.close()
