"""The executor's weights on the real weight plane (weight-plane.md §4).

Real: a TensorFS store holding a block model's weights, `tensorfs.plane`, the runtime's
`PlaneBackend` and `WeightResidency`, torch. The model is small and exact: a stack of
declared no-split blocks whose forward reads every weight, so any byte streamed wrong changes
the output. The host tier runs without a card; the rest needs device 0 and holds the whole
card, so run it under the shared GPU lock.
"""

from __future__ import annotations

import io
import mmap
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tensorfs import Store  # noqa: E402
from tensorfs.derived import Config as NativeConfig  # noqa: E402
from tensorfs.derived import Derivation, Part, Target, Tensor  # noqa: E402

from cozy_runtime.author._loader import census  # noqa: E402
from cozy_runtime.internal import accel, plane, weight_policy  # noqa: E402
from cozy_runtime.internal.derive import Observations, serving_substrate  # noqa: E402
from cozy_runtime.internal.encoding import SPEC_PLAIN  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.weights import (  # noqa: E402
    HostTiers,
    PlaneBackend,
    WeightResidency,
    Weights,
)
from cozy_runtime.internal.weights_sink import weights_transaction_id  # noqa: E402

pytestmark = pytest.mark.skipif(not plane.available(), reason="a TensorFS with the weight plane")
on_card = pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card")

HIDDEN = 2048
BLOCKS = 8


class Block(torch.nn.Module):  # type: ignore[misc,name-defined]
    """16 MiB and a little: at the sub-block grain its two projections are separate regions
    and its own `norm` joins `common`."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.out = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.norm = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))

    def forward(self, x: Any) -> Any:
        return x + torch.tanh(self.out(torch.tanh(self.proj(x * self.norm))))


class Stack(torch.nn.Module):  # type: ignore[misc,name-defined]
    _no_split_modules = ("Block",)

    def __init__(self, reverse: bool = False) -> None:
        super().__init__()
        self.embed = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))
        self.blocks = torch.nn.ModuleList([Block() for _ in range(BLOCKS)])
        self.reverse = reverse

    def forward(self, x: Any) -> Any:
        x = x * self.embed
        for block in reversed(self.blocks) if self.reverse else self.blocks:
            x = block(x)
        return x


def write_checkpoint(root: Path) -> tuple[Checkpoint, dict[str, Any]]:
    """Sane fp16 weights (a fixed seed, small values) in a real TensorFS store."""
    generator = torch.Generator().manual_seed(7)
    values: dict[str, Any] = {"embed": torch.rand(HIDDEN, generator=generator) + 0.5}
    for index in range(BLOCKS):
        for name in ("proj", "out"):
            values[f"blocks.{index}.{name}.weight"] = (
                torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
            )
        values[f"blocks.{index}.norm"] = torch.rand(HIDDEN, generator=generator) + 0.5
    return store_component(root, "stack", values), {k: v.half() for k, v in values.items()}


def store_component(root: Path, name: str, values: dict[str, Any]) -> Checkpoint:
    """`values` as one component's fp16 tensors in a real TensorFS store."""
    tensors = {
        key: Tensor("f16", tuple(value.shape), SPEC_PLAIN, {"value": Part("f16", value.shape)})
        for key, value in values.items()
    }
    store = Store.ensure(str(root / "store"))
    writer = store.begin_derived(
        weights_transaction_id("plane-test", "create", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={name: Target(add=tensors)},
            configs={name: NativeConfig("add")},
            order=tuple((name, key) for key in values),
        ).native_arguments(1 << 30),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for key, value in values.items():
        writer.add_part(name, key, "value", io.BytesIO(value.half().numpy().tobytes()))
    writer.add_config(name, io.BytesIO(b"{}"))
    receipt = writer.commit()
    return Checkpoint(root / "store", "sha256:" + receipt["manifest"]["sha256"])


def load(
    weights: Weights,
    checkpoint: Checkpoint,
    key: str,
    *,
    reverse: bool = False,
    tiers: HostTiers | None = None,
    name: str = "stack",
    factory: Any = None,
    scope: str = "run",
) -> Any:
    """Construct on meta and register with the plane, as the executor's prepare does."""
    backend = PlaneBackend.for_script(
        {name: checkpoint},
        checkpoint.rows(name),
        weights=weights,
        construction=key,
        release="plane-test/1",
        tiers=tiers,
    )
    # the serving substrate: parameters weightless, buffers no checkpoint supplies real
    major, minor = (0, 0) if weights.host_only else torch.cuda.get_device_capability(0)
    variant = "cpu" if weights.host_only else f"sm{major}{minor}"
    with serving_substrate(variant, Observations()):
        module = factory() if factory is not None else Stack(reverse=reverse)
    holder = SimpleNamespace(components={name: module})
    walked = census(holder)
    backend.expect({name: [d.key for d in walked.destinations]})
    live = backend.materialize(holder, walked)
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    residency = WeightResidency(
        weights, backend.components, {scope: (name,)}, refine=backend.refine
    )
    return SimpleNamespace(stack=module, residency=residency, backend=backend)


def run(model: Any, x: Any, passes: int = 3) -> list[Any]:
    model.residency.admit("run", ("stack",))
    try:
        with torch.inference_mode():
            return [model.stack(x) for _ in range(passes)]
    finally:
        model.residency.release("run", ("stack",))


def same(a: list[Any], b: list[Any]) -> bool:
    return all(
        torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b, strict=True)
    )


def tight(model: Any) -> int:
    """The floor at a window of two, plus one resident block: the rest must stream."""
    layout: weight_policy.Layout = model.residency.components["stack"].layout
    return layout.common + 3 * layout.largest


@pytest.fixture
def card(tmp_path: Path) -> Any:
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    return SimpleNamespace(weights=weights, checkpoint=write_checkpoint(tmp_path)[0])


def test_the_pinned_tier_holds_every_part_where_the_layout_says(tmp_path: Path) -> None:
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=1 << 30)
    checkpoint, values = write_checkpoint(tmp_path)
    model = load(weights, checkpoint, "host")
    component = model.backend.components["stack"]
    assert [region.path for region in component.regions] == [""] + [
        f"blocks.{i}" for i in range(BLOCKS)
    ]
    assert_pinned(weights, component, values)
    # A CPU executor computes on its tier in place, exactly as on ordinary tensors.
    x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
    reference = Stack()
    reference.load_state_dict(values)
    with torch.inference_mode():
        expected = reference(x)
    assert same(run(model, x, passes=1), [expected])
    # Below its block floor a component moves to the sub-block grain: each projection is
    # a region, and the blocks' own `norm`s join `common`.
    assert model.residency._refine(component.key)
    fine = model.residency.components["stack"]
    assert [region.path for region in fine.blocks] == [
        f"blocks.{i}.{name}" for i in range(BLOCKS) for name in ("proj", "out")
    ]
    assert fine.common is not None and len(fine.common.weights) == 1 + BLOCKS
    assert_pinned(weights, fine, values)
    # No device, so no block hook (a hook takes the compute stream): still bound in place.
    assert not any("forward" in vars(block) for block in model.stack.modules())
    assert same(run(model, x, passes=1), [expected])


def test_an_unbound_weight_is_parked_as_packages_know_it() -> None:
    """Run 2570: H3's finite-weight check read the storage of a text-encoder weight that was
    streaming (not on the card) and failed. A weight with no plane bytes bound has a zero-byte
    storage, which is how packages already tell a parked weight; its shape, dtype and device
    stay facts."""
    from cozy_runtime.internal.weights import unbound
    from cozy_runtime.models.minimax_h3.official import _inspectable

    held = torch.nn.Parameter(torch.empty(4, 8, dtype=torch.bfloat16), requires_grad=False)
    standin = unbound(torch, held, torch.device("cpu"))
    assert standin.untyped_storage().nbytes() == 0 and standin.numel() == 32
    assert (standin.shape, standin.dtype, standin.device.type) == (held.shape, held.dtype, "cpu")
    assert not _inspectable(standin) and _inspectable(held)
    assert standin.to(torch.float32).untyped_storage().nbytes() == 0


def assert_pinned(weights: Weights, component: Any, values: dict[str, Any]) -> None:
    weights.plane.want(component.ws, "pinned").wait()
    held = mmap.mmap(component.ws.host_fd, component.ws.nbytes, prot=mmap.PROT_READ)
    for region in component.regions:
        for weight in region.weights:
            ((offset, _, _),) = weight.parts.values()
            start = region.offset + offset
            expected = values[weight.row.name].numpy().tobytes()
            assert held[start : start + len(expected)] == expected, weight.row.name


@on_card
def test_streamed_blocks_are_bit_identical_to_resident(card: Any) -> None:
    weights = card.weights
    model = load(weights, card.checkpoint, "exact")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    resident = run(model, x)
    assert weights.facts().streamed["stack"].resident_blocks == BLOCKS
    weights.set_budget(tight(model))
    streamed = run(model, x)
    facts = weights.facts()
    assert facts.streamed["stack"].window == 2
    assert facts.streamed["stack"].resident_blocks < BLOCKS
    assert facts.h2d_bytes > 0
    assert same(resident, streamed)


@on_card
def test_a_budget_below_the_block_floor_runs_at_the_sub_block_grain(card: Any) -> None:
    weights = card.weights
    model = load(weights, card.checkpoint, "fine")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    resident = run(model, x)
    layout = model.residency.components["stack"].layout
    assert layout.fine is not None
    weights.set_budget(layout.fine[0] + 2 * layout.fine[1])  # below common + one block
    assert layout.common + layout.largest > weights.budget
    streamed = run(model, x)
    assert model.residency.components["stack"].fine
    assert same(resident, streamed)


@on_card
def test_the_execution_order_is_learned_then_streamed(card: Any) -> None:
    weights = card.weights
    model = load(weights, card.checkpoint, "reverse", reverse=True)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    resident = run(model, x)
    component = model.residency.components["stack"]
    assert component.order == list(range(BLOCKS, 0, -1))
    weights.set_budget(tight(model))
    assert same(resident, run(model, x))


@on_card
def test_a_stage_entered_once_per_step_keeps_a_resident_set(card: Any) -> None:
    """SDXL's package enters its denoise scope once per step: each stage is one pass, and
    the method repeating is what makes it cyclic, so it keeps blocks resident."""
    weights = card.weights
    model = load(weights, card.checkpoint, "steps")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    resident = run(model, x, passes=1)
    weights.set_budget(tight(model))
    steps = [run(model, x, passes=1)[0] for _ in range(3)]
    facts = weights.facts().streamed["stack"]
    assert 0 < facts.resident_blocks < BLOCKS and facts.window >= 1
    assert all(same(resident, [step]) for step in steps)


@on_card
def test_a_component_converted_with_to_streams_in_that_dtype(card: Any) -> None:
    """SDXL upcasts its VAE with `module.to(dtype)` while parts of it stream: streamed blocks
    bind in the converted dtype, exactly as the resident ones compute."""
    weights = card.weights
    model = load(weights, card.checkpoint, "dtype")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float32)
    weights.set_budget(1 << 30)
    model.stack.to(torch.float32)
    resident = run(model, x)
    weights.set_budget(tight(model))
    streamed = run(model, x)
    assert weights.facts().streamed["stack"].resident_blocks < BLOCKS
    assert all(s.dtype == torch.float32 for s in streamed)
    assert all(torch.equal(a, b) for a, b in zip(resident, streamed, strict=True))


def sdxl_unet() -> Any:
    """SDXL's UNet architecture (convolutions, group norms, cross-attention transformer
    blocks, text-time embeddings), small: what streams in SDXL, in fp16."""
    diffusers = pytest.importorskip("diffusers")
    return diffusers.UNet2DConditionModel(
        sample_size=32,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(64, 128),
        transformer_layers_per_block=(1, 2),
        attention_head_dim=(2, 4),
        cross_attention_dim=64,
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=8 * 6 + 64,
    ).to(torch.float16)


@on_card
@pytest.mark.parametrize("pressure", [False, True])
def test_an_sdxl_unet_streams_bit_exact(tmp_path: Path, pressure: bool) -> None:
    """SDXL's UNet resident, then streamed through a tight budget, computes the same bits.
    Under pressure the streamed arm alone also has a ballast leaving only the plane's budget
    and the resident run's measured activations free, as physical ballast does in G4: every
    streamed lease's bytes are checked against the store (`verify_bytes`), so a difference
    can only be the kernels' choice under memory pressure."""
    torch.manual_seed(11)
    values = {k: v.float() * 0.05 for k, v in sdxl_unet().state_dict().items()}
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    model = load(
        weights, store_component(tmp_path, "unet", values), "unet", name="unet", factory=sdxl_unet
    )
    generator = torch.Generator(device="cuda").manual_seed(3)
    args = (
        torch.randn(2, 4, 32, 32, device="cuda", dtype=torch.float16, generator=generator),
        torch.tensor([500.0], device="cuda"),
        torch.randn(2, 16, 64, device="cuda", dtype=torch.float16, generator=generator),
    )
    added = {
        "text_embeds": torch.randn(2, 64, device="cuda", dtype=torch.float16, generator=generator),
        "time_ids": torch.tensor([[256, 256, 0, 0, 256, 256]] * 2, device="cuda").half(),
    }

    def steps() -> list[Any]:
        out = []
        for _ in range(3):
            model.residency.admit("run", ("unet",))
            try:
                with torch.inference_mode():
                    out.append(model.stack(*args, added_cond_kwargs=added).sample)
            finally:
                model.residency.release("run", ("unet",))
        return out

    weights.set_budget(1 << 30)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    resident = steps()
    peak = torch.cuda.max_memory_allocated() - base
    layout: weight_policy.Layout = model.residency.components["unet"].layout
    weights.set_budget(layout.common + 3 * layout.largest)
    ballast = None
    if pressure:
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        left = weights.budget + peak + weight_policy.MARGIN
        ballast = torch.empty(max(free - left, 0), dtype=torch.uint8, device="cuda")
    wrong: list[str] = []
    acquire = model.residency._acquire

    def checked(component: Any, index: int, stream: int) -> Any:
        lease = acquire(component, index, stream)
        wrong.extend(lease.verify_bytes())
        return lease

    model.residency._acquire = checked
    try:
        streamed = steps()
    finally:
        del model.residency._acquire
        model.residency.unpark()
        del ballast
        torch.cuda.empty_cache()
    assert not wrong, f"plane bytes differ from the store: {wrong[:8]}"
    assert weights.facts().streamed["unet"].resident_blocks < len(layout.blocks)
    exact = [torch.equal(a, b) for a, b in zip(resident, streamed, strict=True)]
    assert all(exact), f"bytes exact, outputs not ({exact}): kernels chose by free memory"


@on_card
def test_the_worker_lowers_a_running_stage_at_its_next_block(card: Any) -> None:
    """Room for another executor's context, made while a call runs: the Worker asks through
    the budget cell, and the stage unmaps and streams from its next block on, unchanged."""
    from cozy_runtime.internal import budget_cell

    weights = card.weights
    model = load(weights, card.checkpoint, "cut")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x)
    cell, fd = budget_cell.Cell.create()
    worker = budget_cell.Cell(fd)
    os.close(fd)
    weights.cell = cell
    model.residency.admit("run", ("stack",))
    try:
        with torch.inference_mode():
            outputs = [model.stack(x)]
            ticket = worker.ask(tight(model))
            outputs += [model.stack(x), model.stack(x)]
    finally:
        model.residency.release("run", ("stack",))
        weights.cell = None
        applied = worker.answered(ticket)
        cell.close()
        worker.close()
    assert applied == tight(model)
    facts = weights.facts()
    assert weights.applied == tight(model) and facts.committed_bytes <= tight(model)
    assert facts.streamed["stack"].resident_blocks < BLOCKS
    assert same(reference, outputs)


class WritingBlock(Block):
    """A block that writes its result into its input (a VAE's in-place activations)."""

    def forward(self, x: Any) -> Any:
        return x.add_(torch.tanh(self.out(torch.tanh(self.proj(x * self.norm)))))


class WritingStack(Stack):
    _no_split_modules = ("WritingBlock",)

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([WritingBlock() for _ in range(BLOCKS)])


@on_card
def test_a_block_that_writes_its_inputs_recovers_op_by_op(card: Any) -> None:
    """Runs 2619-2623 (Anima's VAE at 6 GiB): a block that ran out of memory after writing
    into its input in place could not run again, and the load was refused. Its first run
    learns that it writes; from then on its ops recover one by one and it never runs twice,
    in inference mode too, where torch keeps no version counter to tell."""
    weights = card.weights
    model = load(weights, card.checkpoint, "writes", factory=WritingStack)
    x = torch.randn(8192, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x, passes=1)
    blocks = model.residency.components["stack"].blocks
    assert len(blocks) == BLOCKS and all(region.mutates for region in blocks)
    retries = weights.facts().oom_retries
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    ballast = torch.empty(free - (48 << 20), dtype=torch.uint8, device="cuda")
    try:
        retried = run(model, x, passes=1)
    finally:
        del ballast
        torch.cuda.empty_cache()
    assert weights.facts().oom_retries > retries
    assert same(reference, retried)


@on_card
def test_budgets_cut_and_raised_while_streaming_stay_exact(card: Any) -> None:
    """Xid 31 (a copy engine wrote into an unmapped address): the Worker's cuts land at block
    boundaries while the cursor's copies are in flight, and every cut closes the ring and maps
    a new one. Hundreds of them, alternately tight and roomy, compute the resident bits; run
    under `compute-sanitizer --tool memcheck` this is the reproduction."""
    from cozy_runtime.internal import budget_cell

    weights = card.weights
    model = load(weights, card.checkpoint, "churn")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x, passes=1)
    cell, fd = budget_cell.Cell.create()
    worker = budget_cell.Cell(fd)
    os.close(fd)
    weights.cell = cell
    budgets = (tight(model), 1 << 30, model.residency.components["stack"].layout.common)
    out = []
    model.residency.admit("run", ("stack",))
    try:
        with torch.inference_mode():
            for i in range(200):
                worker.ask(budgets[i % len(budgets)])
                out.append(model.stack(x))
        torch.cuda.synchronize()
    finally:
        model.residency.release("run", ("stack",))
        weights.cell = None
        cell.close()
        worker.close()
    assert all(same(reference, [o]) for o in out)


@on_card
def test_an_op_that_no_unmap_can_help_fails_once(card: Any) -> None:
    """The 3 GiB freeze (W5, 21:53): an allocation retried about 40 times in 1.3 s while the
    driver's free bytes never grew, each retry unmapping weights, closing and remapping the
    stage's ring. A retry runs again only when unmapping gave memory back; else it fails."""
    weights = card.weights
    model = load(weights, card.checkpoint, "stuck")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    layout: weight_policy.Layout = model.residency.components["stack"].layout
    weights.set_budget(tight(model))
    run(model, x, passes=1)
    calls = {"room": 0}
    make_room = weights.make_room

    def counted(need: int, operands: Any) -> None:
        calls["room"] += 1
        make_room(need, operands)

    weights.make_room = counted
    model.residency.admit("run", ("stack",))
    try:
        # far beyond the card: no amount of unmapping makes room for it
        with (
            weights.recovering(),
            torch.inference_mode(),
            pytest.raises(Exception, match=r"gave no device memory back|no weight bytes"),
        ):
            torch.empty(1 << 40, dtype=torch.uint8, device="cuda")
    finally:
        model.residency.release("run", ("stack",))
        del weights.make_room
        torch.cuda.empty_cache()
    assert calls["room"] <= 1 + len(layout.blocks)
    assert same(run(model, x, passes=1), run(model, x, passes=1))


@on_card
def test_an_oom_at_a_block_boundary_unmaps_weights_and_retries(card: Any) -> None:
    weights = card.weights
    model = load(weights, card.checkpoint, "oom")
    x = torch.randn(8192, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x, passes=1)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    # Leave less than one block's activations free beside the fully resident weights.
    ballast = torch.empty(free - (48 << 20), dtype=torch.uint8, device="cuda")
    try:
        retried = run(model, x, passes=1)
    finally:
        del ballast
        torch.cuda.empty_cache()
    assert weights.facts().oom_retries >= 1
    assert same(reference, retried)


@on_card
def test_an_idle_construction_gives_room_and_returns_from_the_pinned_tier(card: Any) -> None:
    weights = card.weights
    first = load(weights, card.checkpoint, "first")
    second = load(weights, card.checkpoint, "second")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    total = first.residency.components["stack"].layout.total
    weights.set_budget(total + (8 << 20))
    a = run(first, x)
    b = run(second, x)  # evicts the idle first: least recently used
    assert weights.resident_bytes(first.residency.components["stack"]) < total
    again = run(first, x)
    assert same(a, again) and same(a, b)


# ------------------------------------------------- H3: a DiT, a request LoRA, the turbo overlay

H3_HIDDEN, H3_LAYERS, H3_RANK = 512, 16, 8
H3_TIMES = (0.0, 0.25, 0.75)
H3_KEYS = [(row, tag) for row in range(3) for tag in range(3)]


def h3_dit() -> Any:
    """MiniMax-H3's real AdaLN-pruned DiT, small, with a request LoRA composed as `--lora`
    composes it: PEFT layers over its attention and output projections."""
    pytest.importorskip("peft")
    from cozy_runtime.author import AdapterRef
    from cozy_runtime.internal.lora_composition import composer
    from cozy_runtime.internal.lora_contract import FORMAT, Graph, Linear
    from cozy_runtime.models.minimax_h3.adaln_pruned import AdaLNPrunedMiniMaxH3Transformer

    dit: Any = AdaLNPrunedMiniMaxH3Transformer(
        table_timesteps=H3_TIMES,
        table_block_keys=H3_KEYS,
        num_attention_heads=4,
        attention_head_dim=H3_HIDDEN // 4,
        hidden_size=H3_HIDDEN,
        num_layers=H3_LAYERS,
        num_refiner_layers=1,
        ffn_dim=4 * H3_HIDDEN,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=4,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    dit.eval().install_lora_consumers()
    paths = [
        f"transformer_blocks.{block}.{site}"
        for block in range(H3_LAYERS)
        for site in ("attn.to_q", "attn.to_out.0", "ff.net.2")
    ]
    graph = Graph(
        FORMAT,
        tuple(Linear("fl2va_dit", path, "wushu", H3_RANK, 8.0, 0.8, "f16") for path in paths),
        (AdapterRef("sha256:" + "12" * 32, 0.8, "lora", "fl2va_dit", "adapter"),),
    )
    composer(graph)(SimpleNamespace(components={"fl2va_dit": dit}))
    return dit.to(torch.float16)


def h3_turbo() -> Any:
    """H3's turbo LoRA: its own model slot, whose factors the DiT's blocks read through hooks."""
    from cozy_runtime.models.minimax_h3.turbo import TurboOverlay, TurboSchedule

    return (
        TurboOverlay(
            hidden_size=H3_HIDDEN,
            inner_dim=H3_HIDDEN,
            ffn_dim=4 * H3_HIDDEN,
            num_layers=H3_LAYERS,
            num_refiner_layers=1,
            video_out=2,
            audio_out=2,
            rank=H3_RANK,
            alpha=8,
            schedule=TurboSchedule((0.75, 0.0), (0.25, 0.0)),
            table_timesteps=H3_TIMES,
            table_block_keys=H3_KEYS,
            block_table_dtype=torch.float16,
            final_table_dtype=torch.float16,
        )
        .eval()
        .to(torch.float16)
    )


def h3_inputs(device: str) -> dict[str, Any]:
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    position, tags, video, audio, text, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=torch.ones(8, dtype=torch.long),
        num_latent_frames=6,
        latent_height=16,
        latent_width=16,
        num_audio_latents=32,
        patch_size=(1, 1, 1),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
        keyframe_anchors=(),
    )
    generator = torch.Generator().manual_seed(5)

    def noise(*shape: int) -> Any:
        return torch.randn(*shape, generator=generator).to(device, torch.float16)

    return {
        "hidden_states": noise(1, len(video), 2),
        "audio_hidden_states": noise(1, len(audio), 2),
        "encoder_hidden_states": noise(1, len(text), 4),
        "timestep": torch.tensor([0.75, 0.25], device=device),
        "timestep_indices": torch.where(tags == 2, 1, 0).to(device),
        "token_tags": tags.to(device),
        "position_ids": position.to(device),
        "video_indices": video.to(device),
        "audio_indices": audio.to(device),
        "text_indices": text.to(device),
        "return_dict": False,
    }


def h3_models(weights: Weights, root: Path, key: str) -> tuple[Any, Any]:
    """The DiT with its request LoRA and the turbo LoRA: two model slots on one plane."""
    slots = []
    for name, factory, scope in (
        ("fl2va_dit", h3_dit, "sample_fl2va_turbo"),
        ("fl2va_turbo", h3_turbo, "sample_fl2va"),
    ):
        torch.manual_seed(292)
        values = {k: torch.randn(v.shape) * 0.05 for k, v in factory().state_dict().items()}
        (root / name).mkdir()
        checkpoint = store_component(root / name, name, values)
        slots.append(
            load(weights, checkpoint, f"{key}-{name}", name=name, factory=factory, scope=scope)
        )
    return slots[0], slots[1]


def h3_denoise(weights: Weights, dit: Any, turbo: Any, inputs: dict[str, Any], between: Any) -> Any:
    """Three evaluations of the DiT under the turbo bank, as H3's `sample_fl2va_turbo` nests
    its scopes: the base's stage, the turbo LoRA's inside it, package arithmetic between."""
    from cozy_runtime.models.minimax_h3.turbo import ATTENTION_KWARG, OVERLAY_KWARG, TURBO_BANK

    selector = {ATTENTION_KWARG: TURBO_BANK, OVERLAY_KWARG: turbo.stack}
    video, audio = inputs["hidden_states"], inputs["audio_hidden_states"]
    dit.residency.admit("sample_fl2va_turbo", ("fl2va_dit",))
    try:
        turbo.residency.admit("sample_fl2va", ("fl2va_turbo",))
        try:
            with weights.recovering(), torch.inference_mode():
                for step in range(3):
                    held = between(step)
                    moved = dit.stack(
                        **{**inputs, "hidden_states": video, "audio_hidden_states": audio},
                        attention_kwargs=selector,
                    )
                    video, audio = video + moved[0] * 0.25, audio + moved[1] * 0.25
                    del held
                return video, audio
        finally:
            turbo.residency.release("sample_fl2va", ("fl2va_turbo",))
    finally:
        dit.residency.release("sample_fl2va_turbo", ("fl2va_dit",))


def test_h3_with_its_loras_computes_on_the_pinned_tier(tmp_path: Path) -> None:
    """The fixture is the real thing: on a CPU executor (weights bound in place) the DiT with
    a request LoRA under the turbo bank computes what the same modules compute unmanaged."""
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=1 << 30)
    dit, turbo = h3_models(weights, tmp_path, "host")
    inputs = h3_inputs("cpu")
    got = h3_denoise(weights, dit, turbo, inputs, lambda step: None)
    plain = []
    for factory in (h3_dit, h3_turbo):
        torch.manual_seed(292)
        module = factory()
        module.load_state_dict(
            {k: (torch.randn(v.shape) * 0.05).half() for k, v in module.state_dict().items()}
        )
        plain.append(module)
    plain_dit, plain_turbo = plain
    unmanaged = SimpleNamespace(admit=lambda *_: None, release=lambda *_: None)
    want = h3_denoise(
        weights,
        SimpleNamespace(stack=plain_dit, residency=unmanaged),
        SimpleNamespace(stack=plain_turbo, residency=unmanaged),
        inputs,
        lambda step: None,
    )
    assert all(torch.isfinite(t.float()).all() for t in got)
    assert same(list(got), list(want))


@on_card
def test_h3_with_its_loras_pages_instead_of_running_out(tmp_path: Path) -> None:
    """Run 2549: H3's DiT with a request LoRA under the turbo LoRA ran out of device memory.
    Here its weights get a budget far below their size, and mid-denoise another allocation
    takes the room its activations had. It still computes the fully resident run's bits: the
    turbo LoRA's stage shares the DiT's budget, blocks stream, and the allocation that fails
    unmaps weights and runs again."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    dit, turbo = h3_models(weights, tmp_path, "h3")
    inputs = h3_inputs("cuda")
    layout: weight_policy.Layout = dit.residency.components["fl2va_dit"].layout
    overlay = turbo.residency.components["fl2va_turbo"].layout.total
    weights.set_budget(1 << 30)
    resident, peak = h3_reference(weights, dit, turbo, inputs)
    assert weights.facts().streamed["fl2va_dit"].resident_blocks == len(layout.blocks)

    spare = 6 * layout.largest  # what an out-of-memory retry can still unmap
    weights.set_budget(weight_policy.floor(layout) + overlay + spare)
    assert weights.budget < (layout.total + overlay) // 2
    was = weights.facts()
    try:
        paged = h3_denoise(weights, dit, turbo, inputs, squeezing(1, peak))
    finally:
        torch.cuda.empty_cache()
    now = weights.facts()
    assert now.streamed["fl2va_dit"].resident_blocks < len(layout.blocks)
    assert now.h2d_bytes > was.h2d_bytes and now.evictions > was.evictions
    assert now.oom_retries > was.oom_retries
    assert same(list(resident), list(paged))


def h3_reference(weights: Weights, dit: Any, turbo: Any, inputs: dict[str, Any]) -> Any:
    """The fully resident denoise and its activation peak."""
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    resident = h3_denoise(weights, dit, turbo, inputs, lambda step: None)
    return resident, torch.cuda.max_memory_allocated() - before


def squeezing(at: int, peak: int) -> Any:
    """Before evaluation `at`, another allocation takes all but half the activations' room."""

    def squeeze(step: int) -> Any:
        if step != at:
            return None
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        return torch.empty(free - peak // 2, dtype=torch.uint8, device="cuda")

    return squeeze


@on_card
def test_one_gpu_of_a_group_never_runs_a_block_twice(tmp_path: Path) -> None:
    """Run 2589 (2x A40): a block that ran out of memory was run again, but in a group it
    holds collectives the other GPUs are already in, and the group hung. One GPU of a group
    runs an unmeasured stage's first pass at the floor and grows at the second by what it
    measured; an op that still runs out of memory unmaps idle weights and that op runs again."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    dit, turbo = h3_models(weights, tmp_path, "h3")
    (tmp_path / "idle").mkdir()
    idle = load(weights, write_checkpoint(tmp_path / "idle")[0], "idle")
    inputs = h3_inputs("cuda")
    layout: weight_policy.Layout = dit.residency.components["fl2va_dit"].layout
    weights.set_budget(1 << 30)
    resident, peak = h3_reference(weights, dit, turbo, inputs)
    calls = {"block": 0}
    block = dit.stack.transformer_blocks[0].attn
    block.register_forward_hook(lambda *_: calls.__setitem__("block", calls["block"] + 1))

    weights.grouped = True
    weights.activation.clear()  # a fresh executor's first request: nothing measured
    grown = h3_denoise(weights, dit, turbo, inputs, lambda step: None)
    assert same(list(resident), list(grown)) and calls["block"] == 3
    assert weights.facts().streamed["fl2va_dit"].resident_blocks == len(layout.blocks)
    assert weights.facts().h2d_bytes > 0  # the first pass streamed every block

    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    run(idle, x, passes=1)
    held = weights.resident_bytes(idle.residency.components["stack"])
    retries = weights.facts().oom_retries
    try:
        squeezed = h3_denoise(weights, dit, turbo, inputs, squeezing(1, peak))
    finally:
        torch.cuda.empty_cache()
    assert weights.facts().oom_retries > retries
    assert weights.resident_bytes(idle.residency.components["stack"]) < held
    assert same(list(resident), list(squeezed)) and calls["block"] == 6


@on_card
def test_an_op_outside_every_block_runs_again_after_weights_leave(card: Any) -> None:
    """Package arithmetic between component calls allocates too: when it cannot, idle and
    then resident weights are unmapped and that one op runs again."""
    weights = card.weights
    model = load(weights, card.checkpoint, "glue")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x, passes=1)
    retries = weights.facts().oom_retries
    model.residency.admit("run", ("stack",))
    try:
        with weights.recovering(), torch.inference_mode():
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            ballast = torch.empty(free - (8 << 20), dtype=torch.uint8, device="cuda")
            scratch = torch.zeros(64 << 20, dtype=torch.uint8, device="cuda")
            out = model.stack(x)
    finally:
        model.residency.release("run", ("stack",))
        del ballast, scratch
        torch.cuda.empty_cache()
    assert weights.facts().oom_retries > retries
    assert same(reference, [out])
