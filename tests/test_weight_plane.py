"""The executor's weights on the real weight plane (weight-plane.md §4).

Real: a TensorFS store holding a block model's weights, `tensorfs.plane`, the runtime's
`PlaneBackend` and `WeightResidency`, torch. The model is small and exact: a stack of
declared no-split blocks whose forward reads every weight, so any byte streamed wrong changes
the output. The host tier runs without a card; the rest needs device 0 and holds the whole
card, so run it under the shared GPU lock.
"""

from __future__ import annotations

import gc
import io
import mmap
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tensorfs import Store  # noqa: E402
from tensorfs.derived import Config as NativeConfig  # noqa: E402
from tensorfs.derived import Derivation, Part, Target, Tensor  # noqa: E402

from cozy_runtime.author._activity import asides  # noqa: E402
from cozy_runtime.author._loader import census  # noqa: E402
from cozy_runtime.author.kernels import rms_rope_split_half  # noqa: E402
from cozy_runtime.internal import accel, paging, plane, spill, weight_policy  # noqa: E402
from cozy_runtime.internal import weights as weights_module  # noqa: E402
from cozy_runtime.internal.derive import Observations, serving_substrate  # noqa: E402
from cozy_runtime.internal.encoding import SPEC_PLAIN  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.weights import (  # noqa: E402
    INDIVISIBLE,
    HostTiers,
    Journal,
    PlaneBackend,
    ResidencyRefusal,
    WeightResidency,
    Weights,
)
from cozy_runtime.internal.weights_sink import weights_transaction_id  # noqa: E402

# These tests fill the card to its last megabytes: never on one a desktop is drawn on.
on_card = pytest.mark.skipif(
    not accel.present(torch, "cuda") or accel.drives_display(0),
    reason="a CUDA card that drives no display",
)
# These hold the process to an owner's cap instead: any card runs them.
on_gpu = pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card")

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
    return Checkpoint.from_store(store, "sha256:" + receipt["manifest"]["sha256"])


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


@pytest.fixture(autouse=True)
def _collected() -> Iterator[None]:
    """A construction is a reference cycle (its blocks' hooks hold it) that keeps one file
    open per stored tensor: the last test's is collected before this one opens its own."""
    gc.collect()
    yield
    gc.collect()


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


def test_a_stored_tensor_the_code_does_not_build_is_skipped_and_never_read(
    tmp_path: Path,
) -> None:
    """Qwen's text encoder stopped building `lm_head` and its checkpoint still stores it
    (1.24 GB). The fit skips it with one warning, and nothing reads or maps its bytes: its
    object leaves the store before the fill, which still lands every built weight."""
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=1 << 30)
    generator = torch.Generator().manual_seed(7)
    values: dict[str, Any] = {"embed": torch.rand(HIDDEN, generator=generator) + 0.5}
    for index in range(BLOCKS):
        for name in ("proj", "out"):
            values[f"blocks.{index}.{name}.weight"] = (
                torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
            )
        values[f"blocks.{index}.norm"] = torch.rand(HIDDEN, generator=generator) + 0.5
    head = torch.randn(2 * HIDDEN, HIDDEN, generator=generator)
    checkpoint = store_component(tmp_path, "stack", {**values, "lm_head.weight": head})
    built = {key: value.half() for key, value in values.items()}
    rows = checkpoint.rows("stack")

    backend = PlaneBackend.for_script(
        {"stack": checkpoint}, rows, weights=weights, construction="skip", release="plane-test/1"
    )
    with serving_substrate("cpu", Observations()):
        module = Stack()
    holder = SimpleNamespace(components={"stack": module})
    walked = census(holder)
    # As the executor does: completeness is over every stored key, and the fit takes the
    # skipped ones out of it.
    backend.expect({"stack": tuple(row.key for row in rows)})
    verdict = backend.fit(walked, encoded_leaves="refuse")
    assert verdict is not None and verdict["ok"], verdict
    assert verdict["ignored"] == ["stack/lm_head.weight"]
    skipped = head.numel() * 2
    assert verdict["warnings"] == [
        "1 stored tensor(s) the code does not build were skipped, not loaded: "
        f"{skipped} B (stack/lm_head.weight)"
    ]

    # A read or a map of the skipped tensor would need its object: it is gone.
    everything = checkpoint.read_plan([("stack", row.name) for row in rows], 1 << 24, ("stack",))
    objects = {
        item["object"].removeprefix("sha256:")
        for item in everything.items()
        if item["what"] == "stack/lm_head.weight#value"
    }
    assert objects, "the skipped tensor is stored as objects"
    for hex_ in objects:
        (tmp_path / "store" / "blobs" / hex_[:2] / hex_[2:4] / hex_).unlink()

    live = backend.materialize(holder, walked)
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    component = backend.components["stack"]
    # The pinned tier is the one a checkpoint without that tensor gets: no room for it.
    plain = load(weights, store_component(tmp_path / "plain", "stack", values), "plain")
    assert component.ws.nbytes == plain.backend.components["stack"].ws.nbytes
    assert all(
        weight.row.name != "lm_head.weight"
        for region in component.regions
        for weight in region.weights
    )
    assert_pinned(weights, component, built)
    filled = sum(value.numel() * 2 for value in built.values())
    ready = {row.name: row.host_ready_bytes for row in plane.stats(weights.plane).sets}
    assert ready["skip/stack"] == filled, ready
    model = SimpleNamespace(
        stack=module,
        residency=WeightResidency(weights, backend.components, {"run": ("stack",)}),
    )
    x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
    reference = Stack()
    reference.load_state_dict(built)
    with torch.inference_mode():
        expected = reference(x)
    assert same(run(model, x, passes=1), [expected])


def test_an_unbound_weight_is_parked_as_packages_know_it() -> None:
    """Run 2570: H3's finite-weight check read the storage of a text-encoder weight that was
    streaming (not on the card) and failed. A weight with no plane bytes bound has a zero-byte
    storage, which is how packages already tell a parked weight; its shape, dtype and device
    stay facts."""
    from cozy_runtime.internal.weights import unbound

    held = torch.nn.Parameter(torch.empty(4, 8, dtype=torch.bfloat16), requires_grad=False)
    standin = unbound(torch, held, torch.device("cpu"))
    assert standin.untyped_storage().nbytes() == 0 and standin.numel() == 32
    assert (standin.shape, standin.dtype, standin.device.type) == (held.shape, held.dtype, "cpu")
    assert standin.to(torch.float32).untyped_storage().nbytes() == 0
    # Run 2793: inside an out-of-memory retry `Tensor.to` reaches the stand-in whole (not as
    # the copy it decomposes into); the plane retyping one there mapped it, and recursed.
    aten, kind = torch.ops.aten, type(standin)
    for func, args, kwargs in (
        (aten.to.dtype, (standin, torch.float32), {}),
        (aten.to.device, (standin, torch.device("cpu"), torch.float32), {}),
        (aten.to.dtype_layout, (standin,), {"dtype": torch.float32}),
        (aten.to.other, (standin, torch.empty(1, dtype=torch.float32)), {}),
    ):
        moved = kind.__torch_dispatch__(func, (kind,), args, kwargs)
        assert type(moved) is kind and moved.shape == held.shape
        assert moved.dtype == torch.float32 and moved.untyped_storage().nbytes() == 0


def test_a_kernel_launch_is_never_handed_a_weight_with_no_bytes() -> None:
    """A Triton kernel reads its operands by pointer. One that is a weight with no plane
    bytes bound, which nothing here can map (no stage owns it), is refused before the launch;
    every other operand passes through untouched."""
    from cozy_runtime.internal.triton_kernels import _Launcher
    from cozy_runtime.internal.weights import ResidencyRefusal, unbound

    launched: list[tuple[Any, ...]] = []
    compiled: Any = {"scale[]": {(4, 1, 1): lambda *args: launched.append(args)}}
    kernel = SimpleNamespace(__name__="scale", arg_names=["x_ptr", "n", "eps"])
    launch = _Launcher(89, compiled, {"scale": ()})
    scale = torch.ones(8)
    launch(kernel, (4, 1, 1), scale, 8, 1e-6)
    assert len(launched) == 1 and launched[0][0] is scale and launched[0][1:] == (8, 1e-6)
    standin = unbound(torch, torch.nn.Parameter(torch.empty(8)), torch.device("cpu"))
    with pytest.raises(ResidencyRefusal, match="no plane bytes") as refused:
        launch(kernel, (4, 1, 1), scale, standin, 8)
    assert refused.value.code == "weight_unbound" and len(launched) == 1


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
def test_a_stage_waits_for_its_weights_before_its_clock_starts(card: Any) -> None:
    """The bytes a new plan wants reach the device before they bind, and that wait is
    recorded as loading: nothing is copied after it and no lease of the stage is late."""
    weights = card.weights
    model = load(weights, card.checkpoint, "arrive")
    spans: list[tuple[str, float]] = []
    weights.set_budget(1 << 30)
    with asides(lambda name, ms: spans.append((name, ms))):
        model.residency.admit("run", ("stack",))
    arrived = weights.facts().h2d_bytes
    assert arrived > 0 and {name for name, _ in spans} == {"loading weights"}
    with torch.inference_mode():
        model.stack(torch.zeros(1, HIDDEN, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    model.residency.release("run", ("stack",))
    stages = plane.stats(weights.plane).stages
    # a lease's own bookkeeping is nanoseconds; a wait for bytes is milliseconds
    assert weights.facts().h2d_bytes == arrived
    assert sum(row.late for row in stages) == 0 and sum(row.stall_ns for row in stages) < 1e6


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


HEADS = 16


class Scale(torch.nn.Module):  # type: ignore[misc,name-defined]
    """A norm's scale, which its owner reads without calling the norm (as fused kernels do)."""

    eps = 1e-6

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(width, dtype=torch.float16))

    def forward(self, x: Any) -> Any:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SiblingBlock(Block):
    """Reads its sub-modules' weights itself: a norm scale, and a whole projection."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = Scale(HIDDEN)

    def forward(self, x: Any) -> Any:
        hidden = torch.tanh(self.proj(x * self.norm * self.scale.weight))
        return x + torch.tanh(torch.nn.functional.linear(hidden, self.out.weight))


class FusedBlock(torch.nn.Module):  # type: ignore[misc,name-defined]
    """A self-attention whose head norms' scales our compiled RMSNorm + RoPE kernel reads by
    pointer, outside torch (Anima's fused plan)."""

    heads = HEADS

    def __init__(self) -> None:
        super().__init__()
        for name in ("to_q", "to_k", "to_v"):
            setattr(self, name, torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16))
        self.to_out = torch.nn.ModuleList(
            [torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16), torch.nn.Identity()]
        )
        self.norm_q, self.norm_k = Scale(HIDDEN // HEADS), Scale(HIDDEN // HEADS)

    def forward(self, x: Any) -> Any:
        q, k, v = (
            getattr(self, name)(x[None]).unflatten(2, (self.heads, -1)).transpose(1, 2)
            for name in ("to_q", "to_k", "to_v")
        )
        turn = torch.eye(2, device=x.device).expand(1, 1, x.shape[0], 64, 2, 2).contiguous()
        q, k = rms_rope_split_half(
            q, k, turn, self.norm_q.weight, self.norm_k.weight, self.norm_q.eps
        )
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return x + torch.tanh(self.to_out[0](out.transpose(1, 2).flatten(2, 3))[0])


def stack_of(
    block: type, matrices: tuple[str, ...], scales: tuple[tuple[str, int], ...]
) -> tuple[Any, dict[str, Any]]:
    """A `Stack` of `block`s and its checkpoint: each block's projections and norm scales."""

    class Stacked(Stack):
        _no_split_modules = (block.__name__,)

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([block() for _ in range(BLOCKS)])

    generator = torch.Generator().manual_seed(7)
    values: dict[str, Any] = {"embed": torch.rand(HIDDEN, generator=generator) + 0.5}
    for index in range(BLOCKS):
        for name in matrices:
            values[f"blocks.{index}.{name}.weight"] = (
                torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
            )
        for name, width in scales:
            key = f"blocks.{index}.{name}"
            values[key if name == "norm" else f"{key}.weight"] = (
                torch.rand(width, generator=generator) + 0.5
            )
    return Stacked, values


def at_the_sub_block_grain(weights: Weights, model: Any, x: Any) -> tuple[list[Any], list[Any]]:
    """The model's outputs resident, then in a budget below its block floor."""
    weights.set_budget(1 << 30)
    resident = run(model, x)
    layout = model.residency.components["stack"].layout
    assert layout.fine is not None
    weights.set_budget(layout.fine[0] + 2 * layout.fine[1])
    streamed = run(model, x)
    assert model.residency.components["stack"].fine
    assert weights.modes["stack.grain"] == "sub-block"  # the run record's confession
    return resident, streamed


@on_card
def test_weights_read_outside_their_modules_forward_stream_exact(tmp_path: Path) -> None:
    """A weight is bound inside its own module's forward, and code reads weights other ways:
    a block multiplies by a sibling norm's scale, a package reaches for `.weight`. At the
    sub-block grain those modules are regions of their own that nothing calls. The small one
    stays in `common` (bound for the stage); the large one is mapped and held for the one op
    that reads it. Either way the bits are the resident run's."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    factory, values = stack_of(SiblingBlock, ("proj", "out"), (("norm", HIDDEN), ("scale", HIDDEN)))
    model = load(weights, store_component(tmp_path, "stack", values), "siblings", factory=factory)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    resident, streamed = at_the_sub_block_grain(weights, model, x)
    component = model.residency.components["stack"]
    assert component.common is not None
    assert {f"blocks.{i}.scale.weight" for i in range(BLOCKS)} <= {
        w.row.name for w in component.common.weights
    }
    assert "blocks.0.out" in {region.path for region in component.blocks}
    assert same(resident, streamed)


@on_card
@pytest.mark.parametrize("small", [paging.SMALL, 1 << 40])
def test_a_kernel_launched_outside_torch_never_reads_an_unmapped_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small: int
) -> None:
    """Run 2744, the laptop's Xid 31: at the sub-block grain Anima's fused self-attention
    handed its head norms' scales to our compiled kernel while no plane bytes were bound to
    them, and the kernel read a pointer with nothing mapped behind it. The scales stay in
    `common`; and were one its own region after all (the second case), the launch maps it
    first (`author.kernels.readable`) and never faults the device."""
    pytest.importorskip("cozy_runtime._kernels._C")
    own_region = small != paging.SMALL
    monkeypatch.setattr(paging, "SMALL", small)
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    width = HIDDEN // HEADS
    factory, values = stack_of(
        FusedBlock, ("to_q", "to_k", "to_v", "to_out.0"), (("norm_q", width), ("norm_k", width))
    )
    model = load(weights, store_component(tmp_path, "stack", values), "kernel", factory=factory)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    resident, streamed = at_the_sub_block_grain(weights, model, x)
    units = {region.path for region in model.residency.components["stack"].blocks}
    assert ("blocks.0.norm_q" in units) == own_region
    assert same(resident, streamed)


@on_card
def test_a_stage_entered_inside_another_leaves_the_outer_its_own_grain(card: Any) -> None:
    """Run 2763: Anima's DiT, mid-stage and so unable to move to the sub-block grain, was
    given only its sub-block floor when the stage was placed again, and the run failed with
    "does not fit at its grain" in a budget that held it. A component that cannot refine now
    keeps at least what its current grain needs; the others stream instead."""
    weights = card.weights
    outer = load(weights, card.checkpoint, "b-outer")
    inner = load(weights, card.checkpoint, "a-inner")  # placed first: it would take the spare
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    want = run(outer, x, passes=1), run(inner, x, passes=1)
    layout: weight_policy.Layout = outer.residency.components["stack"].layout
    assert layout.fine is not None and sum(layout.fine) < layout.common + layout.largest
    weights.set_budget(layout.total + layout.common + layout.largest - (4 << 20))
    outer.residency.admit("run", ("stack",))
    try:
        got_inner = run(inner, x, passes=1)  # placed together with the outer's open stage
        with torch.inference_mode():
            got_outer = [outer.stack(x)]
    finally:
        outer.residency.release("run", ("stack",))
    assert not outer.residency.components["stack"].fine
    assert same(want[0], got_outer) and same(want[1], got_inner)


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


@on_gpu
def test_batch_fits_asks_an_unmeasured_method_with_what_earlier_executors_measured(
    card: Any,
) -> None:
    """A fresh executor has measured nothing, and "unmeasured is yes" would batch cond and
    uncond in a pool that cannot hold both. What the machine seeded for the method answers."""
    weights = card.weights
    residency = load(weights, card.checkpoint, "batch").residency
    assert residency.batch_fits("run", 2), "measured nowhere: the first run measures"
    weights.seed_activations({"run": torch.cuda.get_device_properties(0).total_memory})
    assert not residency.batch_fits("run", 2)
    assert weights.modes["run.batch"] == "1"
    facts = weights.facts()
    assert facts.bounce_copy_bytes == 0 and facts.register_failed >= 0
    assert facts.h2d_bytes == 0 or facts.host_gbps + facts.bounce_gbps + facts.disk_gbps > 0


@on_gpu
def test_weights_mapped_ahead_leave_the_first_stage_nothing_to_copy(card: Any) -> None:
    """The warm set's `gpu` level (`Map`): a loaded construction's weights reach the card
    before any request, as much as a call keeps resident in the same budget. Its first stage
    then copies nothing and computes the same bits, and the map measures nothing of its own."""
    weights = card.weights
    weights.set_budget(1 << 30)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    cold = load(weights, card.checkpoint, "cold")
    expected = run(cold, x, passes=1)
    resident = weights.resident_bytes(cold.residency.components["stack"])
    ahead = load(weights, card.checkpoint, "ahead")
    before = weights.facts().h2d_bytes
    spans = sum(region.span for region in ahead.residency.components["stack"].regions)
    assert ahead.residency.map_ahead() == (spans, True) and resident > 0
    arrived = weights.facts().h2d_bytes
    assert arrived - before >= resident
    assert same(expected, run(ahead, x, passes=1))
    facts = weights.facts()
    assert facts.h2d_bytes == arrived
    assert weights_module.MAPPING not in weights.activation
    assert weights_module.MAPPING not in facts.plan_inputs
    # A budget that holds only part of the weights maps that part, and says so.
    weights.set_budget(tight(ahead))
    partial = load(weights, card.checkpoint, "partial")
    held, complete = partial.residency.map_ahead()
    assert 0 < held < spans and not complete


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


@on_gpu
def test_a_component_cast_and_cast_back_in_its_stage_keeps_no_copy_after_it(card: Any) -> None:
    """SDXL's decode upcasts its VAE and casts it back before its stage ends. The cast back
    left a float16 copy of every weight in torch's memory beside the plane's bytes, across
    requests: 169 MiB at an rtx-3070's 6 GiB, and the UNet streamed 8 to 22 blocks each warm
    step. The stage's end lets the copies go, and the next stage computes the same bits."""
    weights = card.weights
    model = load(weights, card.checkpoint, "cast")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float32)
    weights.set_budget(1 << 30)
    total = model.residency.components["stack"].layout.total

    def decode() -> Any:
        model.residency.open_attempt()
        model.residency.admit("decode", ("stack",))
        try:
            model.stack.to(torch.float32)
            with torch.inference_mode():
                out = model.stack(x)
            model.stack.to(torch.float16)
            return out
        finally:
            model.residency.release("decode", ("stack",))

    before = torch.cuda.memory_allocated()
    first = decode()
    assert torch.cuda.memory_allocated() - before < total // 4
    second = decode()
    assert torch.cuda.memory_allocated() - before < total // 4
    assert torch.equal(first, second) and first.dtype == torch.float32


@on_card
def test_an_activation_estimate_larger_than_the_card_never_refuses_the_stage(card: Any) -> None:
    """F, rtx-3070 with a 3 GiB pool: SDXL's decode seed (3.1 GB, its untiled peak at 1024²)
    was more than the room under the cap, so the stage's plane budget came out 0 and every
    SDXL request was refused for size. An estimate gives way to the stage's floor: the
    weights stream and the stage computes the same bits."""
    weights = card.weights
    model = load(weights, card.checkpoint, "seeded")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x)
    model.residency.unpark()
    weights.set_budget(-1)
    weights.seed_activations({"run": weights.total})
    assert weights.driver_budget("run") == 0, "no stage open: no floor to keep"
    assert same(run(model, x), reference)


@on_gpu
def test_an_earlier_stages_idle_cache_is_neither_kept_nor_counted(card: Any) -> None:
    """SeedVR2 run 5937: a VAE decode left torch 5.5 GiB of idle cache; the transformer's next
    stage was planned in what the driver had beside it (a 0.5 GiB budget), and the cache counted
    as its activations. Before a stage plans, torch's idle cache beyond the stage's activations
    goes back to the driver; what is still cached at its entry is no activation of its own."""
    weights = card.weights
    model = load(weights, card.checkpoint, "cached")
    x = torch.ones(4, HIDDEN, device="cuda", dtype=torch.float16)

    def stage(method: str, freed: int) -> None:
        model.residency.admit(method, ("stack",))
        with torch.inference_mode():
            model.stack(x)
            torch.empty(freed, dtype=torch.uint8, device="cuda")  # an activation, freed
        model.residency.release(method, ("stack",))

    def idle() -> int:
        return torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

    stage("decode", 512 << 20)
    weights.seed_activations({"encode": 1 << 30})  # room for its cache: it stays
    stage("encode", 0)
    assert idle() >= 512 << 20
    assert model.residency.peaks["encode"] < weight_policy.MARGIN, "the cache was not its own"
    model.residency.admit("denoise", ("stack",))
    assert idle() < 512 << 20, "nothing measured for the stage: its plan has the room"
    model.residency.release("denoise", ("stack",))


@on_card
def test_a_bind_that_runs_out_of_device_memory_plans_again_in_what_is_left(card: Any) -> None:
    """Run 2932: on a card without fp8, binding a stage's resident set decodes each weight
    into a float copy, and with the card full that allocation failed inside the plan. The
    stage ended there and the group with it. Binding's copies are torch's bytes (here a
    component converted to fp32: twice its stored bytes): when one does not fit, the plan
    gives room back and binds again, and the stage computes the same bits."""
    weights = card.weights
    model = load(weights, card.checkpoint, "bind")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float32)
    weights.set_budget(1 << 30)
    model.stack.to(torch.float32)
    resident = run(model, x)
    model.residency.unpark()
    weights.set_budget(0)  # nothing mapped: the next stage binds, and converts, everything
    layout: weight_policy.Layout = model.residency.components["stack"].layout
    retries = weights.facts().oom_retries
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    # Room for the stored weights and half of their fp32 copies, no more.
    ballast = torch.empty(free - 2 * layout.total, dtype=torch.uint8, device="cuda")
    try:
        weights.set_budget(layout.total + (8 << 20))
        squeezed = run(model, x)
    finally:
        del ballast
        torch.cuda.empty_cache()
    assert weights.facts().oom_retries > retries
    assert weights.facts().streamed["stack"].resident_blocks < BLOCKS
    assert all(torch.equal(a, b) for a, b in zip(resident, squeezed, strict=True))


@on_card
def test_a_process_cap_holds_torch_and_the_plane_inside_it(card: Any) -> None:
    """The owner's cap bounds the whole process: its CUDA context, torch's reserved bytes and
    the plane's maps. Activations the plan did not count are an out-of-memory torch's
    allocator raises at the cap, and the plane unmaps weights to run them: the same bits, and
    the process never above its cap, whatever the driver has free."""
    weights = card.weights
    model = load(weights, card.checkpoint, "cap")
    x = torch.randn(1 << 15, HIDDEN, device="cuda", dtype=torch.float16)  # 128 MiB each
    torch.cuda.reset_peak_memory_stats()
    live = torch.cuda.memory_allocated()
    uncapped = run(model, x, passes=1)
    # the activations' tensors at their peak, without any segment slack torch kept
    measured = torch.cuda.max_memory_allocated() - live
    model.residency.unpark()
    weights.set_budget(0)
    torch.cuda.empty_cache()
    weights.measure_context()
    # Room for the measured activations and a margin: a stage that keeps every weight (here
    # 128 MiB, more than the margin) leaves its activations short, one that streams does not.
    context = max(weights.context_bytes, 0)  # unreadable per process: counts nothing
    cap = context + torch.cuda.memory_reserved() + measured + weight_policy.MARGIN
    weights.activation.clear()  # an estimate that is wrong: the stage keeps all its weights
    retries = weights.facts().oom_retries
    torch.cuda.reset_peak_memory_stats()
    weights.set_cap(cap)
    try:
        weights.set_budget(-1)
        capped = run(model, x, passes=1)
        peak = torch.cuda.max_memory_reserved()
        facts = weights.facts()
    finally:
        weights.set_cap(-1)
    exact = same(uncapped, capped)
    del x, uncapped, capped  # its 128 MiB segments must not shape the next test's allocations
    torch.cuda.empty_cache()
    assert facts.cap_bytes == cap
    assert facts.oom_retries > retries
    assert peak + context <= cap
    assert 0 <= facts.process_bytes <= cap
    assert exact


@on_gpu
def test_an_op_with_no_room_left_moves_the_tensors_the_process_holds_to_host(card: Any) -> None:
    """Run 5686, qwen-image-2 at 1 MP on an RTX 3070: the VAE decode kept a copy of every
    convolution's input in its feature cache, 6.6 GiB of torch's own tensors, and a
    convolution was refused with the plane already down to 6 MB. Tensors an op does not read
    move to page-locked host memory in place: the stage computes the same bits, and each held
    tensor is still the same CUDA tensor with the same values, read over the bus. Ops note
    what they read for the spill's order only under pressure: a run with room pays nothing."""
    weights = card.weights
    model = load(weights, card.checkpoint, "spill")
    x = torch.randn(1 << 15, HIDDEN, device="cuda", dtype=torch.float16)  # 128 MiB
    spill.release(torch)
    with weights.recovering():
        reference = run(model, x, passes=1)
    assert not spill.PRESSED[0] and not spill.USED, "a run with room notes no op"
    model.residency.unpark()
    weights.set_budget(0)
    held = [torch.full((32 << 20,), float(i), device="cuda") for i in range(8)]  # 1 GiB
    torch.cuda.empty_cache()
    weights.measure_context()
    # Room for the input, the held tensors and the weights' floor: none for activations.
    cap = max(weights.context_bytes, 0) + torch.cuda.memory_reserved() + tight(model)
    moved = weights.facts().spilled_bytes
    weights.set_cap(cap + weight_policy.MARGIN)
    try:
        weights.set_budget(-1)
        with weights.recovering():  # as the executor runs every call
            capped = run(model, x, passes=1)
        facts = weights.facts()
    finally:
        weights.set_cap(-1)
    assert same(reference, capped)
    assert facts.spilled_bytes - moved >= 128 << 20
    assert all(t.is_cuda and bool((t == i).all()) for i, t in enumerate(held))
    assert spill.PRESSED[0] and spill.USED, "after a spill, ops are noted for the next one"
    del held, capped, reference
    gc.collect()
    spill.release(torch)
    assert not spill.HELD, "every moved tensor is gone"
    assert not spill.PRESSED[0] and not spill.USED, "no host block left: no op is noted"


def test_decoded_copies_count_in_a_stages_budget(tmp_path: Path) -> None:
    """A weight the card cannot run as stored (fp8 before sm89) is decoded into a float copy
    when its region binds. That copy is torch's memory beside the plane's stored bytes, so a
    stage's budget counts both: the plane's ring stays sized by the stored bytes alone."""
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=1 << 30)
    model = load(weights, write_checkpoint(tmp_path)[0], "decoded")
    component = model.residency.components["stack"]
    assert component.planned == component.layout
    for block in component.blocks:
        for weight in block.weights:
            weight.route = "decoded_float"
    stored = component.layout
    planned = component.planned
    copies = [
        sum(HIDDEN * HIDDEN * 2 for w in block.weights if w.row.name.endswith(".weight"))
        for block in component.blocks
    ]
    assert all(copy > 0 for copy in copies)
    assert planned.common == stored.common
    assert planned.blocks == tuple(
        size + copy + 2 * HIDDEN for size, copy in zip(stored.blocks, copies, strict=True)
    )
    # The same budget keeps fewer blocks resident, and the ring is the stored blocks' ring.
    budget = stored.common + 5 * stored.largest
    plain = weight_policy.residence(stored, budget)
    counted = weight_policy.residence(planned, budget)
    assert len(counted.resident) < len(plain.resident)
    ring = weight_policy.ring_bytes([stored.blocks[i] for i in counted.streamed], counted.window)
    assert ring < counted.ring


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


def refined(model: Any, name: str) -> weight_policy.Layout:
    """Move a component to the sub-block grain, as a budget below its block floor does."""
    model.residency.unpark()
    assert model.residency._refine(model.residency.components[name].key)
    component = model.residency.components[name]
    assert component.fine and len(component.blocks) > 1
    layout: weight_policy.Layout = component.layout
    return layout


@on_card
@pytest.mark.parametrize("pressure,fine", [(False, False), (True, False), (False, True)])
def test_an_sdxl_unet_streams_bit_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pressure: bool, fine: bool
) -> None:
    """SDXL's UNet resident, then streamed through a tight budget, computes the same bits.
    Under pressure the streamed arm alone also has a ballast leaving only the plane's budget
    and the resident run's measured activations free, as physical ballast does in G4: every
    streamed lease's bytes are checked against the store (`verify_bytes`), so a difference
    can only be the kernels' choice under memory pressure."""
    if fine:  # the sub-block grain of a UNet this small: units of at most 256 KiB
        monkeypatch.setattr(weights_module, "REGION_LIMIT", 256 << 10)
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
    if fine:
        layout = refined(model, "unet")
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


@on_card
def test_the_owner_lowers_a_running_process_cap_at_its_next_block(card: Any) -> None:
    """Room for another executor, or a display's floor, made while a call runs: the owner
    asks for a smaller process cap through the budget cell; from its next block the stage
    streams inside it and the process holds no more than the cap, with the same bits."""
    from cozy_runtime.internal import budget_cell

    weights = card.weights
    model = load(weights, card.checkpoint, "cap-cut")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    reference = run(model, x)
    torch.cuda.empty_cache()  # the cap is measured against what torch holds live
    weights.measure_context()
    context = max(weights.context_bytes, 0)
    cap = context + torch.cuda.memory_reserved() + tight(model) + weight_policy.MARGIN
    cell, fd = budget_cell.Cell.create()
    owner = budget_cell.Cell(fd)
    os.close(fd)
    weights.cell = cell
    model.residency.admit("run", ("stack",))
    try:
        with torch.inference_mode():
            outputs = [model.stack(x)]
            ticket = owner.ask(budget_cell.CAP - cap)
            outputs += [model.stack(x), model.stack(x)]
    finally:
        model.residency.release("run", ("stack",))
        weights.cell = None
        applied = owner.answered(ticket)
        cell.close()
        owner.close()
        weights.set_cap(-1)
    facts = weights.facts()
    assert applied is not None and applied == weights.applied
    assert facts.streamed["stack"].resident_blocks < BLOCKS
    assert context + torch.cuda.memory_reserved() + facts.committed_bytes <= cap
    assert same(reference, outputs)


class Bare(torch.nn.Module):  # type: ignore[misc,name-defined]
    """Blocks and nothing else: no `common`, so its stage holds no lease between blocks."""

    _no_split_modules = ("Block",)

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(BLOCKS)])

    def forward(self, x: Any) -> Any:
        for block in self.blocks:
            x = block(x)
        return x


@on_card
def test_a_block_whose_component_moves_to_the_sub_block_grain_under_it_runs_at_that_grain(
    tmp_path: Path,
) -> None:
    """Run 5894, `tensorfs.plane.Closed`: a stage that held no lease between blocks was placed
    again from inside a block, at its block floor, and moved the component to the sub-block
    grain. That closed the weight set the block had chosen, and its acquire read it. The block
    runs at the grain the component has now instead, with the same bits."""
    from cozy_runtime.internal import budget_cell

    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    generator = torch.Generator().manual_seed(11)
    values: dict[str, Any] = {}
    for index in range(BLOCKS):
        for name in ("proj", "out"):
            values[f"blocks.{index}.{name}.weight"] = (
                torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
            )
        values[f"blocks.{index}.norm"] = torch.rand(HIDDEN, generator=generator) + 0.5
    model = load(weights, store_component(tmp_path, "stack", values), "regrain", factory=Bare)
    layout: weight_policy.Layout = model.residency.components["stack"].layout
    assert model.residency.components["stack"].common is None and layout.fine is not None
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
            # One block at a time: the next block applies it before its acquire, and moves on.
            ticket = worker.ask(layout.common + layout.largest)
            outputs += [model.stack(x), model.stack(x)]
    finally:
        model.residency.release("run", ("stack",))
        weights.cell = None
        worker.answered(ticket)
        cell.close()
        worker.close()
    assert model.residency.components["stack"].fine
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
    into its input in place could not run again, and the load was refused. Such a block runs
    observed: its ops recover one by one, and its writes are journaled so the whole block can
    run again from the input it was given."""
    weights = card.weights
    model = load(weights, card.checkpoint, "writes", factory=WritingStack)
    x = torch.randn(8192, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x, passes=1)
    blocks = model.residency.components["stack"].blocks
    assert len(blocks) == BLOCKS
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
@pytest.mark.parametrize("cut", ["next stage", "worker", "oom"])
def test_weights_leave_only_after_the_kernels_that_read_them(tmp_path: Path, cut: str) -> None:
    """A stage's resident set is bound for the whole stage, and code reads it outside block
    forwards too (a tied embedding, a numerical check, a LoRA merge): those kernels hold no
    lease, and they are queued long before they run (here behind a one-second kernel).
    Whatever unmaps the set next (the next stage's weights, a Worker's mid-call cut, an
    out-of-memory cut) waits for them; unmapping under them was an illegal address (on a
    laptop, an MMU fault that can freeze it)."""
    from cozy_runtime.internal import budget_cell

    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    checkpoint = write_checkpoint(tmp_path)[0]
    first, second = load(weights, checkpoint, "first"), load(weights, checkpoint, "second")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    layout: weight_policy.Layout = first.residency.components["stack"].layout
    weights.set_budget(layout.total + (8 << 20))  # one construction's weights at a time

    def outside(model: Any) -> Any:  # a read outside every block forward
        return x @ model.stack.blocks[3].proj.weight.T

    first.residency.admit("run", ("stack",))
    with torch.inference_mode():
        reference = outside(first)
    first.residency.release("run", ("stack",))
    cell, fd = budget_cell.Cell.create()
    worker = budget_cell.Cell(fd)
    os.close(fd)
    weights.cell = cell
    try:
        with torch.inference_mode():
            first.residency.admit("run", ("stack",))
            torch.cuda._sleep(int(2e9))  # the GPU falls a second behind the host
            late = outside(first)  # reads a resident weight after the sleep
            if cut == "worker":
                worker.ask(layout.common)  # applied at the next block boundary
                first.stack(x)
            elif cut == "oom":
                weights.shrink(layout.total, 0)
            first.residency.release("run", ("stack",))
            if cut == "next stage":
                run(second, x, passes=1)  # wants the room: evicts the first's resident set
        torch.cuda.synchronize()
    finally:
        weights.cell = None
        cell.close()
        worker.close()
    assert torch.equal(reference, late)


@on_card
def test_an_op_that_no_unmap_can_help_fails_once(card: Any) -> None:
    """The 3 GiB freeze (W5, 21:53): an allocation retried about 40 times in 1.3 s while the
    driver's free bytes never grew, each retry unmapping weights, closing and remapping the
    stage's ring. A retry runs again only when memory came back: this process's unmapping,
    else the Worker's other tenants on the GPU (`DeviceRoom`); when neither can, it fails,
    naming the op and its size."""
    weights = card.weights
    model = load(weights, card.checkpoint, "stuck")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    layout: weight_policy.Layout = model.residency.components["stack"].layout
    weights.set_budget(tight(model))
    run(model, x, passes=1)
    calls = {"room": 0}
    make_room = weights.make_room

    def counted(need: int, operands: Any, op: str = "") -> None:
        calls["room"] += 1
        make_room(need, operands, op)

    weights.make_room = counted
    asked: list[int] = []
    weights.room = asked.append  # the Worker: no other tenant here gives anything
    model.residency.admit("run", ("stack",))
    try:
        # far beyond the card: no amount of unmapping makes room for it
        with (
            weights.recovering(),
            torch.inference_mode(),
            pytest.raises(Exception, match=r"aten\.empty.* of \d+ B: nothing on this GPU"),
        ):
            torch.empty(1 << 40, dtype=torch.uint8, device="cuda")
    finally:
        model.residency.release("run", ("stack",))
        del weights.make_room
        weights.room = None
        torch.cuda.empty_cache()
    assert calls["room"] <= 1 + len(layout.blocks)
    assert asked and all(n > 1 << 39 for n in asked)
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


class Table(torch.nn.Module):  # type: ignore[misc,name-defined]
    """A component with no blocks: all of it is its common region (a text encoder's
    embedding table is most of one)."""

    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.empty(4096, HIDDEN, dtype=torch.float16))

    def forward(self, x: Any) -> Any:
        rows = torch.full((x.shape[0],), 7, dtype=torch.long, device=x.device)
        return x * torch.nn.functional.embedding(rows, self.table)


def scoped(tmp_path: Path) -> tuple[Weights, WeightResidency, dict[str, Any]]:
    """One stage that scopes a block stack and a blockless encoder, as Anima's `render` scopes
    its conditioner, text encoder, transformer and VAE: each one's common region stays leased
    for all of it."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(1 << 30, pinned=1 << 30)
    stack_checkpoint, _ = write_checkpoint(tmp_path / "stack")
    generator = torch.Generator().manual_seed(11)
    encoder_checkpoint = store_component(
        tmp_path / "encoder", "encoder", {"table": torch.rand(4096, HIDDEN, generator=generator)}
    )
    checkpoints = {"stack": stack_checkpoint, "encoder": encoder_checkpoint}
    backend = PlaneBackend.for_script(
        checkpoints,
        [*stack_checkpoint.rows("stack"), *encoder_checkpoint.rows("encoder")],
        weights=weights,
        construction="spare",
        release="plane-test/1",
    )
    major, minor = torch.cuda.get_device_capability(0)
    with serving_substrate(f"sm{major}{minor}", Observations()):
        modules = {"stack": Stack(), "encoder": Table()}
    holder = SimpleNamespace(components=modules)
    walked = census(holder)
    backend.expect(
        {name: [d.key for d in walked.destinations if d.key.startswith(name)] for name in modules}
    )
    live = backend.materialize(holder, walked)
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    residency = WeightResidency(
        weights, backend.components, {"run": ("encoder", "stack")}, refine=backend.refine
    )
    return weights, residency, modules


@on_card
def test_a_stage_spares_its_idle_components_for_the_one_that_runs(tmp_path: Path) -> None:
    """At a 1.25 GiB pool Anima's second request's transformer had 173 MB for its activations
    beside 675 MB of idle components. While the running component cannot get memory, the
    others give theirs back; a read of one maps it for that op, and the results are the same
    bits."""
    weights, residency, modules = scoped(tmp_path)
    scope = ("encoder", "stack")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)

    residency.admit("run", scope)
    try:
        with torch.inference_mode():
            encoded = modules["encoder"](x)
            reference = modules["stack"](encoded)
    finally:
        residency.release("run", scope)
    encoder, stack = residency.components["encoder"], residency.components["stack"]
    common = encoder.common
    assert common is not None

    residency.admit("run", scope)
    try:
        with torch.inference_mode():
            assert torch.equal(modules["encoder"](x), encoded)
            mapped = weights.facts().committed_bytes
            assert residency._spare({stack.key}, {stack.key: 0})
            assert bool(encoder.spared) and encoder.lease is None and common.bound is None
            assert weights.facts().committed_bytes <= mapped - common.span
            # the running component is untouched, and a read of the spared one maps it
            assert torch.equal(modules["stack"](encoded), reference)
            assert torch.equal(modules["encoder"](x), encoded)
            assert common.bound is None, "mapped for that op only"
    finally:
        residency.release("run", scope)
    residency.unpark()  # a repeating stage stays open until the next request plans afresh
    assert not encoder.spared, "a stage's end clears it"
    assert weights.facts().oom_retries >= 1


@on_card
def test_an_op_out_of_memory_spares_the_components_not_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs 4360-4369 (Anima in a 1 GiB pool on an RTX 3070, a 758 MiB process cap): an op that
    recovers by itself (in a block's first run, in a kernel between blocks) asked for 36 MiB
    beside 420 MB of the stage's other components, and only a block's own retry spared them.
    When unmapping to the stage's floor gives nothing back, an op's retry spares them too:
    every component but the one whose block ran last and those the op reads."""
    weights, residency, modules = scoped(tmp_path)
    scope = ("encoder", "stack")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    encoder, stack = residency.components["encoder"], residency.components["stack"]
    common = encoder.common
    assert common is not None
    progressed, full = weights.progressed, [ResidencyRefusal(INDIVISIBLE, "a full card")]

    def nothing_came_back(*args: Any) -> None:
        if full:
            raise full.pop()
        progressed(*args)

    residency.admit("run", scope)
    try:
        with torch.inference_mode():
            encoded = modules["encoder"](x)
            reference = modules["stack"](encoded)  # the stack's blocks ran last
            mapped = weights.facts().committed_bytes
            monkeypatch.setattr(weights, "progressed", nothing_came_back)
            weights.make_room(1 << 20, (encoded,), "aten.clone.default")
            assert not full and encoder.spared and not stack.spared
            assert weights.facts().committed_bytes <= mapped - common.span
            assert torch.equal(modules["stack"](encoded), reference)
            assert torch.equal(modules["encoder"](x), encoded)
    finally:
        residency.release("run", scope)
    residency.unpark()


@on_gpu
def test_torch_is_held_above_the_drivers_floor_without_a_cap(card: Any) -> None:
    """Run 5422 (8 GB laptop): with the card at 10 MB free the driver could not allocate its
    own fault buffers (NV_ERR_NO_MEMORY) and libcuda faulted the executor. Torch is held to
    what the driver has free above the floor, cap or none: past it, torch runs out of memory
    (which the plane recovers from), never the driver."""
    weights = card.weights
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    weights.set_floor(free - (64 << 20))
    try:
        weights.limit_torch()
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        small = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
        del small
    finally:
        weights.set_floor(-1)
        weights.limit_torch()
        torch.cuda.empty_cache()
    torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


class Flat(Stack):
    """A component that declares no blocks, as SeedVR2's NaDiT declares none."""

    _no_split_modules = None


@on_gpu
def test_a_component_without_blocks_streams_inside_an_owners_cap(tmp_path: Path) -> None:
    """Run 5424: SeedVR2's 6.4 GB DiT declares no blocks, so it was one region on an 8 GB card,
    leased for its whole stage; nothing could unmap it, and the first activation that did not
    fit beside it was refused. A component without blocks pages at the sub-block grain: its
    ops between regions (`x * norm`, `tanh`) run out of memory at the cap, its weights leave
    for them, and it computes the same bits inside the cap."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    model = load(weights, write_checkpoint(tmp_path)[0], "flat", factory=Flat)
    component = model.residency.components["stack"]
    assert component.fine and len(component.blocks) == 2 * BLOCKS
    x = torch.randn(1 << 15, HIDDEN, device="cuda", dtype=torch.float16)  # 128 MiB each

    def recovered(x: Any) -> list[Any]:
        with weights.recovering():
            return run(model, x, passes=1)

    torch.cuda.reset_peak_memory_stats()
    live = torch.cuda.memory_allocated()
    uncapped = [t.cpu() for t in recovered(x)]  # off the card: its bytes are no room
    measured = torch.cuda.max_memory_allocated() - live
    model.residency.unpark()
    weights.set_budget(0)
    torch.cuda.empty_cache()
    weights.measure_context()
    context = max(weights.context_bytes, 0)
    # Room for the activations and a margin, not for the weights beside them as well (what
    # torch holds live, not its cache: earlier tests' segments are no room).
    cap = context + torch.cuda.memory_allocated() + measured + weight_policy.MARGIN
    assert component.layout.total > weight_policy.MARGIN
    weights.activation.clear()  # unmeasured: the stage keeps every weight it can
    retries = weights.facts().oom_retries
    torch.cuda.reset_peak_memory_stats()
    weights.set_cap(cap)
    try:
        weights.set_budget(-1)
        capped = [t.cpu() for t in recovered(x)]
        peak = torch.cuda.max_memory_reserved()
        facts = weights.facts()
    finally:
        weights.set_cap(-1)
    del x
    torch.cuda.empty_cache()
    # It gave weights back for its activations: unmapped on an out-of-memory, or streamed.
    gave = facts.oom_retries > retries or facts.committed_bytes < component.layout.total
    assert gave and facts.modes["stack.grain"] == "sub-block"
    assert peak + context <= cap
    assert same(uncapped, capped)


class Twice(Block):
    """Its projection called twice in a row, as SeedVR2 calls a shared-weight projection once
    for video and once for text."""

    def forward(self, x: Any) -> Any:
        return x + torch.tanh(self.out(torch.tanh(self.proj(self.proj(x * self.norm)))))


class FlatTwice(Flat):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Twice() for _ in range(BLOCKS)])


@on_gpu
def test_a_module_called_again_right_after_itself_streams_once(tmp_path: Path) -> None:
    """Run 5540: SeedVR2's shared-weight projections are called twice in a row. The cursor
    served the second call by skipping a whole pass, copying every block and handing it on
    unread: 300 GB of copies in one 17-frame window. A block not due within the cursor's
    window is filled on demand instead, and the stream copies each pass about once."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(1 << 30, pinned=1 << 30)
    model = load(weights, write_checkpoint(tmp_path)[0], "twice", factory=FlatTwice)
    layout: weight_policy.Layout = model.residency.components["stack"].layout
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    resident = run(model, x)
    model.residency.unpark()
    weights.set_budget(layout.common + 4 * layout.largest)
    run(model, x, passes=2)  # the order is learned, then streamed
    before = weights.facts().h2d_bytes
    streamed = run(model, x)
    assert weights.facts().streamed["stack"].resident_blocks < len(layout.blocks)
    per_pass = (weights.facts().h2d_bytes - before) / 3
    assert per_pass < 2 * layout.total, per_pass / layout.total
    assert same(resident, streamed)


@on_gpu
def test_a_stage_with_no_component_known_in_use_spares_them_all(tmp_path: Path) -> None:
    """Run 5439: no block of the stage had run, so nothing was known to be in use, and the
    stage's components kept their weights while an op could not get memory. Not knowing which
    is in use, every one the op does not read gives its weights back; a read maps them again
    for that op, and the results are the same bits."""
    weights, residency, modules = scoped(tmp_path)
    scope = ("encoder", "stack")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    encoder, stack = residency.components["encoder"], residency.components["stack"]
    residency.admit("run", scope)
    try:
        with torch.inference_mode():
            reference = modules["stack"](modules["encoder"](x))
            mapped = weights.facts().committed_bytes
            assert residency._spare({"another construction/vae"}, {})
            assert encoder.spared and stack.spared
            assert weights.facts().committed_bytes < mapped
            assert torch.equal(modules["stack"](modules["encoder"](x)), reference)
    finally:
        residency.release("run", scope)
    residency.unpark()


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


# --------------------------------------- a DiT, a request LoRA, a model's consumer overlay

DIT_HIDDEN, DIT_LAYERS, DIT_RANK = 512, 16, 8
#: The overlay's linears: two under the request LoRA's PEFT layers, one that only it adapts.
OVERLAY_SITES = {
    f"transformer_blocks.{block}.{site}": dims
    for block in range(DIT_LAYERS)
    for site, dims in (
        ("attn.to_q", (DIT_HIDDEN, DIT_HIDDEN)),
        ("attn.to_v", (DIT_HIDDEN, DIT_HIDDEN)),
        ("ff.net.2", (4 * DIT_HIDDEN, DIT_HIDDEN)),
    )
}


def lora_dit() -> Any:
    """Diffusers' MiniMax-H3 DiT, small, with a model's consumer hooks and a request LoRA
    composed as `--lora` composes it: PEFT layers over its attention and output projections."""
    pytest.importorskip("peft")
    from diffusers import MiniMaxH3Transformer3DModel

    from cozy_runtime.author import AdapterRef
    from cozy_runtime.internal.lora_composition import composer
    from cozy_runtime.internal.lora_contract import FORMAT, Graph, Linear
    from testdata.consumer_lora import install

    dit: Any = MiniMaxH3Transformer3DModel(
        num_attention_heads=4,
        attention_head_dim=DIT_HIDDEN // 4,
        hidden_size=DIT_HIDDEN,
        num_layers=DIT_LAYERS,
        num_refiner_layers=1,
        ffn_dim=4 * DIT_HIDDEN,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=4,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    install(dit.eval(), OVERLAY_SITES)
    paths = [
        f"transformer_blocks.{block}.{site}"
        for block in range(DIT_LAYERS)
        for site in ("attn.to_q", "attn.to_out.0", "ff.net.2")
    ]
    graph = Graph(
        FORMAT,
        tuple(Linear("fl2va_dit", path, "wushu", DIT_RANK, 8.0, 0.8, "f16") for path in paths),
        (AdapterRef("sha256:" + "12" * 32, 0.8, "lora", "fl2va_dit", "adapter"),),
    )
    composer(graph)(SimpleNamespace(components={"fl2va_dit": dit}))
    return dit.to(torch.float16)


def consumer_overlay() -> Any:
    """The model's consumer LoRA: its own model slot, whose factors the DiT's hooks read."""
    from testdata.consumer_lora import Overlay

    return Overlay(OVERLAY_SITES, DIT_RANK, 1.0).eval().to(torch.float16)


def dit_inputs(device: str) -> dict[str, Any]:
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


def lora_models(weights: Weights, root: Path, key: str) -> tuple[Any, Any]:
    """The DiT with its request LoRA and the consumer overlay: two model slots on one plane."""
    slots = []
    for name, factory, scope in (
        ("fl2va_dit", lora_dit, "sample_adapted"),
        ("fl2va_overlay", consumer_overlay, "sample_fl2va"),
    ):
        torch.manual_seed(292)
        values = {k: torch.randn(v.shape) * 0.05 for k, v in factory().state_dict().items()}
        (root / name).mkdir()
        checkpoint = store_component(root / name, name, values)
        slots.append(
            load(weights, checkpoint, f"{key}-{name}", name=name, factory=factory, scope=scope)
        )
    return slots[0], slots[1]


def lora_denoise(
    weights: Weights, dit: Any, adapter: Any, inputs: dict[str, Any], between: Any
) -> Any:
    """Three evaluations of the DiT with the overlay armed, its scopes nested as a package's
    adapted sampler nests them: the base's stage, the overlay's inside it, package arithmetic
    between."""
    from testdata.consumer_lora import armed

    video, audio = inputs["hidden_states"], inputs["audio_hidden_states"]
    dit.residency.admit("sample_adapted", ("fl2va_dit",))
    try:
        adapter.residency.admit("sample_fl2va", ("fl2va_overlay",))
        try:
            with weights.recovering(), torch.inference_mode(), armed(dit.stack, adapter.stack):
                for step in range(3):
                    held = between(step)
                    moved = dit.stack(
                        **{**inputs, "hidden_states": video, "audio_hidden_states": audio}
                    )
                    video, audio = video + moved[0] * 0.25, audio + moved[1] * 0.25
                    del held
                return video, audio
        finally:
            adapter.residency.release("sample_fl2va", ("fl2va_overlay",))
    finally:
        dit.residency.release("sample_adapted", ("fl2va_dit",))


def test_a_dit_with_its_loras_computes_on_the_pinned_tier(tmp_path: Path) -> None:
    """The fixture is the real thing: on a CPU executor (weights bound in place) the DiT with
    a request LoRA and the consumer overlay armed computes what the same modules compute
    unmanaged."""
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=1 << 30)
    dit, adapter = lora_models(weights, tmp_path, "host")
    inputs = dit_inputs("cpu")
    got = lora_denoise(weights, dit, adapter, inputs, lambda step: None)
    plain = []
    for factory in (lora_dit, consumer_overlay):
        torch.manual_seed(292)
        module = factory()
        module.load_state_dict(
            {k: (torch.randn(v.shape) * 0.05).half() for k, v in module.state_dict().items()}
        )
        plain.append(module)
    plain_dit, plain_overlay = plain
    unmanaged = SimpleNamespace(admit=lambda *_: None, release=lambda *_: None)
    want = lora_denoise(
        weights,
        SimpleNamespace(stack=plain_dit, residency=unmanaged),
        SimpleNamespace(stack=plain_overlay, residency=unmanaged),
        inputs,
        lambda step: None,
    )
    assert all(torch.isfinite(t.float()).all() for t in got)
    assert same(list(got), list(want))


@on_card
def test_a_dit_with_its_loras_pages_instead_of_running_out(tmp_path: Path) -> None:
    """Run 2549: H3's DiT with a request LoRA under its turbo LoRA ran out of device memory.
    Here its weights get a budget far below their size, and mid-denoise another allocation
    takes the room its activations had. It still computes the fully resident run's bits: the
    overlay's stage shares the DiT's budget, blocks stream, and the allocation that fails
    unmaps weights and runs again."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    dit, adapter = lora_models(weights, tmp_path, "dit")
    inputs = dit_inputs("cuda")
    layout: weight_policy.Layout = dit.residency.components["fl2va_dit"].layout
    overlay = adapter.residency.components["fl2va_overlay"].layout.total
    weights.set_budget(1 << 30)
    resident, peak = lora_reference(weights, dit, adapter, inputs)
    assert weights.facts().streamed["fl2va_dit"].resident_blocks == len(layout.blocks)

    spare = 6 * layout.largest  # what an out-of-memory retry can still unmap
    weights.set_budget(weight_policy.floor(layout) + overlay + spare)
    assert weights.budget < (layout.total + overlay) // 2
    was = weights.facts()
    try:
        paged = lora_denoise(weights, dit, adapter, inputs, squeezing(1, peak))
    finally:
        torch.cuda.empty_cache()
    now = weights.facts()
    assert now.streamed["fl2va_dit"].resident_blocks < len(layout.blocks)
    assert now.h2d_bytes > was.h2d_bytes and now.evictions > was.evictions
    assert now.oom_retries > was.oom_retries
    assert same(list(resident), list(paged))


@on_card
def test_a_dit_with_its_loras_at_the_sub_block_grain_is_bit_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DiT (its LoRA consumers, the overlay's factors read from its blocks) with its
    blocks split into their projections computes the resident run's bits."""
    monkeypatch.setattr(weights_module, "REGION_LIMIT", 1 << 20)
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    dit, adapter = lora_models(weights, tmp_path, "dit-fine")
    inputs = dit_inputs("cuda")
    weights.set_budget(1 << 30)
    resident, _ = lora_reference(weights, dit, adapter, inputs)
    adapter.residency.unpark()
    layout = refined(dit, "fl2va_dit")
    overlay = adapter.residency.components["fl2va_overlay"].layout.total
    weights.set_budget(layout.common + overlay + 3 * layout.largest)
    try:
        paged = lora_denoise(weights, dit, adapter, inputs, lambda step: None)
    finally:
        torch.cuda.empty_cache()
    assert weights.facts().streamed["fl2va_dit"].resident_blocks < len(layout.blocks)
    assert same(list(resident), list(paged))


def lora_reference(weights: Weights, dit: Any, adapter: Any, inputs: dict[str, Any]) -> Any:
    """The fully resident denoise and its activation peak."""
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    resident = lora_denoise(weights, dit, adapter, inputs, lambda step: None)
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
    dit, adapter = lora_models(weights, tmp_path, "dit")
    (tmp_path / "idle").mkdir()
    idle = load(weights, write_checkpoint(tmp_path / "idle")[0], "idle")
    inputs = dit_inputs("cuda")
    layout: weight_policy.Layout = dit.residency.components["fl2va_dit"].layout
    weights.set_budget(1 << 30)
    resident, peak = lora_reference(weights, dit, adapter, inputs)
    calls = {"block": 0}
    block = dit.stack.transformer_blocks[0].attn
    block.register_forward_hook(lambda *_: calls.__setitem__("block", calls["block"] + 1))

    weights.grouped = True
    weights.activation.clear()  # a fresh executor's first request: nothing measured
    grown = lora_denoise(weights, dit, adapter, inputs, lambda step: None)
    assert same(list(resident), list(grown)) and calls["block"] == 3
    assert weights.facts().streamed["fl2va_dit"].resident_blocks == len(layout.blocks)
    assert weights.facts().h2d_bytes > 0  # the first pass streamed every block

    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    run(idle, x, passes=1)
    held = weights.resident_bytes(idle.residency.components["stack"])
    retries = weights.facts().oom_retries
    try:
        squeezed = lora_denoise(weights, dit, adapter, inputs, squeezing(1, peak))
    finally:
        torch.cuda.empty_cache()
    assert weights.facts().oom_retries > retries
    assert weights.resident_bytes(idle.residency.components["stack"]) < held
    assert same(list(resident), list(squeezed)) and calls["block"] == 6


@on_card
def test_a_group_gpus_denoise_keeps_a_resident_set_after_a_one_pass_warm_up(card: Any) -> None:
    """Run 2963 (H3 on 2x A40): the leader GPU ran the whole denoise with every DiT block
    streamed through a window of two, nothing resident, while the other GPU kept half the
    blocks on the card. Placed again at its second pass, the stage was judged run-once
    because the component's last stage (the warm-up) had made one pass. A stage on its
    second pass is cyclic."""
    weights = card.weights
    model = load(weights, card.checkpoint, "leader")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)

    def staged(method: str, passes: int) -> list[Any]:
        model.residency.admit(method, ("stack",))
        try:
            with torch.inference_mode():
                return [model.stack(x) for _ in range(passes)]
        finally:
            model.residency.release(method, ("stack",))

    weights.set_budget(1 << 30)
    resident = staged("reference", 3)
    model.residency.open_attempt()
    weights.grouped = True
    weights.activation.clear()  # a fresh executor's first request: nothing measured
    layout: weight_policy.Layout = model.residency.components["stack"].layout
    weights.set_budget(layout.common + (BLOCKS // 2 + 2) * layout.largest)
    staged("warm", 1)
    # The warm-up's first launches (library workspaces) are not the denoise's activations.
    weights.activation.clear()
    denoised = staged("denoise", 3)
    kept = weights.facts().streamed["stack"].resident_blocks
    assert 0 < kept < BLOCKS, kept
    assert same(resident, denoised)


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
    ballast = scratch = None  # a refusal below is the failure reported, not an unbound name
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


def kl_vae(channels: tuple[int, ...] = (64, 128)) -> Any:
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(5)
    return diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * len(channels),
        up_block_types=("UpDecoderBlock2D",) * len(channels),
        block_out_channels=channels,
    ).to(torch.float16)


def plane_vae(weights: Weights, root: Path, factory: Any) -> tuple[Any, Any]:
    """`factory`'s VAE on the plane, and the same weights as a plain module on the card."""
    values = {k: v.float() for k, v in factory().state_dict().items()}
    root.mkdir()
    model = load(weights, store_component(root, "vae", values), "vae", name="vae", factory=factory)
    plain = factory().cuda().eval()
    plain.load_state_dict({k: v.half() for k, v in values.items()})
    return model, plain


def decode(model: Any, z: Any, cap: int = -1) -> Any:
    weights = model.residency.weights
    weights.modes.clear()  # as every call starts
    if cap >= 0:
        weights.set_cap(cap)
    model.residency.admit("run", ("vae",))
    try:
        with weights.recovering(), torch.inference_mode():
            return model.stack.decode(z).sample
    finally:
        model.residency.release("run", ("vae",))
        weights.set_cap(-1)
        torch.cuda.empty_cache()


def room(weights: Weights, nbytes: int) -> int:
    """An owner's cap that leaves `nbytes` for activations beside what the process holds."""
    torch.cuda.empty_cache()
    weights.measure_context()
    held = max(weights.context_bytes, 0) + plane.stats(weights.plane).device(0).committed
    return held + int(torch.cuda.memory_reserved()) + nbytes


@on_gpu
def test_an_image_decode_short_of_memory_is_still_the_untiled_image(tmp_path: Path) -> None:
    """Run 4281 (SDXL in 1.5 GiB) fell back to a blended tiled decode 30 dB from its control.
    Diffusers' own decode runs as it is: with room, untouched; under a cap too small for its
    maps, an op that runs out runs in bands of rows, its result on the card when the tensors
    used least recently can move to host memory, else in host memory, and the ops after it
    read host rows over the bus once each. Either way the image is the untiled one to fp16
    rounding. The cap is the process's own: the card is never filled."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    model, plain = plane_vae(weights, tmp_path / "vae", kl_vae)
    z = torch.randn(1, 4, 256, 256, device="cuda", dtype=torch.float16)  # 512x512 pixels
    with torch.inference_mode():
        reference = plain.decode(z).sample
    del plain
    roomy = decode(model, z)
    assert "bands" not in weights.modes
    short = decode(model, z, room(weights, 96 << 20))
    assert "bands" in weights.modes  # it ran out, and ops ran in bands
    for image in (roomy, short):
        torch.testing.assert_close(image, reference, atol=1e-2, rtol=0)


@on_gpu
def test_a_banded_result_out_of_memory_inside_a_nested_recovery_lives_in_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 6191 (qwen-image-2 at 4 MP): a block's recovery runs inside the call's. A banded
    op's result, which the room said would fit, ran out of memory, and the call's recovery,
    taking the bands' own allocation for the package's op, refused it. The bands' ops reach
    no mode: the result goes to host memory and the image is the untiled one."""
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    model, plain = plane_vae(weights, tmp_path / "vae", kl_vae)
    z = torch.randn(1, 4, 256, 256, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reference = plain.decode(z).sample
    del plain
    monkeypatch.setattr(weights, "activation_room", lambda: (1 << 40, 0))  # every result fits
    weights.modes.clear()
    weights.set_cap(room(weights, 96 << 20))
    model.residency.admit("run", ("vae",))
    try:
        with weights.recovering(), weights.recovering(Journal()), torch.inference_mode():
            image = model.stack.decode(z).sample
    finally:
        model.residency.release("run", ("vae",))
        weights.set_cap(-1)
        torch.cuda.empty_cache()
    assert "host memory" in weights.modes["bands"]
    torch.testing.assert_close(image, reference, atol=1e-2, rtol=0)


@on_gpu
def test_a_qwen_image_vae_decodes_exactly_on_the_plane(tmp_path: Path) -> None:
    """Anima's VAE (diffusers' AutoencoderKLQwenImage) decodes a lone frame through causal 3D
    convolutions, two of each one's three taps on zero padding: each runs as the 2D
    convolution of the tap that meets the frame. With room and short of it, the image is the
    stock decode's to fp16."""
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(5)

    def vae() -> Any:
        return diffusers.AutoencoderKLQwenImage(base_dim=32, num_res_blocks=1).to(torch.float16)

    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(1 << 30, pinned=1 << 30)
    model, plain = plane_vae(weights, tmp_path / "vae", vae)
    z = torch.randn(1, 16, 1, 64, 64, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reference = plain.decode(z).sample
    del plain
    images = [decode(model, z), decode(model, z, room(weights, 48 << 20))]
    assert "host memory" in weights.modes["bands"]
    for image in images:
        assert image.shape == (1, 3, 1, 512, 512)
        torch.testing.assert_close(image, reference, atol=1e-2, rtol=0)


@on_card
def test_a_decode_unmaps_idle_weights_only_for_the_room_it_needs(card: Any, tmp_path: Path) -> None:
    """Run 2955: the first decode at a size unmapped every idle weight, whatever was free, and
    the model's next request uploaded all 6.4 GiB again. A decode that fits beside the idle
    weights unmaps none; on a card with 8 MiB left they leave first (activations first) and
    its maps stay on the card."""
    weights = card.weights
    weights.set_budget(1 << 30)
    idle = load(weights, card.checkpoint, "idle")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    stack = idle.residency.components["stack"]
    model, plain = plane_vae(weights, tmp_path / "vae", lambda: kl_vae((32, 64)))
    latents = [
        torch.randn(1, 4, side, side, device="cuda", dtype=torch.float16) for side in (64, 128)
    ]
    with torch.inference_mode():
        references = [plain.decode(z).sample for z in latents]
    del plain

    def squeezed(z: Any, left: int = 0) -> Any:
        ballast = None
        model.residency.admit("run", ("vae",))
        try:
            with weights.recovering(), torch.inference_mode():
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
                ballast = torch.empty(free - left if left else 0, dtype=torch.uint8, device="cuda")
                return model.stack.decode(z).sample
        finally:
            model.residency.release("run", ("vae",))
            del ballast
            torch.cuda.empty_cache()

    run(idle, x, passes=1)
    torch.cuda.synchronize()  # its last copies land
    was, mapped = weights.facts(), weights.resident_bytes(stack)
    assert mapped > 0
    image = squeezed(latents[0])
    assert weights.facts().evictions == was.evictions and weights.resident_bytes(stack) >= mapped
    torch.testing.assert_close(image, references[0], atol=1e-2, rtol=0)
    weights.modes.clear()
    image = squeezed(latents[1], left=8 << 20)
    assert "host memory" not in weights.modes.get("bands", "")
    assert weights.resident_bytes(stack) < mapped
    torch.testing.assert_close(image, references[1], atol=1e-2, rtol=0)


@on_gpu
def test_a_decode_takes_no_room_from_its_stages_floor(card: Any, tmp_path: Path) -> None:
    """Run 3143 (Anima with 3 GiB free): the VAE decodes inside a stage that also holds a
    streamed component. Making room for the decode cut the plane to what was resident then,
    under that stage's floor; the VAE's next block had no room to arrive in, the plan it asked
    for was refused at every tile size, and the run ended `device_shortfall` with 2.6 GiB
    allocatable. The room a decode takes is never its open stages' floor."""
    diffusers = pytest.importorskip("diffusers")
    weights = card.weights
    weights.set_budget(-1, pinned=2 << 30)
    denoiser = load(weights, card.checkpoint, "beside", scope="render")

    def vae() -> Any:
        torch.manual_seed(5)
        return diffusers.AutoencoderKL(
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            block_out_channels=(256, 512),
        ).to(torch.float16)

    values = {k: v.float() for k, v in vae().state_dict().items()}
    (tmp_path / "vae").mkdir()
    model = load(
        weights, store_component(tmp_path / "vae", "vae", values), "kl", name="vae", factory=vae
    )
    plain = vae().cuda().eval()
    plain.load_state_dict({k: v.half() for k, v in values.items()})
    z = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reference = plain.decode(z).sample
    del plain
    # Both stream: the VAE in regions larger than the stage's common weights together.
    stack = denoiser.residency.components["stack"].layout
    fine = refined(model, "vae")
    assert fine.largest > stack.common + fine.common
    weights.set_budget(weight_policy.floor(stack) + weight_policy.floor(fine) + (2 << 20))
    denoiser.residency.admit("render", ("stack",))
    try:
        model.residency.admit("run", ("vae",))
        try:
            with weights.recovering(), torch.inference_mode():
                denoiser.stack(x)  # its last blocks stay mapped: room the decode may take
                image = model.stack.decode(z).sample
            assert weights.applied >= weights.floor()
        finally:
            model.residency.release("run", ("vae",))
    finally:
        denoiser.residency.release("render", ("stack",))
        torch.cuda.empty_cache()
    torch.testing.assert_close(image, reference, atol=1e-2, rtol=0)
