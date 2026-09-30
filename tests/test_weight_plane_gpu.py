"""The executor's weights on the real weight plane, on the card (weight-plane.md §4).

Real: a TensorFS store holding a block model's weights, `tensorfs.plane` on device 0, the
runtime's `PlaneBackend` and `WeightResidency`, torch on CUDA. The model is small and exact:
a stack of declared no-split blocks whose forward reads every weight, so any byte streamed
wrong changes the output. Run it under the shared GPU lock: it holds the whole card.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tensorfs import Store  # noqa: E402
from tensorfs.derived import Config as NativeConfig  # noqa: E402
from tensorfs.derived import Derivation, Part, Target, Tensor  # noqa: E402

from cozy_runtime.author._loader import census  # noqa: E402
from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.encoding import SPEC_PLAIN  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights  # noqa: E402
from cozy_runtime.internal.weights_sink import weights_transaction_id  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not plane.available(),
    reason="the weight plane on a CUDA card",
)

HIDDEN = 2048
BLOCKS = 8


class Block(torch.nn.Module):  # type: ignore[misc,name-defined]
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.norm = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))

    def forward(self, x: Any) -> Any:
        return x + torch.tanh(self.proj(x * self.norm))


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


def write_checkpoint(root: Path) -> Checkpoint:
    """Sane fp16 weights (a fixed seed, small values) in a real TensorFS store."""
    generator = torch.Generator().manual_seed(7)
    values: dict[str, Any] = {"embed": torch.rand(HIDDEN, generator=generator) + 0.5}
    for index in range(BLOCKS):
        values[f"blocks.{index}.proj.weight"] = (
            torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
        )
        values[f"blocks.{index}.norm"] = torch.rand(HIDDEN, generator=generator) + 0.5
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
            targets={"stack": Target(add=tensors)},
            configs={"stack": NativeConfig("add")},
            order=tuple(("stack", key) for key in values),
        ).native_arguments(4096),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for key, value in values.items():
        writer.add_part("stack", key, "value", io.BytesIO(value.half().numpy().tobytes()))
    writer.add_config("stack", io.BytesIO(b"{}"))
    receipt = writer.commit()
    return Checkpoint(root / "store", "sha256:" + receipt["manifest"]["sha256"])


def load(weights: Weights, checkpoint: Checkpoint, key: str, *, reverse: bool = False) -> Any:
    """Construct on meta and register with the plane, as the executor's prepare does."""
    backend = PlaneBackend.for_script(
        {"stack": checkpoint},
        checkpoint.rows("stack"),
        weights=weights,
        construction=key,
        release="plane-test/1",
    )
    with torch.device("meta"):
        stack = Stack(reverse=reverse)
    holder = SimpleNamespace(components={"stack": stack})
    walked = census(holder)
    backend.expect({"stack": [d.key for d in walked.destinations]})
    live = backend.materialize(holder, walked)
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    residency = WeightResidency(weights, backend.components, {"run": ("stack",)})
    return SimpleNamespace(stack=stack, residency=residency, backend=backend)


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
    layout = model.residency.components["stack"].layout
    return layout.common + 3 * layout.largest


@pytest.fixture
def card(tmp_path: Path) -> Any:
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    return SimpleNamespace(weights=weights, checkpoint=write_checkpoint(tmp_path))


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


def test_an_oom_at_a_block_boundary_unmaps_weights_and_retries(card: Any) -> None:
    weights = card.weights
    model = load(weights, card.checkpoint, "oom")
    x = torch.randn(8192, HIDDEN, device="cuda", dtype=torch.float16)
    weights.set_budget(1 << 30)
    reference = run(model, x, passes=1)
    free, _ = torch.cuda.mem_get_info()
    # Leave less than one block's activations free beside the fully resident weights.
    ballast = torch.empty(free - (48 << 20), dtype=torch.uint8, device="cuda")
    try:
        retried = run(model, x, passes=1)
    finally:
        del ballast
    assert weights.facts().oom_retries >= 1
    assert same(reference, retried)


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
