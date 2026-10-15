"""A stage whose room makes torch drop its cache again and again gets more: the components it
does not read leave the device before its next entry, and fill again when their own stage
opens. Under a per-process limit torch refuses a request when its reserved bytes plus the
request pass the limit, even one that only has to extend a block it holds, so a peak that fits
can still squeeze. Real planes, a real allocator under a real cap, on one CUDA card."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import accel  # noqa: E402
from cozy_runtime.internal import weights as weights_module  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402
from test_device_tier import Custody, executor_with  # noqa: E402
from test_weight_plane import (  # noqa: E402
    BLOCKS,
    HIDDEN,
    load,
    run,
    same,
    store_component,
    write_checkpoint,
)

pytestmark = pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card")
MIB = 1 << 20
ROOM = 600 * MIB


def other_checkpoint(root: Path) -> Any:
    """A second component ("other") with weights of its own, in a store of its own."""
    generator = torch.Generator().manual_seed(11)
    values: dict[str, Any] = {"embed": torch.rand(HIDDEN, generator=generator) + 0.5}
    for index in range(BLOCKS):
        for name in ("proj", "out"):
            values[f"blocks.{index}.{name}.weight"] = (
                torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
            )
        values[f"blocks.{index}.norm"] = torch.rand(HIDDEN, generator=generator) + 0.5
    return store_component(root, "other", values)


def rest(model: Any, x: Any) -> list[Any]:
    """One entry of the other component's own stage."""
    model.residency.admit("rest", ("other",))
    try:
        with torch.inference_mode():
            return [model.stack(x)]
    finally:
        model.residency.release("rest", ("other",))


def growing(model: Any, x: Any) -> list[Any]:
    """One entry of "run": the stack's forward, then three buffers each larger than the last
    (256, 384, 512 MiB), each freed at once. Every one has to grow what torch holds."""
    model.residency.admit("run", ("stack",))
    try:
        with torch.inference_mode():
            out = [model.stack(x)]
            for mib in (256, 384, 512):
                torch.empty(mib * MIB, dtype=torch.uint8, device="cuda")
            return out
    finally:
        model.residency.release("run", ("stack",))


def capped(weights: Weights, committed: int, room: int = ROOM) -> None:
    """Hold the process to its context, `committed` weight bytes, torch's live bytes and `room`."""
    weights.measure_context()
    torch.cuda.empty_cache()
    live = int(torch.cuda.memory_allocated())
    weights.set_cap(max(weights.context_bytes, 0) + committed + live + room)


def test_a_squeezed_stage_pages_out_the_component_it_does_not_read(tmp_path: Path) -> None:
    a_at, _ = write_checkpoint(tmp_path / "a")
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    weights.set_cap(4 << 30)  # under an owner's cap other stages' weights stay mapped
    a = load(weights, a_at, "a")
    b = load(weights, other_checkpoint(tmp_path / "b"), "b", name="other", scope="rest")
    try:
        x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
        expected, expected_rest = run(a, x, passes=1), rest(b, x)
        stack, other = a.backend.components["stack"], b.backend.components["other"]
        own, idle = weights.resident_bytes(stack), weights.resident_bytes(other)
        assert own > 0 and idle > 0
        capped(weights, own + idle)

        # The peak (512 MiB) fits the room (600 MiB), and torch still drops its cache twice:
        # 256 cached + 384 asked and 384 cached + 512 asked both pass its limit.
        room, dropped = weights.allocatable(), weights.retries()
        assert same(growing(a, x), expected)
        assert weights.retries() - dropped == 2
        squeezed = weights.facts().squeezed["run"]
        assert abs(squeezed - room) < 8 * MIB, "the room that entry had"
        assert weights.resident_bytes(other) == idle, "nothing left the device during it"

        # Before the next entry the idle component leaves: torch has its bytes too. From an
        # empty cache (as a new process starts) one drop is left: the last buffer's.
        torch.cuda.empty_cache()
        dropped = weights.retries()
        assert same(growing(a, x), expected)
        assert weights.retries() - dropped == 1
        assert (weights.resident_bytes(stack), weights.resident_bytes(other)) == (own, 0)
        assert weights.facts().squeezed["run"] == squeezed, "that entry did not squeeze"

        # Its own stage fills it again, unmodified.
        copied = weights.facts().h2d_bytes
        assert same(rest(b, x), expected_rest)
        assert weights.facts().h2d_bytes - copied == idle

        # A component an open stage scopes (spared by `_spare` or not) is never paged out:
        # with both stages open there is nothing to give.
        b.residency.admit("rest", ("other",))
        a.residency.admit("run", ("stack",))
        try:
            assert weights.page_out(weights_module._in_use(weights)) is False
        finally:
            a.residency.release("run", ("stack",))
            b.residency.release("rest", ("other",))
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        a.backend.close()
        b.backend.close()


def test_a_stage_squeezed_inside_one_entry_pages_out_at_its_next_block(tmp_path: Path) -> None:
    """One entry can be a whole denoise: the idle component leaves at the next block boundary
    once torch has dropped its cache twice, and what the entry asks after that fits."""
    a_at, _ = write_checkpoint(tmp_path / "a")
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(-1, pinned=1 << 30)
    weights.set_cap(4 << 30)
    a = load(weights, a_at, "a")
    b = load(weights, other_checkpoint(tmp_path / "b"), "b", name="other", scope="rest")
    try:
        x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
        expected, expected_rest = run(a, x, passes=1), rest(b, x)
        stack, other = a.backend.components["stack"], b.backend.components["other"]
        own, idle = weights.resident_bytes(stack), weights.resident_bytes(other)
        capped(weights, own + idle)
        retries: list[int] = []
        a.residency.admit("run", ("stack",))
        try:
            with torch.inference_mode():
                room = weights.allocatable()
                # what the stage's plan left of the idle component (its LRU may take some)
                held = weights.resident_bytes(other)
                dropped = weights.retries()
                # Each larger than the last, each fits alone, and each with the one before it
                # passes torch's limit: two cache drops.
                for share in (0.48, 0.54, 0.55):
                    torch.empty(int(room * share), dtype=torch.uint8, device="cuda")
                retries.append(weights.retries() - dropped)
                out = [a.stack(x)]  # its first block boundary: the idle component leaves
                retries.append(weights.retries())
                # Larger than the block torch keeps (so it has to grow it): over the room
                # beside that block, and under that room plus the idle component's bytes.
                larger = int(room * 0.56)
                assert room < int(room * 1.11) < room + held - 16 * MIB
                torch.empty(larger, dtype=torch.uint8, device="cuda")
                retries.append(weights.retries() - retries[1])
        finally:
            a.residency.release("run", ("stack",))
        assert retries[0] == 2 and retries[2] == 0, retries
        assert same(out, expected) and weights.resident_bytes(other) == 0
        facts = weights.facts()
        assert facts.paged_out == 1 and facts.paged_out_bytes >= held  # with what was filling
        assert abs(weights.facts().squeezed["run"] - room) < 8 * MIB
        assert same(rest(b, x), expected_rest)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        a.backend.close()
        b.backend.close()


def test_a_component_mapped_from_the_machine_is_given_back_and_fills_again(tmp_path: Path) -> None:
    """Degree 2: the idle component's bytes are the machine's. The executor lets its mapping go
    and asks for room; the machine drops the holding nobody maps and raises the cap. A room an
    earlier executor squeezed in is seeded, so this one pages before its first entry."""
    a_at, _ = write_checkpoint(tmp_path / "a")
    b_at = other_checkpoint(tmp_path / "b")
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    custody = Custody()
    opened: list[Any] = []
    try:
        filler, _ = executor_with(custody, tmp_path / "filler")
        if not filler.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        first = load(filler, a_at, "a"), load(filler, b_at, "b", name="other", scope="rest")
        expected, expected_rest = run(first[0], x, passes=1), rest(first[1], x)
        for model in first:  # the filler ends: custody keeps both sets, nobody maps them
            model.backend.close()
        assert len(custody.held) == 2 and all(custody.ended())

        weights, _ = executor_with(custody, tmp_path / "replacement")
        a = load(weights, a_at, "a")
        b = load(weights, b_at, "b", name="other", scope="rest")
        opened.extend([a, b])
        stack, other = a.backend.components["stack"], b.backend.components["other"]
        mapped = weights.resident_bytes(other)
        assert stack.shared and other.shared and weights.facts().committed_bytes == 0
        capped(weights, 0)
        asked: list[int] = []

        def room(free_bytes: int) -> None:
            asked.append(custody.collect())
            weights.set_cap(weights.cap + asked[-1])

        weights.room = room
        weights.seed_squeezed({"run": weights.allocatable()})
        free = torch.cuda.mem_get_info(0)[0]
        assert same(run(a, x, passes=1), expected)
        assert weights.facts().squeezed == {}, "a seed is the owner's already: nothing to report"
        # The other set left this process and custody, and its memory is the driver's again.
        (freed,) = asked  # the holding's whole chunks: a little more than its regions' bytes
        assert freed >= mapped and not other.shared and weights.resident_bytes(other) == 0
        assert list(custody.held) == [(stack.ws.digest, "")]
        assert torch.cuda.mem_get_info(0)[0] - free > mapped - 32 * MIB
        assert weights.facts().h2d_bytes == 0, "the stage's own set stayed mapped"

        # Its own stage fills a private copy and offers it: the machine keeps it again.
        assert same(rest(b, x), expected_rest)
        assert weights.facts().h2d_bytes == mapped
        assert sorted(custody.held) == sorted([(stack.ws.digest, ""), (other.ws.digest, "")])
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        for model in opened:
            model.backend.close()
        custody.close()
