"""The machine keeps GPU weights across executors (`weights.attach/1`, Degree 2).

Real: two `tensorfs.plane` device planes (an executor and its replacement), the executor's
own durable exchange over a seam socketpair with exported CUDA fds, and a custody stand-in
that does what cozy-machine's does: keep the fds, hand duplicates out. The replacement copies
no byte to the GPU, computes bit-identically, and after a revoke fills its own copy again.
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import accel, plane  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.weights import DeviceTiers, Weights  # noqa: E402
from test_weight_plane import BLOCKS, HIDDEN, load, run, same, tight, write_checkpoint  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not plane.sharing(), reason="a TensorFS that shares GPU regions"),
    pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card"),
]


class Custody:
    """cozy-machine's `ResidentCustody` in miniature: layout -> (regions, fds)."""

    def __init__(self) -> None:
        self.held: dict[str, tuple[list[dict[str, Any]], list[int]]] = {}

    def serve(self, channel: Channel) -> None:
        while (frame := channel.recv()) is not None:
            assert frame["kind"] == "device_tier", frame
            fds = [channel.recv_shared() for _ in range(frame.get("descriptors", 0))]
            answer: dict[str, Any] = {"event": "answer", "seq": frame["seq"], "ok": True}
            if frame["offer"]:
                if frame["layout"] in self.held:
                    answer["duplicate"] = True
                    for fd in fds:
                        os.close(fd)
                else:
                    self.held[frame["layout"]] = (frame["regions"], fds)
                    answer["generation"] = 1
                channel.send(answer)
                continue
            held = self.held.get(frame["layout"])
            if held is None:
                channel.send(answer)
                continue
            regions, kept = held
            channel.send(
                {
                    **answer,
                    "held": True,
                    "generation": 1,
                    "regions": regions,
                    "descriptors": len(kept),
                }
            )
            for fd in kept:
                channel.send_shared(fd)

    def close(self) -> None:
        for _, fds in self.held.values():
            for fd in fds:
                os.close(fd)
        self.held.clear()


def executor_with(custody: Custody, tmp_path: Path) -> tuple[Weights, Executor]:
    tmp_path.mkdir()
    ours, theirs = socket.socketpair()
    threading.Thread(target=custody.serve, args=(Channel(theirs),), daemon=True).start()
    executor = Executor(Channel(ours), tmp_path)
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    weights.set_budget(1 << 30, pinned=1 << 30)
    weights.sharing = DeviceTiers(ask=executor._ask_device, offer=executor._offer_device)
    return weights, executor


def test_a_replacement_attaches_the_kept_weights_and_computes_bit_identically(
    tmp_path: Path,
) -> None:
    checkpoint, _ = write_checkpoint(tmp_path)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    custody = Custody()
    try:
        first, _ = executor_with(custody, tmp_path / "a")
        if not first.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        model = load(first, checkpoint, "share")
        expected = run(model, x)
        filled = first.facts().h2d_bytes
        assert filled > 0
        total = sum(r.span for r in model.backend.components["stack"].regions)
        facts = first.facts()
        assert (facts.shared_bytes, facts.committed_bytes) == (total, 0), "offered at stage exit"
        assert first.share() == 0, "regions are offered once"
        assert same(run(model, x), expected), "the filler still computes on its read-only copy"

        replacement, _ = executor_with(custody, tmp_path / "b")
        again = load(replacement, checkpoint, "share")
        component = again.backend.components["stack"]
        assert component.shared == set(range(len(component.regions)))
        assert same(run(again, x), expected)
        facts = replacement.facts()
        assert facts.h2d_bytes == 0, "attached weights are never copied"
        assert facts.shared_bytes == total

        # Revoked: queued work first, then unmapped; the next stage fills a private copy.
        layout = component.ws.digest
        assert replacement.revoke(layout) == total
        assert replacement.facts().shared_bytes == 0
        assert same(run(again, x), expected)
        assert replacement.facts().h2d_bytes == filled
    finally:
        custody.close()


def test_a_duplicate_offer_is_replaced_by_the_kept_copy(tmp_path: Path) -> None:
    checkpoint, _ = write_checkpoint(tmp_path)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    custody = Custody()
    try:
        one, _ = executor_with(custody, tmp_path / "a")
        if not one.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        two, _ = executor_with(custody, tmp_path / "b")
        first, second = load(one, checkpoint, "dup"), load(two, checkpoint, "dup")
        expected = run(first, x)
        total = sum(r.span for r in first.backend.components["stack"].regions)
        assert one.facts().shared_bytes == total
        # The second fills its own copy; offered at its stage exit, it goes for the kept one.
        assert same(run(second, x), expected)
        assert two.facts().committed_bytes == 0 and two.facts().shared_bytes == total
        assert same(run(second, x), expected)
    finally:
        custody.close()


def test_attached_weights_cost_no_budget_and_are_never_streamed(tmp_path: Path) -> None:
    """A budget that would stream the component holds it whole once it is attached: the plan
    pays only for unattached regions, so no ring, no copy, and no attached region dropped."""
    checkpoint, _ = write_checkpoint(tmp_path)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    custody = Custody()
    try:
        first, _ = executor_with(custody, tmp_path / "a")
        if not first.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        model = load(first, checkpoint, "free")
        expected = run(model, x)
        replacement, _ = executor_with(custody, tmp_path / "b")
        again = load(replacement, checkpoint, "free")
        replacement.set_budget(tight(again))
        assert same(run(again, x), expected)
        facts = replacement.facts()
        assert facts.h2d_bytes == 0 and facts.committed_bytes == 0
        assert facts.streamed["stack"].resident_blocks == BLOCKS
        assert facts.streamed["stack"].window == 0
        assert facts.shared_bytes == sum(r.span for r in again.backend.components["stack"].regions)
    finally:
        custody.close()
