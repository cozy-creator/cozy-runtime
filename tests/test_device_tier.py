"""The machine keeps GPU weights across executors (`weights.attach/1`, Degree 2).

Real: two `tensorfs.plane` device planes (an executor and its replacement), the executor's
own durable exchange over a seam socketpair with exported CUDA fds, and a custody stand-in
that does what TensorD's does: keep the fds, hand duplicates out. The replacement copies
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
    """TensorD's `ResidentCustody` in miniature: (layout, variant) -> (regions, fds), and
    one lease socket per offer kept or attachment, whose other end the executor holds. An ask
    for variant "*" is answered with the layout's baked variant offered last."""

    def __init__(self, layout_says: str = "", variants: bool = True) -> None:
        self.held: dict[tuple[str, str], tuple[list[dict[str, Any]], list[int]]] = {}
        #: an older machine keys holdings by layout alone and never says `variants`
        self.variants = variants
        self.leases: list[socket.socket] = []
        #: each holding's leases
        self.readers: dict[tuple[str, str], list[socket.socket]] = {}
        #: a holder that names another layout than the one it was asked for
        self.layout_says = layout_says

    def lease(self, channel: Channel, key: tuple[str, str]) -> None:
        ours, theirs = socket.socketpair()
        channel.send_lease(theirs.fileno())
        theirs.close()
        self.leases.append(ours)
        self.readers.setdefault(key, []).append(ours)

    def ended(self, leases: list[socket.socket] | None = None) -> list[bool]:
        """Which leases (all, in the order given out) the executor ended (closed, or died)."""
        out = []
        for ours in self.leases if leases is None else leases:
            try:
                out.append(ours.recv(1, socket.MSG_DONTWAIT) == b"")
            except BlockingIOError:
                out.append(False)
        return out

    def collect(self) -> int:
        """Drop every holding no executor maps, as the machine does when one asks for room:
        its fds close, and with them the last references to its GPU memory. Returns the bytes."""
        freed = 0
        for key in [k for k in self.held if all(self.ended(self.readers.get(k, [])))]:
            regions, fds = self.held.pop(key)
            for fd in fds:
                os.close(fd)
            freed += sum(sum(row["chunks"]) for row in regions)
        return freed

    def serve(self, channel: Channel) -> None:
        while (frame := channel.recv()) is not None:
            assert frame["kind"] == "device_tier", frame
            fds = [channel.recv_shared() for _ in range(frame.get("descriptors", 0))]
            answer: dict[str, Any] = {"event": "answer", "seq": frame["seq"], "ok": True}
            variant = frame.get("variant", "") if self.variants else ""
            if self.variants:
                answer["variants"] = True
            if variant == "*":
                baked = [v for layout, v in self.held if layout == frame["layout"] and v]
                variant = baked[-1] if baked else "*"
            key = (frame["layout"], variant)
            if frame["offer"]:
                if key in self.held:
                    answer["duplicate"] = True
                    for fd in fds:
                        os.close(fd)
                    channel.send(answer)
                else:
                    self.held[key] = (frame["regions"], fds)
                    channel.send({**answer, "generation": 1, "lease": True})
                    self.lease(channel, key)
                continue
            held = self.held.get(key)
            if held is None:
                channel.send(answer)
                continue
            regions, kept = held
            channel.send(
                {
                    **answer,
                    "held": True,
                    "generation": 1,
                    "layout": self.layout_says or frame["layout"],
                    "regions": regions,
                    "descriptors": len(kept),
                    "lease": True,
                }
                | ({"variant": variant} if variant else {})
            )
            for fd in kept:
                channel.send_shared(fd)
            self.lease(channel, key)

    def regions(self, layout: str, variant: str) -> list[int]:
        return [row["region"] for row in self.held.get((layout, variant), ([], []))[0]]

    def close(self) -> None:
        for _, fds in self.held.values():
            for fd in fds:
                os.close(fd)
        self.held.clear()
        for ours in self.leases:
            ours.close()


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
        first.measure_context()
        before = first.context_bytes
        expected = run(model, x)
        filled = first.facts().h2d_bytes
        assert filled > 0
        total = sum(r.span for r in model.backend.components["stack"].regions)
        facts = first.facts()
        assert (facts.shared_bytes, facts.committed_bytes) == (total, 0), "offered at stage exit"
        assert first.share() == 0, "regions are offered once"
        # The driver charges the exported regions to this process: never its context (which
        # may still grow by a library workspace, never by the shared bytes).
        first.measure_context()
        assert first.context_bytes < before + total, (before, first.context_bytes, total)
        assert same(run(model, x), expected), "the filler still computes on its read-only copy"

        replacement, _ = executor_with(custody, tmp_path / "b")
        again = load(replacement, checkpoint, "share")
        component = again.backend.components["stack"]
        assert component.shared == set(range(len(component.regions)))
        assert same(run(again, x), expected)
        facts = replacement.facts()
        assert facts.h2d_bytes == 0, "attached weights are never copied"
        assert facts.shared_bytes == total

        # Two live leases: the filler's offer and the replacement's attachment.
        assert custody.ended() == [False, False]

        # Revoked: queued work first, then unmapped; the next stage fills a private copy. The
        # replacement's lease ends with it, with no message; the filler's stays.
        layout = component.ws.digest
        assert replacement.revoke(layout) == total
        assert custody.ended() == [False, True]
        assert replacement.facts().shared_bytes == 0
        assert same(run(again, x), expected)
        assert replacement.facts().h2d_bytes == filled
    finally:
        custody.close()


def test_a_trim_lets_some_regions_go_and_keeps_the_rest_attached(tmp_path: Path) -> None:
    """Switches on a 12 GB card (phil): the machine needs part of a layout's room, not all of
    it (`weights.trim/1`). The trimmed regions go, the rest stay mapped under the same lease,
    and the next stage copies only the trimmed regions."""
    checkpoint, _ = write_checkpoint(tmp_path)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    custody = Custody()
    try:
        first, _ = executor_with(custody, tmp_path / "a")
        if not first.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        expected = run(load(first, checkpoint, "share"), x)
        replacement, _ = executor_with(custody, tmp_path / "b")
        again = load(replacement, checkpoint, "share")
        component = again.backend.components["stack"]
        everything = set(range(len(component.regions)))
        assert component.shared == everything
        trimmed = sorted(everything)[-2:]
        spans = sum(component.regions[i].span for i in trimmed)
        assert replacement.revoke(component.ws.digest, trimmed) == spans
        assert component.shared == everything - set(trimmed)
        assert custody.ended() == [False, False], "its other regions are still mapped"
        assert same(run(again, x), expected)
        assert 0 < replacement.facts().h2d_bytes <= spans
    finally:
        custody.close()


def test_an_executor_loaded_without_device_weights_shares_when_the_machine_asks(
    tmp_path: Path,
) -> None:
    """Degree 2 after the fact: the machine learns from a call that the construction fits and
    sends `share` to an executor it loaded without device weights."""
    checkpoint, _ = write_checkpoint(tmp_path)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    custody = Custody()
    try:
        first, executor = executor_with(custody, tmp_path / "a")
        if not first.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        first.sharing = None
        model = load(first, checkpoint, "late")
        expected = run(model, x)
        total = sum(r.span for r in model.backend.components["stack"].regions)
        assert first.facts().shared_bytes == 0, "Degree 1: nothing offered"
        executor.weights = first
        assert executor.share()["shared_bytes"] == total
        facts = first.facts()
        assert (facts.shared_bytes, facts.committed_bytes) == (total, 0)
        assert same(run(model, x), expected)

        replacement, _ = executor_with(custody, tmp_path / "b")
        again = load(replacement, checkpoint, "late")
        assert same(run(again, x), expected)
        assert replacement.facts().h2d_bytes == 0, "attached, never copied"
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


def test_a_holder_naming_another_layout_is_never_mapped(tmp_path: Path) -> None:
    """The holder's answer carries the layout it holds; a reader whose own differs maps
    nothing (and the plane would refuse it too) and fills its own copy."""
    checkpoint, _ = write_checkpoint(tmp_path)
    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float16)
    custody = Custody(layout_says="f" * 64)
    try:
        first, _ = executor_with(custody, tmp_path / "a")
        if not first.plane.device_sharing(0):
            pytest.skip("this GPU cannot export memory")
        expected = run(load(first, checkpoint, "lie"), x)
        replacement, _ = executor_with(custody, tmp_path / "b")
        again = load(replacement, checkpoint, "lie")
        assert again.backend.components["stack"].shared == set()
        assert custody.ended()[-1], "the refused answer's lease is closed at once"
        assert same(run(again, x), expected)
        assert replacement.facts().h2d_bytes > 0
    finally:
        custody.close()
