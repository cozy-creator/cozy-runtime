"""Copy-while-building on the real weight plane (CPU core): a held host layout is adopted from the
regions it carries before its module exists; the load binds to it when the module's regions are
the deposit's, and a module of other regions is a miss that loads from the chunk files. The run
completes and equals the reference either way."""

from __future__ import annotations

import os
import socket
import threading
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import tensorfs  # noqa: E402
from tensorfs import varena  # noqa: E402
from tensorfs.varena.torch import Declared, Memory, WeightGrant, regions_of  # noqa: E402
from torch import nn  # noqa: E402

from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.weights import Backend  # noqa: E402
from tensord_peer import answer_read_grants, model_source  # noqa: E402

DIM, BLOCKS = 64, 4


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.lin(x))


class Net(nn.Module):
    _no_split_modules = ("Block",)

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(DIM, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.head = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def built(grant: WeightGrant, blocks: list[str] | None = None) -> Net:
    with torch.device("meta"):
        net = Net()
    vars(net)["_varena"] = Declared(grant, blocks=blocks)
    return net


def test_a_held_layout_binds_before_the_module_and_other_regions_miss(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference = Net()
    store = tensorfs.Store(tmp_path / "store")
    writer = store.writer()
    for key, t in reference.state_dict().items():
        t = t.detach().contiguous()
        writer.put("net", key, "F32", list(t.shape), t.view(torch.uint8).numpy().tobytes())
    ck = store.checkpoint(writer.commit())
    grant = WeightGrant(ck, "net", f"sha256:{ck.manifest.sha256}")
    held = varena.seal(ck, regions_of(built(grant)), "deposit")
    grant = replace(grant, host_fd=held)
    x = torch.randn(8, DIM, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = reference(x)
    memory = Memory.open(torch.device("cpu"))
    try:
        memory.adopt(grant)
        adopted = memory.adopted[grant.key]
        assert adopted.regions == regions_of(built(grant))
        net = built(grant)
        model = memory.load(net)
        assert model.ws is adopted and not memory.adopted
        with torch.no_grad(), memory.request("adopted"):
            assert torch.equal(net(x), want)
        model.close()

        memory.adopt(grant)
        other = built(grant, blocks=["blocks.0"])
        assert regions_of(other) != memory.adopted[grant.key].regions
        missed = memory.load(other)
        assert not memory.adopted and missed.ws.regions == regions_of(other)
        with torch.no_grad(), memory.request("missed"):
            assert torch.equal(other(x), want)
        missed.close()
    finally:
        memory.unclaimed()
        os.close(held)


def test_a_changed_held_layout_regrants_chunks_for_a_header_only_executor(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference = Net()
    store = tensorfs.Store(tmp_path / "store")
    writer = store.writer()
    for key, t in reference.state_dict().items():
        t = t.detach().contiguous()
        writer.put("net", key, "F32", list(t.shape), t.numpy())
    manifest = f"sha256:{writer.commit().sha256}"
    ck = Checkpoint.granted(manifest, lambda name: model_source(store, manifest, name))
    assert ck.ck.manifest is None
    grant = WeightGrant(ck.ck, "net", manifest)
    held = varena.seal(store.checkpoint(manifest), regions_of(built(grant)), "old-layout")
    memory = Memory.open(torch.device("cpu"))
    backend = Backend.for_script(
        {"net": ck},
        ck.rows("net"),
        torch=torch,
        memory=memory,
        device=torch.device("cpu"),
        construction="held-mismatch",
        release="test",
    )
    left, right = socket.socketpair()
    executor, asked = Executor(Channel(left), tmp_path), []
    serving = threading.Thread(
        target=answer_read_grants,
        args=(Channel(right), store, {backend._layout("net"): held}, asked),
    )
    serving.start()
    backend.grant = executor._read_grant
    try:
        backend.ahead()
        with torch.device("meta"):
            net = Net()
        net._no_split_modules = ("Linear",)
        model = backend._register("net", net, [row.key for row in ck.rows("net")])
        assert asked == [backend._layout("net"), ""], "a mismatch explicitly asks for chunk files"
        assert not memory.adopted and model.ws.regions == regions_of(net)
        x = torch.randn(8, DIM)
        with torch.no_grad(), memory.request("changed-layout"):
            assert torch.equal(net(x), reference(x))
        model.close()
    finally:
        backend.close()
        memory.unclaimed()
        left.close()
        serving.join()
        right.close()
        os.close(held)
