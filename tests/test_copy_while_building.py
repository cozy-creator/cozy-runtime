"""A held host layout on the real weight plane (CPU core): a module of other regions than the
deposit's is a miss and refills from the stored checkpoint; the run completes and equals the
reference."""

from __future__ import annotations

import os
import socket
import threading
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import tensorfs  # noqa: E402
from tensorfs import varena  # noqa: E402
from tensorfs.varena.torch import Declared, Memory, WeightGrant, regions_of  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import Module  # noqa: E402

from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.weights import Backend  # noqa: E402
from tensord_peer import answer_read_grants  # noqa: E402

DIM, BLOCKS = 64, 4


def network() -> Module:
    net = nn.Sequential(
        OrderedDict(
            embed=nn.Linear(DIM, DIM),
            blocks=nn.Sequential(
                *(
                    nn.Sequential(OrderedDict(lin=nn.Linear(DIM, DIM), relu=nn.ReLU()))
                    for _ in range(BLOCKS)
                )
            ),
            head=nn.Linear(DIM, DIM),
        )
    )
    net._no_split_modules = ("Sequential",)
    return net


def built(grant: WeightGrant, blocks: list[str] | None = None) -> Module:
    with torch.device("meta"):
        net = network()
    net._varena = Declared(grant, blocks=blocks)
    return net


def test_a_changed_held_layout_refills_directly_from_the_stored_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference = network()
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("net", cls=("", "", ""), config={})
    for key, t in reference.state_dict().items():
        t = t.detach().contiguous()
        component.put(key, t.numpy(), dtype="F32", shape=list(t.shape))
    native = writer.commit()
    assert native.id is not None
    manifest = native.id
    ck = Checkpoint(native)
    grant = WeightGrant(ck.ck, "net", manifest)
    held = varena.seal(native, regions_of(built(grant)), "old-layout")
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
    executor = Executor(Channel(left), tmp_path)
    asked: list[str] = []
    serving = threading.Thread(
        target=answer_read_grants,
        args=(Channel(right), {backend._layout("net"): held}, asked),
    )
    serving.start()
    backend.grant = executor._read_grant
    try:
        backend.ahead()
        with torch.device("meta"):
            net = network()
        net._no_split_modules = ()
        model = backend._register("net", net, [row.key for row in ck.rows("net")])
        assert asked == [backend._layout("net")], (
            "the stored checkpoint refills without another grant"
        )
        assert model.ws.regions == regions_of(net)
        x = torch.randn(8, DIM)
        with torch.no_grad(), memory.request("changed-layout"):
            assert torch.equal(net(x), reference(x))
        model.close()
    finally:
        backend.close()
        left.close()
        serving.join()
        right.close()
        os.close(held)


def test_a_standalone_checkpoint_load_owns_its_parameter_views(tmp_path: Path) -> None:
    torch.manual_seed(7)
    reference = network()
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("net", cls=("", "", ""), config={})
    for key, value in reference.state_dict().items():
        component.put(
            key, value.detach().contiguous().numpy(), dtype="F32", shape=list(value.shape)
        )
    ck = writer.commit()
    with torch.device("meta"):
        net = network()
    memory = Memory.open(torch.device("cpu"))
    model = memory.load(net, ck=ck, component="net")
    try:
        part = next(
            part
            for part in model.ws.parts
            if part.component == "net" and part.key == "embed.weight"
        )
        pinned = model.ws.bind(part.region)
        storage = torch.from_dlpack(pinned)
        retained = (
            storage[part.offset : part.offset + part.nbytes].view(torch.float32).reshape(part.shape)
        )
        assert retained.data_ptr() == net.embed.weight.data_ptr() == pinned.ptr + part.offset
        ck.close()
        x = torch.randn(8, DIM)
        with torch.no_grad(), memory.request("standalone"):
            assert torch.equal(net(x), reference(x))
        model.close()
        assert torch.equal(retained, reference.embed.weight), "the consumer still owns its host pin"
    finally:
        model.close()
        ck.close()
