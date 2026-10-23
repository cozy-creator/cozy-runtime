"""A Backend's binds on varena: a decoded tensor decodes once a generation, not once a bind
(after v5's L40S cells 13367/13375 re-decoded every int8 UNet weight on every block bind), and a
registered component serves in eval mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tensorfs = pytest.importorskip("tensorfs")
from tensorfs.varena import GpuId  # noqa: E402
from tensorfs.varena.torch import Memory  # noqa: E402

from cozy_runtime.derive.microscale import encode_int8_convrot256  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.weights import Backend  # noqa: E402
from test_int8_convrot import weight  # noqa: E402


def test_a_decoded_tensor_decodes_once_across_twenty_binds(tmp_path: Path) -> None:
    values = weight(64, 512, 7)
    encoded = encode_int8_convrot256("weight", values)
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("net", cls=("torch", "", "Linear"), config={"bias": False})
    component.put_encoded(
        "weight",
        dtype="F32",
        shape=[64, 512],
        scheme="int8.convrot",
        params={"rotation": "kron_regular_h4", "rotation_size": 256},
        parts={
            "weight": ("I8", [64, 512], encoded.payload),
            "scale": ("F32", [64], encoded.companions[0].raw),
        },
    )
    native = writer.commit()
    del component, writer
    checkpoint = Checkpoint(native)
    rows = checkpoint.rows("net")
    memory = Memory.open(GpuId(f"cozy-family-default-{os.getpid()}", "cpu", 1 << 30))
    backend = Backend.for_script(
        {"net": checkpoint},
        rows,
        torch=torch,
        memory=memory,
        device=torch.device("cpu"),
        construction="family-default",
        release="family-default",
    )
    with torch.device("meta"):
        net = torch.nn.Linear(512, 64, bias=False)
    model = backend._register("net", net, [r.key for r in rows])
    backend.models["net"] = model
    assert "forward" not in vars(net)  # Runtime's wrappers come after the plan
    provider = backend.selected["net.weight"].provider
    assert provider.route == "decoded_float"
    decodes: list[int] = []
    code = type(provider).decode.__code__
    original = sys.getprofile()

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is code:
            decodes.append(1)

    x = torch.randn(3, 512)
    try:
        sys.setprofile(profile)
        outputs = []
        with torch.no_grad():
            for turn in range(20):
                with memory.request(f"bind-{turn}"):
                    outputs.append(net(x))
        assert len(decodes) == 1, "a decoded tensor decodes once a generation"
        assert all(torch.equal(o, outputs[0]) for o in outputs)
        routes = backend.routes["net"]
        assert routes.region is not None  # kept in the component's derived region
        routes.forget()  # the region came back empty (varena dropped it): the next bind decodes
        assert not routes.decoded
        with torch.no_grad(), memory.request("after-drop"):
            assert torch.equal(net(x), outputs[0])
        assert len(decodes) == 2
    finally:
        sys.setprofile(original)
        model.close()
        backend.close()
        native.close()
    assert not routes.decoded and routes.region is None


def test_a_component_is_served_in_eval_mode(tmp_path: Path) -> None:
    """Constructed in training mode (as Diffusers and transformers build), its dropout would
    act on every request; registered, it serves as written for inference."""
    values = torch.randn(64, 32)
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("net", cls=("torch", "", "Sequential"), config={})
    component.put("0.weight", values.numpy().tobytes(), dtype="F32", shape=[64, 32])
    native = writer.commit()
    del component, writer
    checkpoint = Checkpoint(native)
    rows = checkpoint.rows("net")
    memory = Memory.open(GpuId(f"cozy-eval-mode-{os.getpid()}", "cpu", 1 << 30))
    backend = Backend.for_script(
        {"net": checkpoint},
        rows,
        torch=torch,
        memory=memory,
        device=torch.device("cpu"),
        construction="eval-mode",
        release="eval-mode",
    )
    with torch.device("meta"):
        net = torch.nn.Sequential(torch.nn.Linear(32, 64, bias=False), torch.nn.Dropout(0.5))
    assert net.training
    model = backend._register("net", net, [r.key for r in rows])
    try:
        assert not any(m.training for m in net.modules())
        x = torch.randn(4, 32)
        with torch.no_grad(), memory.request("eval-mode"):
            assert torch.equal(net(x), x @ values.T)
    finally:
        model.close()
        backend.close()
        native.close()
