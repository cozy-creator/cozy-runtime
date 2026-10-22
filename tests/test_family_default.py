"""A lane runs on its family's default kernel where no offline verdict refines it, and a decoded
tensor decodes once a generation, not once a bind (coordinator ruling 10-22, after v5's L40S cells
13367/13375 re-decoded every int8 UNet weight on every block bind)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tensorfs

from cozy_runtime.author._services import Attempt
from cozy_runtime.derive.microscale import encode_int8_convrot256
from cozy_runtime.internal import int8_linear, prebuilt, probe, qualification
from cozy_runtime.internal.encoding import (
    DeviceFacts,
    Int8ConvRotNativeLeaf,
    launch_providers,
    measure_runtime,
    params_of,
)
from cozy_runtime.internal.encoding.leaves import sites as leaf_sites
from cozy_runtime.internal.executor import Executor
from test_int8_convrot import weight
from test_refused_leaves import PARAMS, plan, row

torch = pytest.importorskip("torch")


def _card(sm: int) -> DeviceFacts:
    return DeviceFacts("cuda", f"sm{sm} card", sm, "595.91.07", "", 0, "13.0", "Linux/x86_64")


def _qualified(sm: int, tmp_path: Path) -> qualification.Qualification:
    return qualification.qualified(
        torch,
        launch_providers(),
        _card(sm),
        qualification.TABLE_DTYPES,
        release="family-default",
        runtime=measure_runtime(torch, "family-default"),
        tables=tmp_path / "no-tables",
    )


def test_an_sm89_card_without_verdicts_runs_its_int8_lane_on_the_family_default(
    tmp_path: Path,
) -> None:
    table = _qualified(89, tmp_path)
    cells = [
        o
        for o in table.result.observations
        if o.implementation == "cozy.int8-convrot256.native-leaf/1"
    ]
    by_dtype = {o.output_dtype: o.evidence for o in cells}
    assert isinstance(by_dtype["bfloat16"], probe.FamilyDefaultEvidence)
    assert isinstance(by_dtype["float16"], probe.FamilyDefaultEvidence)
    assert not by_dtype["float32"].passed  # an 8-bit GEMM writes half precision only
    rows = [row("unet.a.weight", "int8.convrot", "bf16", (64, 256))]
    served = plan(rows, table.capabilities, _card(89))
    assert [t.route for t in served.tensors] == ["encoded_gemm"]
    assert served.refused_leaves() == ()
    # At execution the family's GEMM serves without a probe; the run says why.
    ops = prebuilt.Prebuilt(torch, 0, "mma.sync", False)
    try:
        verdict = prebuilt.admission(torch, ops, prebuilt.INT8, "mma.sync", 89)
        assert verdict.ok and "family default" in verdict.detail
        assert prebuilt.DEFAULTED[89, "int8"] == verdict.detail
        encoded = encode_int8_convrot256("w", weight(64, 512, 5))
        roles = {
            "weight": torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).reshape(
                64, 512
            ),
            "scale": torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32),
        }
        leaf = Int8ConvRotNativeLeaf(params=params_of(PARAMS["int8.convrot"]))
        installed = {
            "unet.a.weight": (
                leaf.leaf(torch, roles, probe._StandIn(512, 64), torch.bfloat16),
                None,
                (),
            )
        }
        executor = Executor(None, tmp_path)  # type: ignore[arg-type]
        executor.backend = SimpleNamespace(gemm_sites=leaf_sites(installed))
        executor.models = {"sdxl": object()}
        executor.device_facts = _card(89)
        executor.refused_leaves = ()
        int8_linear._OPS[0] = ops.int8_linear  # the family's op, as `native` installs it
        record = Attempt("gemm", tmp_path, 0)
        Executor._emit_gemm(record, executor)
        said = {
            (r["name"], r["value"]): r["fields"]
            for r in record.ring.rows()
            if r["kind"] == "confession"
        }
        assert said[("gemm.family_default", int8_linear.OP)]["reason"] == verdict.detail  # type: ignore[index]
        assert ("gemm.fallback", int8_linear.OP) not in said
    finally:
        prebuilt.DEFAULTED.pop((89, "int8"), None)
        int8_linear._OPS.pop(0, None)


def test_a_card_without_the_lanes_tensor_cores_decodes(tmp_path: Path) -> None:
    table = _qualified(75, tmp_path)
    rows = [row("unet.a.weight", "int8.convrot", "bf16", (64, 256))]
    assert [t.route for t in plan(rows, table.capabilities, _card(75)).tensors] == ["decoded_float"]


def test_a_decoded_tensor_decodes_once_across_twenty_binds(tmp_path: Path) -> None:
    from tensorfs.varena import GpuId
    from tensorfs.varena.torch import Memory

    from cozy_runtime.internal.fill import Checkpoint
    from cozy_runtime.internal.weights import Backend

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
        held = 64 * 512 * 4
        assert routes.shed(0) == held and not routes.decoded  # varena's ask: the next bind decodes
        with torch.no_grad(), memory.request("after-shed"):
            assert torch.equal(net(x), outputs[0])
        assert len(decodes) == 2
    finally:
        sys.setprofile(original)
        model.close()
        backend.close()
        native.close()
    assert not routes.decoded and routes.cache is None
