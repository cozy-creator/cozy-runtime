"""`svdq.nvfp4` (xs-066): the calibrated producer on a real model's linears, the store and
TensorFS's decode, the decode cores' law, the leaf as the weight plane installs and binds it, a
request's LoRAs in its branch, the kernels' build, and on a Blackwell card the kernels against the
law."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cozy_runtime.author import Weights
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.derive import svdq
from cozy_runtime.derive.microscale import e4m3_decode, unpack_nibbles
from cozy_runtime.derive.operations import (
    EncodingRule,
    QuantizationPlan,
    calibrate,
    quantize_into,
)
from cozy_runtime.internal import (
    decode_cores,
    decode_schemes,
    probe,
    svdq_linear,
    triton_kernels,
    weights,
)
from cozy_runtime.internal.encoding import LeafProvider, SvdqNvfp4NativeLeaf, params_of
from cozy_runtime.internal.encoding.formats import E2M1_VALUES
from cozy_runtime.weights_store import read_into
from store_fixtures import adopted, checkpoint, context, home

PARAMS = params_of(
    {
        "block_size": 16,
        "global_scale": "multiply",
        "nibble_order": "low_first",
        "scale_layout": "row_major",
    }
)
#: The stored parts as the fill plane hands them over: role -> (torch dtype name, shape).
PARTS = {
    "weight": ("uint8", lambda n, k: (n, k // 2)),
    "scale": ("float8_e4m3fn", lambda n, k: (n, k // 16)),
    "scale_2": ("float32", lambda n, k: ()),
    "pre_quant_scale": ("float32", lambda n, k: (k,)),
    "lowrank_down": ("bfloat16", lambda n, k: (svdq.RANK, k)),
    "lowrank_up": ("bfloat16", lambda n, k: (n, svdq.RANK)),
    "input_scale": ("float32", lambda n, k: ()),
}


def model(torch: Any) -> Any:
    """Two linears whose inputs carry a few outlier channels, as a DiT block's do."""
    generator = torch.Generator().manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(512, 384), torch.nn.GELU(), torch.nn.Linear(384, 256))
    with torch.no_grad():
        for layer in (net[0], net[2]):
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.05)
    return net


def inputs(torch: Any, seed: int) -> Any:
    x = torch.randn(4, 96, 512, generator=torch.Generator().manual_seed(seed))
    x[..., :6] *= 25.0
    return x


def calibrated(torch: Any) -> tuple[Any, svdq.Calibration]:
    net = model(torch)
    keys = ["transformer.0.weight", "transformer.2.weight"]
    with svdq.Calibration(torch, {"transformer": net}, keys) as calibration, torch.no_grad():
        for seed in range(3):
            net(inputs(torch, seed))
    return net, calibration


def parts_of(torch: Any, raws: dict[str, bytes], rows: int, cols: int) -> dict[str, Any]:
    """Stored parts from their bytes."""
    return {
        role: torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .view(getattr(torch, PARTS[role][0]))
        .view(PARTS[role][1](rows, cols))
        for role, raw in raws.items()
    }


def stored(torch: Any, fitted: svdq.Fitted, rows: int, cols: int) -> dict[str, Any]:
    """A fit's parts, from the encoder's bytes."""
    roles = [role for role, _, _ in svdq.companions(rows, cols)]
    return parts_of(
        torch,
        {"weight": fitted.payload, **dict(zip(roles, fitted.companions, strict=True))},
        rows,
        cols,
    )


def fitted_parts(torch: Any) -> tuple[Any, dict[str, Any]]:
    net, calibration = calibrated(torch)
    weight = net[0].weight.detach().numpy()
    fitted = svdq.fit(torch, "w", weight, calibration.stats("transformer", "0.weight"), "cpu")
    return net, stored(torch, fitted, *weight.shape)


def law(torch: Any, parts: dict[str, Any]) -> Any:
    """The spec's decode from the stored bytes, element by element in f32 and r order."""
    rows, cols = parts["lowrank_up"].shape[0], parts["pre_quant_scale"].numel()
    codes = unpack_nibbles(parts["weight"].numpy(), cols)
    scale = e4m3_decode(parts["scale"].view(torch.uint8).numpy()).reshape(rows, cols // 16, 1)
    value = np.asarray(E2M1_VALUES, np.float32)[codes].reshape(rows, cols // 16, 16) * scale
    value = value.reshape(rows, cols).astype(np.float32) * parts["scale_2"].numpy().reshape(1)[0]
    value = value * parts["pre_quant_scale"].numpy()[None, :]
    up, down = parts["lowrank_up"].float().numpy(), parts["lowrank_down"].float().numpy()
    for r in range(up.shape[1]):
        value = value + up[:, r : r + 1] * down[r : r + 1, :]
    return torch.from_numpy(value)


def test_calibrated_svdq_beats_plain_nvfp4_on_outlier_activations() -> None:
    """Fitted to its own inputs, each linear's whole scheme (W4A4 plus the branch) is closer to
    the float output, token by token, than plain nvfp4 W4A4, as the fit reports; its weight
    decodes near the original, and the residual's round trip is within the encoder's bound."""
    torch = pytest.importorskip("torch")
    net, calibration = calibrated(torch)
    for key, layer in (("0.weight", net[0]), ("2.weight", net[2])):
        w = layer.weight.detach()
        fitted = svdq.fit(torch, key, w.numpy(), calibration.stats("transformer", key), "cpu")
        parts = stored(torch, fitted, *w.shape)
        relative = float((law(torch, parts) - w).norm() / w.norm())
        assert relative < 0.1 and relative == pytest.approx((fitted.sse / fitted.energy) ** 0.5)
        assert 0 < fitted.max_abs_err <= fitted.amax / 4
        x = calibration.stats("transformer", key).rows.float()
        exact = x @ w.t()
        got = svdq_linear.eager(torch, x, parts, None, torch.float32)
        mine = float(((got - exact).norm(dim=1) / exact.norm(dim=1)).mean())
        assert mine == pytest.approx(fitted.error, rel=1e-3)
        assert mine < 0.8 * fitted.nvfp4_error, (key, mine, fitted.nvfp4_error, fitted.alpha)


def test_a_recipe_writes_the_lane_from_its_calibration_and_tensorfs_decodes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As a package recipe runs it: the model at 16 bits runs real inputs under `calibrate`, then
    `quantize_into` fits each svdq linear to its own. TensorFS holds the parts, and its decode is
    the spec's law of them; the plan's other weights keep their lane."""
    torch = pytest.importorskip("torch")
    store = home(tmp_path / "tensorfs", monkeypatch)
    net = model(torch)
    held = {f"{i}.weight": net[i].weight.detach().numpy().copy() for i in (0, 2)}
    text = np.random.default_rng(4).standard_normal((64, 256)).astype(np.float32)
    source = checkpoint(store, {"transformer": held, "text_encoder": {"q.weight": text}})
    plan = QuantizationPlan(
        components=("transformer", "text_encoder"),
        include=("transformer.*.weight", "text_encoder.*.weight"),
        encodings=(EncodingRule("transformer.*", "svdq.nvfp4"),),
        output_precision="preserve",
    )
    shown: list[tuple[str, str, str, str, str]] = []
    ctx = context(store, shown=shown, outputs={"model": 64 << 20})
    recorder = calibrate(
        ctx, torch, source=Weights(source), components={"transformer": net}, plan=plan,
        encoding="int8.convrot",
    )  # fmt: skip
    with recorder, torch.no_grad():
        for seed in range(3):
            net(inputs(torch, seed))
    quantize_into(
        ctx, fake_telemetry(fake_attempt("svdq")), source=Weights(source), plan=plan,
        encoding="int8.convrot", output="model", calibration=recorder,
    )  # fmt: skip
    out = adopted(store, shown)
    assert out.components["text_encoder"].tensors["q.weight"].encoding.scheme == "int8.convrot"
    for key, (rows, cols) in (("0.weight", (384, 512)), ("2.weight", (256, 384))):
        tensor = out.components["transformer"].tensors[key]
        assert tensor.encoding is not None and tensor.encoding.scheme == "svdq.nvfp4"
        raws = {}
        for role, (dtype, shape) in PARTS.items():
            size = int(np.prod(shape(rows, cols), dtype=np.int64)) * {
                "uint8": 1,
                "float8_e4m3fn": 1,
                "bfloat16": 2,
            }.get(dtype, 4)
            raws[role] = bytearray(size)
            read_into(out, "transformer", key, role, 0, raws[role])
        parts = parts_of(torch, {r: bytes(b) for r, b in raws.items()}, rows, cols)
        decoded = out.components["transformer"].tensors.read(key, decode=True)
        assert torch.equal(torch.from_dlpack(decoded).float(), law(torch, parts)), key
        relative = float(
            (law(torch, parts) - torch.from_numpy(held[key])).norm() / np.linalg.norm(held[key])
        )
        assert relative < 0.1, (key, relative)


def test_the_decode_cores_are_the_spec_law() -> None:
    torch = pytest.importorskip("torch")
    _, parts = fitted_parts(torch)
    shape = (384, 512)
    adapter = decode_schemes.BY_SCHEME["svdq.nvfp4"]
    names = {torch.uint8: "u8", torch.float8_e4m3fn: "f8_e4m3fn", torch.float32: "f32"}
    view = {r: (names.get(t.dtype, "bf16"), tuple(t.shape)) for r, t in parts.items()}
    assert decode_schemes.parts_rule(adapter, shape, dict(PARAMS), view) == ""
    for dtype in (torch.float32, torch.bfloat16):
        out = torch.empty(shape, dtype=dtype)
        cores = decode_cores.Cores(torch, kernel=False)
        adapter.decode(cores, parts, dict(PARAMS), shape, out)
        want = torch.empty(shape, dtype=dtype)
        decode_cores.round_into(torch, law(torch, parts), want.view(-1))
        assert torch.equal(out, want), dtype


def test_the_leaf_installs_on_meta_binds_the_stored_roles_and_serves_the_law() -> None:
    """As the weight plane installs it (meta parts) and binds a page's parts: a canonical tensor's
    roles are the stored tensors themselves, nothing is made at the bind, and the leaf's output is
    the law's (`svdq_linear.eager`, what the kernels compute) and near the float product."""
    torch = pytest.importorskip("torch")
    provider = SvdqNvfp4NativeLeaf(params=PARAMS)
    assert isinstance(provider, LeafProvider)
    assert provider.supports([256, 512], dict(PARAMS)) == ""
    assert "K %" in provider.supports([256, 528], dict(PARAMS))
    net, parts = fitted_parts(torch)
    replaced = probe._StandIn(512, 384)
    meta = {role: t.to("meta") for role, t in parts.items()}
    planned = provider.leaf(torch, meta, replaced, torch.bfloat16)
    assert weights._derived(planned, meta) == ()
    made = weights._bound(
        planned, (), parts, lambda: provider.leaf(torch, parts, replaced, torch.bfloat16)
    )
    for name, tensor in made.items():
        planned._buffers[name] = tensor
    planned._buffers["_logical_dtype"] = torch.empty(0, dtype=torch.bfloat16)
    x = inputs(torch, 7).to(torch.bfloat16)
    got = planned(x)
    assert got.shape == (4, 96, 384) and got.dtype == torch.bfloat16
    assert torch.equal(got, svdq_linear.eager(torch, x, parts, None, torch.bfloat16))
    exact = x.float() @ net[0].weight.detach().t()
    assert float((got.float() - exact).norm() / exact.norm()) < 0.1
    wide = {**parts, "lowrank_down": torch.zeros(160, 512, dtype=torch.bfloat16)}
    with pytest.raises(Exception, match="decodes instead"):
        provider.leaf(torch, wide, replaced, torch.bfloat16)


def _row(key: str, scheme: str, roles: dict[str, str]) -> Any:
    parts = [SimpleNamespace(role=r, dtype=d) for r, d in roles.items()]
    return SimpleNamespace(
        key=key, encoded=SimpleNamespace(scheme=scheme, params=PARAMS, parts=parts)
    )


def test_a_request_lora_joins_the_low_rank_branch() -> None:
    """A baked request (`weights._Bakes.prepare`) hands an svdq leaf its PEFT layer's LoRA, which
    joins its branch as more ranks when it runs; no page is refilled and no stored byte changes.
    Per step, the leaf drops it (PEFT adds it). A LoRA past the kernels' ranks adds its own term."""
    torch = pytest.importorskip("torch")
    with torch.no_grad():  # as a forward runs: the PEFT factors are parameters
        _, parts = fitted_parts(torch)
        before = {role: t.clone() for role, t in parts.items()}
        leaf = SvdqNvfp4NativeLeaf(params=PARAMS).leaf(
            torch, parts, probe._StandIn(512, 384), torch.bfloat16
        )
        x = inputs(torch, 7).to(torch.bfloat16)
        base = leaf(x).float()
        generator = torch.Generator().manual_seed(5)

        def adapter(rank: int, size: float) -> tuple[Any, Any]:
            down = torch.nn.Linear(512, rank, bias=False).to(torch.bfloat16)
            up = torch.nn.Linear(rank, 384, bias=False).to(torch.bfloat16)
            with torch.no_grad():
                down.weight.copy_(torch.randn(rank, 512, generator=generator) * size)
                up.weight.copy_(torch.randn(384, rank, generator=generator) * size)
            return down, up

        layer = torch.nn.Module()
        layer.base_layer, layer.active_adapters, layer.scaling = leaf, ["default"], {"default": 0.7}
        down, up = adapter(16, 0.1)
        layer.lora_A, layer.lora_B = (
            torch.nn.ModuleDict({"default": down}),
            torch.nn.ModuleDict({"default": up}),
        )
        root = torch.nn.Module()
        root.lin = layer
        roles = {role: str(t.dtype).removeprefix("torch.") for role, t in parts.items()}
        rows = {
            "w": _row("transformer.lin.base_layer.weight", "svdq.nvfp4", roles),
            "a": _row("transformer.lin.lora_A.default.weight", "plain", {"value": "bf16"}),
            "b": _row("transformer.lin.lora_B.default.weight", "plain", {"value": "bf16"}),
        }
        bakes = weights._Bakes(torch, root, rows, set(), lambda module: None)
        assert bakes.prepare() == ("bake",) and layer._cozy_baked
        bakes(None, {"w": parts, "a": {"value": down.weight}, "b": {"value": up.weight}})
        assert all(
            torch.equal(
                parts[r].reshape(-1).view(torch.uint8), before[r].reshape(-1).view(torch.uint8)
            )
            for r in parts
        )

        def term(down: Any, up: Any, scale: float) -> Any:
            return (x.float() @ down.weight.float().t()) @ (up.weight.float() * scale).t()

        def off(got: Any, want: Any) -> float:
            return float((got - want).norm() / want.norm())

        assert off(leaf(x).float() - base, term(down, up, 0.7)) < 1e-2
        bakes.strategy = "per_step"
        bakes.prepare()
        assert leaf.lora is None and not layer._cozy_baked and torch.equal(leaf(x).float(), base)
        bakes.strategy = "bake"
        wide = adapter(112, 0.05)
        layer.lora_A["default"], layer.lora_B["default"] = wide
        bakes.prepare()
        assert off(leaf(x).float() - base, term(*wide, 0.7)) < 1e-2


@pytest.mark.parametrize(
    ("arch", "mma"),
    [
        (120, "mma.sync.aligned"),
        (121, "mma.sync.aligned"),
        (100, "tcgen05.mma"),
        (103, "tcgen05.mma"),
    ],
)
def test_every_kernel_builds_for_blackwell_on_the_block_scaled_fp4_mma(arch: int, mma: str) -> None:
    """Off a card: each entry the executor's start builds compiles for each Blackwell the leaf
    serves, and the GEMM multiplies on the FP4 tensor cores' block-scaled MMA, not an emulation."""
    if svdq_linear.TRITON_ABSENT:
        pytest.skip(svdq_linear.TRITON_ABSENT)
    triton = pytest.importorskip("triton")
    gemms = 0
    for entry in svdq_linear.KERNELS.entries:
        compiled = triton.compile(
            triton_kernels._source(entry),
            target=triton.backends.compiler.GPUTarget("cuda", arch, 32),
            options={"num_warps": entry.num_warps},
        )
        if entry.kernel is svdq_linear._gemm:
            ptx = compiled.asm["ptx"]
            assert mma in ptx and "kind::mxf4nvf4.block_scale" in ptx, entry.key
            gemms += 1
    assert gemms == 2 * len(svdq_linear.RANKS)


def test_the_kernels_are_the_law_on_a_blackwell_card() -> None:
    """The Triton build as an executor gets it, verified (`svdq_linear.verify`: each stage against
    the law), then DiT shapes at each rank and dtype against the law."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the svdq kernels run on a Blackwell card")
    device = torch.device("cuda", 0)
    launch = triton_kernels.load(svdq_linear.KERNELS, device=device)
    svdq_linear.verify(torch, device, launch)
    generator = torch.Generator(device=device).manual_seed(3)

    def draw(*shape: int) -> Any:
        return torch.randn(*shape, generator=generator, device=device)

    for rows, n, k in ((1, 512, 1024), (300, 3072, 3072), (4100, 1280, 5120)):
        for r in svdq_linear.RANKS:
            roles = {
                "weight": torch.randint(0, 256, (n, k // 2), generator=generator, device=device).to(
                    torch.uint8
                ),
                "scale": (draw(n, k // 16).abs() + 0.5).to(torch.float8_e4m3fn),
                "scale_2": torch.full((), 0.01, device=device),
                "pre_quant_scale": draw(k).abs() + 0.5,
                "lowrank_down": (draw(r, k) * 0.05).to(torch.bfloat16),
                "lowrank_up": (draw(n, r) * 0.05).to(torch.bfloat16),
            }
            for half in (torch.bfloat16, torch.float16):
                x, bias = draw(rows, k).to(half), draw(n).to(half)
                q, sx, low, g = svdq_linear._quantized(torch, launch, x, roles)
                got = svdq_linear._multiplied(torch, launch, q, sx, low, g, roles, bias, half)
                want = svdq_linear.eager(torch, x, roles, bias, half)
                error = float((got.float() - want.float()).norm() / want.float().norm())
                assert error < 1e-3, (rows, n, k, r, half, error)


def test_the_probe_checks_the_leaf_against_the_exact_product() -> None:
    """The Blackwell conformance's native suite: the leaf's output against the f64 product of the
    exact weight (residual, smoothing and branch)."""
    torch = pytest.importorskip("torch")
    provider = SvdqNvfp4NativeLeaf(params=PARAMS)
    agreement = probe._native_agreement(torch, provider, torch.device("cpu"), torch.bfloat16)
    assert agreement.passed and agreement.deviation_rel < 0.15, agreement
