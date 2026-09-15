"""Real TensorFS -> CUDA fill -> upstream PEFT, including base-only restaging."""

from __future__ import annotations

import io
import weakref
from pathlib import Path
from typing import Any

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author._loader import census
from cozy_runtime.internal.encoding import SPEC_PLAIN, SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM
from cozy_runtime.internal.executor_adapters import AdapterModel
from cozy_runtime.internal.fill import Checkpoint, StreamingFillBackend, _Holder
from cozy_runtime.internal.forensics import distinct_storage_bytes
from cozy_runtime.internal.lora import AdapterInput

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("peft")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="streamed native FP8 qualification requires CUDA SM89 or newer",
)


@pytest.mark.parametrize("encoding", [SPEC_PLAIN, SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM])
def test_streamed_base_restage_preserves_peft_and_native_dispatch(
    tmp_path: Path, encoding: str
) -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    torch.manual_seed(739)
    torch.set_num_threads(4)
    shape = (128, 128)
    weight = torch.randn(shape, dtype=torch.bfloat16) * 0.05
    if encoding == SPEC_PLAIN:
        parts = {"value": weight}
        roles = {"value": Part("bf16", shape)}
    else:
        scales = weight.float().abs().amax(dim=1).clamp_min(1e-12) / 448
        parts = {
            "data": (weight.float() / scales[:, None]).to(torch.float8_e4m3fn),
            "scale": scales[:, None] if encoding == SPEC_ROWWISE_KEEPDIM else scales,
        }
        roles = {
            "data": Part("f8_e4m3fn", shape),
            "scale": Part("f32", tuple(parts["scale"].shape)),
        }
    stored_bytes = sum(value.nbytes for value in parts.values())
    store = tensorfs.Store.init(tmp_path / "store")
    declaration = Derivation(
        {},
        {"dit": Target(add={"linear.weight": Tensor("bf16", shape, encoding, roles)})},
        {},
        [("dit", "linear.weight")],
    )
    identity = tensorfs.object_id(encoding.encode())
    writer = store.begin_derived(
        identity, 1, *declaration.native_arguments(stored_bytes), work_fingerprint=identity
    )
    for role, value in parts.items():
        writer.add_part(
            "dit", "linear.weight", role, io.BytesIO(value.view(torch.uint8).numpy().tobytes())
        )
    manifest = "sha256:" + writer.commit()["manifest"]["sha256"]
    checkpoint = Checkpoint(tmp_path / "store", manifest)
    backend = StreamingFillBackend.for_script(
        checkpoint,
        checkpoint.rows("dit"),
        release="peft-streaming/1",
        store="fixture",
        snapshot=manifest,
        placement="component_staged",
        objective="footprint",
        window_bytes=1 << 20,
        slots=2,
        readers=1,
        inflight=1,
    )
    selected = backend.resolved["dit.linear.weight"]
    assert selected.route == ("verbatim" if encoding == SPEC_PLAIN else "encoded_gemm")
    if encoding != SPEC_PLAIN:
        assert selected.implementation == "torchao.fp8-rowwise.native/1"
    root = torch.nn.Module()
    base = root.linear = torch.nn.Linear(128, 128, bias=False, device="meta", dtype=torch.bfloat16)
    holder = _Holder({"dit": root})
    walked = census(holder)
    backend.expect({"dit": [destination.key for destination in walked.destinations]})
    backend.hold_lease = True
    live = backend.materialize(holder, walked, encoded_leaves="accept")
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    del live, walked
    backend.preserve_adapter_bases(["dit"])
    model = AdapterModel(
        {"dit": root},
        {"dit": manifest},
        [
            {
                "component": "dit",
                "base_snapshot": manifest,
                "target": "linear",
                "shape": list(shape),
                "dtype": "bf16",
            }
        ],
        str(tmp_path / "store"),
    )
    a = torch.randn(8, 128, dtype=torch.bfloat16) * 0.03
    b = torch.randn(128, 8, dtype=torch.bfloat16) * 0.03
    adapter = AdapterInput(
        "fixture",
        "dit",
        manifest,
        {"linear.lora_A.weight": a, "linear.lora_B.weight": b},
        scale=0.3,
    )
    value = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        baseline = root.linear(value).clone()
        residual = (
            torch.nn.functional.linear(torch.nn.functional.linear(value, a.cuda()), b.cuda()) * 0.3
        )
    scaled_operands: list[tuple[str, str]] = []

    class NativeOperands(TorchDispatchMode):  # type: ignore[misc]
        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if func is torch.ops.aten._scaled_mm.default:
                scaled_operands.append((str(args[0].dtype), str(args[1].dtype)))
            return func(*args, **(kwargs or {}))

    try:
        with model.linear.request(
            [adapter],
            device="cuda",
            max_device_bytes=a.nbytes + b.nbytes,
            check_cancelled=lambda: None,
            check_capacity=lambda size: None,
        ):
            layer = root.linear
            factors = tuple(
                p
                for name in layer.active_adapters
                for p in (layer.lora_A[name].weight, layer.lora_B[name].weight)
            )
            with (
                NativeOperands(),
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile,
            ):
                actual = layer(value)
            assert torch.equal(actual, baseline + residual)
            if encoding != SPEC_PLAIN:
                assert scaled_operands == [("torch.float8_e4m3fn", "torch.float8_e4m3fn")]
                assert any(event.name == "aten::_scaled_mm" for event in profile.events())
                cuda_events = [
                    event.name
                    for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
                assert cuda_events
                profile.export_chrome_trace(str(tmp_path / "native-peft-trace.json"))
            for _ in range(3):
                assert backend.evict("dit") == stored_bytes
                assert base.weight.is_meta
                assert all(p.device.type == "cuda" for p in factors)
                backend.stage("dit")
                assert root.linear is layer and layer.base_layer is base
                assert tuple(backend.base_weights["dit"].live()) == ("dit.linear.weight",)
                assert backend.vram_charge["dit"] == stored_bytes
                assert distinct_storage_bytes(backend.base_weights["dit"].tensors()) == stored_bytes
                assert all(
                    p is q
                    for p, q in zip(
                        factors,
                        (
                            p
                            for name in layer.active_adapters
                            for p in (layer.lora_A[name].weight, layer.lora_B[name].weight)
                        ),
                        strict=True,
                    )
                )
                assert torch.equal(layer(value), actual)
        assert root.linear.active_adapters == []
        held = [weakref.ref(p) for p in factors]
        del factors
        assert all(reference() is None for reference in held)
        with torch.no_grad():
            assert torch.equal(root.linear(value), baseline)
    finally:
        model.linear.close()
        backend.evict("dit")
        backend.close()
