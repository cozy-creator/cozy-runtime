"""Native checkpoint reads through the public tensor bridge and Runtime ownership."""

import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author._loader import census
from cozy_runtime.internal.base_weights import BaseWeights
from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM
from cozy_runtime.internal.encoding.torchao_rowwise import TorchaoRowwiseLeaf, bind_rowwise_weight
from cozy_runtime.internal.forensics import distinct_storage_bytes
from cozy_runtime.internal.lora import AdapterInput, LinearAdapters

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("peft")
pytest.importorskip("diffusers")


@pytest.mark.parametrize("encoding", [SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM])
def test_native_read_bind_refill_and_eviction_keep_request_factors(
    tmp_path: Path, encoding: str
) -> None:
    from cozy_runtime.models.minimax_h3.official import ResidentWeights

    shape = (8, 16)
    scales = torch.linspace(0.01, 0.08, 8, dtype=torch.float32)
    if encoding == SPEC_ROWWISE_KEEPDIM:
        scales = scales[:, None]
    original = {
        "data": torch.arange(128, dtype=torch.float32).reshape(shape).to(torch.float8_e4m3fn),
        "scale": scales,
    }
    spec = Tensor(
        "bf16",
        shape,
        encoding,
        {
            "data": Part("f8_e4m3fn", shape),
            "scale": Part("f32", tuple(scales.shape)),
        },
    )
    store = tensorfs.Store.init(tmp_path / "store")
    declaration = Derivation({}, {"dit": Target(add={"0.weight": spec})}, {}, [("dit", "0.weight")])
    identity = tensorfs.object_id(encoding.encode())
    writer = store.begin_derived(
        identity,
        1,
        *declaration.native_arguments(sum(part.nbytes for part in original.values())),
        work_fingerprint=identity,
    )
    for role, value in original.items():
        writer.add_part(
            "dit", "0.weight", role, io.BytesIO(value.view(torch.uint8).numpy().tobytes())
        )
    manifest = "sha256:" + writer.commit()["manifest"]["sha256"]
    base = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
    root = torch.nn.Sequential(base)
    owned = BaseWeights("dit", root, census(SimpleNamespace(components={"dit": root})))
    resident = ResidentWeights()
    with tensorfs.tensors(store, manifest, buffer_bytes=31) as reader:
        loaded = reader.get("dit", "0.weight")
        # A flat encoding has a flat stored scale; the upstream weight uses keepdim.
        parts = {"data": loaded.qdata, "scale": loaded.scale.view(spec.parts["scale"].shape)}
        previous = bind_rowwise_weight(
            base,
            parts["data"],
            parts["scale"],
            spec=reader.spec("dit", "0.weight"),
        )
        assert previous is not base.weight
        assert base.weight.qdata.data_ptr() == loaded.qdata.data_ptr()
        assert base.weight.scale.data_ptr() == loaded.scale.data_ptr()
        assert base.weight.mm_config.use_fast_accum
        assert base.weight.kernel_preference.name == "TORCH"
        assert base.weight.act_quant_kwargs is None
        assert distinct_storage_bytes(base.parameters()) == 160
        assert torch.equal(base.weight.qdata.view(torch.uint8), original["data"].view(torch.uint8))
        # Exercise the actual provider-probe boundary too, retaining the same declared codec.
        probe = TorchaoRowwiseLeaf(encoding).leaf(torch, parts, base, torch.bfloat16)
        assert probe.weight.qdata.data_ptr() == loaded.qdata.data_ptr()
        resident.record(base.weight)
        assert resident.verified(base.weight)
        reader.get("dit", "0.weight", out=base.weight)
        assert not resident.verified(base.weight)
        binding = LinearAdapters({"dit": root}, {"dit": manifest}, {"dit": ["0"]})
        adapter = AdapterInput(
            "one",
            "dit",
            manifest,
            {
                "0.lora_A.weight": torch.ones(2, 16, dtype=torch.bfloat16),
                "0.lora_B.weight": torch.ones(8, 2, dtype=torch.bfloat16),
            },
        )
        with binding.request(
            [adapter], device="cpu", max_device_bytes=4096, check_cancelled=lambda: None
        ):
            layer = root[0]
            factors = tuple(
                weight
                for name in layer.active_adapters
                for weight in (layer.lora_A[name].weight, layer.lora_B[name].weight)
            )
            assert all(value.device.type == "cpu" for value in factors)
            copied = [value.clone() for value in factors]
            aliases = tensorfs.physical_tensors(base.weight)
            owned.evict()
            assert base.weight.is_meta
            assert all(value.device.type == "cpu" for value in factors)
            assert all(
                torch.equal(value, expected)
                for value, expected in zip(factors, copied, strict=True)
            )
            assert all(value.device.type == "cpu" for value in aliases)
            owned.reserve("cpu")
            refreshed = reader.get("dit", "0.weight")
            bind_rowwise_weight(
                base, refreshed.qdata, refreshed.scale.view(spec.parts["scale"].shape), spec=spec
            )
            assert root[0] is layer and layer.base_layer is base
            assert all(
                actual is expected
                for actual, expected in zip(
                    (
                        weight
                        for name in layer.active_adapters
                        for weight in (layer.lora_A[name].weight, layer.lora_B[name].weight)
                    ),
                    factors,
                    strict=True,
                )
            )
            assert not resident.verified(base.weight)
        binding.close()
        assert root[0] is base


@pytest.mark.parametrize("fault", ["encoding", "scale_shape", "logical_dtype", "weight_shape"])
def test_declared_spec_mismatch_refuses_before_rebinding(fault: str) -> None:
    base = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
    weight = base.weight
    spec = Tensor(
        "bf16",
        (8, 16),
        SPEC_ROWWISE_KEEPDIM,
        {
            "data": Part("f8_e4m3fn", (8, 16)),
            "scale": Part("f32", (8, 1)),
        },
    )
    if fault == "encoding":
        spec = replace(spec, encoding="sha256:" + "0" * 64)
    elif fault == "scale_shape":
        spec = replace(spec, encoding=SPEC_ROWWISE)
    elif fault == "logical_dtype":
        spec = replace(spec, logical_dtype="f16")
    else:
        spec = replace(spec, shape=(16, 8))
    with pytest.raises(ValueError):
        bind_rowwise_weight(
            base, torch.ones(8, 16, dtype=torch.float8_e4m3fn), torch.ones(8, 1), spec=spec
        )
    assert base.weight is weight and "forward" not in base.__dict__
