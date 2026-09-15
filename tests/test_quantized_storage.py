"""Count actual upstream weight storage without rewriting logical model slots."""

import pytest

from cozy_runtime.internal.forensics import distinct_storage_bytes

torch = pytest.importorskip("torch")


def test_plain_views_count_storage_once_and_meta_has_no_storage() -> None:
    tensor = torch.arange(24, dtype=torch.float32)
    assert distinct_storage_bytes((tensor, tensor[4:9], tensor.view(2, 12))) == 96
    assert distinct_storage_bytes((tensor.to("meta"),)) == 0


def test_torchao_weight_counts_encoded_parts_without_changing_parameter_keys() -> None:
    pytest.importorskip("torchao")
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

    payload = torch.zeros(32, 16, dtype=torch.float8_e4m3fn)
    scales = torch.ones(32, 1, dtype=torch.float32)
    weight = Float8Tensor(
        payload,
        scales,
        block_size=[1, 16],
        dtype=torch.bfloat16,
        mm_config=Float8MMConfig(use_fast_accum=True),
        act_quant_kwargs=None,
        kernel_preference=KernelPreference.TORCH,
    )
    base = torch.nn.Linear(16, 32, bias=False, device="meta")
    base.weight = torch.nn.Parameter(weight, requires_grad=False)
    parameter = base.weight

    assert distinct_storage_bytes(base.parameters()) == payload.nbytes + scales.nbytes
    assert distinct_storage_bytes((base.weight, weight, payload, scales.view(-1))) == 640
    assert base.weight is parameter
    assert list(base.state_dict()) == ["weight"]
    assert base.weight.qdata.data_ptr() == payload.data_ptr()
    assert base.weight.scale.data_ptr() == scales.data_ptr()
    base.to("meta")
    assert distinct_storage_bytes(base.parameters()) == 0
