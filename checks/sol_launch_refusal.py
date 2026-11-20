"""Exercise the Hopper prefix route when CuTe refuses an individual head shard."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from cozy_runtime.accel.attention import AttentionLayout
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.internal import attention_sol as sol


def dense(*, query: Tensor, key: Tensor, value: Tensor, scale: float | None) -> Tensor:
    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    return scaled_dot_product_attention(q, k, v, scale=scale).transpose(1, 2).contiguous()


def check() -> None:
    generator = torch.Generator().manual_seed(43)
    inputs = [torch.randn(1, 257, 2, 16, generator=generator) for _ in range(3)]
    layout = AttentionLayout(250, 64, 4, dense_until_step=4)
    site = sol.Site("dit", "attention")
    attempts: list[int] = []

    def refuses(*_: object) -> None:
        attempts.append(len(attempts))
        raise RuntimeError("CUDA Error Code: 1")

    def missing_keys(*_: object) -> None:
        raise AssertionError("a rejected launch has no output to merge")

    native = sol.Loaded(refuses)
    modules = {
        "sol_attn.interface": SimpleNamespace(_stream=lambda _: None),
        "sol_attn.common": SimpleNamespace(to_cute_tensor=lambda tensor: tensor),
        "sol_attn.preprocess": SimpleNamespace(
            prepare=lambda q, k, v, **_: (k[:, ::64], v[:, ::64], q[:, ::64, :, 0])
        ),
    }
    with (
        patch.object(sol, "import_module", modules.__getitem__),
        patch.object(sol, "_pin_preprocess"),
        patch.object(sol, "_card", return_value=(9, 0)),
        patch.object(sol, "_preprocessed", return_value=""),
        patch.object(sol, "_ready", return_value=(native, "")),
        patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()),
        sol.observing(kernels=(kernels := [])) as counts,
    ):
        q, k, v = inputs
        for _ in range(2):
            got = sol._execute(q, k, v, None, layout, site, dense=dense, keys=missing_keys)
            expected = dense(query=q[:, :250], key=k[:, :250], value=v[:, :250], scale=None)
            torch.testing.assert_close(got[:, :250], expected)
            assert torch.count_nonzero(got[:, 250:]) == 0
        assert len(attempts) == 1
        assert counts["dense_launch_refused"] == 2
        assert kernels[0]["source"] == "launch_refused"

        def accepts(*args: object) -> None:
            rows = args[3]
            assert isinstance(rows, Tensor)
            rows.fill_(0)

        native.function = accepts
        got = sol._execute(
            q, k, v, None, AttentionLayout(251, 64, 4), site, dense=dense, keys=lambda *_: None
        )
        assert got.shape == q.shape
        assert counts["sparse"] == 1

        def sticky(*_: object) -> None:
            raise RuntimeError("CUDA Error Code: 700")

        native.function = sticky
        try:
            sol._execute(
                q, k, v, None, AttentionLayout(252, 64, 4), site, dense=dense, keys=missing_keys
            )
        except RuntimeError as error:
            assert str(error) == "CUDA Error Code: 700"
            assert fault(error) is Fault.STICKY
        else:
            raise AssertionError("a device fault cannot be treated as an unsupported launch")


if __name__ == "__main__":
    check()
