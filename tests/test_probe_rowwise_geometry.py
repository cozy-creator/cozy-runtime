"""Native qualification fixtures carry the encoding's actual stored scale shape."""

import pytest
import tensorfs
from tensorfs.derived import Part, Tensor

from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM
from cozy_runtime.internal.encoding.torchao_rowwise import TorchaoRowwiseLeaf
from cozy_runtime.internal.probe import _weight_roles

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")


def test_probe_flat_and_keepdim_codes_agree_and_both_match_their_declared_spec() -> None:
    previous = None
    for encoding in (SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM):
        _, parts = _weight_roles(torch, TorchaoRowwiseLeaf(encoding), torch.device("cpu"))
        rows, cols = parts["data"].shape
        shape = (rows,) if encoding == SPEC_ROWWISE else (rows, 1)
        spec = Tensor(
            "bf16",
            (rows, cols),
            encoding,
            {
                "data": Part("f8_e4m3fn", (rows, cols)),
                "scale": Part("f32", shape),
            },
        )
        assert tuple(parts["scale"].shape) == shape
        weight = tensorfs.from_parts(spec, parts)
        assert weight.qdata.data_ptr() == parts["data"].data_ptr()
        assert weight.scale.data_ptr() == parts["scale"].data_ptr()
        if previous is not None:
            assert torch.equal(weight.qdata.view(torch.uint8), previous.qdata.view(torch.uint8))
            assert torch.equal(weight.scale, previous.scale)
        previous = weight
