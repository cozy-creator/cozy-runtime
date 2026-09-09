"""Checkpoint dtype is immutable constructor input, before census and allocation."""

from types import SimpleNamespace

import pytest
import torch

from cozy_runtime.author import Artifact, Config, ConformanceError, Loader
from cozy_runtime.author._loader import TensorSpec
from cozy_runtime.internal import derive_child


@pytest.mark.parametrize("dtype,width", [("f32", 4), ("f16", 2)])
def test_checkpoint_dtype_controls_actual_destination_census(dtype: str, width: int) -> None:
    class Pipeline:
        def __init__(self, config: Config) -> None:
            selected = {"f32": torch.float32, "f16": torch.float16}[
                config.tensor_dtype("linear.weight", default="f32")
            ]
            with torch.device("meta"):
                self.components = {"linear": torch.nn.Linear(2, 3, bias=False, dtype=selected)}

    loader = Loader(
        Artifact(
            "native-header",
            {"linear.weight": TensorSpec((3, 2), dtype, nbytes=6 * width)},
            Config({}),
        ),
        owner=SimpleNamespace(),
        mode="derive",
    )
    model = loader.construct(Pipeline, factory=Pipeline)
    tensor = model.components["linear"].weight
    assert tensor.is_meta
    assert tensor.numel() * tensor.element_size() == 6 * width
    (destination,) = loader.records()[0].destinations
    assert destination.spec.dtype == dtype
    assert destination.spec.nbytes == 6 * width


def test_dtype_view_is_read_only_and_missing_supplied_key_refuses() -> None:
    source = {"linear.weight": "f16"}
    config = Config({"linear": {}}, tensor_dtypes=source)
    source["linear.weight"] = "f32"
    assert config.tensor_dtype("linear.weight", default="f32") == "f16"
    assert config.section("linear").tensor_dtype("linear.weight", default="f32") == "f16"
    assert config.mapping() == {"linear": {}}
    with pytest.raises(ConformanceError, match="no tensor"):
        config.tensor_dtype("linear.typo", default="f32")
    assert Config({}).tensor_dtype("linear.weight", default="f32") == "f32"


def test_private_derive_request_carries_dtype_metadata() -> None:
    import msgspec

    request = derive_child.DeriveRequest(
        (
            derive_child.SlotRequest(
                "sample.models.model", b"{}", tensor_dtypes={"linear.weight": "f16"}
            ),
        ),
        application="fixture:app",
    )
    assert derive_child.read_request(msgspec.msgpack.encode(request.render())) == request
