"""Checkpoint dtype is immutable constructor input, before census and allocation."""

import io
import mmap
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
import tensorfs
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import Artifact, Config, ConformanceError, Loader
from cozy_runtime.author._loader import TensorSpec
from cozy_runtime.internal import derive_child, plane
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of
from cozy_runtime.internal.weights import PlaneBackend, Weights
from cozy_runtime.internal.weights_sink import weights_transaction_id

torch = pytest.importorskip("torch")


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
    request = derive_child.DeriveRequest(
        (
            derive_child.SlotRequest(
                "sample.models.model", b"{}", tensor_dtypes={"linear.weight": "f16"}
            ),
        ),
        application="fixture:app",
    )
    assert derive_child.read_request(msgspec.msgpack.encode(request.render())) == request


@pytest.mark.skipif(not plane.available(), reason="a TensorFS with the weight plane")
@pytest.mark.parametrize("dtype,width", [("f32", 4), ("f16", 2)])
def test_native_checkpoint_registers_the_dtype_selected_constructor(
    tmp_path: Path,
    dtype: str,
    width: int,
) -> None:
    selected_dtype = {"f32": torch.float32, "f16": torch.float16}[dtype]
    source = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=selected_dtype)
    store = tensorfs.Store.ensure(str(tmp_path / "store"))
    transaction = store.begin_derived(
        weights_transaction_id("dtype-proof", "proof", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={
                "linear": Target(
                    add={
                        "weight": Tensor(dtype, (3, 2), SPEC_PLAIN, {"value": Part(dtype, (3, 2))})
                    }
                )
            },
            configs={"model": NativeConfig("add")},
            order=(("linear", "weight"),),
        ).native_arguments(8192),
        work_fingerprint="sha256:" + "2" * 64,
    )
    transaction.add_part("linear", "weight", "value", io.BytesIO(source.numpy().tobytes()))
    transaction.add_config("model", io.BytesIO(b"{}"))
    receipt = transaction.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    checkpoint = Checkpoint(tmp_path / "store", manifest)
    rows = checkpoint.rows("linear")
    # The host tier alone: registration and the pinned fill need no card.
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=64 << 20)
    backend = PlaneBackend.for_script(
        {"linear": checkpoint}, rows, weights=weights, construction="dtype", release="dtype/1"
    )

    class Pipeline:
        def __init__(self, config: Config) -> None:
            requested = {"f32": torch.float32, "f16": torch.float16}[
                config.tensor_dtype("linear.weight", default="f32")
            ]
            with torch.device("meta"):
                self.components = {"linear": torch.nn.Linear(2, 3, bias=False, dtype=requested)}

    try:
        backend.expect({"linear": ("linear.weight",)})
        loader = Loader(
            Artifact(manifest, tensor_schema_of(rows), Config({})),
            owner=SimpleNamespace(),
            backend=backend,
        )
        model = loader.construct(Pipeline, factory=Pipeline)
        # A CPU executor binds the registered bytes in place, in the selected dtype.
        assert torch.equal(model.components["linear"].weight, source)
        assert loader.records()[0].filled_bytes == width * 6
        component = backend.components["linear"]
        (region,) = component.regions
        (weight,) = region.weights
        ((offset, stored, shape),) = weight.parts.values()
        assert (stored, shape) == (dtype, (3, 2))
        weights.plane.want(component.ws, plane.PINNED).wait()
        with mmap.mmap(component.ws.host_fd, component.ws.nbytes, prot=mmap.PROT_READ) as held:
            start = region.offset + offset
            assert held[start : start + width * 6] == source.numpy().tobytes()
    finally:
        backend.close()
