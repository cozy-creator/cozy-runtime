"""Real CPU source/consumer checks; GPU placement is covered on rented devices."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import tensorfs  # noqa: E402
from tensorfs.varena.torch import Declared, Memory, Source, regions_of  # noqa: E402
from torch import nn  # noqa: E402

DIM, BLOCKS = 64, 4


def network() -> nn.Module:
    model = nn.Sequential(
        OrderedDict(
            embed=nn.Linear(DIM, DIM),
            blocks=nn.Sequential(
                *(
                    nn.Sequential(OrderedDict(lin=nn.Linear(DIM, DIM), relu=nn.ReLU()))
                    for _ in range(BLOCKS)
                )
            ),
            head=nn.Linear(DIM, DIM),
        )
    )
    model._no_split_modules = ("Sequential",)
    return model


def checkpoint(reference: nn.Module, folder: Path) -> tensorfs.Checkpoint:
    writer = tensorfs.Writer(cache_dir=folder)
    component = writer.component("net", cls="", config={})
    for key, tensor in reference.state_dict().items():
        component.put(
            key, tensor.detach().contiguous().numpy(), dtype="F32", shape=list(tensor.shape)
        )
    return writer.commit()


@pytest.mark.parametrize("blocks", [None, ["blocks.0"]])
def test_component_source_loads_without_a_daemon_or_a_host_deposit(
    tmp_path: Path, blocks: list[str] | None
) -> None:
    torch.manual_seed(0)
    reference = network()
    ck = checkpoint(reference, tmp_path / "store")
    with torch.device("meta"):
        model = network()
    model._varena = Declared(Source(ck, "net", ck.id), blocks=blocks)
    assert regions_of(model)
    memory = Memory.open(torch.device("cpu"))
    loaded = memory.load(model)
    sample = torch.randn(8, DIM, generator=torch.Generator().manual_seed(1))
    try:
        with torch.no_grad(), memory.request("source-load"):
            assert torch.equal(model(sample), reference(sample))
    finally:
        loaded.close()
        memory.close()
        ck.close()


def test_loaded_tensor_owns_its_bytes_after_checkpoint_and_model_close(tmp_path: Path) -> None:
    torch.manual_seed(7)
    reference = network()
    ck = checkpoint(reference, tmp_path / "store")
    with torch.device("meta"):
        model = network()
    memory = Memory.open(torch.device("cpu"))
    loaded = memory.load(model, ck=ck, component="net")
    retained = model.embed.weight
    expected_pointer = retained.data_ptr()
    ck.close()
    sample = torch.randn(8, DIM, generator=torch.Generator().manual_seed(8))
    with torch.no_grad(), memory.request("source-lifetime"):
        assert torch.equal(model(sample), reference(sample))
    loaded.close()
    memory.close()
    assert retained.data_ptr() == expected_pointer
    assert torch.equal(retained, reference.embed.weight)
