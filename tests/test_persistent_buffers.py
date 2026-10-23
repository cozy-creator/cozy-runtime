"""A persistent buffer is checkpoint state, bound like a parameter, and a load says where its
construction and checkpoint disagree (se-074).

Klein's VAE normalizes its latents with a batch norm's `running_mean`/`running_var`, read outside
any forward. The serving substrate kept persistent buffers as the host tensors their construction
made, varena binds only meta tensors, so on a GPU the checkpoint's statistics were bound only
inside the module's forward and a direct read returned mean 0, var 1: no latent normalization,
washed-out images with an 8 px speckle grid (runs 14398-14403), and nothing said so.

Real serving substrate, real TensorFS checkpoint, real Runtime fill and real varena on the host:
every stored tensor reaches varena on meta, so it binds at every name; a persistent buffer the
checkpoint does not store keeps its constructed value; stored tensors no destination reads and
unstored persistent buffers are named once in the load's warnings, and the model serves.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tensorfs = pytest.importorskip("tensorfs")
pytest.importorskip("tensorfs.varena.torch")

from tensorfs.varena.torch import Memory  # noqa: E402

from cozy_runtime.author import Model, uses_components  # noqa: E402
from cozy_runtime.author._context import Device  # noqa: E402
from cozy_runtime.author._loader import Artifact, Config, Loader  # noqa: E402
from cozy_runtime.author._model import WARNINGS, ModelRegistry  # noqa: E402
from cozy_runtime.internal import canonical, model_config  # noqa: E402
from cozy_runtime.internal.derive import Observations, serving_substrate  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of  # noqa: E402
from cozy_runtime.internal.weights import Backend  # noqa: E402
from store_fixtures import home  # noqa: E402

CHANNELS = 8


class Net(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(CHANNELS, CHANNELS, 1)
        self.bn = torch.nn.BatchNorm2d(CHANNELS, eps=1e-4, affine=False)
        self.register_buffer("scale", torch.full((CHANNELS,), 3.0))  # persistent, never stored
        self.register_buffer("derived", torch.arange(CHANNELS, dtype=torch.float32), persistent=False)


class Graph:
    def __init__(self, config: Config) -> None:
        self.components = {"net": Net()}


class Normalized(Model[Graph]):
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=Graph)

    @uses_components("net")
    def normalize(self, x: Any) -> Any:
        net = self.graph.components["net"]
        return net.bn.eval()(net.conv(x))


class Declared(Normalized, unloaded=("net.head.weight",), derived=("net.scale",)):
    """The same model, declaring the stored tensor it leaves unloaded and the buffer it derives."""


class Recording(Memory):  # type: ignore[misc]
    """varena's memory, noting which stored tensors it was handed off meta."""

    stored: tuple[str, ...] = ()
    real: list[str]

    def load(self, module: Any, *args: Any, **kwargs: Any) -> Any:
        state = module.state_dict(keep_vars=True)
        self.real = [k for k in self.stored if not state[k].is_meta]
        return super().load(module, *args, **kwargs)


def _stored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Checkpoint, dict[str, np.ndarray]]:
    rng = np.random.default_rng(74)
    values = {
        "conv.weight": rng.standard_normal((CHANNELS, CHANNELS, 1, 1), dtype=np.float32),
        "conv.bias": rng.standard_normal(CHANNELS, dtype=np.float32),
        "bn.running_mean": rng.standard_normal(CHANNELS, dtype=np.float32),
        "bn.running_var": rng.uniform(0.5, 4.0, CHANNELS).astype(np.float32),
        "bn.num_batches_tracked": np.array(7, dtype=np.int64),
        "head.weight": rng.standard_normal((2, CHANNELS), dtype=np.float32),  # nothing builds it
    }
    writer = tensorfs.Writer(cache_dir=home(tmp_path, monkeypatch))
    component = writer.component("net", cls=("torch", "", "Net"), config={"channels": CHANNELS})
    for key, value in values.items():
        dtype = "I64" if value.dtype == np.int64 else "F32"
        component.put(key, value, dtype=dtype, shape=list(value.shape))
    return Checkpoint(writer.commit()), values


def _load(cls: type[Normalized], ck: Checkpoint) -> tuple[Any, Recording, Backend]:
    rows = ck.rows("net")
    memory = Recording.open(torch.device("cpu"))
    memory.stored = tuple(r.key.removeprefix("net.") for r in rows if r.key != "net.head.weight")
    backend = Backend.for_script(
        {"net": ck}, rows, torch=torch, memory=memory, device=torch.device("cpu"),
        construction="se-074", release="se-074",
    )  # fmt: skip
    document = canonical.parse_canonical(model_config.construction_config(ck.configs()))
    artifact = Artifact(ck.id, tensor_schema_of(rows), Config(document, "net.config"))
    registry = ModelRegistry(
        release="se-074", backend=backend, device=Device("cpu"),
        substrate=lambda: serving_substrate("cpu", Observations()),
    )  # fmt: skip
    model = registry.acquire("serve.models.net", cls, artifact)
    return model.graph.components["net"], memory, backend


def test_declared_gaps_load_without_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ck, _ = _stored(tmp_path, monkeypatch)
    WARNINGS.clear()
    net, _, backend = _load(Declared, ck)
    try:
        assert torch.equal(net.scale, torch.full((CHANNELS,), 3.0))
        assert list(WARNINGS) == [], list(WARNINGS)
    finally:
        WARNINGS.clear()
        backend.close()


def test_every_stored_tensor_binds_and_a_load_names_what_it_did_not_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ck, values = _stored(tmp_path, monkeypatch)
    WARNINGS.clear()
    net, memory, backend = _load(Normalized, ck)
    try:
        assert memory.real == [], f"stored tensors handed to varena off meta: {memory.real}"
        with torch.inference_mode():
            mean = net.bn.running_mean.view(1, -1, 1, 1) + 0
            var = net.bn.running_var.view(1, -1, 1, 1) + 0
        assert torch.equal(mean.flatten(), torch.from_numpy(values["bn.running_mean"]))
        assert torch.equal(var.flatten(), torch.from_numpy(values["bn.running_var"]))
        assert int(net.bn.num_batches_tracked) == 7
        assert torch.equal(net.scale, torch.full((CHANNELS,), 3.0)), "an unstored buffer keeps its value"
        assert torch.equal(net.derived, torch.arange(CHANNELS, dtype=torch.float32))
        with torch.inference_mode(), memory.request("se-074"):
            x = torch.randn(1, CHANNELS, 4, 4)
            out = net.bn.eval()(net.conv(x))
            want = (net.conv(x) - mean) / torch.sqrt(var + 1e-4)
        torch.testing.assert_close(out, want)
        rows_seen = {message: fields for message, fields in WARNINGS}
        assert len(rows_seen) == 2, rows_seen
        assert any(f.get("tensors") == "net/head.weight" for f in rows_seen.values()), rows_seen
        assert any(f.get("buffers") == "net/scale" for f in rows_seen.values()), rows_seen
    finally:
        WARNINGS.clear()
        backend.close()


CHILD = r"""
import sys
import numpy as np
import tensorfs
import torch
from tensorfs.varena.torch import Memory, Placeholder
from cozy_runtime.internal.derive import Observations, serving_substrate
from cozy_runtime.internal.weights import _weightless_buffers

CHANNELS = 8
rng = np.random.default_rng(74)
stored = {
    "conv.weight": rng.standard_normal((CHANNELS, CHANNELS, 1, 1), dtype=np.float32),
    "conv.bias": rng.standard_normal(CHANNELS, dtype=np.float32),
    "bn.running_mean": rng.standard_normal(CHANNELS, dtype=np.float32),
    "bn.running_var": rng.uniform(0.5, 4.0, CHANNELS).astype(np.float32),
    "bn.num_batches_tracked": np.array(7, dtype=np.int64),
}
writer = tensorfs.Writer(cache_dir=sys.argv[1])
component = writer.component("net", cls=("torch", "", "Net"), config={})
for key, value in stored.items():
    component.put(key, value, dtype="I64" if value.dtype == np.int64 else "F32", shape=list(value.shape))
ck = writer.commit()


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(CHANNELS, CHANNELS, 1)
        self.bn = torch.nn.BatchNorm2d(CHANNELS, eps=1e-4, affine=False)


with serving_substrate("cuda", Observations()):
    net = Net()
_weightless_buffers(net, "net", {f"net.bn.{k}" for k in ("running_mean", "running_var", "num_batches_tracked")})
Memory.open(torch.device("cuda", 0)).load(net, ck=ck)
with torch.inference_mode():
    assert isinstance(net.bn.running_mean, Placeholder), type(net.bn.running_mean)
    mean = (net.bn.running_mean.view(1, -1, 1, 1) + 0).cpu()
    assert torch.equal(mean.flatten(), torch.from_numpy(stored["bn.running_mean"])), mean
print("OK")
"""


@pytest.mark.real_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_stored_buffer_reads_its_checkpoint_value_outside_a_forward_on_a_gpu(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path / "store")], capture_output=True, text=True
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), done.stderr[-3000:]
