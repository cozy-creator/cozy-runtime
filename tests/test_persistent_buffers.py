"""A persistent buffer is checkpoint state, bound like a parameter (se-074).

Klein's VAE normalizes its latents with a batch norm's `running_mean`/`running_var`, read outside
any forward. The serving substrate kept persistent buffers as the host tensors their construction
made, so on a GPU varena bound the checkpoint's statistics only inside the module's forward and a
direct read returned mean 0, var 1: no latent normalization, washed-out images with an 8 px
speckle grid (runs 14398-14403). Built on meta like parameters, they become placeholders that
read the checkpoint's values anywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorfs")

from cozy_runtime.internal.derive import Observations, serving_substrate  # noqa: E402

NET = r"""
import numpy as np
import tensorfs
import torch

CHANNELS = 8


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(CHANNELS, CHANNELS, 1)
        self.bn = torch.nn.BatchNorm2d(CHANNELS, eps=1e-4, affine=False)
        self.register_buffer("derived", torch.arange(CHANNELS, dtype=torch.float32), persistent=False)

    def forward(self, x):
        return self.bn(self.conv(x))


def checkpoint(store):
    rng = np.random.default_rng(74)
    stored = {
        "conv.weight": rng.standard_normal((CHANNELS, CHANNELS, 1, 1), dtype=np.float32),
        "conv.bias": rng.standard_normal(CHANNELS, dtype=np.float32),
        "bn.running_mean": rng.standard_normal(CHANNELS, dtype=np.float32),
        "bn.running_var": rng.uniform(0.5, 4.0, CHANNELS).astype(np.float32),
        "bn.num_batches_tracked": np.array(7, dtype=np.int64),
    }
    writer = tensorfs.Writer(cache_dir=store)
    component = writer.component("net", cls=("torch", "", "Net"), config={})
    for key, value in stored.items():
        dtype = "I64" if value.dtype == np.int64 else "F32"
        component.put(key, value.tobytes(), dtype=dtype, shape=list(value.shape))
    return writer.commit(), stored
"""

CHILD = NET + r"""
import sys
from tensorfs.varena.torch import Memory, Placeholder
from cozy_runtime.internal.derive import Observations, serving_substrate

ck, stored = checkpoint(sys.argv[1])
with serving_substrate("cuda", Observations()):
    net = Net()
Memory.open(torch.device("cuda", 0)).load(net, ck=ck)
with torch.inference_mode():
    assert isinstance(net.bn.running_mean, Placeholder), type(net.bn.running_mean)
    mean = (net.bn.running_mean + 0).cpu()
    var = (net.bn.running_var + 0).cpu()
    assert torch.equal(mean, torch.from_numpy(stored["bn.running_mean"])), mean
    assert torch.equal(var, torch.from_numpy(stored["bn.running_var"])), var
    assert int(net.bn.num_batches_tracked) == 7
    x = torch.randn(1, CHANNELS, 4, 4, device="cuda")
    expected = (net.conv(x) - mean.cuda().view(1, -1, 1, 1)) / torch.sqrt(var.cuda().view(1, -1, 1, 1) + 1e-4)
    assert torch.allclose(net.eval()(x), expected, atol=1e-5)
print("OK")
"""


def test_construction_leaves_persistent_buffers_weightless() -> None:
    """Persistent buffers wait for the fill like parameters; derived ones keep their values."""
    with serving_substrate("cpu", Observations()):
        bn = torch.nn.BatchNorm2d(8, affine=True)
        bn.register_buffer("derived", torch.arange(8, dtype=torch.float32), persistent=False)
    assert bn.weight.is_meta
    assert bn.running_mean.is_meta and bn.running_var.is_meta and bn.num_batches_tracked.is_meta
    assert torch.equal(bn.derived, torch.arange(8, dtype=torch.float32))


@pytest.mark.real_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_persistent_buffer_reads_its_checkpoint_value_outside_a_forward(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path / "store")], capture_output=True, text=True
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), done.stderr[-3000:]
