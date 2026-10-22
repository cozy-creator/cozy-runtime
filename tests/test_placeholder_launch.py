"""An encoded leaf run outside its model's forward launches on its weight's bytes (cr-208).

Real GPU, real varena, the real int8 leaf and the Runtime wheel's prebuilt op. varena binds a
model's common weights only while the model or a direct child runs; elsewhere a weight is a
placeholder whose pointer is 0. Klein calls its modulation heads (ModuleDict entries, grandchildren
of its transformer) directly, so their int8 GEMV read address 0 and the device faulted (runs 13569,
13601). A device fault poisons its process, so the case runs in a child.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorfs")
pytestmark = [
    pytest.mark.real_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

CHILD = r"""
import sys
import numpy as np
import tensorfs
import torch
from tensorfs.varena.torch import Memory, Placeholder
from cozy_runtime.derive.microscale import encode_int8_convrot256
from cozy_runtime.internal import int8_linear
from cozy_runtime.internal.encoding import Int8ConvRotNativeLeaf

OUT, IN = 512, 1024
weight = np.random.default_rng(208).standard_normal((OUT, IN), dtype=np.float32)
encoded = encode_int8_convrot256("heads.txt", weight)
writer = tensorfs.Writer(cache_dir=sys.argv[1])
component = writer.component("net", cls=("torch", "", "Net"), config={})
component.put("heads.txt.data", encoded.payload, dtype="I8", shape=[OUT, IN])
component.put("heads.txt.scale", encoded.companions[0].raw, dtype="F32", shape=[OUT])
checkpoint = writer.commit()


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        meta = {"weight": torch.empty(OUT, IN, dtype=torch.int8, device="meta"),
                "scale": torch.empty(OUT, device="meta")}
        replaced = torch.nn.Linear(IN, OUT, bias=False, device="meta")
        leaf = Int8ConvRotNativeLeaf().leaf(torch, meta, replaced, torch.bfloat16)
        self.heads = torch.nn.ModuleDict({"txt": leaf})

    def forward(self, x):
        return self.heads["txt"](x)


net, memory = Net(), Memory.open(torch.device("cuda", 0))
memory.load(net, ck=checkpoint)
head = net.heads["txt"]
data = torch.frombuffer(bytearray(encoded.payload), dtype=torch.int8).view(OUT, IN).cuda()
scale = torch.frombuffer(bytearray(encoded.companions[0].raw), dtype=torch.float32).cuda()
generator = torch.Generator(device="cuda").manual_seed(208)
with torch.inference_mode():
    assert isinstance(head.data, Placeholder), type(head.data)
    for rows in (1, 8):  # the one-launch GEMV, then quantize and GEMM
        x = torch.randn(rows, IN, generator=generator, device="cuda").to(torch.bfloat16)
        inside = net(x)  # the model's forward: its common is bound
        outside = head(x)  # a grandchild run directly: its weight is a placeholder
        law = int8_linear.eager(torch, x, data, scale, torch.bfloat16)
        torch.cuda.synchronize()
        assert torch.equal(inside, outside), rows
        error = float((outside.float() - law.float()).norm() / law.float().norm())
        assert error < 1e-2, (rows, error)
    assert int8_linear.serving(0) == int8_linear.OP, int8_linear.serving(0)
    assert isinstance(head.data, Placeholder), "the read must not leave the weight bound"
print("OK")
"""


def test_a_leaf_outside_its_models_forward_reads_its_weight(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path / "store")], capture_output=True, text=True
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), done.stderr[-3000:]
