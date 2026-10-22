"""A conv's cuDNN engine never depends on an earlier attempt's free memory (cr-206).

Real GPU, real cuDNN, varena's real catch. Each arm is a fresh process: an attempt runs its first
conv of each shape starved of workspace and then crashes; the next attempt must be byte-identical
to a process that never saw the crash. Without `library_oom` cuDNN silently takes another engine,
caches it on the thread, and the next attempt differs (the control arm shows it on this GPU).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

#: H3 audio-VAE-like convolutions: (in, out, kernel, dilation, stride, groups, transposed, length).
SHAPES = [
    (512, 512, 7, 1, 1, 1, False, 5175),
    (256, 256, 11, 3, 1, 1, False, 10350),
    (128, 128, 7, 5, 1, 1, False, 20700),
    (64, 64, 3, 1, 1, 1, False, 41400),
    (512, 256, 4, 1, 2, 1, True, 5175),
    (256, 128, 4, 1, 2, 1, True, 10350),
    (256, 256, 12, 1, 1, 256, False, 20700),
    (128, 128, 12, 1, 2, 128, True, 20700),
]

CHILD = r"""
import hashlib, json, sys
import torch
from tensorfs.varena import cuts
from cozy_runtime.internal import library_oom

arm, shapes = sys.argv[1], json.loads(sys.argv[2])
torch.manual_seed(206)
seen = []
torch._C._cuda_attach_out_of_memory_observer(lambda d, a, t, f: seen.append(a))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
gen = torch.Generator().manual_seed(206)
def layer(i, o, k, d, s, g, transposed, n):
    if transposed:
        conv = torch.nn.ConvTranspose1d(i, o, k, stride=s, padding=(k - s) // 2, groups=g)
    else:
        conv = torch.nn.Conv1d(i, o, k, stride=s, dilation=d, padding=d * (k - 1) // 2, groups=g)
    return conv.cuda().eval(), torch.randn(1, i, n, generator=gen).cuda()


layers = [layer(*shape) for shape in shapes]
torch.nn.functional.conv1d(torch.ones(1, 1, 8, device="cuda"), torch.ones(1, 1, 3, device="cuda"))


class Hog:
    # varena's rungs over one tensor: making room releases it.
    held = None

    def free(self, device, need):
        released, Hog.held = Hog.held is not None, None
        torch.cuda.empty_cache()
        return released

    def lower(self, device, need):
        return False

    def available(self, device):
        return torch.cuda.mem_get_info()[0]

    def exhausted(self, device, need, op):
        return torch.OutOfMemoryError(f"{op} needs {need} B")

    def noted(self, rung, why, nbytes):
        pass


memo = cuts.Memo()
cuts.serve(memo, Hog())
if arm == "surfaced":
    library_oom.install(torch)


def attempt(starve):
    with torch.no_grad(), cuts.caught(), (library_oom.surfaced() if arm == "surfaced" else _none()):
        out = []
        for conv, x in layers:
            if starve:  # its output fits; any workspace above 1 MiB does not
                torch.cuda.empty_cache()
                numel = _out_numel(conv, x)
                room = -(-numel * 4 // (2 << 20)) * (2 << 20) + (1 << 20)
                Hog.held = torch.empty(torch.cuda.mem_get_info()[0] - room, dtype=torch.uint8,
                                       device="cuda")
            out.append(hashlib.sha256(conv(x).cpu().numpy().tobytes()).hexdigest()[:16])
            Hog.held = None
        if starve:
            raise RuntimeError("the attempt crashed")
        return out


def _out_numel(conv, x):
    n = x.shape[-1]
    if isinstance(conv, torch.nn.ConvTranspose1d):
        k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        length = (n - 1) * s - 2 * p + k
    else:
        k, s, p, d = conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.dilation[0]
        length = (n + 2 * p - d * (k - 1) - 1) // s + 1
    return conv.out_channels * length


class _none:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if arm != "fresh":
    try:
        attempt(starve=True)
    except RuntimeError as exc:
        assert "crashed" in str(exc), exc
starved = list(seen)
print("RESULT " + json.dumps(attempt(starve=False)))
print("INFO " + json.dumps({"starved_ooms": starved, "later_ooms": seen[len(starved):], "retries": memo.retries}))
"""


def _arm(arm: str) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, arm, json.dumps(SHAPES)], capture_output=True, text=True
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, done.stderr[-2000:]
    return list(json.loads(lines[-1][7:]))


def test_a_crashed_starved_attempt_leaves_the_next_attempt_bytewise_fresh() -> None:
    assert _arm("surfaced") == _arm("fresh")


def test_without_it_the_starved_engine_outlives_the_crash() -> None:
    """The control: the defect the module removes, on this GPU's cuDNN (RTX PRO 4500: 6 of 8)."""
    if _arm("unsurfaced") == _arm("fresh"):
        pytest.skip("this GPU's cuDNN chose the same engines starved: nothing to show")
