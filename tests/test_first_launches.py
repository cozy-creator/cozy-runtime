"""A GPU library's first launch takes device memory for itself (cuBLAS 36 MiB for a bf16 GEMM
on an L4), and with none left it fails under names that are not out-of-memory (run 3095:
CUBLAS_STATUS_INTERNAL_ERROR in a block's first Linear). Each case runs in a fresh process:
a library's handle is made once per process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import accel, weight_policy  # noqa: E402
from cozy_runtime.internal.weights import _room_for  # noqa: E402

on_card = pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card")

#: Leave about `left` bytes of the card to the driver: the rest is this process's ballast.
FILL = """
def fill(left):
    free = lambda: torch.cuda.mem_get_info(dev)[0]
    held = [torch.empty(max(free() - (96 << 20), 0), dtype=torch.uint8, device=dev)]
    while free() >= left + (2 << 20):  # the allocator's smallest segment
        held.append(torch.empty(1 << 20, dtype=torch.uint8, device=dev))
    return held
"""

STARTED = (
    """
import torch
from cozy_runtime.internal import accel
dev = torch.device("cuda", 0)
accel.initialize(torch, "cuda")
accel.first_launches(torch, "cuda")
"""
    + FILL
    + """
ops = []
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    a = torch.ones(64, 64, device=dev, dtype=dtype)
    x = torch.ones(1, 4, 16, 16, device=dev, dtype=dtype)
    w = torch.ones(4, 4, 3, 3, device=dev, dtype=dtype)
    ops.append(lambda a=a: torch.mm(a, a))
    ops.append(lambda x=x, w=w: torch.nn.functional.conv2d(x, w, padding=1))
ballast = fill(4 << 20)
with torch.inference_mode():
    for op in ops:
        op()
torch.cuda.synchronize()
print("ran", len(ops))
"""
)

#: A stage holds its weights and the card has `argv[3]` MiB left when bf16 and fp16 GEMMs
#: launch for the first time.
REFUSED = (
    """
import sys
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[1])
import test_weight_plane as twp
from cozy_runtime.internal.weights import Weights
dev = torch.device("cuda", 0)
"""
    + FILL
    + """
weights = Weights(torch, dev, "cuda")
weights.set_budget(-1, pinned=1 << 30)
model = twp.load(weights, twp.write_checkpoint(Path(sys.argv[2]))[0], "refused")
x = torch.ones(4, twp.HIDDEN, device=dev, dtype=torch.float16)
a = torch.ones(64, 64, device=dev, dtype=torch.bfloat16)
b = torch.ones(64, 64, device=dev, dtype=torch.float16)
model.residency.admit("run", ("stack",))
torch.cuda.synchronize()
ballast = fill(int(sys.argv[3]) << 20)
with weights.recovering(), torch.inference_mode():
    torch.mm(a, a)
    torch.mm(b, b)
    model.stack(x)
torch.cuda.synchronize()
model.residency.release("run", ("stack",))
print("ran")
"""
)

#: Idle weights are mapped and the card has `argv[3]` MiB left (0: nothing else is held) when a
#: convolution runs for the first time. Prints the workspace it ran with and the plane's bytes.
STARVED = (
    """
import sys
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[1])
import test_weight_plane as twp
from cozy_runtime.internal import plane
from cozy_runtime.internal.weights import Weights
dev = torch.device("cuda", 0)
"""
    + FILL
    + """
weights = Weights(torch, dev, "cuda")
weights.set_budget(-1, pinned=1 << 30)
model = twp.load(weights, twp.write_checkpoint(Path(sys.argv[2]))[0], "starved")
twp.run(model, torch.ones(4, twp.HIDDEN, device=dev, dtype=torch.float16), passes=1)
x = torch.randn(1, 64, 5, 128, 256, device=dev, dtype=torch.float16)
w = torch.randn(64, 64, 3, 3, 3, device=dev, dtype=torch.float16)
mapped = plane.stats(weights.plane).device(0).committed
torch.cuda.empty_cache()  # cuDNN takes a workspace from torch's cache before the driver
ballast = fill(int(sys.argv[3]) << 20) if int(sys.argv[3]) else []
torch.cuda.reset_peak_memory_stats(dev)
before = torch.cuda.memory_allocated(dev)
with weights.recovering(), torch.inference_mode():
    y = torch.nn.functional.conv3d(x, w, padding=1)
workspace = torch.cuda.max_memory_allocated(dev) - before - y.nbytes
print(workspace, mapped, plane.stats(weights.plane).device(0).committed)
"""
)


def fresh(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *args], capture_output=True, text=True, check=False
    )


@on_card
def test_a_started_device_runs_its_first_library_calls_on_a_full_card() -> None:
    done = fresh(STARTED)
    assert done.returncode == 0 and done.stdout.split()[-2:] == ["ran", "6"], done.stderr[-800:]


@on_card
@pytest.mark.parametrize("left", [2, 8])
def test_a_library_refused_inside_a_stage_runs_again_once_weights_made_room(
    tmp_path: Path, left: int
) -> None:
    """cuBLAS has its handle from the load. With a few MiB left on an L4 it refuses a dtype's
    first GEMM as CUBLAS_STATUS_NOT_SUPPORTED or INTERNAL_ERROR (bf16 under about 6 MiB, fp16
    under about 10), and runs it once the stage's weights made room."""
    done = fresh(REFUSED, str(Path(__file__).parent), str(tmp_path), str(left))
    assert done.returncode == 0 and done.stdout.split()[-1:] == ["ran"], done.stderr[-800:]


def test_a_library_refusal_is_given_room_once_and_then_is_its_own_failure() -> None:
    refusal = RuntimeError(
        "CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx( handle, opa, opb,"
    )
    assert _room_for(refusal, False) == (weight_policy.LIBRARY_ROOM, True)
    with pytest.raises(RuntimeError, match="CUBLAS_STATUS_INTERNAL_ERROR"):
        _room_for(refusal, True)
    cudnn = RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")
    assert _room_for(cudnn, False) == (weight_policy.LIBRARY_ROOM, True)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _room_for(RuntimeError("shape mismatch"), False)
    oom = RuntimeError("CUDA error: out of memory")
    assert _room_for(oom, True) == (0, True), "out of memory is always answered"


@on_card
def test_a_convolution_on_a_full_card_runs_the_kernel_it_runs_with_room(tmp_path: Path) -> None:
    """cuDNN catches a refused workspace and falls back to a convolution that needs none (on
    an RTX 4060 Ti, implicit_convolveNd_sgemm: 4-7x slower for SeedVR2's VAE), and torch keeps
    that choice for the shape as long as the process lives. Idle weights give the workspace
    room before cuDNN's next try: the card has room for the output, not for both."""
    here = str(Path(__file__).parent)
    roomy = fresh(STARVED, here, str(tmp_path / "roomy"), "0")
    full = fresh(STARVED, here, str(tmp_path / "full"), "40")
    assert roomy.returncode == 0 and full.returncode == 0, (roomy.stderr + full.stderr)[-1600:]
    wanted = int(roomy.stdout.split()[-3])
    if wanted <= 0:
        pytest.skip("this card's cuDNN runs the convolution without a workspace")
    workspace, mapped, kept = map(int, full.stdout.split()[-3:])
    assert workspace > 0, (roomy.stdout, full.stdout)
    assert kept < mapped, "the workspace came from idle weights"
