"""The executor's pinned host tier: the seam passes tiers only as memfds, a pinned budget
bounds the fill from its first byte, and the tier stays under the cgroup's `memory.high`.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.seam import Channel, SeamError  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402
from test_weight_plane import load, write_checkpoint  # noqa: E402


def test_the_seam_carries_only_memfds_as_tiers(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    with left, right, (tmp_path / "file").open("w") as regular:
        with pytest.raises(SeamError, match="memfd"):
            Channel(left).send_memfd(regular.fileno())
        memfd = ctypes.CDLL(None, use_errno=True).memfd_create(b"tier", 0)
        assert memfd >= 0
        try:
            Channel(left).send_memfd(memfd)
            received = Channel(right).recv_memfd()
            assert os.path.samestat(os.fstat(received), os.fstat(memfd))
            os.close(received)
        finally:
            os.close(memfd)


class GpuTier(Weights):
    """A GPU executor's host tier, without its device: `register` fills as a card's does."""

    def register(self, component: object) -> None:
        self.host_only = False
        try:
            super().register(component)  # type: ignore[arg-type]
        finally:
            self.host_only = True


def test_a_pinned_budget_given_with_the_load_bounds_the_fill_from_its_first_byte(
    tmp_path: Path,
) -> None:
    """TensorD's pilot pinned SDXL whole (7.13 GB) and only then cut it to the 4 GiB
    it sent after Load. The budget now comes with the Load (`Load.pinned_bytes`), applied
    before the weight set registers: a prefix fills inside it, and nothing is pinned past it
    and punched back. Without one, the executor sizes its own tier from the host."""
    checkpoint = write_checkpoint(tmp_path)[0]
    budget = 96 << 20  # below the set's 128 MiB
    weights = GpuTier(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, budget)  # what `Executor.load` applies from `Load.pinned_bytes`
    filling = load(weights, checkpoint, "bounded").residency.components["stack"].filling
    assert filling
    for ticket in filling:
        ticket.wait()
    held = plane.stats(weights.plane).host
    assert 0 < held.fill_bytes <= held.used <= budget == held.budget, held
    assert held.evictions == 0, held

    unbounded = GpuTier(torch, torch.device("cpu"), "cpu")
    filling = load(unbounded, checkpoint, "own").residency.components["stack"].filling
    assert filling
    for ticket in filling:
        ticket.wait()
    assert plane.stats(unbounded.plane).host.used > budget  # its own share holds the set


def test_a_weight_set_closes_after_its_pinned_tier_gave_regions_back(tmp_path: Path) -> None:
    """Run 4365 (SDXL in a 1 GiB pool): the host had less, the pinned tier gave back regions
    its fill had asked for, and closing the set (to refine it) reported `region 80 left the
    pinned tier` as the run's failure. The tier is optional: the set closes."""
    weights = GpuTier(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, 1 << 30)
    component = load(weights, write_checkpoint(tmp_path)[0], "shrunk").residency.components["stack"]
    for ticket in component.filling:
        ticket.wait()
    weights._pin(component.regions[0].span)  # the host has room for one region now
    assert plane.stats(weights.plane).host.evictions > 0
    weights.forget("shrunk")
    assert not component.filling


#: Runs inside a small cgroup: fill the host tier of every weight set given, as a GPU executor
#: does at load (`Weights.register`), and report what the cgroup holds. It stops the moment
#: memory that cannot be reclaimed reaches `memory.high`, where the kernel throttles the fill.
FILL = """
import json, os, sys, time
from pathlib import Path
import torch
torch.set_num_threads(2)
sys.path.insert(0, sys.argv[1])
from cozy_runtime.internal import plane
from cozy_runtime.internal.fill import Checkpoint
from tensorfs import Store
from cozy_runtime.internal.weights import Weights
from test_weight_plane import load

own = Path("/proc/self/cgroup").read_text().split("::")[1].strip()
cgroup = Path("/sys/fs/cgroup") / own.lstrip("/")
high = int((cgroup / "memory.high").read_text())

def held():
    rows = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    return int(rows["anon"]), int(rows["shmem"])

class GpuTier(Weights):  # a GPU executor's host tier, without its device
    def register(self, component):
        self.host_only = False
        try:
            super().register(component)
        finally:
            self.host_only = True

    def hold_host(self):
        self.host_only = False
        try:
            getattr(super(), "hold_host", lambda: None)()
        finally:
            self.host_only = True

if sys.argv[3]:  # fill the cgroup with page cache first, as a pod's downloaded weights do
    with open(sys.argv[3], "wb") as file:
        for _ in range(96):
            file.write(os.urandom(8 << 20))
    with open(sys.argv[3], "rb") as file:
        while file.read(8 << 20):
            pass
cached = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
weights = GpuTier(torch, torch.device("cpu"), "cpu")
started, models, throttled = time.monotonic(), [], False
for index, (root, manifest) in enumerate(json.loads(sys.argv[2])):
    models.append(load(weights, Checkpoint.from_store(Store.open(root), manifest), f"set{index}"))
    tickets = models[-1].residency.components["stack"].filling
    while not all(ticket.done() for ticket in tickets) and not throttled:
        throttled = sum(held()) >= high
        time.sleep(0.01)
    if throttled:
        break
pinned = plane.stats(weights.plane).host.used
# Process memory grows afterwards (a stage's activations): the tier gives way at each pass.
grown = []
while not throttled and len(grown) < 20:
    grown.append(bytearray(os.urandom(16 << 20)))
    weights.hold_host()
    throttled = sum(held()) >= high
anon, shmem = held()
print(json.dumps({"high": high, "anon": anon, "shmem": shmem, "loaded": len(models),
                  "throttled": throttled or anon + shmem >= high, "pinned_before": pinned,
                  "cache": int(cached["inactive_file"]) + int(cached["active_file"]),
                  "pinned": plane.stats(weights.plane).host.used,
                  "seconds": round(time.monotonic() - started, 2)}), flush=True)
os._exit(0)
"""


@pytest.mark.skipif(shutil.which("systemd-run") is None, reason="systemd-run for a cgroup")
@pytest.mark.parametrize("cached", [False, True])
def test_the_pinned_tier_stays_under_the_cgroups_memory_high(tmp_path: Path, cached: bool) -> None:
    """The laptop under MemoryHigh 12 GiB / MemoryMax 16 GiB, no swap (rebench, 2026-10-02):
    the pinned tiers of SDXL and Anima grew to 10 GiB of shared memory beside 2.7 GiB of
    process memory, the cgroup sat over `memory.high`, and the kernel throttled a weight load
    for 11 minutes. The host budget was `memory.max - memory.current`: it never read
    `memory.high`. Here five weight sets (640 MiB) load in a cgroup whose `memory.high` is
    below them: every load completes, a prefix of the weights is pinned, the rest stays on
    disk, and what cannot be reclaimed stays under `memory.high`, also while the process
    then grows by 320 MiB. `cached`: the cgroup is first filled with page cache (a rented
    pod sits at its limit with the weights it downloaded); that memory is reclaimable, so
    the loads proceed and the tier still pins."""
    sets = [write_checkpoint(tmp_path / str(index))[0] for index in range(5)]
    scope = ("systemd-run", "--user", "--scope", "--quiet", "--collect")
    limits = (
        "-p",
        f"MemoryHigh={800 << 20}",
        "-p",
        f"MemoryMax={4 << 30}",
        "-p",
        "MemorySwapMax=0",
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    if subprocess.run([*scope, *limits, "true"], capture_output=True, env=env).returncode != 0:
        pytest.skip("this session cannot create a memory-limited scope")
    rows = json.dumps([[checkpoint.root, checkpoint.manifest_id] for checkpoint in sets])
    tests = str(Path(__file__).parent)
    cache = str(tmp_path / "cache.bin") if cached else ""
    ran = subprocess.run(
        [*scope, *limits, sys.executable, "-c", FILL, tests, rows, cache],
        capture_output=True,
        text=True,
        env=env,
    )
    assert ran.returncode == 0 and ran.stdout.strip(), ran.stderr[-2000:]
    facts = json.loads(ran.stdout.strip().splitlines()[-1])
    assert not facts["throttled"] and facts["loaded"] == len(sets), facts
    assert facts["anon"] + facts["shmem"] < facts["high"], facts
    assert 0 < facts["pinned_before"] < 640 << 20, facts  # a prefix is pinned, never all
    assert facts["pinned"] < facts["pinned_before"], facts  # and it gave way to the process
    assert not cached or facts["cache"] > 256 << 20, facts  # the cgroup really was full of it
