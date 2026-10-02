"""The Worker keeps each weight set's pinned tier across executors (weight-plane.md §3).

Real: two host-only `tensorfs.plane` instances (an executor and its replacement), the
executor's own durable exchange over a seam socketpair, the Worker's reply path and its
`HostTiers`. The replacement adopts the filled tier: it reads no byte from the store.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author._executor_requests import (  # noqa: E402
    Handler,
    HostTier,
    Reply,
    Request,
    respond,
)
from cozy_runtime.internal import plane  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.seam import Channel, SeamError  # noqa: E402
from cozy_runtime.internal.weights import HostTiers as Tiers  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402
from cozy_runtime.internal.worker.host_tier import HostTiers  # noqa: E402
from test_weight_plane import assert_pinned, load, write_checkpoint  # noqa: E402

pytestmark = pytest.mark.skipif(not plane.available(), reason="a TensorFS with the weight plane")


def answer_tier(tiers: HostTiers) -> Handler:
    def answer(request: Request) -> Reply:
        assert isinstance(request, HostTier)
        return tiers.answer(request)

    return answer


def serve(channel: Channel, tiers: HostTiers, stop: threading.Event) -> None:
    """The Worker's half of `child.Executor.call`'s request branch."""
    while not stop.is_set():
        frame = channel.recv()
        if frame is None:
            return
        received = channel.recv_memfd() if frame.get("descriptor") is True else None
        answer, handoff = respond(frame, answer_tier(tiers), received)
        channel.send(answer)
        if isinstance(handoff, int):
            channel.send_memfd(handoff)
            os.close(handoff)


def test_a_replacement_executor_adopts_the_tier_the_worker_kept(tmp_path: Path) -> None:
    checkpoint, values = write_checkpoint(tmp_path)
    tiers = HostTiers(lambda: 1 << 40)
    ours, theirs = socket.socketpair()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(Channel(theirs), tiers, stop), daemon=True)
    worker.start()
    executor = Executor(Channel(ours), tmp_path)
    seam = Tiers(ask=executor._ask_tier, offer=executor._offer_tier)
    try:
        first = Weights(torch, torch.device("cpu"), "cpu")
        first.set_budget(-1, pinned=1 << 30)
        filled = load(first, checkpoint, "tier", tiers=seam).backend.components["stack"]
        assert_pinned(first, filled, values)
        assert len(tiers.held) == 1 and tiers.nbytes() > 0
        filled_bytes = plane.stats(first.plane).host.fill_bytes
        assert filled_bytes > 0

        # The first executor goes (its weight set closes): the Worker's claim keeps the RAM.
        probe = os.dup(filled.ws.host_fd)
        first.forget("tier")
        assert os.fstat(probe).st_blocks * 512 >= filled_bytes

        second = Weights(torch, torch.device("cpu"), "cpu")
        second.set_budget(-1, pinned=1 << 30)
        adopted = load(second, checkpoint, "tier", tiers=seam).backend.components["stack"]
        assert_pinned(second, adopted, values)
        assert plane.stats(second.plane).host.fill_bytes == 0  # nothing read from the store

        # Released by the Worker and closed by the last executor, the tier holds no RAM.
        second.forget("tier")
        tiers.close()
        assert os.fstat(probe).st_blocks * 512 <= 4096  # its state page only
        os.close(probe)
    finally:
        stop.set()
        ours.close()
        worker.join()
        theirs.close()
        tiers.close()


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
    models.append(load(weights, Checkpoint(root, manifest), f"set{index}"))
    ticket = models[-1].residency.components["stack"].filling
    while ticket is not None and not ticket.done() and not throttled:
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
