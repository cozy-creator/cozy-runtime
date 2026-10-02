"""Native Rust machine -> unchanged SDK TensorFS host adoption, CPU-only."""
import json
import mmap
import os
import subprocess
from pathlib import Path

import pytest
from tensorfs import plane as native_plane

from cozy_runtime.author._executor_requests import HostTierPlan
from cozy_runtime.internal import canonical
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.model_sources import BrokeredModelSources
from cozy_runtime.internal.seam import connect
from test_machine_model_sources import owner_model


@pytest.fixture
def machine_host_owner_binary(request):
    binary = request.config.getoption("machine_host_owner_binary", default=None)
    if binary is None:
        pytest.skip("requires explicitly built native CPU owner fixture binary")
    return Path(binary)


def test_real_machine_owner_fills_sdk_adopts_without_private_fill(
    tmp_path, machine_host_owner_binary
):
    _store, manifest, lease, _files, _header, data = owner_model(tmp_path)
    socket_path = tmp_path / "owner.sock"
    config = tmp_path / "owner.json"
    config.write_text(json.dumps({
        "store": str(tmp_path / "owner"),
        "selection": {"manifest": manifest, "components": ["linear"]},
        "scope": {"actor": "owner-fixture", "plan": "selected-fixture-plan"},
        "cache": {"budget_bytes": 8 << 20, "readers": 2, "max_entries": 2},
        "socket": str(socket_path),
    }))
    stderr = tmp_path / "rust-owner-stderr.log"
    with stderr.open("wb") as errors:
        process = subprocess.Popen([str(machine_host_owner_binary), str(config)],
                                   stdout=subprocess.PIPE, stderr=errors, text=True)
        assert process.stdout.readline().strip() == "ready"
        channel = connect(str(socket_path))
        executor = Executor(channel, tmp_path / "executor")
        provider = BrokeredModelSources(executor._durable)
        consumer = native_plane.Plane(devices=[], readers=2, direct_io=False)
        try:
            checkpoint = Checkpoint(Path(""), manifest, source_provider=provider)
            traversal = (("linear", "weight"),)
            regions = (("linear/weight",),)
            value = HostTierPlan(manifest=manifest, name="generation/linear",
                                 layout=canonical.digest(
                                     {"manifest": manifest, "regions": regions}),
                                 traversal=traversal, components=("linear",), regions=regions)
            executor._prepare_tier(value)
            plan = checkpoint.read_plan(traversal, 4 << 20, ("linear",))
            source = provider.source(consumer, manifest, plan)
            fd = executor._ask_tier(value.name, value.layout)
            assert fd is not None
            try:
                ws = consumer.register(value.name, source, plan, regions, host_fd=fd)
            finally:
                os.close(fd)
            consumer.set_pinned_budget(ws.nbytes)
            consumer.want(ws, "pinned").wait()
            with mmap.mmap(ws.host_fd, ws.nbytes, access=mmap.ACCESS_READ) as mapping:
                part = ws.parts[0]
                assert mapping[part["offset"]:part["offset"] + part["nbytes"]] == data
            # Native recipient claims Ready backing; it did not copy from its descriptor
            # source or allocate/fill a second private host data pool.
            stats = consumer.stats()
            assert stats["host"]["fills"] == 0
            assert stats["host"]["buffered_bytes"] == 0
            assert "libcuda.so" not in Path("/proc/self/maps").read_text()
            assert "libnvidia-ml" not in Path("/proc/self/maps").read_text()
            ws.close()
            del source
            provider.close()
            consumer.close()
            channel.close()
            assert process.wait() == 0
            facts = json.loads(stderr.read_text().strip().splitlines()[-1])
            assert facts["filled_bytes"] == len(data)
            assert facts["backing_bytes"] > 0 and facts["recipients"] == 1
        finally:
            channel.close()
            if process.poll() is None:
                process.terminate()  # explicit owned CPU fixture cleanup; no GPU/process policy
                process.wait()
            lease.release()
