"""Copied into the proof-only SDXL package; never installed in production SDXL."""

import json
import os
import signal
import sys
from pathlib import Path


def host_snapshot():
    """Read only. No limits, cgroup membership, process signals or allocations change."""
    try:
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        relative = next(
            line.removeprefix("0::")
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        root = Path("/sys/fs/cgroup").resolve()
        group = (root / relative.lstrip("/")).resolve()
        if not group.is_relative_to(root):
            raise ValueError("cgroup namespace does not resolve inside its mount")
        groups = []
        while True:
            events = {
                key: int(value)
                for key, value in (
                    line.split() for line in (group / "memory.events").read_text().splitlines()
                )
            }
            groups.append({"path": str(group.relative_to(root)), "events": events})
            if group == root:
                break
            group = group.parent
        return {"available": True, "mem_available_bytes": available, "cgroups": groups}
    except (OSError, ValueError, StopIteration) as error:
        return {"available": False, "reason": str(error)}


def after_step(index, kill, case, device, telemetry, complete=False):
    if index != 0 and not complete:
        return
    import torch

    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise RuntimeError("Lifecycle proof requires exactly one assigned CUDA device")
    torch.cuda.synchronize(device)
    event = {
        "pid": os.getpid(),
        "uuid": str(torch.cuda.get_device_properties(device).uuid),
        "step": index,
        "cuda_completed": True,
        "kill": bool(kill),
        "case": case,
        "complete": bool(complete),
        "host": host_snapshot(),
    }
    line = "SLIM_LIFECYCLE " + json.dumps(event, sort_keys=True)
    telemetry.log(line, level="info")
    print(line, file=sys.stderr, flush=True)
    if kill:
        os.kill(os.getpid(), signal.SIGKILL)
