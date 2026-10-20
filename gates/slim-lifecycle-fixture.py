"""Copied into the proof-only SDXL package; never installed in production SDXL."""

import json
import os
import signal
import sys


def after_step(index, kill, case, device, telemetry):
    if index != 0:
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
    }
    line = "SLIM_LIFECYCLE " + json.dumps(event, sort_keys=True)
    telemetry.log(line, level="info")
    print(line, file=sys.stderr, flush=True)
    if kill:
        os.kill(os.getpid(), signal.SIGKILL)
