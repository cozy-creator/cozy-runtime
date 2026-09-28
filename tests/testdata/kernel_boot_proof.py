"""SageAttention compiled on THIS card at boot, through the compile manager, from pinned source.

Run as a child by `tests/test_kernel_boot.py` with a store directory and the kernel-source
registry. Nothing is faked except a builder's word in the first arm (a live lock and its
progress file, what a running builder leaves): selection falls through past a kernel still
compiling. Then the real builder compiles upstream's `setup.py` for this card, the next
selection takes the kernel, and a real site runs a forward through it. Prints one JSON line.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

from cozy_runtime.internal import attention, kernel_cache, kernel_compile, kernel_sources
from cozy_runtime.internal.encoding import DeviceFacts

PREFER = ("sageattention", "sdpa")


def main() -> None:
    store_dir, sources = Path(sys.argv[1]), Path(sys.argv[2])
    kernel_sources.ROOT = sources
    store = kernel_cache.Store(store_dir)
    kernel_cache.configure(store)
    major, minor = torch.cuda.get_device_capability()
    device = DeviceFacts("cuda", torch.cuda.get_device_name(0), major * 10 + minor, "", "", 0)
    site = WanAttention(dim=1024, heads=8, dim_head=128, processor=WanAttnProcessor())
    roots = {"dit": site.to("cuda", torch.bfloat16)}
    out: dict[str, Any] = {"sm": device.sm}
    job = kernel_sources.job("sageattention", device.sm)

    # A builder's word: its lock held and its progress written. Selection serves the next
    # ready kernel, a pin refuses with the progress, and the run evidence says both.
    lock = store.claim(job.key)
    assert lock is not None
    store.set_state(job.key, {"pid": os.getpid(), "progress": 0.42, "started_unix_ms": 1})
    compiling = attention.select(device, roots, choose=lambda _: PREFER)
    out["while_compiling"] = {"hosts": compiling.hosts, "skipped": compiling.skipped}
    try:
        attention.pinned("sageattention", device)
    except attention.AttentionRefusal as exc:
        out["pin_while_compiling"] = {"code": exc.code, "detail": str(exc)}
    out["evidence_while_compiling"] = attention.evidence(compiling.totals())
    out["pending_while_compiling"] = attention.pending(compiling, device)
    store.set_state(job.key, None)
    os.close(lock)

    # The real compile, in the background process the startup warm starts.
    started = time.monotonic()
    out["warm"] = attention.expect(device, list(PREFER))
    progress: list[float] = []
    while (state := kernel_compile.status(store, job)).state in ("compiling", "pending"):
        if state.progress is not None and (not progress or state.progress != progress[-1]):
            progress.append(state.progress)
        time.sleep(1)
    out["compile"] = {
        "state": state.state,
        "detail": state.detail,
        "wall_s": round(time.monotonic() - started, 1),
        "compile_ms": state.ms,
        "progress": progress,
    }
    if state.state != "ready":
        print(json.dumps(out))
        return

    # A request boundary: what the construction skipped while compiling is ready now.
    out["pending_after"] = attention.pending(compiling, device)
    upgraded = attention.select(device, roots, choose=lambda _: PREFER)
    out["after"] = {"hosts": upgraded.hosts, "artifacts": upgraded.artifacts}
    hidden = torch.randn(1, 4096, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        served = site(hidden)
        attention.select(device, roots)
        floor = site(hidden)
    out["forward"] = {
        "shape": list(served.shape),
        "finite": bool(torch.isfinite(served).all()),
        "rel_l2_to_sdpa": float((served.float() - floor.float()).norm() / floor.float().norm()),
    }
    out["evidence_after"] = attention.evidence(upgraded.totals())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
