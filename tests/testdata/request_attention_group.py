"""Request attention selection across a real degree-2 gloo group, over the real seam (cr-191).

The worker's stand-in and the CPU executor program are `multi_construction`'s. One
construction serves every request; the pin is a per-request execution parameter.
Prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
import multi_construction as mc  # type: ignore[import-not-found]

from cozy_runtime.author import Config

DEVICES = "cpu0,cpu1"


def ranks(execution: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """rank -> requested, observed and dispatched attention with its probes, and the rank's
    device arch, from the execution record."""
    return {row["rank"]: {**row["attention"], "arch": row["arch"]} for row in execution["ranks"]}


def _serve(driver: Any, root: Path, key: str, name: str, pin: str) -> dict[str, Any]:
    request = {"request_id": name, "entrypoint": "generate", "payload": {"value": 3}}
    prepared = driver.call(
        {"cmd": "prepare_request", "construction": key, "attention_kernel": pin, **request}
    )
    if not prepared["ok"]:
        return prepared
    spool = root / "spool" / name
    spool.mkdir(parents=True)
    served = driver.call(
        {
            "cmd": "invoke",
            "construction": key,
            "attention_kernel": pin,
            "deadline_s": None,
            "spool": str(spool),
            "placement": "all_resident",
            **request,
        }
    )
    assert served["ok"] and served["outcome"]["terminal"] == "succeeded", served
    assert not served["poisoned"], served
    return served


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    root = Path(parser.parse_args().root)
    store = root / "store"
    torch.manual_seed(231)
    mc.native.native_model(
        store, "base", mc.BasePipe(Config({"dit": mc.CONFIG})).components, {"dit": mc.CONFIG}
    )
    mc.native.native_model(store, "overlay", mc.OverlayPipe(Config({})).components, {})
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES
    overrides = {"generate.models.base": "test/base@1", "generate.models.overlay": "test/overlay@1"}
    binding = mc._stage(root, mc.DUAL, "A", overrides, store)
    key = binding.construction_key()
    driver = mc.Driver(root / "run", "cpu", DEVICES)
    pins = {"base": "", "pinned": "base/dit=sdpa", "base-again": ""}
    facts: dict[str, Any] = {"attention": {}, "pids": []}
    try:
        assert mc._start(driver, binding, 2, DEVICES)["ok"]
        assert driver.call(mc._load_command(binding, key, 2, "cpu", 64 << 30))["ok"]
        assert driver.call({"cmd": "activate", "construction": key})["ok"]
        facts["pids"].append(mc._pids(driver))
        for name in ("base", "unsupported", "pinned", "base-again"):
            reply = _serve(driver, root, key, name, pins.get(name, "base/dit=flash-attn3"))
            if not reply["ok"]:
                assert reply.get("origin") == "request", reply
                facts["refused"] = reply["code"]
                continue
            facts["attention"][name] = ranks(reply["execution"])
            facts["pids"].append(mc._pids(driver))
    finally:
        driver.close()
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
