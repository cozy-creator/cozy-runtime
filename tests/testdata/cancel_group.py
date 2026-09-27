"""Cancel one attempt mid-step on a real degree-2 gloo group, over the real seam.

The worker's stand-in and the CPU executor program are `multi_construction`'s; the cancel
marker is published exactly as `ExecutorSupervision.cooperative_cancel` publishes it.
Prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
import multi_construction as mc  # type: ignore[import-not-found]

from cozy_runtime.author import Config
from cozy_runtime.internal import execution_evidence

STEPPED = Path(__file__).parent / "stepped_group_package"
sys.path.insert(0, str(STEPPED))
from stepped_group_package import CONFIG, StepPipe  # type: ignore[import-not-found]  # noqa: E402

DEVICES = "cpu0,cpu1"


def _invoke(
    driver: Any, root: Path, key: str, name: str, payload: dict[str, Any], cancel_after: int
) -> dict[str, Any]:
    """One attempt; the marker is published once `cancel_after` steps have reported."""
    request = {"request_id": name, "entrypoint": "generate", "payload": payload}
    assert driver.call({"cmd": "prepare_request", "construction": key, **request})["ok"]
    spool = root / "spool" / name
    spool.mkdir(parents=True)
    driver.channel.send(
        {
            "cmd": "invoke",
            "construction": key,
            "deadline_s": None,
            "spool": str(spool),
            "placement": "all_resident",
            **request,
        }
    )
    marked: dict[str, float] = {}
    while True:
        frame = driver.channel.recv(timeout=None)
        assert frame is not None, "the executor died during the invoke"
        if frame.get("event") != "progress":
            assert frame.get("reply") == "invoke", frame
            return {**frame, "marked": marked, "replied_at": time.monotonic()}
        if frame.get("kind") == "progress" and frame.get("position") == cancel_after:
            marker = root / "run" / "executor" / "executor.cancel"
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(name)
            os.replace(temporary, marker)
            marked = {"at": time.monotonic(), "us": execution_evidence.now_us()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    arguments = parser.parse_args()
    root = Path(arguments.root)
    store = root / "store"
    torch.manual_seed(231)
    mc.native.native_model(
        store, "base", StepPipe(Config({"dit": CONFIG})).components, {"dit": CONFIG}
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES
    binding = mc._stage(root, STEPPED, "A", {"generate.models.base": "test/base@1"}, store)
    key = binding.construction_key()
    driver = mc.Driver(root / "run", "cpu", DEVICES)
    facts: dict[str, Any] = {"pids": []}
    try:
        assert mc._start(driver, binding, 2, DEVICES)["ok"]
        assert driver.call(mc._load_command(binding, key, 2, "cpu", 64 << 30))["ok"]
        assert driver.call({"cmd": "activate", "construction": key})["ok"]
        facts["pids"].append(mc._pids(driver))
        steps = {"steps": 1, "seconds": 0.0}
        for name, payload, cancel_after in (
            ("before", steps, -1),
            ("cancelled", {"steps": 4, "seconds": arguments.seconds}, 1),
            ("after", steps, -1),
        ):
            reply = _invoke(driver, root, key, name, payload, cancel_after)
            facts[name] = {
                "outcome": reply["outcome"],
                "poisoned": reply["poisoned"],
                "ranks": reply["execution"]["ranks"],
                "marked": reply["marked"],
                "replied_at": reply["replied_at"],
                "result": (
                    json.loads((root / "spool" / name / mc.RESULT_DOCUMENT).read_bytes())
                    if reply["outcome"]["terminal"] == "succeeded"
                    else None
                ),
            }
            facts["pids"].append(mc._pids(driver))
    finally:
        driver.close()
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
