"""Real CPU native fill, staged warm, then five resident executor requests.

Uses the existing subprocess fixture's host-memory accounting and CPU byte ring.
This proves lifecycle and actual tensors, not GPU performance or CUDA capacity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import multi_construction as native  # type: ignore[import-not-found]
import torch

from cozy_runtime.author import Config
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import PlanChooser, PreparedModel, PreparedRequest

PACKAGE = Path(__file__).parent / "warm_restore_package"
sys.path.insert(0, str(PACKAGE))
from warm_restore_package import Pipeline  # type: ignore[import-not-found]  # noqa: E402


def main(root: Path) -> None:
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(53)
    model = Pipeline(Config({}))
    native.native.native_model(root / "store", "warm-restore", model.components, {})
    binding = native._stage(root, PACKAGE, "warm", {}, root / "store")
    driver = native.Driver(root / "run", "cpu", "")
    try:
        started = native._start(driver, binding, 1, "")
        assert started["ok"], started
        loaded = driver.call(native._load_command(binding, "warm", 1, "cpu", 64 << 30))
        assert loaded["ok"], loaded
        facts = loaded["facts"]
        assert facts["parked"] == [], facts
        assert set(facts["constructed"]) == {"encoder", "decoder"}, facts
        assert set(facts["resident"]) == {"decoder"}, facts
        ledger = Ledger()
        ledger.observe_construction(facts)
        assert set(ledger.missing_resident()) == {"encoder"}
        assert driver.call({"cmd": "activate", "construction": "warm"})["ok"]
        restored = driver.call({"cmd": "restore", "names": sorted(ledger.missing_resident())})
        assert restored["ok"], restored
        ledger.observe_restore(restored, "warm completion")
        assert not ledger.missing_resident()
        pids = native._pids(driver)
        counters = []
        for index in range(5):
            request = f"request-{index}"
            prepared = driver.call(
                {
                    "cmd": "prepare_request",
                    "construction": "warm",
                    "request_id": request,
                    "entrypoint": "generate",
                    "payload": {"value": index + 1},
                }
            )
            assert prepared["ok"], prepared
            plan = PlanChooser(ledger).choose(
                binding,
                PreparedModel.from_reply(loaded),
                PreparedRequest.unresolved("generate", {}),
            )
            assert plan.placement == "all_resident", plan.document()
            spool = root / "spool" / request
            spool.mkdir(parents=True)
            reply = driver.call(
                {
                    "cmd": "invoke",
                    "construction": "warm",
                    "request_id": request,
                    "entrypoint": "generate",
                    "deadline_s": None,
                    "spool": str(spool),
                    "placement": plan.placement,
                }
            )
            assert reply["ok"] and reply["outcome"]["terminal"] == "succeeded", reply
            residency = reply["residency"]
            assert residency["attempt_stages"] == residency["attempt_evictions"] == 0, residency
            with torch.no_grad():
                value = model.components["encoder"](torch.full((1, 16), float(index + 1)))
                expected = float(model.components["decoder"](value).sum())
            actual = json.loads((spool / RESULT_DOCUMENT).read_bytes())["checksum"]
            assert actual == expected, (actual, expected)
            counters.append(
                {"stages": residency["attempt_stages"], "evictions": residency["attempt_evictions"]}
            )
        assert native._pids(driver) == pids
        print(json.dumps({"backend": "cpu", "pids": pids, "requests": counters}), flush=True)
    finally:
        driver.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
