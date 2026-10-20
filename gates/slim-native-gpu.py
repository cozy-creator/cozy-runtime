# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4,<0.31"]
# ///
"""Run the portable native tests through `cozy run --on RENTAL --gpus 1`.

The outer Runtime launches a standalone Rust binary; it does not replace the worker's SDK.
The binary checks its actual visible GPU and runs each case in a fresh process.
The required suite JSON chooses ten existing cases or adds the new eleventh case:
{"warm_admission": false|true}. The eleventh needs the updated placement binary;
coverage is explicit in result.json.
"""
from pathlib import Path
import json
import os
import re
import signal
import subprocess
import tempfile
from typing import Annotated

from cozy_runtime.author import AssetBound, FileAsset, Outputs, Tree

CASES = {
    "pages_pin_evict_refill_and_survive_clean_owner_exit": "PASS independent pages, pin protection, RAM refill and kept-page import",
    "escaped_writable_view_retains_pages_and_recovers": "PASS escaped writable retention and same-process recovery",
    "read_only_fault_is_contained_in_its_process": "PASS read-only VMM protection and healthy replacement",
    "cuda_oom_is_recovered_in_process": "PASS real CUDA OOM recovery",
    "killed_owner_keeps_read_only_pages": "PASS SIGKILL read-only page custody",
    "retired_page_releases_custody_after_last_view": "PASS retired custody closes after last view completion",
}


PLACEMENT_CASES = {
    "placement::gpu_tests::failed_begin_rolls_back_and_reuses_the_request_identity":
        "PASS failed begin rolls back and reuses its request",
    "placement::gpu_tests::failed_direct_ram_fill_retains_its_source_until_checked_recovery":
        "PASS failed RAM fill retains its source through recovery",
    "placement::gpu_tests::ram_warm_returns_before_export_and_close_prevents_stale_registration":
        "PASS RAM warming stays asynchronous and closes without stale registration",
    "placement::gpu_tests::idle_worker_retires_after_pending_consumer_finishes":
        "PASS idle worker retires after pending consumer",
}


def main(*, binary: Annotated[FileAsset, AssetBound(max_bytes=67108864)],
         placement_binary: Annotated[FileAsset, AssetBound(max_bytes=67108864)],
         suite: Annotated[FileAsset, AssetBound(max_bytes=4096)], out: Outputs) -> Tree:
    selection = json.loads(suite.read_bytes())
    if (not isinstance(selection, dict) or set(selection) != {"warm_admission"}
            or type(selection["warm_admission"]) is not bool):
        raise ValueError('suite must be exactly {"warm_admission": true|false}')
    warm_admission = selection["warm_admission"]
    root = Path(tempfile.mkdtemp(prefix="varena-slim-gpu-"))
    executable = root / "slim-gpu"
    executable.write_bytes(binary.read_bytes())
    executable.chmod(0o700)
    placement_executable = root / "slim-placement"
    placement_executable.write_bytes(placement_binary.read_bytes())
    placement_executable.chmod(0o700)
    logs = root / "logs"
    logs.mkdir()
    results = []
    gpu_uuids = set()
    cases = [(executable, name, marker) for name, marker in CASES.items()]
    cases += [(placement_executable, name, marker) for name, marker in PLACEMENT_CASES.items()]
    if warm_admission:
        cases.append((
            placement_executable,
            "placement::gpu_tests::idle_cached_weight_pins_cannot_block_their_own_warm_admission",
            "PASS idle cache pins release before warm admission without weight eviction",
        ))
    for program, name, marker in cases:
        command = [str(program), "--ignored", "--exact", name, "--nocapture", "--test-threads=1"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            print(stdout + stderr, flush=True)
            raise RuntimeError(f"Native GPU case {name} exceeded its test-harness bound") from None
        text = stdout + stderr
        (logs / f"{name.replace('::', '_')}.log").write_text(text)
        print(text, flush=True)
        summary = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
        if process.returncode or marker not in text or not summary or summary[-1] != ("1", "0", "0"):
            raise RuntimeError(f"Native GPU case {name} failed (exit {process.returncode})")
        identities = [line.split(marker, 1)[1].split()[0]
                      for line in text.splitlines() if marker in line]
        if not identities:
            raise RuntimeError(f"Native GPU case {name} did not report its physical GPU")
        gpu_uuids.update(identities)
        if len(gpu_uuids) != 1:
            raise RuntimeError(f"Native GPU cases used different physical GPUs: {sorted(gpu_uuids)}")
        results.append(name)
    (logs / "result.json").write_text(json.dumps({"passed": results, "warm_admission_requested": warm_admission, "gpu_uuid": next(iter(gpu_uuids)), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, indent=2))
    return out.save_tree(logs)
