"""The machine compiles H3's kernels at worker boot inside the CUDA worker image, and every
executor on it serves them: a newer one from the store, an older one from the image kernel
site (`testdata/kernel_boot_proof.py`, `testdata/kernel_site_old_executor.py`).

The image is `COZY_WORKER_IMAGE` (the cohort's CUDA worker, pulled locally). This checkout's
Runtime runs in it over the image's own Python, torch, toolkit and kernel sources: the machine
as root like a pod's worker, in a container of `CPUS` CPUs it measures itself; executors under
their own uids from package-style venvs over the image's torch, one of them the Runtime H3's
and Qwen's releases lock. `COZY_GPU_LOCK` names a shared box's GPU lock file, held only while
an executor runs attention on the card. Skips without docker, the image or a GPU. Prints each
artifact's compile time and its slowest compiled units.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

IMAGE = os.environ.get("COZY_WORKER_IMAGE", "")
GPU_LOCK = os.environ.get("COZY_GPU_LOCK", "")
ROOT = Path(__file__).resolve().parents[1]
#: H3 1.19.0's lock; the older executor is the Runtime H3 1.19.0 and qwen-image-2 0.2.3 lock.
PACKAGES = ("diffusers==0.40.0", "transformers==5.17.0")
OLD = "0.18.67"
CPUS = 2
EXECUTOR = "setpriv --reuid=64001 --regid=64001 --clear-groups env HOME=/tmp TMPDIR=/tmp"
#: The older executor imports its own release, never this checkout.
OLD_EXECUTOR = EXECUTOR.replace("env ", "env -u PYTHONPATH ")
PROOF = "/work/venv/bin/python /proof/kernel_boot_proof.py"
#: A package venv over the image's own interpreter and packages, wherever the image keeps them.
VENV = (
    "base=$(python3 -c 'import sys; print(sys.executable)') && "
    "uv venv -q --python $base {0} && "
    "$base -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))' "
    "> {0}/lib/python3.12/site-packages/image.pth && "
    "uv pip install -q --python {0}/bin/python {1}"
)


def _skip() -> str:
    if not IMAGE:
        return "COZY_WORKER_IMAGE names no CUDA worker image"
    if shutil.which("docker") is None or shutil.which("nvidia-smi") is None:
        return "no docker or no NVIDIA GPU"
    found = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return "" if found.returncode == 0 else f"{IMAGE} is not pulled"


def _exec(name: str, script: str, *, gpu: bool = False) -> dict[str, Any]:
    command = ["docker", "exec", name, "sh", "-c", script]
    if gpu and GPU_LOCK:
        command = ["flock", GPU_LOCK, *command]
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(done.stdout.splitlines()[-1]) if done.stdout.strip() else {}


def test_the_machine_compiles_h3s_kernels_at_boot_and_every_executor_serves_them(
    tmp_path: Path,
) -> None:
    why = _skip()
    if why:
        pytest.skip(why)
    tmp_path.chmod(0o755)  # executors run under their own uids
    uv_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "uv"
    name = f"cozy-kernel-boot-{os.getpid()}"
    mounts = {
        str(tmp_path): "/work",
        str(uv_cache): "/uv-cache",
        str(ROOT / "src"): "/runtime/src:ro",
        str(ROOT / "tests" / "testdata"): "/proof:ro",
    }
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "--gpus", "all", "--cpus", str(CPUS)]
        + ["-e", "HOME=/work", "-e", "UV_CACHE_DIR=/uv-cache", "-e", "PYTHONPATH=/runtime/src"]
        + [arg for host, inside in mounts.items() for arg in ("-v", f"{host}:{inside}")]
        + ["--entrypoint", "sleep", IMAGE, "7200"],
        check=True,
        capture_output=True,
    )
    try:
        _exec(name, VENV.format("/work/venv", " ".join(PACKAGES)))
        _exec(name, VENV.format("/work/old", " ".join(PACKAGES)))
        _exec(
            name, f"uv pip install -q --no-deps --python /work/old/bin/python cozy-runtime=={OLD}"
        )
        booting = subprocess.Popen(
            ["docker", "exec", name, "sh", "-c", f"exec nice -n 19 {PROOF} machine /work/kernels"],
            stdout=subprocess.PIPE,
            text=True,
        )
        while not (tmp_path / "machine.started").exists():
            assert booting.poll() is None, "the machine's boot ended before it compiled"
            time.sleep(0.2)
        warm = _exec(name, f"{EXECUTOR} {PROOF} executor-warm /work/kernels")
        machine = json.loads(booting.communicate()[0].splitlines()[-1])
        assert booting.returncode == 0
        serve = _exec(name, f"{EXECUTOR} {PROOF} executor-serve /work/kernels", gpu=True)
        old = _exec(
            name,
            f"{PROOF} unpublish /work/kernels && "
            f"({OLD_EXECUTOR} /work/old/bin/python /proof/kernel_site_old_executor.py /tmp/site "
            "> /tmp/old.json &) && while [ ! -e /tmp/site.waiting ]; do sleep 0.2; done && "
            f"{PROOF} publish /work/kernels && touch /tmp/site && "
            "while [ ! -s /tmp/old.json ]; do sleep 0.2; done && cat /tmp/old.json",
            gpu=True,
        )
    finally:
        owner = f"{os.getuid()}:{os.getgid()}"
        subprocess.run(["docker", "exec", name, "chown", "-R", owner, "/work"])
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    summary = {"machine": machine["compiles"], "sol": serve["sol"], "sage": serve["sage"]}
    print(json.dumps({**summary, "old_executor_probes": old["probes"]}, indent=1))

    # The machine compiles H3's ladder best first, as wide as the container, and fills the
    # image kernel site.
    assert machine["ladder"] == machine["h3_ladder"] == ["sol-attn", "sageattention", "flash-attn3"]
    assert machine["order"] == ["kernel-python", "sageattention"] and machine["site"] == ""
    assert machine["width"] == CPUS
    compiles = machine["compiles"]
    for artifact, compiled in compiles.items():
        assert compiled["state"] == "ready" and compiled["compile_ms"] > 0, (artifact, compiled)
    assert compiles["sageattention"]["width"] == CPUS and compiles["sageattention"]["units_ms"]
    assert machine["published"] == ["kernel-python", "sageattention"]
    for entry in ("sageattention", "sageattention-2.2.0.dist-info", "sol_attn", "cutlass"):
        assert entry in machine["site_entries"], machine["site_entries"]

    # A newer executor's warm and construction mid-compile: the machine's builds are its.
    rows = {row["kernel"]: row for row in warm["warm"]}
    assert rows["sageattention"]["state"] == "compiling", rows
    assert rows["flash-attn3"]["status"] == "unsupported" and rows["sdpa"]["status"] == "ready"
    # It serves the best kernel ready then: Sol once its sources are unpacked (seconds), else SDPA.
    served = warm["construction"]["hosts"]["ref2va_dit"]
    assert served in ({"sol-attn": 1}, {"sdpa": 1}), served
    assert warm["pin_while_compiling"]["code"] == "attention_kernel_compiling"
    assert set(warm["own_builders"]) <= {"sol-attn"}, warm  # Sol's card object is its own

    # Later it serves Sol and SageAttention2 from the worker's namespace and builds its own.
    assert serve["h3_hosts"] == {"ref2va_dit": {"sol-attn": 1}}
    assert "fusion" in serve["own_builders"] and set(serve["own_builders"]) <= {
        "fusion",
        "sol-attn",
    }, serve["own_builders"]
    assert serve["sol_counts"]["sparse"] == 1 and serve["sol_site_finite"]
    assert serve["sol"]["rel_l2_to_upstream"] < 0.03, serve["sol"]
    sage = serve["sage"]
    assert sage["hosts"] == {"ref2va_dit": {"sageattention": 1}}
    assert sage["finite"] and sage["rel_l2_to_sdpa"] < 0.05, sage
    assert serve["fusion"]["arch"] == machine["sm"]

    # An older executor started before SageAttention2 landed in the site takes it at its next
    # selection, through its own probe, with no restart and no relock.
    assert old["runtime"] == OLD and old["site"]["path"] == "/opt/cozy/kernels/site", old
    before = {row["kernel"]: row["status"] for row in old["before"]}
    after = {row["kernel"]: row["status"] for row in old["after"]}
    assert before["sageattention"] == "attention_kernel_absent", before
    assert after["sageattention"] == "ready", after
    [probe] = old["probes"]
    assert probe["kernel"] == "sageattention" and probe["status"] == "ready", probe
