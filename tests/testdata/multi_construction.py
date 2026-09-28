"""Several constructions in ONE executor process, driven over the real seam (proto-061 D1).

The driver is the worker's stand-in: it binds the control socket, starts the executor
program as a subprocess, and speaks the executor's own command vocabulary to it. On CPU the
executor and its follower ranks run under the same host-RAM accounting and synchronous ring
the native multi-model fixture uses; on CUDA nothing is adapted at all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))  # followers run isolated (`python -I`)
import native_parallel_models as native  # type: ignore[import-not-found]

from cozy_runtime.author import Config
from cozy_runtime.internal import accel, package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import main as executor_main
from cozy_runtime.internal.local import stage_development_placement
from cozy_runtime.internal.parallel import group as group_module
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel, listener
from cozy_runtime.internal.worker.session import executor_load_command

TESTDATA = Path(__file__).parent
DUAL = TESTDATA / "dual_model_package"
BLOCK = TESTDATA / "block_package"
sys.path[:0] = [str(DUAL), str(BLOCK)]
from block_package import BlockPipe  # type: ignore[import-not-found]  # noqa: E402
from dual_model_package import (  # type: ignore[import-not-found]  # noqa: E402
    CONFIG,
    BasePipe,
    OverlayPipe,
    batch,
)

MIB = 1 << 20


def _follower_argv(*, rank: int, world: int, rank_fd: int, root: str) -> list[str]:
    return [
        str(Path(__file__)),
        "--serve",
        "cpu",
        "--root",
        root,
        "--rank",
        str(rank),
        "--world",
        str(world),
        "--rank-fd",
        str(rank_fd),
        "--leader",
        str(os.getpid()),
    ]


def serve(backend: str, argv: list[str]) -> None:
    """The executor program itself; on CPU its accounting and ring are host-memory ones."""
    if backend == "cpu":
        setattr(accel, "host_backend_family", lambda: "cpu")  # noqa: B010
        native.install_cpu_transport()
        native.cpu_accounting()
        setattr(group_module, "executor_argv", _follower_argv)  # noqa: B010
    torch.set_num_threads(1)
    raise SystemExit(executor_main(argv))


class Driver:
    """The worker's side of the seam, reduced to what these arms need."""

    def __init__(self, root: Path, backend: str, devices: str, *, scope: str = "") -> None:
        root.mkdir(parents=True, exist_ok=True)
        path = str(root / "seam.sock")
        sock = listener(path)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": devices,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONFAULTHANDLER": "1",
        }
        # A named transient systemd scope puts the executor in its own memory cgroup.
        prefix = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={scope}"] if scope else []
        self.process = subprocess.Popen(
            [
                *prefix,
                sys.executable,
                str(Path(__file__)),
                "--serve",
                backend,
                "--socket",
                path,
                "--root",
                str(root / "executor"),
            ],
            env=env,
        )
        sock.settimeout(120)
        connection, _ = sock.accept()
        sock.close()
        self.channel = Channel(connection)

    def call(self, command: dict[str, Any]) -> dict[str, Any]:
        self.channel.send(command)
        while True:
            frame = self.channel.recv(timeout=None)
            assert frame is not None, f"the executor died during {command['cmd']!r}"
            if frame.get("event") == "progress":
                continue
            if frame.get("event") == "request":
                # No co-tenant to vacate: answered as a worker with no handler answers.
                self.channel.send({"event": "answer", "seq": frame["seq"], "ok": False})
                continue
            assert frame.get("reply") == command["cmd"], frame
            return frame

    def close(self) -> None:
        try:
            self.channel.send({"cmd": "shutdown"})
            self.process.wait(timeout=60)
        finally:
            if self.process.poll() is None:
                self.process.kill()
            self.channel.close()


def _load_command(
    binding: Any, key: str, degree: int, backend: str, authorized: int
) -> dict[str, Any]:
    command = executor_load_command(
        binding,
        construction=key,
        devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        sequence_parallel_degree=degree,
        authorized_device_limit_bytes=authorized,
    )
    rows = command.get("models") or [command]
    for row in rows:
        row["binding"].update(window_bytes=4 * MIB, slots=4, readers=2, inflight=2)
        if backend == "cpu":
            row["binding"]["variant"] = "cpu"
    return command


def _stage(root: Path, project: Path, name: str, overrides: dict[str, str], store: Path) -> Any:
    interface = package_interface.build(discover(project))
    staged = stage_development_placement(
        root / f"home-{name}",
        interface,
        package=f"test/{project.name.replace('_', '-')}",
        release="0.0.1",
        installation_id="local-" + uuid.uuid4().hex,
        artifact_cache=root / f"cache-{name}",
        tensorfs_root=store,
        project=project,
        binding_overrides=overrides,
    )
    return next(iter(staged.declared_bindings.values()))


def _start(driver: Driver, binding: Any, degree: int, devices: str) -> dict[str, Any]:
    return driver.call(
        {
            "cmd": "start",
            "devices": devices,
            "sequence_parallel_degree": degree,
            "application": binding.application,
            "package_interface": binding.interface_path,
            "development_project": binding.development_project,
        }
    )


def _serve(driver: Driver, root: Path, key: str, request: str) -> dict[str, Any]:
    prepared = driver.call(
        {
            "cmd": "prepare_request",
            "construction": key,
            "request_id": request,
            "entrypoint": "generate",
            "payload": {"value": 3},
        }
    )
    assert prepared["ok"], prepared
    spool = root / "spool" / request
    spool.mkdir(parents=True)
    served = driver.call(
        {
            "cmd": "invoke",
            "construction": key,
            "request_id": request,
            "entrypoint": "generate",
            "deadline_s": None,
            "spool": str(spool),
            "placement": "all_resident",
        }
    )
    assert served["ok"] and served["outcome"]["terminal"] == "succeeded", served
    assert not served["poisoned"], served
    result = json.loads((spool / RESULT_DOCUMENT).read_bytes())
    return {"checksum": result["checksum"], "metrics": served["metrics"]}


def _states(driver: Driver) -> dict[str, str]:
    reply = driver.call({"cmd": "residency"})
    assert reply["ok"], reply
    return {row["key"]: row["state"] for row in reply["constructions"]}


def _pids(driver: Driver) -> list[int]:
    probe = driver.call({"cmd": "probe"})
    return [row["pid"] for row in probe["processes"]]


def two_models(root: Path, degree: int) -> dict[str, Any]:
    """Two many-model constructions (base or base2, plus overlay) in one CPU process."""
    store = root / "store"
    torch.manual_seed(231)
    base = BasePipe(Config({"dit": CONFIG}))
    torch.manual_seed(17)
    other = BasePipe(Config({"dit": CONFIG}))
    overlay = OverlayPipe(Config({}))
    with torch.no_grad():
        overlay.components["projection"].weight.copy_(torch.tensor([[1.0, 0.25], [-0.5, 1.25]]))
    native.native_model(store, "base", base.components, {"dit": CONFIG}, encoded=True)
    native.native_model(store, "base2", other.components, {"dit": CONFIG})
    native.native_model(store, "overlay", overlay.components, {})
    projection = {"projection": overlay.components["projection"]}
    with torch.no_grad():
        expected = {
            key: sum(
                float(value.sum())
                for value in pipe.components["dit"](**batch(), attention_kwargs=projection)
            )
            for key, pipe in (("A", base), ("B", other))
        }
    devices = ",".join(f"cpu{rank}" for rank in range(degree))
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    bindings = {
        key: _stage(
            root,
            DUAL,
            key,
            {
                "generate.models.base": f"test/{model}@1",
                "generate.models.overlay": "test/overlay@1",
            },
            store,
        )
        for key, model in (("A", "base"), ("B", "base2"))
    }
    driver = Driver(root / "run", "cpu", devices)
    facts: dict[str, Any] = {}
    try:
        pid = driver.call({"cmd": "hello"})["pid"]
        refused = driver.call(_load_command(bindings["A"], "A", degree, "cpu", 64 << 30))
        assert refused["code"] == "executor_not_started", refused
        started = _start(driver, bindings["A"], degree, devices)
        assert started["ok"] and started["torch"], started
        assert len(started["follower_pids"]) == degree - 1, started
        names = [name for name, _ in started["stages"]]
        assert "package_import" in names and "runtime_imports" in names, started
        for rank_stages in started["follower_stages"]:
            assert "package_import" in [name for name, _ in rank_stages], started
        if degree > 1:
            assert "group_form" in names, started
        ranks = _pids(driver)
        assert ranks[0] == pid and len(ranks) == degree
        loads = {}
        for key in ("A", "B"):
            loads[key] = driver.call(_load_command(bindings[key], key, degree, "cpu", 64 << 30))
            assert loads[key]["ok"] and not loads[key]["reused"], loads[key]
            assert loads[key]["construction"] == key and loads[key]["resident_bytes"] > 0
            assert loads[key]["parked"] == [], loads[key]
        again = driver.call(_load_command(bindings["A"], "A", degree, "cpu", 64 << 30))
        assert again["ok"] and again["reused"], again
        assert again["constructed_model_digest"] == loads["A"]["constructed_model_digest"]
        assert loads["A"]["constructed_model_digest"] != loads["B"]["constructed_model_digest"]
        assert _states(driver) == {"A": "resident", "B": "resident"}
        served: dict[str, Any] = {}
        for key in ("A", "B", "A"):
            activated = driver.call({"cmd": "activate", "construction": key})
            assert activated["ok"] and activated["parked"] == [], activated
            served[key] = _serve(driver, root, key, f"{key}-{len(served)}-{time.monotonic_ns()}")
            assert math.isclose(
                served[key]["checksum"], expected[key], rel_tol=2e-5, abs_tol=2e-6
            ), (key, served[key], expected)
            metrics = served[key]["metrics"]
            assert isinstance(metrics["shape_cell"], str) and metrics["shape_cell"]
            assert metrics["working_peak_vram_bytes"] >= 0
        inactive = driver.call(
            {
                "cmd": "invoke",
                "construction": "B",
                "request_id": "inactive",
                "entrypoint": "generate",
                "deadline_s": None,
                "spool": str(root),
            }
        )
        assert inactive["code"] == "construction_inactive", inactive
        unloaded = driver.call({"cmd": "unload", "construction": "A"})
        assert unloaded["ok"] and unloaded["construction"] == "A", unloaded
        assert _states(driver) == {"B": "resident"}
        gone = driver.call(
            {
                "cmd": "prepare_request",
                "construction": "A",
                "request_id": "gone",
                "entrypoint": "generate",
                "payload": {},
            }
        )
        assert gone["code"] == "construction_not_loaded", gone
        reloaded = driver.call(_load_command(bindings["A"], "A", degree, "cpu", 64 << 30))
        assert reloaded["ok"] and not reloaded["reused"], reloaded
        assert driver.call({"cmd": "activate", "construction": "A"})["ok"]
        again_a = _serve(driver, root, "A", "A-reloaded")
        assert math.isclose(again_a["checksum"], expected["A"], rel_tol=2e-5, abs_tol=2e-6)
        assert _states(driver) == {"A": "resident", "B": "resident"}
        assert _pids(driver) == ranks, "every construction ran in the same processes"
        facts = {"pids": ranks, "checksums": {k: v["checksum"] for k, v in served.items()}}
    finally:
        driver.close()
    return facts


def sized_blocks(root: Path, size: int) -> dict[str, Any]:
    """Two single-block constructions on one CUDA card under a ceiling that holds one."""
    store = root / "store"
    pipes = {}
    for key, seed in (("A", 5), ("B", 6)):
        torch.manual_seed(seed)
        pipes[key] = BlockPipe(Config({"block": {"size": size}}))
        native.native_model(
            store, f"block-{key.lower()}", pipes[key].components, {"block": {"size": size}}
        )
    expected = {
        key: float(pipe.components["block"](torch.ones(1, size)).sum().detach())
        for key, pipe in pipes.items()
    }
    block_bytes = size * size * 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    bindings = {
        key: _stage(
            root, BLOCK, key, {"generate.models.block": f"test/block-{key.lower()}@1"}, store
        )
        for key in ("A", "B")
    }
    driver = Driver(root / "run", "cuda", "0")
    try:
        pid = driver.call({"cmd": "hello"})["pid"]
        started = _start(driver, bindings["A"], 1, "0")
        assert started["ok"] and started["device_total_bytes"] > 0, started
        first = driver.call(_load_command(bindings["A"], "A", 1, "cuda", 4 << 30))
        assert first["ok"] and first["resident_bytes"] == block_bytes, first
        held = int(first["facts"]["allocator_bytes"])
        # A ceiling that holds either block and not both: loading B parks A whole.
        ceiling = held + block_bytes // 2
        second = driver.call(_load_command(bindings["B"], "B", 1, "cuda", ceiling))
        assert second["ok"] and second["parked"] == ["A"], second
        assert _states(driver) == {"A": "parked", "B": "resident"}
        timings = []
        for key, other in (("A", "B"), ("B", "A"), ("A", "B")):
            activated = driver.call({"cmd": "activate", "construction": key})
            assert activated["ok"] and activated["parked"] == [other], activated
            assert activated["restored_bytes"] == block_bytes, activated
            # From a warm page cache a restore is a copy: bytes / 10 GB/s + 1 s. A read the
            # block layer served (the shared box evicted the pages) is a disk read instead.
            if 0 <= activated["read_bytes"] <= block_bytes // 2:
                budget_ms = block_bytes / 10e9 * 1000 + 1000
                assert activated["ms"] <= budget_ms, (activated, budget_ms)
            timings.append(activated["ms"])
            assert _states(driver) == {key: "resident", other: "parked"}
            served = _serve(driver, root, key, f"{key}-{len(timings)}")
            assert math.isclose(served["checksum"], expected[key], rel_tol=1e-3), served
            metrics = served["metrics"]
            # Working memory excludes the resident weights: the block was allocated at entry.
            assert metrics["allocated_at_start_bytes"] >= block_bytes, metrics
            assert 0 < metrics["working_peak_vram_bytes"] < 64 * MIB, metrics
        again = driver.call({"cmd": "activate", "construction": "A"})
        assert again["ok"] and again["parked"] == [] and again["restored_bytes"] == 0, again
        unloaded = driver.call({"cmd": "unload", "construction": "A"})
        assert unloaded["ok"] and unloaded["freed_bytes"] >= block_bytes, unloaded
        assert _states(driver) == {"B": "parked"}
        assert driver.call({"cmd": "hello"})["pid"] == pid
        return {"activate_ms": timings, "block_bytes": block_bytes}
    finally:
        driver.close()


def refused_loads(root: Path) -> dict[str, Any]:
    """Loads refused before they hold device bytes leave the process and its models serving."""
    store = root / "store"
    torch.manual_seed(5)
    pipe = BlockPipe(Config({"block": {"size": 8}}))
    native.native_model(store, "block-a", pipe.components, {"block": {"size": 8}})
    # The checkpoint's inline config asks for 16x16 while its tensor is 8x8: a fit refusal.
    native.native_model(store, "block-bad", pipe.components, {"block": {"size": 16}})
    expected = float(pipe.components["block"](torch.ones(1, 8)).sum().detach())
    os.environ["CUDA_VISIBLE_DEVICES"] = "cpu0"
    bindings = {
        key: _stage(root, BLOCK, key, {"generate.models.block": f"test/{model}@1"}, store)
        for key, model in (("A", "block-a"), ("B", "block-bad"), ("C", "block-a"))
    }
    driver = Driver(root / "run", "cpu", "cpu0")
    try:
        pid = driver.call({"cmd": "hello"})["pid"]
        assert _start(driver, bindings["A"], 1, "cpu0")["ok"]
        assert driver.call(_load_command(bindings["A"], "A", 1, "cpu", 64 << 30))["ok"]
        misfit = driver.call(_load_command(bindings["B"], "B", 1, "cpu", 64 << 30))
        assert not misfit["ok"] and misfit["code"] == "model_fit_refused", misfit
        assert misfit["poisoned"] == "", misfit
        starved = driver.call(_load_command(bindings["C"], "C", 1, "cpu", 1))
        assert starved["code"] == "device_shortfall" and starved["poisoned"] == "", starved
        assert set(_states(driver)) == {"A"}
        activated = driver.call(
            {"cmd": "activate", "construction": "A", "authorized_device_limit_bytes": 64 << 30}
        )
        assert activated["ok"], activated
        served = _serve(driver, root, "A", "after-refusals")
        assert math.isclose(served["checksum"], expected, rel_tol=1e-5), served
        assert driver.call({"cmd": "hello"})["pid"] == pid
        return {"codes": [misfit["code"], starved["code"]]}
    finally:
        driver.close()


def _executor_cgroup(pid: int) -> Path:
    line = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[0]
    return Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")


def host_budget(root: Path, size: int) -> dict[str, Any]:
    """Fill a checkpoint with host memory capped at what is held plus 1.5x the checkpoint.

    On CPU the destinations ARE host memory, so the cap is the model plus half of it: a
    fill that also held a whole host image of the checkpoint would exceed it and be
    OOM-killed. Page cache from the store reads is charged too and must be reclaimed.
    """
    store = root / "store"
    torch.manual_seed(7)
    pipe = BlockPipe(Config({"block": {"size": size}}))
    native.native_model(store, "block-big", pipe.components, {"block": {"size": size}})
    expected = float(pipe.components["block"](torch.ones(1, size)).sum().detach())
    checkpoint = size * size * 4
    del pipe
    os.environ["CUDA_VISIBLE_DEVICES"] = "cpu0"
    binding = _stage(root, BLOCK, "A", {"generate.models.block": "test/block-big@1"}, store)
    driver = Driver(root / "run", "cpu", "cpu0", scope=f"cozy-host-budget-{os.getpid()}")
    try:
        pid = driver.call({"cmd": "hello"})["pid"]
        assert _start(driver, binding, 1, "cpu0")["ok"]
        cgroup = _executor_cgroup(pid)
        held = int((cgroup / "memory.current").read_text())
        limit = held + checkpoint + checkpoint // 2
        (cgroup / "memory.swap.max").write_text("0")
        (cgroup / "memory.max").write_text(str(limit))
        loaded = driver.call(_load_command(binding, "A", 1, "cpu", 64 << 30))
        assert loaded["ok"], loaded
        assert driver.call({"cmd": "activate", "construction": "A"})["ok"]
        served = _serve(driver, root, "A", "host-budget")
        assert math.isclose(served["checksum"], expected, rel_tol=1e-3), served
        events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
        assert int(events["oom_kill"]) == 0, events
        peak = int((cgroup / "memory.peak").read_text())
        return {"checkpoint": checkpoint, "limit": limit, "peak": peak, "held": held}
    finally:
        driver.close()


def main() -> None:
    if "--serve" in sys.argv:
        index = sys.argv.index("--serve")
        serve(sys.argv[index + 1], sys.argv[1:index] + sys.argv[index + 2 :])
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("two-models", "sized-blocks", "refused-loads", "host-budget"),
        required=True,
    )
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--size", type=int, default=8192)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.arm == "two-models":
        facts = two_models(args.root, args.degree)
    elif args.arm == "refused-loads":
        facts = refused_loads(args.root)
    elif args.arm == "host-budget":
        facts = host_budget(args.root, args.size)
    else:
        facts = sized_blocks(args.root, args.size)
    print(json.dumps(facts), flush=True)


if __name__ == "__main__":
    main()
