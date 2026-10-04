"""Owned allocator diagnostics; no models, SDK edits, global switch or full decode."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import pathlib
import subprocess
import time

MIB = 1024 * 1024
STAT_NAMES = (
    "alloc_calls",
    "free_calls",
    "live_bytes",
    "peak_bytes",
    "largest_request",
    "limit_denials",
    "last_alloc_rc",
    "last_free_rc",
    "invalid_frees",
    "sync_failures",
    "limit",
    "successful_frees",
)


def read_cgroup() -> dict[str, object]:
    root = pathlib.Path("/sys/fs/cgroup")
    result: dict[str, object] = {}
    for name in (
        "memory.max",
        "memory.current",
        "memory.peak",
        "memory.events",
        "memory.stat",
        "memory.pressure",
        "memory.swap.max",
    ):
        p = root / name
        result[name] = p.read_text().strip() if p.exists() else None
    return result


def command(*args: str) -> str:
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


def proc_identity(pid: int) -> dict[str, object]:
    p = pathlib.Path("/proc") / str(pid)
    # Field22 follows the parenthesized process name; do not split that name.
    fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "cmdline": (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--holder-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--cap-bytes", type=int, default=512 * MIB)
    args = parser.parse_args()
    assert args.cap_bytes == 512 * MIB, "Declared probe cap cannot silently change"
    args.out.mkdir(parents=True, exist_ok=False)
    journal = args.out / "journal.jsonl"
    result: dict[str, object] = {
        "proof_level": "GPU_ALLOCATOR_DIAGNOSTIC_ONLY",
        "pid": os.getpid(),
        "cap_bytes": args.cap_bytes,
        "boundary_request_bytes": args.cap_bytes + 2 * MIB,
        "small_tensor_bytes": MIB,
        "bridge_virtual_limit_bytes": 16 * MIB,
        "full_decode": False,
        "sdk_or_physical_grant_changes": False,
        "source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "bridge_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
    }
    bridge = None
    torch = None

    def stats() -> dict[str, int]:
        if bridge is None:
            return {}
        return {name: int(bridge.probe_stat(i)) for i, name in enumerate(STAT_NAMES)}

    def record(stage: str, **details: object) -> dict[str, object]:
        sample: dict[str, object] = {
            "stage": stage,
            "observed_unix_ns": time.time_ns(),
            "cgroup": read_cgroup(),
            "gpu": command(
                "nvidia-smi",
                "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ),
            "cuda_apps": command(
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ),
            "proc_status": pathlib.Path("/proc/self/status").read_text(),
            "bridge": stats(),
            **details,
        }
        if torch is not None and torch.cuda.is_initialized():
            sample["torch"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
            }
        with journal.open("a") as f:
            f.write(json.dumps(sample) + "\n")
            f.flush()
            os.fsync(f.fileno())
        assert sample["gpu"].startswith("GPU-73b71e35-ea6f-21f9-316e-2d86174a91a8,")
        apps = {
            int(line.split(",")[0]): int(line.split(",")[1])
            for line in sample["cuda_apps"].splitlines()
        }
        assert set(apps) <= {113, 2059, os.getpid()}, "Unexpected CUDA app; stop new probes"
        assert apps[113] * MIB >= 6439436288
        assert proc_identity(113)["start_ticks"] == 33024000
        assert proc_identity(2059)["start_ticks"] == 33529715
        assert int(sample["gpu"].split(",")[2]) * MIB <= 1610612736 + 2 * MIB
        return sample

    initial = record("before_torch_import")
    holder = json.loads(args.holder_receipt.read_text())
    assert holder["pid"] == 113 and holder["start_ticks"] == 33024000
    assert holder["held_bytes"] == 6439436288
    assert holder["gpu_uuid"] == "GPU-73b71e35-ea6f-21f9-316e-2d86174a91a8"
    assert proc_identity(113)["start_ticks"] == holder["start_ticks"]
    executor = proc_identity(2059)
    assert executor["start_ticks"] == 33529715
    assert "generations/455e625ff9e342598602a8a388bd4a5b/" in executor["cmdline"]
    assert initial["gpu"].startswith(holder["gpu_uuid"] + ",")
    assert {int(line.split(",")[0]) for line in initial["cuda_apps"].splitlines()} == {113, 2059}
    cg = initial["cgroup"]
    assert int(cg["memory.max"]) - int(cg["memory.current"]) >= 1024 * MIB
    result["holder"] = holder
    result["existing_owned_executor"] = executor
    small = pool = allocator = None
    try:
        import torch as torch_module

        torch = torch_module
        torch.set_num_threads(2)
        result["software"] = {
            "torch": torch.__version__,
            "git": torch.version.git_version,
            "cuda": torch.version.cuda,
            "path": torch.__file__,
        }
        assert torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90"
        torch.cuda.init()
        free, total = torch.cuda.mem_get_info()
        assert free >= args.cap_bytes + 64 * MIB, "No silent relaxation of physical preflight"
        after_import = record("after_torch_context", free_bytes=free, total_bytes=total)
        cg = after_import["cgroup"]
        assert int(cg["memory.max"]) - int(cg["memory.current"]) >= 64 * MIB
        torch.cuda.set_per_process_memory_fraction(float(args.cap_bytes / total))
        result["actual_fraction"] = torch.cuda.get_per_process_memory_fraction()
        result["actual_fraction_limit_bytes"] = round(result["actual_fraction"] * total)
        assert result["actual_fraction_limit_bytes"] == args.cap_bytes
        bridge = ctypes.CDLL(str(args.library.resolve()))
        bridge.probe_set_limit.argtypes = [ctypes.c_uint64]
        bridge.probe_set_limit.restype = ctypes.c_int
        bridge.probe_stat.argtypes = [ctypes.c_int]
        bridge.probe_stat.restype = ctypes.c_uint64
        bridge.probe_pointer_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        bridge.probe_pointer_info.restype = ctypes.c_int
        assert bridge.probe_set_limit(16 * MIB) == 0
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            str(args.library.resolve()), "probe_managed_alloc", "probe_managed_free"
        )
        pool = torch.cuda.MemPool(allocator.allocator(), use_on_oom=False)
        result["pool_id"] = pool.id
        record("pool_created", pool_use_count=pool.use_count())
        with torch.cuda.use_mem_pool(pool):
            small = torch.empty(MIB, dtype=torch.uint8, device="cuda")
            small.fill_(37)
        torch.cuda.synchronize()
        pointer = (ctypes.c_uint64 * 4)()
        pointer_rc = bridge.probe_pointer_info(ctypes.c_void_p(small.data_ptr()), pointer)
        assert pointer_rc == 0 and pointer[0] == 3, "Actual pointer must be CUDA managed"
        copied = small.cpu()
        assert copied.numel() == MIB and bool(torch.all(copied == 37).item())
        del copied
        result["small"] = {
            "pointer_type": int(pointer[0]),
            "device": int(pointer[1]),
            "pointer_returncode": pointer_rc,
            "data_verified": True,
        }
        record(
            "small_managed_verified",
            pool_snapshot=pool.snapshot(include_traces=False),
            pool_use_count=pool.use_count(),
            pointer=list(pointer),
        )
        before_boundary = stats()
        try:
            with torch.cuda.use_mem_pool(pool):
                unexpected = torch.empty(args.cap_bytes + 2 * MIB, dtype=torch.uint8, device="cuda")
            result["boundary"] = {"outcome": "UNEXPECTED_SUCCESS"}
            del unexpected
        except torch.OutOfMemoryError as error:
            result["boundary"] = {"outcome": "OUT_OF_MEMORY", "error": str(error)}
        after_boundary = stats()
        result["boundary"].update(
            {
                "alloc_calls_before": before_boundary["alloc_calls"],
                "alloc_calls_after": after_boundary["alloc_calls"],
                "custom_callback_reached": after_boundary["alloc_calls"]
                != before_boundary["alloc_calls"],
                "bridge_denials_delta": after_boundary["limit_denials"]
                - before_boundary["limit_denials"],
            }
        )
        record("boundary_complete", boundary=result["boundary"])
        assert result["boundary"]["outcome"] == "OUT_OF_MEMORY"
        assert not result["boundary"]["custom_callback_reached"]
        assert result["boundary"]["bridge_denials_delta"] == 0
        result["outcome"] = "MANAGED_SMALL_AND_PRE_CALLBACK_VIRTUAL_CAP_REFUSAL"
    except Exception as error:
        result["outcome"] = "FAILED"
        result["error"] = repr(error)
        raise
    finally:
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        small = None
        gc.collect()
        if pool is not None:
            record("before_pool_release", pool_use_count=pool.use_count())
        pool = None
        gc.collect()
        record("after_pool_release")
        result["final_bridge"] = stats()
        result["holder_after"] = proc_identity(113)
        result["executor_after"] = proc_identity(2059)
        result["cleanup_verified"] = (
            bridge is not None
            and stats()["live_bytes"] == 0
            and stats()["invalid_frees"] == 0
            and stats()["sync_failures"] == 0
            and result["holder_after"]["start_ticks"] == 33024000
            and result["executor_after"]["start_ticks"] == 33529715
        )
        with (args.out / "result.json").open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(result))
    assert result["cleanup_verified"]


if __name__ == "__main__":
    main()
