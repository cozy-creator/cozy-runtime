# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.27.4,<0.31"]
# [tool.cozy]
# accelerator = false
# ///
"""Inspect current-UID Varena tables and kernel OFD claims; never acquires a lock or writes them."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import struct
import tempfile
from pathlib import Path
from typing import Any

from cozy_runtime.author import Outputs, Tree

HEADER, SLOT, ROWS, SHARED = 64, 576, 64, 90176
CLASSES = ("free", "orphan", "speculative", "unused", "declared", "working", "pinned")
FLOCK = struct.Struct("hh4xqqi4x")


def read_at(fd: int, count: int, offset: int) -> bytes:
    data = os.pread(fd, count, offset)
    if len(data) != count:
        raise ValueError(f"short table read at {offset}: {len(data)} != {count}")
    return data


def process(pid: int, uid: int) -> dict[str, Any]:
    folder = Path(f"/proc/{pid}")
    try:
        if folder.stat().st_uid != uid:
            return {"exists": True, "owner_differs": True}
        raw = (folder / "stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        return {
            "exists": True,
            "state": fields[0],
            "parent_pid": int(fields[1]),
            "start_ticks": int(fields[19]),
        }
    except OSError as exc:
        return {"exists": False, "errno": exc.errno}


def inspect(path: Path, uid: int) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != uid or not stat.S_ISREG(info.st_mode) or info.st_size != SHARED:
            raise ValueError("owner/type/table size differs from the checked current layout")
        header = read_at(fd, HEADER, 0)
        out: dict[str, Any] = {
            "path": str(path),
            "device": info.st_dev,
            "inode": info.st_ino,
            "magic": hex(struct.unpack_from("<Q", header, 0)[0]),
            "capacity": struct.unpack_from("<Q", header, 8)[0],
            "changed": struct.unpack_from("<I", header, 16)[0],
            "rows": [],
        }
        for index in range(ROWS):
            offset = HEADER + SLOT * index
            raw = b""
            stable = False
            for _ in range(3):
                raw = read_at(fd, SLOT, offset)
                state, _, _, seq = struct.unpack_from("<QIIQ", raw)
                again = read_at(fd, 24, offset)
                state_after, _, _, seq_after = struct.unpack("<QIIQ", again)
                if state == state_after and seq == seq_after:
                    stable = True
                    break
            flags = state & 3
            if not flags:
                continue
            # F_OFD_GETLK is a query on this freshly opened, read-only OFD. The
            # range is the exact claim range used by Ledger::claim/alive.
            probe = FLOCK.pack(fcntl.F_WRLCK, os.SEEK_SET, offset, SLOT, 0)
            answer = fcntl.fcntl(fd, fcntl.F_OFD_GETLK, probe)
            lock_type, _, lock_start, lock_length, lock_pid = FLOCK.unpack(answer)
            claimed = lock_type != fcntl.F_UNLCK
            pid = struct.unpack_from("<I", raw, 8)[0]
            words = struct.unpack_from("<25Q", raw, 24 + 200 * (seq & 1))
            running = [
                {
                    "id": words[9 + 2 * i],
                    "remaining": None if words[10 + 2 * i] == (1 << 64) - 1 else words[10 + 2 * i],
                }
                for i in range(min(words[8], 8))
            ]
            out["rows"].append(
                {
                    "row": index,
                    "generation": state >> 2,
                    "raw_state": state,
                    "state": "reaping" if flags == 2 else "live" if claimed else "dead",
                    "stable_snapshot": stable,
                    "pid": pid,
                    "process": process(pid, uid),
                    "claim_present": claimed,
                    "lock_type": lock_type,
                    "lock_start": lock_start,
                    "lock_length": lock_length,
                    "lock_pid": lock_pid,
                    "classes": dict(zip(CLASSES, words[:7], strict=True)),
                    "context": words[7],
                    "running": running,
                    "wakeups": struct.unpack_from("<Q", raw, 536)[0],
                    "progress": struct.unpack_from("<Q", raw, 544)[0],
                    "cache_release": struct.unpack_from("<Q", raw, 552)[0],
                    "cache_inflight": struct.unpack_from("<Q", raw, 560)[0],
                }
            )
        return out
    finally:
        os.close(fd)


def main(*, out: Outputs) -> Tree:
    uid = os.geteuid()
    tables, failures = [], []
    for path in sorted(Path("/dev/shm").glob(f"varena-*-{uid}")):
        try:
            tables.append(inspect(path, uid))
        except (OSError, ValueError, struct.error) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    result = {
        "pid": os.getpid(),
        "uid": uid,
        "read_only": True,
        "tables": tables,
        "failures": failures,
    }
    folder = Path(tempfile.mkdtemp(prefix="varena-ledger-inspect-"))
    (folder / "ledger.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return out.save_tree(folder)
