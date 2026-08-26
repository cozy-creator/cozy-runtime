"""The out-of-process postmortem witness: it names deaths the dying process cannot report.

Executor death was already covered — the worker outlives it, journals a terminal, and
converges. The worker's OWN death was not, and it is the one that matters most: the
journal owner is the process that says what happened, so when the kernel kills it there is
nobody left to write the sentence. v1's postmortem gap is exactly this shape, and every
attempt to close it from inside the dying process failed for the same reason — an OOM kill
is SIGKILL and SIGKILL runs no handler.

So a WITNESS parent execs the worker and outlives it. It is deliberately tiny: no torch,
no protocol, no journal writes, no threads. It writes a BOOT record before the child starts,
waits, and writes a DEATH record naming what it saw from outside:

    exit status, and the SIGNAL when there was one
    whether the kernel OOM-killed it — read from the cgroup's own `memory.events` counter,
      as a DELTA across the child's life rather than an absolute nobody can interpret
    how long it ran, and the argv it ran

The record lives OFF tmpfs. A boot record on a ramdisk is a boot record that dies with the
same event it exists to explain — proven by `statfs`, refused rather than warned about,
because a witness that silently wrote to RAM is worse than no witness at all.

Hardware and kernel facts here are TRI-STATE: present, absent, or UNREADABLE. An unreadable
cgroup counter is named `unreadable`, never reported as "not OOM-killed" — the whole value
of this record is that it does not guess.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

BOOT_FORMAT = "cozy.runtime.WitnessBoot/1"
DEATH_FORMAT = "cozy.runtime.WitnessDeath/1"

#: `statfs.f_type` values for memory-backed filesystems. A boot record on one of these dies
#: with the machine event it exists to explain.
TMPFS_MAGIC = frozenset({0x01021994, 0x858458F6, 0x9041934})  # tmpfs, ramfs, older ramfs


class WitnessRefusal(Exception):
    """The witness cannot do its job honestly and says so instead of pretending."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require_durable(directory: Path) -> str:
    """Refuse a memory-backed record location. Returns the filesystem's magic, as evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(directory)
    magic = getattr(stat, "f_fsid", 0)
    try:
        with open("/proc/self/mountinfo") as handle:
            mounts = [line.split() for line in handle]
    except OSError:
        return "unreadable"
    best, kind = "", ""
    for parts in mounts:
        if "-" not in parts:
            continue
        point = parts[4]
        fstype = parts[parts.index("-") + 1]
        if str(directory).startswith(point) and len(point) > len(best):
            best, kind = point, fstype
    if kind in ("tmpfs", "ramfs"):
        raise WitnessRefusal(
            "boot_record_on_ram",
            f"{directory} is on {kind} ({best}): a boot record kept in RAM dies with the "
            "very event it exists to explain, so the witness refuses rather than writing "
            "one that will not survive",
        )
    return kind or f"fsid:{magic}"


def _cgroup_path() -> Path | None:
    """This process's cgroup v2 directory, or None when it cannot be resolved."""
    try:
        line = Path("/proc/self/cgroup").read_text().strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    _, _, relative = line.partition("::")
    if not relative:
        return None
    candidate = Path("/sys/fs/cgroup") / relative.lstrip("/")
    return candidate if candidate.is_dir() else None


def oom_kills() -> int:
    """The cgroup's cumulative `oom_kill` count, or -1 when UNREADABLE (never 0)."""
    cgroup = _cgroup_path()
    if cgroup is None:
        return -1
    try:
        for line in (cgroup / "memory.events").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "oom_kill":
                return int(value)
    except (OSError, ValueError):
        return -1
    return -1


def _write(path: Path, document: dict[str, Any]) -> None:
    """Write and fsync — a witness record that is only in the page cache witnessed nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(handle, data)
        os.fsync(handle)
    finally:
        os.close(handle)


def name_death(status: int, before: int, after: int) -> dict[str, Any]:
    """Turn a wait status into a NAMED death. Tri-state on the OOM question."""
    signalled = os.WIFSIGNALED(status)
    number = os.WTERMSIG(status) if signalled else 0
    code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
    if before < 0 or after < 0:
        oom = "unreadable"
    elif after > before:
        oom = "true"
    else:
        oom = "false"
    if signalled and number == signal.SIGKILL and oom == "false":
        how = "killed_externally"
    elif signalled and number == signal.SIGKILL and oom == "true":
        how = "oom_killed"
    elif signalled:
        how = f"signalled_{signal.Signals(number).name}"
    elif code == 0:
        how = "exited_clean"
    else:
        how = f"exited_{code}"
    return {
        "how": how,
        "exit_code": code,
        "signal": number,
        "signal_name": signal.Signals(number).name if number else "",
        "oom_killed": oom,
        "cgroup_oom_kill_before": before,
        "cgroup_oom_kill_after": after,
        "method": (
            "wait status observed by the PARENT, plus the cgroup memory.events oom_kill "
            "counter read before and after the child's life; an unreadable counter is named "
            "unreadable and never reported as 'not OOM-killed'"
        ),
    }


def witness(root: Path, argv: list[str]) -> int:
    """Exec `argv` as a child, outlive it, and record how it died. Returns its exit code."""
    directory = root / "witness"
    filesystem = require_durable(directory)
    before = oom_kills()
    started = time.time()
    child = os.fork()
    if child == 0:  # the worker, in its own process group so a group kill is fenced
        os.setpgid(0, 0)
        os.execv(argv[0], argv)
        os._exit(127)
    _write(
        directory / "boot.json",
        {
            "format": BOOT_FORMAT,
            "witness_pid": os.getpid(),
            "child_pid": child,
            "argv": argv,
            "started_unix_ms": int(started * 1000),
            "record_filesystem": filesystem,
            "cgroup_oom_kill_at_boot": before,
        },
    )
    _, status = os.waitpid(child, 0)
    death = name_death(status, before, oom_kills())
    death.update(
        format=DEATH_FORMAT,
        child_pid=child,
        ran_ms=round((time.time() - started) * 1000, 1),
        ended_unix_ms=int(time.time() * 1000),
    )
    _write(directory / "death.json", death)
    print(f"[witness] {death['how']} (oom_killed={death['oom_killed']})", flush=True)
    return death["exit_code"] if death["exit_code"] >= 0 else 128 + int(death["signal"])


def read_death(root: Path) -> dict[str, Any]:
    """The surviving record, read from disk with no live process anywhere."""
    path = root / "witness" / "death.json"
    if not path.is_file():
        raise WitnessRefusal("death_unrecorded", f"no witness death record under {root}")
    document: dict[str, Any] = json.loads(path.read_text())
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cozy-runtime worker witness")
    parser.add_argument("--root", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = [a for a in args.command if a != "--"]
    if not command:
        print("[witness] nothing to witness: pass -- <command>", file=sys.stderr)
        return 2
    try:
        return witness(Path(args.root), command)
    except WitnessRefusal as exc:
        print(f"[witness] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
