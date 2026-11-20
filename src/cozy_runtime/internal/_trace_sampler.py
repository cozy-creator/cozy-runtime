"""Proof-only (cl-1318): an in-process stack sampler, py-spy's view where ptrace is denied. Every
`PERIOD_S` it records each thread's Python stack; a native call that released the GIL shows as
its Python caller. Written to /tmp/cozy-trace/<pid>.ndjson: `{"k": id, "f": [frames]}` once per
stack, then `{"t": unix ms, "th": thread, "k": id}` whenever a thread's stack changes (it holds
that stack until its next line). Never merged."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

PERIOD_S = 0.002
DEPTH = 60
_STARTED: set[int] = set()


def start() -> None:
    pid = os.getpid()
    if pid in _STARTED:
        return
    _STARTED.add(pid)
    os.makedirs("/tmp/cozy-trace", exist_ok=True)
    thread = threading.Thread(target=_run, args=(pid,), name="cl1318-sampler", daemon=True)
    thread.start()


def _run(pid: int) -> None:
    me = threading.get_ident()
    stacks: dict[tuple[str, ...], int] = {}
    out = open(f"/tmp/cozy-trace/{pid}.ndjson", "a", buffering=1 << 20)
    names: dict[int, str] = {}
    held: dict[int, int] = {}
    last_flush = time.monotonic()
    while True:
        time.sleep(PERIOD_S)
        now = time.time() * 1000
        frames = sys._current_frames()
        if len(names) != threading.active_count():
            names = {t.ident: t.name for t in threading.enumerate() if t.ident is not None}
        lines = []
        for ident, frame in frames.items():
            if ident == me:
                continue
            stack: list[str] = []
            f = frame
            while f is not None and len(stack) < DEPTH:
                code = f.f_code
                stack.append(f"{os.path.basename(code.co_filename)}:{code.co_name}:{f.f_lineno}")
                f = f.f_back
            key = tuple(stack)
            sid = stacks.get(key)
            if sid is None:
                sid = stacks[key] = len(stacks)
                lines.append(json.dumps({"k": sid, "f": stack}))
            if held.get(ident) != sid:
                held[ident] = sid
                lines.append(json.dumps({"t": round(now, 1), "th": names.get(ident, str(ident)), "k": sid}))
        del frames
        if lines:
            out.write("\n".join(lines) + "\n")
        if time.monotonic() - last_flush > 1.0:
            out.flush()
            last_flush = time.monotonic()
