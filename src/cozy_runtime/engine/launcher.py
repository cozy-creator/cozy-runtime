"""Which of the engine's launch paths this machine's driver runs (xs-213): its WHILE-node launcher
(`_engine.Segment`, a decode's steps in one launch), else the host replaying the step's graph, else
neither, and Transformers' engine serves. A driver that crashes a path crashes a child process, never
the executor (driver 615.71.09 crashed the launcher: engine-routes 44605985214).

At a model's install, off any request (`judge`): a child process runs the varena suite's engine-loop
sections on tiny graphs (`internal.bench.engine_loop`), the launcher's (loop, polled, copies) then
the host-replayed one, each said as it ends, for each path not yet decided. A child that a signal
ends decides the path it was on: crashed, kept as a crash marker for this boot (`verify`'s, keyed by
the `_engine` artifact, the card, the driver and the Runtime build), and a next child checks the
paths after it. A section that fails refuses its path in this process only. A pass is kept on the
machine under the same key, so a later executor starts no child for that path.

A process's first launch of the launcher is guarded too (`first`): one that crashes refuses the
launcher in this boot's next processes (`verify.guarded`), whatever the check said.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

#: The launcher, as `verify` keys its crash marker.
LAUNCHER = "cozy-engine-launcher"
#: Each path's engine-loop sections, in the order a check runs them: a crash it cannot place (one
#: before any section, as a check's process starting) is the first path's.
SECTIONS = {"launcher": ("loop", "polled", "copies"), "replayed": ("replayed",)}
#: The step-graph node kinds (`_engine.kinds`) the check's launcher sections hold, so a driver that
#: faults on one fails the check, not a call: every kind a decode step holds. A step holding another
#: is never launched (`Decoder`): the check must grow with the step. An empty node does no work.
COVERED = frozenset({"kernel", "memcpy host-to-device", "memcpy device-to-device", "memset", "empty"})
#: Each path as `verify` keys its crash marker.
KEYS = {"launcher": LAUNCHER, "replayed": f"{LAUNCHER}-replayed"}
#: The child's longest run.
CHILD_S = 180


@dataclass(frozen=True)
class Paths:
    """Why each path can't serve here: "" where it can."""

    launcher: str
    replayed: str


_JUDGED: dict[int, Paths] = {}
_JUDGING: set[int] = set()
_FIRST: set[int] = set()
_LOCK = threading.Lock()


def judge(index: int) -> None:
    """Starts cuda:`index`'s check on a thread of its own, once a process."""
    with _LOCK:
        if index in _JUDGED or index in _JUDGING:
            return
        _JUDGING.add(index)
    threading.Thread(target=_judge, args=(index,), name=f"engine-launcher cuda:{index}", daemon=True).start()


def judged(index: int) -> Paths | None:
    """cuda:`index`'s paths once checked, else None."""
    return _JUDGED.get(index)


def _judge(index: int) -> None:
    try:
        paths = _paths(index)
    except Exception as exc:  # the check itself broke: neither path is proven
        why = f"its check failed: {type(exc).__name__}: {exc}"[:300]
        paths = Paths(why, why)
    with _LOCK:
        _JUDGED[index] = paths
        _JUDGING.discard(index)
    print(f"[engine] cuda:{index}: launcher {paths.launcher or 'serves'}; replayed {paths.replayed or 'serves'}",
          file=sys.stderr)  # fmt: skip


def _key(index: int) -> tuple[str, int, str, str]:
    import torch

    from cozy_runtime.internal import accel, prebuilt, verify

    major, minor = torch.cuda.get_device_capability(index)
    return prebuilt.artifact("_engine"), major * 10 + minor, accel.driver_version(), verify.RUNTIME


def _paths(index: int) -> Paths:
    from cozy_runtime.internal import verify

    artifact, sm, _, _ = key = _key(index)
    why: dict[str, str | None] = {}
    for path in SECTIONS:
        if verify.crashed(KEYS[path], artifact, sm):
            why[path] = verify.CRASHED
        else:
            why[path] = "" if _passed(key, path) else None
    while todo := [path for path in SECTIONS if why[path] is None]:
        said, code = _child(index, [section for path in todo for section in SECTIONS[path]])
        for path in todo:
            failed = [f"its {s} section failed: {said[s][1]}"[:300] for s in SECTIONS[path] if s in said and not said[s][0]]
            missing = [s for s in SECTIONS[path] if s not in said]
            if failed:
                why[path] = failed[0]
            elif not missing:
                why[path] = ""
                _keep(key, path)
            else:  # the child ended on this path: a signal is its crash; the paths after it go to the next child
                ended = f"its check's process ended (exit {code}) in its {missing[0]} section"
                why[path] = ended
                if code < 0:
                    verify.remember_crash(KEYS[path], artifact, sm, ended)
                break
        if code >= 0 and any(why[path] is None for path in todo):  # it ended without a signal, saying nothing more
            for path in todo:
                why[path] = why[path] if why[path] is not None else f"its check's process ended (exit {code})"
    return Paths(why["launcher"] or "", why["replayed"] or "")


def _child(index: int, sections: list[str]) -> tuple[dict[str, tuple[bool, str]], int]:
    """Each of `sections` the child said, and its exit code (negative: the signal that ended it). The child runs in
    the executor's own place, never a call's: a call's is removed when it ends, and a child that outlived one failed
    its section on a missing working directory (engine-routes 44633999562)."""
    from cozy_runtime.internal import call_scope

    command = [sys.executable, "-I", "-B", "-m", __name__, str(index), *sections]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=CHILD_S, check=False,
                              env=call_scope.process_environment(), cwd=call_scope.process_cwd())  # fmt: skip
    except subprocess.TimeoutExpired:
        return {}, 0
    said: dict[str, tuple[bool, str]] = {}
    for line in done.stdout.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            said[str(row["section"])] = (bool(row["ok"]), str(row.get("detail", "")))
    return said, done.returncode


def _record(key: tuple[str, int, str, str], path: str) -> Path | None:
    from cozy_runtime.internal import kernel_cache, verify

    store = kernel_cache.configured()
    if store is None:
        return None
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]
    return store.own / verify.VERDICTS / f"{KEYS[path]}-passed-{digest}.json"


def _passed(key: tuple[str, int, str, str], path: str) -> bool:
    record = _record(key, path)
    return record is not None and record.is_file()


def _keep(key: tuple[str, int, str, str], path: str) -> None:
    record = _record(key, path)
    if record is None:
        return
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"artifact": key[0], "sm": key[1], "driver": key[2], "runtime": key[3]}))
    except OSError:
        return


@contextmanager
def first(index: int, launch: bool) -> Iterator[None]:
    """A launcher's build (`launch` False) or launch on cuda:`index`, under a crash marker
    (`verify.guarded`) until this process's first launch there has run; later ones as they are."""
    if index in _FIRST:
        yield
        return
    from cozy_runtime.internal import verify

    artifact, sm, _, _ = _key(index)
    with verify.guarded(LAUNCHER, artifact, sm):
        yield
    if launch:
        _FIRST.add(index)


def _main(index: int, sections: list[str]) -> None:
    """The child: each of `sections` on cuda:`index`, one JSON line as each ends."""
    import tensorfs.varena.torch as vt
    import torch

    from cozy_runtime.internal.bench import engine_loop

    vt.install(index)
    torch.cuda.set_device(index)
    for section in sections:
        wrong: list[str] = []
        try:
            getattr(engine_loop, f"_{section}")(wrong)
            torch.cuda.synchronize()
        except Exception as exc:
            wrong.append(f"{type(exc).__name__}: {exc}")
        print(json.dumps({"section": section, "ok": not wrong, "detail": "; ".join(wrong)[:300]}), flush=True)


if __name__ == "__main__":
    _main(int(sys.argv[1]), sys.argv[2:])
