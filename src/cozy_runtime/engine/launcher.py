"""Which of the engine's launch paths this machine's driver runs (xs-213): its WHILE-node launcher
(`_engine.Segment`, a decode's steps in one launch), else the host replaying the step's graph, else
neither, and Transformers' engine serves. A driver that crashes a path crashes a child process, never
the executor (driver 615.71.09 crashed the launcher: engine-routes 44605985214).

At a model's install, off any request (`judge`): a child process runs the varena suite's engine-loop
sections on tiny graphs (`internal.bench.engine_loop`), the host-replayed one first, then the
launcher's (loop, polled, copies), each said as it ends. A section that fails, or that the child
never finished, refuses its path. A pass is kept on the machine, keyed by the `_engine` artifact,
the card, the driver and the Runtime build, so a later executor starts no child; a refusal is kept
nowhere, and the next executor checks again.

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
#: Each path's engine-loop sections, in the order the child runs them.
SECTIONS = {"replayed": ("replayed",), "launcher": ("loop", "polled", "copies")}
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
    crashed = verify.crashed(LAUNCHER, artifact, sm)
    if not crashed and _passed(key):
        return Paths("", "")
    said, code = _child(index)
    replayed = _refused(said, SECTIONS["replayed"], code)
    launcher = verify.CRASHED if crashed else _refused(said, SECTIONS["launcher"], code)
    if not replayed and not launcher:
        _keep(key)
    return Paths(launcher, replayed)


def _child(index: int) -> tuple[dict[str, tuple[bool, str]], int]:
    """Each section the child said, and its exit code (negative: the signal that ended it)."""
    command = [sys.executable, "-I", "-B", "-m", __name__, str(index)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=CHILD_S, check=False)
    except subprocess.TimeoutExpired:
        return {}, -1
    said: dict[str, tuple[bool, str]] = {}
    for line in done.stdout.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            said[str(row["section"])] = (bool(row["ok"]), str(row.get("detail", "")))
    return said, done.returncode


def _refused(said: dict[str, tuple[bool, str]], sections: tuple[str, ...], code: int) -> str:
    for section in sections:
        if section not in said:
            return f"its check's process ended (exit {code}) before its {section} section finished"
        ok, detail = said[section]
        if not ok:
            return f"its {section} section failed: {detail}"[:300]
    return ""


def _record(key: tuple[str, int, str, str]) -> Path | None:
    from cozy_runtime.internal import kernel_cache, verify

    store = kernel_cache.configured()
    if store is None:
        return None
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]
    return store.own / verify.VERDICTS / f"{LAUNCHER}-passed-{digest}.json"


def _passed(key: tuple[str, int, str, str]) -> bool:
    path = _record(key)
    return path is not None and path.is_file()


def _keep(key: tuple[str, int, str, str]) -> None:
    path = _record(key)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"artifact": key[0], "sm": key[1], "driver": key[2], "runtime": key[3]}))
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


def _main(index: int) -> None:
    """The child: each section on cuda:`index`, one JSON line as each ends."""
    import tensorfs.varena.torch as vt
    import torch

    from cozy_runtime.internal.bench import engine_loop

    vt.install(index)
    torch.cuda.set_device(index)
    for sections in SECTIONS.values():
        for section in sections:
            wrong: list[str] = []
            try:
                getattr(engine_loop, f"_{section}")(wrong)
                torch.cuda.synchronize()
            except Exception as exc:
                wrong.append(f"{type(exc).__name__}: {exc}")
            print(json.dumps({"section": section, "ok": not wrong, "detail": "; ".join(wrong)[:300]}), flush=True)


if __name__ == "__main__":
    _main(int(sys.argv[1]))
