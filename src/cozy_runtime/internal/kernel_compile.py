"""The machine's compile manager: every artifact compiles once per key, on this machine, for the
card present, in a niced background process, and is published whole to the kernel store.

`submit` never blocks and never compiles in the caller. The store holds the artifact (`ready`),
or a live builder holds its lock (`compiling`, with the progress it reports), or this worker
boot already failed it (`failed`, retried at the next boot only), or `submit` starts the
builder and says `compiling`. Nothing waits on a compile: a consumer serves the best READY
artifact and falls through otherwise.

The builder is `python -m cozy_runtime.internal.kernel_compile` under the caller's own
interpreter and import path, so an extension builds against the torch that will import it. It
inherits the key's lock, takes one of the machine's build slots (flock files, so a dead
builder frees its slot), runs at nice 19, and is ended only on a PROVEN wedge (`liveness`): its
whole process tree stopped burning CPU and writing output, against its own observed pace.
A failure is recorded in the worker boot's scratch scope (`TMPDIR`), which the next boot
starts without.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from cozy_runtime.internal import config, kernel_cache, liveness, proctree

#: kind -> the `module:function` a builder runs: `build(spec, staging, report) -> facts`. The
#: function fills `staging` (an `object` file, a `site/` tree, or nothing when the artifact
#: lives in its compiler's own keyed cache) and returns producer facts.
BUILDERS = {
    "setup": "cozy_runtime.internal.kernel_sources:build_setup",
    "python": "cozy_runtime.internal.kernel_sources:build_python",
    "sol": "cozy_runtime.internal.attention_sol:build",
    "fusion": "cozy_runtime.internal.fusion:build",
}
#: Kinds that run many compiler processes: one at a time on the machine, each as wide as the
#: machine measures (`width`). The others are one Python process each and run at once.
WIDE = frozenset({"setup"})
_BOOT = (
    "import json, sys; boot = json.loads(sys.argv.pop(1)); sys.path[:] = boot['path']; "
    "sys.pycache_prefix = boot['pycache_prefix']; "
    "from cozy_runtime.internal.kernel_compile import main; raise SystemExit(main(sys.argv[1:]))"
)
_PROGRESS = re.compile(rb"^\[(\d+)/(\d+)\]")


class CompileFailed(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class Job:
    """One artifact: its exact key, the builder kind and the builder's JSON spec."""

    kind: str
    key: kernel_cache.Key
    spec: Mapping[str, Any] = field(default_factory=dict)
    #: variables the builder runs under beyond the caller's own, e.g. a hidden GPU
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class State:
    """What a consumer may do with an artifact now."""

    kernel: str
    state: str  # ready | compiling | pending | failed | absent | unsupported
    progress: float | None = None
    ms: float = 0.0
    detail: str = ""
    path: str = ""

    def line(self) -> str:
        if self.state == "compiling" and self.progress is not None:
            return f"compiling ({self.progress:.0%})"
        if self.state == "ready" and self.ms:
            return f"ready (compiled in {self.ms / 1000:.1f} s)"
        return f"{self.state}: {self.detail}" if self.detail else self.state

    def document(self) -> dict[str, Any]:
        row: dict[str, Any] = {"kernel": self.kernel, "state": self.state, "line": self.line()}
        if self.progress is not None:
            row["progress"] = round(self.progress, 3)
        if self.ms:
            row["compile_ms"] = round(self.ms, 1)
        if self.detail:
            row["detail"] = self.detail[:300]
        return row


#: This process's builders, by key digest, so their exit is observed without a clock.
_STARTED: dict[str, subprocess.Popen[bytes]] = {}
_GUARD = threading.Lock()


def status(store: kernel_cache.Store, job: Job) -> State:
    """The artifact's state, reading only: the store, the boot's failures, the builder's word.
    `pending` is nobody building it: a builder's word outlives it only until its lock frees."""
    key = job.key
    found = store.entry(key)
    if found is not None:
        return State(key.kernel, "ready", ms=_compile_ms(found), path=str(found))
    failed = _failure(key)
    if failed is not None:
        return State(key.kernel, "failed", detail=f"{failed['code']}: {failed['detail']}"[:300])
    live = store.state(key)
    free = store.claim(key)
    if free is not None:
        os.close(free)
        if not store.built_by_worker(key):
            return State(key.kernel, "pending")
    live = live or {}
    progress, started = live.get("progress"), live.get("started_unix_ms")
    return State(
        key.kernel,
        "compiling",
        progress if isinstance(progress, (int, float)) else None,
        round(time.time() * 1000 - started, 1) if isinstance(started, (int, float)) else 0.0,
        str(live.get("phase") or "") if live.get("phase") != "compiling" else "",
    )


def submit(store: kernel_cache.Store, job: Job) -> State:
    """`status`, after starting the builder when nobody on this machine is building the key."""
    current = status(store, job)
    if current.state != "pending":
        return current
    with _GUARD:
        running = _STARTED.get(job.key.digest)
        lock = store.claim(job.key) if running is None or running.poll() is not None else None
        if lock is not None:
            try:
                if store.entry(job.key) is None and _failure(job.key) is None:
                    store.set_state(job.key, None)  # a dead builder's last word, if any
                    _STARTED[job.key.digest] = _spawn(store, job, lock)
            finally:
                os.close(lock)
    current = status(store, job)
    return current if current.state != "pending" else State(job.key.kernel, "compiling")


def in_order(
    store: kernel_cache.Store,
    jobs: Sequence[Job],
    environment: Mapping[str, str],
    done: Callable[[Job, State], None],
) -> threading.Thread:
    """The machine's own compiles, one after another in `jobs`' order (its priority), each
    as wide as the machine. Every key nobody builds yet is claimed at once, so no executor
    on this machine starts a second build of it while it waits its turn; `done` sees each
    one ready or failed, and at once one another process already built or builds."""
    queued: list[tuple[Job, int]] = []
    for job in jobs:
        lock = store.claim(job.key) if store.entry(job.key) is None else None
        if lock is None:
            continue
        store.set_state(job.key, {"pid": os.getpid(), "progress": 0.0, "phase": "queued"})
        queued.append((job, lock))

    def run() -> None:
        for job in jobs:
            held = next((lock for queued_job, lock in queued if queued_job is job), None)
            if held is not None:
                try:
                    with _GUARD:
                        _STARTED[job.key.digest] = _spawn(store, job, held, environment)
                except OSError as exc:
                    store.set_state(job.key, None)
                    done(job, State(job.key.kernel, "failed", detail=f"compile_unstarted: {exc}"))
                    continue
                finally:
                    os.close(held)
            with store.building(job.key):
                pass
            done(job, status(store, job))

    thread = threading.Thread(target=run, name="machine-kernels", daemon=True)
    thread.start()
    return thread


def learn(store: kernel_cache.Store, scope: str, job: Job) -> None:
    """Remember, on this machine only, an artifact a request needed, so the next boot compiles
    it ahead (`learned`). One line per key under the store's own namespace; never shared."""
    path = store.own / "learned" / f"{scope}.jsonl"
    row = json.dumps(
        {
            "kind": job.kind,
            "kernel": job.key.kernel,
            "inputs": json.loads(job.key.inputs),
            "spec": dict(job.spec),
            "env": dict(job.env),
        },
        sort_keys=True,
    )
    with contextlib.suppress(OSError):
        if row not in (path.read_text().splitlines() if path.exists() else ()):
            path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            with path.open("a") as log:
                log.write(row + "\n")


def learned(store: kernel_cache.Store, scope: str) -> list[Job]:
    """What `learn` remembered under `scope`, as jobs to submit."""
    try:
        lines = (store.own / "learned" / f"{scope}.jsonl").read_text().splitlines()
    except OSError:
        return []
    jobs = []
    for line in lines:
        with contextlib.suppress(ValueError, KeyError, TypeError, kernel_cache.KernelCacheRefusal):
            row = json.loads(line)
            key = kernel_cache.Key.of(row["kernel"], row["inputs"])
            jobs.append(Job(row["kind"], key, row["spec"], row["env"]))
    return jobs


def _spawn(
    store: kernel_cache.Store,
    job: Job,
    lock: int,
    environment: Mapping[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    argv = [
        sys.executable,
        "-I",
        "-c",
        _BOOT,
        json.dumps({"path": sys.path, "pycache_prefix": sys.pycache_prefix}),
        "--store",
        str(store.own),
        *(["--trusted", str(store.trusted)] if store.trusted is not None else []),
        "--lock-fd",
        str(lock),
        "--job",
        json.dumps(
            {
                "kind": job.kind,
                "kernel": job.key.kernel,
                "inputs": json.loads(job.key.inputs),
                "spec": dict(job.spec),
            }
        ),
    ]
    process = subprocess.Popen(
        argv,
        env={**(config.inherited_environment() if environment is None else environment), **job.env},
        pass_fds=(lock,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    scope = _failures()

    def reap() -> None:
        _, err = process.communicate()
        if process.returncode and _failure(job.key, scope) is None:
            # Died before it could say why (killed, or an import failed): say it for it.
            tail = err.decode(errors="replace").strip()[-600:]
            detail = f"exit {process.returncode}: {tail}"
            _record_failure(job.key, "compile_crashed", detail, scope)

    threading.Thread(target=reap, name=f"kernel-builder-{job.key.kernel}", daemon=True).start()
    return process


def _compile_ms(entry_dir: Path) -> float:
    try:
        facts = json.loads((entry_dir / "entry.json").read_text()).get("producer") or {}
        return float(facts.get("compile_ms") or 0.0)
    except (OSError, ValueError, TypeError):
        return 0.0


# ------------------------------------------------------------------ boot-scoped failures


def _failures() -> Path:
    return Path(tempfile.gettempdir()) / "cozy-kernel-failures"


def _failure(key: kernel_cache.Key, scope: Path | None = None) -> dict[str, Any] | None:
    try:
        found = json.loads(((scope or _failures()) / f"{key.kernel}-{key.digest}.json").read_text())
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def _record_failure(
    key: kernel_cache.Key, code: str, detail: str, scope: Path | None = None
) -> None:
    directory = scope or _failures()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{key.kernel}-{key.digest}.json"
    scratch = path.with_suffix(f".{os.getpid()}")
    kept = detail if len(detail) <= 4000 else detail[:2000] + "\n...\n" + detail[-2000:]
    scratch.write_text(json.dumps({"code": code, "detail": kept}))
    os.replace(scratch, path)


# --------------------------------------------------------------------- the builder process


def width(unit_bytes: int) -> int:
    """How many compiler processes one wide build runs: every CPU this process may use (its
    affinity and its cgroup's quota), as far as the memory it may still take holds one
    `unit_bytes` process each. Measured each build; the machine is the pod's."""
    cpus = len(os.sched_getaffinity(0))
    quota = _read("/sys/fs/cgroup/cpu.max").split()
    if len(quota) == 2 and quota[0].isdigit() and quota[1].isdigit() and int(quota[1]):
        cpus = min(cpus, -(-int(quota[0]) // int(quota[1])))
    memory = [_meminfo("MemAvailable")]
    limit, used = _read("/sys/fs/cgroup/memory.max"), _read("/sys/fs/cgroup/memory.current")
    if limit.isdigit() and used.isdigit():
        memory.append(int(limit) - int(used))
    return max(1, min(cpus, min(memory) // unit_bytes))


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _meminfo(field: str) -> int:
    found = re.search(rf"^{field}:\s+(\d+) kB", _read("/proc/meminfo"), re.MULTILINE)
    return int(found.group(1)) * 1024 if found else 0


def machine_slot(namespace: Path, *, create: bool = False) -> Path:
    """The build slot's lock file; the worker creates its own at boot for its executors."""
    path = namespace / ".slots" / "0.lock"
    if create:
        path.parent.mkdir(mode=0o755, exist_ok=True)
        os.close(os.open(path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o644))
    return path


@contextlib.contextmanager
def _slot(store: kernel_cache.Store, kind: str) -> Iterator[None]:
    """A wide build holds the machine's one build slot, waiting on a live builder that has
    it; others need none. The slot lives in the worker's namespace when this process reads
    one (an executor under uid isolation), so the machine has one; it is an flock, so a
    builder that dies frees it."""
    if kind not in WIDE:
        yield
        return
    try:
        fd = os.open(machine_slot(store.trusted or store.own), os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        fd = os.open(machine_slot(store.own, create=True), os.O_RDONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    report: Callable[[float], None],
) -> str:
    """Run one compiler command to completion; return its output's tail.

    Progress is ninja's `[done/total]`. The command runs in its own session so a proven wedge
    ends its whole tree: the meter is the tree's CPU and bytes moved plus the output it wrote,
    judged against its own longest pause (`liveness.Pace`), never against a clock."""
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    tail: list[bytes] = []
    errors: list[bytes] = []  # a compiler's first errors, which a traceback tail buries
    written = [0]
    # Units of every ninja run so far: a project that builds several extensions, one after
    # another or at once, starts a new `[1/N]` for each, so progress is the units done over
    # all the units announced.
    done, announced = [0], [0]

    def drain() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            written[0] += len(line)
            tail.append(line)
            del tail[:-40]
            if b"error" in line.lower() and len(errors) < 12:
                errors.append(line)
            found = _PROGRESS.match(line)
            if found and int(found.group(2)):
                done[0] += 1
                if int(found.group(1)) == 1:
                    announced[0] += int(found.group(2))
                report(min(1.0, done[0] / max(announced[0], 1)))

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    pace = liveness.Pace()
    while True:
        try:
            process.wait(timeout=liveness.SAMPLE_SECONDS)
            break
        except subprocess.TimeoutExpired:
            pass
        tree = [process.pid, *proctree.descendants(process.pid)]
        burns = [liveness.burn(pid) for pid in tree]
        pace.observe((sum(b for b in burns if b is not None), written[0]))
        if pace.wedged(liveness.noise_floor()):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise CompileFailed("compile_stalled", pace.verdict(liveness.noise_floor()))
    reader.join()
    output = b"".join(tail).decode(errors="replace")
    if process.returncode:
        first = b"".join(errors).decode(errors="replace")
        detail = f"{argv[0]} exited {process.returncode}: {first}...\n{output}"
        raise CompileFailed("compile_failed", detail)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cozy_runtime.internal.kernel_compile")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--trusted", type=Path, default=None)
    parser.add_argument("--lock-fd", type=int, default=-1)
    parser.add_argument("--job", required=True)
    options = parser.parse_args(argv)
    os.nice(19)
    raw = json.loads(options.job)
    store = kernel_cache.Store(options.store, options.trusted)
    key = kernel_cache.Key.of(raw["kernel"], raw["inputs"])
    started, started_unix_ms = time.perf_counter(), round(time.time() * 1000)
    last = [(-1.0, "")]

    def report(progress: float, phase: str = "compiling") -> None:
        # A new phase is always written: the slot's taker must stop reading as queued.
        if abs(progress - last[0][0]) >= 0.01 or phase != last[0][1]:
            last[0] = (progress, phase)
            live = {"pid": os.getpid(), "started_unix_ms": started_unix_ms}
            store.set_state(key, {**live, "progress": progress, "phase": phase})

    report(0.0, "queued for a build slot")
    try:
        with _slot(store, raw["kind"]):
            report(0.0)
            module, name = BUILDERS[raw["kind"]].split(":")
            build = getattr(import_module(module), name)

            def fill(staging: Path) -> Mapping[str, object]:
                facts = dict(build(raw["spec"], staging, report))
                elapsed = (time.perf_counter() - started) * 1000
                return {"host": os.uname().nodename, "compile_ms": round(elapsed, 1), **facts}

            store.publish(key, fill)
    except CompileFailed as exc:
        _record_failure(key, exc.code, str(exc))
        return 1
    except BaseException as exc:
        _record_failure(key, "compile_failed", f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        store.set_state(key, None)
    print(json.dumps({"key": key.digest, "ms": round((time.perf_counter() - started) * 1000)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
