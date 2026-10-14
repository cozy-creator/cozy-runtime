"""The machine's compile manager: every artifact compiles once per key, on this machine, for the
card present, in a niced background process, and is published whole to the kernel store.

`submit` never blocks and never compiles in the caller. The store holds the artifact (`ready`),
or a live builder holds its lock (`compiling`, with the progress it reports), or this machine
run already failed it (`failed`, retried at the next run only), or `submit` starts the builder
and says `compiling`. A consumer that needs the artifact `wait`s for its build and never runs
another kernel in its place; the machine compiles the ones every card needs ahead of any
request (`machine_kernels`).

The builder is `python -m cozy_runtime.internal.kernel_compile` under the caller's own
interpreter and import path, so an extension builds against the torch that will import it. It
inherits the key's lock, runs at nice 19, and is ended only on a PROVEN wedge (`liveness`):
every compiler on the machine stopped burning CPU and writing output, against its own
observed pace. Each of its compiler jobs is admitted by the machine's compile ledger
(`compiler_job`), within the CPUs and the memory each job is measured to take.
A failure is recorded in the machine run's scratch scope (`TMPDIR`), which the next run starts
without.
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
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
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
    "triton": "cozy_runtime.internal.triton_kernels:build",
}
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


class Canceled(Exception):
    """The request a wait served was canceled: the wait ends, and the build goes on."""


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


#: This process's builders and their reapers, by key digest: their exit is observed without a
#: clock, and a dead one's failure is on record before its key is read again.
_STARTED: dict[str, tuple[subprocess.Popen[bytes], threading.Thread]] = {}
_GUARD = threading.Lock()
#: How often a wait reads the state it waits on: resolution, never a verdict.
POLL_SECONDS = 0.5
#: Told the state a wait reads, each sample, named for its waiter: an executor reports it to
#: its machine as the run's progress.
waiting: Callable[[State], None] | None = None


def _sleep(seconds: float) -> bool:
    time.sleep(seconds)
    return False


#: How a request's executor holds each compile wait it makes, named for its waiter: around the
#: wait, giving the sleep its polls take, which wakes at the request's cancel and says whether
#: it came. The wait then ends `Canceled`, and its build goes on for later requests.
held: ContextVar[Callable[[str], AbstractContextManager[Callable[[float], bool]]]] = ContextVar(
    "cozy_compile_held", default=lambda waiter: contextlib.nullcontext(_sleep)
)


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
        process, reaper = _STARTED.get(job.key.digest) or (None, None)
        if process is not None and reaper is not None and process.poll() is not None:
            reaper.join()
        lock = store.claim(job.key) if process is None or process.poll() is not None else None
        if lock is not None:
            try:
                if store.entry(job.key) is None and _failure(job.key) is None:
                    store.set_state(job.key, None)  # a dead builder's last word, if any
                    _spawn(store, job, lock)
            finally:
                os.close(lock)
    current = status(store, job)
    return current if current.state != "pending" else State(job.key.kernel, "compiling")


def wait(store: kernel_cache.Store, job: Job, waiter: str = "") -> State:
    """`submit`, then its build's end: `ready`, or `failed` and why. Judged by measured progress
    alone (`liveness`): while any build on this machine moves, the wait goes on; when they have
    all stopped, against their own pace, it ends `failed` for this machine run, and the
    builder's owner ends the builder by the same rule. `waiting` hears each sample under
    `waiter`'s name (the kernel a request asked for), else the artifact's; `held` holds it."""
    state = submit(store, job)
    if state.state != "compiling":
        return state
    name = waiter or state.kernel
    pace, sampled = liveness.Pace(), float("-inf")
    with held.get()(name) as sleep:
        while state.state == "compiling":
            now = time.monotonic()
            if now - sampled >= liveness.SAMPLE_SECONDS:
                sampled = now
                if waiting is not None:
                    waiting(replace(state, kernel=name))
                pace.observe(_activity(store), now)
                if pace.wedged(liveness.noise_floor(), now):
                    verdict = pace.verdict(liveness.noise_floor(), now)
                    _record_failure(job.key, "compile_stalled", verdict)
                    return status(store, job)
            if sleep(POLL_SECONDS):
                raise Canceled(f"canceled waiting for {name} to compile; the compile goes on")
            state = submit(store, job)
    return state


def _activity(store: kernel_cache.Store) -> object:
    """The meter a wait is judged by: every live build's word in this machine's store, and
    the CPU and bytes its process tree has burned. None when no builder's burn is readable."""
    words: list[str] = []
    pids: set[int] = set()
    for namespace in (store.own, store.trusted):
        for path in sorted(namespace.glob("*/*.state.json")) if namespace is not None else ():
            with contextlib.suppress(OSError, ValueError, AttributeError):
                word = path.read_text()
                pid = json.loads(word).get("pid")
                words.append(word)
                if type(pid) is int:
                    pids.add(pid)
    try:
        tree = {member for pid in pids for member in (pid, *proctree.descendants(pid))}
    except proctree.ProcessTreeUnsupported:
        return None
    burns = [burned for member in tree if (burned := liveness.burn(member)) is not None]
    return (tuple(words), sum(burns)) if burns else None


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


def _spawn(store: kernel_cache.Store, job: Job, lock: int) -> None:
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
        env={**config.inherited_environment(), **job.env},
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

    reaper = threading.Thread(target=reap, name=f"kernel-builder-{job.key.kernel}", daemon=True)
    reaper.start()
    _STARTED[job.key.digest] = (process, reaper)
    # Its pid is on record before it can say anything itself: a wait meters it from birth.
    with contextlib.suppress(OSError):
        store.set_state(job.key, {"pid": process.pid, "progress": 0.0, "phase": "starting"})


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


def cpus() -> int:
    """Every CPU this process may use: its affinity and its cgroup's quota."""
    count = len(os.sched_getaffinity(0))
    quota = _read("/sys/fs/cgroup/cpu.max").split()
    if len(quota) == 2 and quota[0].isdigit() and quota[1].isdigit() and int(quota[1]):
        count = min(count, -(-int(quota[0]) // int(quota[1])))
    return count


def available_memory() -> int:
    """The memory this process may still take: the host's available, within its cgroup's."""
    memory = [_meminfo("MemAvailable")]
    limit, used = _read("/sys/fs/cgroup/memory.max"), _read("/sys/fs/cgroup/memory.current")
    if limit.isdigit() and used.isdigit():
        memory.append(int(limit) - int(used))
    return min(memory)


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _meminfo(field: str) -> int:
    found = re.search(rf"^{field}:\s+(\d+) kB", _read("/proc/meminfo"), re.MULTILINE)
    return int(found.group(1)) * 1024 if found else 0


# ----------------------------------------------------------------- the machine's compile ledger


def ledger(store: kernel_cache.Store) -> Path:
    """The machine's one compile ledger: in the trusted namespace when this process may write
    it (an executor under uid isolation may not, and keeps its own)."""
    shared = store.trusted is not None and os.access(store.trusted, os.W_OK)
    return ((store.trusted if shared else None) or store.own) / ".compile.json"


@contextlib.contextmanager
def _ledger(path: Path) -> Iterator[dict[str, Any]]:
    """The ledger, read and rewritten under its flock: `jobs` (pid -> admitted compiler job,
    pruned to live processes) and `budget`, the memory and CPUs the machine's compilers may
    take, measured again whenever no compiler job runs."""
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    fd = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            book = json.loads(path.read_text())
        except (OSError, ValueError):
            book = {}
        book["jobs"] = {pid: v for pid, v in dict(book.get("jobs") or {}).items() if _alive(pid)}
        if not book["jobs"]:
            book["budget"] = {"bytes": available_memory(), "cpus": cpus()}
        yield book
        scratch = path.with_suffix(f".{os.getpid()}")
        scratch.write_text(json.dumps(book))
        os.chmod(scratch, 0o666)
        os.replace(scratch, path)
    finally:
        os.close(fd)


def _alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (ValueError, ProcessLookupError):
        return False
    return True


def _admits(book: dict[str, Any], job_bytes: int) -> bool:
    """A compiler job runs when the CPUs and memory the ledger holds have room for it; the
    first always runs."""
    jobs, budget = book["jobs"].values(), book["budget"]
    held = sum(job["bytes"] for job in jobs)
    return not jobs or (
        len(jobs) < budget["cpus"]
        and held + job_bytes <= budget["bytes"]
        and available_memory() >= job_bytes
    )


def compiler_job(path: str, key: str, job_bytes: int, compiler: str) -> None:
    """A build's compiler, through the ledger at `path`: it waits until admitted, then this
    process becomes the compiler, on record until it exits. `sys.argv[1:]` are its arguments."""
    pid = str(os.getpid())
    with contextlib.suppress(OSError):  # an unwritable ledger admits every job
        while True:
            with _ledger(Path(path)) as book:
                if _admits(book, job_bytes):
                    book["jobs"][pid] = {"key": key, "bytes": job_bytes}
                    break
            time.sleep(POLL_SECONDS)
    os.execv(compiler, [compiler, *sys.argv[1:]])


def compilers(path: Path) -> set[int]:
    """The machine's admitted compiler jobs: a build waiting its turn is judged by theirs."""
    try:
        return {int(pid) for pid in json.loads(path.read_text()).get("jobs", {})}
    except (OSError, ValueError, AttributeError):
        return set()


def run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    report: Callable[[float], None],
    alongside: Callable[[], set[int]] = set,
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
        tree = {process.pid, *proctree.descendants(process.pid), *alongside()}
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

    try:
        report(0.0)
        module, name = BUILDERS[raw["kind"]].split(":")
        build = getattr(import_module(module), name)
        spec = {**raw["spec"], "ledger": str(ledger(store)), "key": key.digest}

        def fill(staging: Path) -> Mapping[str, object]:
            facts = dict(build(spec, staging, report))
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
