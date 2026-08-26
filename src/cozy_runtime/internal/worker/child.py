"""The disposable device executor, from the worker's side.

ONE ordinal over one execution context: `executor_incarnation`, the TRANSPORT fence, which
bumps on child respawn and rides `PlacementStatus.executor_generation`.

`readiness_epoch` is GONE — DELETED, not renamed (#482/#486b). It was a second counter for
in-place state invalidation (an orphan sweep + reload leaves the process alive and its state
worthless), and rev-2 answers that with the two facts that were actually wanted: the
placement's SERVING axis says what it can serve, and the worker-level `admission_generation`
says whether an offer is admissible. One fence, not two to keep consistent. `invalidate()`
still exists and still withdraws dispatchability; what it no longer does is mint an ordinal
nobody could reconcile against the other one.

Cancellation is cooperative then forceful and the WORKER owns both clocks: the grace
budget is a runtime constant here (`GRACE_SECONDS`), overridable by the operator through
`CancelAttempt.grace_ms`, and never an author knob. The forceful path kills the executor's
whole cgroup subtree, so even a detached grandchild cannot outlive the terminal that closed
its attempt.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import accel, proctree, spawn
from cozy_runtime.internal.seam import Channel, SeamError, listener
from cozy_runtime.internal.worker.observe import GIVE_UP_AFTER
from cozy_runtime.internal.worker.plan import PreparedModel

#: The grace budget: long enough for the longest uninterruptible kernel plus one loop step,
#: short enough that a caller's cancel is not a lifetime. A RUNTIME CONSTANT; `grace_ms` on
#: CancelAttempt is the operator override door, and no author surface reaches it.
GRACE_SECONDS = 3.0

#: How often the worker LOOKS while a spawned executor is dialling back. A sampling
#: cadence, not a bound: `spawn` waits while the child is alive and burning CPU, and this
#: only says how finely it can tell. Short because the answer is normally milliseconds away.
DIAL_SAMPLE_SECONDS = 0.5

#: A dialled-back child has already imported the executor and has no legitimate long work
#: before answering ``hello``.  Bound this read so a stale or malicious connector cannot
#: retain the lifecycle lock forever.
HELLO_SECONDS = 10.0

#: A driver allocation is allowed this long to disappear.  Both PRESENT and UNREADABLE
#: block a successor: only a measured ABSENT row proves the device ownership was returned.
RECLAIM_SECONDS = 10.0
RECLAIM_SAMPLE_SECONDS = 0.25


#: `sockaddr_un.sun_path` is 108 bytes on Linux, NUL included. A COZY_HOME deep enough to
#: pass it makes the executor's control socket unbindable, and the only symptom was a worker
#: that exited before advertising a plan (cl-013). The cause is measurable here, so it is
#: named here.
SUN_PATH_MAX = 108


def work(pid: int) -> int:
    """How much WORK a process has done, read from `/proc` by whoever is watching it.

    THE PROGRESS METER, and the only thing that separates a slow subject from a wedged one
    (cr-011). Two monotone counters summed: CPU nanoseconds burned, and bytes actually
    fetched from storage. Summing is exactly "either one moved", because neither ever
    falls — and both hands are needed, because the two ways to be busy and silent are to
    compute (a hash-bound conversion leg) and to wait on the disk (a cold-page-cache import
    of a multi-gigabyte extension). A meter with only the first hand kills the second.

    THIS PROCESS, threads included, and its unreaped children excluded — which is the right
    reading because the trampoline EXECS the executor, so the pid that is watched is the pid
    that does the work, and torch's own parallelism is threads.

    -1 is UNREADABLE, never a zero — an unreadable meter decides nothing, so it neither
    rescues a subject nor condemns it.
    """
    return proctree.progress_burn(pid)


def socket_path_refusal(path: str) -> str:
    """Why this control socket cannot exist, or "" when it can."""
    encoded = len(path.encode()) + 1
    if encoded <= SUN_PATH_MAX:
        return ""
    return (
        f"executor_socket_unbindable: the worker/executor control socket would be "
        f"{path!r}, which is {encoded} B including its terminator and the kernel's "
        f"`sockaddr_un.sun_path` holds {SUN_PATH_MAX} B. Nothing can bind it, so no executor "
        f"can ever dial back. Point COZY_HOME (or `serve --out`) at a shorter directory: "
        f"{encoded - SUN_PATH_MAX} B shorter is enough"
    )


class ExecutorGone(Exception):
    """The executor died. Always a fact about the incarnation, never about the attempt."""

    def __init__(self, detail: str, status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


@dataclass(slots=True)
class Executor:
    """One incarnation: a pid, a seam, and the facts it reported."""

    incarnation: int
    process: proctree.ProcessIdentity
    cgroup: proctree.CgroupScope
    channel: Channel
    seal_digest: str
    hello: dict[str, Any] = field(default_factory=dict)
    ready_plans: set[str] = field(default_factory=set)
    construction_key: str = ""
    prepared_binding_set: frozenset[str] = frozenset()
    prepared_model: PreparedModel | None = None
    purpose: str = "fresh"
    poisoned: str = ""
    spawned_at: float = 0.0
    ready_at: float = 0.0
    exit_status: int | None = None
    _calls: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def pid(self) -> int:
        return self.process.pid

    def alive(self) -> bool:
        # Observation must not reap.  Keeping an exited direct child as a zombie until
        # driver reclaim is proven prevents its PID from being reused underneath NVML's
        # raw-PID process table.
        state = proctree.process_state(self.process)
        return state is not None and state != "Z"

    def call(
        self,
        command: dict[str, Any],
        *,
        timeout: float | None = None,
        total_timeout: float | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_request: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """One command, its reply, and every frame in between.

        Two in-between kinds, and the difference between them is DURABILITY, not shape:

        * `progress` is the lossy narration. It is forwarded and never answered.
        * `request` is the DURABLE mid-attempt exchange (cr-009's job checkpoint): the
          executor blocks on it, this side makes the fact durable, and the answer travels
          back on the same channel. A durable fact never rides the lossy lane, so a
          checkpoint save cannot be shed the way a progress frame can.
        """
        # One stream has one in-flight transaction.  Preparation, probes and attempts can
        # originate on different worker lanes; serializing here prevents their frames and
        # replies from corrupting each other even if a caller violates the higher-level lane.
        with self._calls:
            name = command["cmd"]
            deadline = time.monotonic() + total_timeout if total_timeout is not None else None
            self.channel.send(command)
            while True:
                try:
                    remaining = None if deadline is None else max(deadline - time.monotonic(), 0)
                    frame = self.channel.recv(timeout, total_timeout=remaining)
                except SeamError as exc:
                    raise ExecutorGone(f"seam: {exc}") from exc
                if frame is None:
                    raise ExecutorGone(
                        f"the executor died during {name!r} (exit status {self.exit_status})",
                        self.exit_status,
                    )
                if frame.get("event") == "progress":
                    if on_progress is not None:
                        on_progress(frame)
                    continue
                if frame.get("event") == "request":
                    answer = (
                        on_request(frame)
                        if on_request is not None
                        else {
                            "ok": False,
                            "code": "no_durable_exchange",
                            "detail": f"this attempt installed no handler for "
                            f"{frame.get('kind')!r}",
                        }
                    )
                    self.channel.send({"event": "answer", "seq": frame.get("seq"), **answer})
                    continue
                if frame.get("reply") != name:
                    raise ExecutorGone(f"out-of-order reply {frame.get('reply')!r} for {name!r}")
                return frame

    # ------------------------------------------------------------------ signals

    def kill_group(self, sig: int = signal.SIGKILL) -> int:
        """Fault-injection hook: signal this exact executor group without retiring it."""
        try:
            known = proctree.freeze_process_group(self.process)
            for process in known:
                if proctree.process_state(process) != "Z":
                    with contextlib.suppress(ProcessLookupError):
                        proctree.signal_process(process, sig)
            return len(known)
        except (ProcessLookupError, PermissionError):
            return 0

    def close(self) -> None:
        self.channel.close()


@dataclass(frozen=True, slots=True)
class ReclaimEvidence:
    """What retirement proved before ownership was cleared."""

    exit_status: int
    device: accel.ProcessMemory


@dataclass(frozen=True, slots=True)
class OwnerRecord:
    """The exact kernel scope, provisionally or finally bound to its leader birth."""

    cgroup: proctree.CgroupScope
    process: proctree.ProcessIdentity | None = None
    members: tuple[proctree.ProcessIdentity, ...] = ()
    """Frozen pre-kill census, durable so reclaim proof survives worker death."""
    reclaimed: bool = False


class ExecutorSupervision:
    """Owns the worker's one disposable executor process and incarnation fence."""

    def __init__(
        self,
        *,
        root: Path,
        python: str,
        base_env: tuple[tuple[str, str], ...],
        cozy_home: Path,
        device_process: Callable[[int], accel.ProcessMemory] | None = None,
    ) -> None:
        self.root = root
        self.python = python
        self.base_env = base_env
        self.cozy_home = cozy_home
        self.root.mkdir(parents=True, exist_ok=True)
        self._cgroup_namespace = proctree.executor_cgroup_namespace(self.root)
        try:
            self._root_lease = proctree.acquire_worker_root_lease(self.root / "worker.lock")
        except BlockingIOError as exc:
            raise ExecutorGone(
                f"worker root {self.root} is already supervised by another live worker"
            ) from exc
        # The incarnation starts at 1 (02 §0): a worker that has not spawned yet already
        # carries a valid fence rather than a zero.
        self.incarnation = 1
        self.current: Executor | None = None
        self.spawns = 0
        self.socket_path = str(root / "executor.sock")
        self._listener: socket.socket | None = None
        self._lifecycle = threading.Lock()
        self._closed = False
        self._device_process = device_process
        self.on_invalidate: Callable[[str], None] | None = None
        """Called AFTER every invalidation, once this object's own state already says the
        executor cannot serve. The session installs the half it owns — dropping the
        advertised intake state — because readiness is a SESSION fact and the process
        worker has no business publishing one. See `invalidate`."""
        self.on_change: Callable[[Executor | None], None] | None = None
        """Session-owned projection updated at the same publication/clear boundary."""
        self._launches: queue.Queue[tuple[dict[str, Any], queue.Queue[Any]] | None] = queue.Queue()
        self._launcher: threading.Thread | None = None
        self._launcher_lock = threading.Lock()

    @property
    def _owner_path(self) -> Path:
        return self.root / "executor.owner"

    def _observe_processes(
        self, processes: set[proctree.ProcessIdentity]
    ) -> dict[int, accel.ProcessMemory]:
        """One driver-table sample over all owned births.

        Tests may inject the scalar reader; production takes one coherent NVML snapshot
        instead of launching one ``nvidia-smi`` process per PID per poll.
        """
        pids = {process.pid for process in processes}
        if self._device_process is not None:
            return {pid: self._device_process(pid) for pid in pids}
        return accel.process_memories(pids, accel.host_backend_family())

    def _owner_payload(self, owner: OwnerRecord) -> str:
        process = owner.process
        return json.dumps(
            {
                "cgroup": {
                    "relative_path": owner.cgroup.relative_path,
                    "inode": owner.cgroup.inode,
                },
                "process": (
                    {"pid": process.pid, "started_ticks": process.started_ticks}
                    if process is not None
                    else None
                ),
                "members": [
                    {"pid": member.pid, "started_ticks": member.started_ticks}
                    for member in owner.members
                ],
                "reclaimed": owner.reclaimed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _write_owner(self, owner: OwnerRecord) -> None:
        """Publish a scope before spawn, closing the crash-before-PID-record window."""
        temporary = self.root / f"executor.owner.tmp.{os.getpid()}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(self._owner_payload(owner))
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        try:
            # Publish without replacement.  A second worker pointed at the same root must
            # fail closed instead of erasing the first worker's ownership evidence.
            os.link(temporary, self._owner_path)
        except FileExistsError as exc:
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} appeared during spawn"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        self._fsync_root()

    def _bind_owner(
        self, provisional: OwnerRecord, process: proctree.ProcessIdentity
    ) -> OwnerRecord:
        if provisional.process is not None or self._read_owner() != provisional:
            raise ExecutorGone("executor cgroup ownership changed before leader binding")
        owner = OwnerRecord(provisional.cgroup, process, provisional.members)
        self._replace_owner(provisional, owner)
        return owner

    def _replace_owner(self, expected: OwnerRecord, owner: OwnerRecord) -> None:
        if self._read_owner() != expected:
            raise ExecutorGone(f"executor ownership changed while replacing {expected}")
        temporary = self.root / f"executor.owner.tmp.{os.getpid()}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(self._owner_payload(owner))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._owner_path)
            self._fsync_root()
        finally:
            temporary.unlink(missing_ok=True)

    def _read_owner(self) -> OwnerRecord | None:
        try:
            document = json.loads(self._owner_path.read_text())
            cgroup = proctree.CgroupScope(
                relative_path=str(document["cgroup"]["relative_path"]),
                inode=int(document["cgroup"]["inode"]),
            )
            raw_process = document["process"]
            process = (
                proctree.ProcessIdentity(
                    pid=int(raw_process["pid"]),
                    started_ticks=int(raw_process["started_ticks"]),
                )
                if raw_process is not None
                else None
            )
            members = tuple(
                sorted(
                    (
                        proctree.ProcessIdentity(
                            pid=int(member["pid"]),
                            started_ticks=int(member["started_ticks"]),
                        )
                        for member in document.get("members", [])
                    ),
                    key=lambda member: (member.pid, member.started_ticks),
                )
            )
            reclaimed = document["reclaimed"]
            if not isinstance(reclaimed, bool):
                raise TypeError("reclaimed must be a boolean")
        except FileNotFoundError:
            return None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} is unreadable; refusing "
                "to guess that the device is unowned"
            ) from exc
        if cgroup.inode <= 0 or not cgroup.relative_path.startswith("/"):
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} has invalid cgroup {cgroup}"
            )
        try:
            proctree.validate_executor_cgroup(cgroup, self._cgroup_namespace)
        except proctree.CgroupUnavailable as exc:
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} names an unsafe cgroup: {exc}"
            ) from exc
        if process is not None and (process.pid <= 0 or process.started_ticks <= 0):
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} has invalid identity {process}"
            )
        if len(set(members)) != len(members) or any(
            member.pid <= 0 or member.started_ticks <= 0 for member in members
        ):
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} has invalid member census"
            )
        return OwnerRecord(cgroup, process, members, reclaimed)

    def _clear_owner(self, expected: OwnerRecord) -> None:
        """Clear only the exact durable owner whose OS and driver absence was proved."""
        recorded = self._read_owner()
        if recorded is None:
            return
        if recorded != expected:
            raise ExecutorGone(
                f"refusing to clear executor ownership for {expected}; durable owner is {recorded}"
            )
        self._owner_path.unlink()
        self._fsync_root()

    def _fsync_root(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def owns(self, executor: Executor) -> bool:
        """Whether `executor` is still the exact live child this supervision owns."""
        with self._lifecycle:
            return self.current is executor and executor.alive() and not self._closed

    def cooperative_cancel(self, expected: Executor) -> bool:
        """Signal only the exact executor still owned at this lifecycle boundary."""
        with self._lifecycle:
            if self.current is not expected or self._closed:
                return False
            try:
                # The lifecycle lock excludes the only code allowed to reap this direct
                # child, so the checked PID cannot be reused before this signal is sent.
                proctree.signal_process(expected.process, signal.SIGUSR1)
                return True
            except ProcessLookupError:
                return False

    @contextlib.contextmanager
    def hold(self, expected: Executor) -> Any:
        """Keep one exact executor current across an acceptance commit."""
        with self._lifecycle:
            yield self.current is expected and expected.alive() and not self._closed

    # ------------------------------------------------------------------ spawn

    def _launch(self, **request: Any) -> spawn.Spawned:
        """`posix_spawn` from ONE thread that outlives every control stream.

        THE TRAMPOLINE'S CONTAINMENT SEAL IS THREAD-SCOPED. Linux delivers
        `PR_SET_PDEATHSIG` when the child's parent THREAD exits, not when the parent process
        does — and executor preparation runs on the accepted stream's own thread, because
        that is where the Directive arrives. So an owner reconnect, which ends the previous
        stream and its thread, delivered SIGKILL to a perfectly healthy executor: the
        running attempt died as ABANDONED / EXECUTOR_INVALIDATED and the worker rebuilt,
        for no reason but who happened to call `posix_spawn`. The seal is worth keeping —
        it is the containment guarantee — so the CALLER moves instead: every spawn is
        performed by this one thread, created once and alive for the life of the process.
        """
        with self._launcher_lock:
            if self._launcher is None:

                def serve_launches() -> None:
                    while True:
                        request = self._launches.get()
                        if request is None:
                            return
                        call, answer = request
                        try:
                            answer.put(("ok", spawn.spawn_executor(**call)))
                        except BaseException as exc:  # returned to the asking thread
                            answer.put(("raise", exc))

                self._launcher = threading.Thread(
                    target=serve_launches, daemon=True, name="executor-launcher"
                )
                self._launcher.start()
        answer: queue.Queue[Any] = queue.Queue()
        self._launches.put((request, answer))
        kind, value = answer.get()
        if kind == "raise":
            raise value
        return value  # type: ignore[no-any-return]

    def spawn(self, *, imposed: dict[str, str]) -> Executor:
        """EXEC-FIRST spawn of a fresh incarnation. Bumps BOTH ordinals: a fresh executor
        has no state, so every incarnation bump implies a readiness bump.

        THE DIAL-BACK IS WAITED OUT BY OBSERVATION, not by a clock. It used to be
        `DIAL_SECONDS = 60.0`, which got both ends wrong at once: a child that died on its
        first import still cost a full minute of silence before anyone said so, and a child
        still legitimately importing on a cold page cache past the minute was killed for it.
        The two facts that decide are the child's own: it is GONE (nothing will ever dial
        back — say so now), or it is alive and BURNING CPU (it is working, wait). Alive and
        burning nothing across `GIVE_UP_AFTER` samples is a stall, and gets a named refusal.
        """
        with self._lifecycle:
            if self._closed:
                raise ExecutorGone("executor supervision is closed")
            if self.current is not None:
                raise ExecutorGone(
                    f"spawn refused: incarnation {self.current.incarnation} pid "
                    f"{self.current.pid} is still owned; retire or replace that exact child"
                )
            return self._spawn_locked(imposed=imposed)

    def _spawn_locked(self, *, imposed: dict[str, str]) -> Executor:
        """Spawn while `_lifecycle` is held; publish only after a verified hello."""
        recorded = self._read_owner()
        if recorded is not None:
            raise ExecutorGone(
                f"spawn refused: durable executor owner {recorded} remains; restart "
                "reclamation must prove OS and driver absence first"
            )
        if self._listener is None:
            refusal = socket_path_refusal(self.socket_path)
            if refusal:
                raise ExecutorGone(refusal)
            Path(self.socket_path).unlink(missing_ok=True)
            self._listener = listener(self.socket_path)
        if self.spawns:
            # A RESPAWN bumps the transport fence, and with it the readiness fence: a fresh
            # executor has no state, so no incarnation bump can leave readiness unmoved.
            self.incarnation += 1
        self.spawns += 1
        imposed = {
            **imposed,
            # Backend-mandated laws come from the accelerator boundary (#450): on MPS this
            # forces PYTORCH_ENABLE_MPS_FALLBACK=0; on CUDA it is empty. The FAMILY is a
            # cheap platform fact (present-but-unqualified, #446).
            **accel.child_env_law(accel.host_backend_family()),
            "COZY_EXECUTOR_SOCKET": self.socket_path,
            "COZY_HOME": str(self.cozy_home),
        }
        started = time.perf_counter()
        spawned: spawn.Spawned | None = None
        process: proctree.ProcessIdentity | None = None
        owner: OwnerRecord | None = None
        conn: socket.socket | None = None
        try:
            try:
                owner = OwnerRecord(proctree.create_executor_cgroup(self._cgroup_namespace))
            except (proctree.CgroupUnavailable, proctree.ProcessTreeUnsupported) as exc:
                raise ExecutorGone(
                    f"executor containment unavailable before allocation: {exc}"
                ) from exc
            self._write_owner(owner)
            spawned = self._launch(
                python=self.python,
                module_argv=[
                    "-c",
                    "import sys; from cozy_runtime.internal.executor import main; "
                    "raise SystemExit(main(sys.argv[1:]))",
                    "--socket",
                    self.socket_path,
                    "--root",
                    str(self.root),
                ],
                base_env=self.base_env,
                imposed=imposed,
                cwd=str(self.root),
                cgroup=owner.cgroup,
            )
            process = proctree.process_identity(spawned.pid)
            owner = self._bind_owner(owner, process)
            conn = self._dial_back(process)
            executor = Executor(
                incarnation=self.incarnation,
                process=process,
                cgroup=owner.cgroup,
                channel=Channel(conn),
                seal_digest=spawned.seal_digest,
                spawned_at=started,
            )
            # Identity is checked before `current` can point at it: a stale connector is not
            # this spawn merely because it reached the shared listener.  Unlike cold import,
            # hello performs no long work, so a bounded read is a lifecycle requirement.
            executor.hello = executor.call(
                {"cmd": "hello"}, timeout=HELLO_SECONDS, total_timeout=HELLO_SECONDS
            )
            sealed = executor.hello.get("sealed")
            if (
                executor.hello.get("pid") != spawned.pid
                or executor.hello.get("ppid") != os.getpid()
                or executor.hello.get("pgid") != spawned.pid
                or not proctree.cgroup_contains(owner.cgroup, process)
                or not isinstance(sealed, dict)
                or spawn.seal_digest(sealed) != spawned.seal_digest
            ):
                raise ExecutorGone(
                    f"executor hello identity mismatch for incarnation {self.incarnation}: "
                    f"expected pid/group {spawned.pid}, parent {os.getpid()}, seal "
                    f"{spawned.seal_digest}; received pid {executor.hello.get('pid')}, "
                    f"parent {executor.hello.get('ppid')}"
                )
            executor.ready_at = time.perf_counter()
            self.current = executor
            if self.on_change is not None:
                self.on_change(executor)
            return executor
        except BaseException as exc:
            if conn is not None:
                with contextlib.suppress(OSError):
                    conn.close()
            if spawned is not None:
                try:
                    assert owner is not None
                    self._abort_spawn(spawned.pid, process, owner)
                except ExecutorGone as cleanup:
                    raise ExecutorGone(f"{exc}; {cleanup}") from exc
            elif owner is not None:
                try:
                    # `_write_owner` may have linked the durable record and then failed
                    # its directory fsync. Observe the actual boundary instead of trusting
                    # a local boolean set only after the call returned.
                    if self._read_owner() == owner:
                        self._reclaim_owner(owner, "spawn before launch")
                    else:
                        proctree.remove_executor_cgroup(owner.cgroup)
                except (ExecutorGone, proctree.CgroupUnavailable) as cleanup:
                    raise ExecutorGone(f"{exc}; {cleanup}") from exc
            raise

    def _abort_spawn(
        self, pid: int, process: proctree.ProcessIdentity | None, owner: OwnerRecord
    ) -> None:
        """Kill and prove reclaim of a provisional direct child never published."""
        if process is not None:
            owner, known, _ = self._stop_and_observe_reclaim(owner, "provisional spawn")
            self._reap_and_prove_absent(known, process)
            self._remove_scope_and_owner(owner)
            return

        # Birth capture itself failed.  The unreaped direct-child wait slot still pins this
        # numeric PID, so this is the one safe numeric signal in the lifecycle.
        try:
            reaped, _ = os.waitpid(pid, os.WNOHANG)
            if reaped:
                self._reclaim_owner(owner, "provisional spawn without a captured birth")
                return
            os.kill(pid, signal.SIGKILL)
        except (ChildProcessError, ProcessLookupError):
            self._reclaim_owner(owner, "provisional spawn without a captured birth")
            return
        deadline = time.monotonic() + RECLAIM_SECONDS
        while time.monotonic() < deadline:
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                self._reclaim_owner(owner, "provisional spawn without a captured birth")
                return
            if reaped:
                self._reclaim_owner(owner, "provisional spawn without a captured birth")
                return
            time.sleep(0.01)
        raise ExecutorGone(f"spawn cleanup could not reap provisional pid {pid}")

    def _dial_back(self, process: proctree.ProcessIdentity) -> socket.socket:
        """The accepted connection from a freshly spawned child, or a NAMED reason there is
        none. Three answers, and a clock is not one of them."""
        assert self._listener is not None
        pid = process.pid
        self._listener.settimeout(DIAL_SAMPLE_SECONDS)
        still, burned = 0, -1
        while True:
            try:
                conn, _ = self._listener.accept()
                return conn
            except TimeoutError:
                pass
            state = proctree.process_state(process)
            if state is None or state == "Z":
                # GONE. Nothing will ever dial back, and the old constant spent a full
                # minute not saying so — the reason is knowable the moment the child exits.
                raise ExecutorGone(
                    f"incarnation {self.incarnation} (pid {pid}) exited before dialling back",
                )
            still += 1
            burned, before = work(pid), burned
            if burned > before >= 0:
                still = 0  # importing, and an import on a cold page cache takes what it takes
            if still >= GIVE_UP_AFTER:
                raise ExecutorGone(
                    f"incarnation {self.incarnation} (pid {pid}) is alive and has burned no "
                    f"CPU across {still} observations while owing a dial-back: it is stalled, "
                    "not slow"
                )

    def invalidate(self, expected: Executor, why: str) -> bool:
        """In-place readiness invalidation: the PROCESS may survive, its state did not.

        THE ADVERTISEMENT GOES WITH IT (cr-007, cl-003). A device process that cannot serve
        must stop SAYING it can at the moment it stops, which is here: the ready-plan set
        empties and `on_invalidate` withdraws dispatchability before this call returns.
        Advertising READY until the rebuild started spent a coordinator's whole requeue
        budget on a worker that was about to be healthy.
        """
        with self._lifecycle:
            return self._invalidate_locked(expected, why)

    def _invalidate_locked(self, expected: Executor, why: str) -> bool:
        if self.current is not expected:
            return False
        expected.ready_plans.clear()
        expected.poisoned = expected.poisoned or why
        if self.on_invalidate is not None:
            self.on_invalidate(why)
        return True

    def _stop_and_observe_reclaim(
        self, owner: OwnerRecord, why: str
    ) -> tuple[OwnerRecord, set[proctree.ProcessIdentity], accel.ProcessMemory]:
        """Kill the exact cgroup subtree, then retain births until driver absence."""
        if owner.reclaimed:
            raise ExecutorGone(f"cannot reclaim already-reclaimed owner {owner}")
        known = set(owner.members)
        if owner.process is not None:
            known.add(owner.process)
        try:
            known.update(proctree.freeze_executor_cgroup(owner.cgroup, seconds=RECLAIM_SECONDS))
            members = tuple(sorted(known, key=lambda item: (item.pid, item.started_ticks)))
            if members != owner.members:
                expanded = OwnerRecord(owner.cgroup, owner.process, members)
                self._replace_owner(owner, expanded)
                owner = expanded
            proctree.kill_frozen_executor_cgroup(owner.cgroup, seconds=RECLAIM_SECONDS)
        except (proctree.CgroupUnavailable, TimeoutError) as exc:
            raise ExecutorGone(
                f"could not reclaim exact executor cgroup {owner.cgroup} ({why}): {exc}; "
                "ownership retained and successor spawn refused"
            ) from exc

        deadline = time.monotonic() + RECLAIM_SECONDS
        observations: dict[int, accel.ProcessMemory] = {}
        while True:
            states = {member.pid: proctree.process_state(member) for member in known}
            observations = self._observe_processes(known)
            if all(state in {None, "Z"} for state in states.values()) and all(
                item.state == "absent" for item in observations.values()
            ):
                leader_device = (
                    observations[owner.process.pid]
                    if owner.process is not None
                    else accel.ProcessMemory("absent", 0)
                )
                return owner, known, leader_device
            if time.monotonic() >= deadline:
                raise ExecutorGone(
                    f"could not prove reclaim of {owner.cgroup} ({why}): "
                    f"process_states={states}, device_states="
                    f"{ {pid: item.state for pid, item in observations.items()} }; "
                    "ownership retained and successor spawn refused"
                )
            time.sleep(RECLAIM_SAMPLE_SECONDS)

    @staticmethod
    def _reap_and_prove_absent(
        known: set[proctree.ProcessIdentity], leader: proctree.ProcessIdentity | None
    ) -> int:
        """Reap subreaper-owned zombies, then prove every captured birth absent."""
        deadline = time.monotonic() + RECLAIM_SECONDS
        leader_status = -1
        while True:
            for process in known:
                status = proctree.reap_child(process)
                if process == leader and status is not None:
                    leader_status = status
            states = {process.pid: proctree.process_state(process) for process in known}
            if all(state is None for state in states.values()):
                return leader_status
            if time.monotonic() >= deadline:
                raise ExecutorGone(
                    f"could not reap executor subtree "
                    f"{leader.pid if leader is not None else 'without leader'}: "
                    f"process_states={states}; "
                    "ownership retained and successor spawn refused"
                )
            time.sleep(0.01)

    def _remove_scope_and_owner(self, owner: OwnerRecord) -> None:
        """Durably mark reclaim before removing its exact empty kernel scope."""
        reclaimed = OwnerRecord(owner.cgroup, owner.process, owner.members, True)
        self._replace_owner(owner, reclaimed)
        try:
            if proctree.executor_cgroup_exists(reclaimed.cgroup):
                proctree.remove_executor_cgroup(reclaimed.cgroup)
        except proctree.CgroupUnavailable as exc:
            raise ExecutorGone(f"could not remove reclaimed executor cgroup: {exc}") from exc
        self._clear_owner(reclaimed)

    def _reclaim_owner(self, owner: OwnerRecord, why: str) -> ReclaimEvidence:
        owner, known, device = self._stop_and_observe_reclaim(owner, why)
        status = self._reap_and_prove_absent(known, owner.process)
        self._remove_scope_and_owner(owner)
        return ReclaimEvidence(status, device)

    def retire_current(self, expected: Executor, why: str) -> ReclaimEvidence | None:
        """Withdraw, kill, and prove reclaim of exactly `expected`.

        A stale caller never acts on a successor. The owned handle is retained if reap
        fails, which also makes every later spawn refuse: losing the handle and claiming
        the slot is free was the process leak behind cr-024.
        """
        with self._lifecycle:
            return self._retire_locked(expected, why)

    def _retire_locked(self, expected: Executor, why: str) -> ReclaimEvidence | None:
        if self.current is not expected:
            return None
        self._invalidate_locked(expected, why)
        owner = self._read_owner()
        if (
            owner is None
            or owner.cgroup != expected.cgroup
            or owner.process != expected.process
            or owner.reclaimed
        ):
            raise ExecutorGone(
                f"durable executor owner does not match incarnation {expected.incarnation} "
                f"pid {expected.pid}: {owner}"
            )
        evidence = self._reclaim_owner(owner, why)
        expected.exit_status = evidence.exit_status

        expected.close()
        self.current = None
        if self.on_change is not None:
            self.on_change(None)
        return evidence

    def replace(self, expected: Executor, *, imposed: dict[str, str], why: str) -> Executor:
        """Retire one exact incarnation and publish at most one successor."""
        with self._lifecycle:
            if self.current is not expected:
                raise ExecutorGone(
                    f"replacement refused: incarnation {expected.incarnation} pid "
                    f"{expected.pid} is no longer the owned executor; re-evaluate current state"
                )
            self._retire_locked(expected, why)
            return self._spawn_locked(imposed=imposed)

    # ------------------------------------------------------------------ orphans

    def sweep_orphans(self) -> list[int]:
        """Reclaim the one exact durable scope left by a previous worker birth."""
        with self._lifecycle:
            return self._sweep_orphans_locked("worker restart")

    def _sweep_orphans_locked(self, why: str) -> list[int]:
        owner = self._read_owner()
        if owner is None:
            return []
        if owner.reclaimed:
            try:
                if proctree.executor_cgroup_exists(owner.cgroup):
                    proctree.remove_executor_cgroup(owner.cgroup)
            except proctree.CgroupUnavailable as exc:
                raise ExecutorGone(
                    f"could not finish durable reclaimed-scope cleanup: {exc}"
                ) from exc
            self._clear_owner(owner)
            return []
        was_present = owner.process is not None and proctree.same_process(owner.process)
        self._reclaim_owner(owner, why)
        return [owner.process.pid] if was_present and owner.process is not None else []

    def close(self) -> None:
        launcher: threading.Thread | None = None
        with self._lifecycle:
            if self._closed:
                return
            executor = self.current
            if executor is not None:
                self._retire_locked(executor, "worker shutdown")
            else:
                self._sweep_orphans_locked("worker shutdown")
            if self._listener is not None:
                self._listener.close()
                Path(self.socket_path).unlink(missing_ok=True)
                self._listener = None
            self._closed = True
            launcher = self._launcher
            self._launcher = None
            if launcher is not None:
                self._launches.put(None)
        if launcher is not None:
            launcher.join()
        proctree.release_worker_root_lease(self._root_lease)
