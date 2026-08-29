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
`CancelAttempt.grace_ms`, and never an author knob. Writable cgroup v2 remains the kernel
boundary; a strictly proven credential-unwritable cgroup-v1 container uses the narrower
first-party process-tree backend described in :mod:`cozy_runtime.internal.proctree`.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import secrets
import shutil
import signal
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import accel, jit_cache, proctree, spawn
from cozy_runtime.internal.child_env import EXECUTOR_SCOPE_ENV
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
    scope: proctree.ExecutorScope
    channel: Channel
    seal_digest: str
    hello: dict[str, Any] = field(default_factory=dict)
    ready_plans: set[str] = field(default_factory=set)
    construction_key: str = ""
    prepared_binding_set: frozenset[str] = frozenset()
    prepared_model: PreparedModel | None = None
    reserved_for_job: bool = False
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

    def close(self) -> None:
        self.channel.close()


@dataclass(frozen=True, slots=True)
class ReclaimEvidence:
    """What retirement proved before ownership was cleared."""

    exit_status: int
    device: accel.ProcessMemory


@dataclass(frozen=True, slots=True)
class OwnerRecord:
    """The exact local scope, provisionally or finally bound to its leader birth."""

    scope: proctree.ExecutorScope
    process: proctree.ProcessIdentity | None = None
    members: tuple[proctree.ProcessIdentity, ...] = ()
    """Frozen pre-kill census, durable so reclaim proof survives worker death."""


class ExecutorSupervision:
    """Owns the worker's one disposable executor process and incarnation fence."""

    def __init__(
        self,
        *,
        root: Path,
        python: str,
        base_env: tuple[tuple[str, str], ...],
        cozy_home: Path,
        executor_uid: int = -1,
        executor_gid: int = -1,
        jit_pod_scope: str = "",
        scratch_root: Path | None = None,
        device_process: Callable[[int], accel.ProcessMemory] | None = None,
    ) -> None:
        self.root = root
        self.python = python
        self.overlay = ""
        """The endpoint-only import root appended after image-owned packages."""
        self.overlay_content_digest = ""
        self.jit_cache = ""
        """One worker-boot/content cache path, deleted after worker shutdown."""
        self.base_env = base_env
        self.cozy_home = cozy_home
        self.executor_uid = executor_uid
        self.executor_gid = executor_gid
        self.jit_pod_scope = jit_pod_scope or f"local-{secrets.token_hex(16)}"
        self.scratch_root = scratch_root
        self.root.mkdir(parents=True, exist_ok=True)
        if self.isolated:
            # Group execute admits the fixed executor to the named socket/cancel paths but
            # not to list or read the worker's journal, owner record, or credentials.
            os.chown(self.root, 0, self.executor_gid)
            self.root.chmod(0o710)
        self._scope_namespace = proctree.executor_scope_namespace(self.root)
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
        self.cancel_path = root / "executor.cancel"
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
        self.on_exit: Callable[[Executor], None] | None = None
        """Session-owned reaction to an exact current child becoming waitable.

        The watcher never reaps.  It reports the exact :class:`Executor` handle so the
        session can invalidate only that incarnation and serialize any rebuild with device
        work; a stale watcher has no authority over a successor.
        """
        self._launches: queue.Queue[tuple[dict[str, Any], queue.Queue[Any]] | None] = queue.Queue()
        self._launcher: threading.Thread | None = None
        self._launcher_lock = threading.Lock()

    @property
    def isolated(self) -> bool:
        return self.executor_uid >= 0 and self.executor_gid >= 0

    def grant_directory(self, path: Path, generation: int) -> None:
        """Hand one already-populated attempt/scratch directory to the current child."""

        if not self.isolated:
            return
        current = self.current
        if current is None or current.incarnation != generation:
            raise ExecutorGone(
                f"cannot grant {path}: executor generation {generation} is not current"
            )
        os.chown(path, self.executor_uid, self.executor_gid)
        path.chmod(0o700)

    def generation_scratch(self, generation: int) -> Path:
        if self.scratch_root is None:
            raise ExecutorGone("executor scratch root is absent")
        if not self.isolated:
            return self.scratch_root
        return self.scratch_root / f"generation-{generation}"

    def _prepare_generation_scratch(self) -> None:
        if not self.isolated:
            return
        if self.scratch_root is None:
            raise ExecutorGone("isolated executor scratch root is absent")
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        os.chown(self.scratch_root, 0, 0)
        self.scratch_root.chmod(0o711)
        # No old executor exists while _spawn_locked holds the lifecycle lock. Remove every
        # retired generation before making the successor reachable; a fixed uid therefore
        # cannot read a previous incarnation's scratch even though the numeric uid is reused.
        for entry in self.scratch_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                raise ExecutorGone(f"unexpected entry in executor scratch root: {entry}")
            shutil.rmtree(entry)
        generation = self.generation_scratch(self.incarnation)
        generation.mkdir(mode=0o700)
        os.chown(generation, self.executor_uid, self.executor_gid)

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
        scope = owner.scope
        return json.dumps(
            {
                "scope": (
                    {
                        "backend": "cgroup_v2",
                        "relative_path": scope.relative_path,
                        "inode": scope.inode,
                    }
                    if isinstance(scope, proctree.CgroupScope)
                    else {"backend": "process_tree", "token": scope.token, "uid": scope.uid}
                ),
                "process": (
                    {"pid": process.pid, "started_ticks": process.started_ticks}
                    if process is not None
                    else None
                ),
                "members": [
                    {"pid": member.pid, "started_ticks": member.started_ticks}
                    for member in owner.members
                ],
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
            raise ExecutorGone("executor scope ownership changed before leader binding")
        owner = OwnerRecord(provisional.scope, process, provisional.members)
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
            if not isinstance(document, dict) or set(document) != {"scope", "process", "members"}:
                raise TypeError("owner fields differ")
            raw_scope = document["scope"]
            if not isinstance(raw_scope, dict):
                raise TypeError("scope is not an object")
            if raw_scope.get("backend") == "cgroup_v2" and set(raw_scope) == {
                "backend",
                "relative_path",
                "inode",
            }:
                scope: proctree.ExecutorScope = proctree.CgroupScope(
                    relative_path=str(raw_scope["relative_path"]),
                    inode=int(raw_scope["inode"]),
                )
            elif raw_scope.get("backend") == "process_tree" and set(raw_scope) == {
                "backend",
                "token",
                "uid",
            }:
                scope = proctree.ProcessTreeScope(str(raw_scope["token"]), int(raw_scope["uid"]))
            else:
                raise TypeError("scope fields differ")
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
        except FileNotFoundError:
            return None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} is unreadable; refusing "
                "to guess that the device is unowned"
            ) from exc
        if isinstance(scope, proctree.CgroupScope) and (
            scope.inode <= 0 or not scope.relative_path.startswith("/")
        ):
            raise ExecutorGone(f"executor ownership record {self._owner_path} has invalid {scope}")
        try:
            proctree.validate_executor_scope(scope, self._scope_namespace)
        except proctree.ContainmentUnavailable as exc:
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} names an unsafe scope: {exc}"
            ) from exc
        if isinstance(scope, proctree.ProcessTreeScope) and scope.uid != self.executor_uid:
            raise ExecutorGone(
                f"executor ownership record {self._owner_path} names uid {scope.uid}, "
                f"want {self.executor_uid}"
            )
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
        return OwnerRecord(scope, process, members)

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

    def cooperative_cancel(self, expected: Executor, request_id: str) -> bool:
        """Publish an attempt-keyed marker for only the exact executor still owned."""
        with self._lifecycle:
            if self.current is not expected or self._closed:
                return False
            temporary = self.root / f"executor.cancel.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                temporary.write_text(request_id)
                os.replace(temporary, self.cancel_path)
                return True
            except OSError:
                return False
            finally:
                temporary.unlink(missing_ok=True)

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

    def use_overlay(self, python: str, overlay: str, content_digest: str) -> bool:
        """Point the executor at one base interpreter, overlay, and node-local JIT scope."""

        if not re.fullmatch(r"sha256:[0-9a-f]{64}", content_digest):
            raise ExecutorGone(f"invalid endpoint content digest for JIT cache: {content_digest!r}")
        if (self.python, self.overlay, self.overlay_content_digest) == (
            python,
            overlay,
            content_digest,
        ):
            return False
        self.python = python
        self.overlay = overlay
        self.overlay_content_digest = content_digest
        return True

    def _prepare_jit_cache(self) -> dict[str, str]:
        """Create/reuse one worker-boot/content cache without publishing its bytes."""

        if not self.overlay_content_digest:
            return {}
        try:
            prepared = jit_cache.scope(
                self.cozy_home,
                self.overlay_content_digest,
                pod_scope=self.jit_pod_scope,
                uid=self.executor_uid,
                gid=self.executor_gid,
            )
            self.jit_cache = str(prepared.root)
            return prepared.environment
        except jit_cache.JITCacheRefusal as exc:
            raise ExecutorGone(str(exc)) from exc

    def _clear_jit_pod(self) -> None:
        try:
            jit_cache.remove_pod(self.cozy_home, self.jit_pod_scope)
            self.jit_cache = ""
        except jit_cache.JITCacheRefusal as exc:
            raise ExecutorGone(str(exc)) from exc

    def _spawn_locked(self, *, imposed: dict[str, str]) -> Executor:
        """Spawn while `_lifecycle` is held; publish only after a verified hello."""
        recorded = self._read_owner()
        if recorded is not None:
            raise ExecutorGone(
                f"spawn refused: durable executor owner {recorded} remains; restart "
                "reclamation must prove OS and driver absence first"
            )
        # Cancellation is scoped to one executor generation as well as one attempt. A
        # marker from the retired generation cannot become authority in its successor.
        self.cancel_path.unlink(missing_ok=True)
        if self._listener is None:
            refusal = socket_path_refusal(self.socket_path)
            if refusal:
                raise ExecutorGone(refusal)
            Path(self.socket_path).unlink(missing_ok=True)
            self._listener = listener(self.socket_path)
            if self.isolated:
                os.chown(self.socket_path, 0, self.executor_gid)
                os.chmod(self.socket_path, 0o660)
        if self.spawns:
            # A RESPAWN bumps the transport fence, and with it the readiness fence: a fresh
            # executor has no state, so no incarnation bump can leave readiness unmoved.
            self.incarnation += 1
        self._prepare_generation_scratch()
        self.spawns += 1
        started = time.perf_counter()
        spawned: spawn.Spawned | None = None
        process: proctree.ProcessIdentity | None = None
        owner: OwnerRecord | None = None
        conn: socket.socket | None = None
        exit_published: threading.Event | None = None
        try:
            imposed = {
                **imposed,
                **self._prepare_jit_cache(),
                # Backend-mandated laws come from the accelerator boundary (#450): on MPS this
                # forces PYTORCH_ENABLE_MPS_FALLBACK=0; on CUDA it is empty. The FAMILY is a
                # cheap platform fact (present-but-unqualified, #446).
                **accel.child_env_law(accel.host_backend_family()),
                "COZY_EXECUTOR_SOCKET": self.socket_path,
                "COZY_HOME": str(self.cozy_home),
            }
            try:
                owner = OwnerRecord(
                    proctree.create_executor_scope(
                        self._scope_namespace,
                        self.executor_uid if self.isolated else None,
                    )
                )
            except (proctree.ContainmentUnavailable, proctree.ProcessTreeUnsupported) as exc:
                raise ExecutorGone(
                    f"executor containment unavailable before allocation: {exc}"
                ) from exc
            if isinstance(owner.scope, proctree.ProcessTreeScope):
                imposed[EXECUTOR_SCOPE_ENV] = owner.scope.token
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
                    *(["--overlay", self.overlay] if self.overlay else []),
                ],
                base_env=self.base_env,
                imposed=imposed,
                cwd=str(self.root),
                scope=owner.scope,
                uid=self.executor_uid,
                gid=self.executor_gid,
            )
            process = proctree.process_identity(spawned.pid)
            if isinstance(owner.scope, proctree.ProcessTreeScope):
                proctree.wait_process_stopped(process, seconds=RECLAIM_SECONDS)
                if not proctree.process_scope_contains(owner.scope, process):
                    raise ExecutorGone(
                        f"process-tree trampoline {process.pid} stopped outside its exact scope"
                    )
            owner = self._bind_owner(owner, process)
            if isinstance(owner.scope, proctree.ProcessTreeScope):
                proctree.signal_process(process, signal.SIGCONT)
            conn = self._dial_back(process)
            executor = Executor(
                incarnation=self.incarnation,
                process=process,
                scope=owner.scope,
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
            contained = (
                proctree.cgroup_contains(owner.scope, process)
                if isinstance(owner.scope, proctree.CgroupScope)
                else proctree.process_scope_contains(owner.scope, process)
            )
            if (
                executor.hello.get("pid") != spawned.pid
                or executor.hello.get("ppid") != os.getpid()
                or executor.hello.get("pgid") != spawned.pid
                or not contained
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
            exit_published = self._start_exit_watcher(executor)
            self.current = executor
            if self.on_change is not None:
                self.on_change(executor)
            exit_published.set()
            return executor
        except BaseException as exc:
            if exit_published is not None:
                # Release a watcher installed before publication.  `current` either still
                # names this exact child or never did; the session callback rechecks it.
                exit_published.set()
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
                        proctree.remove_executor_scope(owner.scope)
                except (ExecutorGone, proctree.ContainmentUnavailable) as cleanup:
                    raise ExecutorGone(f"{exc}; {cleanup}") from exc
            raise

    def _start_exit_watcher(self, executor: Executor) -> threading.Event:
        """Install one exact-child observer before ``current`` is published.

        Opening the pidfd is synchronous, so a worker never advertises an executor it cannot
        observe.  The thread waits for publication before notifying the session, closing the
        only race where a very short-lived child could exit between hello and ``current``.
        """

        observer = proctree.observe_process_exit(executor.process)
        published = threading.Event()

        def wait_for_exit() -> None:
            try:
                observer.wait()
            finally:
                observer.close()
            published.wait()
            callback = self.on_exit
            if callback is not None:
                callback(executor)

        try:
            threading.Thread(
                target=wait_for_exit,
                daemon=True,
                name=f"executor-exit-{executor.incarnation}",
            ).start()
        except BaseException:
            observer.close()
            raise
        return published

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
        Advertising READY until the rebuild started spent a RecordOwner's whole requeue
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
        """Stop one exact scope, persist its census, then prove OS and driver absence."""
        known = set(owner.members)
        if owner.process is not None:
            known.add(owner.process)
        try:
            if isinstance(owner.scope, proctree.CgroupScope):
                if proctree.executor_scope_exists(owner.scope):
                    known.update(
                        proctree.freeze_executor_cgroup(owner.scope, seconds=RECLAIM_SECONDS)
                    )
            else:
                known.update(
                    proctree.freeze_process_tree(
                        owner.scope,
                        owner.process,
                        known,
                        worker_pid=os.getpid(),
                        seconds=RECLAIM_SECONDS,
                    )
                )
            members = tuple(sorted(known, key=lambda item: (item.pid, item.started_ticks)))
            if members != owner.members:
                expanded = OwnerRecord(owner.scope, owner.process, members)
                self._replace_owner(owner, expanded)
                owner = expanded
            if isinstance(owner.scope, proctree.CgroupScope):
                if proctree.executor_scope_exists(owner.scope):
                    proctree.kill_frozen_executor_cgroup(owner.scope, seconds=RECLAIM_SECONDS)
            else:
                proctree.kill_process_tree(known)
        except (proctree.ContainmentUnavailable, TimeoutError) as exc:
            raise ExecutorGone(
                f"could not reclaim exact executor scope {owner.scope} ({why}): {exc}; "
                "ownership retained and successor spawn refused"
            ) from exc

        deadline = time.monotonic() + RECLAIM_SECONDS
        observations: dict[int, accel.ProcessMemory] = {}
        while True:
            states = {member.pid: proctree.process_state(member) for member in known}
            observations = self._observe_processes(known)
            try:
                token_leftovers = (
                    proctree.process_scope_processes(owner.scope)
                    if isinstance(owner.scope, proctree.ProcessTreeScope)
                    else set()
                )
            except proctree.ContainmentUnavailable as exc:
                raise ExecutorGone(
                    f"could not prove process-tree scope empty ({why}): {exc}; "
                    "ownership retained and successor spawn refused"
                ) from exc
            if (
                all(state in {None, "Z"} for state in states.values())
                and all(item.state == "absent" for item in observations.values())
                and not token_leftovers
            ):
                leader_device = (
                    observations[owner.process.pid]
                    if owner.process is not None
                    else accel.ProcessMemory("absent", 0)
                )
                return owner, known, leader_device
            if time.monotonic() >= deadline:
                raise ExecutorGone(
                    f"could not prove reclaim of {owner.scope} ({why}): "
                    f"process_states={states}, device_states="
                    f"{ {pid: item.state for pid, item in observations.items()} }; "
                    f"token_leftovers={sorted(p.pid for p in token_leftovers)}; "
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
        """Remove an empty kernel scope when present, then atomically clear its owner."""
        try:
            if isinstance(owner.scope, proctree.CgroupScope):
                if proctree.executor_scope_exists(owner.scope):
                    proctree.remove_executor_scope(owner.scope)
            else:
                # Re-census immediately at the clear boundary. A same-UID process can
                # deliberately replay the non-secret token; that documented first-party
                # residual must still fail closed when it is observable.
                proctree.remove_executor_scope(owner.scope)
        except proctree.ContainmentUnavailable as exc:
            raise ExecutorGone(f"could not remove reclaimed executor scope: {exc}") from exc
        self._clear_owner(owner)

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
        if owner is None or owner.scope != expected.scope or owner.process != expected.process:
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
        if isinstance(owner.scope, proctree.ProcessTreeScope):
            try:
                proctree.validate_process_tree_host()
            except (proctree.ContainmentUnavailable, proctree.ProcessTreeUnsupported) as exc:
                raise ExecutorGone(
                    f"cannot recover process-tree executor ownership on this host: {exc}; "
                    "ownership retained and successor spawn refused"
                ) from exc
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
            self._clear_jit_pod()
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
