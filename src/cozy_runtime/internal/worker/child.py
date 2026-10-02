"""The disposable device executor, from the worker's side.

ONE ordinal over one execution context: the executor EPOCH, the TRANSPORT fence, which bumps
on child respawn and rides `PlacementStatus.executor_epoch`.

`readiness_epoch` is GONE — DELETED, not renamed (#482/#486b). It was a second counter for
in-place state invalidation (an orphan sweep + reload leaves the process alive and its state
worthless), and rev-2 answers that with the two facts that were actually wanted: the
placement's SERVING axis says what it can serve, and the worker-level `admission_epoch`
says whether an offer is admissible. One fence, not two to keep consistent. `invalidate()`
still exists and still withdraws dispatchability; what it no longer does is mint an ordinal
nobody could reconcile against the other one.

Cancellation is cooperative then forceful. The executor stops at its next safe point and
stays warm; the WORKER escalates only when the attempt has stopped moving against its own
measured frame cadence (`AttemptRecord.cancel_stall`), never on a flat grace.

Writable cgroup v2 remains the kernel boundary; a strictly proven credential-unwritable
container uses the narrower first-party process-tree backend described in
:mod:`cozy_runtime.internal.proctree`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import secrets
import signal
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict, Unpack

from cozy_runtime import __version__ as RUNTIME_VERSION
from cozy_runtime.author._executor_requests import (
    Answer,
    BudgetCell,
    Handler,
    Reply,
    Request,
    refuse,
    respond,
)
from cozy_runtime.internal import (
    accel,
    budget_cell,
    executor_commands,
    jit_cache,
    liveness,
    proctree,
    spawn,
)
from cozy_runtime.internal.child_env import (
    ALLOWLIST,
    EXECUTOR_SCOPE_ENV,
    FLASH_ATTENTION_CACHE_ENABLED_ENV,
)
from cozy_runtime.internal.executor_commands import Command, Hello
from cozy_runtime.internal.seam import (
    EXECUTOR_PROTOCOL_REVISION,
    Channel,
    SeamError,
    listener,
)
from cozy_runtime.internal.worker.plan import PreparedModel

_LOG = logging.getLogger(__name__)

#: How often the worker LOOKS while a spawned executor is dialling back. A sampling
#: cadence, not a bound: `spawn` waits while the child is alive, and this only says how
#: promptly it observes either a connection or process death.
DIAL_SAMPLE_SECONDS = 0.5

#: A dialled-back child has already imported the executor and has no legitimate long work
#: before answering ``hello``.  Bound this read so a stale or malicious connector cannot
#: retain the lifecycle lock forever.
HELLO_SECONDS = 10.0

#: How often reclaim RE-READS the kernel and the driver while a killed executor is torn
#: down. A sampling cadence, not a bound: both PRESENT and UNREADABLE block a successor, and
#: only a measured ABSENT row proves the device ownership was returned — the wait ends on
#: that answer, or when the whole observation (process states, resident bytes, device rows)
#: has provably stopped moving (`liveness.Pace`, against the runtime's one noise floor). A
#: 100 GB context whose teardown takes a minute is moving the whole time and is waited for.
RECLAIM_SAMPLE_SECONDS = 0.25

#: The weight plane copies to the device from read-only file mappings.
SIGBUS_MEANS = (
    ": a read fault on mapped weights (a disk read error, or a store object changed under "
    "its mapping)"
)


def work(pid: int) -> int:
    """Monotone CPU and byte work observed for one process, or -1 if unreadable."""
    return proctree.progress_burn(pid)


#: `sockaddr_un.sun_path` is 108 bytes on Linux, NUL included. A COZY_HOME deep enough to
#: pass it makes the executor's control socket unbindable, and the only symptom was a worker
#: that exited before advertising a plan (cl-013). The cause is measurable here, so it is
#: named here.
SUN_PATH_MAX = 108


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
    """The executor died. Always a fact about the epoch, never about the attempt."""

    def __init__(self, detail: str, status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


def death(status: int | None, name: object) -> str:
    """An executor's end during `name`, from its wait status; "died" when none was readable."""
    if status is None or status < 0:
        return f"the executor died during {name!r}"
    code = os.waitstatus_to_exitcode(status)
    if code >= 0:
        return f"the executor exited with status {code} during {name!r}"
    try:
        cause = signal.Signals(-code).name
    except ValueError:
        cause = f"signal {-code}"
    killed = f"the executor was killed by {cause} during {name!r}"
    return killed + SIGBUS_MEANS if code == -signal.SIGBUS else killed


class ExecutorProtocolMismatch(ExecutorGone):
    """Only this executor is refused before dispatch when its IPC is unsupported."""

    code = "executor_protocol_incompatible"


def require_executor_protocol(hello: dict[str, Any]) -> None:
    revision = hello.get("executor_protocol_revision")
    if (
        type(revision) is int
        and revision == EXECUTOR_PROTOCOL_REVISION
        and isinstance(hello.get("native_interfaces"), dict)
    ):
        return
    # The executor states its revision and native interfaces in its hello (cozy-runtime
    # 0.18.51 on). Say which side to update.
    executor = hello.get("runtime_version", "unknown")
    remedy = (
        "update the rental's Runtime: cozy rental update <rental>"
        if type(revision) is int and revision > EXECUTOR_PROTOCOL_REVISION
        else "publish the package against cozy-runtime>=0.18.51 (its lock pins "
        f"cozy-runtime {executor}), then run it again"
    )
    raise ExecutorProtocolMismatch(
        f"executor_protocol_incompatible: the package environment runs cozy-runtime "
        f"{executor} (executor protocol {revision!r}) and this worker runs {RUNTIME_VERSION} "
        f"(protocol {EXECUTOR_PROTOCOL_REVISION}); {remedy}."
    )


@dataclass(slots=True)
class LoadedConstruction:
    """One construction the executor built: what it resolved and whom it serves."""

    key: str
    prepared_model: PreparedModel
    #: every model parameter name it is bound under (h3a-018)
    parameters: frozenset[str]
    #: the entrypoint bindings it serves
    bindings: frozenset[str]
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Executor:
    """One epoch: a pid, a seam, and the facts it reported."""

    epoch: int
    process: proctree.ProcessIdentity
    scope: proctree.ExecutorScope
    channel: Channel
    seal_digest: str
    #: `CUDA_VISIBLE_DEVICES` as this process was SEALED with it, verbatim (cr-066). The
    #: seal is imposed once, at spawn, and a running process can never be re-sealed — so
    #: this is what the executor actually serves, and reusing it for a lane whose
    #: `devices` differ is what `executor.py` refuses as `env_seal_broken`.
    sealed_devices: str = ""
    #: Frozen launch environment, independent of the supervision's next selection.
    launch_python: str = ""
    environment_installation_id: str = ""
    hello: dict[str, Any] = field(default_factory=dict)
    #: the bindings this process is DISPATCHABLE for: started, package imported; their
    #: constructions load on the attempt path
    ready_bindings: set[str] = field(default_factory=set)
    #: the `start` reply once every rank is up; empty until then
    started: dict[str, Any] = field(default_factory=dict)
    #: every construction built in this process (resident or parked), by construction key
    loaded: dict[str, LoadedConstruction] = field(default_factory=dict)
    #: the construction key the executor last activated
    active: str = ""
    reserved_for_job: bool = False
    poisoned: str = ""
    spawned_at: float = 0.0
    ready_at: float = 0.0
    exit_status: int | None = None
    #: The measurement a `watched` call killed this process on, so the ExecutorGone the
    #: waiting caller then sees names the wedge and not merely the death.
    wedged: str = ""
    #: the longest gap between frames this process has shown across its attempts (seconds);
    #: a later cancel's patience is never derived from less
    worst_pause: float = 0.0
    #: its plane budget cell (`budget_cell`), handed over at its first load; None before, or
    #: from an executor older than the cell
    cell: budget_cell.Cell | None = None
    _calls: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def pid(self) -> int:
        return self.process.pid

    def loaded_bindings(self) -> set[str]:
        """Dispatchable bindings whose construction is built in this process."""
        built = {digest for row in self.loaded.values() for digest in row.bindings}
        return built & self.ready_bindings

    def alive(self) -> bool:
        # Observation must not reap.  Keeping an exited direct child as a zombie until
        # driver reclaim is proven prevents its PID from being reused underneath NVML's
        # raw-PID process table.
        state = proctree.process_state(self.process)
        return state is not None and state != "Z"

    def call(
        self,
        command: Command,
        *,
        timeout: float | None = None,
        total_timeout: float | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_request: Handler | None = None,
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
            name = command.__struct_config__.tag
            deadline = time.monotonic() + total_timeout if total_timeout is not None else None
            try:
                self.channel.send(executor_commands.encode(command))
            except OSError as exc:  # its seam is closed: it was reaped (a broken group's rank 0)
                raise ExecutorGone(f"seam: {exc}", self.exit_status) from exc
            while True:
                try:
                    remaining = None if deadline is None else max(deadline - time.monotonic(), 0)
                    frame = self.channel.recv(timeout, total_timeout=remaining)
                except SeamError as exc:
                    raise ExecutorGone(f"seam: {exc}") from exc
                if frame is None:
                    raise self._gone(name)
                if frame.get("event") == "progress":
                    if on_progress is not None:
                        try:
                            on_progress(frame)
                        except Exception as exc:  # the lossy lane sheds; the call goes on
                            _LOG.warning("dropped an executor progress frame: %r", exc)
                    continue
                if frame.get("event") == "request":
                    try:
                        received = (
                            self.channel.recv_memfd() if frame.get("descriptor") is True else None
                        )
                    except SeamError as exc:
                        raise ExecutorGone(f"seam: {exc}") from exc
                    answer, handoff = respond(frame, self._cells(on_request), received)
                    try:
                        self.channel.send(answer)
                        if isinstance(handoff, int):
                            self.channel.send_memfd(handoff)
                        elif handoff is not None:
                            self.channel.send_descriptor(handoff)
                    finally:
                        if isinstance(handoff, int):
                            os.close(handoff)
                        elif handoff is not None:
                            handoff.close()
                    continue
                if "reply" not in frame and frame.get("event") is not None:
                    # An event kind from a newer executor that this worker does not consume.
                    continue
                if frame.get("reply") != name:
                    raise ExecutorGone(f"out-of-order reply {frame.get('reply')!r} for {name!r}")
                return frame

    def _gone(self, name: object) -> ExecutorGone:
        """This process's death during `name`, read without reaping: retirement owns that."""
        if self.wedged:
            return ExecutorGone(self.wedged, self.exit_status)
        status: int | None = None
        with (
            contextlib.suppress(OSError),
            contextlib.closing(proctree.observe_process_exit(self.process)) as exited,
        ):
            # Its seam closed before its exit is reportable. One that is exiting is waited for,
            # however long its device teardown takes; one that only closed its seam is not.
            if proctree.dying(self.process):
                exited.wait()
                status = proctree.peek_child_exit(self.process)
        if status is None:
            status = self.exit_status
        return ExecutorGone(death(status, name), status)

    def _cells(self, handler: Handler | None) -> Handler | None:
        """Keep this executor's budget cell; every other request goes to `handler`."""

        def answer(request: Request) -> Reply:
            if isinstance(request, BudgetCell):
                if self.cell is not None:
                    self.cell.close()
                try:
                    self.cell = budget_cell.Cell(request.memfd)
                finally:
                    os.close(request.memfd)
                return Answer(ok=True)
            if handler is None:
                return refuse("no_durable_exchange", f"no handler for {request!r}"[:200])
            return handler(request)

        return answer

    def cut(self, vram_bytes: int, *, every: float = 0.005) -> int | None:
        """Lower (or raise) a running call's plane budget at its next block boundary: the
        bytes it applied, or None when it has no cell or its call ended first (the next call
        carries its grant). Waits on the call itself, never on a clock."""
        cell = self.cell
        if cell is None:
            return None
        ticket = cell.ask(vram_bytes)
        while (applied := cell.answered(ticket)) is None:
            if not (self.busy() and self.alive()):
                return cell.answered(ticket)
            threading.Event().wait(every)
        return applied

    def busy(self) -> bool:
        """A call holds the seam now (an attempt, a load): a new one would wait for it."""
        return self._calls.locked()

    @contextlib.contextmanager
    def watched(self, what: str) -> Iterator[None]:
        """Bound a seam call by this process's OWN work, not by a clock.

        A cold `describe_installed` imports the package — torch included — and a `probe`
        after an attempt waits on a device sync backlog; neither has a length anyone can
        state in advance, and both used to be cut at a number of seconds a warm box happened
        to fit inside. Inside this context the call is made with no timeout, and this
        process is sampled on `liveness.SAMPLE_SECONDS`: it is killed — exact birth, through
        its pidfd — only when its meter (`progress_burn`: CPU plus bytes moved) has been flat
        for longer than `liveness.Pace` allows against the pauses it has itself shown. The
        kill closes the seam, the blocked `call` returns, and the ExecutorGone it raises
        carries the measurement. Process death by any other cause ends the call exactly as
        before; an unreadable meter decides nothing.
        """
        stop = threading.Event()
        pace = liveness.Pace()

        def watch() -> None:
            while not stop.wait(liveness.SAMPLE_SECONDS):
                pace.observe(liveness.burn(self.pid))
                if pace.wedged(liveness.noise_floor()):
                    self.wedged = (
                        f"the executor wedged during {what!r}: "
                        f"{pace.verdict(liveness.noise_floor())}"
                    )
                    with (
                        contextlib.suppress(ProcessLookupError, OSError),
                        contextlib.closing(proctree.observe_process_exit(self.process)) as exited,
                    ):
                        proctree.signal_process(self.process, signal.SIGKILL)
                        # Socket EOF can precede process exit. Finish this
                        # exact-birth observation before returning the wedge;
                        # retirement still owns reaping and driver reclaim.
                        exited.wait()
                    return

        thread = threading.Thread(target=watch, name=f"watch-{what}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    def close(self) -> None:
        self.channel.close()
        if self.cell is not None:
            self.cell.close()
            self.cell = None


@dataclass(frozen=True, slots=True)
class ReclaimEvidence:
    """What retirement proved before ownership was cleared."""

    exit_status: int
    device: accel.ProcessMemory
    #: every birth the scope held when it was frozen - rank 0 AND its followers under a
    #: group (cr-068) - each proven gone from the OS and from the driver's table
    members: tuple[int, ...] = ()
    #: per device of the lane's seal: the fold of every member's driver row on THAT card.
    #: Evidence for the record; the merged table over every device is what the reclaim
    #: waited on. In a PID namespace the driver names host pids, so `absent` here never
    #: overrides the OS-state half - it can only agree with it.
    devices: dict[str, accel.ProcessMemory] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OwnerRecord:
    """The exact local scope, provisionally or finally bound to its leader birth."""

    scope: proctree.ExecutorScope
    process: proctree.ProcessIdentity | None = None
    members: tuple[proctree.ProcessIdentity, ...] = ()
    """Frozen pre-kill census, durable so reclaim proof survives worker death."""


class _LaunchRequest(TypedDict):
    """Arguments crossing onto the long-lived executor launch thread."""

    python: str
    module_argv: list[str]
    base_env: tuple[tuple[str, str], ...]
    imposed: Mapping[str, str]
    cwd: str
    scope: proctree.ExecutorScope
    uid: NotRequired[int]
    gid: NotRequired[int]


type _LaunchResult = spawn.Spawned | BaseException


#: The control document an executor derives into its handoff: one installed package's
#: PackageInterface, written by `describe_installed` and unlinked by the reader at once.
INTERFACE_DOCUMENT = "package-interface.json"


class ExecutorSupervision:
    """Owns the worker's one disposable executor process and epoch fence."""

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
        kernel_cache: Path | None = None,
        device_process: Callable[[int], accel.ProcessMemory] | None = None,
        socket_path: Path | None = None,
    ) -> None:
        self.root = root
        self.kernel_cache = kernel_cache
        self.python = python
        self.environment_installation_id = ""
        self.jit_cache = ""
        """One worker-boot/content cache path, deleted after worker shutdown."""
        self.base_env = base_env
        self.cozy_home = cozy_home
        self.executor_uid = executor_uid
        self.executor_gid = executor_gid
        self.jit_pod_scope = jit_pod_scope or f"local-{secrets.token_hex(16)}"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.isolated:
            # Group execute admits the fixed executor to the named socket/cancel paths but
            # not to list or read the worker's records, owner record, or credentials.
            os.chown(self.root, 0, self.executor_gid)
            self.root.chmod(0o710)
        self._scope_namespace = proctree.executor_scope_namespace(self.root)
        lock = self.root / "worker.lock"
        try:
            self._root_lease = proctree.acquire_worker_root_lease(lock)
        except BlockingIOError as exc:
            holder = proctree.lease_holder(lock)
            if holder is None or holder == os.getpid():
                raise ExecutorGone(
                    f"worker root {self.root} is already supervised by another live worker"
                ) from exc
            # The previous generation still holds this root while it exits; its Host waits
            # for it and kills it if it stalls, so this wait ends (run 1486).
            print(
                f"[worker] boot: waiting for the previous worker (pid {holder}) "
                f"to release {self.root}",
                file=sys.stderr,
                flush=True,
            )
            self._root_lease = proctree.acquire_worker_root_lease(lock, wait=True)
        # An older executor's rank 0 writes `Model.warm`'s call tensors before any request has
        # an attempt spool. Provision that subtree before dropping privileges; the worker root
        # remains traverse-only for the executor and cannot be created beneath by that UID.
        warm = self.root / "warm"
        warm.mkdir(mode=0o700, exist_ok=True)
        if self.isolated:
            os.chown(warm, self.executor_uid, self.executor_gid)
        warm.chmod(0o700)
        # The epoch starts at 1 (02 §0): a worker that has not spawned yet already
        # carries a valid fence rather than a zero.
        self.epoch = 1
        self.current: Executor | None = None
        self.spawns = 0
        self.socket_path = str(socket_path if socket_path is not None else root / "executor.sock")
        self.cancel_path = root / "executor.cancel"
        #: the seal's `CUDA_VISIBLE_DEVICES` entries of the current spawn: the devices a
        #: reclaim proves absence on, one row each (cr-068)
        self.devices: tuple[str, ...] = ()
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
        self.on_lane_failure: Callable[[str], None] | None = None
        """Session-owned reaction to one of THIS object's own threads dying.

        The launcher and the per-epoch exit watcher are sole producers: nothing else
        launches a child, and nothing else notices one dying. A thread that dies here is a
        producer that has stopped without saying so, and the session's answer is a machine
        failure an owner can read — never a silence (cr-061)."""
        self.on_exit: Callable[[Executor], None] | None = None
        """Session-owned reaction to an exact current child becoming waitable.

        The watcher never reaps.  It reports the exact :class:`Executor` handle so the
        session can invalidate only that epoch and serialize any rebuild with device
        work; a stale watcher has no authority over a successor.
        """
        self._launches: queue.Queue[tuple[_LaunchRequest, queue.Queue[_LaunchResult]] | None] = (
            queue.Queue()
        )
        self._launcher: threading.Thread | None = None
        self._launcher_lock = threading.Lock()

    @property
    def isolated(self) -> bool:
        return self.executor_uid >= 0 and self.executor_gid >= 0

    def grant_directory(self, path: Path, epoch: int) -> None:
        """Hand one already-populated attempt/scratch directory to the current child."""

        if not self.isolated:
            return
        current = self.current
        if current is None or current.epoch != epoch:
            raise ExecutorGone(f"cannot grant {path}: executor epoch {epoch} is not current")
        os.chown(path, self.executor_uid, self.executor_gid)
        path.chmod(0o700)

    def handoff(self, epoch: int, document: str) -> Path:
        """Grant the current executor a derived control document; the reader unlinks it
        at once. Request scratch remains in the owner's `tmp/<id>/` through its grant;
        preparation's warm tensor spool is provisioned separately at supervisor startup.

        THE GRANTED DIRECTORY IS THE WHOLE SURFACE, which is why this returns the document
        and not the directory holding it. Callers used to append their own subdirectory and
        mkdir it -- as this ROOT worker, inside a directory just handed to an unprivileged
        executor -- so the executor's own `open(..., "xb")` was refused on a tree it
        nominally owned. Run 187 died there, on a rented GPU, after a 92 GiB fetch had
        already succeeded: `PermissionError: [Errno 13] Permission denied:
        '/run/cozy/worker/handoff/metadata/package-interface.json'`. A path this method
        returns is granted; one a caller builds beneath it is not.

        A stale document is cleared rather than inherited. `describe_installed` creates it
        exclusively, so one left behind by an interrupted preparation would refuse the next
        attempt -- and preparation is re-issued after exactly that kind of interruption."""
        if document != Path(document).name or document in ("", ".", ".."):
            raise ExecutorGone(f"handoff document {document!r} is not one file name")
        path = self.root / "handoff"
        path.mkdir(exist_ok=True)
        self.grant_directory(path, epoch)
        document_path = path / document
        document_path.unlink(missing_ok=True)
        return document_path

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
        if not self.devices:
            # Sealed to no device: it cannot hold device memory, and no driver is asked.
            return {pid: accel.ProcessMemory("absent", 0) for pid in pids}
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
    def hold(self, expected: Executor) -> Iterator[bool]:
        """Keep one exact executor current across an acceptance commit."""
        with self._lifecycle:
            yield self.current is expected and expected.alive() and not self._closed

    # ------------------------------------------------------------------ spawn

    def _launch(self, **request: Unpack[_LaunchRequest]) -> spawn.Spawned:
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
                    try:
                        while True:
                            request = self._launches.get()
                            if request is None:
                                return
                            call, answer = request
                            try:
                                answer.put(spawn.spawn_executor(**call))
                            except BaseException as exc:  # returned to the asking thread
                                answer.put(exc)
                    except BaseException as exc:
                        self._lane_died("executor-launcher", exc)

                self._launcher = threading.Thread(
                    target=serve_launches, daemon=True, name="executor-launcher"
                )
                self._launcher.start()
        answer: queue.Queue[_LaunchResult] = queue.Queue()
        self._launches.put((request, answer))
        value = answer.get()
        if isinstance(value, BaseException):
            raise value
        return value

    def spawn(self, *, imposed: dict[str, str]) -> Executor:
        """EXEC-FIRST spawn of a fresh epoch. Bumps BOTH ordinals: a fresh executor
        has no state, so every epoch bump implies a readiness bump.

        THE DIAL-BACK IS WAITED OUT BY OBSERVATION, not by a clock. A child that exits can
        never dial back, so that observed death refuses immediately. A live child keeps its
        chance to connect: CPU, I/O and scheduler counters do not prove that a valid import
        is stuck, and no arbitrary number of unchanged samples is allowed to kill it.
        """
        with self._lifecycle:
            if self._closed:
                raise ExecutorGone("executor supervision is closed")
            if self.current is not None:
                raise ExecutorGone(
                    f"spawn refused: epoch {self.current.epoch} pid "
                    f"{self.current.pid} is still owned; retire or replace that exact child"
                )
            return self._spawn_locked(imposed=imposed)

    def use_environment(self, python: str, installation_id: str) -> bool:
        """Point the executor at one package venv and node-local JIT scope."""

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", installation_id):
            raise ExecutorGone(f"invalid installation ID for JIT cache: {installation_id!r}")
        with self._lifecycle:
            if (self.python, self.environment_installation_id) == (python, installation_id):
                return False
            self.python = python
            self.environment_installation_id = installation_id
            return True

    def _prepare_jit_cache(self) -> dict[str, str]:
        """Impose every upstream cache path before Python can import Torch.

        A package environment gets worker-boot/exact-content reuse. A Runtime-only or
        otherwise unbound executor gets a fresh unpredictable scope on every spawn: it may
        not reuse compiler bytes without a content identity, but leaving the variables
        absent is also wrong. PyTorch lazily writes ``TORCHINDUCTOR_CACHE_DIR`` when a
        scheduler first imports its compile helpers; that post-CUDA write broke the env
        seal in local SDXL. The fresh scope fixes the ordering without inventing reuse.
        """
        installation_id = self.environment_installation_id or f"executor-{secrets.token_hex(16)}"
        try:
            prepared = jit_cache.scope(
                self.cozy_home,
                installation_id,
                pod_scope=self.jit_pod_scope,
                uid=self.executor_uid,
                gid=self.executor_gid,
            )
            self.jit_cache = str(prepared.root)
            if self.kernel_cache is None:
                return prepared.environment
            # Runtime-served kernel objects (and FA4's own cache) outlive this boot scope.
            return {
                **prepared.environment,
                **jit_cache.machine(
                    self.kernel_cache,
                    uid=self.executor_uid,
                    gid=self.executor_gid,
                    installation=self.environment_installation_id,
                ),
            }
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
        # Cancellation is scoped to one executor epoch as well as one attempt. A marker
        # from the retired epoch cannot become authority in its successor.
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
            # executor has no state, so no epoch bump can leave readiness unmoved.
            self.epoch += 1
        self.spawns += 1
        started = time.perf_counter()
        spawned: spawn.Spawned | None = None
        process: proctree.ProcessIdentity | None = None
        owner: OwnerRecord | None = None
        conn: socket.socket | None = None
        exit_published: threading.Event | None = None
        try:
            self.devices = tuple(
                entry.strip()
                for entry in imposed.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if entry.strip()
            )
            imposed = {
                **imposed,
                **self._prepare_jit_cache(),
                # Backend-mandated laws come from the accelerator boundary (#450): on MPS this
                # forces PYTORCH_ENABLE_MPS_FALLBACK=0; on CUDA it is empty. The FAMILY is a
                # cheap platform fact (present-but-unqualified, #446).
                **accel.child_env_law(accel.host_backend_family()),
                "COZY_EXECUTOR_SOCKET": self.socket_path,
                "COZY_HOME": str(self.cozy_home),
                # Beside the JIT scope, never inside it: that scope is DIRECTORIES, and
                # every value in it is an absolute path. FlashAttention-4 compiles on the pod
                # (h3a-024) and upstream gates its persistent cache on a switch read at
                # import, separate from the directory the scope already imposes. Imposed
                # unconditionally because it is meaningless without that directory and the
                # directory is always imposed.
                FLASH_ATTENTION_CACHE_ENABLED_ENV: "1",
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
                proctree.wait_process_stopped(process)
                if not proctree.process_scope_contains(owner.scope, process):
                    raise ExecutorGone(
                        f"process-tree trampoline {process.pid} stopped outside its exact scope"
                    )
            owner = self._bind_owner(owner, process)
            if isinstance(owner.scope, proctree.ProcessTreeScope):
                proctree.signal_process(process, signal.SIGCONT)
            conn = self._dial_back(process)
            executor = Executor(
                epoch=self.epoch,
                process=process,
                scope=owner.scope,
                channel=Channel(conn),
                seal_digest=spawned.seal_digest,
                sealed_devices=spawned.env.get("CUDA_VISIBLE_DEVICES", ""),
                launch_python=self.python,
                environment_installation_id=self.environment_installation_id,
                spawned_at=started,
            )
            # Identity is checked before `current` can point at it: a stale connector is not
            # this spawn merely because it reached the shared listener.  Unlike cold import,
            # hello performs no long work, so a bounded read is a lifecycle requirement.
            executor.hello = executor.call(
                Hello(), timeout=HELLO_SECONDS, total_timeout=HELLO_SECONDS
            )
            sealed = executor.hello.get("sealed")
            # An executor from another Runtime version reports its own allowlist: compare
            # every name both sides know, each against the value this worker imposed.
            seal_matches = isinstance(sealed, dict) and all(
                sealed[key] == spawned.env.get(key, "") for key in ALLOWLIST if key in sealed
            )
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
                or not seal_matches
            ):
                raise ExecutorGone(
                    f"executor hello identity mismatch for epoch {self.epoch}: "
                    f"expected pid/group {spawned.pid}, parent {os.getpid()}, seal "
                    f"{spawned.seal_digest}; received pid {executor.hello.get('pid')}, "
                    f"parent {executor.hello.get('ppid')}"
                )
            require_executor_protocol(executor.hello)
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

    def _lane_died(self, name: str, exc: BaseException) -> None:
        """One of this supervision's own threads stopped producing. Say so, loudly."""
        traceback.print_exception(exc)
        callback = self.on_lane_failure
        if callback is not None:
            with contextlib.suppress(Exception):
                callback(f"the {name} lane died: {type(exc).__name__}: {exc}"[:400])

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
                try:
                    observer.wait()
                finally:
                    observer.close()
                published.wait()
                callback = self.on_exit
                if callback is not None:
                    callback(executor)
            except BaseException as exc:
                self._lane_died(f"executor-exit-{executor.epoch}", exc)

        try:
            threading.Thread(
                target=wait_for_exit,
                daemon=True,
                name=f"executor-exit-{executor.epoch}",
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
        pace = liveness.Pace()
        while True:
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                self._reclaim_owner(owner, "provisional spawn without a captured birth")
                return
            if reaped:
                self._reclaim_owner(owner, "provisional spawn without a captured birth")
                return
            # A killed child is reapable within milliseconds unless the kernel is still
            # tearing it down; its state and resident bytes move while that happens.
            pace.observe((proctree.process_state_of(pid), proctree.rss_bytes(pid)))
            if pace.wedged(liveness.noise_floor()):
                raise ExecutorGone(
                    f"spawn cleanup could not reap provisional pid {pid}: "
                    f"{pace.verdict(liveness.noise_floor())}"
                )
            time.sleep(0.01)

    def _dial_back(self, process: proctree.ProcessIdentity) -> socket.socket:
        """The accepted connection from a freshly spawned child, or a NAMED reason there is
        none. Three answers, and a clock is not one of them."""
        assert self._listener is not None
        pid = process.pid
        self._listener.settimeout(DIAL_SAMPLE_SECONDS)
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
                    f"epoch {self.epoch} (pid {pid}) exited before dialling back",
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
        expected.ready_bindings.clear()
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
                    known.update(proctree.freeze_executor_cgroup(owner.scope))
            else:
                known.update(
                    proctree.freeze_process_tree(
                        owner.scope,
                        owner.process,
                        known,
                        worker_pid=os.getpid(),
                    )
                )
            members = tuple(sorted(known, key=lambda item: (item.pid, item.started_ticks)))
            if members != owner.members:
                expanded = OwnerRecord(owner.scope, owner.process, members)
                self._replace_owner(owner, expanded)
                owner = expanded
            if isinstance(owner.scope, proctree.CgroupScope):
                if proctree.executor_scope_exists(owner.scope):
                    proctree.kill_frozen_executor_cgroup(owner.scope)
            else:
                proctree.kill_process_tree(known)
        except (proctree.ContainmentUnavailable, TimeoutError) as exc:
            raise ExecutorGone(
                f"could not reclaim exact executor scope {owner.scope} ({why}): {exc}; "
                "ownership retained and successor spawn refused"
            ) from exc

        pace = liveness.Pace()
        observations: dict[int, accel.ProcessMemory] = {}
        while True:
            states = {member.pid: proctree.process_state(member) for member in known}
            observations = self._observe_processes(known)
            # A zombie has already completed exit: its mm and device contexts are gone, and
            # only the direct-child wait slot remains. Driver APIs commonly answer unreadable
            # for that state, which is not uncertainty about live memory. Preserve the birth
            # until `_reap_and_prove_absent`, but treat the released device side as absent.
            for member in known:
                if states[member.pid] in {None, "Z"}:
                    # A vanished birth has completed exit even further than a zombie: the
                    # kernel has already torn down its mm and device contexts, and on a
                    # host with no driver (a CPU pod) the driver API answers unreadable
                    # for every pid, which is not uncertainty about a process that no
                    # longer exists.
                    observations[member.pid] = accel.ProcessMemory("absent", 0)
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
            # The whole observation is the meter. Resident bytes fall while the kernel
            # unmaps a dying process and device rows shrink while the driver frees, so a
            # long teardown keeps moving; only one that has stopped is given up on.
            pace.observe(
                (
                    tuple(sorted(states.items())),
                    tuple(
                        sorted((pid, item.state, item.bytes) for pid, item in observations.items())
                    ),
                    tuple(sorted((member.pid, proctree.rss_bytes(member.pid)) for member in known)),
                    tuple(sorted(process.pid for process in token_leftovers)),
                )
            )
            if pace.wedged(liveness.noise_floor()):
                raise ExecutorGone(
                    f"could not prove reclaim of {owner.scope} ({why}): "
                    f"process_states={states}, device_states="
                    f"{ {pid: item.state for pid, item in observations.items()} }; "
                    f"token_leftovers={sorted(p.pid for p in token_leftovers)}; "
                    f"census={proctree.describe_scope_census(owner.scope)}; "
                    f"{pace.verdict(liveness.noise_floor())}; "
                    "ownership retained and successor spawn refused"
                )
            time.sleep(RECLAIM_SAMPLE_SECONDS)

    @staticmethod
    def _reap_and_prove_absent(
        known: set[proctree.ProcessIdentity], leader: proctree.ProcessIdentity | None
    ) -> int:
        """Reap subreaper-owned zombies, then prove every captured birth absent."""
        pace = liveness.Pace()
        leader_status = -1
        while True:
            for process in known:
                status = proctree.reap_child(process)
                if process == leader and status is not None:
                    leader_status = status
            states = {process.pid: proctree.process_state(process) for process in known}
            if all(state is None for state in states.values()):
                return leader_status
            pace.observe(tuple(sorted(states.items())))
            if pace.wedged(liveness.noise_floor()):
                raise ExecutorGone(
                    f"could not reap executor subtree "
                    f"{leader.pid if leader is not None else 'without leader'}: "
                    f"process_states={states}; {pace.verdict(liveness.noise_floor())}; "
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

    def _device_evidence(
        self, known: set[proctree.ProcessIdentity]
    ) -> dict[str, accel.ProcessMemory]:
        """The per-device half of the record: for each entry of the seal, every known birth
        folded to one row - present if any is, unreadable if any is, else absent."""
        pids = {process.pid for process in known}
        if not pids or not self.devices or self._device_process is not None:
            return {}
        rows: dict[str, accel.ProcessMemory] = {}
        by_device = accel.process_memories_by_device(
            pids, accel.host_backend_family(), self.devices
        )
        for device in self.devices:
            states = {row.state for row in by_device[device].values()}
            if "present" in states:
                rows[device] = accel.ProcessMemory(
                    "present", sum(max(row.bytes, 0) for row in by_device[device].values())
                )
            elif "unreadable" in states:
                rows[device] = accel.ProcessMemory("unreadable")
            else:
                rows[device] = accel.ProcessMemory("absent", 0)
        return rows

    def _reclaim_owner(self, owner: OwnerRecord, why: str) -> ReclaimEvidence:
        owner, known, device = self._stop_and_observe_reclaim(owner, why)
        if isinstance(owner.scope, proctree.CgroupScope):
            known |= proctree.adopted_zombies(owner.scope)
        devices = self._device_evidence(known)
        status = self._reap_and_prove_absent(known, owner.process)
        self._remove_scope_and_owner(owner)
        return ReclaimEvidence(
            status,
            device,
            members=tuple(sorted(process.pid for process in known)),
            devices=devices,
        )

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
                f"durable executor owner does not match epoch {expected.epoch} "
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
        """Retire one exact epoch and publish at most one successor."""
        with self._lifecycle:
            if self.current is not expected:
                raise ExecutorGone(
                    f"replacement refused: epoch {expected.epoch} pid "
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
                proctree.validate_process_tree_host(owner.scope.uid)
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
