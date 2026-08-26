"""The disposable device executor, from the supervisor's side.

Two coupled ordinals over one execution context, kept as TWO counters on purpose
(worker-protocol/02 §0):

* `executor_incarnation` — the TRANSPORT fence. Bumps on child respawn.
* `readiness_epoch` — the READINESS fence. Bumps on every state invalidation, which
  includes every incarnation bump AND the in-place case where the process survives but its
  state did not (an orphan sweep + reload). That in-place case is exactly why one merged
  counter is not enough: it would force a full respawn — a 5-second refill — for every
  orphan sweep.

Cancellation is cooperative then forceful and the SUPERVISOR owns both clocks: the grace
budget is a runtime constant here (`GRACE_SECONDS`), overridable by the operator through
`CancelAttempt.grace_ms`, and never an author knob. The forceful path kills the executor's
whole process GROUP, so a grandchild cannot outlive the terminal that closed its attempt.
"""

from __future__ import annotations

import errno
import os
import signal
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import proctree, spawn
from cozy_runtime.internal.seam import Channel, SeamError, listener
from cozy_runtime.internal.worker.observe import GIVE_UP_AFTER

#: The grace budget: long enough for the longest uninterruptible kernel plus one loop step,
#: short enough that a caller's cancel is not a lifetime. A RUNTIME CONSTANT; `grace_ms` on
#: CancelAttempt is the operator override door, and no author surface reaches it.
GRACE_SECONDS = 3.0

#: How often the supervisor LOOKS while a spawned executor is dialling back. A sampling
#: cadence, not a bound: `spawn` waits while the child is alive and burning CPU, and this
#: only says how finely it can tell. Short because the answer is normally milliseconds away.
DIAL_SAMPLE_SECONDS = 0.5


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
        f"executor_socket_unbindable: the supervisor/executor control socket would be "
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
    pid: int
    channel: Channel
    seal_digest: str
    hello: dict[str, Any] = field(default_factory=dict)
    ready_plans: set[str] = field(default_factory=set)
    poisoned: str = ""
    spawned_at: float = 0.0
    ready_at: float = 0.0
    exit_status: int | None = None

    def alive(self) -> bool:
        if self.exit_status is not None:
            return False
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            self.exit_status = -1
            return False
        if pid == 0:
            return True
        self.exit_status = status
        return False

    def call(
        self,
        command: dict[str, Any],
        *,
        timeout: float | None = None,
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
        name = command["cmd"]
        self.channel.send(command)
        while True:
            try:
                frame = self.channel.recv(timeout)
            except SeamError as exc:
                raise ExecutorGone(f"seam: {exc}") from exc
            if frame is None:
                self.alive()
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
                        "detail": f"this attempt installed no handler for {frame.get('kind')!r}",
                    }
                )
                self.channel.send({"event": "answer", "seq": frame.get("seq"), **answer})
                continue
            if frame.get("reply") != name:
                raise ExecutorGone(f"out-of-order reply {frame.get('reply')!r} for {name!r}")
            return frame

    # ------------------------------------------------------------------ signals

    def cooperative_cancel(self) -> bool:
        """Signal the running handler to unwind. It may ignore this; that is the point."""
        try:
            os.kill(self.pid, signal.SIGUSR1)
            return True
        except ProcessLookupError:
            return False

    def kill_group(self, sig: int = signal.SIGKILL) -> int:
        """Forceful: the executor AND every grandchild, by process group."""
        killed = 0
        try:
            proctree.tree_kill(self.pid, sig)
            killed += 1
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(self.pid, sig)
                killed += 1
            except ProcessLookupError:
                pass
        return killed

    def reap(self, seconds: float = 10.0) -> int | None:
        """Wait for the process to be gone AND reaped — a zombie is not a dead executor."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not self.alive():
                return self.exit_status
            time.sleep(0.01)
        return None

    def close(self) -> None:
        self.channel.close()


class ExecutorSupervision:
    """Spawns, watches, kills and replaces the executor; owns the two ordinals."""

    def __init__(
        self,
        *,
        root: Path,
        python: str,
        base_env: tuple[tuple[str, str], ...],
        cozy_home: Path,
    ) -> None:
        self.root = root
        self.python = python
        self.base_env = base_env
        self.cozy_home = cozy_home
        # Both ordinals START at 1 (02 §0): a worker that has not spawned yet is already
        # at incarnation 1, so its Register carries a valid fence rather than a zero.
        self.incarnation = 1
        self.readiness_epoch = 1
        self.current: Executor | None = None
        self.needs_rebuild = False
        """Set by every invalidation. It exists because `alive()` is a RACE against a
        process that is being killed on another thread: a dying executor answers `waitpid`
        with 0 for a few milliseconds, and a rebuild decision that reads liveness alone
        silently skips exactly when it matters most."""
        self.spawns = 0
        self.socket_path = str(root / "executor.sock")
        self._listener: socket.socket | None = None
        self.on_invalidate: Callable[[str], None] | None = None
        """Called AFTER every invalidation, once this object's own state already says the
        executor cannot serve. The session installs the half it owns — dropping the
        advertised intake state — because readiness is a SESSION fact and the process
        supervisor has no business publishing one. See `invalidate`."""

    # ------------------------------------------------------------------ spawn

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
            self.readiness_epoch += 1
        self.spawns += 1
        imposed = {
            **imposed,
            "COZY_EXECUTOR_SOCKET": self.socket_path,
            "COZY_HOME": str(self.cozy_home),
        }
        started = time.perf_counter()
        spawned = spawn.spawn_executor(
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
        )
        conn = self._dial_back(spawned.pid)
        executor = Executor(
            incarnation=self.incarnation,
            pid=spawned.pid,
            channel=Channel(conn),
            seal_digest=spawned.seal_digest,
            spawned_at=started,
        )
        # No clock on the hello either: the child has just proved it is alive by dialling
        # back, and if it dies mid-answer the seam's own EOF is what says so — which is how
        # every other command on this channel is waited for.
        executor.hello = executor.call({"cmd": "hello"}, timeout=None)
        executor.ready_at = time.perf_counter()
        self.current = executor
        return executor

    def _dial_back(self, pid: int) -> Any:
        """The accepted connection from a freshly spawned child, or a NAMED reason there is
        none. Three answers, and a clock is not one of them."""
        assert self._listener is not None
        self._listener.settimeout(DIAL_SAMPLE_SECONDS)
        still, burned = 0, -1
        while True:
            try:
                conn, _ = self._listener.accept()
                return conn
            except TimeoutError:
                pass
            try:
                reaped, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                reaped, status = pid, -1
            if reaped:
                # GONE. Nothing will ever dial back, and the old constant spent a full
                # minute not saying so — the reason is knowable the moment the child exits.
                raise ExecutorGone(
                    f"incarnation {self.incarnation} (pid {pid}) exited "
                    f"(status {status}) before dialling back",
                    status,
                )
            still += 1
            burned, before = work(pid), burned
            if burned > before >= 0:
                still = 0  # importing, and an import on a cold page cache takes what it takes
            if still >= GIVE_UP_AFTER:
                os.kill(pid, signal.SIGKILL)
                raise ExecutorGone(
                    f"incarnation {self.incarnation} (pid {pid}) is alive and has burned no "
                    f"CPU across {still} observations while owing a dial-back: it is stalled, "
                    "not slow"
                )

    def invalidate(self, why: str) -> int:
        """In-place readiness invalidation: the PROCESS may survive, its state did not.

        THE ADVERTISEMENT GOES WITH IT (cr-007, cl-003). A device process that cannot serve
        must stop SAYING it can at the moment it stops, which is here: the ready-plan set
        empties and `on_invalidate` drops the intake state before this call returns.
        Advertising READY until the rebuild started spent a coordinator's whole requeue
        budget on a worker that was about to be healthy.
        """
        self.readiness_epoch += 1
        self.needs_rebuild = True
        if self.current is not None:
            self.current.ready_plans.clear()
            self.current.poisoned = self.current.poisoned or why
        if self.on_invalidate is not None:
            self.on_invalidate(why)
        return self.readiness_epoch

    def replace(self, *, imposed: dict[str, str]) -> Executor:
        """Kill the current incarnation (group and all) and spawn a fresh one."""
        if self.current is not None:
            self.current.kill_group()
            self.current.reap()
            self.current.close()
            self.current = None
        return self.spawn(imposed=imposed)

    # ------------------------------------------------------------------ orphans

    def sweep_orphans(self, *, exclude: int = 0) -> list[int]:
        """Fence or kill leftover executors from before a supervisor restart (§6.2 step 1).

        Identity is the executor's own argv — it names this worker's socket path — so the
        sweep can never touch another worker's executor or an unrelated process.
        """
        killed: list[int] = []
        markers = (b"cozy_runtime.internal.executor", self.socket_path.encode())
        for pid in proctree.pids_with_cmdline_markers(markers, exclude=exclude):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise
        return killed

    def close(self) -> None:
        # Bind ONCE. `current` is written by the rebuild path on another thread, so a
        # check-then-use here races a concurrent replacement and dies on `None` mid-shutdown
        # — observed live, as an AttributeError that took the whole run down after its
        # terminal was already durable.
        executor, self.current = self.current, None
        if executor is not None:
            executor.kill_group()
            executor.reap(2.0)
            executor.close()
        if self._listener is not None:
            self._listener.close()
            Path(self.socket_path).unlink(missing_ok=True)
            self._listener = None
