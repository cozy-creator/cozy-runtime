"""RankGroup - the K-1 follower ranks that execute ONE generation beside rank 0.

Rank 0 is the executor the worker spawned and addresses. It spawns the followers itself:
each is the same executor program (`internal/executor.py --rank r --world K`), a direct
child inside rank 0's cgroup (inherited), armed with pdeathsig so it dies with rank 0,
sealed to the same `CUDA_VISIBLE_DEVICES` and pinned to card `r` of it. The worker never
sees a follower except as a process in the scope it already freezes, kills and proves
absent.

Two planes, deliberately distinct:

* THE COMMAND PLANE is a `socketpair` per follower carrying the worker/executor seam's own
  length-framed canonical-JSON frames (`internal/seam.py`): control, never pickle, never a
  path on disk. Rank 0 broadcasts a command to every follower and collects one reply from
  each; a follower parks on `recv`, so teardown cannot hang on a collective.
* THE COLLECTIVE PLANE is torch.distributed over a loopback `TCPStore` on port 0 -
  kernel-chosen, in-executor, never a constant and never on the wire - with NVLS
  multicast off before the communicator forms (pgw#929: CUDA 401 on the first all-to-all
  otherwise). Collectives happen only inside the gated call (`cp.py`).

Nothing here waits on a clock. A follower that has not dialled back or replied is judged
by `liveness.Pace` over its own progress meter: one that is still importing torch or
filling is left alone, one that has stopped against its own evidence is `group_broken`.
A follower's death is observed the moment its seam closes, and the group is then broken
for good: rank 0 poisons itself and the worker rebuilds the whole cgroup.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import queue
import signal
import socket
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from cozy_runtime.internal import liveness, spawn
from cozy_runtime.internal.config import inherited_environment
from cozy_runtime.internal.parallel.plan import GroupRefusal
from cozy_runtime.internal.seam import Channel, SeamError

#: NCCL reads this once, at communicator creation. NVLink SHARP multicast needs a privilege
#: our containers do not have and Ulysses' all-to-all never uses it (pgw#929: CUDA 401 on
#: the first all-to-all otherwise). The WORKER imposes it in a group lane's seal - erased
#: then written, never "only when unset" - and every rank REQUIRES it before forming a
#: communicator: the executor asserts the seal, it does not write its own environment.
NVLS_ENV = "NCCL_NVLS_ENABLE"


@dataclass(frozen=True, slots=True)
class RankSpec:
    """What one rank needs to join: its rank, the world, rank 0's store port, the backend."""

    rank: int
    world: int
    port: int
    backend: str


class ChildProcess(Protocol):
    """What rank 0 needs of a follower's process: `spawn.Child` or a `subprocess.Popen`."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def wait(self) -> int: ...

    def send_signal(self, number: int) -> None: ...


@dataclass(slots=True)
class Follower:
    """One follower rank as rank 0 holds it: the process and its command seam."""

    rank: int
    process: ChildProcess
    channel: Channel
    hello: dict[str, Any] = field(default_factory=dict)
    replies: queue.Queue[dict[str, Any] | None] = field(default_factory=queue.Queue)
    #: the seam-closed verdict once the reader saw EOF or a torn frame; "" while live
    gone: str = ""

    @property
    def pid(self) -> int:
        return self.process.pid

    def exit_status(self) -> int | None:
        return self.process.poll()


def executor_argv(*, rank: int, world: int, rank_fd: int, root: str) -> list[str]:
    """The follower's program: the executor itself, told its rank and its seam fd."""
    return [
        "-c",
        "import sys; from cozy_runtime.internal.executor import main; "
        "raise SystemExit(main(sys.argv[1:]))",
        "--root",
        root,
        "--rank",
        str(rank),
        "--world",
        str(world),
        "--rank-fd",
        str(rank_fd),
        "--leader",
        str(os.getpid()),
    ]


def require_nvls_off(sealed: Mapping[str, str], backend: str) -> None:
    """Refuse to form an NCCL communicator under a seal that did not turn NVLS off."""
    if backend == "nccl" and sealed.get(NVLS_ENV) != "0":
        raise GroupRefusal(
            f"the seal carries {NVLS_ENV}={sealed.get(NVLS_ENV)!r}; a group lane's seal "
            "imposes 0 before any communicator forms (pgw#929)",
            code="group_unformed",
        )


def join_rank(torch: Any, spec: RankSpec, sealed: Mapping[str, str]) -> Any:
    """A follower joins rank 0's store and forms its half of the process group.

    The caller has already made this rank's device current (`torch.cuda.set_device(r)`)
    through the executor's one explicit init boundary. Returns the group handle every
    collective is given explicitly.
    """
    dist = torch.distributed

    require_nvls_off(sealed, spec.backend)
    store = dist.TCPStore("127.0.0.1", spec.port, spec.world, is_master=False)
    dist.init_process_group(spec.backend, store=store, rank=spec.rank, world_size=spec.world)
    dist.barrier()
    return dist.group.WORLD


class RankGroup:
    """Rank 0's handle on the K-1 followers of one generation."""

    def __init__(
        self,
        *,
        degree: int,
        backend: str,
        python: str = sys.executable,
        root: str = "",
        sample_seconds: float = liveness.SAMPLE_SECONDS,
        forward: Callable[[dict[str, Any]], None] | None = None,
        launch: Callable[[list[str], int], ChildProcess] | None = None,
    ) -> None:
        if degree < 2:
            raise ValueError("a rank group is degree >= 2; degree 1 is a plain executor")
        self.degree = int(degree)
        self.backend = backend
        self.python = python
        self.root = root
        self.sample_seconds = float(sample_seconds)
        #: where a follower's progress frame goes: rank 0's own seam, so the worker keeps
        #: reading movement while rank 0 waits on its followers
        self.forward = forward
        #: how a follower's program is started: the trampoline's `inherit` backend under
        #: rank 0's own environment, or the suite's launcher for the gloo arms
        self._launch = launch or self._spawn_follower
        self.followers: list[Follower] = []
        self.store: Any = None
        self.port = 0
        self.pg: Any = None
        self.formed = False
        #: the first fault, latched: a broken group never heals, it is rebuilt
        self.broken = ""
        self._readers: list[threading.Thread] = []

    # ------------------------------------------------------------------ lifecycle

    def _spawn_follower(self, module_argv: list[str], rank_fd: int) -> ChildProcess:
        return spawn.spawn_follower(
            python=self.python,
            module_argv=module_argv,
            env=inherited_environment(),
            inherit_fd=rank_fd,
        )

    def spawn(self) -> None:
        """Start ranks 1..K-1. Returns as soon as the processes exist; nothing is joined."""
        for rank in range(1, self.degree):
            ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                argv = executor_argv(
                    rank=rank, world=self.degree, rank_fd=theirs.fileno(), root=self.root
                )
                process = self._launch(argv, theirs.fileno())
            finally:
                theirs.close()
            follower = Follower(rank=rank, process=process, channel=Channel(ours))
            self.followers.append(follower)
            reader = threading.Thread(
                target=self._read, args=(follower,), name=f"rank{rank}-reader", daemon=True
            )
            reader.start()
            self._readers.append(reader)

    def _read(self, follower: Follower) -> None:
        """One follower's inbound frames: progress is forwarded, a reply is queued, and
        the seam closing is the death notice."""
        while True:
            try:
                frame = follower.channel.recv(timeout=None)
            except SeamError as exc:
                follower.gone = f"rank {follower.rank} seam: {exc}"
                break
            if frame is None:
                follower.gone = f"rank {follower.rank} closed its seam"
                break
            if frame.get("event") == "progress":
                if self.forward is not None:
                    frame = {**frame, "name": f"rank{follower.rank}:{frame.get('name', '')}"}
                    with contextlib.suppress(SeamError):
                        self.forward(frame)
                continue
            follower.replies.put(frame)
            if frame.get("reply") == "run" and not frame.get("ok"):
                self._break(
                    f"rank {follower.rank} failed its mirrored call: "
                    f"{frame.get('code')}: {frame.get('detail', '')}"[:600]
                )
        follower.replies.put(None)
        self._break(follower.gone)

    def _break(self, why: str) -> None:
        if self.broken:
            return
        self.broken = why
        self._abort_collectives()
        # A live follower can fail before entering a collective its peers have entered.
        # Closing those processes releases the peer connections too, including on gloo
        # where the experimental NCCL abort API cannot unblock the leader.
        for follower in self.followers:
            with contextlib.suppress(ProcessLookupError, OSError):
                if follower.exit_status() is None:
                    follower.process.send_signal(signal.SIGKILL)

    def _abort_collectives(self) -> None:
        """A dead follower must not park rank 0 on a collective that cannot complete:
        abort the communicator so the collective raises instead of waiting on NCCL's
        clock. Best effort - the worker's own liveness watch on rank 0 is the authority."""
        if self.pg is None:
            return
        try:
            c10d = importlib.import_module("torch.distributed.distributed_c10d")
            abort = getattr(c10d, "_abort_process_group", None)
            if abort is not None:
                abort()
        except Exception:  # teardown must not raise
            return

    def dial_back(self, sealed: Mapping[str, str]) -> None:
        """Every follower answers `hello` under EXACTLY this rank's seal."""
        self.broadcast({"cmd": "hello"})
        for follower, reply in zip(self.followers, self.collect("hello"), strict=True):
            follower.hello = reply
            if dict(reply.get("sealed") or {}) != dict(sealed):
                raise GroupRefusal(
                    f"rank {follower.rank} dialled back under a different seal "
                    f"({reply.get('sealed')!r} vs {dict(sealed)!r})",
                    code="group_unformed",
                )

    def form(self, torch: Any, sealed: Mapping[str, str]) -> Any:
        """Form the process group: the store on a kernel-chosen loopback port, every
        follower joining it, rank 0 joining last, one barrier as the proof."""
        dist = torch.distributed

        self.check_alive()
        require_nvls_off(sealed, self.backend)
        self.store = dist.TCPStore(
            "127.0.0.1", 0, self.degree, is_master=True, wait_for_workers=False
        )
        self.port = int(self.store.port)
        self.broadcast({"cmd": "join", "port": self.port, "backend": self.backend})
        dist.init_process_group(self.backend, store=self.store, rank=0, world_size=self.degree)
        self.pg = dist.group.WORLD
        dist.barrier()
        for follower, reply in zip(self.followers, self.collect("join"), strict=True):
            if not reply.get("ok"):
                raise GroupRefusal(
                    f"rank {follower.rank} could not join: {reply.get('code')}: "
                    f"{reply.get('detail', '')}"[:400],
                    code="group_unformed",
                )
        self.formed = True
        return self.pg

    # ------------------------------------------------------------------ commands

    def broadcast(self, command: dict[str, Any]) -> None:
        """Deliver one command to every follower, or refuse with the group still whole."""
        self.check_alive()
        for follower in self.followers:
            try:
                follower.channel.send(command)
            except SeamError as exc:
                raise GroupRefusal(
                    f"rank {follower.rank} cannot be commanded: {exc}", code="group_broken"
                ) from exc

    def collect(self, name: str) -> list[dict[str, Any]]:
        """One reply named `name` from every follower, in rank order.

        The wait is on the follower's OWN progress meter: sampled every `sample_seconds`,
        wedged only against the pauses it has itself shown (`liveness.Pace`). Death is
        observed through the seam, not inferred from silence.
        """
        replies: list[dict[str, Any]] = []
        for follower in self.followers:
            pace = liveness.Pace()
            floor = liveness.noise_floor(self.sample_seconds)
            while True:
                try:
                    frame = follower.replies.get(timeout=self.sample_seconds)
                except queue.Empty:
                    pace.observe(liveness.burn(follower.pid))
                    if pace.wedged(floor):
                        self._break(
                            f"rank {follower.rank} wedged during {name!r}: {pace.verdict(floor)}"
                        )
                        raise GroupRefusal(self.broken, code="group_broken") from None
                    continue
                if frame is None:
                    raise GroupRefusal(f"{self.verdict()} during {name!r}", code="group_broken")
                if frame.get("reply") != name:
                    self._break(f"rank {follower.rank} answered {frame.get('reply')!r} to {name!r}")
                    raise GroupRefusal(self.broken, code="group_broken")
                replies.append(frame)
                break
        return replies

    def call(self, command: dict[str, Any]) -> list[dict[str, Any]]:
        self.broadcast(command)
        return self.collect(str(command["cmd"]))

    def verdict(self) -> str:
        """The latched fault, with every exit status known by now beside it."""
        exited = [
            f"rank {follower.rank} exit status {status}"
            for follower in self.followers
            if (status := follower.exit_status()) is not None
        ]
        return self.broken + (f" ({'; '.join(exited)})" if exited else "")

    def check_alive(self) -> None:
        """A dead or gone follower fails the caller LOUDLY, now - never a park on a
        collective that cannot complete."""
        if self.broken:
            raise GroupRefusal(self.verdict(), code="group_broken")
        for follower in self.followers:
            if follower.exit_status() is not None:
                self._break(f"rank {follower.rank} exited")
                raise GroupRefusal(self.verdict(), code="group_broken")

    def pids(self) -> tuple[int, ...]:
        return tuple(follower.pid for follower in self.followers)

    def close(self) -> None:
        """Teardown of THIS group only, idempotent. The cgroup is the authority: followers
        are told to leave, then killed; the worker proves their absence."""
        pg, self.pg = self.pg, None
        for follower in self.followers:
            with contextlib.suppress(SeamError, OSError):
                follower.channel.send({"cmd": "shutdown"})
        for follower in self.followers:
            with contextlib.suppress(ProcessLookupError, OSError):
                if follower.exit_status() is None:
                    follower.process.send_signal(signal.SIGKILL)
            with contextlib.suppress(OSError):
                follower.process.wait()
            follower.channel.close()
        if pg is not None:
            with contextlib.suppress(Exception):
                importlib.import_module("torch.distributed").destroy_process_group()
        self.store = None
        self.formed = False
