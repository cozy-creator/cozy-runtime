"""Manage follower executors inside the leader's process scope.

Each follower has a framed JSON command socket and joins the group's distributed
collectives. A failed run or lost follower breaks the group and terminates its
remaining followers, allowing blocked collectives to unwind. Nothing here waits on a clock:
every wait ends with the reply or the follower's death, and one that blocks says which GPU it waits
on. A group forms only after every follower said it is ready; a follower that fails while GPU 0 is
inside group init (which nothing can interrupt) ends GPU 0's process, so TensorD sees the exit at
once. torch's own deadlines (`NEVER`) never end a group.
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
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from types import ModuleType
from typing import Any, Protocol, TypedDict

import msgspec

from cozy_runtime.author._errors import DEVICE_OOM, MEMORY_EXHAUSTED
from cozy_runtime.internal import execution_evidence, spawn
from cozy_runtime.internal.config import inherited_environment, update_environment
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.parallel.plan import GroupRefusal
from cozy_runtime.internal.seam import Channel, SeamError

#: NCCL reads this once, at communicator creation. NVLink SHARP multicast needs a privilege
#: our containers do not have and Ulysses' all-to-all never uses it (pgw#929: CUDA 401 on
#: the first all-to-all otherwise). The WORKER imposes it in a group lane's seal - erased
#: then written, never "only when unset" - and every rank REQUIRES it before forming a
#: communicator: the executor asserts the seal, it does not write its own environment.
NVLS_ENV = "NCCL_NVLS_ENABLE"
#: A GPU's part of a call ended short of memory: the attempt fails, the group stays whole.
SHORT = frozenset({DEVICE_OOM, MEMORY_EXHAUSTED})
#: Another GPU ended the collective this one waited in (`RankGroup.interrupt`).
ENDED = "group_call_ended"
#: Every group wait's deadline. torch has no "none": its store gives up after 5 min, group init and
#: every collective after 10 (NCCL) or 30 (gloo), failing a healthy slow group. A group ends with
#: its members instead: a follower's death or refusal (`RankGroup._break`), or a GPU fault TensorD
#: sees. Longer than any run, and within what torch's clocks hold.
NEVER = timedelta(days=36500)


@dataclass(frozen=True, slots=True)
class RankSpec:
    """What one rank needs to join: its rank, the world, rank 0's store port, the backend,
    and the NCCL route rank 0 chose from the group's measured links (`links.Links.nccl`)."""

    rank: int
    world: int
    port: int
    backend: str
    nccl: Mapping[str, str] = field(default_factory=dict)


class ChildProcess(Protocol):
    """What rank 0 needs of a follower's process: `spawn.Child` or a `subprocess.Popen`."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def wait(self) -> int: ...

    def send_signal(self, number: int) -> None: ...


class Frame(msgspec.Struct, frozen=True):
    """What rank 0 reads of every follower frame to route it: a progress event is
    forwarded, anything else is the reply to the command in flight."""

    event: str = ""
    name: str = ""
    reply: str = ""
    #: None: the frame states no verdict
    ok: bool | None = None
    code: str | None = None
    detail: str = ""


class _Hello(msgspec.Struct, frozen=True):
    """A follower's `hello`, as far as rank 0 checks it: who it is and under which seal."""

    rank: int
    world: int
    sealed: dict[str, str]


class SpreadLog(TypedDict):
    """One spread method this attempt: the calls each rank ran, why a rank declined, and
    whether the spread was optional. Keys are rank numbers as strings: it crosses as JSON."""

    calls: dict[str, int]
    declined: dict[str, str]
    spare: bool


@dataclass(slots=True)
class Follower:
    """One follower rank as rank 0 holds it: the process and its command seam."""

    rank: int
    process: ChildProcess
    channel: Channel
    #: how people name the GPU this process drives (`execution_evidence.gpu_name`)
    gpu: str = ""
    #: routed replies; the body stays a mapping because its readers index it by key
    replies: queue.Queue[tuple[Frame, dict[str, Any]] | None] = field(default_factory=queue.Queue)
    #: the seam-closed verdict once the reader saw EOF or a torn frame; "" while live
    gone: str = ""
    #: a byte here aborts its collectives (its watcher thread; `RankGroup.interrupt`)
    abort: socket.socket | None = None

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


def abortable(backend: str) -> None:
    """Before a process joins an NCCL group: an aborted collective raises in the thread that
    waits on it, and the process lives on to form the group again (`RankGroup.interrupt`).
    torch's watchdog otherwise tears the process down on the error an abort leaves (its
    default handling), and `_abort_process_group` asks for it off. Its heartbeat monitor kills
    the process after 8 min without a watchdog beat, a clock on a GPU that may be slow: off too.
    Read when the communicator is made, and these are its only switches."""
    if "nccl" in backend:
        update_environment(
            {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0", "TORCH_NCCL_ENABLE_MONITORING": "0"}
        )


def route(nccl: Mapping[str, str]) -> None:
    """NCCL reads its route once, when this process's first communicator forms; every GPU of
    the group writes the same values, chosen by rank 0, before it joins."""
    update_environment(nccl)


def ready_rank(torch: Any, spec: RankSpec, sealed: Mapping[str, str]) -> Any:
    """A follower's first half of joining: its route set and rank 0's store reached. Returns the
    store `form_rank` forms the group on once every follower said it is ready.

    The caller has already made this rank's device current (`torch.cuda.set_device(r)`)
    through the executor's one explicit init boundary.
    """
    require_nvls_off(sealed, spec.backend)
    abortable(spec.backend)
    route(spec.nccl)
    return torch.distributed.TCPStore(
        "127.0.0.1", spec.port, spec.world, is_master=False, timeout=NEVER
    )


def form_rank(torch: Any, spec: RankSpec, store: Any) -> Any:
    """A follower's half of the group, on the store `ready_rank` reached. Returns the group
    handle every collective is given explicitly."""
    dist = torch.distributed
    dist.init_process_group(
        spec.backend,
        store=store,
        rank=spec.rank,
        world_size=spec.world,
        device_id=_device(torch, spec.backend),
        timeout=NEVER,
    )
    dist.barrier()
    warm(torch, spec.backend, spec.rank, spec.world)
    return dist.group.WORLD


def _device(torch: ModuleType, backend: str) -> object:
    """NCCL's device, which makes its communicator eager: its buffers are allocated as the group
    forms, before weights fill the GPU, never inside a request where no rung can make room
    for the driver (12298, 12379: "NCCL Error 1: unhandled cuda error")."""
    return torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend else None


def warm(torch: ModuleType, backend: str, rank: int, world: int) -> None:
    """One of each collective a request uses, so NCCL connects them all as the group forms
    (varena counts what they hold as context): GPU 0 and each follower exchange one element
    (a request's broadcasts and results ride that pair's communicator), then the pass's level
    agreement (an int64 max), the Ulysses row gather and its all-to-all."""
    if "nccl" not in backend:
        return
    dist, device = torch.distributed, _device(torch, backend)
    one = torch.zeros(1, device=device)
    for peer in range(1, world) if rank == 0 else (0,):
        if rank == 0:
            dist.send(one, peer)
        else:
            dist.recv(one, peer)
    level = torch.zeros(1, dtype=torch.int64, device=device)
    dist.all_reduce(level, op=dist.ReduceOp.MAX)
    rows = torch.zeros(world, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(rows, level)
    sent = torch.zeros(world, dtype=torch.bfloat16, device=device)
    dist.all_to_all_single(torch.empty_like(sent), sent)
    torch.cuda.synchronize()


def abort_process_group() -> None:
    """Abort this process's communicator, so a collective in flight raises instead of waiting
    on a GPU that left it. Best effort (NCCL; a gloo group is only dropped)."""
    try:
        c10d = importlib.import_module("torch.distributed.distributed_c10d")
        abort = getattr(c10d, "_abort_process_group", None)
        if abort is not None:
            abort()
    except Exception:  # teardown must not raise
        return


class RankGroup:
    """Rank 0's handle on the K-1 followers of one generation."""

    def __init__(
        self,
        *,
        degree: int,
        backend: str,
        python: str = sys.executable,
        root: str = "",
        forward: Callable[[dict[str, object]], None] | None = None,
        launch: Callable[[list[str], int], ChildProcess] | None = None,
        devices: Iterable[str] = (),
        waiting: Callable[[str], None] = lambda what: None,
    ) -> None:
        if degree < 2:
            raise ValueError("a rank group is degree >= 2; degree 1 is a plain executor")
        self.degree = int(degree)
        self.backend = backend
        self.python = python
        self.root = root
        #: the seal's `CUDA_VISIBLE_DEVICES` entries: rank r drives entry r
        self.devices = tuple(devices)
        #: where a follower's progress frame goes: rank 0's own seam, so the worker keeps
        #: reading movement while rank 0 waits on its followers
        self.forward = forward
        #: told what rank 0 waits on as it starts to block: the call's stage, for people
        self.waiting = waiting
        #: how a follower's program is started: the trampoline's `inherit` backend under
        #: rank 0's own environment, or the suite's launcher for the gloo arms
        self._launch = launch or self._spawn_follower
        self.followers: list[Follower] = []
        self.store: Any = None
        self.port = 0
        self.pg: Any = None
        #: the NCCL route every formation of this group uses (`links.Links.nccl`)
        self.nccl: dict[str, str] = {}
        self.formed = False
        #: GPU 0 is inside group init, which no abort reaches: a follower's failure ends it
        self.initializing = False
        #: the first fault, latched: a broken group never heals, it is rebuilt
        self.broken = ""
        #: a call in flight may run again when a GPU runs out of memory (`mirror.mirrored`)
        self.rerunnable = False
        #: `interrupt` ended this formation's collectives; `form(again=...)` clears it
        self.interrupted = False
        #: set once this formation's interrupt has aborted its communicator (set: none due)
        self._ended = threading.Event()
        self._ended.set()
        #: per follower rank, the largest working peak its mirrored calls reported this attempt
        self.working_peaks: dict[int, int] = {}
        #: per follower rank, its executor's plane facts as its latest reply stated them
        #: per follower rank, its `execution_evidence` record folded over this attempt's calls
        self.rank_records: dict[int, execution_evidence.GpuEvidence] = {}
        #: per spread method this attempt
        self.spread_log: dict[str, SpreadLog] = {}
        #: each kernel a follower began serving at a command (`kernel_ready.boundary`), and its
        #: rank, until the attempt's record takes them
        self.began: list[tuple[int, str]] = []
        self._readers: list[threading.Thread] = []
        #: Followers with a command in flight. A follower answers one command at a time, so
        #: every caller that sends one holds its rank until the reply is read: calls made
        #: from overlapped threads (`author.concurrently`) never interleave on one seam.
        self._busy: set[int] = set()
        #: ranks homing a component whose remote scope is open: kept free for its calls
        self._reserved: Counter[int] = Counter()
        self._leases = threading.Condition()

    def gpu(self, rank: int) -> str:
        """How people name the GPU process `rank` drives: `GPU 3`, as nvidia-smi numbers it."""
        return execution_evidence.gpu_name(self.devices, rank)

    # ------------------------------------------------------------------ lifecycle

    def _spawn_follower(self, module_argv: list[str], rank_fd: int) -> ChildProcess:
        return spawn.spawn_follower(
            python=self.python,
            module_argv=module_argv,
            env=inherited_environment(),
            inherit_fd=rank_fd,
        )

    def spawn(self) -> None:
        """Start ranks 1..K-1. Returns as soon as the processes exist; nothing is joined. Every
        follower is launched before any reader thread starts, so a launch may fork."""
        for rank in range(1, self.degree):
            ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                argv = executor_argv(
                    rank=rank, world=self.degree, rank_fd=theirs.fileno(), root=self.root
                )
                process = self._launch(argv, theirs.fileno())
            finally:
                theirs.close()
            self.followers.append(
                Follower(rank=rank, process=process, channel=Channel(ours), gpu=self.gpu(rank))
            )
        for follower in self.followers:
            reader = threading.Thread(
                target=self._read,
                args=(follower,),
                name=f"rank{follower.rank}-reader",
                daemon=True,
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
                follower.gone = f"{follower.gpu} lost its seam: {exc}"
                break
            if frame is None:
                follower.gone = f"{follower.gpu} closed its seam"
                break
            try:
                head = msgspec.convert(frame, Frame)
            except msgspec.ValidationError as exc:
                follower.gone = f"{follower.gpu} sent a malformed frame: {exc}"
                break
            if head.event == "progress":
                if self.forward is not None:
                    with contextlib.suppress(SeamError):
                        self.forward({**frame, "name": f"{follower.gpu}:{head.name}"})
                continue
            follower.replies.put((head, frame))
            if head.reply == "form" and not head.ok:
                self._break(
                    f"{follower.gpu} could not form the group: {head.code}: {head.detail}"[:600]
                )
            if head.reply != "run" or head.ok:
                continue
            if "rerun" in frame and self.rerunnable:
                self.interrupt()  # GPU 0 may wait on it in a collective
            elif "rerun" in frame or head.code not in SHORT:
                self._break(
                    f"{follower.gpu} failed its part of the call: {head.code}: {head.detail}"[:600]
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
        if self.initializing:
            # GPU 0 waits inside group init for a member that will never arrive, and init has
            # no abort: this process ends, saying why, and TensorD fails the call on its exit.
            sys.stderr.write(
                f"group_unformed: {why}; GPU 0 cannot leave group init, so its process ends\n"
            )
            sys.stderr.flush()
            os._exit(int(Exit.failed))

    def _abort_collectives(self) -> None:
        """A dead follower must not park rank 0 on a collective that cannot complete:
        abort the communicator so the collective raises instead of waiting on NCCL's
        clock. Best effort: TensorD's watch on the call's progress is the authority."""
        if self.pg is not None:
            abort_process_group()

    def interrupt(self) -> None:
        """End the collective in flight on every GPU and keep the group whole: each
        follower's watcher aborts its communicator, and this process aborts its own. The group
        forms again (`form(again=...)`) before its next call. Once per formation: a second
        caller (the reader thread and the call's own thread both may) returns only once the
        first one's abort is done, so the group is never formed again under an abort still
        in progress (that deadlocks torch's process-group teardown)."""
        with self._leases:
            first = not self.interrupted
            if first:
                self.interrupted = True
                self._ended.clear()
        if not first:
            self._ended.wait()
            return
        try:
            for follower in self.followers:
                if follower.abort is not None:
                    with contextlib.suppress(OSError):
                        follower.abort.send(b"\1")
            self._abort_collectives()
        finally:
            self._ended.set()

    def dial_back(self, sealed: Mapping[str, str]) -> None:
        """Every follower answers `hello` as its own rank of this world, under EXACTLY this
        rank's seal."""
        self.broadcast({"cmd": "hello"})
        for follower, reply in zip(self.followers, self.collect("hello"), strict=True):
            try:
                hello = msgspec.convert(reply, _Hello)
            except msgspec.ValidationError as exc:
                raise GroupRefusal(
                    f"{follower.gpu} answered hello malformed: {exc}", code="group_unformed"
                ) from exc
            if (hello.rank, hello.world) != (follower.rank, self.degree):
                raise GroupRefusal(
                    f"{follower.gpu} dialled back as process {hello.rank} of {hello.world}, "
                    f"not {follower.rank} of {self.degree}",
                    code="group_unformed",
                )
            if hello.sealed != dict(sealed):
                raise GroupRefusal(
                    f"{follower.gpu} dialled back under a different seal "
                    f"({hello.sealed!r} vs {dict(sealed)!r})",
                    code="group_unformed",
                )

    def form(
        self,
        torch: Any,
        sealed: Mapping[str, str],
        again: bool = False,
        nccl: Mapping[str, str] | None = None,
        formed: Callable[[Any], None] | None = None,
    ) -> Any:
        """Form the process group in two steps, so a partial group never forms. `join`: every
        follower reaches the store on a kernel-chosen loopback port and says it is ready (or
        refuses, or dies), each followed by the socket its watcher reads (`interrupt`). `form`:
        only then do all GPUs enter group init, rank 0 last, with one barrier as the proof.
        `again` forms it AGAIN after an `interrupt`: every follower joins a new store.
        `nccl` is the route every GPU forms under; `formed` runs on rank 0 right after the
        barrier, while each follower runs its half before it answers `form`."""
        dist = torch.distributed

        self.check_alive()
        require_nvls_off(sealed, self.backend)
        abortable(self.backend)
        if nccl is not None:
            self.nccl = dict(nccl)
            route(self.nccl)
        if again and dist.is_initialized():
            dist.destroy_process_group()
        self.pg, self.formed = None, False
        self.store = dist.TCPStore(
            "127.0.0.1", 0, self.degree, is_master=True, wait_for_workers=False, timeout=NEVER
        )
        self.port = int(self.store.port)
        join = {"cmd": "join", "port": self.port, "backend": self.backend, "nccl": self.nccl}
        for follower in self.followers:
            if again:
                self.send_one(follower.rank, {**join, "again": True})
                continue
            follower.abort, watcher = socket.socketpair()
            try:
                self.send_one(follower.rank, {**join, "abort": True})
                follower.channel.send_descriptor(watcher)
            except SeamError as exc:
                raise GroupRefusal(
                    f"{follower.gpu} cannot be commanded: {exc}", code="group_broken"
                ) from exc
            finally:
                watcher.close()
        self._unformed(self.collect("join"), "join")
        self.initializing = True
        try:
            self.broadcast({"cmd": "form", "calibrate": formed is not None})
            self.waiting(f"forming the group: {self.gpu(0)} and {len(self.followers)} more in init")
            dist.init_process_group(
                self.backend,
                store=self.store,
                rank=0,
                world_size=self.degree,
                device_id=_device(torch, self.backend),
                timeout=NEVER,
            )
            self.pg = dist.group.WORLD
        finally:
            self.initializing = False
        if self.broken:
            raise GroupRefusal(self.broken, code="group_unformed")
        dist.barrier()
        warm(torch, self.backend, 0, self.degree)
        if formed is not None:
            formed(self.pg)
        self._unformed(self.collect("form"), "form")
        self.formed, self.interrupted = True, False
        return self.pg

    def _unformed(self, replies: list[dict[str, Any]], step: str) -> None:
        """Refuse the group on any follower's refusal of `step`."""
        for follower, reply in zip(self.followers, replies, strict=True):
            if not reply.get("ok"):
                raise GroupRefusal(
                    f"{follower.gpu} could not {step} the group: {reply.get('code')}: "
                    f"{reply.get('detail', '')}"[:400],
                    code="group_unformed",
                )

    # ------------------------------------------------------------------ commands

    def broadcast(self, command: Mapping[str, object]) -> None:
        """Deliver one command to every follower, or refuse with the group still whole."""
        self.check_alive()
        for follower in self.followers:
            try:
                follower.channel.send(dict(command))
            except SeamError as exc:
                raise GroupRefusal(
                    f"{follower.gpu} cannot be commanded: {exc}", code="group_broken"
                ) from exc

    def start(self, command: Mapping[str, object]) -> None:
        """Hand every follower rank 0's Start. The descriptors that follow TensorD's Start are
        rank 0's: the Start a follower gets names none, or it would wait for descriptors nobody
        sends."""
        self.broadcast({**command, "ledgers": 0})

    def collect(self, name: str, shown: bool = True) -> list[dict[str, Any]]:
        """One reply named `name` from every follower, in rank order. Each wait ends with the
        reply or the follower's death, observed through its seam, never inferred from silence.
        A call's replies follow its collectives every step, so those waits are not `shown`."""
        return [self._reply(follower, name, shown) for follower in self.followers]

    def call_one(self, rank: int, command: Mapping[str, object]) -> dict[str, Any]:
        """One command to ONE follower and its reply: a hosted component's call."""
        with self.hold((rank,)):
            self.send_one(rank, command)
            return self.reply_one(rank, str(command["cmd"]))

    @contextlib.contextmanager
    def hold(self, ranks: Iterable[int]) -> Iterator[None]:
        """Wait until every rank in `ranks` is free, then keep them until the block ends."""
        wanted = frozenset(ranks)
        with self._leases:
            busy = wanted & self._busy
        if busy:  # told outside the lock; the wait below reads it again
            self.waiting(f"waiting for {', '.join(map(self.gpu, sorted(busy)))}: another call")
        with self._leases:
            self._leases.wait_for(lambda: not wanted & self._busy)
            self._busy |= wanted
        try:
            yield
        finally:
            self.give(wanted)

    def take(self, ranks: Iterable[int]) -> tuple[int, ...]:
        """The ranks among `ranks` that are free and not reserved, now kept by the caller.
        Never waits: a spread round runs on whoever is idle."""
        with self._leases:
            taken = tuple(r for r in ranks if r not in self._busy and not self._reserved[r])
            self._busy.update(taken)
        return taken

    def give(self, ranks: Iterable[int]) -> None:
        with self._leases:
            self._busy.difference_update(ranks)
            self._leases.notify_all()

    def reserve(self, ranks: Iterable[int]) -> None:
        """Keep `ranks` out of spread rounds while a remote scope homed there is open."""
        with self._leases:
            self._reserved.update(ranks)

    def unreserve(self, ranks: Iterable[int]) -> None:
        with self._leases:
            self._reserved.subtract(ranks)

    def send_one(self, rank: int, command: Mapping[str, object]) -> None:
        self.check_alive()
        try:
            self.followers[rank - 1].channel.send(dict(command))
        except SeamError as exc:
            raise GroupRefusal(
                f"{self.gpu(rank)} cannot be commanded: {exc}", code="group_broken"
            ) from exc

    def reply_one(self, rank: int, name: str, shown: bool = True) -> dict[str, Any]:
        return self._reply(self.followers[rank - 1], name, shown)

    def _reply(self, follower: Follower, name: str, shown: bool = True) -> dict[str, Any]:
        if shown and follower.replies.empty():
            self.waiting(f"waiting on {follower.gpu}: {name}")
        frame = follower.replies.get()
        if frame is None:
            raise GroupRefusal(f"{self.verdict()} during {name!r}", code="group_broken")
        head, body = frame
        if head.reply != name:
            refused = f": {head.code}: {head.detail}"[:600] if head.ok is False else ""
            self._break(f"{follower.gpu} answered {head.reply!r} to {name!r}{refused}")
            raise GroupRefusal(self.broken, code="group_broken")
        return body

    def call(self, command: Mapping[str, object]) -> list[dict[str, Any]]:
        self.broadcast(command)
        return self.collect(str(command["cmd"]))

    def verdict(self) -> str:
        """The latched fault, with every exit status known by now beside it."""
        exited = [
            f"{follower.gpu} exit status {status}"
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
                self._break(f"{follower.gpu}'s process exited")
                raise GroupRefusal(self.verdict(), code="group_broken")

    def pids(self) -> tuple[int, ...]:
        """The followers' pids in rank order, rank 1 first: TensorD pairs them with GPUs by
        position."""
        return tuple(follower.pid for follower in self.followers)

    def close(self) -> None:
        """Teardown of THIS group only, idempotent. The cgroup is the authority: followers
        are told to leave, then killed; the worker proves their absence. A killed follower is
        reaped by its own thread, so one stuck in the kernel (D state) never holds the teardown."""
        pg, self.pg = self.pg, None
        for follower in self.followers:
            with contextlib.suppress(SeamError, OSError):
                follower.channel.send({"cmd": "shutdown"})
        for follower in self.followers:
            with contextlib.suppress(ProcessLookupError, OSError):
                if follower.exit_status() is None:
                    follower.process.send_signal(signal.SIGKILL)
            threading.Thread(target=_reap, args=(follower.process,), daemon=True).start()
            follower.channel.close()
            if follower.abort is not None:
                follower.abort.close()
        if pg is not None:
            with contextlib.suppress(Exception):
                importlib.import_module("torch.distributed").destroy_process_group()
        self.store = None
        self.formed = False


def _reap(process: ChildProcess) -> None:
    with contextlib.suppress(OSError):
        process.wait()
