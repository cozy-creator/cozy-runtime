"""Manage follower executors inside the leader's process scope.

Each follower has a framed JSON command socket and joins the group's distributed
collectives. A failed run or lost follower breaks the group and terminates its
remaining followers, allowing blocked collectives to unwind. Import/fill progress
is observed through the existing liveness meter.
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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, TypedDict

import msgspec

from cozy_runtime.internal import execution_evidence, liveness, spawn
from cozy_runtime.internal.config import inherited_environment, update_environment
from cozy_runtime.internal.parallel.plan import GpuDivergence, GroupPlan, GroupRefusal
from cozy_runtime.internal.seam import Channel, SeamError

#: NCCL reads this once, at communicator creation. NVLink SHARP multicast needs a privilege
#: our containers do not have and Ulysses' all-to-all never uses it (pgw#929: CUDA 401 on
#: the first all-to-all otherwise). The WORKER imposes it in a group lane's seal - erased
#: then written, never "only when unset" - and every rank REQUIRES it before forming a
#: communicator: the executor asserts the seal, it does not write its own environment.
NVLS_ENV = "NCCL_NVLS_ENABLE"


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
    forwarded, a request is relayed to the machine, anything else is the reply to the
    command in flight."""

    event: str = ""
    name: str = ""
    reply: str = ""
    #: None: the frame states no verdict
    ok: bool | None = None
    code: str | None = None
    detail: str = ""


class _Hello(msgspec.Struct, frozen=True):
    """A follower's `hello`, as far as rank 0 checks it: who it is and under which seal, and
    what its memory answers beyond the floor's commands (`weight_plane/1`)."""

    rank: int
    world: int
    sealed: dict[str, str]
    memory: tuple[str, ...] = ()


_Bytes = Annotated[int, msgspec.Meta(ge=0)]


class _Offer(msgspec.Struct, frozen=True, tag_field="reply", tag="resident_budget"):
    """A follower's plan and the bytes it needs, for `agree_residence`."""

    rank: int
    model: str
    plan: GroupPlan
    required: _Bytes


class _Parking(msgspec.Struct, frozen=True, tag_field="cmd", tag="resident_budget"):
    """Rank 0's answer to the offers: the constructions every rank parks."""

    model: str
    park: list[str]


class _Measured(msgspec.Struct, frozen=True, tag_field="reply", tag="resident_ceiling"):
    """A follower's ceiling, measured after it parked."""

    rank: int
    model: str
    ceiling: _Bytes


class _Ceiling(msgspec.Struct, frozen=True, tag_field="cmd", tag="resident_ceiling"):
    """The group's ceiling: the smallest any rank measured."""

    model: str
    ceiling: _Bytes


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
    #: its `hello`'s memory capabilities: rank 0 sends it only the commands it answers
    memory: tuple[str, ...] = ()
    #: a request of its own is with the machine: that wait is not its stillness
    relaying: bool = False
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
    default handling), and `_abort_process_group` asks for it off. Read when the
    communicator is made, and this is its only switch."""
    if "nccl" in backend:
        update_environment({"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"})


#: The NCCL names this process's group set from its measured links: Runtime-owned, so the
#: seal's restore leaves them alone (an older machine still seals NCCL_P2P_LEVEL=NVL).
ROUTED: dict[str, str] = {}


def route(nccl: Mapping[str, str]) -> None:
    """NCCL reads its route once, when this process's first communicator forms; every GPU of
    the group writes the same values, chosen by rank 0, before it joins."""
    ROUTED.update(nccl)
    update_environment(nccl)


def join_rank(torch: Any, spec: RankSpec, sealed: Mapping[str, str]) -> Any:
    """A follower joins rank 0's store and forms its half of the process group.

    The caller has already made this rank's device current (`torch.cuda.set_device(r)`)
    through the executor's one explicit init boundary. Returns the group handle every
    collective is given explicitly.
    """
    dist = torch.distributed

    require_nvls_off(sealed, spec.backend)
    abortable(spec.backend)
    route(spec.nccl)
    store = dist.TCPStore("127.0.0.1", spec.port, spec.world, is_master=False)
    dist.init_process_group(spec.backend, store=store, rank=spec.rank, world_size=spec.world)
    dist.barrier()
    return dist.group.WORLD


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


def _decode[T](
    frame: Mapping[str, object] | None, kind: type[T], gpu: str, field_name: str, what: str
) -> T:
    """One residence message, or a divergence naming the GPU and the field it broke."""
    try:
        return msgspec.convert(frame, kind)
    except msgspec.ValidationError as exc:
        raise GpuDivergence(gpu, field_name, f"{what}: {exc}") from exc


def receive_residence(
    channel: Channel,
    *,
    rank: int,
    model: str,
    plan: Mapping[str, object],
    required: int,
    park: Callable[[list[str]], None],
    ceiling: Callable[[], int],
) -> int:
    """A follower's half of `RankGroup.agree_residence`.

    Offer this rank's plan and need, park what rank 0 parked, offer the ceiling measured
    after that, and accept only a matching, no-larger group ceiling.
    """
    offer = _Offer(rank, model, GroupPlan.read(plan), required)
    gpu = execution_evidence.gpu_name(offer.plan.devices, rank)
    channel.send(msgspec.to_builtins(offer))
    what = "invalid group parking decision"
    parking = _decode(channel.recv(timeout=None), _Parking, gpu, "parked_constructions", what)
    if parking.model != model:
        raise GpuDivergence(gpu, "parked_constructions", f"{what} for {parking.model!r}")
    park(parking.park)
    local = ceiling()
    channel.send(msgspec.to_builtins(_Measured(rank, model, local)))
    what = "invalid group residence ceiling"
    shared = _decode(channel.recv(timeout=None), _Ceiling, gpu, "resident_budget_bytes", what)
    if shared.model != model or shared.ceiling > local:
        raise GpuDivergence(gpu, "resident_budget_bytes", what)
    return shared.ceiling


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
        forward: Callable[[dict[str, object]], None] | None = None,
        relay: Callable[[Channel, dict[str, Any]], None] | None = None,
        launch: Callable[[list[str], int], ChildProcess] | None = None,
        devices: Iterable[str] = (),
    ) -> None:
        if degree < 2:
            raise ValueError("a rank group is degree >= 2; degree 1 is a plain executor")
        self.degree = int(degree)
        self.backend = backend
        self.python = python
        self.root = root
        self.sample_seconds = float(sample_seconds)
        #: the seal's `CUDA_VISIBLE_DEVICES` entries: rank r drives entry r
        self.devices = tuple(devices)
        #: where a follower's progress frame goes: rank 0's own seam, so the worker keeps
        #: reading movement while rank 0 waits on its followers
        self.forward = forward
        #: answers a follower's own request to the machine on its channel (rank 0's `_relay`)
        self.relay = relay
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
        #: the first fault, latched: a broken group never heals, it is rebuilt
        self.broken = ""
        #: a call in flight may run again when a GPU runs out of memory (`mirror.mirrored`)
        self.rerunnable = False
        #: `interrupt` ended this formation's collectives; `form(room=...)` clears it
        self.interrupted = False
        #: set once this formation's interrupt has aborted its communicator (set: none due)
        self._ended = threading.Event()
        self._ended.set()
        #: device bytes the followers gave back at the last `form(room=...)`
        self.gave = 0
        #: per mirrored component, the pace of its calls (`watched`): the longest pause any
        #: call of it showed carries over to the next
        self.paces: dict[str, liveness.Pace] = {}
        #: per follower rank, the largest working peak its mirrored calls reported this attempt
        self.working_peaks: dict[int, int] = {}
        #: per follower rank, its executor's plane facts as its latest reply stated them
        self.rank_planes: dict[int, dict[str, Any]] = {}
        #: per follower rank, its `execution_evidence` record folded over this attempt's calls
        self.rank_records: dict[int, execution_evidence.RankEvidence] = {}
        #: per spread method this attempt
        self.spread_log: dict[str, SpreadLog] = {}
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
            if head.event == "request" and self.relay is not None:
                follower.relaying = True
                try:
                    self.relay(follower.channel, frame)
                except SeamError as exc:
                    follower.gone = f"{follower.gpu}'s request was not relayed: {exc}"
                    break
                finally:
                    follower.relaying = False
                continue
            if isinstance(frame.get("plane"), dict):
                self.rank_planes[follower.rank] = frame["plane"]
            follower.replies.put((head, frame))
            if head.reply == "run" and not head.ok and self.rerunnable and "rerun" in frame:
                self.interrupt()
            elif head.reply == "run" and not head.ok:
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

    def _abort_collectives(self) -> None:
        """A dead follower must not park rank 0 on a collective that cannot complete:
        abort the communicator so the collective raises instead of waiting on NCCL's
        clock. Best effort - the worker's own liveness watch on rank 0 is the authority."""
        if self.pg is not None:
            abort_process_group()

    @contextlib.contextmanager
    def watched(self, name: str, device: Callable[[], int]) -> Iterator[None]:
        """A mirrored call in flight, judged by its own pace (`liveness.Pace`), never a clock:
        the reading is GPU 0's device progress (`device`), the bytes each follower moved and
        the answers that arrived. Every GPU of a call waits on the others in its
        collectives, so a GPU stuck in one stops them all; a spinning wait burns CPU, which is
        why CPU time is not in the reading. Still for longer than STILL_FACTOR times the
        longest pause any call of this component showed (the first call learns its own, from
        its device progress) and the noise floor: the group breaks, naming the GPUs that
        have not answered. NCCL's own watchdog is off (`abortable`): this is the one that
        ends a wedged call."""
        pace = self.paces.setdefault(name, liveness.Pace())
        pace.reading = None  # this call opens its own clock
        floor = liveness.noise_floor(self.sample_seconds)
        finished = threading.Event()

        def watch() -> None:
            while not finished.wait(self.sample_seconds):
                pace.observe(
                    (
                        device(),
                        tuple(liveness.moved(f.pid) for f in self.followers),
                        tuple(f.replies.qsize() for f in self.followers),
                    )
                )
                if pace.wedged(floor):
                    quiet = [f.gpu for f in self.followers if f.replies.empty()] or ["GPU 0"]
                    self._break(f"{', '.join(quiet)} wedged in {name!r}: {pace.verdict(floor)}")
                    return

        watcher = threading.Thread(target=watch, name="group-watch", daemon=True)
        watcher.start()
        try:
            yield
        finally:
            finished.set()
            watcher.join()

    def interrupt(self) -> None:
        """End the collective in flight on every GPU and keep the group whole: each
        follower's watcher aborts its communicator, and this process aborts its own. The group
        forms again (`form(room=...)`) before its next call. Once per formation: a second
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
            follower.memory = hello.memory

    def form(
        self,
        torch: Any,
        sealed: Mapping[str, str],
        cells: Sequence[int] = (),
        room: int = -1,
        nccl: Mapping[str, str] | None = None,
        formed: Callable[[Any], None] | None = None,
    ) -> Any:
        """Form the process group: the store on a kernel-chosen loopback port, every
        follower joining it, rank 0 joining last, one barrier as the proof. `cells`, one per
        follower in rank order, are their budget cells: each follows its `join`, then the
        socket its watcher reads (`interrupt`). `room` >= 0 forms it AGAIN after an
        `interrupt`: every follower joins a new store and gives that many device bytes back.
        `nccl` is the route every GPU forms under; `formed` runs on rank 0 right after the
        barrier, while each follower runs its half before it answers the join."""
        dist = torch.distributed

        self.check_alive()
        require_nvls_off(sealed, self.backend)
        abortable(self.backend)
        if nccl is not None:
            self.nccl = dict(nccl)
            route(self.nccl)
        if room >= 0 and dist.is_initialized():
            dist.destroy_process_group()
        self.store = dist.TCPStore(
            "127.0.0.1", 0, self.degree, is_master=True, wait_for_workers=False
        )
        self.port = int(self.store.port)
        join = {
            "cmd": "join",
            "port": self.port,
            "backend": self.backend,
            "nccl": self.nccl,
            "calibrate": formed is not None,
        }
        for index, follower in enumerate(self.followers):
            if room >= 0:
                self.send_one(follower.rank, {**join, "room": room})
                continue
            follower.abort, watcher = socket.socketpair()
            try:
                self.send_one(follower.rank, {**join, "cell": bool(cells), "abort": True})
                if cells:
                    self.memfd_to(follower.rank, cells[index])
                follower.channel.send_descriptor(watcher)
            except SeamError as exc:
                raise GroupRefusal(
                    f"{follower.gpu} cannot be commanded: {exc}", code="group_broken"
                ) from exc
            finally:
                watcher.close()
        dist.init_process_group(self.backend, store=self.store, rank=0, world_size=self.degree)
        self.pg = dist.group.WORLD
        dist.barrier()
        if formed is not None:
            formed(self.pg)
        replies = self.collect("join")
        for follower, reply in zip(self.followers, replies, strict=True):
            if not reply.get("ok"):
                raise GroupRefusal(
                    f"{follower.gpu} could not join the group: {reply.get('code')}: "
                    f"{reply.get('detail', '')}"[:400],
                    code="group_unformed",
                )
        self.gave = sum(int(reply.get("freed") or 0) for reply in replies)
        self.formed, self.interrupted = True, False
        return self.pg

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

    def collect(self, name: str) -> list[dict[str, Any]]:
        """One reply named `name` from every follower, in rank order.

        The wait is on the follower's OWN progress meter: sampled every `sample_seconds`,
        wedged only against the pauses it has itself shown (`liveness.Pace`). Death is
        observed through the seam, not inferred from silence.
        """
        return [self._reply(follower, name) for follower in self.followers]

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

    def memfd_to(self, rank: int, memfd: int) -> None:
        """Hand follower `rank` one pinned host tier (right after the frame naming it)."""
        try:
            self.followers[rank - 1].channel.send_memfd(memfd)
        except SeamError as exc:
            raise GroupRefusal(
                f"{self.gpu(rank)} cannot take a host tier: {exc}", code="group_broken"
            ) from exc

    def reply_one(self, rank: int, name: str) -> dict[str, Any]:
        return self._reply(self.followers[rank - 1], name)

    def _reply(self, follower: Follower, name: str) -> dict[str, Any]:
        pace = liveness.Pace()
        floor = liveness.noise_floor(self.sample_seconds)
        while True:
            try:
                frame = follower.replies.get(timeout=self.sample_seconds)
            except queue.Empty:
                if follower.relaying:
                    continue
                pace.observe(liveness.burn(follower.pid))
                if pace.wedged(floor):
                    self._break(f"{follower.gpu} wedged during {name!r}: {pace.verdict(floor)}")
                    raise GroupRefusal(self.broken, code="group_broken") from None
                continue
            if frame is None:
                raise GroupRefusal(f"{self.verdict()} during {name!r}", code="group_broken")
            head, body = frame
            if head.reply != name:
                refused = f": {head.code}: {head.detail}"[:600] if head.ok is False else ""
                self._break(f"{follower.gpu} answered {head.reply!r} to {name!r}{refused}")
                raise GroupRefusal(self.broken, code="group_broken")
            return body

    def agree_residence(
        self,
        *,
        model: str,
        plan: Mapping[str, object],
        required: int,
        park: Callable[[int], list[str]],
        ceiling: Callable[[], int],
    ) -> tuple[int, list[str]]:
        """Agree the plan, the constructions to park, and the ceiling, before any fill.

        Every rank resolved on its own card; each offer carries its plan and the bytes it
        needs. A plan that differs is divergence, refused before a destination exists. Rank
        0 parks for the largest need any rank reported and every rank parks the same
        constructions; each then measures its own ceiling and the group fills under the
        smallest. The control seam also observes an early load failure or rank death,
        without parking the leader in a device collective.
        """
        need = int(required)
        mine = GroupPlan.read(plan)
        for follower, reply in zip(self.followers, self.collect("resident_budget"), strict=True):
            offer = _decode(reply, _Offer, follower.gpu, "resident_budget", "invalid GPU offer")
            if (offer.rank, offer.model) != (follower.rank, model):
                raise GpuDivergence(follower.gpu, "resident_budget", "invalid GPU offer")
            offer.plan.assert_agrees(mine, rank=follower.rank)
            need = max(need, offer.required)
        parked = park(need)
        self.broadcast(msgspec.to_builtins(_Parking(model, parked)))
        shared = ceiling()
        for follower, reply in zip(self.followers, self.collect("resident_ceiling"), strict=True):
            measured = _decode(
                reply,
                _Measured,
                follower.gpu,
                "resident_budget_bytes",
                "invalid GPU residence ceiling",
            )
            if (measured.rank, measured.model) != (follower.rank, model):
                raise GpuDivergence(
                    follower.gpu, "resident_budget_bytes", "invalid GPU residence ceiling"
                )
            shared = min(shared, measured.ceiling)
        self.broadcast(msgspec.to_builtins(_Ceiling(model, shared)))
        return shared, parked

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
            if follower.abort is not None:
                follower.abort.close()
        if pg is not None:
            with contextlib.suppress(Exception):
                importlib.import_module("torch.distributed").destroy_process_group()
        self.store = None
        self.formed = False
