"""Start a serving installation's executor before its GPU grant (h3a-087, the startup warm).

An executor's start is CPU work: two interpreters, torch, the Runtime and the package (4-8 s
for SDXL and Anima, ~14 s for H3). None of it needs a GPU or a weight byte, so as soon as the
machine knows an installation will run it starts that installation's process at the lowest
CPU priority and imports everything without touching a device (`Start(import_only)`), which
also submits the fused glue's compile. Three things say an installation will run: a serving
execution waiting for its grant (`waiting`), a child call preparing its models (`request`),
and the Host landing weights for a root it has not submitted yet (`request_landing`,
h3a-089), which the first root to ask for that installation, or be granted its cards for
it, adopts. The grant prefers cards where a process waits (`ordinals`); the replica binding
them adopts the process (`claim`/`settle`), and its start only initializes the device.

The seal (`CUDA_VISIBLE_DEVICES`) is fixed at spawn, so the cards must be known first: every
readable card when the machine has exactly the width, else the root's own lease and free
cards. Where the grant depends on who leaves first, nothing starts. At most one unclaimed
process imports per card set, for the root the scheduler serves first there; the next one
starts when it is claimed or leaves (`_admit`), and roots of one installation share it.

A restarted, idle machine prewarms at boot (`prewarm`): the installation each GPU set last
served, from its own journal, starts for no root on that set.

Nothing here reserves a GPU or allocates device memory. A slot yields when every root waiting
for it ends, when its installation's replica binds other cards or is collected, and, started
for no root, when another installation is granted its cards. An executor that predates
import-only starts (`import_only` absent from its hello) is only spawned: sent the start it
does not know, it would initialize the device. A failure anywhere is a missed saving, never
a verdict: the replica then starts its own process exactly as it did before this existed.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import itertools
import json
import math
import os
import resource
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime.internal import package_environment, package_installation, package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.execution_evidence import gpu_names
from cozy_runtime.internal.executor_commands import Start
from cozy_runtime.internal.worker import machine_lanes
from cozy_runtime.internal.worker.child import Executor, ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.gpu_scheduler import Ordinals, Want, model_degrees, width_for
from cozy_runtime.internal.worker.lanes import DeviceLane, lane_id_for
from cozy_runtime.internal.worker.stage_progress import Emit

if TYPE_CHECKING:
    from .machine_child_target import Target
    from .session import HostedPlacement, Worker


class _Cards(msgspec.Struct, frozen=True):
    """The GPU scheduler's `view()`: each root's leased cards and each grant's root and cards."""

    leases: dict[str, tuple[int, ...]]
    grants: dict[str, tuple[str, tuple[int, ...]]]


@dataclass
class Slot:
    installation: str
    ordinals: Ordinals
    #: the roots waiting for this process; empty until one adopts a Host-started slot
    roots: set[str]
    supervision: ExecutorSupervision
    #: set once the start is done trying; a claim waits on it, never a clock
    started: threading.Event = field(default_factory=threading.Event)
    claimed: threading.Event = field(default_factory=threading.Event)
    #: the process is spawned and, when it can, has imported everything
    ready: bool = False
    executor: Executor | None = None
    #: the dead replica of these cards whose own slot this starts its successor in
    replica: str = ""
    #: the installation generation's interpreter the process was started from
    python: str = ""
    #: where phase rows go; rows wait here until a root's journal takes them
    sink: Emit | None = None
    frames: list[dict[str, Json]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, frame: dict[str, Json]) -> None:
        with self.lock:
            sink = self.sink
            if sink is None:
                self.frames.append(frame)
                return
        sink(frame)

    def attach(self, sink: Emit) -> None:
        """Send this slot's rows to `sink` from now on, the waiting ones first."""
        with self.lock:
            if self.sink is not None:
                return
            self.sink, waiting, self.frames = sink, self.frames, []
            for frame in waiting:
                sink(frame)


@dataclass
class Ask:
    """An installation roots (or the Host's landing) will run, waiting for its cards to have
    no unclaimed slot."""

    installation: str
    interface: bytes
    width: int
    roots: set[str]
    sink: Emit | None
    arrival: int
    #: the card set a boot prewarm names (what it last served); empty: foreseen at admission
    ordinals: Ordinals = ()


class Prespawns:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.lock = threading.Lock()
        self.slots: dict[tuple[str, Ordinals], Slot] = {}
        self.asks: dict[str, Ask] = {}
        #: each open root's scheduling priority, as the last scheduling pass saw it
        self.priorities: dict[str, int] = {}
        self.arrivals = itertools.count()
        self.admitting = threading.Lock()

    def request(self, owner: str, parent: str, target: Target, emit: Emit) -> None:
        """Start `target`'s executor while its call prepares, for the root `parent`
        descends from."""
        try:
            assert self.worker.executions is not None
            root, _ = self.worker.executions.scheduling_root(owner, parent)
            raw = base64.b64decode(
                target.prepared_installation["placement"]["package_interface"], validate=True
            )
            width = self._width(target.declaration.get("models") or [])
            self._request(root, target.installation_id, raw, width, emit)
        except Exception as exc:  # a missed saving, never the preparation's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def request_landing(
        self, installation: str, interface: bytes, models: Sequence[object]
    ) -> None:
        """Start `installation`'s executor while the Host lands the weights of a root not yet
        submitted; the first root to ask for it or be granted its cards adopts it."""
        try:
            self._request("", installation, interface, self._width(models), None)
        except Exception as exc:  # a missed saving, never the preparation's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def waiting(self, want: Want, interface: bytes, emit: Emit) -> None:
        """Start the executor a serving execution waiting for its grant will run in."""
        try:
            self._request(want.root, want.template[0], interface, want.width, emit)
        except Exception as exc:  # a missed saving, never the execution's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def _width(self, models: Sequence[object]) -> int:
        """A model-bearing call's width on this machine; 0 for none."""
        readable = self.worker.readable_gpus()
        return width_for(model_degrees(models), len(readable)) if models and readable else 0

    def prewarm(self, owner: str) -> None:
        """A restarted, idle machine's first grant pays no executor start either: the
        installation each GPU set last served, read from this machine's own journal, starts
        importing there for no root; the first root to ask for it, or be granted those cards
        for it, adopts it."""
        assert self.worker.executions is not None
        for request, ordinals in self.worker.executions.last_grants(owner):
            try:
                shape = machine_lanes.recorded(self.worker, owner, request)
                if shape.kind == "serving":
                    installation, interface = shape.template[0], shape.interface
                    self._request("", installation, interface, len(ordinals), None, ordinals)
            except Exception as exc:  # a missed saving, never the boot's failure
                self.worker.note("warm", f"no prewarm from {request}: {exc!r}"[:256])

    def _request(
        self,
        root: str,
        installation: str,
        interface: bytes,
        width: int,
        emit: Emit | None,
        ordinals: Ordinals = (),
    ) -> None:
        if not width or self.worker.options.install_root is None:
            return
        with self.lock:
            slot = next((s for s in self.slots.values() if s.installation == installation), None)
            adopts = bool(root) and slot is not None and not slot.roots
            if slot is not None:
                # One process serves every root waiting for its installation.
                slot.roots |= {root} - {""}
            elif (ask := self.asks.get(installation)) is not None:
                ask.roots |= {root} - {""}
                ask.sink = ask.sink or emit
            else:
                self.asks[installation] = Ask(
                    installation,
                    interface,
                    width,
                    {root} - {""},
                    emit,
                    next(self.arrivals),
                    ordinals,
                )
        if slot is None:
            self._admit()
            return
        if adopts:
            self.worker.note("warm", f"{root!r} adopts the prespawn on {list(slot.ordinals)}")
        if emit is not None:
            slot.attach(emit)

    def _admit(self) -> None:
        """Start the ask first in line on every card set no unclaimed slot holds: at most one
        process imports ahead per set, for the root the scheduler serves first there; a started
        import keeps its place. Runs again whenever a slot is claimed or leaves and on every
        scheduling pass."""
        with self.admitting:
            with self.lock:
                asks = sorted(self.asks.values(), key=self._precedence)
                held = {o for slot in self.slots.values() for o in slot.ordinals}
            for ask in asks:
                with self.lock:
                    slot = next(
                        (s for s in self.slots.values() if s.installation == ask.installation), None
                    )
                    if slot is not None:  # started meanwhile: its roots wait on that process
                        slot.roots |= ask.roots
                        self.asks.pop(ask.installation, None)
                        continue
                root = min(ask.roots, key=lambda r: self.priorities.get(r, math.inf), default="")
                ordinals = ask.ordinals or self._foreseen(root, ask.width)
                if ask.ordinals and not set(ask.ordinals) <= set(self.worker.readable_gpus()):
                    ordinals = None  # a card set this machine no longer reads
                if ordinals is None or held & set(ordinals):
                    continue
                held |= set(ordinals)
                with self.lock:
                    self.asks.pop(ask.installation, None)
                self._start(ask, ordinals)

    def _readmit(self) -> None:
        """`_admit` off the caller's thread: a claim, a leaving slot or a scheduling pass."""
        if self.asks:
            threading.Thread(target=self._admit, name="prespawn-admit", daemon=True).start()

    def _precedence(self, ask: Ask) -> tuple[float, int]:
        """The GPU scheduler's order: the ask's first root's priority, then arrival. A landing
        no root has asked for yet comes last."""
        return (
            min((self.priorities.get(r, math.inf) for r in ask.roots), default=math.inf),
            ask.arrival,
        )

    def _start(self, ask: Ask, ordinals: Ordinals) -> None:
        replica = self._replica(ask.installation, ordinals)
        if replica is not None and replica.supervision.current is not None:
            return  # a replica's process holds (or is leaving) these cards: nothing to start
        slot_id = "warm-" + _digest([ask.installation, list(ordinals)])
        if replica is not None:
            # A dead replica of these cards: its successor starts early, in its own slot.
            slot = Slot(ask.installation, ordinals, ask.roots, replica.supervision, sink=ask.sink)
            slot.replica = replica.placement.placement_id
        else:
            supervision = self.worker._new_placement_supervision(
                slot_id, uid_key=uid_key(ask.installation, ordinals)
            )
            slot = Slot(ask.installation, ordinals, ask.roots, supervision, sink=ask.sink)
            supervision.on_exit = functools.partial(_forget, supervision)
        with self.lock:
            self.slots[(slot.installation, ordinals)] = slot
        threading.Thread(
            target=self._run, args=(slot, ask.interface), name=f"prespawn-{slot_id}", daemon=True
        ).start()

    def _foreseen(self, root: str, width: int) -> Ordinals | None:
        """The cards `root`'s grant of `width` will take, when that is known before it: every
        readable card when there are exactly that many, else its lease and free cards."""
        readable = self.worker.readable_gpus()
        if len(readable) == width:
            return readable
        unreadable = set(range(len(self.worker.lanes.entries))) - set(readable)
        return self.worker.gpu.spare(root, width, exclude=unreadable)

    def _replica(self, installation: str, ordinals: Ordinals) -> HostedPlacement | None:
        """The replica of this installation already bound to these cards, alive or not."""
        return next(
            (
                hosted
                for hosted in list(self.worker.hosted.values())
                if hosted.placement.installation_id == installation
                and hosted.placement.device_pin == ordinals
            ),
            None,
        )

    def _run(self, slot: Slot, raw: bytes) -> None:
        worker = self.worker
        started, began = time.time(), time.perf_counter()
        lane = _lane(worker, slot.ordinals)
        try:
            # A Runtime update leaves the installation's SDK generation behind: rebuild it
            # first, so no executor starts from the old one (run 2495).
            installed = package_installation.refresh(
                Path(worker.options.install_root or ""),
                slot.installation,
                cache=worker.config.dependency_cache,
            )
            slot.python = str(installed.python)
            interface = package_interface.parse(raw, "prespawned installation")
            path = (
                Path(worker.options.artifact_cache or "")
                / "selections"
                / slot.installation
                / "package-interface.json"
            )
            package_interface.publish(path, raw)
            slot.supervision.use_environment(str(installed.python), slot.installation)
            executor = slot.executor = slot.supervision.spawn(imposed=worker.imposed(lane))
            # The imports yield every CPU to running requests; a claim restores the adopted
            # process's priority, so a worker that cannot restore it leaves it unchanged.
            if not slot.claimed.is_set() and _RESTORABLE:
                _nice(executor, 19)
            if "import_only" in executor.hello.get("memory", ()):
                reply = executor.call(
                    Start(
                        devices=lane.devices,
                        sequence_parallel_degree=len(slot.ordinals),
                        application=interface.application,
                        package_interface=str(path),
                        import_only=True,
                    ),
                    timeout=None,
                )
                if not reply.get("ok"):
                    raise ExecutorGone(f"start refused: {reply.get('code')}: {reply.get('detail')}")
                detail = "; ".join(
                    part
                    for part in (
                        ", ".join(f"{k} {v:.0f}ms" for k, v in reply.get("stages") or ()),
                        str(reply.get("compiles") or ""),
                    )
                    if part
                )
            else:
                detail = (
                    f"executor cozy-runtime {executor.hello.get('runtime_version')} "
                    "imports at its grant"
                )
            slot.ready = True
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            slot.started.set()
        self._phase("Preparing model executor", started, began, slot.ready, slot, detail)
        if not slot.ready:
            worker.note("warm", f"prespawn on {gpu_names(lane.entries)} failed: {detail}")
            self._drop(slot)
            return
        worker.note(
            "warm", f"prespawned {slot.installation} on {gpu_names(lane.entries)}: {detail}"
        )

    def _phase(
        self, name: str, started: float, began: float, ok: bool, slot: Slot, detail: str
    ) -> None:
        slot.emit(
            {
                "kind": "log",
                "name": name,
                "value": "info",
                "at_unix_ms": int(time.time() * 1000),
                "fields": {
                    "phase": name,
                    "completed": ok,
                    "started_unix_ms": int(started * 1000),
                    "elapsed_ms": round((time.perf_counter() - began) * 1000, 3),
                    "ordinals": list(slot.ordinals),
                    "detail": detail[:500],
                },
            }
        )

    def ordinals(self, installation: str) -> Ordinals:
        """Where a started process of this installation waits, so its grant goes there."""
        with self.lock:
            return tuple(
                sorted({o for key in self.slots if key[0] == installation for o in key[1]})
            )

    def claim(
        self, installation: str, ordinals: Ordinals, sink: Emit | None = None, placement: str = ""
    ) -> Slot | None:
        """The replica about to bind these cards takes the slot started for them; every other
        slot of its installation is now useless and yields. A slot started for no root sends
        its phase rows to `sink`, the claiming execution's journal. A slot started in another
        (dead) replica's own executor slot is that replica's: another `placement` sharing it
        would share one executor between two rows, so it yields instead (run 2493)."""
        with self.lock:
            slot = self.slots.pop((installation, ordinals), None)
            stale = [s for key, s in self.slots.items() if key[0] == installation]
            self.asks.pop(installation, None)
        if slot is not None and (
            (slot.replica and placement and slot.replica != placement) or not self._current(slot)
        ):
            stale.append(slot)
            slot = None
        for other in stale:
            self._drop(other)
        self._readmit()
        if slot is not None:
            if sink is not None:
                slot.attach(sink)
            slot.claimed.set()
            if slot.executor is not None:
                _nice(slot.executor, _WORKER)
        return slot

    def _current(self, slot: Slot) -> bool:
        """Whether the slot's process runs its installation's current generation: one a
        refresh has since replaced yields, and the replica starts from the new one."""
        if not slot.python:
            return True
        try:
            installed = package_installation.open_installation(
                Path(self.worker.options.install_root or ""), slot.installation
            )
        except (package_environment.EnvironmentRefusal, OSError):
            return False
        return str(installed.python) == slot.python

    def settle(self, slot: Slot, placement_id: str) -> None:
        """Wait out the claimed slot's imports (the replica would pay them anyway), then hand
        its process to the replica at ordinary priority, or retire it so the replica starts
        its own."""
        slot.started.wait()
        executor = slot.executor
        worker = self.worker
        supervision = slot.supervision
        supervision.on_invalidate = lambda why: worker._hosted_capacity_dropped(placement_id, why)
        supervision.on_exit = lambda gone: worker._hosted_executor_exited(placement_id, gone)
        supervision.on_change = lambda current: worker._row_changed(placement_id, current)
        if not slot.ready or executor is None or not executor.alive():
            if executor is not None:
                self._reclaim(supervision, executor, "prespawned start did not finish")
            return
        _nice(executor, _WORKER)
        worker.note("warm", f"{placement_id!r} adopts prespawned epoch {executor.epoch}")

    def reconcile(
        self,
        roots: Mapping[str, int],
        view: Mapping[str, object],
        installations: Mapping[str, str] | None = None,
    ) -> None:
        """Yield every slot whose roots all ended. A slot started for no root is adopted by the
        one root granted its cards for its installation (`installations`: grant key ->
        installation) and yields when another installation is granted them. Cards another
        root holds never make a slot yield: it holds nothing on them."""
        installations = installations or {}
        self.priorities = dict(roots)
        cards = msgspec.convert(view, _Cards, strict=True)
        taken: dict[int, set[str]] = {}
        granted: dict[int, set[tuple[str, str]]] = {}
        for root, held in cards.leases.items():
            for ordinal in held:
                taken.setdefault(ordinal, set()).add(root)
        for key, (root, held) in cards.grants.items():
            for ordinal in held:
                taken.setdefault(ordinal, set()).add(root)
                granted.setdefault(ordinal, set()).add((root, installations.get(key, "")))
        with self.lock:
            for slot in self.slots.values():
                takers = {r for o in slot.ordinals for r in taken.get(o, ())}
                grants = {g for o in slot.ordinals for g in granted.get(o, ())}
                if (
                    not slot.roots
                    and len(takers) == 1
                    and grants
                    and all(installation == slot.installation for _, installation in grants)
                ):
                    slot.roots = set(takers)
                    self.worker.note(
                        "warm", f"{takers} adopts the prespawn on {list(slot.ordinals)}"
                    )
            leaving = [
                slot
                for slot in self.slots.values()
                if (slot.roots and not slot.roots & roots.keys())
                or (
                    not slot.roots
                    and any(
                        installation != slot.installation
                        for o in slot.ordinals
                        for _, installation in granted.get(o, ())
                    )
                )
            ]
            for ask in [a for a in self.asks.values() if a.roots and not a.roots & roots.keys()]:
                del self.asks[ask.installation]
        for slot in leaving:
            self.worker.note("warm", f"prespawn on {list(slot.ordinals)} yields")
            self._drop(slot)
        self._readmit()

    def _drop(self, slot: Slot) -> None:
        """Forget `slot` and reclaim its process off the caller's thread."""
        with self.lock:
            if self.slots.get((slot.installation, slot.ordinals)) is slot:
                del self.slots[(slot.installation, slot.ordinals)]
        threading.Thread(target=self._retire, args=(slot,), daemon=True).start()
        self._readmit()

    def _retire(self, slot: Slot) -> None:
        """Reclaim the process now, mid-start or not; its warm thread then finds it gone."""
        executor = slot.supervision.current
        if executor is not None:
            self._reclaim(slot.supervision, executor, "prespawned executor yields")
        if not slot.replica:
            slot.supervision.close()

    def _reclaim(self, supervision: ExecutorSupervision, executor: Executor, why: str) -> None:
        try:
            supervision.retire_current(executor, why)
        except ExecutorGone as exc:
            self.worker.note("recovery", f"prespawn reclaim failed: {exc}"[:256])

    def forget(self, installation: str) -> None:
        """Yield every slot of an installation that is being collected."""
        with self.lock:
            leaving = [slot for slot in self.slots.values() if slot.installation == installation]
            self.asks.pop(installation, None)
        for slot in leaving:
            self._drop(slot)

    def close(self) -> None:
        with self.lock:
            slots, self.slots, self.asks = list(self.slots.values()), {}, {}
        for slot in slots:
            self._retire(slot)


def uid_key(installation: str, ordinals: Ordinals) -> str:
    """One executor uid per (installation, cards), so a prespawned process and the replica
    that adopts or later replaces it read and write one kernel-store namespace."""
    return f"{installation}@{','.join(map(str, ordinals))}"


def _lane(worker: Worker, ordinals: Ordinals) -> DeviceLane:
    """The seal of the lane a replica on these cards binds, without binding one."""
    devices = ",".join(worker.lanes.by_id[lane_id_for((o,))].devices for o in ordinals)
    return DeviceLane(
        lane_id_for(ordinals),
        ordinals,
        devices,
        worker_pid=os.getpid(),
        locks=tuple(worker.lanes.by_id[lane_id_for((o,))].locks[0] for o in ordinals),
    )


#: The priority an adopted process returns to: the worker's own.
_WORKER = os.getpriority(os.PRIO_PROCESS, 0)
#: Whether this worker may raise a process back to `_WORKER` after lowering it: root, or
#: an RLIMIT_NICE that reaches it (the ceiling is 20 - its soft limit).
_RESTORABLE = os.geteuid() == 0 or 20 - resource.getrlimit(resource.RLIMIT_NICE)[0] <= _WORKER


def _nice(executor: Executor, value: int) -> None:
    """Every rank shares rank 0's process group. Raising priority back needs the worker's
    privilege; without it the adopted process keeps the warm's priority."""
    with contextlib.suppress(OSError):
        os.setpriority(os.PRIO_PGRP, executor.pid, value)


def _forget(supervision: ExecutorSupervision, gone: Executor) -> None:
    supervision.invalidate(gone, "prespawned executor exited")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]
