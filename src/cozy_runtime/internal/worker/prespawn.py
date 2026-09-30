"""Start a serving call's executor while its weights download (h3a-087, the startup warm).

A download is network-bound; an executor's start is CPU-bound (torch, CUDA and the package:
~14 s for H3) and so is proving its attention kernels. Neither needs a weight byte, so the
preparation that begins a download also starts the placement's process on the GPUs its
root would be granted — ones the root already holds, else free ones — at the lowest CPU
priority, and proves its kernels while those cards are idle. The grant prefers those
ordinals (`ordinals`), the replica adopts the started process (`claim`/`settle`) and its
construction fill begins the moment the weights land.

Two preparations begin a download. A serving call's own (`request`) knows its root. The
Host's, for a root it has not submitted yet (`request_landing`, h3a-089), knows none: its
slot starts on free cards and the first root that asks for the installation, or is granted
one of those cards for it, adopts it. Its phase rows wait in the slot and reach the
adopting root's journal when that root's replica takes the process.

Nothing here reserves a GPU. A slot yields — its process is reclaimed — as soon as another
root leases or is granted one of its cards, when its root ends, or when its installation's
replica lands elsewhere or is collected. A failure anywhere is a missed saving, never a
verdict: the replica then starts its own process exactly as it did before this existed.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import accel, package_installation, package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.execution_evidence import gpu_name, gpu_names
from cozy_runtime.internal.executor_commands import Start, Warm
from cozy_runtime.internal.worker.child import Executor, ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.gpu_scheduler import Ordinals, model_degrees, width_for
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
    #: the root this process is started for; empty until one adopts a Host-started slot
    root: str
    supervision: ExecutorSupervision
    #: set once the process is started (or failed to); a claim waits on it, never a clock
    started: threading.Event = field(default_factory=threading.Event)
    claimed: threading.Event = field(default_factory=threading.Event)
    #: set when the warm thread has nothing more to send this process
    finished: threading.Event = field(default_factory=threading.Event)
    #: set by every scheduling pass, claim and drop: the warm thread looks again
    nudge: threading.Event = field(default_factory=threading.Event)
    executor: Executor | None = None
    #: the dead replica of these cards whose own slot this starts its successor in
    replica: str = ""
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


class Prespawns:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.lock = threading.Lock()
        self.slots: dict[tuple[str, Ordinals], Slot] = {}

    def request(self, owner: str, parent: str, target: Target, emit: Emit) -> None:
        """Start `target`'s executor beside its download, if a GPU set is spare for the root
        `parent` descends from."""
        try:
            assert self.worker.executions is not None
            root, _ = self.worker.executions.scheduling_root(owner, parent)
            raw = base64.b64decode(
                target.prepared_installation["placement"]["package_interface"], validate=True
            )
            models = target.declaration.get("models") or []
            self._request(root, target.installation_id, raw, models, emit)
        except Exception as exc:  # a missed saving, never the preparation's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def request_landing(
        self, installation: str, interface: bytes, models: Sequence[object]
    ) -> None:
        """Start `installation`'s executor while the Host lands the weights of a root not yet
        submitted, on free cards; the first root to ask for it or be granted there adopts it."""
        try:
            self._request("", installation, interface, models, None)
        except Exception as exc:  # a missed saving, never the preparation's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def _request(
        self,
        root: str,
        installation: str,
        interface: bytes,
        models: Sequence[object],
        emit: Emit | None,
    ) -> None:
        worker = self.worker
        entries = worker.lanes.entries
        if not models or not entries or worker.options.install_root is None:
            return
        with self.lock:
            waiting = [s for s in self.slots.values() if s.installation == installation]
            unrooted = next((s for s in waiting if not s.root), None)
            if root and unrooted is not None:
                # The Host started this installation for a root it had not submitted yet.
                unrooted.root = root
            elif root or not waiting:
                unrooted = None
            else:
                return  # already started beside an earlier download of it
        if unrooted is not None:
            worker.note("warm", f"{root!r} adopts the prespawn on {list(unrooted.ordinals)}")
            if emit is not None:
                unrooted.attach(emit)
            unrooted.nudge.set()
            return
        family = accel.host_backend_family()
        readable = [
            o for o, e in enumerate(entries) if accel.device_memory(e, family).state == "measured"
        ]
        width = width_for(model_degrees(models), len(readable))
        unreadable = set(range(len(entries))) - set(readable)
        ordinals = worker.gpu.spare(root, width, exclude=unreadable)
        replica = None if ordinals is None else self._replica(installation, ordinals)
        current = replica.supervision.current if replica is not None else None
        if ordinals is None or (current is not None and current.alive()):
            return
        if current is not None:
            replica = None  # its dead process is still being reclaimed: a slot of its own
        slot_id = "warm-" + _digest([installation, list(ordinals)])
        with self.lock:
            if (installation, ordinals) in self.slots:
                return
            if replica is not None:
                # A dead replica of these cards: its successor starts early, in its own slot.
                slot = Slot(installation, ordinals, root, replica.supervision, sink=emit)
                slot.replica = replica.placement.placement_id
            else:
                supervision = worker._new_placement_supervision(
                    slot_id, uid_key=uid_key(installation, ordinals)
                )
                slot = Slot(installation, ordinals, root, supervision, sink=emit)
                supervision.on_exit = functools.partial(_forget, supervision)
            self.slots[(slot.installation, ordinals)] = slot
        threading.Thread(
            target=self._finishing,
            args=(slot, interface),
            name=f"prespawn-{slot_id}",
            daemon=True,
        ).start()

    def _finishing(self, slot: Slot, interface: bytes) -> None:
        try:
            self._run(slot, interface)
        finally:
            slot.started.set()
            slot.finished.set()

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
        detail = legs = ""
        lane = _lane(worker, slot.ordinals)
        try:
            installed = package_installation.open_installation(
                Path(worker.options.install_root or ""), slot.installation
            )
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
            # The warm yields every CPU to the download and to running requests; a claim
            # restores the adopted process's priority.
            if not slot.claimed.is_set():
                _nice(executor, 19)
            startup = "warm-" + _digest([slot.installation, list(slot.ordinals)])
            with worker.memory.starting(lane, startup):
                reply = executor.call(
                    Start(
                        devices=lane.devices,
                        sequence_parallel_degree=len(slot.ordinals),
                        application=interface.application,
                        package_interface=str(path),
                    ),
                    timeout=None,
                )
            if not reply.get("ok"):
                raise ExecutorGone(f"start refused: {reply.get('code')}: {reply.get('detail')}")
            executor.started = dict(reply)
            legs = ", ".join(f"{k} {v:.0f}ms" for k, v in reply.get("stages") or ())
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            slot.started.set()
        self._phase("Starting model executor", started, began, not detail, slot, detail or legs)
        if detail:
            worker.note("warm", f"prespawn on {gpu_names(lane.entries)} failed: {detail}")
            self._drop(slot)
            return
        worker.note("warm", f"prespawned {slot.installation} on {gpu_names(lane.entries)}: {legs}")
        # Launch checks wait for idle cards; the compiles they start run beside the download.
        while self._busy(slot.ordinals):
            slot.nudge.wait()
            slot.nudge.clear()
            if slot.claimed.is_set() or not self._held(slot):
                return
        if slot.claimed.is_set() or not self._held(slot):
            return
        started, began = time.time(), time.perf_counter()
        try:
            reply = executor.call(Warm(), timeout=None)
            ranks = reply.get("ranks") or []
            detail = "; ".join(
                f"{gpu_name(lane.entries, int(row.get('rank', -1)))}: "
                + ", ".join(
                    f"{k['kernel']} {k.get('line') or k['status']}" for k in row.get("kernels", ())
                )
                for row in ranks
            ) or str(reply.get("skipped") or reply.get("detail") or "")
            ok = bool(reply.get("ok"))
            detail = "; ".join(
                part for part in (_before_boot_compile(executor) + detail, _machine(worker)) if part
            )
        except ExecutorGone as exc:
            detail, ok = f"executor gone: {exc}"[:300], False
        self._phase("Starting kernel compiles", started, began, ok, slot, detail)
        worker.note("warm", f"prespawned {slot.installation} kernels: {detail}"[:256])

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

    def _held(self, slot: Slot) -> bool:
        with self.lock:
            return self.slots.get((slot.installation, slot.ordinals)) is slot

    def _busy(self, ordinals: Ordinals) -> bool:
        grants = msgspec.convert(self.worker.gpu.view(), _Cards, strict=True).grants
        return any(set(held) & set(ordinals) for _root, held in grants.values())

    def ordinals(self, installation: str) -> Ordinals:
        """Where a started process of this installation waits, so its grant goes there."""
        with self.lock:
            return tuple(
                sorted({o for key in self.slots if key[0] == installation for o in key[1]})
            )

    def claim(self, installation: str, ordinals: Ordinals, sink: Emit | None = None) -> Slot | None:
        """The replica about to bind these cards takes the slot started for them; every other
        slot of its installation is now useless and yields. A slot started for no root sends
        its phase rows to `sink`, the claiming execution's journal."""
        with self.lock:
            slot = self.slots.pop((installation, ordinals), None)
            stale = [s for key, s in self.slots.items() if key[0] == installation]
        for other in stale:
            self._drop(other)
        if slot is not None:
            if sink is not None:
                slot.attach(sink)
            slot.claimed.set()
            slot.nudge.set()
            if slot.executor is not None:
                _nice(slot.executor, _WORKER)
        return slot

    def settle(self, slot: Slot, placement_id: str) -> None:
        """Wait out the claimed slot's start (the replica would pay it anyway), then hand
        its process to the replica at ordinary priority, or retire it so the replica starts
        its own."""
        slot.started.wait()
        executor = slot.executor
        worker = self.worker
        supervision = slot.supervision
        supervision.on_invalidate = lambda why: worker._hosted_capacity_dropped(placement_id, why)
        supervision.on_exit = lambda gone: worker._hosted_executor_exited(placement_id, gone)
        supervision.on_change = lambda current: worker._row_changed(placement_id, current)
        if executor is None or not executor.started or not executor.alive():
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
        """Yield every slot another root now leases or was granted a card of, and every slot
        whose root ended. A slot started for no root is adopted by the one root granted its
        cards for its installation (`installations`: grant key -> installation)."""
        installations = installations or {}
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
                    not slot.root
                    and len(takers) == 1
                    and grants
                    and all(installation == slot.installation for _, installation in grants)
                ):
                    (slot.root,) = takers
                    self.worker.note(
                        "warm", f"{slot.root!r} adopts the prespawn on {list(slot.ordinals)}"
                    )
            leaving = [
                slot
                for slot in self.slots.values()
                if (slot.root and slot.root not in roots)
                or any(taken.get(o, set()) - {slot.root} for o in slot.ordinals)
            ]
            for slot in self.slots.values():
                slot.nudge.set()
        for slot in leaving:
            self.worker.note("warm", f"prespawn on {list(slot.ordinals)} yields its cards")
            self._drop(slot)

    def _drop(self, slot: Slot) -> None:
        """Forget `slot` and reclaim its process off the caller's thread."""
        with self.lock:
            if self.slots.get((slot.installation, slot.ordinals)) is slot:
                del self.slots[(slot.installation, slot.ordinals)]
        slot.nudge.set()
        threading.Thread(target=self._retire, args=(slot,), daemon=True).start()

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
        for slot in leaving:
            self._drop(slot)

    def close(self) -> None:
        with self.lock:
            slots, self.slots = list(self.slots.values()), {}
        for slot in slots:
            slot.nudge.set()
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


#: The first Runtime whose executor compiles its kernels on the machine (e345e63c).
BOOT_COMPILE = Version("0.18.70")


def _before_boot_compile(executor: Executor) -> str:
    """An older executor compiles nothing and reads only the image kernel site: its rows
    say absent until the machine's build of a kernel lands there, and it takes one at its
    next construction. The package's lock picks it."""
    found = str(executor.hello.get("runtime_version", ""))
    try:
        if Version(found) >= BOOT_COMPILE:
            return ""
    except InvalidVersion:
        return ""
    return (
        f"executor cozy-runtime {found} (before {BOOT_COMPILE}) takes each machine kernel "
        "from the image kernel site at its next construction after it is ready; "
    )


def _machine(worker: Worker) -> str:
    machine = worker.machine_kernels
    return machine.line() if machine is not None else ""


def _nice(executor: Executor, value: int) -> None:
    """Every rank shares rank 0's process group. Raising priority back needs the worker's
    privilege; without it the adopted process keeps the warm's priority."""
    with contextlib.suppress(OSError):
        os.setpriority(os.PRIO_PGRP, executor.pid, value)


def _forget(supervision: ExecutorSupervision, gone: Executor) -> None:
    supervision.invalidate(gone, "prespawned executor exited")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]
