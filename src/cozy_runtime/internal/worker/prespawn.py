"""Start a serving call's executor while its weights download (h3a-087, the startup warm).

A download is network-bound; an executor's start is CPU-bound (torch, CUDA and the package:
~14 s for H3) and so is proving its attention kernels. Neither needs a weight byte, so the
preparation that begins a download also starts the placement's process on the GPUs its
root would be granted — ones the root already holds, else free ones — at the lowest CPU
priority, and proves its kernels while those cards are idle. The grant prefers those
ordinals (`ordinals`), the replica adopts the started process (`claim`/`settle`) and its
construction fill begins the moment the weights land.

Nothing here reserves a GPU. A slot yields — its process is reclaimed — as soon as another
root leases or is granted one of its cards, when its root ends, or when its installation's
replica lands elsewhere. A failure anywhere is a missed saving, never a verdict: the
replica then starts its own process exactly as it did before this existed.
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime.internal import accel, package_installation, package_interface
from cozy_runtime.internal.worker.child import Executor, ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.gpu_scheduler import Ordinals, model_degrees, width_for
from cozy_runtime.internal.worker.lanes import DeviceLane, lane_id_for

if TYPE_CHECKING:
    from .machine_child_target import Target
    from .session import Worker

Emit = Callable[[dict[str, Any]], None]


@dataclass
class Slot:
    installation: str
    ordinals: Ordinals
    root: str
    supervision: ExecutorSupervision
    #: set once the process is started (or failed to); a claim waits on it, never a clock
    started: threading.Event = field(default_factory=threading.Event)
    claimed: threading.Event = field(default_factory=threading.Event)
    #: set when the warm thread has nothing more to send this process
    finished: threading.Event = field(default_factory=threading.Event)
    executor: Executor | None = None


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
            self._request(root, target, emit)
        except Exception as exc:  # a missed saving, never the preparation's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def _request(self, root: str, target: Target, emit: Emit) -> None:
        worker = self.worker
        models = target.declaration.get("models") or []
        entries = worker.lanes.entries
        if not models or not entries or worker.options.install_root is None:
            return
        family = accel.host_backend_family()
        readable = [
            o for o, e in enumerate(entries) if accel.device_memory(e, family).state == "measured"
        ]
        width = width_for(model_degrees(models), len(readable))
        unreadable = set(range(len(entries))) - set(readable)
        ordinals = worker.gpu.spare(root, width, exclude=unreadable)
        if ordinals is None or self._warm(target.installation_id, ordinals):
            return
        slot_id = "warm-" + _digest([target.installation_id, list(ordinals)])
        with self.lock:
            if (target.installation_id, ordinals) in self.slots:
                return
            supervision = worker._new_placement_supervision(
                slot_id, uid_key=uid_key(target.installation_id, ordinals)
            )
            slot = Slot(target.installation_id, ordinals, root, supervision)
            supervision.on_exit = functools.partial(_forget, supervision)
            self.slots[(slot.installation, ordinals)] = slot
        threading.Thread(
            target=self._finishing,
            args=(slot, target, emit),
            name=f"prespawn-{slot_id}",
            daemon=True,
        ).start()

    def _finishing(self, slot: Slot, target: Target, emit: Emit) -> None:
        try:
            self._run(slot, target, emit)
        finally:
            slot.started.set()
            slot.finished.set()

    def _warm(self, installation: str, ordinals: Ordinals) -> bool:
        """A replica of this installation already runs on these cards."""
        for hosted in list(self.worker.hosted.values()):
            executor = hosted.supervision.current
            if (
                hosted.placement.installation_id == installation
                and hosted.placement.device_pin == ordinals
                and executor is not None
                and executor.alive()
            ):
                return True
        return False

    def _run(self, slot: Slot, target: Target, emit: Emit) -> None:
        worker, placement = self.worker, target.prepared_installation["placement"]
        started, began = time.time(), time.perf_counter()
        detail = legs = ""
        try:
            installed = package_installation.open_installation(
                Path(worker.options.install_root or ""), target.installation_id
            )
            raw = base64.b64decode(placement["package_interface"], validate=True)
            interface = package_interface.read_bytes(raw, "prespawned installation")
            path = (
                Path(worker.options.artifact_cache or "")
                / "selections"
                / target.installation_id
                / "package-interface.json"
            )
            package_interface.publish(path, raw)
            slot.supervision.use_environment(str(installed.python), target.installation_id)
            lane = _lane(worker, slot.ordinals)
            executor = slot.executor = slot.supervision.spawn(imposed=worker.imposed(lane))
            # The warm yields every CPU to the download and to running requests; a claim
            # restores the adopted process's priority.
            if not slot.claimed.is_set():
                _nice(executor, 19)
            reply = executor.call(
                {
                    "cmd": "start",
                    "devices": lane.devices,
                    "sequence_parallel_degree": len(slot.ordinals),
                    "application": str(interface.get("application", "")),
                    "package_interface": str(path),
                    "development_project": "",
                },
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
        self._phase(
            emit, "Starting model executor", started, began, not detail, slot, detail or legs
        )
        if detail:
            worker.note("warm", f"prespawn on {list(slot.ordinals)} failed: {detail}")
            self._drop(slot)
            return
        worker.note("warm", f"prespawned {slot.installation} on {list(slot.ordinals)}: {legs}")
        if slot.claimed.is_set() or self._busy(slot.ordinals):
            return  # a request is ready: its construction proves what it needs itself
        with self.lock:
            if self.slots.get((slot.installation, slot.ordinals)) is not slot:
                return  # yielded
        started, began = time.time(), time.perf_counter()
        try:
            reply = executor.call({"cmd": "warm"}, timeout=None)
            ranks = reply.get("ranks") or []
            detail = "; ".join(
                f"rank {row.get('rank')}: "
                + ", ".join(f"{k['kernel']} {k['status']}" for k in row.get("kernels", ()))
                for row in ranks
            ) or str(reply.get("skipped") or reply.get("detail") or "")
            ok = bool(reply.get("ok"))
        except ExecutorGone as exc:
            detail, ok = f"executor gone: {exc}"[:300], False
        self._phase(emit, "Proving attention kernels", started, began, ok, slot, detail)
        worker.note("warm", f"prespawned {slot.installation} kernels: {detail}"[:256])

    def _phase(
        self, emit: Emit, name: str, started: float, began: float, ok: bool, slot: Slot, detail: str
    ) -> None:
        emit(
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

    def _busy(self, ordinals: Ordinals) -> bool:
        grants = self.worker.gpu.view()["grants"]
        return any(set(held) & set(ordinals) for _root, held in grants.values())

    def ordinals(self, installation: str) -> Ordinals:
        """Where a started process of this installation waits, so its grant goes there."""
        with self.lock:
            return tuple(
                sorted({o for key in self.slots if key[0] == installation for o in key[1]})
            )

    def claim(self, installation: str, ordinals: Ordinals) -> Slot | None:
        """The replica about to bind these cards takes the slot started for them; every other
        slot of its installation is now useless and yields."""
        with self.lock:
            slot = self.slots.pop((installation, ordinals), None)
            stale = [s for key, s in self.slots.items() if key[0] == installation]
        for other in stale:
            self._drop(other)
        if slot is not None:
            slot.claimed.set()
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

    def reconcile(self, roots: Mapping[str, int], view: Mapping[str, Any]) -> None:
        """Yield every slot another root now leases or was granted a card of, and every slot
        whose root ended."""
        taken: dict[int, set[str]] = {}
        for root, held in view["leases"].items():
            for ordinal in held:
                taken.setdefault(ordinal, set()).add(root)
        for root, held in view["grants"].values():
            for ordinal in held:
                taken.setdefault(ordinal, set()).add(root)
        with self.lock:
            leaving = [
                slot
                for slot in self.slots.values()
                if slot.root not in roots
                or any(taken.get(o, set()) - {slot.root} for o in slot.ordinals)
            ]
        for slot in leaving:
            self.worker.note("warm", f"prespawn on {list(slot.ordinals)} yields its cards")
            self._drop(slot)

    def _drop(self, slot: Slot) -> None:
        """Forget `slot` and reclaim its process off the caller's thread."""
        with self.lock:
            if self.slots.get((slot.installation, slot.ordinals)) is slot:
                del self.slots[(slot.installation, slot.ordinals)]
        threading.Thread(target=self._retire, args=(slot,), daemon=True).start()

    def _retire(self, slot: Slot) -> None:
        """Reclaim the process now, mid-start or not; its warm thread then finds it gone."""
        executor = slot.supervision.current
        if executor is not None:
            self._reclaim(slot.supervision, executor, "prespawned executor yields")
        slot.supervision.close()

    def _reclaim(self, supervision: ExecutorSupervision, executor: Executor, why: str) -> None:
        try:
            supervision.retire_current(executor, why)
        except ExecutorGone as exc:
            self.worker.note("recovery", f"prespawn reclaim failed: {exc}"[:256])

    def close(self) -> None:
        with self.lock:
            slots, self.slots = list(self.slots.values()), {}
        for slot in slots:
            self._retire(slot)


def uid_key(installation: str, ordinals: Ordinals) -> str:
    """One executor uid per (installation, cards), so a prespawned process and the replica
    that adopts or later replaces it read and write one kernel-store namespace."""
    return f"{installation}@{','.join(map(str, ordinals))}"


def _lane(worker: Worker, ordinals: Ordinals) -> DeviceLane:
    """The seal of the lane a replica on these cards binds, without binding one."""
    devices = ",".join(worker.lanes.by_id[lane_id_for((o,))].devices for o in ordinals)
    return DeviceLane(lane_id_for(ordinals), ordinals, devices, worker_pid=os.getpid())


#: The priority an adopted process returns to: the worker's own.
_WORKER = os.getpriority(os.PRIO_PROCESS, 0)


def _nice(executor: Executor, value: int) -> None:
    """Every rank shares rank 0's process group. Raising priority back needs the worker's
    privilege; without it the adopted process keeps the warm's priority."""
    with contextlib.suppress(OSError):
        os.setpriority(os.PRIO_PGRP, executor.pid, value)


def _forget(supervision: ExecutorSupervision, gone: Executor) -> None:
    supervision.invalidate(gone, "prespawned executor exited")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]
