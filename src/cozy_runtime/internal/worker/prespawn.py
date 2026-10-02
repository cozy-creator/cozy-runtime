"""Start a serving installation's executor before its call reaches it (h3a-087, startup warm).

An executor's start is CPU work: two interpreters, torch, the Runtime and the package (4-8 s
for SDXL and Anima, ~14 s for H3). None of it needs a GPU or a weight byte, so as soon as the
machine knows an installation will run on a GPU set it starts that installation's process
there at the lowest CPU priority and imports everything without touching a device
(`Start(import_only)`), which also submits the fused glue's compile. Its hello also tells the
stage scheduler whether the installation takes stage turns, so a queued call can prepare ahead.

Four things say an installation will run, and where: a queued call the stage scheduler placed
(`waiting`), a child call preparing its models (`request`), the Host landing weights for a root
not submitted yet (`request_landing`, h3a-089), and a restarted machine's journal (`prewarm`:
the installations each GPU set served last). The replica that binds those GPUs adopts the
process (`claim`/`settle`) and its start only initializes the device.

The seal (`CUDA_VISIBLE_DEVICES`) is fixed at spawn, so the GPUs must be known first: the
scheduler's placement, or where it would place such a call now. Nothing here reserves a GPU or
allocates device memory. A process yields when its installation's replica claims another set,
or its installation is collected. An executor that predates import-only starts
(`import_only` absent from its hello) is only spawned: sent the start it does not know, it would
initialize the device. A failure anywhere is a missed saving, never a verdict: the replica then
starts its own process exactly as it did before this existed.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import json
import os
import resource
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cozy_runtime.internal import package_environment, package_installation, package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.execution_evidence import gpu_names
from cozy_runtime.internal.executor_commands import Start
from cozy_runtime.internal.worker import machine_lanes
from cozy_runtime.internal.worker.child import Executor, ExecutorGone, ExecutorSupervision
from cozy_runtime.internal.worker.lanes import DeviceLane, lane_id_for
from cozy_runtime.internal.worker.stage_progress import Emit
from cozy_runtime.internal.worker.stage_scheduler import Ordinals, Want

if TYPE_CHECKING:
    from .machine_child_target import Target
    from .session import HostedPlacement, Worker


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
    #: the interpreter of the installation generation the process was started from
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


#: Installations prewarmed per GPU set at boot: the two it served last, the pair an
#: alternating workload (SDXL, then Anima, then SDXL) comes back to after a restart.
PREWARM = 2


class Prespawns:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.lock = threading.Lock()
        #: serializes slot creation, never held while a process starts
        self.starting = threading.Lock()
        self.slots: dict[tuple[str, Ordinals], Slot] = {}

    def waiting(self, key: str, want: Want, interface: bytes, emit: Emit) -> None:
        """Start the executor a queued call will run in, on the GPUs it was placed on."""
        try:
            ordinals = self.worker.stages.planned(key)
            if ordinals:
                self._request(want.root, want.template[0], interface, ordinals, emit)
        except Exception as exc:  # a missed saving, never the execution's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def request(self, owner: str, parent: str, target: Target, emit: Emit) -> None:
        """Start `target`'s executor while its call prepares, for the root `parent`
        descends from, where the scheduler would place it."""
        try:
            assert self.worker.executions is not None
            root, _ = self.worker.executions.scheduling_root(owner, parent)
            raw = base64.b64decode(
                target.prepared_installation["placement"]["package_interface"], validate=True
            )
            models = target.declaration.get("models") or []
            self._foresee(root, target.installation_id, raw, models, emit)
        except Exception as exc:  # a missed saving, never the preparation's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def request_landing(
        self, installation: str, interface: bytes, models: Sequence[object]
    ) -> None:
        """Start `installation`'s executor while the Host lands the weights of a root not yet
        submitted; the first replica of it on those GPUs adopts it."""
        try:
            self._foresee("", installation, interface, models, None)
        except Exception as exc:  # a missed saving, never the preparation's failure
            self.worker.note("warm", f"no prespawn: {type(exc).__name__}: {exc}"[:256])

    def _foresee(
        self,
        root: str,
        installation: str,
        interface: bytes,
        models: Sequence[object],
        emit: Emit | None,
    ) -> None:
        readable = self.worker.readable_gpus()
        width = _width(models, len(readable))
        if not width:
            return
        excluded = tuple(o for o in range(len(self.worker.lanes.entries)) if o not in readable)
        ordinals = self.worker.stages.predict(
            Want(
                root=root or installation,
                widths=(width,),
                template=(installation, ""),
                exclude=excluded,
            )
        )
        if ordinals:
            self._request(root, installation, interface, ordinals, emit)

    def prewarm(self, owner: str) -> None:
        """A restarted machine's first calls pay no executor start either: the installations
        each GPU set served most recently (`PREWARM`, the pair a workload alternates between),
        read from this machine's own journal, start importing there for no call; a replica of
        one on that set adopts it."""
        assert self.worker.executions is not None
        chosen: dict[Ordinals, list[str]] = {}
        readable = set(self.worker.readable_gpus())
        for request, ordinals in self.worker.executions.last_grants(owner):
            served = chosen.setdefault(ordinals, [])
            if len(served) >= PREWARM or not set(ordinals) <= readable:
                continue
            try:
                shape = machine_lanes.recorded(self.worker, owner, request)
                installation = shape.template[0]
                if shape.kind == "serving" and installation not in served:
                    served.append(installation)
                    self._request("", installation, shape.interface, ordinals, None)
            except Exception as exc:  # a missed saving, never the boot's failure
                self.worker.note("warm", f"no prewarm from {request}: {exc!r}"[:256])

    def _request(
        self,
        root: str,
        installation: str,
        interface: bytes,
        ordinals: Ordinals,
        emit: Emit | None,
    ) -> None:
        """One import-only process per installation and GPU set, unless a replica of it already
        runs there; a later caller's rows follow the one it started."""
        if self.worker.options.install_root is None or not installation:
            return
        # One at a time: two callers for one installation and GPU set (the boot's prewarm and
        # a queued call) must find one slot, never each make a supervision for it.
        with self.starting:
            replica = self._replica(installation, ordinals)
            if replica is not None and (
                replica.supervision.current is not None
                or replica.placement.placement_id in self.worker._activating
            ):
                return  # its replica holds these cards, or is starting on them: nothing to start
            with self.lock:
                slot = self.slots.get((installation, ordinals))
                if slot is not None:
                    slot.roots |= {root} - {""}
            if slot is None:
                slot = self._slot(installation, ordinals, {root} - {""}, emit, replica)
                threading.Thread(
                    target=self._run,
                    args=(slot, interface),
                    name=f"prespawn-{installation}",
                    daemon=True,
                ).start()
                return
        if emit is not None:
            slot.attach(emit)

    def _slot(
        self,
        installation: str,
        ordinals: Ordinals,
        roots: set[str],
        emit: Emit | None,
        replica: HostedPlacement | None,
    ) -> Slot:
        if replica is not None:
            # A dead replica of these cards: its successor starts early, in its own slot.
            slot = Slot(installation, ordinals, roots, replica.supervision, sink=emit)
            slot.replica = replica.placement.placement_id
        else:
            supervision = self.worker._new_placement_supervision(
                "warm-" + _digest([installation, list(ordinals)]),
                uid_key=uid_key(installation, ordinals),
            )
            slot = Slot(installation, ordinals, roots, supervision, sink=emit)
            supervision.on_exit = functools.partial(_forget, supervision)
        with self.lock:
            self.slots[(installation, ordinals)] = slot
        return slot

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
            # A Runtime update leaves the installation's SDK generation behind. Rebuild it
            # first, so no executor starts from the old one (runs 2495, 2575, 2710).
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
            # Its hello says whether the installation takes stage turns: a call queued for
            # it may now prepare ahead of its turn.
            worker.stages.resync()
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

    def sets(self, installation: str) -> list[Ordinals]:
        """Where a started process of this installation waits, so its call is placed there."""
        with self.lock:
            return [key[1] for key in self.slots if key[0] == installation]

    def hellos(self, installation: str) -> list[dict[str, object]]:
        """The hellos of this installation's started processes."""
        with self.lock:
            return [
                slot.executor.hello
                for key, slot in self.slots.items()
                if key[0] == installation and slot.executor is not None
            ]

    def claim(
        self, installation: str, ordinals: Ordinals, sink: Emit | None = None, placement: str = ""
    ) -> Slot | None:
        """The replica about to bind these cards takes the slot started for them; every other
        slot of its installation is now useless and yields. A slot started for no root sends
        its phase rows to `sink`, the claiming execution's journal. The slot stays known until
        `settle`, so a caller arriving while the replica starts finds it and starts nothing.

        Two slots yield instead of being taken, and the replica starts its own process: one
        started from a generation of the installation that a Runtime update has since replaced,
        and one started in another (dead) replica's own executor slot, which `placement` would
        otherwise share with it (run 2493, `executor_replaced_before_entry`)."""
        with self.lock:
            slot = self.slots.get((installation, ordinals))
            yields = slot is not None and (
                bool(slot.replica and placement and slot.replica != placement)
                or not self._current(slot)
            )
            stale = [s for key, s in self.slots.items() if key[0] == installation and s is not slot]
            if slot is not None and yields:
                del self.slots[(installation, ordinals)]
            elif slot is not None:
                slot.claimed.set()
        for other in stale:
            self._drop(other)
        if slot is not None and yields:
            self._retire(slot)  # now: the replica's own start never meets this process
            return None
        if slot is not None:
            if sink is not None:
                slot.attach(sink)
            if slot.executor is not None:
                _nice(slot.executor, _WORKER)
        return slot

    def _current(self, slot: Slot) -> bool:
        """Whether the slot's process runs its installation's current generation."""
        if not slot.python:
            return True  # not spawned yet: its start refreshes the installation first
        try:
            installed = package_installation.open_installation(
                Path(self.worker.options.install_root or ""), slot.installation
            )
        except package_environment.EnvironmentRefusal:
            return False
        return str(installed.python) == slot.python

    def settle(self, slot: Slot, placement_id: str) -> None:
        """Wait out the claimed slot's imports (the replica would pay them anyway), then hand
        its process to the replica at ordinary priority, or retire it so the replica starts
        its own."""
        slot.started.wait()
        with self.lock:
            if self.slots.get((slot.installation, slot.ordinals)) is slot:
                del self.slots[(slot.installation, slot.ordinals)]
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

    def _drop(self, slot: Slot) -> None:
        """Forget `slot` and reclaim its process off the caller's thread. A claimed slot is
        its replica's: `settle` reclaims a start that failed, and the replica starts its own."""
        with self.lock:
            if slot.claimed.is_set():
                return
            if self.slots.get((slot.installation, slot.ordinals)) is slot:
                del self.slots[(slot.installation, slot.ordinals)]
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
            if not slot.claimed.is_set():
                self._retire(slot)


def _width(models: Sequence[object], readable: int) -> int:
    """A model-bearing call's width here: the widest count every model slot declares (one
    always is) that `readable` GPUs form; 0 for no model or no GPU."""
    if not models or not readable:
        return 0
    common = set(range(1, readable + 1))
    for model in models:
        parallel = model.get("sequence_parallel") if isinstance(model, dict) else None
        degrees = parallel.get("degrees") if isinstance(parallel, dict) else None
        declared = {d for d in degrees if type(d) is int} if isinstance(degrees, list) else set()
        common &= {1, *declared}
    return max(common)


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
