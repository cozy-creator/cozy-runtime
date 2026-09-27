"""GPU scheduling policy: one lease per root, exact device sets, level-triggered.

Policy only, no I/O. Every drive tick passes the open roots with their durable priority
and every demand not yet released from its device; anything missing is gone, so a hold
cannot leak. A root's lease grows only from free ordinals and ends when the root stops or
says it is done with GPUs (`release_root`, the author's `ctx.release_gpus()`; a root that is
itself the GPU call is done at its device exit), so the Python gap between two children
never lets another root in. Priority orders the waiters: the oldest waiter that cannot start
reserves the free ordinals, so a younger narrow demand never starves it. Nothing is revoked,
lent or preempted.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Ordinals = tuple[int, ...]


@dataclass(frozen=True)
class Demand:
    """One GPU phase of an execution attempt: `key` is `request#attempt`."""

    key: str
    root: str
    width: int
    warm: Ordinals = ()  # where this exact placement is already resident
    exclude: Ordinals = ()  # ordinals it must not take


def model_degrees(models: Iterable[Any]) -> list[set[int]]:
    """Declared `sequence_parallel.degrees` per model slot; an unreadable field is none."""
    rows = []
    for model in models:
        parallel = model.get("sequence_parallel") if isinstance(model, dict) else None
        degrees = parallel.get("degrees") if isinstance(parallel, dict) else None
        rows.append({d for d in degrees if type(d) is int} if isinstance(degrees, list) else set())
    return rows


def width_for(degrees: Iterable[Iterable[int]], devices: int, exact: int = 0) -> int:
    """The largest device count every model slot declares, capped by the machine.

    One device is always declared. Authored rung GPU counts are a renting decision and
    never narrow this on a machine that is already wider. An `exact` count the owner
    submitted is the width itself when every slot declares it and the machine forms it;
    otherwise there is no width (0).
    """
    common = set(range(1, max(devices, 1) + 1))
    for declared in degrees:
        common &= {1, *declared}
    if exact:
        return exact if exact in common else 0
    return max(common)


class GpuScheduler:
    def __init__(self, devices: int) -> None:
        self.devices = devices
        self.lock = threading.Lock()
        self.leases: dict[str, set[int]] = {}
        self.grants: dict[str, tuple[str, Ordinals]] = {}
        self.waiting: dict[str, tuple[str, int, tuple[str, ...]]] = {}
        self.released: set[str] = set()
        #: roots done with GPUs -> why: their gap holds nothing until their next demand
        self.yielded: dict[str, str] = {}
        self.events: list[tuple[str, str, dict[str, Any]]] = []
        self._arrival: dict[str, int] = {}
        self._counter = itertools.count()

    def sync(self, roots: Mapping[str, int], demands: Sequence[Demand]) -> dict[str, Ordinals]:
        """Reconcile to this tick's roots (name -> priority) and demands; return grants."""
        with self.lock:
            # A stopped root's attempt keeps its grant until it leaves the device.
            live = {d.key: d for d in demands if d.key not in self.released}
            self.released &= {d.key for d in demands}
            for key in [key for key in self.grants if key not in live]:
                self._end(key, "gone")
            self.waiting = {k: v for k, v in self.waiting.items() if k in live}
            for root in [root for root in self.leases if root not in roots]:
                self._lease(root, set(), "stopped")
            self.yielded = {r: c for r, c in self.yielded.items() if r in roots}
            self._arrival = {k: self._arrival.get(k, next(self._counter)) for k in live}
            busy = {o for _, ordinals in self.grants.values() for o in ordinals}
            stalled: set[str] = set()
            pending = sorted(
                (d for d in live.values() if d.key not in self.grants and d.root in roots),
                key=lambda d: (roots[d.root], d.root, self._arrival[d.key]),
            )
            for demand in pending:  # a GPU call after a release starts a new lease
                self.yielded.pop(demand.root, None)
            for demand in pending:
                if demand.root in stalled:
                    continue  # FIFO within a root: a later call never overtakes an earlier one
                lease = self.leases.get(demand.root, set())
                taken = {o for held in self.leases.values() for o in held}
                usable = [
                    o
                    for o in range(self.devices)
                    if o not in busy and o not in demand.exclude and (o in lease or o not in taken)
                ]
                if len(usable) >= demand.width:
                    ranked = sorted(usable, key=lambda o: (o not in demand.warm, o not in lease, o))
                    chosen = tuple(sorted(ranked[: demand.width]))
                    self.grants[demand.key] = (demand.root, chosen)
                    self.waiting.pop(demand.key, None)
                    busy.update(chosen)
                    self._lease(demand.root, lease | set(chosen), f"granted {demand.key}")
                    self.events.append(
                        (demand.root, "gpu.grant", {"key": demand.key, "ordinals": list(chosen)})
                    )
                    continue
                if not stalled:
                    # The head of the line reserves what is free; nothing else moves.
                    self._lease(demand.root, lease | set(usable), f"reserved for {demand.key}")
                stalled.add(demand.root)
                blocked = tuple(
                    sorted(
                        {r for r, _ in self.grants.values()}
                        | {r for r, held in self.leases.items() if held and r != demand.root}
                    )
                )
                if self.waiting.get(demand.key) != (demand.root, demand.width, blocked):
                    self.waiting[demand.key] = (demand.root, demand.width, blocked)
                    self.events.append(
                        (
                            demand.root,
                            "gpu.wait",
                            {"key": demand.key, "width": demand.width, "blocked_by": list(blocked)},
                        )
                    )
            return {key: ordinals for key, (_, ordinals) in self.grants.items()}

    def release(self, key: str, **facts: Any) -> None:
        """Device exit: the ordinals return to the root's lease before the next tick. A root
        that is itself this call (a direct serving or device-job root) has no GPU work after
        it, so its lease ends here and its encode and output transfer hold no GPU."""
        with self.lock:
            self.released.add(key)
            if key in self.grants:
                root = self.grants[key][0]
                self._end(key, "exited", facts)
                if key.rpartition("#")[0] == root:
                    self._yield(root, "device_exit")

    def release_root(self, root: str, cause: str = "released_by_root") -> None:
        """`root` is done with GPUs: its lease ends now and returns to the pool.

        A call still on its GPUs keeps them until its exit, and they return to the pool
        then. The root's next demand waits at its own priority like any other and starts
        a new lease; it never takes GPUs back from whoever was granted them. Idempotent."""
        with self.lock:
            self._yield(root, cause)

    def drain(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Observations since the last drain, for the caller to journal outside any lock."""
        with self.lock:
            events, self.events = self.events, []
            return events

    def spare(self, root: str, width: int, exclude: Iterable[int] = ()) -> Ordinals | None:
        """`width` cards `root` could be granted without taking another root's: its own lease
        first, then free ones. A look, never a hold: nothing here reserves or yields."""
        with self.lock:
            others = {o for r, held in self.leases.items() if r != root for o in held}
            others |= {o for r, held in self.grants.values() if r != root for o in held}
            lease = self.leases.get(root, set())
            usable = sorted(
                (o for o in range(self.devices) if o not in others and o not in set(exclude)),
                key=lambda o: (o not in lease, o),
            )
            return tuple(sorted(usable[:width])) if len(usable) >= width else None

    def granted(self, key: str) -> Ordinals | None:
        with self.lock:
            grant = self.grants.get(key)
            return grant[1] if grant is not None else None

    def status(self, request: str, on_device: Callable[[str], bool]) -> dict[str, Any]:
        """One execution and, for a root, its calls: waiting, executing, granted, holding."""
        with self.lock:

            def mine(key: str, root: str) -> bool:
                return root == request or key.rpartition("#")[0] == request

            status: dict[str, Any] = {"phase": "none", "width": 0, "ordinals": [], "blocked_by": []}
            for key, (root, width, blocked) in self.waiting.items():
                if mine(key, root):
                    return {
                        **status,
                        "phase": "waiting",
                        "width": width,
                        "blocked_by": list(blocked),
                    }
            granted = {key: o for key, (root, o) in self.grants.items() if mine(key, root)}
            if granted:
                return {
                    **status,
                    "phase": "executing" if any(map(on_device, granted)) else "granted",
                    "width": max(map(len, granted.values())),
                    "ordinals": sorted({o for ordinals in granted.values() for o in ordinals}),
                }
            if self.leases.get(request):
                return {**status, "phase": "holding", "ordinals": sorted(self.leases[request])}
            return status

    def view(self) -> dict[str, Any]:
        with self.lock:
            return {
                "leases": {root: sorted(held) for root, held in self.leases.items()},
                "grants": {key: [root, list(o)] for key, (root, o) in self.grants.items()},
                "waiting": {key: list(blocked) for key, (_, _, blocked) in self.waiting.items()},
            }

    def _end(self, key: str, cause: str, facts: dict[str, Any] | None = None) -> None:
        root, ordinals = self.grants.pop(key)
        body = {"key": key, "ordinals": list(ordinals), "cause": cause, **(facts or {})}
        self.events.append((root, "gpu.release", body))
        if root in self.yielded:
            self._lease(root, self.leases.get(root, set()) - set(ordinals), self.yielded[root])

    def _yield(self, root: str, cause: str) -> None:
        if root in self.yielded:
            return
        self.yielded[root] = cause
        active = {o for r, ordinals in self.grants.values() if r == root for o in ordinals}
        self._lease(root, active, cause)

    def _lease(self, root: str, ordinals: set[int], cause: str) -> None:
        if ordinals != self.leases.get(root, set()):
            self.events.append((root, "gpu.lease", {"ordinals": sorted(ordinals), "cause": cause}))
        if ordinals:
            self.leases[root] = ordinals
        else:
            self.leases.pop(root, None)
