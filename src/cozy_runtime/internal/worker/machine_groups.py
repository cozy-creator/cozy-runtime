"""Private root priority and GPU groups, projected from the existing execution journal."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json

from . import machine_lanes
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .session import Worker


@dataclass
class Root:
    accepted_ms: int
    width: int = 1
    claimed: bool = False
    preferred: tuple[int, ...] = ()
    downloads: int = 0


def pinned(preparation: dict[str, Any]) -> tuple[int, ...]:
    if not preparation or not preparation.get("state"):
        return ()
    desired = machine_lanes.state(preparation)
    if desired.WhichOneof("mode") != "placement_set":
        return ()
    return tuple(
        sorted(
            {
                ordinal
                for pin in desired.placement_set.device_pins
                for ordinal in pin.device_ordinals
            }
        )
    )


def layout(
    roots: dict[str, Root], width: int, occupied: set[int] | None = None
) -> dict[str, tuple[int, ...]]:
    """Older roots keep priority when their explicit required width grows."""
    free = set(range(width))
    occupied = occupied or set()
    groups = {}
    for name, root in sorted(roots.items(), key=lambda item: (item[1].accepted_ms, item[0])):
        if not root.claimed or root.width <= 1 or root.width > len(free):
            continue
        selected = root.preferred
        if len(selected) != root.width or not set(selected) <= free:
            selected = tuple(
                sorted(
                    sorted(free, key=lambda ordinal: (ordinal in occupied, ordinal))[: root.width]
                )
            )
        groups[name] = selected
        root.preferred = selected
        free.difference_update(selected)
    return groups


class Groups:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.roots: dict[str, Root] = {}
        self.ancestry: dict[str, str] = {}
        self.owner = ""
        self.recovered = False
        self.busy: set[str] = set()
        self.occupied: set[int] = set()
        self.waiting: set[str] = set()
        self.lock = threading.RLock()

    def root(self, owner: str, request: str) -> str:
        executions = self.worker.executions
        assert executions is not None
        if owner != self.owner:
            self.roots.clear()
            self.ancestry.clear()
            self.owner = owner
            self.recovered = False
            self.waiting.clear()
        root = self.ancestry.get(request)
        if root is None:
            root, accepted = executions.scheduling_root(owner, request)
            self.ancestry[request] = root
            self.roots.setdefault(root, Root(accepted))
        return root

    def priority(self, owner: str, request: str) -> tuple[int, str]:
        with self.lock:
            root = self.root(owner, request)
            return self.roots[root].accepted_ms, root

    def announce(self, owner: str, request: str, width: int) -> None:
        with self.lock:
            root = self.roots[self.root(owner, request)]
            root.width = max(root.width, width)

    def transfer(self, owner: str, request: str, change: int) -> None:
        with self.lock:
            root = self.roots.get(self.root(owner, request))
            if root is not None:
                root.downloads = max(0, root.downloads + change)

    def refresh(self, owner: str) -> None:
        executions = self.worker.executions
        assert executions is not None
        with self.lock:
            active = executions.scheduling_roots(owner)
            changed_owner = owner != self.owner
            fresh = changed_owner or not self.recovered
            if changed_owner:
                self.roots.clear()
                self.ancestry.clear()
                self.owner = owner
                self.waiting.clear()
            for name in set(self.roots) - active.keys():
                del self.roots[name]
            self.ancestry = {
                request: root for request, root in self.ancestry.items() if root in active
            }
            self.waiting.intersection_update(self.ancestry)
            for name, accepted in active.items():
                self.roots.setdefault(name, Root(accepted))
            if fresh:
                for name, raw in executions.scheduling_history(owner):
                    group = pinned(canonical_json.decode(raw))
                    record = self.roots.get(name)
                    if record is None:
                        continue
                    if group and len(group) >= record.width:
                        record.width, record.preferred, record.claimed = len(group), group, True
            self.recovered = True
            self.busy.clear()
            self.occupied.clear()
            for request in executions.pending(owner):
                preparation = executions.preparation(owner, request)
                desired = machine_lanes.state(preparation) if preparation else None
                self.occupied.update(pinned(preparation))
                if pinned(preparation) or (
                    desired is not None
                    and desired.WhichOneof("mode") == "job"
                    and not desired.job.orchestration
                ):
                    root = self.root(owner, request)
                    self.busy.add(root)
            for attempt in self.worker.engine.history.values():
                if attempt.state in (
                    *self.worker.engine.PRE_RELEASE,
                    *self.worker.engine.HELD,
                ) and not attempt.lane_id.startswith("cpu-"):
                    lane = self.worker.lanes.by_id.get(attempt.lane_id)
                    if lane is not None:
                        self.occupied.update(lane.ordinals)
                    # Other-owner/public attempts keep the existing physical lane guards.
                    with suppress(WorkspaceRefusal):
                        self.busy.add(self.root(owner, attempt.request_id))
            calls = self.worker.machine_calls
            if calls is not None:
                for (subject, child), future in calls.serving.pending.items():
                    if (
                        subject == owner
                        and future.done()
                        and not future.cancelled()
                        and future.exception() is None
                    ):
                        self.busy.add(self.root(owner, child))

    def available(
        self, owner: str, parent: str, degree: int, loads: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        """Called only for GPU-ready work. Hints announce width without claiming."""
        with self.lock:
            name = self.root(owner, parent)
            root = self.roots[name]
            root.width = max(root.width, degree)
            root.claimed = True
            self.busy.add(name)
            occupied = (
                self.occupied if loads is None else {i for i, load in enumerate(loads) if load}
            )
            groups = layout(self.roots, len(self.worker.lanes.entries), occupied)
            if name in groups:
                return groups[name]
            reserved = {ordinal for group in groups.values() for ordinal in group}
            available = set(range(len(self.worker.lanes.entries))) - reserved
            # A known transfer can lend a group, but a brief no-child handoff cannot.
            for holder, group in groups.items():
                held = self.roots[holder]
                if held.downloads and holder not in self.busy:
                    available.update(group)
            return tuple(sorted(available))

    def admits(self, owner: str, request: str, preparation: dict[str, Any]) -> bool:
        group = pinned(preparation)
        if not group:
            return True
        allowed = self.available(owner, request, len(group))
        admitted = set(group) <= set(allowed)
        if admitted:
            self.waiting.discard(request)
        return admitted

    def wait(self, owner: str, request: str, parent: str, ordinal: int) -> None:
        if request in self.waiting:
            return
        self.waiting.add(request)
        with self.lock:
            root = self.root(owner, parent)
            self.ancestry[request] = root
        self.worker.emit_progress(
            parent,
            ordinal,
            {
                "kind": "progress",
                "stage": "Waiting for GPUs / earlier workflow",
            },
        )
        if root != parent:
            executions = self.worker.executions
            assert executions is not None
            state = executions.status(owner, root)
            self.worker.emit_progress(
                root,
                state.attempt_ordinal,
                {
                    "kind": "progress",
                    "stage": "Waiting for GPUs / earlier workflow",
                },
            )
