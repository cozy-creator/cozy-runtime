"""Lifetime ownership for a committed eager retention scope; no memory budgeting here."""

from __future__ import annotations

import sys
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from cozy_runtime.author._errors import CapabilityError

_LOCK = threading.RLock()
_CREATING: dict[int, dict[object, str]] = {}
_EAGER: dict[int, EagerLease] = {}
_GENERATION: dict[int, int] = {}


class OwnerConflict(CapabilityError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="execution_owner_conflict")


@dataclass(frozen=True)
class EagerCommit:
    """Issued by the execution layer after ruling out preparation for this execution."""

    request: Any
    observation_ref: weakref.ReferenceType[Any]
    device: int
    reason: str

    @property
    def observation(self) -> Any:
        found = self.observation_ref()
        if found is None:
            raise OwnerConflict("the eager execution owner has ended")
        return found

    @classmethod
    def issue(cls, request: Any, observation: Any, reason: str) -> EagerCommit:
        return cls(request, weakref.ref(observation), int(observation.device), reason)


@dataclass
class EagerLease:
    commit: EagerCommit
    completion_uncertain: bool = False


def _changed(device: int) -> None:
    _GENERATION[device] = _GENERATION.get(device, 0) + 1


@contextmanager
def preparing(device: int, kind: str, *, optional: bool = False) -> Iterator[bool]:
    """Claim before allocation; publish into the existing owner registry before release.

    No metadata lock spans allocation, driver work, a callback or completion waits.
    """
    claim = object()
    with _LOCK:
        conflict = device in _EAGER
        if conflict and not optional:
            raise OwnerConflict(f"{kind} conflicts with committed eager retention")
        if not conflict:
            _CREATING.setdefault(device, {})[claim] = kind
            _changed(device)
    if conflict:
        yield False
        return
    try:
        yield True
    finally:
        with _LOCK:
            creating = _CREATING[device]
            del creating[claim]
            if not creating:
                del _CREATING[device]
            _changed(device)


def _blockers(commit: EagerCommit) -> tuple[str, ...]:
    """Read real owners outside the claim lock; creation generation closes the race."""
    import tensorfs.varena.torch as vt
    import torch

    from cozy_runtime import graphs
    from cozy_runtime.author import _loop
    from cozy_runtime.graphs import domain
    from cozy_runtime.internal import prepared
    from cozy_runtime.internal.execution import phase
    from cozy_runtime.internal.residency import _REQUEST

    request, observation = _REQUEST.get(), phase._ACTIVE.get()
    if (
        request is None or observation is None
        or request is not commit.request or observation is not commit.observation
    ):
        return ("no live Loop-committed eager execution owner",)
    memory = request.memory
    reasons: list[str] = []
    if (
        request.idle or not memory.cuda or observation.completion_stream is None
        or observation.eager_commit is not commit or observation.completed
        or (memory.device.index or 0) != commit.device
    ):
        reasons.append("no live Loop-committed eager execution owner")
    # This checkpoint deliberately proves one visible CUDA device. Global Runtime owner
    # inventories below are conservative outside that cohort; no byte-count proxy is used.
    if commit.device != 0 or torch.cuda.device_count() != 1:
        reasons.append("eager retention currently qualifies one visible CUDA device")
    if not vt.split_context(commit.device)["eligible"]:
        reasons.append("native capture/replay/leaf ownership is active")
    with memory.lock:
        if any(other is not request and not other.idle for other in memory.requests):
            reasons.append("another serving request")
        regions = tuple(memory.regions)
        models = tuple(memory.models.values())
        bindings = tuple(binding for model in models for binding in model.bindings)
        if any(model.retiring or (model.users and model.thread != threading.get_ident())
               for model in models):
            reasons.append("model ownership is changing or concurrent")
    # Individual owner locks are never acquired while holding Memory or claim locks.
    for owner in regions:
        if not owner.lock.acquire(blocking=False):
            reasons.append("Runtime cache owner is changing")
            continue
        try:
            if not owner.closed or owner.pins or owner.reclaiming:
                reasons.append("Runtime disposable-cache owner")
        finally:
            owner.lock.release()
    for binding in bindings:
        if not binding.lock.acquire(blocking=False):
            reasons.append("prepared weight binding is changing")
            continue
        try:
            if binding.completion_lost or (not binding.closed and (
                binding.used or binding.held or binding._load_slot is not None
                or binding._stage_source is not None or binding._load_data is not None
            )):
                reasons.append("prepared weight binding")
        finally:
            binding.lock.release()
    runtime: Any = _loop._RUNTIME
    if runtime is not None and hasattr(runtime, "_lock"):
        with runtime._lock:
            if runtime._kept or any(site.pending for site in runtime._sites.values()):
                reasons.append("kept or queued loop program")
    for recorder in tuple(graphs._RECORDERS.values()):
        if not recorder.lock.acquire(blocking=False):
            reasons.append("graph recorder ownership is changing")
            continue
        try:
            # Legacy queued signatures contain only layouts, not captured tensors. Their
            # actual idle preparation is guarded; an empty recorder is not a live cache.
            if recorder.recordings or recorder.prepared or recorder.pool is not None:
                reasons.append("graph recorder owner")
        finally:
            recorder.lock.release()
    if any(not definition._closed for definition in tuple(prepared._OWNERS)):
        reasons.append("prepared definition or sequence")
    if any(
        not owner.closed and (
            owner.stream is not None or owner.control is not None or owner.owner is not None
            or owner.done is not None or owner.graphs
        )
        for owner in tuple(domain._LIVE)
    ):
        reasons.append("graph domain or hidden library owner")
    lm: Any = sys.modules.get("cozy_runtime.lm")
    if lm is not None:
        with lm._ENGINES_LOCK:
            engines = tuple(lm._ENGINES.values())
        for engine in engines:
            with engine.cond:
                if (
                    engine.cache is not None or engine.graphs or engine.pool is not None
                    or engine.addresses or engine.completions or engine.unfenced
                    or engine.driver is not None or engine.pending or engine.incoming
                    or (engine.region is not None and not engine.region.closed)
                ):
                    reasons.append("language-model execution/cache owner")
    if any(lm is None or snapshot is not lm._graph_snapshot for snapshot, _ in graphs.DECODERS):
        reasons.append("unrepresented decoder owner provider")
    return tuple(dict.fromkeys(reasons))


@contextmanager
def eager(commit: EagerCommit) -> Iterator[EagerLease]:
    """Claim a closed owner registry until the actual eager completion/abort boundary.

    This checkpoint is policy-neutral: it neither clears limits nor changes retention.
    The execution layer must issue the commit only after its preparation decision.
    """
    device = commit.device
    with _LOCK:
        if device in _EAGER or _CREATING.get(device):
            raise OwnerConflict("device execution/preparation ownership is busy")
        generation = _GENERATION.get(device, 0)
    try:
        reasons = _blockers(commit)
    except RuntimeError as error:
        # Empty metadata containers may register concurrently without owning GPU state.
        # A changing weak registry is a refusal to claim closure, never evidence of it.
        if "changed" in str(error) and "iteration" in str(error):
            raise OwnerConflict("owner registry changed during inventory") from error
        raise
    with _LOCK:
        if generation != _GENERATION.get(device, 0) or _CREATING.get(device) or device in _EAGER:
            raise OwnerConflict("owner registry changed while admitting eager retention")
        if reasons:
            raise OwnerConflict("eager retention unavailable: " + "; ".join(reasons))
        lease = EagerLease(commit)
        _EAGER[device] = lease
        _changed(device)
    try:
        yield lease
    finally:
        if not commit.observation.completion_proven:
            # Normal eager completion already synchronizes its owning stream. An abort
            # must join queued work before another execution owner may start preparing.
            import torch

            try:
                torch.cuda.synchronize(device)
            except BaseException:
                # An unproven device completion remains quarantined until executor exit.
                lease.completion_uncertain = True
                raise
        with _LOCK:
            if _EAGER.get(device) is not lease:
                raise RuntimeError("eager retention lost its execution owner")
            del _EAGER[device]
            _changed(device)
