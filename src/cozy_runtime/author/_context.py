"""`Context` and `RequestView` — where a handler runs, and its request-local model state.

`Context` carries execution lifetime and capabilities for admitted model inputs and
declared outputs. TensorFS owns the returned source/output handles and native receipts;
Context binds them to this invocation. Store paths, placement/offload controls and
provider credentials remain outside this surface. The seeded request generator is
`view.generator`.

The `Model` boundary itself is `author/_model.py`; this module holds only what both it and
the handler surface need, so the construction plane can import the request plane and never
the reverse.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, final, overload

from cozy_runtime.author._errors import Cancelled, CapabilityError, DeadlineExceeded

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from cozy_runtime.author._artifacts import ModelArtifact
    from cozy_runtime.author._loader import Scheduler
    from cozy_runtime.author._prepared import Execution, PreparedWork
    from cozy_runtime.author._recipes import DeclaredRecipe
    from cozy_runtime.author._weights import Weights
    from cozy_runtime.weights_store import Checkpoint, Derivation, Output

DeviceType = Literal["cuda", "cpu", "mps"]


@dataclass(frozen=True, slots=True)
class Device:
    """A read-only device identity. `str(device)` is the torch spelling (`cuda:0`)."""

    type: DeviceType = "cpu"
    index: int | None = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


_PLACED: list[Device] = [Device()]


def device() -> Device:
    """This process's device, the one every model, call and placement here uses: its GPU (a
    group rank's own), MPS, or the CPU where none is present. A package never picks one; reading
    it is not a door. The CPU until an executor names it."""
    return _PLACED[0]


def place(placed: Device) -> None:
    """The executor names its device once its accelerator is up (`device()`)."""
    _PLACED[0] = placed


@dataclass(frozen=True, slots=True)
class AdapterRef:
    """One entry of the applied stack. Gates read `kind`, NEVER `ref` (§1.4 hard rule)."""

    ref: str
    scale: float
    kind: str
    component: str = ""
    source_component: str = ""
    family: str = ""


@final
@dataclass(frozen=True, slots=True)
class RequestView:
    """Request-local model state. Its SDK-owned members are EXACTLY three:
    `make_scheduler(name)`, `generator`, and `adapters`.

    There is no pipeline passthrough and no dynamic attribute proxying — `view(...)` and
    `view.<anything else>` do not exist, which is what removes the author surface's last
    declared `Any` seam (§1.3/§1.7). Model code calls its own typed pipeline and hands it
    `view.generator` and fresh schedulers explicitly. It cannot be subclassed: a wider view
    would be that deleted proxy under another name.

    `generator` is a `random.Random` seeded from the request. Noise for a model is drawn with
    `cozy_runtime.rng`, on the CPU, so a seed makes the same output on any device.
    """

    generator: random.Random
    adapters: tuple[AdapterRef, ...] = ()
    _seed: int = 0
    _prototypes: Mapping[str, Scheduler] = field(default_factory=dict, repr=False)
    _sampler: str | None = field(default=None, repr=False)

    def __init_subclass__(cls) -> None:
        raise CapabilityError(
            f"class {cls.__name__}(RequestView): the request view is exactly three SDK-owned "
            "members — a subclass is the deleted pipeline proxy under another name (§1.3)",
            code="view_subclass",
        )

    @classmethod
    def _make(
        cls,
        seed: int,
        adapters: tuple[AdapterRef, ...],
        prototypes: Mapping[str, Scheduler],
        sampler: str | None,
    ) -> RequestView:
        return cls(random.Random(seed), adapters, seed, prototypes, sampler)

    def make_scheduler(self, name: str) -> Scheduler:
        """A FRESH request-lifetime instance of the named family — never shared, and it
        dies with the attempt. The SDK owns the sampler table; the request owns nothing."""
        prototype = self._prototypes.get(name)
        if prototype is None:
            raise CapabilityError(
                f"no scheduler named {name!r} in this construction "
                f"({', '.join(sorted(self._prototypes)) or 'none'}) — schedulers are "
                "construction facts, not request data",
                code="unknown_scheduler",
                fields=[name],
            )
        return prototype.clone(self._sampler)


def derive_seed(request_id: str) -> int:
    """A stable seed for an unseeded request: deterministic RNG without an authored number."""
    return int.from_bytes(hashlib.blake2b(request_id.encode(), digest_size=8).digest(), "big")


@dataclass(slots=True)
class Context:
    """The execution surface: where and under what conditions this code runs.

    Injected into a handler when its signature names it.
    """

    request_id: str
    deadline: float
    """Monotonic-clock instant the attempt must not outlive."""
    device: Device = field(default_factory=Device)
    _cancel: Callable[[], bool] = field(default=lambda: False, repr=False)
    _tensorfs_source: Callable[[Weights[Any]], Checkpoint] | None = field(default=None, repr=False)
    _tensorfs_output: Callable[[str, Derivation], Output] | None = field(default=None, repr=False)
    _tensorfs_adopt: Callable[[Output], ModelArtifact] | None = field(default=None, repr=False)
    _release_gpus: Callable[[], None] | None = field(default=None, repr=False)
    _plan: Callable[[Sequence[str | Collection[str]]], None] | None = field(
        default=None, repr=False
    )
    _recipes: tuple[DeclaredRecipe, ...] = field(default=(), repr=False)
    _submit_prepared: Callable[[PreparedWork], Execution] | None = field(default=None, repr=False)
    _pause_stall_watch: Callable[[], AbstractContextManager[None]] | None = field(
        default=None, repr=False
    )

    def tensorfs_source(self, weights: Weights[Any]) -> Checkpoint:
        """An admitted `Weights` argument's checkpoint: its pipeline, components and stored
        bytes."""
        self.raise_if_cancelled()
        if self._tensorfs_source is None:
            raise CapabilityError(
                "this attempt has no TensorFS source binding", code="weights_source_ungranted"
            )
        return self._tensorfs_source(weights)

    def output(self, slot: str, derivation: Derivation) -> Output:
        """Open one interface-declared weights output: a new checkpoint written into the store
        from `derivation`'s base and sources, resumable, within the output's bound."""
        self.raise_if_cancelled()
        if self._tensorfs_output is None:
            raise CapabilityError(
                "this attempt has no weights output binding", code="weights_output_ungranted"
            )
        return self._tensorfs_output(slot, derivation)

    def adopt_model(self, output: Output) -> ModelArtifact:
        """Commit `output` and keep it as this attempt's model. A committed output replays;
        TensorD uploads it when the run names a destination."""
        self.raise_if_cancelled()
        if self._tensorfs_adopt is None:
            raise CapabilityError(
                "this attempt has no weights adopt binding", code="weights_receipt_mismatch"
            )
        return self._tensorfs_adopt(output)

    def release_gpus(self) -> None:
        """Say this request is done with GPUs for a long while, so the next request's GPU
        work starts now while this one finishes its CPU and network work.

        A request holds its GPUs from its first GPU call until it ends, across the gaps
        between its calls. Call this before a long CPU-only stretch, usually the end
        (assembly, encoding, uploads), after awaiting the last GPU call. Its GPUs go to the
        oldest waiting request; the request keeps running. A GPU call still in flight keeps
        its GPUs until it returns. A GPU call awaited later waits in line at this request's
        original priority (its acceptance) and never takes GPUs back from whoever got them
        meanwhile. Idempotent. Where there is no allocation to end (a request with no GPU
        calls, a child call, or a GPU handler, whose GPU ends when it returns) it does nothing.
        """
        if self._release_gpus is not None:
            self._release_gpus()

    @property
    def cancelled(self) -> bool:
        """THE cancellation question. A passed deadline IS cancellation, not a second flag."""
        return self._cancel() or time.monotonic() >= self.deadline

    @overload
    def plan(self, prepared: PreparedWork, /) -> Execution: ...

    @overload
    def plan(self, *steps: str | Collection[str]) -> None: ...

    def plan(self, *steps: str | Collection[str] | PreparedWork) -> Execution | None:
        """The models this call runs, in order, so GPU memory evicts and restores by them: each
        step a component name, or a set of names used together (`{"unet", "controlnet"}`).
        Declare it before the first model runs; calling again replaces it. No sizes or times.

        A single PreparedWork instead submits already prepared resident GPU work and returns
        its Execution observer. Further submissions are ordered after earlier completion;
        a prior failure prevents later work. PreparedSequence submits a whole ordered native
        plan at once, without Python calls between its stages. Ordinary component hints are
        not reinterpreted as executable code.
        """
        from cozy_runtime.author._prepared import PreparedWork

        if len(steps) == 1 and isinstance(steps[0], PreparedWork):
            if self._submit_prepared is not None:
                return self._submit_prepared(steps[0])
            return steps[0].submit(self)
        if any(isinstance(step, PreparedWork) for step in steps):
            raise TypeError("prepared work must be the only ctx.plan argument")
        if self._plan is not None:
            self._plan(steps)  # type: ignore[arg-type]
        return None

    def raise_if_cancelled(self) -> None:
        """THE cancellation spelling."""
        if time.monotonic() >= self.deadline:
            raise DeadlineExceeded(f"attempt {self.request_id} outlived its deadline")
        if self._cancel():
            raise Cancelled(f"attempt {self.request_id} was cancelled")


#: The running attempt's `Context`, for Runtime code a handler calls without one (`lm`): its
#: long loops answer cancellation as the handler would.
CURRENT: ContextVar[Context | None] = ContextVar("cozy_attempt_context", default=None)
