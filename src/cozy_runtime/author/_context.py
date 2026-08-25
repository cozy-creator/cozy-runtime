"""`Context`, `RequestView`, and the minimal `Model` base.

`Context` answers exactly one question — where and under what conditions am I executing.
The member list below is the COMPLETE committed inventory (§1.3): anything absent is
absent on purpose. No checkpoint paths, no placement or offload knobs, no config or
environment, no provider facts, no child-call surface — and no `ctx.generator`: the
seeded per-request generator is `view.generator`, the ONE spelling.

cr-002b owns the `Model` boundary (Loader, `construct`/`load`, `uses_components`,
`make_scheduler`). What lives here is only what the HANDLER surface needs: a model
parameter to derive `gpu` from, its construction-identity fact, and the request view that
carries the seeded generator and the applied adapter stack.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from cozy_runtime.author._errors import Cancelled, DeadlineExceeded

PipelineT = TypeVar("PipelineT")

DeviceType = Literal["cuda", "cpu", "mps"]


@dataclass(frozen=True, slots=True)
class Device:
    """A read-only device identity. `str(device)` is the torch spelling (`cuda:0`)."""

    type: DeviceType = "cpu"
    index: int | None = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


@dataclass(frozen=True, slots=True)
class AdapterRef:
    """One entry of the applied stack. Gates read `kind`, NEVER `ref` (§1.4 hard rule)."""

    ref: str
    scale: float
    kind: str


@dataclass(frozen=True, slots=True)
class RequestView:
    """Request-local model state. SDK-owned members are exactly `generator` and `adapters`.

    No pipeline passthrough and no dynamic attribute proxying exist: model code calls its
    own typed pipeline and hands it `view.generator` explicitly (§1.3/§1.7).
    cr-002b adds the third member, `make_scheduler(name)`.
    """

    generator: random.Random
    seed: int
    adapters: tuple[AdapterRef, ...] = ()


def derive_seed(request_id: str) -> int:
    """A stable seed for an unseeded request: deterministic RNG without an authored number."""
    return int.from_bytes(hashlib.blake2b(request_id.encode(), digest_size=8).digest(), "big")


@dataclass(slots=True)
class Context:
    """The request-scoped execution surface. Injected when the signature names it."""

    request_id: str
    deadline: float
    """Monotonic-clock instant the attempt must not outlive."""
    boot_warmup: bool = False
    """True only during the boot warm pass — cheapen the body off it (cr-007)."""
    device: Device = field(default_factory=Device)
    _cancel: Callable[[], bool] = field(default=lambda: False, repr=False)
    _adapters: tuple[AdapterRef, ...] = field(default=(), repr=False)

    @property
    def cancelled(self) -> bool:
        """THE cancellation question. A passed deadline IS cancellation, not a second flag."""
        return self._cancel() or time.monotonic() >= self.deadline

    def raise_if_cancelled(self) -> None:
        """THE cancellation spelling."""
        if time.monotonic() >= self.deadline:
            raise DeadlineExceeded(f"attempt {self.request_id} outlived its deadline")
        if self._cancel():
            raise Cancelled(f"attempt {self.request_id} was cancelled")


class Model(Generic[PipelineT]):
    """Base of an endpoint's reusable model class — a bare typed parameter is the whole
    declaration, and its presence in a signature is what derives `gpu` (§1.2).

    cr-002b brings `load(loader)`, `loader.construct`, `@uses_components`, and the
    construction transaction. `checkpoint_ref` is a pinned-ref REPRODUCTION FACT for
    result structs — an observation, never catalog authority.
    """

    checkpoint_ref: str = ""

    def for_request(self, ctx: Context, *, seed: int | None = None) -> RequestView:
        """The request-local view: seeded generator plus the immutable applied stack.

        Request-local state lives where its lifetime is true — never as mutable fields on
        the long-lived model.
        """
        effective = derive_seed(ctx.request_id) if seed is None else seed
        return RequestView(random.Random(effective), effective, ctx._adapters)
