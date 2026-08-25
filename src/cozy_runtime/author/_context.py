"""`Context` and `RequestView` — where a handler runs, and its request-local model state.

`Context` answers exactly one question — where and under what conditions am I executing.
The member list below is the COMPLETE committed inventory (§1.3): anything absent is
absent on purpose. No checkpoint paths, no placement or offload knobs, no config or
environment, no provider facts, no child-call surface — and no `ctx.generator`: the
seeded per-request generator is `view.generator`, the ONE spelling.

The `Model` boundary itself is `author/_model.py`; this module holds only what both it and
the handler surface need, so the construction plane can import the request plane and never
the reverse.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, final

from cozy_runtime.author._errors import Cancelled, CapabilityError, DeadlineExceeded

if TYPE_CHECKING:
    from cozy_runtime.author._loader import Scheduler

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

    `generator` is a `random.Random` while the runtime is weightless; it becomes a
    `torch.Generator` with the first real fills (cr-005). Author code passes it through and
    never inspects it, so the swap is source-compatible.
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
