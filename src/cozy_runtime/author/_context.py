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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, final

from cozy_runtime.author._errors import Cancelled, CapabilityError, DeadlineExceeded

if TYPE_CHECKING:
    from tensorfs.derived import OutputCapability, SourceCapability

    from cozy_runtime.author._artifacts import ModelArtifact
    from cozy_runtime.author._loader import Scheduler
    from cozy_runtime.author._model import Model

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
    """The execution surface: where and under what conditions this code runs.

    Injected into a handler when its signature names it, and handed to `Model.warm` by
    the runtime. Under `warm` there is no attempt: `request_id` is empty, the adapter
    stack is empty, and no package-call broker is bound (model-lifecycle.md, cr-110).
    """

    request_id: str
    deadline: float
    """Monotonic-clock instant the attempt must not outlive."""
    device: Device = field(default_factory=Device)
    _cancel: Callable[[], bool] = field(default=lambda: False, repr=False)
    _adapters: tuple[AdapterRef, ...] = field(default=(), repr=False)
    _tensorfs_source: Callable[[Model[object]], SourceCapability] | None = field(
        default=None, repr=False
    )
    _tensorfs_output: Callable[[str], OutputCapability] | None = field(default=None, repr=False)
    _tensorfs_adopt: Callable[[Mapping[str, Any]], ModelArtifact] | None = field(
        default=None, repr=False
    )
    _release_gpus: Callable[[], None] | None = field(default=None, repr=False)

    def tensorfs_source(self, model: Model[object]) -> SourceCapability:
        """Bind an admitted model input to TensorFS's source capability."""
        self.raise_if_cancelled()
        if self._tensorfs_source is None:
            raise CapabilityError(
                "this attempt has no TensorFS source binding", code="weights_source_ungranted"
            )
        return self._tensorfs_source(model)

    def output(self, slot: str) -> OutputCapability:
        """Return one interface-declared TensorFS output, without enlarging its bound."""
        self.raise_if_cancelled()
        if self._tensorfs_output is None:
            raise CapabilityError(
                "this attempt has no TensorFS output binding", code="weights_output_ungranted"
            )
        return self._tensorfs_output(slot)

    def adopt_model(self, receipt: Mapping[str, Any]) -> ModelArtifact:
        """Adopt the exact native result of this attempt's declared output."""
        self.raise_if_cancelled()
        if self._tensorfs_adopt is None:
            raise CapabilityError(
                "this attempt has no TensorFS receipt binding", code="weights_receipt_mismatch"
            )
        return self._tensorfs_adopt(receipt)

    def release_gpus(self) -> None:
        """Say this request is done with GPUs, so the next request's GPU work starts now
        while this one finishes its CPU and network work (assembly, encoding, uploads).

        Call it after awaiting the last GPU call. The request's GPU lease ends at once;
        the request keeps running. A GPU call still in flight keeps its GPUs until it
        returns. A GPU call awaited later is a new demand that waits in line at this
        request's original priority and never takes GPUs back from whoever got them
        meanwhile. Idempotent. Where there is no lease to end (a request with no GPU
        calls, or a GPU handler, whose GPUs end when it returns) it does nothing.
        """
        if self._release_gpus is not None:
            self._release_gpus()

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
