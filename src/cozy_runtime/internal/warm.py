"""`Model.warm` — the author's one post-fill, pre-serving moment (cr-110, model-lifecycle.md).

The executor calls `warm_generation` once per construction fill: after the fill and its
verification, after every runtime application on the constructed generation (the execution
plan, the attention contract) and the residency install, and before it reports Ready — so
the placement is never DISPATCHABLE with a cold generation behind it, on a dynamic placement
as on any other. It never runs on a component stage or evict: residency moves bytes and the
module objects survive, so what `warm` did to them survives too.

Before calling the author, the executor opens the existing component-staged residency
contract. Each declared use evicts unrelated completed weights, giving dry forwards room
for their work; no request's all-resident promise exists yet. Followers receive the same
contract through the ordinary mirror. The actual post-warm resident set is reported to
the request planner. This can stage additional components and therefore adds warm I/O.

Inside `warm` the name-renaming `torch.compile(module)` spelling is refused typed, exactly
as under derive; the in-place `nn.Module.compile` is allowed (#702). The Context carries the
device, the fill's deadline and cancellation, and can reserve no package call: the broker
that only `invoke` binds is absent, which is a property of the Context and not a flag.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from cozy_runtime.author._context import Device
from cozy_runtime.author._model import Model, warm_context
from cozy_runtime.internal.derive import refuse_compile


class WarmFailed(Exception):
    """The author's `warm` raised — or reached a refused spelling. Names the exception."""

    code = "warm_failed"

    def __init__(self, exc: BaseException) -> None:
        super().__init__(f"{type(exc).__name__}: {exc}")


def warm_generation(model: Model[Any], *, device: Device, cancel: Callable[[], bool]) -> float:
    """Run `model.warm` once on a filled generation; return its wall time in milliseconds.

    The deadline is the fill's, and the fill has none: a prepare is silence-bounded by the
    worker, never clocked (§3.3), so the honest bound is no bound. `cancel` is the one
    cancellation fact this process holds about a construction — whether it is already
    poisoned — and it is what `ctx.raise_if_cancelled()` reads.
    """
    ctx = warm_context(device, deadline=math.inf, cancel=cancel)
    started = time.perf_counter()
    try:
        with refuse_compile(lazy=False):
            model.warm(ctx)
    except Exception as exc:
        raise WarmFailed(exc) from exc
    return (time.perf_counter() - started) * 1000
