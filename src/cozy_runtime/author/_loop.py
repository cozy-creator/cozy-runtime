"""`denoise_loop`: a package declares its loop; the Runtime decides how it runs.

The Runtime runs it eagerly or as one resident captured program, by the device's memory and
capabilities. It owns capture, scratch pools, fallback, cancellation and the program's
lifetime; the package writes one step function that every mode runs.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, Protocol


class Loops(Protocol):
    """The Runtime's side of `denoise_loop`, installed by an executor."""

    def run(
        self,
        step: Callable[..., object],
        steps: range,
        state: tuple[Any, ...],
        carried: tuple[Any, ...],
        inputs: tuple[Any, ...],
        phases: tuple[Hashable, ...] | None,
        key: Hashable | None,
        on_step: Callable[[int], object] | None,
        before: Callable[..., object] | None,
        after: Callable[..., object] | None,
    ) -> None: ...


_RUNTIME: Loops | None = None


def install(runtime: Loops | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def denoise_loop(
    step: Callable[..., object],
    steps: range,
    *,
    state: Sequence[Any],
    carried: Sequence[Any] = (),
    inputs: Sequence[Any] = (),
    phases: Sequence[Hashable] | None = None,
    key: Hashable | None = None,
    on_step: Callable[[int], object] | None = None,
    before: Callable[..., object] | None = None,
    after: Callable[..., object] | None = None,
) -> None:
    """Run `step` once per index in `steps`; `state` and `carried` advance in place.

    The Runtime calls `step(i, *state, *carried, *inputs)`, plus `phase=phases[k]` when phases
    are declared. `i` is a one-element int64 tensor on the device holding the global step index.
    The step reads its schedule with `table.index_select(0, i)` and updates `state` and `carried`
    in place. It runs tensor operations only: no host read of a device value, no random draw
    (noise is an input) and no host decision except one taken from its `phase`. Every mode runs
    this same function, so eager and native execution compute the same values.

    - `state`: what the loop advances (latents). On return the tensors hold the final values.
    - `carried`: tensors one step leaves for the next that are not outputs.
    - `inputs`: read-only tensors of this call.
    - `phases`: one hashable host-side plan per step; each run of equal phases is one captured
      segment, and the step receives its phase.
    - `key`: with `inputs` and an `after`, lets the Runtime keep the whole call for a later
      call with an equal key and equal tensor layouts. It names everything the step bakes in
      that is not a declared tensor (a guidance float, say). The step, `before` and `after` must
      then reach every per-call tensor through their arguments; the Runtime keeps its own
      copies and copies values in and out.
    - `on_step(index)`: called on the host after each completed step.
    - `before` and `after`: work that runs once before the first step and after the last,
      called like the step without its index (`before(*state, *carried, *inputs)`): a text
      encoder writing carried conditioning, a decoder writing an image held in `state`. The
      Runtime records them into the program when their components stay on the device too.

    A step that runs components is called inside a `@uses_components` method, whose components
    they are. A step of tensor work alone may run outside one; it records resident too.
    """
    if not isinstance(steps, range) or not len(steps) or steps.step != 1:
        raise ValueError("denoise_loop steps are a nonempty range with step 1")
    state, carried, inputs = tuple(state), tuple(carried), tuple(inputs)
    if not state:
        raise ValueError("denoise_loop needs state to advance")
    if phases is not None:
        phases = tuple(phases)
        if len(phases) != len(steps):
            raise ValueError("denoise_loop declares one phase per step")
    if key is not None:
        hash(key)
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("denoise_loop runs inside a Runtime executor")
    runtime.run(step, steps, state, carried, inputs, phases, key, on_step, before, after)
