"""`denoise_loop`: a package declares its loop; the Runtime decides how it runs.

The Runtime runs it eagerly or as one resident captured program, by the device's memory and
capabilities. It owns capture, scratch pools, fallback, cancellation and the program's
lifetime; the package writes one step function that every mode runs.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, Protocol

from cozy_runtime.author import _attention


class Loops(Protocol):
    """The Runtime's side of `denoise_loop`, installed by an executor."""

    def run(
        self,
        step: Callable[..., object],
        steps: range,
        state: tuple[Any, ...],
        carried: tuple[Any, ...],
        inputs: tuple[Any, ...],
        labels: tuple[Hashable, ...] | None,
        key: Hashable | None,
        on_step: Callable[[int], object] | None,
        before: Callable[..., object] | None,
        after: Callable[..., object] | None,
        cache: Any = None,
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
    labels: Sequence[Hashable] | None = None,
    key: Hashable | None = None,
    on_step: Callable[[int], object] | None = None,
    before: Callable[..., object] | None = None,
    after: Callable[..., object] | None = None,
    cache: Any = None,
) -> None:
    """Run `step` once per index in `steps`; `state` and `carried` advance in place.

    The Runtime calls `step(i, *state, *carried, *inputs)`, plus `label=labels[k]` when labels
    are declared. `i` is a one-element int64 tensor on the device holding the global step index.
    The step reads its schedule with `table.index_select(0, i)` and updates `state` and `carried`
    in place. It runs tensor operations only: no host read of a device value, no random draw
    (noise is an input, or `rng.counter_noise` from the step's `i`) and no host decision except
    one taken from its `label`. Every mode runs this same function, so eager and native
    execution compute the same values.

    - `state`: what the loop advances (latents). On return the tensors hold the final values.
    - `carried`: tensors one step leaves for the next that are not outputs.
    - `inputs`: read-only tensors of this call, and `retained` States (a prefix's K/V): the
      step receives the tensors a State's `get()` returns in its place. Each step run eagerly
      reads it again, so its memory may leave the card between steps; a recorded program reads
      it once, beside the program's pool, when there is room for both, and otherwise the call
      runs eagerly and the run says why. Keyed or not, the same. Never call `get()` in the step.
    - `labels`: one hashable host-side label per step, which the step receives: what differs
      between steps (dense or sparse attention, an expert, a batch). Each distinct label records
      one body, run for each of its steps in order; a first call runs its steps eagerly through
      each label's first.
    - `key`: with `inputs`, lets the Runtime keep the whole call for a later call with an
      equal key and equal tensor layouts. It names everything the step bakes in that is not
      a declared tensor (a guidance float, say), an overlay its step arms on only some calls
      among them: overlays armed before the loop are its own. `before` and `after` remain
      optional. The step and any declared `before`/`after` must reach every per-call tensor
      through their arguments; the Runtime keeps its own copies and copies values in and out.
    - `on_step(index)`: called on the host after each completed step.
    - `before` and `after`: work that runs once before the first step and after the last,
      called like the step without its index (`before(*state, *carried, *inputs)`): a text
      encoder writing carried conditioning, a decoder writing an image held in `state`. The
      Runtime records them into the program when their components stay on the device too.
    - `cache`: a `stepcache` policy for the transformers whose stacks the package installed
      (`stepcache.install`): approximate, for a preset that asks for it, never a default.

    LoRA overlays (`lora.Hook`) are armed before the loop; one a step arms raises. As it
    returns, each attention producer's request state ends its loop (`end_loop()`).

    A step that runs components is called inside a `@uses_components` method, whose components
    they are. A step of tensor work alone may run outside one; it records resident too.
    """
    if not isinstance(steps, range) or not len(steps) or steps.step != 1:
        raise ValueError("denoise_loop steps are a nonempty range with step 1")
    state, carried, inputs = tuple(state), tuple(carried), tuple(inputs)
    if not state:
        raise ValueError("denoise_loop needs state to advance")
    if labels is not None:
        labels = tuple(labels)
        if len(labels) != len(steps):
            raise ValueError("denoise_loop declares one label per step")
    if key is not None:
        hash(key)
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("denoise_loop runs inside a Runtime executor")
    try:
        runtime.run(step, steps, state, carried, inputs, labels, key, on_step, before, after, cache)
    finally:
        _attention.end_loop()
