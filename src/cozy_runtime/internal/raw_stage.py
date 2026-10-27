"""Internal own-address raw-stage mechanism; no package adapter opts in yet.

The callback receives ephemeral views of existing Region addresses during preparation.
Every view must disappear after capture. Native detachment refuses escaped aliases.
This is a single one-shot stage using the existing continuation, not a full streamed plan.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from tensorfs import varena

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.compute import PreparedRepeat, _native_error


def capture_raw_stage(
    checkpoint: Any,
    groups: Sequence[tuple[Any, Sequence[tuple[str, int, int]]]],
    operation: Callable[[Any, tuple[Any, ...]], None],
    *,
    state: Sequence[Any],
    ordering: str = "events",
) -> PreparedRepeat:
    """Capture once at existing raw Region VAs, then refill those VAs natively on submit.

    The preparation callback must be data-independent, must update only declared state,
    return None, and must not retain input views. No private banks, parameter-address
    substitution or physical aliases are created. Groups must belong exclusively to the
    immutable checkpoint recipe supplied by the caller. Actual model binding/generation
    validation remains the production adapter's responsibility; this private mechanism is
    not that adapter. The first slice requires at least one nonresident input at capture.
    """
    import torch

    tensors = tuple(state)
    if not tensors or any(not isinstance(t, torch.Tensor) or not t.is_cuda for t in tensors):
        raise ValueError("raw stage needs CUDA boundary tensors")
    device = tensors[0].device.index
    if device is None or any(t.device.index != device or t.is_inference() for t in tensors):
        raise ValueError("raw stage uses non-inference state on one device")
    if ordering not in {"events", "checked"}:
        raise ValueError("ordering must be events or checked")
    slot = varena.LoadSlot.from_regions(checkpoint, groups)
    if slot is None:
        raise CapabilityError(
            "the initial raw-stage mechanism requires a nonresident capture input",
            code="prepared_graph_ineligible",
        )
    if not callable(getattr(slot._native, "detach_source", None)):
        slot.close()
        raise CapabilityError("TensorFS lacks stage detachment", code="prepared_graph_ineligible")
    plan = None
    try:

        def record():
            inputs = slot.tensors
            original = tuple(t.detach() for t in tensors)
            layouts = tuple(PreparedRepeat._layout(t) for t in tensors)

            def body(index):
                try:
                    result = operation(index, inputs)
                    if result is not None:
                        raise CapabilityError(
                            "raw stages write declared state and return None",
                            code="prepared_graph_ineligible",
                        )
                finally:
                    changed = [
                        i
                        for i, tensor in enumerate(tensors)
                        if PreparedRepeat._layout(tensor) != layouts[i]
                    ]
                    for i in changed:
                        # Metadata-only recovery precedes _capture_ready's value restore;
                        # never copy saved values into an input aliased by invalid .data/set_.
                        tensors[i].data = original[i]
                    if changed:
                        raise CapabilityError(
                            "raw stage changed boundary tensor metadata",
                            code="prepared_graph_ineligible",
                        )

            return PreparedRepeat._capture_ready(
                body,
                state=tensors,
                steps=1,
                owners=(slot,),
                _release_loaded_inputs=True,
            )

        with torch.cuda.device(device), torch.inference_mode(False):
            plan = record()
            slot.seal()
            # record's callback frame is gone; remove this wrapper's only private views.
            # No gc sweep or reference-count guess proves alias safety: native physical
            # ownership checks the actual remaining DLPack/foreign Pins below.
            slot._tensor = None
            slot._tensors = None
            source = slot._native.detach_source()
            plan._load = source
            plan._load_ordering_requested = ordering
            plan._load_ordering = ordering
            return plan
    except BaseException as error:
        if plan is not None:
            plan.close()
        if isinstance(error, Exception):
            _native_error(error)
        raise
    finally:
        slot.close()
