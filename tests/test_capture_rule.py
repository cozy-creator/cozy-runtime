"""A region records its graph only where a replay is known to follow (lead's rule after v7:
CLIP recorded 97 graphs on its last call of a cold request, each replayed once)."""

from __future__ import annotations

from cozy_runtime.graphs import records
from cozy_runtime.internal.weights import Steps


def _calls(steps: Steps, attempt: int, component: str, seen: dict[str, int]) -> bool:
    """One call of `component`'s region in `attempt`: whether it records."""
    steps.called(component)
    now = records(seen.get(component), attempt, steps.later(component), placed=True)
    seen.setdefault(component, attempt)
    return now


def test_the_unet_records_cold_clip_waits_for_the_warm_request() -> None:
    seen: dict[str, int] = {}
    cold = Steps()
    cold.plan([{"text_encoder", "text_encoder_2"}, "unet", "unet", "unet", "vae"])
    # CLIP runs twice in its one step (the prompt and the negative): no step follows either call
    assert [_calls(cold, 1, "text_encoder", seen) for _ in range(2)] == [False, False]
    # the UNet's first call runs as written; its second records (steps follow it), and its last,
    # with no step after it, would not record (it replays the second's recording)
    assert [_calls(cold, 1, "unet", seen) for _ in range(3)] == [False, True, False]
    warm = Steps()
    warm.plan([{"text_encoder", "text_encoder_2"}, "unet", "unet", "unet", "vae"])
    # seen in the cold request: CLIP records at its first warm call (its second replays)
    assert _calls(warm, 2, "text_encoder", seen) is True


def test_unplaced_regions_record_from_their_second_call() -> None:
    assert records(None, 1, False, placed=False) is False
    assert records(1, 2, False, placed=False) is True
