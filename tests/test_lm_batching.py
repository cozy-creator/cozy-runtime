"""Concurrent `lm.generate` calls on one model decode in one loop (`lm._Engine`): calls join at
the next step and leave when done, with prompts of different lengths, limits, penalties and
images, drafted and not, and each row keeps the greedy tokens Transformers gives it alone."""

from __future__ import annotations

import threading
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cozy_runtime import lm  # noqa: E402
from test_lm_mtp import DEVICE, STOP, _head, _qwen3, _qwen3_5, _reference  # noqa: E402


def _together(model: Any, calls: list[dict[str, Any]], draft: Any = None) -> list[Any]:
    """Every call on its own thread, released at once; each returns its tokens."""
    results: list[Any] = [None] * len(calls)
    errors: list[BaseException] = []
    start = threading.Barrier(len(calls))

    def run(i: int, call: dict[str, Any]) -> None:
        start.wait()
        try:
            ids = call["ids"]
            inputs = call.get("inputs") or {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }
            results[i] = lm.generate(
                model,
                inputs,
                max_new_tokens=call["new"],
                stop=STOP,
                pad=0,
                presence_penalty=call.get("penalty", 0.0),
                draft=draft,
            )
        except BaseException as exc:  # surfaced by the test
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def _rows_a_step(err: str) -> list[float]:
    return [
        float(line.split(" rows a step")[0].rsplit(", ", 1)[1])
        for line in err.splitlines()
        if " rows a step" in line
    ]


@pytest.mark.parametrize("make", [_qwen3, _qwen3_5], ids=["qwen3", "qwen3_5"])
def test_concurrent_calls_keep_their_own_tokens(
    make: Any, capfd: pytest.CaptureFixture[str]
) -> None:
    torch.manual_seed(7)
    model = make()
    shapes = [(30, 40, 0.0), (250, 12, 1.5), (260, 60, 0.0), (5, 33, 0.0), (120, 7, 1.5)]
    calls = [
        {"ids": torch.randint(10, 44, (1, prompt), device=DEVICE), "new": new, "penalty": penalty}
        for prompt, new, penalty in shapes
    ]
    wants = [_reference(model, c["ids"], c["new"], c["penalty"]) for c in calls]
    capfd.readouterr()
    for got, want in zip(_together(model, calls), wants, strict=True):
        assert torch.equal(got, want)
    assert max(_rows_a_step(capfd.readouterr().err)) > 1  # they shared steps


def test_drafted_concurrent_calls_keep_their_own_tokens(capfd: pytest.CaptureFixture[str]) -> None:
    """Rows drafted with one head keep each its own accepted prefix, a verify pass for all."""
    torch.manual_seed(9)
    model = _qwen3_5()
    head = _head(model)
    calls = [
        {"ids": torch.randint(10, 44, (1, prompt), device=DEVICE), "new": new}
        for prompt, new in ((40, 30), (250, 50), (8, 20), (90, 41))
    ]
    wants = [_reference(model, c["ids"], c["new"], 0.0) for c in calls]
    capfd.readouterr()
    for got, want in zip(_together(model, calls, draft=head), wants, strict=True):
        assert torch.equal(got, want)
    err = capfd.readouterr().err
    assert "drafting" in err and max(_rows_a_step(err)) > 1


def test_an_image_call_and_text_calls_decode_together() -> None:
    torch.manual_seed(1)
    model = _qwen3_5()
    vision = model.config.vision_config
    patch = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    merged = 16 // vision.spatial_merge_size**2
    image = [46, *[45] * merged, 47]
    text = torch.randint(10, 44, (1, 30), device=DEVICE)
    ids = torch.cat([text[:, :4], torch.tensor([image], device=DEVICE), text[:, 4:]], 1)
    pictured = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.randn(16, patch).to(DEVICE),
        "image_grid_thw": torch.tensor([[1, 4, 4]], device=DEVICE),
        "mm_token_type_ids": (ids == 45).int(),
    }
    calls = [{"ids": ids, "inputs": pictured, "new": 30}]
    calls += [{"ids": torch.randint(10, 44, (1, n), device=DEVICE), "new": 25} for n in (17, 64)]
    with torch.inference_mode():
        out = model.generate(
            **pictured, max_new_tokens=30, do_sample=False, eos_token_id=STOP, pad_token_id=0
        )
    wants = [out[:, ids.shape[1] :]] + [_reference(model, c["ids"], 25, 0.0) for c in calls[1:]]
    for got, want in zip(_together(model, calls), wants, strict=True):
        assert torch.equal(got, want)


def test_more_calls_than_the_largest_bucket_wait_for_a_slot() -> None:
    torch.manual_seed(3)
    model = _qwen3()
    calls = [
        {"ids": torch.randint(10, 44, (1, 10 + i), device=DEVICE), "new": 6 + i % 5}
        for i in range(lm.BUCKETS[-1] + 3)
    ]
    wants = [_reference(model, c["ids"], c["new"], 0.0) for c in calls]
    for got, want in zip(_together(model, calls), wants, strict=True):
        assert torch.equal(got, want)
