"""Memoized Model methods (tracker #298): the real author wrapper, the executor's engine and
the Worker's machine tier, on CPU torch. Nothing here is package- or model-specific: the
models below are synthetic, so the mechanism is proven general."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from cozy_runtime.author import ConformanceError, Model, uses_components  # noqa: E402
from cozy_runtime.author._executor_requests import (  # noqa: E402
    MemoEntry,
    MemoStored,
    StageMemoLookup,
    StageMemoStore,
)
from cozy_runtime.author._stage_memo import inexact  # noqa: E402
from cozy_runtime.internal import stage_memo  # noqa: E402
from cozy_runtime.internal.executor_commands import MemoSettings  # noqa: E402


class Conditioning(msgspec.Struct, frozen=True):
    prompt: Any
    pooled: Any
    tokens: int


def _tokens(text: str) -> Any:
    return torch.tensor([ord(c) % 97 for c in text] or [0], dtype=torch.int64)


class Encoder(Model[object]):
    runs = 0

    @uses_components("text_encoder", memoize=True, memo_dependencies=(_tokens,))
    def encode(self, text: str, scale: float = 1.0) -> Conditioning:
        type(self).runs += 1
        ids = _tokens(text)
        weight = torch.linspace(-1, 1, 64, dtype=torch.float32).reshape(1, 64)
        prompt = (ids[:, None].float() * weight * scale).to(torch.float16)
        return Conditioning(prompt, prompt.float().mean(0).to(torch.bfloat16), int(ids.numel()))

    @uses_components("text_encoder", memoize=True)
    def noisy(self, text: str) -> Any:
        return torch.rand(4)

    @uses_components("text_encoder", memoize=True)
    def retried(self, text: str) -> Any:
        type(self).runs += 1
        inexact()
        return torch.ones(4)

    @uses_components("vae", memoize=True)
    def pair(self, n: int) -> tuple[Any, list[Any], dict[str, Any]]:
        type(self).runs += 1
        n = n if isinstance(n, int) else 7
        return torch.full((2, 3), n), [torch.zeros(1, dtype=torch.bool), n * 0.5], {"n": n}


def _model(weights: str = "w1", engine: stage_memo.Engine | None = None) -> Encoder:
    model = Encoder.for_test()
    memo = stage_memo.GenerationMemo(
        torch=torch,
        environment="sha256:env",
        components={"text_encoder": weights, "vae": "sha256:vae"},
        assets="sha256:assets",
        device="sha256:cpu",
        sites={},
        engine=engine or stage_memo.Engine(),
    )
    object.__setattr__(model, "_cozy_memo", memo)
    Encoder.runs = 0
    return model


def _calls(model: Encoder) -> list[stage_memo.Call]:
    memo = model._cozy_memo
    assert isinstance(memo, stage_memo.GenerationMemo)
    return memo.engine.calls


def test_a_hit_runs_nothing_and_returns_the_same_bytes_in_the_same_form() -> None:
    model = _model()
    first = model.encode("a red fox")
    second = model.encode(text="a red fox", scale=1.0)
    assert Encoder.runs == 1
    assert isinstance(second, Conditioning) and second.tokens == first.tokens
    assert torch.equal(first.prompt, second.prompt) and first.prompt is not second.prompt
    assert second.prompt.dtype == torch.float16 and second.pooled.dtype == torch.bfloat16
    # The hit opened no component scope: nothing was staged for it.
    model.harness.assert_scopes("encode")
    outcomes = [(c.outcome, c.sha256) for c in _calls(model)]
    assert outcomes[0][0] == "miss" and outcomes[1] == ("hit:memory", outcomes[0][1])
    model.encode("a red fox", 2.0)
    model.encode("a blue fox")
    assert Encoder.runs == 3


def test_neither_a_miss_nor_a_hit_holds_its_result_past_the_call() -> None:
    """The attempt ledger reads device bytes at the terminal. A result the memo still held,
    even as uncollected garbage, reads there as a leak and costs the executor."""
    model = _model()
    gc.disable()
    try:
        for _ in ("miss", "hit"):
            result = model.encode("a prompt nobody keeps")
            held = weakref.ref(result.prompt)
            del result
            assert held() is None
    finally:
        gc.enable()


def test_any_result_structure_round_trips() -> None:
    model = _model()
    first = model.pair(3)
    second = model.pair(3)
    assert Encoder.runs == 1
    assert type(second[1]) is list and second[1][1] == 1.5 and second[2] == {"n": 3}
    assert torch.equal(first[0], second[0]) and second[1][0].dtype == torch.bool


def test_weights_and_numerics_scope_the_key() -> None:
    engine = stage_memo.Engine()
    same = _model("w1", engine)
    same.encode("shared")
    other = _model("w1", engine)
    other.encode("shared")
    assert Encoder.runs == 0, "a generation sharing bit-identical encoder weights hits"
    different = _model("w2", engine)
    different.encode("shared")
    assert Encoder.runs == 1, "different encoder weights miss"
    torch.backends.cuda.matmul.allow_tf32 = not torch.backends.cuda.matmul.allow_tf32
    try:
        different.encode("shared")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = not torch.backends.cuda.matmul.allow_tf32
    assert Encoder.runs == 2, "a math flag read at call time changes the key"


def test_drawing_from_the_default_generator_refuses_memo_rng() -> None:
    model = _model()
    with pytest.raises(ConformanceError) as refused:
        model.noisy("x")
    assert refused.value.code == "memo_rng"


def test_changed_arithmetic_is_returned_but_never_stored() -> None:
    model = _model()
    model.retried("x")
    model.retried("x")
    assert Encoder.runs == 2
    assert [c.reason for c in _calls(model)] == ["inexact", "inexact"]


def test_unkeyable_arguments_run_unmarked() -> None:
    model = _model()
    model.pair(lambda: 1)  # type: ignore[arg-type]
    model.pair(lambda: 1)  # type: ignore[arg-type]
    assert Encoder.runs == 2
    assert [c.reason for c in _calls(model)] == ["argument of type function"] * 2


def test_the_process_tier_keeps_its_cap_least_recent_first() -> None:
    engine = stage_memo.Engine()
    model = _model(engine=engine)
    model.encode("one")
    engine.configure(MemoSettings(process_bytes=2 * engine.held + 1))
    model.encode("two")
    model.encode("thr")
    assert list(engine.entries) and len(engine.entries) == 2
    model.encode("thr")
    model.encode("one")
    assert Encoder.runs == 4, "'one' left first; 'thr' stayed"
    engine.configure(MemoSettings(process_bytes=0))
    assert not engine.entries
    model.encode("thr")
    assert Encoder.runs == 5


def test_value_gate_stores_only_costly_small_results(tmp_path: Path) -> None:
    stored: list[StageMemoStore] = []

    def exchange(request: Any, into: type[Any]) -> Any:
        if isinstance(request, StageMemoLookup):
            return MemoEntry(ok=True)
        stored.append(request)
        return MemoStored(ok=True, stored=bool(request.local))

    engine = stage_memo.Engine()
    engine.configure(MemoSettings(entry_bytes=4096))
    engine.open(exchange, tmp_path)
    model = _model(engine=engine)
    model.encode("x" * 20)
    assert stored[-1].local and stored[-1].reason == ""
    model.encode("y" * 200)
    assert stored[-1].reason == "too_large" and not stored[-1].local
    stats = engine.stats["Encoder.encode"]
    stats.read_s = 10.0  # a stored entry measured slower to read than this encode is to run
    model.encode("z" * 20)
    assert stored[-1].reason == "cheap"
    assert stats.skips == {"too_large": 1, "cheap": 1} and stats.stored_bytes > 0
    rows: list[tuple[str, dict[str, Any]]] = []
    engine.close(lambda kind, name, value, **fields: rows.append((name, fields)))
    assert [name for name, _ in rows].count("memo.call") == 3
    (method,) = [fields for name, fields in rows if name == "memo.method"]
    assert method["misses"] == 3 and method["skips"] == "cheap=1,too_large=1"


def test_install_keys_each_component_and_never_fails_a_load() -> None:
    device = SimpleNamespace(kind="cpu", sm=0, name="cpu")
    header = {"components": {"text_encoder": {"w": {"parts": {"value": [1, 2]}}}}, "configs": {}}

    def install(model: Encoder, headers: dict[str, Any]) -> None:
        stage_memo.install(
            model,
            torch=torch,
            headers=headers,
            assets=header,
            plan_rows=[{"component": "text_encoder", "key": "w", "encoding": "plain/1"}],
            adapters=[],
            roots={"text_encoder": torch.nn.Linear(2, 2), "vae": torch.nn.Linear(2, 2)},
            device=device,
            world=1,
            sealed={},
            fusion=None,
        )

    model = Encoder.for_test()
    install(model, {"text_encoder": header, "vae": header})
    Encoder.runs = 0
    memo = model._cozy_memo
    assert isinstance(memo, stage_memo.GenerationMemo)
    assert memo.components["text_encoder"] != memo.components["vae"]
    model.encode("x")
    model.encode("x")
    assert Encoder.runs == 1
    broken = Encoder.for_test()
    install(broken, {"text_encoder": {"components": {"text_encoder": object()}}})
    assert broken._cozy_memo is None, "an unreadable identity serves unmemoized"


def test_arrays_and_images_key_and_round_trip_and_non_finite_results_are_not_kept() -> None:
    class Imaging(Model[object]):
        runs = 0

        @uses_components("vae", memoize=True)
        def embed(self, pixels: Any, image: Any, scale: float) -> tuple[Any, Any, Any]:
            type(self).runs += 1
            return torch.tensor([float(pixels.sum()) * scale]), image.convert("L"), pixels[:2]

    model = Imaging.for_test()
    object.__setattr__(model, "_cozy_memo", _model()._cozy_memo)
    pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    image = Image.fromarray(np.full((4, 4, 3), 7, dtype=np.uint8))
    first = model.embed(pixels, image, 1.0)
    second = model.embed(pixels.copy(), image.copy(), 1.0)
    assert Imaging.runs == 1
    assert second[1].tobytes() == first[1].tobytes() and second[1].mode == "L"
    assert np.array_equal(second[2], first[2]) and second[2].dtype == np.uint8
    model.embed(pixels, image, float("nan"))
    model.embed(pixels, image, float("nan"))
    assert Imaging.runs == 3, "a non-finite result is returned and never kept"
