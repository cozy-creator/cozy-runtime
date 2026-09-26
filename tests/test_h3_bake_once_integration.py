"""CPU contracts for direct, request-isolated H3 Turbo activation."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest


def _model_types() -> tuple[Any, Any]:
    pytest.importorskip("torch")
    try:
        from cozy_runtime.models.minimax_h3.model import H3TurboLoRA
        from cozy_runtime.models.minimax_h3.official import OfficialH3TurboLoRA
    except ImportError as exc:
        pytest.skip(f"installed Diffusers does not expose the H3 API: {exc}")
    return H3TurboLoRA, OfficialH3TurboLoRA


def test_prepared_overlay_is_reused_for_a_variant() -> None:
    _model_types()
    from cozy_runtime.models.minimax_h3.official import OfficialH3TurboLoRA

    adapter = cast(Any, OfficialH3TurboLoRA.__new__(OfficialH3TurboLoRA))
    config = {"hidden_size": 4}
    marker = object()
    adapter.specs = {"fl2va": (config, None), "ref2va": (config, None)}
    adapter.components = {"fl2va_turbo": marker, "ref2va_turbo": object()}
    base = SimpleNamespace(
        _dit_specs={
            "fl2va": (config, "adaln-pruned", None),
            "ref2va": (config, "adaln-pruned", None),
        }
    )
    first = adapter.overlay(base, "fl2va")
    second = adapter.overlay(base, "fl2va")
    assert first is second is marker


def test_request_state_is_restored_and_overlay_is_loaded_once() -> None:
    H3TurboLoRA, _ = _model_types()

    class State:
        def __init__(self) -> None:
            self.values = {"attention_kwargs": {"caller": "A"}}

        def get(self, name: str) -> Any:
            return self.values.get(name)

        def set(self, name: str, value: Any) -> None:
            self.values[name] = value

    class Checks:
        def component(self, _name: str, _root: Any) -> None:
            return None

        @contextmanager
        def forwards(self, _root: Any, _name: str) -> Any:
            yield

    state = State()
    checks = Checks()
    overlay = object()
    calls: list[str] = []

    def denoise(_task: str, current: State, **_kwargs: Any) -> str:
        calls.append("denoise")
        selected = current.get("attention_kwargs")
        assert selected["cozy_h3_bank"] == "turbo"
        assert selected["cozy_h3_overlay"] is overlay
        return "done"

    def load_overlay(_base: Any, _trunk: Any) -> Any:
        calls.append("overlay")
        return overlay

    adapter = cast(Any, H3TurboLoRA.__new__(H3TurboLoRA))
    adapter.pipe = SimpleNamespace(overlay=load_overlay)
    base = SimpleNamespace(
        pipe=SimpleNamespace(
            components={"fl2va_dit": object()},
            denoise=denoise,
        )
    )
    result = adapter._sample(
        base,
        "fl2va",
        state,
        on_step=None,
        cancel=None,
        checks=checks,
        sol_dense_steps=4,
    )
    assert result == "done"
    assert calls == ["overlay", "denoise"]
    assert state.get("attention_kwargs") == {"caller": "A"}
