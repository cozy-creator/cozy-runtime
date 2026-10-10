"""CPU contract proof for full unprofiled warmup through the diagnostic entrypoint."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NoReturn

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def package(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    root = Path(__file__).parents[1] / "research/h3-request-profile/h3_request_profile"
    name = "h3_profile_warmup_fixture"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name + ".instrument", None)


@pytest.mark.parametrize("name,steps", [("turbo", 8), ("regular", 30)])
def test_unprofiled_entrypoint_delegates_full_request_with_original_objects(
    package: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, steps: int,
) -> None:
    from cozy_runtime.author import Outputs, VideoAsset
    from cozy_runtime.author.fakes import fake_attempt

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("unprofiled warmup must not construct a profiler or wrap the handler")

    monkeypatch.setattr(package, "profile_request", forbidden)
    monkeypatch.setattr(torch.profiler, "profile", forbidden)
    ctx, assets, lora, tel = object(), object(), object(), object()
    out = Outputs(fake_attempt(spool=tmp_path / "spool"))
    counters = {"h2d_bytes": 11, "lora_bakes": 0}
    model = SimpleNamespace(
        _cozy_residency=SimpleNamespace(weights=SimpleNamespace(facts=lambda: dict(counters)))
    )
    payload_type = package.TurboInput if name == "turbo" else package.RegularInput
    payload = payload_type(prompt="full warmup", duration_s=15, seed=581204, profile_enabled=False)
    expected = (ctx, payload, assets, model, lora, out, tel) if name == "turbo" else (
        ctx, payload, assets, model, out, tel
    )
    video, warnings = VideoAsset("original-video"), ["unchanged"]
    calls: list[int] = []

    def retained(*args: object) -> SimpleNamespace:
        assert len(args) == len(expected)
        assert all(actual is wanted for actual, wanted in zip(args, expected, strict=True))
        for step in range(steps):
            calls.append(step)
            counters["h2d_bytes"] += 7
        return SimpleNamespace(video=video, warnings=warnings)

    selected: list[str] = []

    def handler(symbol: str) -> Callable[..., SimpleNamespace]:
        selected.append(symbol)
        return retained

    monkeypatch.setattr(package, "retained_handler", handler)
    result = package.app.get(name).fn(*expected)
    assert selected == ["fl2va_turbo" if name == "turbo" else "fl2va"]
    assert calls == list(range(steps))
    assert result.render.video is video and result.render.warnings is warnings
    assert {p.name for p in result.trace.path.iterdir()} == {"summary.json"}
    facts = json.loads((result.trace.path / "summary.json").read_bytes())
    assert facts["profile_enabled"] is False
    assert facts["render_call"]["plane_delta"]["h2d_bytes"] == steps * 7
    assert facts["render_call"]["wall_ms"] >= 0


def test_enabled_default_keeps_existing_profiler_delegation(
    package: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = package.TurboInput(prompt="trace", duration_s=15, seed=581204)
    assert payload.profile_enabled is True
    ctx, assets, base, lora, out, tel = (object() for _ in range(6))
    observed_out, observed_tel, rendered, trace, summary = (object() for _ in range(5))
    calls: list[tuple[object, ...]] = []

    def retained(*args: object) -> object:
        calls.append(args)
        return rendered

    def profile(
        model: object, dense: int, render: Callable[[object, object], object], **options: object,
    ) -> tuple[object, object, object]:
        assert model is base and dense == 4
        assert options == {
            "outputs": out, "telemetry": tel, "detailed": False,
            "full_request_trace": True, "trace_cpu_ops": False,
        }
        assert render(observed_out, observed_tel) is rendered
        return rendered, trace, summary

    monkeypatch.setattr(package, "retained_handler", lambda name: retained)
    monkeypatch.setattr(package, "profile_request", profile)
    monkeypatch.setattr(package, "saved", lambda result, output: (result, output))
    assert package.app.get("turbo").fn(ctx, payload, assets, base, lora, out, tel) == (
        (rendered, trace, summary), out
    )
    assert calls == [(ctx, payload, assets, base, lora, observed_out, observed_tel)]


def test_unprofiled_failure_preserves_handler_exception_without_artifacts(package: ModuleType) -> None:
    instrument = sys.modules[package.__name__ + ".instrument"]
    failure = ValueError("original handler failure")

    def broken() -> NoReturn:
        raise failure

    with pytest.raises(ValueError) as caught:
        instrument.unprofiled_request(object(), broken, outputs=object())
    assert caught.value is failure
