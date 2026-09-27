"""Admission classification controls on CPU; actual Sol arithmetic is a CUDA gate."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT
from cozy_runtime.internal import attention, attention_sol
from cozy_runtime.internal.attention_registry import register
from cozy_runtime.internal.encoding import DeviceFacts


def _dense(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    import torch

    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale
    ).transpose(1, 2)


def _inputs(device: str = "cpu") -> tuple[Any, ...]:
    import torch

    generator = torch.Generator(device=device).manual_seed(0x5A6E)
    return tuple(
        (torch.randn(1, 257, 4, 128, generator=generator, device=device) * 0.5).to(torch.bfloat16)
        for _ in range(3)
    )


def _cpu_wrapper(query: Any, key: Any, value: Any) -> Any:
    # CPU tests invoke the actual policy after its separately tested CUDA gate.
    layout, site = _ACTIVE_LAYOUT.get(), attention_sol._SITE.get()
    assert layout is not None and site is not None
    return attention_sol._execute(query, key, value, None, layout, site)


def _control(monkeypatch: pytest.MonkeyPatch, *, broken: bool = False) -> attention.Ready:
    """Real SDPA over full/subset keys tests classification, not Sol GPU math."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    def upstream(q: Any, k: Any, v: Any, **kwargs: Any) -> Any:
        assert kwargs == dict(
            scale=None,
            tau=1.0,
            thresh_type="exact",
            kv_splits=1,
            sink_start=0,
            sink_tokens=kwargs["sink_tokens"],
        )
        if kwargs["sink_tokens"] == q.shape[1]:
            return torch.zeros_like(q) if broken else _dense(query=q, key=k, value=v, scale=None)
        assert kwargs["sink_tokens"] == 33
        return _dense(query=q, key=k[:, :64], value=v[:, :64], scale=None)

    monkeypatch.setattr(
        attention_sol, "import_module", lambda _: SimpleNamespace(sol_attn=upstream)
    )
    monkeypatch.setattr(attention_sol, "_DENSE", _dense)
    member = register("_cozy_sol_admission_cpu_control", _cpu_wrapper)
    return attention.Ready(attention.BY_NAME["sol-attn"], member)


def test_sparse_approximation_is_reported_after_separate_arithmetic_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _control(monkeypatch)
    device = DeviceFacts(kind="cpu", name="CPU control", sm=0, driver="", configuration="", index=0)
    report = attention._probe_sol(entry, device, *_inputs())
    assert report["arithmetic_rel_l2_to_fp32"] < 0.02
    assert report["sparse_policy_rel_l2_to_fp32"] > 0.2
    assert report["sparse_error_meaning"] == "approximation_only_not_model_quality"
    assert report["calls"] == dict(sparse=2, dense_step=0, dense_path=0, dense_prefix=2)
    assert _ACTIVE_LAYOUT.get() is None
    assert attention_sol._SITE.get() is None
    applied = attention.Swap(replace(entry, admission=report), ()).apply()
    assert applied.document()["admission"]["sol-attn"] == report
    merged = attention.merge({"model": replace(applied, hosts={"dit": {"sol-attn": 1}}).document()})
    assert merged is not None and merged["admission"]["model"]["sol-attn"] == report


def test_broken_native_all_keys_control_refuses_before_prefix_can_hide_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _control(monkeypatch, broken=True)
    device = DeviceFacts(kind="cpu", name="CPU control", sm=0, driver="", configuration="", index=0)
    with pytest.raises(attention.AttentionRefusal, match="past its declared budget"):
        attention._probe_sol(entry, device, *_inputs())


def test_native_sol_admission_separates_all_keys_from_sparse_policy() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires an owned SM90 CUDA qualification worker")
    if importlib.util.find_spec("sol_attn") is None:
        pytest.skip("requires the declared upstream sol-attn package")
    attention_sol.bind_dense(_dense, "test-native-dense")
    candidate = attention.BY_NAME["sol-attn"]
    entry = attention.Ready(candidate, attention._member(candidate))
    device = DeviceFacts(
        kind="cuda",
        name=torch.cuda.get_device_name(),
        sm=90,
        driver="",
        configuration="",
        index=torch.cuda.current_device(),
    )
    with torch.no_grad():
        report = attention._computes(entry, device)
    assert report is not None
    assert report["arithmetic_control"] == "native_all_keys_selected"
    assert report["arithmetic_rel_l2_to_fp32"] <= 0.02
    assert report["repeat_equal"] and report["inputs_unchanged"]
