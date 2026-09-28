"""Real preparation selector and CPU attention; no storage/GPU fill is claimed."""

from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import Executor, _local_attention_pin
from cozy_runtime.internal.executor_commands import Binding, Budgets, Load, ModelLoad


def _roots() -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    processor_type: Any = WanAttnProcessor
    return {
        name: torch.nn.ModuleList(
            [WanAttention(dim=8, heads=1, dim_head=8, processor=processor_type())]
        )
        for name in ("dit", "vae")
    }


@pytest.mark.parametrize("owner", ["model", "alias"])
def test_preparation_accepts_declared_single_model_names_and_retains_pin(
    tmp_path: Path, owner: str
) -> None:
    torch = pytest.importorskip("torch")
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    roots = _roots()
    binding = Binding(model_parameter_name="model", model_parameter_names=("alias", "model"))
    device = DeviceFacts(kind="cpu", name="CPU", sm=0, driver="", configuration="", index=0)
    chosen: list[str] = []

    def choose(context: Any) -> None:
        chosen.append(context.component)

    pin = f"{owner}/dit=sdpa"
    applied = executor._prepare_attention(binding, roots, pin, device, choose=choose)
    assert applied.pin == pin and applied.document()["pin"] == pin
    assert applied.hosts == {"dit": {"sdpa": 1}, "vae": {"sdpa": 1}}
    assert chosen == ["vae"]
    with torch.no_grad():
        assert roots["dit"][0](torch.ones(1, 2, 8)).isfinite().all()
    for invalid in ("unknown/dit=sdpa", f"{owner}/missing=sdpa"):
        with pytest.raises(attention.AttentionRefusal, match="matches no"):
            executor._prepare_attention(binding, roots, invalid, device, optional_scope=True)


def test_prepare_many_validates_aliases_before_model_work(tmp_path: Path) -> None:
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    rows = (
        ModelLoad(
            binding=Binding(
                model_binding_path="generate.models.base",
                model_parameter_name="base",
                model_parameter_names=("base", "alternate"),
            ),
            budgets=Budgets(),
        ),
        ModelLoad(
            binding=Binding(
                model_binding_path="generate.models.other",
                model_parameter_name="other",
                model_parameter_names=("other",),
            ),
            budgets=Budgets(),
        ),
    )
    # The actual aggregate validation must recognize all aliases. An unknown
    # model refuses before the deliberately incomplete binding can reach fill.
    command = Load(
        construction="k",
        models=rows,
        authorized_device_limit_bytes=1,
        attention_pin="absent/dit=sdpa",
    )
    reply = executor._prepare_many(command, rows)
    assert reply["code"] == "attention_kernel_unsupported"
    assert "alternate" in reply["detail"]
    for row in rows:
        names = row.binding.parameter_names()
        translated = attention.for_model("alternate/dit=sdpa", names)
        assert translated == ("dit=sdpa" if row is rows[0] else "")
    assert (
        _local_attention_pin("alternate/dit=sdpa", rows[0].binding.parameter_names()) == "dit=sdpa"
    )


def test_component_scope_can_skip_unrelated_construction_without_broadening(tmp_path: Path) -> None:
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    device = DeviceFacts(kind="cpu", name="CPU", sm=0, driver="", configuration="", index=0)
    roots = _roots()
    del roots["dit"]
    result = executor._prepare_attention(
        Binding(model_parameter_name="other"), roots, "dit=sdpa", device, optional_scope=True
    )
    assert result.pin == ""
    assert result.hosts == {"vae": {"sdpa": 1}}
