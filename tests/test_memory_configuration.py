"""An unreadable configured limit must not silently become an unlimited GPU."""

from pathlib import Path

import pytest

from cozy_runtime.internal.config import ConfigError, parse_gpu_budget, read_gpu_budget


def test_iec_si_and_ordinal_budgets() -> None:
    assert parse_gpu_budget("3GiB") == ((-1, 3 * 1024**3),)
    assert parse_gpu_budget({"0": "3GB", "1": "1.5e9"}) == ((0, 3_000_000_000), (1, 1_500_000_000))


@pytest.mark.parametrize(
    "value", ["0", "-1GiB", "3GIBB", "1e999", False, {}, {0: "3GiB", 1: "bad"}, {-2: "3GiB"}]
)
def test_invalid_limit_is_a_configuration_refusal(value: object) -> None:
    with pytest.raises(ConfigError) as refused:
        parse_gpu_budget(value)
    assert refused.value.name == "gpu.budget"


def test_only_an_absent_budget_is_unconfigured(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    assert read_gpu_budget(path) == ()
    path.write_text("gpu:\n  future_setting: value\n")
    assert read_gpu_budget(path) == ()
    path.write_text("gpu:\n  budget: {0: 3GiB, 1: nope}\n")
    with pytest.raises(ConfigError):
        read_gpu_budget(path)
    path.write_text("gpu: false\n")
    with pytest.raises(ConfigError):
        read_gpu_budget(path)


@pytest.mark.parametrize("text", [
    '{"gpu": {"budget": "1GiB"}}',
    'defaults: &limits\n  budget: 1GiB\ngpu:\n  <<: *limits\n',
    'gpu: {budget: {0: 1GiB, 1: 512MiB}}',
])
def test_standard_yaml_budget_forms_never_silently_remove_limit(tmp_path: Path, text: str) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(text)
    budget = dict(read_gpu_budget(path))
    assert budget and next(iter(budget.values())) == 1024**3


@pytest.mark.parametrize(
    "text", ['gpu: {budget: [}', '[gpu, budget]', 'gpu: !!python/object:foo {}']
)
def test_malformed_or_nonmapping_runtime_yaml_refuses(tmp_path: Path, text: str) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        read_gpu_budget(path)
