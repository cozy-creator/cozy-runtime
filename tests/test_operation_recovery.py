"""Real dispatcher operators fail after a side effect; recovery must not repeat them.

These CPU faults exercise the dispatch recovery contract without allocating on a GPU.
They do not qualify low-VRAM inference or CUDA allocation recovery.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal.weights import (  # noqa: E402
    ResidencyRefusal,
    Weights,
    _recovery_type,
)


@pytest.mark.parametrize("effect", ["mutation", "rng", "opaque"])
def test_a_failed_operator_with_side_effects_is_never_replayed(effect: str) -> None:
    calls: list[int] = []
    generator = torch.Generator().manual_seed(11)
    expected = torch.Generator().manual_seed(11)
    torch.rand(1, generator=expected)
    # A custom kernel may change bytes or hidden state before reporting allocation failure.
    # Its schema alone is insufficient to establish that the failure was atomic.
    library = torch.library.Library("cozy_recovery_fault", "FRAGMENT")
    schema = "Tensor(a!) value" if effect == "mutation" else "Tensor value"
    library.define(f"{effect}({schema}) -> Tensor")

    def fail(value: Any) -> Any:
        calls.append(1)
        if effect == "mutation":
            value.add_(1)
        elif effect == "rng":
            torch.rand(1, generator=generator)
        raise torch.OutOfMemoryError("Tried to allocate 1 MiB")

    library.impl(effect, fail, "CPU")
    weights = Weights(torch, torch.device("cpu"), "cuda")
    with torch.inference_mode():
        value = torch.ones(1)
        with _recovery_type(torch)(weights), pytest.raises(ResidencyRefusal) as failed:
            getattr(torch.ops.cozy_recovery_fault, effect)(value)
        assert failed.value.code == "recovery_contract_unavailable"
        assert effect in str(failed.value)
        assert isinstance(failed.value.__cause__, torch.OutOfMemoryError)
        assert calls == [1]
        assert value.item() == (2 if effect == "mutation" else 1)
        if effect == "rng":
            assert torch.equal(generator.get_state(), expected.get_state())


def test_safe_successful_operations_preserve_outputs_and_input_mutations() -> None:
    weights = Weights(torch, torch.device("cpu"), "cuda")
    with torch.inference_mode():
        value = torch.arange(8, dtype=torch.float32)
        with _recovery_type(torch)(weights):
            result = torch.tanh(value + 2)
            value.add_(1)
        assert torch.equal(result, torch.tanh(torch.arange(8, dtype=torch.float32) + 2))
        assert torch.equal(value, torch.arange(8, dtype=torch.float32) + 1)


def test_nested_recovery_scopes_install_one_dispatch_mode_and_unwind_it() -> None:
    weights = Weights(torch, torch.device("cpu"), "cuda")
    # Activate dispatcher recovery over a real CPU plane without creating a CUDA context.
    # This exercises scope ownership, not device allocation recovery.
    weights.host_only = False
    current = torch.utils._python_dispatch._get_current_dispatch_mode
    with torch.inference_mode(), weights.recovering():
        mode = current()
        assert mode is not None
        with weights.recovering():
            assert current() is mode
            assert torch.equal(torch.ones(1) + 1, torch.full((1,), 2.0))
        assert current() is mode
    assert current() is None
    assert not weights._recovering
