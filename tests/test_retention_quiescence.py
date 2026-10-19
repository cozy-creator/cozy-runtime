"""CPU control-flow checks only: no CUDA idle or retained allocation is simulated as proof."""

from __future__ import annotations

from typing import cast

import pytest

torch = pytest.importorskip("torch")

from tensorfs.varena import WeightSet  # noqa: E402
from tensorfs.varena.torch import Declared, Memory, Model, Unbound  # noqa: E402


class Exports:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def kept(self) -> list[tuple[int, int, int]]:
        self.calls.append("conservative")
        return []

    def kept_quiescent(self) -> list[tuple[int, int, int]]:
        self.calls.append("quiescent")
        return []


def test_quiescent_exports_reject_active_requests_and_forwards() -> None:
    memory, exports = Memory.open(torch.device("cpu")), Exports()
    model = Model(
        memory, "component", "key", torch.nn.Identity(), cast(WeightSet, exports), Declared()
    )
    # Only route/guard metadata: Memory stays CPU and no actual CUDA allocation is claimed.
    memory.granted, model.gpu, model.held = True, 0, 1
    assert model.kept() == ("", []) and exports.calls == ["conservative"]
    for request_id in ("failed", "next"):
        with pytest.raises(ValueError, match="handler failed"), memory.request(request_id):
            with pytest.raises(Unbound, match="no active request or forward"):
                model.kept(quiescent=True)
            raise ValueError("handler failed")
        assert not memory.active
        assert model.kept(quiescent=True) == ("", [])
    assert exports.calls == ["conservative", "quiescent", "quiescent"]
    memory.models[model.name] = model
    model.flight = 1
    with pytest.raises(Unbound, match="no active request or forward"):
        model.kept(quiescent=True)
    model.flight = 0
    other = Model(
        memory, "other", "other", torch.nn.Identity(), cast(WeightSet, exports), Declared()
    )
    memory.models[other.name], other.flight = other, 1
    with pytest.raises(Unbound, match="no active request or forward"):
        model.kept(quiescent=True)
    assert exports.calls == ["conservative", "quiescent", "quiescent"]
