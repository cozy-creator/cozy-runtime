"""Reclaim diagnostics explain actual gates without changing retained tensor storage."""

from typing import Any

import pytest

from cozy_runtime.author import pure
from cozy_runtime.internal import memory
from test_host_reclaim_tensors import resident


@pytest.mark.parametrize(
    "reason",
    ["mutable_state", "compiled_component", "active_scope", "no_store", "unsafe_storage_geometry"],
)
def test_retained_reason_and_bytes_follow_the_actual_gate(reason: str) -> None:
    owner, held = resident("model")
    root = held.part.root
    if reason == "mutable_state":
        root._cozy_retry_tensors = lambda: (root.weight,)
    elif reason == "compiled_component":
        root._compiled_call_impl = lambda *args: None
    elif reason == "active_scope":
        owner.scopes.append(("model", frozenset()))
    elif reason == "no_store":
        held.part.store = None
    else:
        root.register_buffer("alias", root.weight.detach()[1:])
    assert owner.reclaim_host(128) == 0
    status = owner.host_reclaim_status()
    assert status["bytes_by_reason"] == {reason: 128}
    assert owner.host_memory()["managed_host_bytes"] == 128


def test_late_foreign_hook_records_the_first_loss_of_readonly_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held = resident("model")
    root = held.part.root
    pure(root)
    assert owner._state_contract(held, [])
    hook = root.register_forward_hook(lambda module, args, output: None)
    assert owner._state_contract_reason(held, []) == "unclaimed_forward_hook"
    owner.cuda = True  # Drive pre-placement contract logic; the plan never reaches a device.

    def stop(*args: Any) -> memory.Plan:
        raise memory.MemoryRefusal("test_boundary", "stop before device planning")

    monkeypatch.setattr(owner, "_plan", stop)
    with pytest.raises(memory.MemoryRefusal, match="stop before"):
        owner.run(held, "forward", root, (), {})
    assert held.readonly_lost_reason == "unclaimed_forward_hook"
    hook.remove()
    assert owner._state_contract(held, [])
    assert not held.part.common.readonly  # A later declaration cannot validate old weight bytes.
    owner.cuda = False
    assert owner.reclaim_host(128) == 0
    status = owner.host_reclaim_status()
    assert status["bytes_by_reason"] == {"readonly_authority_lost": 128}
    components = status["components"]
    assert isinstance(components, dict)
    assert components[held.key]["readonly_lost_reason"] == "unclaimed_forward_hook"


def test_explicit_pure_hook_and_unknown_hook_remain_distinct() -> None:
    owner, held = resident("model")
    root = held.part.root
    pure(root)

    def capture(module: Any, args: Any, output: Any) -> None:
        return None

    root.register_forward_hook(pure(capture))
    assert owner._state_contract_reason(held, []) == ""
    unknown = root.register_forward_pre_hook(lambda module, args: None)
    try:
        assert owner._state_contract_reason(held, []) == "unclaimed_forward_hook"
    finally:
        unknown.remove()
    assert owner._state_contract(held, [])
