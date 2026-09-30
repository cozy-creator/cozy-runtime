"""Real CPU/meta owner/alias tests; simulated device accounting never opens CUDA."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.author._retry import pure, read_whole, retry_state
from cozy_runtime.internal import memory, tiers


class Block(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))

    def forward(self, value: Any) -> Any:
        return value @ self.weight


class Network(torch.nn.Module):  # type: ignore[misc]
    _no_split_modules: ClassVar[list[str]] = ["Block"]

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Linear(1, 1)
        self.other = torch.nn.Linear(3, 3)
        self.blocks = torch.nn.ModuleList([Block(), Block()])

    @property
    def device(self) -> Any:
        return next(self.parameters()).device

    @property
    def dtype(self) -> Any:
        return next(self.parameters()).dtype

    def forward(self, value: Any) -> Any:
        return value + self.other.weight.sum()


class Store:
    """Only immutable-census capability; tests do not invoke a disk or device loader."""

    def __init__(self, root: Any, name: str) -> None:
        self.keys = frozenset(f"{name}.{key}" for key in root.state_dict())

    def unit_loads(self, name: str, keys: frozenset[str]) -> bool:
        return keys == self.keys

    def transient(self, name: str) -> int:
        return 0

    def load(self, name: str) -> object:
        raise AssertionError("no actual device or store fill in this CPU fixture")

    def load_unit(self, name: str, unit: tiers.Unit) -> None:
        raise AssertionError("no actual device or store fill in this CPU fixture")

    def drop(self, name: str) -> None:
        raise AssertionError("no actual device or store fill in this CPU fixture")

    def drop_unit(self, unit: tiers.Unit) -> None:
        raise AssertionError("no actual device or store fill in this CPU fixture")


def adopted(root: Any, *, forward_only: bool = True) -> tuple[memory.Planner, memory.Held]:
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt(
        "c", "network", root, Store(root, "network"), tiers.HOST, forward_only=forward_only
    )
    return owner, held


def test_split_keeps_whole_first_owner_real_and_preserves_exact_census() -> None:
    root = pure(Network())
    marker = root.marker.weight
    original = {name: tensor.clone() for name, tensor in root.state_dict().items()}
    owner, held = adopted(root)
    part = held.part
    assert part.anchor is not None and part.anchor is not part.common
    assert part.anchor.nbytes == 8 and part.common.nbytes == 48
    assert len(part.blocks) == 2 and part.nbytes == 88
    assert {id(module) for _, module in part.anchor.owners()} == {id(root.marker)}
    assert {key for unit in part.units for key in unit.live("network")} == Store(
        root, "network"
    ).keys
    assert root.marker.weight is marker and root.device == torch.device("cpu")
    assert root.dtype == torch.float32
    assert owner._scope_common(held) == (part.anchor,)
    assert all(torch.equal(root.state_dict()[name], tensor) for name, tensor in original.items())


@pytest.mark.parametrize(
    "kind",
    ["package-unknown", "forward-unknown", "read-whole", "mutable", "store-unknown", "buffer-only"],
)
def test_unknown_or_stronger_contract_retains_ordinary_common(kind: str) -> None:
    root = Network()
    if kind != "forward-unknown":
        pure(root)
    if kind == "read-whole":
        read_whole(root)
    if kind == "mutable":
        retry_state(root, lambda: None, lambda _: None, tensors=lambda: (root.other.weight,))
    if kind == "buffer-only":
        root = pure(torch.nn.Module())
        root.register_buffer("marker", torch.ones(2))
        root.other = torch.nn.Module()
        root.other.register_buffer("value", torch.ones(4))
    owner = memory.Planner(torch, torch.device("cpu"))
    store = None if kind == "store-unknown" else Store(root, "network")
    held = owner.adopt(
        "c", "network", root, store, tiers.HOST, forward_only=kind != "package-unknown"
    )
    assert held.part.anchor is None
    assert owner._scope_common(held) == (held.part.common,)


def test_tied_storage_pulls_complete_alias_owner_into_anchor() -> None:
    root = pure(Network())
    root.other.weight = torch.nn.Parameter(root.marker.weight.view(1, 1))
    _, held = adopted(root)
    part = held.part
    assert part.anchor is not None
    assert {id(module) for _, module in part.anchor.owners()} >= {id(root.marker), id(root.other)}
    assert (
        root.marker.weight.untyped_storage().data_ptr()
        == root.other.weight.untyped_storage().data_ptr()
    )
    assert part.nbytes == 52  # 4 shared weight + 4 marker bias + 12 other bias + two16B blocks
    tiers.validate_anchor(part)


def test_meta_view_aliases_are_not_confused_with_unrelated_zero_data_pointers() -> None:
    root = Network().to(device="meta")
    root.other.weight = torch.nn.Parameter(root.marker.weight.view(1, 1))
    part = tiers.partition("network", root, tiers.DISK, Store(root, "network"))
    tiers.split_common(part)
    assert part.anchor is not None
    assert {id(module) for _, module in part.anchor.owners()} >= {id(root.marker), id(root.other)}
    assert len(part.blocks) == 2
    tiers.validate_anchor(part)


def test_alias_into_repeated_block_is_not_split_across_units() -> None:
    root = pure(Network())
    root.blocks[0].weight = root.marker.weight
    _, held = adopted(root)
    assert len(held.part.blocks) == 1
    assert held.part.anchor is not None
    assert id(root.blocks[0]) in {id(module) for _, module in held.part.anchor.owners()}
    tiers.validate_anchor(held.part)


@pytest.mark.parametrize("change", ["owner", "slot", "shape", "alias"])
def test_positive_split_contract_rejects_untracked_tensor_graph_mutation(change: str) -> None:
    root = pure(Network())
    owner, held = adopted(root)
    if change == "owner":
        root.extra = torch.nn.Linear(2, 2)
    elif change == "slot":
        root.register_parameter("new_first", torch.nn.Parameter(torch.ones(1)))
    elif change == "shape":
        root.other.weight = torch.nn.Parameter(torch.ones(4, 3))
    else:
        root.other.weight = torch.nn.Parameter(root.marker.weight.expand(3, 3))
    with pytest.raises(memory.MemoryRefusal, match=r"changed|alias") as failure:
        owner._validate_component(held)
    assert not failure.value.reusable and owner.unusable


def test_unproven_dynamic_component_does_not_acquire_a_new_topology_refusal() -> None:
    root = Network()
    owner, held = adopted(root, forward_only=False)
    root.extra = torch.nn.Linear(2, 2)
    owner._validate_component(held)
    assert held.part.anchor is None
    assert id(root.extra) in {id(module) for _, module in held.part.common.owners()}


def test_retry_python_cache_mutation_is_not_tensor_topology_mutation() -> None:
    root = Network()
    root.cache = []
    retry_state(
        root, lambda: list(root.cache), lambda saved: root.cache.__setitem__(slice(None), saved)
    )
    owner, held = adopted(root)
    assert held.part.anchor is not None
    root.cache.append(torch.ones(2))
    root.counter = 12
    owner._validate_component(held)


def test_all_operation_common_is_loaded_but_only_anchor_pins_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = pure(Network())
    owner, held = adopted(root)
    part = held.part
    assert part.anchor is not None
    owner.scopes = [("c", frozenset(("network",)))]
    owner.cuda = True
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(owner, "allocatable", lambda: 1024)
    loaded = []

    def load(_part: tiers.Component, unit: tiers.Unit) -> None:
        loaded.append(unit)
        unit.tier = tiers.DEVICE

    monkeypatch.setattr(owner, "_load", load)
    owner._enter_scope("c", "render", ("network",))
    assert loaded == [part.anchor]
    loaded.clear()
    owner._load_common(held)
    assert loaded == list(part.common_units)
    assert owner._evictable(held) == part.common.nbytes
    assert root.marker.weight.device == torch.device("cpu")  # still the original actual tensor
    held.whole = True
    assert owner._evictable(held) == 0


def test_dtype_conversion_visits_each_common_owner_once_and_keeps_real_metadata() -> None:
    class CountingLinear(torch.nn.Linear):  # type: ignore[misc]
        def _apply(self, fn: Any, recurse: bool = True) -> Any:
            self.conversions = getattr(self, "conversions", 0) + 1
            return super()._apply(fn, recurse)

    root = pure(Network())
    root.marker = CountingLinear(1, 1)
    root.other = CountingLinear(3, 3)
    _, held = adopted(root)
    part = held.part
    before = part.nbytes
    tiers.intercept_apply(torch, part, lambda: tiers.validate_anchor(part))
    root.to(dtype=torch.float64)
    assert root.marker.conversions == root.other.conversions == 1
    assert root.dtype == torch.float64 and root.device == torch.device("cpu")
    assert part.nbytes == 2 * before
    assert all(tensor.dtype == torch.float64 for unit in part.units for *_, tensor in unit.slots())
    tiers.validate_anchor(part)


def test_compiled_component_keeps_conservative_common() -> None:
    root = pure(Network())
    root._compiled_call_impl = lambda *args: None
    _, held = adopted(root)
    assert held.compiled and held.part.anchor is None


def test_actual_replay_reclaims_remainder_without_losing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling = pure(Network())
    owner, other = adopted(sibling)
    current = pure(torch.nn.Linear(1, 1))
    held = owner.adopt("c", "current", current, None, tiers.DEVICE)
    for unit in other.part.units:
        unit.tier = tiers.DEVICE
    owner.scopes = [("c", frozenset(("network", "current")))]
    owner.cuda = True

    def allocated() -> int:
        return sum(part.part.on(tiers.DEVICE) for part in owner.held.values())

    monkeypatch.setattr(owner, "_allocated", allocated)
    monkeypatch.setattr(owner, "_reserved", allocated)
    monkeypatch.setattr(owner, "allocatable", lambda: 104 - allocated())
    monkeypatch.setattr(owner, "_cap", lambda: None)
    monkeypatch.setattr(owner, "_sample", lambda: None)
    monkeypatch.setattr(owner, "_home", lambda _part, unit: setattr(unit, "tier", tiers.HOST))
    monkeypatch.setattr(tiers, "trim", lambda *_: None)
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None, raising=False)
    key = memory.OpKey(held.kind, "forward", (((1,), "torch.float32"),), mode=memory._mode(current))
    owner.records[key] = memory.Record(ok=0)
    arm, rollback = owner._arm, owner._rollback

    def cpu_arm(*args: Any) -> memory.Snapshot:
        owner.cuda = False
        try:
            return arm(*args)
        finally:
            owner.cuda = True

    def cpu_rollback(*args: Any) -> None:
        owner.cuda = False
        try:
            rollback(*args)
        finally:
            owner.cuda = True

    monkeypatch.setattr(owner, "_arm", cpu_arm)
    monkeypatch.setattr(owner, "_rollback", cpu_rollback)
    calls = []
    marker = sibling.marker.weight

    def forward(value: Any) -> Any:
        calls.append(owner.allocatable())
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4 B")
        return value + 1

    with torch.no_grad():
        result = owner.run(held, "forward", forward, (torch.ones(1),), {})
    assert torch.equal(result, torch.tensor([2.0]))
    assert calls == [8, 56]
    assert other.part.anchor is not None and other.part.anchor.tier == tiers.DEVICE
    assert other.part.common.tier == tiers.HOST
    assert sibling.marker.weight is marker and sibling.device == torch.device("cpu")


def test_new_mutable_sibling_state_is_rechecked_between_quote_and_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = pure(Network())
    owner, held = adopted(root)
    for unit in held.part.units:
        unit.tier = tiers.DEVICE
    owner.scopes = [("c", frozenset(("network",)))]
    assert owner._evictable(held) == 80  # 48B remainder plus32B blocks
    retry_state(root, lambda: None, lambda _: None, tensors=lambda: (root.other.weight,))
    moved = []

    def home(_part: tiers.Component, unit: tiers.Unit) -> None:
        moved.append(unit)
        unit.tier = tiers.HOST

    monkeypatch.setattr(owner, "_home", home)
    owner._offload(held)
    assert held.part.common.mutable and held.part.common.tier == tiers.DEVICE
    assert held.part.anchor is not None and held.part.anchor.tier == tiers.DEVICE
    assert moved == list(held.part.blocks)
    assert owner._evictable(held) == 0


def test_new_cross_unit_alias_is_rejected_before_actual_victim_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = pure(Network())
    root.marker = torch.nn.Linear(3, 3)
    owner, held = adopted(root)
    for unit in held.part.units:
        unit.tier = tiers.DEVICE
    owner.scopes = [("c", frozenset(("network",)))]
    assert owner._evictable(held) == 80
    root.other.weight = torch.nn.Parameter(root.marker.weight)
    moved = []
    monkeypatch.setattr(owner, "_home", lambda *_: moved.append(True))
    with pytest.raises(memory.MemoryRefusal, match="alias"):
        owner._offload(held)
    assert not moved


def test_scope_layout_and_validation_cost_are_reported_without_fake_byte_savings() -> None:
    owner, held = adopted(pure(Network()))
    owner._validate_component(held)
    document = owner.document("c")
    layout = cast(dict[str, dict[str, Any]], document["scope_backing"])[held.key]
    assert layout["metadata_anchor_bytes"] == 8
    assert layout["operation_common_bytes"] == 48
    assert layout["operation_common_device_bytes"] == 0
    assert layout["scope_minimum_bytes"] == 8 and layout["anchor_only_scope"]
    validation = cast(dict[str, int], document["anchor_validation"])
    assert validation["calls"] == 1
    assert validation["ns"] >= 0


def test_meta_views_charge_full_unique_backing_but_do_not_claim_store_reconstructibility() -> None:
    root = pure(torch.nn.Module())
    backing = torch.empty(20, device="meta")
    root.first = torch.nn.Module()
    root.first.weight = torch.nn.Parameter(backing[2:6].view(2, 2))
    root.other = torch.nn.Module()
    root.other.weight = torch.nn.Parameter(backing[10:14].view(2, 2))
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "network", root, Store(root, "network"), tiers.DISK, forward_only=True)
    assert held.part.nbytes == 80  # not32 view bytes, and not160 duplicated backing bytes
    assert not tiers.store_safe(held.part.common)
    assert held.part.anchor is None  # store lacks alias/view reconstruction authority
    assert root.first.weight.untyped_storage()._cdata == root.other.weight.untyped_storage()._cdata


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_store_rejects_noncontiguous_full_backing_in_both_materialized_and_meta_forms(
    device: str,
) -> None:
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(torch.empty((2, 3), device=device).t())
    part = tiers.partition("network", root, tiers.HOST if device == "cpu" else tiers.DISK)
    assert part.nbytes == 24
    assert not tiers.store_safe(part.common)


@pytest.mark.parametrize("whole", [False, True])
def test_initial_parked_alias_refill_refuses_before_loader_or_binding_mutation(whole: bool) -> None:
    root = pure(torch.nn.Module())
    backing = torch.empty(20, device="meta")
    root.first = torch.nn.Parameter(backing[2:6])
    root.second = torch.nn.Parameter(backing[10:14])
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "network", root, Store(root, "network"), tiers.DISK, forward_only=True)
    before = (root.first, root.second)
    # Store.load/load_unit raise AssertionError if called: geometry must refuse first.
    with pytest.raises(RuntimeFailure, match="store cannot reconstruct") as failure:
        tiers.from_disk(torch, held.part, None if whole else held.part.common)
    assert failure.value.code == "memory_capability"
    assert root.first is before[0] and root.second is before[1]
    assert held.part.common.tier == tiers.DISK and not held.part.moving
