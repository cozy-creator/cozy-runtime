"""Real CPU/meta tensors through production adoption; simulated device placement only."""

from typing import Any

import pytest

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import memory, tiers

torch = pytest.importorskip("torch")


class Block(torch.nn.Linear):  # type: ignore[name-defined, misc]  # optional Torch test base
    pass


class Network(torch.nn.Module):  # type: ignore[name-defined, misc]  # optional Torch test base
    _no_split_modules = ["Block"]

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Linear(2, 2, bias=False)
        self.blocks = torch.nn.ModuleList([Block(2, 2, bias=False), Block(2, 2, bias=False)])
        self.register_buffer("count", torch.ones(2, dtype=torch.int64))


class Store:
    def __init__(self, root: Any) -> None:
        self.keys = frozenset("network." + key for key in root.state_dict())
        self.fail = False

    def unit_loads(self, name: str, keys: frozenset[str]) -> bool:
        return keys == self.keys

    def load_schema(self, name: str, unit: tiers.Unit) -> tiers.RefillSchema:
        return tiers.RefillSchema(tuple(tensor.dtype for *_, tensor in unit.slots()))

    def transient(self, name: str) -> int:
        return 0

    def load_unit(self, name: str, unit: tiers.Unit) -> None:
        for _, module in unit.owners():
            module.to_empty(device="cpu", recurse=False)
        if self.fail:
            raise RuntimeError("fill failed")

    def drop_unit(self, unit: tiers.Unit) -> None:
        for _, module in unit.owners():
            module.to_empty(device="meta", recurse=False)

    def load(self, name: str) -> object:
        raise AssertionError("unit store")

    def drop(self, name: str) -> None:
        raise AssertionError("unit store")


def adopted() -> tuple[memory.Planner, memory.Held, Store]:
    root = Network().to("meta")
    store = Store(root)
    owner = memory.Planner(torch, torch.device("cpu"))
    # Install production interception without invoking a CUDA method/allocation.
    owner.cuda = True
    held = owner.adopt("c", "network", root, store, tiers.DISK)
    return owner, held, store


def test_production_adopt_quotes_parked_common_and_streamed_block_after_double() -> None:
    owner, held, store = adopted()
    part = held.part
    key = memory.OpKey(held.kind, "forward", ())
    owner.records[key] = memory.Record(ok=999)
    part.root.double()
    assert key not in owner.records
    # common: 16B floating weight +16B integer buffer; double changes only weight.
    assert part.common.nbytes == 32
    assert tiers.load_demand(torch, part.common) == tiers.LoadDemand(48, 64)
    assert tiers.load_demand(torch, part.blocks[0]) == tiers.LoadDemand(32, 48)
    assert owner._need(part, 0) == 96  # settled common48 +one streamed conversion48
    assert owner._need(part, 2) == 128  # first final80 +last peak48
    assert part.root.marker.weight.dtype == torch.float32  # metadata not falsified
    tiers.from_disk(torch, part, part.common)
    assert part.common.nbytes == 48 and part.root.marker.weight.dtype == torch.float64
    assert part.root.count.dtype == torch.int64
    assert owner._need(part, 0) == 48  # common already charged on device


@pytest.mark.parametrize(
    "method,dtype",
    [("half", torch.float16), ("bfloat16", torch.bfloat16), ("double", torch.float64)],
)
def test_canonical_dtype_paths_preserve_requested_mode(method: str, dtype: Any) -> None:
    _, held, _ = adopted()
    getattr(held.part.root, method)()
    tiers.from_disk(torch, held.part, held.part.common)
    assert held.part.root.marker.weight.dtype == dtype
    assert held.part.root.count.dtype == torch.int64


def test_chained_casts_quote_largest_intermediate_not_just_final_dtype() -> None:
    _, held, _ = adopted()
    held.part.root.double().half()
    assert tiers.load_demand(torch, held.part.blocks[0]) == tiers.LoadDemand(8, 48)
    tiers.from_disk(torch, held.part, held.part.blocks[0])
    assert held.part.blocks[0].nbytes == 8
    assert held.part.root.blocks[0].weight.dtype == torch.float16


def test_whole_fill_peak_is_distinct_from_sequential_fill_peak() -> None:
    _, held, _ = adopted()
    held.part.root.double()
    assert tiers.load_sequence(torch, held.part.units) == tiers.LoadDemand(112, 128)
    assert tiers.load_sequence(torch, held.part.units, whole=True) == tiers.LoadDemand(112, 128)
    held.part.root.half()
    assert tiers.load_sequence(torch, held.part.units).resident == 40
    assert tiers.load_sequence(torch, held.part.units).peak == 80
    assert tiers.load_sequence(torch, held.part.units, whole=True).peak == 96


def test_unknown_transform_refuses_before_mutating_any_queue() -> None:
    _, held, _ = adopted()
    with pytest.raises(RuntimeFailure, match="unbounded"):
        held.part.root._apply(lambda tensor: tensor.double())
    assert not any(unit.pending for unit in held.part.units)
    assert held.part.root.marker.weight.dtype == torch.float32


def test_failed_refill_retains_source_and_pending_mode_for_same_retry() -> None:
    _, held, store = adopted()
    held.part.root.double()
    unit = held.part.blocks[0]
    promised = tiers.load_demand(torch, unit)
    store.fail = True
    with pytest.raises(RuntimeError, match="fill failed"):
        tiers.from_disk(torch, held.part, unit)
    assert unit.tier == tiers.DISK and unit.pending and not unit.applied
    assert unit.nbytes == 16 and tiers.load_demand(torch, unit) == promised
    store.fail = False
    tiers.from_disk(torch, held.part, unit)
    assert unit.nbytes == 32 and unit.tier == tiers.DEVICE


def test_meta_alias_conversion_is_refused_before_changing_registered_backing() -> None:
    _, held, _ = adopted()
    root = held.part.root
    root.marker.register_parameter("alias", root.marker.weight)
    with pytest.raises(RuntimeFailure, match="alias"):
        root.double()
    assert root.marker.weight is root.marker.alias
    assert not any(unit.pending for unit in held.part.units)


def test_device_only_request_keeps_existing_managed_placement() -> None:
    _, held, _ = adopted()
    held.part.root.cpu()
    assert all(unit.tier == tiers.DISK and not unit.pending for unit in held.part.units)


def test_post_fill_conversion_failure_preserves_production_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, held, _ = adopted()
    held.part.root.double()
    unit = held.part.blocks[0]
    original = unit.pending[0]
    failed = False

    def conversion(tensor: Any) -> Any:
        nonlocal failed
        if not tensor.is_meta and not failed:
            failed = True
            raise ValueError("cast failed after fill")
        return original(tensor)

    # Inject a transient execution failure into the queued canonical conversion;
    # the requested target remains double on both attempts.
    unit.pending[0] = tiers.DTypeConvert(conversion)
    with pytest.raises(ValueError, match="cast failed"):
        tiers.from_disk(torch, held.part, unit)
    assert unit.tier == tiers.DISK and unit.nbytes == 16
    assert unit.pending and not unit.applied
    tiers.from_disk(torch, held.part, unit)
    assert unit.nbytes == 32 and held.part.root.blocks[0].weight.dtype == torch.float64


def test_converted_profile_identity_cannot_leak_to_same_kind_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, held, _ = adopted()
    root = Network().to("meta")
    sibling = owner.adopt("other", "network", root, Store(root), tiers.DISK)
    assert sibling.kind == held.kind
    held.part.root.double()
    sibling.part.root.half()
    assert held.conversion_epoch == sibling.conversion_epoch
    modes = []

    class StopBeforePreparation(Exception):
        pass

    def plan(_held: Any, key: memory.OpKey, *_: Any) -> Any:
        modes.append(key.mode)
        raise StopBeforePreparation

    monkeypatch.setattr(owner, "_plan", plan)
    for item in [held, sibling]:
        with pytest.raises(StopBeforePreparation):
            owner.run(item, "forward", lambda: None, (), {})
    assert modes[0] != modes[1]
    assert "weight_conversion" in dict(modes[0])
    assert "weight_conversion" in dict(modes[1])
    first = memory.OpKey(held.kind, "forward", (((1,), "torch.int64"),), mode=modes[0])
    second = memory.OpKey(sibling.kind, "forward", (((1,), "torch.int64"),), mode=modes[1])
    owner.records[first] = memory.Record(ok=999)
    assert owner._estimate(second) == (None, False)


def test_noop_cast_preserves_full_shared_meta_backing_quote() -> None:
    _, held, _ = adopted()
    held.part.root.float()  # canonical no-op queue obtained through production interception
    conversion = held.part.common.pending[0]
    root = torch.nn.Module()
    backing = torch.empty(16, device="meta", dtype=torch.float32)
    root.register_parameter("a", torch.nn.Parameter(backing[4:8]))
    root.register_parameter("b", torch.nn.Parameter(backing[:2]))
    unit = tiers.Unit("", root, tier=tiers.DISK, nbytes=backing.untyped_storage().nbytes())
    unit.pending.append(conversion)
    assert tiers.load_demand(torch, unit) == tiers.LoadDemand(64, 64)


def test_partial_host_conversion_reservation_failure_refreshes_actual_bytes() -> None:
    root = Network()
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    owner.cuda = True  # production adapters, no CUDA operation
    held = owner.adopt("c", "network", root, None, tiers.HOST)
    key = memory.OpKey(held.kind, "forward", ())
    owner.records[key] = memory.Record(ok=999)
    # Common's floating weight needs32B; the next block's32B is not prepaid.
    with owner.host_credit(32), pytest.raises(RuntimeFailure) as failure:
        root.double()
    assert failure.value.code == "host_shortfall"
    assert root.marker.weight.dtype == torch.float64
    assert root.blocks[0].weight.dtype == torch.float32
    assert held.part.common.nbytes == 48  # actual32B weight +unchanged16B integer buffer
    assert [unit.nbytes for unit in held.part.blocks] == [16, 16]
    assert tiers.load_demand(torch, held.part.common) == tiers.LoadDemand(48, 48)
    assert owner._need(held.part, 0) == 64
    assert len(held.part.common.applied) == 1
    assert not held.part.blocks[0].applied and not held.part.blocks[0].pending
    assert key not in owner.records and not owner.unusable


def test_partial_owner_cast_failure_preserves_exception_and_truthful_sizes() -> None:
    failure = ValueError("owner cast failed after changing backing")

    class FailingBlock(Block):
        def _apply(self, fn: Any, recurse: bool = True) -> Any:
            super()._apply(fn, recurse)
            raise failure

    root = Network()
    root._no_split_modules = ["FailingBlock"]
    root.blocks = torch.nn.ModuleList([FailingBlock(2, 2, bias=False) for _ in range(2)])
    owner = memory.Planner(torch, torch.device("cpu"), grouped=True)
    owner.cuda = True
    held = owner.adopt("c", "network", root, None, tiers.HOST)
    with owner.host_credit(128), pytest.raises(ValueError) as caught:
        root.double()
    assert caught.value is failure
    assert held.part.common.nbytes == 48
    assert [unit.nbytes for unit in held.part.blocks] == [32, 16]
    assert tiers.load_demand(torch, held.part.blocks[0]) == tiers.LoadDemand(32, 32)
    assert not held.part.blocks[0].applied  # partial owner work is not a completed replay
    assert not held.part.blocks[0].pending
