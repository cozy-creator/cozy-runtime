"""Native TensorFS bytes + production strictfill/refill; CPU transport replaces CUDA DMA only."""

from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any
import io

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor
from cozy_runtime.author import pure
from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.author._loader import canonical_dtype
from cozy_runtime.internal import fill, memory, tiers
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.encoding.formats import (
    ScaledScalarDequant,
    SPEC_SCALED_SCALAR_WEIGHT_ONLY,
)

torch = pytest.importorskip("torch")


def fixture(
    tmp_path: Path,
    *,
    whole: bool = False,
    mixed: bool = False,
    decoded: bool = False,
    width: int = 2,
    derived_count: int = 1,
    derived_last: bool = False,
    derived_second_count: int = 0,
) -> tuple[Any, ...]:
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList(
        [torch.nn.Linear(width, 2, bias=False, dtype=torch.float16) for _ in range(3)]
    )
    root._no_split_modules = ["Linear"]
    with torch.no_grad():
        for layer in root.layers:
            layer.weight.copy_(
                torch.tensor(
                    [[1.0009765625, 0.333251953125], [-2.00390625, 0.000123]], dtype=torch.float16
                ).repeat(1, width // 2)
            )
    if mixed:
        root.register_buffer("count", torch.tensor([17], dtype=torch.int64))
        (root.layers[-1] if derived_last else root).register_buffer(
            "table", torch.full((derived_count,), 0.123456, dtype=torch.float32), persistent=False
        )
    if derived_second_count:
        root.register_buffer("other_table", torch.zeros(derived_second_count), persistent=False)
    root.requires_grad_(False)
    pure(root)
    originals = {k: v.detach().clone() for k, v in root.state_dict().items()}
    native = tensorfs.Store.ensure(str(tmp_path / "store"))

    def dtype(t: Any) -> str:
        return canonical_dtype(t.dtype)

    payloads = {
        k: (
            {"data": v.to(torch.float8_e4m3fn), "weight_scale": torch.tensor(1.25)}
            if decoded
            else {"value": v}
        )
        for k, v in originals.items()
    }
    codec = ScaledScalarDequant("fp8-scaled-scalar/1")
    if decoded:
        assert not mixed
        with torch.no_grad():
            for key, tensor in root.state_dict().items():
                codec.decode(torch, payloads[key], tensor)
        originals = {k: v.detach().clone() for k, v in root.state_dict().items()}
    writer = native.begin_derived(
        "sha256:" + "71" * 32,
        1,
        *Derivation(
            sources={},
            targets={
                "network": Target(
                    add={
                        k: Tensor(
                            dtype(v),
                            tuple(v.shape),
                            SPEC_SCALED_SCALAR_WEIGHT_ONLY if decoded else SPEC_PLAIN,
                            {
                                role: Part(dtype(data), tuple(data.shape))
                                for role, data in payloads[k].items()
                            },
                        )
                        for k, v in originals.items()
                    }
                )
            },
            configs={},
            order=tuple(("network", k) for k in originals),
        ).native_arguments(8192),
        work_fingerprint="sha256:" + "72" * 32,
    )
    for k, parts in payloads.items():
        for role, value in parts.items():
            writer.add_part(
                "network",
                k,
                role,
                io.BytesIO(value.reshape(-1).view(torch.uint8).numpy().tobytes()),
            )
    receipt = writer.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    checkpoint = fill.Checkpoint(tmp_path / "store", manifest)
    rows = checkpoint.rows("network")
    # Only device construction/transport are substituted. Both production load entry
    # points, strict fill(), canonical metadata and native verified reads are actual.
    backend: Any = fill.StreamingFillBackend.__new__(fill.StreamingFillBackend)
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.device_str = "cpu"
    backend.kind = "cpu"
    backend.plan = {row.key: row for row in rows}
    backend.specs = fill.tensor_schema_of(rows)
    backend.selected = {
        key: SimpleNamespace(provider=codec if decoded else SimpleNamespace(route="verbatim"))
        for key in backend.plan
    }
    backend.resolved = {
        key: SimpleNamespace(route="decoded_float" if decoded else "verbatim")
        for key in backend.plan
    }
    backend.expected = {"network": frozenset(backend.plan)}
    backend.components = {"network": root}
    backend.parked = {}
    backend.derived = {"network": fill._derived_values(root)}
    backend.replaced = {}
    backend.leaf_sites = {}
    backend.scratch = {}
    backend.enqueued = {}
    backend.stage_log = []
    backend.stats = {}
    backend.unit_fill = False
    backend.state = "ready"
    backend.census_keys = {"network": frozenset(backend.plan)}
    total = sum(v.numel() * v.element_size() for v in originals.values())
    backend.component_memory = {
        "network": fill.ComponentMemory(destinations=total, decode=8 if decoded else 0)
    }
    backend._mark = lambda *a, **kw: None
    backend._acquire = lambda name: checkpoint.acquire()
    backend.reads = []
    backend.fail = False

    def commit(self: Any) -> None:
        with native.acquire_cozytensors(manifest) as lease:
            header = native.manifest(manifest)["header"]
            assert isinstance(header, bytes)
            for key, value in self.enqueued.items():
                self.reads.append((key, value.dtype))
                if decoded:
                    parts = {}
                    for role, spec in payloads[key.removeprefix("network.")].items():
                        data = torch.empty_like(spec)
                        lease.read_part_into(
                            header,
                            "network",
                            key.removeprefix("network."),
                            role,
                            0,
                            memoryview(data.reshape(-1).view(torch.uint8).numpy()).cast("B"),
                        )
                        parts[role] = data
                    self.selected[key].provider.decode(torch, parts, value)
                else:
                    lease.read_part_into(
                        header,
                        "network",
                        key.removeprefix("network."),
                        "value",
                        0,
                        memoryview(value.detach().view(torch.uint8).numpy()).cast("B"),
                    )
                if self.fail:
                    raise RuntimeError("injected native read failure")
        self.enqueued.clear()

    backend._commit_enqueued = MethodType(commit, backend)
    owner = memory.Planner(torch, torch.device("cpu"))
    owner.cuda = True
    held = owner.adopt("test", "network", root, backend, tiers.DEVICE)
    owner.cuda = False
    held.part.unit_loads = not whole
    return owner, held, backend, originals


def raw(t: Any) -> bytes:
    return bytes(t.detach().view(torch.uint8).numpy().tobytes())


@pytest.mark.parametrize("whole", [False, True])
@pytest.mark.parametrize("via_host", [False, True])
def test_real_strict_refill_replays_retained_bfloat16_repeatedly(
    tmp_path: Path, whole: bool, via_host: bool
) -> None:
    owner, held, backend, _ = fixture(tmp_path, whole=whole, mixed=True)
    root = held.part.root
    root.bfloat16()
    expected = {k: raw(v) for k, v in root.state_dict().items()}
    table = raw(root.table)
    for _ in range(3):
        if via_host:
            for unit in held.part.units:
                tiers.to_host(torch, unit)
        tiers.to_disk(torch, held.part, None)
        assert root.layers[0].weight.dtype == torch.bfloat16  # parked metadata stays authored
        assert root.table.dtype == torch.bfloat16
        before = [(unit, list(unit.pending)) for unit in held.part.units]
        demand = tiers.load_sequence(torch, held.part.units, whole=whole, component=held.part)
        assert demand.resident == sum(unit.nbytes for unit in held.part.units)
        tiers.from_disk(torch, held.part, None)
        assert {k: raw(v) for k, v in root.state_dict().items()} == expected and raw(
            root.table
        ) == table
        assert root.count.dtype == torch.int64
        assert all(unit.applied == pending and not unit.pending for unit, pending in before)
    assert all(dtype in (torch.float16, torch.int64) for _, dtype in backend.reads)


@pytest.mark.parametrize(
    "target,expected_resident,expected_peak", [(torch.bfloat16, 8, 16), (torch.float32, 16, 24)]
)
def test_source_quote_is_canonical_not_parked_dtype(
    tmp_path: Path, target: Any, expected_resident: int, expected_peak: int
) -> None:
    _, held, _, _ = fixture(tmp_path)
    root = held.part.root
    root.to(target)
    tiers.to_disk(torch, held.part, None)
    unit = held.part.blocks[0]
    d = tiers.load_demand(torch, unit, component=held.part)
    assert d.source == 8 and d.resident == expected_resident and d.peak == expected_peak
    slots = list(unit.slots())
    assert slots[0][3].dtype == target
    units = held.part.blocks
    all_ = tiers.load_sequence(torch, units, whole=True, component=held.part)
    assert all_.resident == expected_resident * 2
    assert all_.peak == (24 if target == torch.bfloat16 else 40)
    assert all(t.is_meta for *_, t in slots)


def test_rounding_chain_replayed_in_order_not_collapsed(tmp_path: Path) -> None:
    _, held, _, original = fixture(tmp_path)
    root = held.part.root
    root.bfloat16().half()
    expected = {k: raw(v) for k, v in root.state_dict().items()}
    assert any(expected[k] != raw(v) for k, v in original.items())
    for _ in range(2):
        tiers.to_disk(torch, held.part, None)
        assert all(len(unit.pending) == 2 for unit in held.part.units)
        tiers.from_disk(torch, held.part, None)
        assert {k: raw(v) for k, v in root.state_dict().items()} == expected
        assert all(len(unit.applied) == 2 and not unit.pending for unit in held.part.units)


@pytest.mark.parametrize("whole", [False, True])
def test_failed_fill_restores_exact_parked_metadata_objects_and_queue(
    tmp_path: Path, whole: bool
) -> None:
    _, held, backend, _ = fixture(tmp_path, whole=whole)
    root = held.part.root
    root.float()
    tiers.to_disk(torch, held.part, None)
    original = [
        (module, key, parameter, tensor)
        for unit in held.part.units
        for module, key, parameter, tensor in unit.slots()
    ]
    queues = [(unit, list(unit.pending), unit.nbytes) for unit in held.part.units]
    backend.fail = True
    with pytest.raises(RuntimeError, match="native read failure"):
        tiers.from_disk(torch, held.part, None)
    for module, key, parameter, tensor in original:
        assert (module._parameters if parameter else module._buffers)[key] is tensor
    assert all(
        unit.pending == pending and not unit.applied and unit.nbytes == nbytes
        for unit, pending, nbytes in queues
    )
    backend.fail = False
    tiers.from_disk(torch, held.part, None)
    assert all(t.dtype == torch.float32 for t in root.parameters())


def test_initial_mismatch_and_unknown_route_remain_strict(tmp_path: Path) -> None:
    _, held, backend, _ = fixture(tmp_path)
    key = "network.layers.0.weight"
    with pytest.raises(fill.FillRefusal):
        backend.fill(key, backend.specs[key], held.part.root.layers[0].weight.bfloat16())
    held.part.root.bfloat16()
    tiers.to_disk(torch, held.part, None)
    unit = held.part.blocks[0]
    del backend.resolved[next(iter(unit.live("network")))]
    with pytest.raises(fill.FillRefusal, match="route missing"):
        tiers.load_demand(torch, unit, component=held.part)
    assert not backend.reads and all(t.is_meta for *_, t in unit.slots())


def test_real_decoded_float_whole_refill_and_transient_quote(tmp_path: Path) -> None:
    owner, held, backend, _ = fixture(tmp_path, whole=True, decoded=True)
    root = held.part.root
    root.float()
    expected = {k: raw(v) for k, v in root.state_dict().items()}
    tiers.to_disk(torch, held.part, None)
    d = tiers.load_sequence(torch, held.part.units, whole=True, component=held.part)
    assert d.resident == 48 and d.peak == 56  # canonical24; final48 plus8 overlap
    assert backend.transient("network") == 8
    assert owner._need(held.part, len(held.part.blocks)) == 64
    tiers.from_disk(torch, held.part, None)
    assert {k: raw(v) for k, v in root.state_dict().items()} == expected


def test_encoded_without_conversion_quote_unchanged_and_pending_refuses(tmp_path: Path) -> None:
    owner, held, backend, _ = fixture(tmp_path, whole=True)
    tiers.to_disk(torch, held.part, None)
    # Stand-in encoded settled footprint and truthful provider expansion overhead.
    for unit in held.part.units:
        unit.nbytes = 4
    backend.component_memory["network"] = fill.ComponentMemory(
        destinations=24, native=12, native_destinations=24, native_resident=12
    )
    for resolved in backend.resolved.values():
        resolved.route = "encoded_gemm"
    assert backend.transient("network") == 24
    assert (
        owner._need(held.part, len(held.part.blocks)) == 36
    )  # original12+24; no new canonical doublecharge
    key = next(iter(held.part.common.live("network")))
    backend.resolved[key].route = "encoded_gemm"
    held.part.common.pending = [tiers.DTypeConvert(lambda t: t.float())]
    with pytest.raises(RuntimeFailure, match="encoded leaves"):
        tiers.load_demand(torch, held.part.common, component=held.part)
    assert not backend.reads


@pytest.mark.parametrize("whole", [False, True])
def test_postfill_conversion_failure_restores_earlier_converted_unit(
    tmp_path: Path, whole: bool
) -> None:
    _, held, backend, _ = fixture(tmp_path, whole=whole, mixed=True)
    root = held.part.root
    root.float()
    expected = {k: raw(v) for k, v in root.state_dict().items()}
    tiers.to_disk(torch, held.part, None)
    original = [
        (module, key, parameter, tensor)
        for unit in held.part.units
        for module, key, parameter, tensor in unit.slots()
    ]
    second = held.part.blocks[0]
    convert = second.pending[0]
    error = ValueError("conversion after native fill")

    def failing(tensor: Any) -> Any:
        result = convert(tensor)
        if not tensor.is_meta:
            assert held.part.common.tier == tiers.DEVICE
            assert backend.reads
            raise error
        return result

    second.pending[0] = tiers.DTypeConvert(failing)
    queues = [(unit, list(unit.pending), unit.nbytes) for unit in held.part.units]
    with pytest.raises(ValueError) as caught:
        tiers.from_disk(torch, held.part, None)
    assert caught.value is error
    for module, key, parameter, tensor in original:
        assert (module._parameters if parameter else module._buffers)[key] is tensor
    assert all(
        unit.tier == tiers.DISK
        and unit.pending == pending
        and not unit.applied
        and unit.nbytes == nbytes
        for unit, pending, nbytes in queues
    )
    second.pending[0] = convert
    tiers.from_disk(torch, held.part, None)
    assert {k: raw(v) for k, v in root.state_dict().items()} == expected


def test_native_read_failure_after_an_earlier_unit_commits_rolls_back(tmp_path: Path) -> None:
    _, held, backend, _ = fixture(tmp_path)
    held.part.root.bfloat16()
    tiers.to_disk(torch, held.part, None)
    original = [tensor for unit in held.part.units for *_, tensor in unit.slots()]
    commit = backend._commit_enqueued

    def later() -> None:
        if backend.reads:
            assert held.part.common.tier == tiers.DEVICE
            backend.fail = True
        commit()

    backend._commit_enqueued = later
    with pytest.raises(RuntimeError, match="native read failure"):
        tiers.from_disk(torch, held.part, None)
    assert all(
        a is b
        for a, b in zip(
            original, [t for unit in held.part.units for *_, t in unit.slots()], strict=True
        )
    )
    assert all(
        unit.tier == tiers.DISK and unit.pending and not unit.applied for unit in held.part.units
    )


def test_regression_disabling_canonicalization_reproduces_f16_bf16_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, held, backend, _ = fixture(tmp_path)
    held.part.root.bfloat16()
    tiers.to_disk(torch, held.part, None)
    canonicalize = backend._canonicalize_refill
    monkeypatch.setattr(backend, "_canonicalize_refill", lambda *_: None)
    with pytest.raises(fill.FillRefusal, match="LOGICAL dtype"):
        tiers.from_disk(torch, held.part, held.part.common)
    monkeypatch.setattr(backend, "_canonicalize_refill", canonicalize)
    tiers.from_disk(torch, held.part, held.part.common)
    assert held.part.root.layers[0].weight.dtype == torch.bfloat16


def test_f16_to_f32_source_16_resident_32_peak_48(tmp_path: Path) -> None:
    _, held, _, _ = fixture(tmp_path, width=4)
    held.part.root.float()
    tiers.to_disk(torch, held.part, None)
    unit = held.part.blocks[0]
    demand = tiers.load_demand(torch, unit, component=held.part)
    assert (demand.source, demand.resident, demand.peak) == (16, 32, 48)
    assert unit.nbytes == 32 and next(unit.slots())[3].dtype == torch.float32


@pytest.mark.parametrize("chain", [False, True])
def test_derived_restoration_peak_dominates_noop_or_roundtrip_cast(
    tmp_path: Path, chain: bool
) -> None:
    _, held, backend, _ = fixture(tmp_path, mixed=True, derived_count=25)
    if chain:
        held.part.root.bfloat16().float()
    tiers.to_disk(torch, held.part, None)
    demand = tiers.load_demand(torch, held.part.common, component=held.part)
    assert demand.source == 116 and demand.restoration_peak_extra == 100
    assert (
        demand.peak == 216 and demand.replay_peak is not None and demand.replay_peak < demand.peak
    )
    assert demand.resident == (124 if chain else 116)
    expected_dtype = torch.float32
    tiers.from_disk(torch, held.part, None)
    assert held.part.root.table.dtype == expected_dtype
    assert backend.reads


def test_whole_derived_restore_is_before_earlier_unit_widening(tmp_path: Path) -> None:
    _, held, _, _ = fixture(tmp_path, whole=True, mixed=True, derived_count=25, derived_last=True)
    held.part.root.float()
    tiers.to_disk(torch, held.part, None)
    demand = tiers.load_sequence(torch, held.part.units, whole=True, component=held.part)
    # int64root8 +threeF16weights24 +lastderived100 =132 beforecasts.
    # All derived restoration finishes before any unit's conversion chain.
    assert demand.resident == 156 and demand.peak == 232
    sequential = tiers.load_sequence(torch, held.part.units, component=held.part)
    assert sequential.peak == 248  # sequentially, earlier units really have widened
    tiers.from_disk(torch, held.part, None)
    assert held.part.root.layers[-1].table.dtype == torch.float32


def test_derived_copy_overlap_is_largest_single_buffer_not_sum(tmp_path: Path) -> None:
    _, held, backend, _ = fixture(tmp_path, mixed=True, derived_count=25, derived_second_count=3)
    tiers.to_disk(torch, held.part, None)
    demand = tiers.load_demand(torch, held.part.common, component=held.part)
    assert (demand.source, demand.resident, demand.restoration_peak_extra, demand.peak) == (
        128,
        128,
        100,
        228,
    )
    assert demand.replay_peak == 128
    assert not backend.reads
    tiers.from_disk(torch, held.part, None)
    assert torch.equal(held.part.root.other_table, torch.zeros(3))


def test_encoded_derived_excess_is_not_provider_transient_doublecharge(tmp_path: Path) -> None:
    owner, held, backend, _ = fixture(tmp_path, whole=True, mixed=True, derived_count=25)
    tiers.to_disk(torch, held.part, None)
    held.part.common.nbytes = 112  # encoded4 +int8 +derived100
    for unit in held.part.blocks:
        unit.nbytes = 4
    for resolved in backend.resolved.values():
        resolved.route = "encoded_gemm"
    backend.component_memory["network"] = fill.ComponentMemory(
        destinations=132, native=12, native_destinations=24, native_resident=12
    )
    assert backend.transient("network") == 24
    schema = backend.load_schema("network", held.part.common)
    assert schema.dtypes is None and schema.restoration_peak_extra == 88
    assert (
        owner._need(held.part, len(held.part.blocks)) == 232
    )  # max(provider144, canonical132+derived100)


def test_shape_refusal_before_allocation_restores_parked_objects(tmp_path: Path) -> None:
    _, held, backend, _ = fixture(tmp_path)
    held.part.root.bfloat16()
    tiers.to_disk(torch, held.part, None)
    unit = held.part.blocks[0]
    original = torch.nn.Parameter(
        torch.empty((1, 4), dtype=torch.bfloat16, device="meta"), requires_grad=False
    )
    unit.module.weight = original
    pending = list(unit.pending)
    with pytest.raises(fill.FillRefusal, match="shape/layout"):
        tiers.from_disk(torch, held.part, unit)
    assert unit.module.weight is original and unit.pending == pending and unit.tier == tiers.DISK
    assert not backend.reads
