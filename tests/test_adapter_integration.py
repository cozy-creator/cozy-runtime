"""Real TensorFS snapshots and CPU executor scopes, without a model-package LoRA API."""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs

from cozy_runtime.internal import adapters
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.executor_adapters import AdapterAttempt, AdapterModel
from cozy_runtime.internal.lora import AdapterRefusal
from cozy_runtime.internal.worker.package_prepare import _slot_adapters
from cozy_runtime.internal.worker.plan import ModelBinding

torch = pytest.importorskip("torch")


def _snapshot(store: Any, component: str, tensors: dict[str, Any]) -> str:
    encoded = {name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()}
    identity = hashlib.sha256(
        component.encode() + b"".join(value.tobytes() for value in encoded.values())
    ).hexdigest()
    targets = {
        component: {
            "drop": [],
            "add": {
                name: {
                    "logical_dtype": "f32",
                    "shape": list(value.shape),
                    "encoding": SPEC_PLAIN,
                    "parts": {"value": {"dtype": "f32", "shape": list(value.shape)}},
                }
                for name, value in encoded.items()
            },
        }
    }
    writer = store.begin_derived(
        "sha256:" + identity,
        1,
        {},
        targets,
        {},
        [(component, name) for name in encoded],
        sum(value.nbytes for value in encoded.values()),
        work_fingerprint="sha256:" + identity,
    )
    for name, value in encoded.items():
        writer.add_part(component, name, "value", io.BytesIO(value.tobytes()))
    return "sha256:" + str(writer.commit()["manifest"]["sha256"])


def _fixture(tmp_path: Path) -> tuple[Path, str, str, Any, Any, Any]:
    root = tmp_path / "store"
    store = tensorfs.Store.init(root)
    weights = np.arange(12, dtype=np.float32).reshape(3, 4) / 20
    a = np.arange(8, dtype=np.float32).reshape(2, 4) / 10
    b = np.arange(6, dtype=np.float32).reshape(3, 2) / 10
    base = _snapshot(store, "dit", {"0.weight": weights})
    adapter = _snapshot(
        store,
        "adapter",
        {
            "0.lora_A.weight": a,
            "0.lora_B.weight": b,
            "0.alpha": np.asarray(3.0),
        },
    )
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    model[0].weight.data.copy_(torch.from_numpy(weights))
    return root, base, adapter, model, torch.from_numpy(a), torch.from_numpy(b)


def test_snapshot_fit_stage_apply_and_same_base_independent_slots(tmp_path: Path) -> None:
    root, base, ref, first, a, b = _fixture(tmp_path)
    second = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    second.load_state_dict(first.state_dict())
    one = adapters.AdapterSelection("dit", ref, "adapter", "0.5")
    two = replace(one, scale="-0.25")
    geometry = adapters.inspect(root, [one], {"dit": base})
    assert geometry.device_bytes == 56 and geometry.source_bytes == 60
    assert geometry.targets == {"dit": ("0",)}
    controllers = {
        name: AdapterModel({"dit": model}, {"dit": base}, geometry.compatibility, str(root))
        for name, model in (("first", first), ("second", second))
    }
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20
    baseline = first(x)
    delta = torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
    capacity: list[int] = []
    selected = {"first": [one.document()], "second": [two.document()]}
    for _ in range(2):
        attempt = AdapterAttempt(
            controllers,
            selected,
            device="cpu",
            check_cancelled=lambda: None,
            check_capacity=capacity.append,
        )
        try:
            torch.testing.assert_close(first(x), baseline + delta * 0.75, rtol=5e-6, atol=5e-7)
            torch.testing.assert_close(second(x), baseline + delta * -0.375, rtol=5e-6, atol=5e-7)
            assert attempt.refs["first"][0].scale == 0.5
            assert attempt.refs["second"][0].scale == -0.25
            assert attempt.device_bytes == 112
        finally:
            attempt.close()
        assert torch.equal(first(x), baseline) and torch.equal(second(x), baseline)
    assert 112 + geometry.scratch_bytes in capacity  # factors + static scratch at admission
    assert geometry.scratch_bytes in capacity  # same bound enforced at every actual target


def test_coalesced_model_slots_refuse_even_when_only_one_is_adapted(tmp_path: Path) -> None:
    root, base, ref, model, _, _ = _fixture(tmp_path)
    selection = adapters.AdapterSelection("dit", ref, "adapter", "1")
    geometry = adapters.inspect(root, [selection], {"dit": base})
    controller = AdapterModel({"dit": model}, {"dit": base}, geometry.compatibility, str(root))
    with pytest.raises(AdapterRefusal) as error:
        AdapterAttempt(
            {"adapted": controller, "plain": controller},
            {"adapted": [selection.document()]},
            device="cpu",
            check_cancelled=lambda: None,
            check_capacity=lambda _: None,
        )
    assert error.value.code == "adapter_slot_alias"


def test_binding_identity_reuses_targets_but_not_wrong_base(tmp_path: Path) -> None:
    root, base, ref, model, _, _ = _fixture(tmp_path)
    selection = adapters.AdapterSelection("dit", ref, "adapter", "1")
    geometry = adapters.inspect(root, [selection], {"dit": base})
    binding = ModelBinding(
        "Model",
        "generate.models.model",
        "model",
        str(root),
        "",
        base,
        ("dit",),
        {"dit": base},
        adapters=(selection,),
        adapter_compatibility=geometry.compatibility,
    )
    another = replace(binding, adapters=(replace(selection, scale="0.5"),))
    assert binding.key_document() == another.key_document()
    assert binding.key_document() != replace(binding, reference_snapshot="other").key_document()
    controller = AdapterModel({"dit": model}, {"dit": base}, geometry.compatibility, str(root))
    with pytest.raises(AdapterRefusal) as error:
        controller.inspect([replace(selection, component="ref2va_dit").document()])
    assert error.value.code == "adapter_component"


def test_preparation_keeps_nested_order_and_separate_adapter_models(tmp_path: Path) -> None:
    root, base, ref, _, _, _ = _fixture(tmp_path)
    source = {
        "model": "owner/base",
        "manifest": base,
        "adapters": [
            {
                "component": "dit",
                "model": "owner/lora",
                "manifest": ref,
                "source_component": "adapter",
                "scale": scale,
            }
            for scale in ("0.5", "0", "-0.25")
        ],
    }
    models, slots = _slot_adapters(source, ["dit"], root, 1)
    assert [row.scale for row in slots] == ["0.5", "0", "-0.25"]
    assert [row.model_id for row in slots] == [row.id for row in models]
    assert all(row.manifest.digest.hex() == ref[7:] for row in models)


@pytest.mark.parametrize(
    "spelling,value",
    [("1", 1.0), ("0", -0.0), ("0.5", 0.5), ("-0.25", -0.25), ("1e+06", 1e6), ("1e-05", 1e-5)],
)
def test_scale_matches_creator_shortest_decimal(spelling: str, value: float) -> None:
    assert adapters.scale_string(value) == spelling
    assert adapters.parse_scale(spelling) == value


@pytest.mark.parametrize(
    "spelling", ["", "-0", "0.0", "+.5", "NaN", "Inf", "1e309", "0x1p0", "1000000", "1e6"]
)
def test_scale_refuses_noncanonical_or_nonfinite_wire_values(spelling: str) -> None:
    with pytest.raises(AdapterRefusal):
        adapters.parse_scale(spelling)


def test_placement_reserves_factors_and_static_tile_scratch(tmp_path: Path) -> None:
    import os

    from cozy_runtime.internal.worker.ledger import Ledger
    from cozy_runtime.internal.worker.plan import (
        DeclaredBinding,
        PlanChooser,
        PreparedModel,
        PreparedRequest,
    )

    root, base, ref, _, _, _ = _fixture(tmp_path)
    selection = adapters.AdapterSelection("dit", ref, "adapter", "1")
    geometry = adapters.inspect(root, [selection, selection], {"dit": base})
    member = ModelBinding(
        "Model",
        "serve.models.model",
        "model",
        str(root),
        "",
        base,
        ("dit",),
        {"dit": base},
        model_weight_bytes=4096,
        adapters=(selection, selection),
        adapter_compatibility=geometry.compatibility,
        adapter_device_bytes=geometry.device_bytes,
        adapter_scratch_bytes=geometry.scratch_bytes,
    )
    declared = DeclaredBinding(
        "sha256:" + "0" * 64,
        "serve",
        "Model",
        "serve.models.model",
        "model",
        "release",
        models=(member,),
        model_weight_bytes=4096,
    )
    ledger = Ledger(worker_pid=os.getpid())
    ledger.begin_generation(1)
    ledger.observe_construction(
        {
            "filled_bytes": 4096,
            "filled": 1,
            "allocator_bytes": 4096,
            "reserved_bytes": 4096,
            "resident": {"dit": 4096},
            "device_free_bytes": 2 << 30,
            "device_total_bytes": 3 << 30,
            "declared_scopes": {"forward": ["dit"]},
        }
    )
    chooser = PlanChooser(ledger)
    request = PreparedRequest.unresolved("serve", {})
    plain = replace(
        declared,
        models=(
            replace(
                member,
                adapters=(),
                adapter_compatibility=(),
                adapter_device_bytes=0,
                adapter_scratch_bytes=0,
            ),
        ),
    )
    chooser.choose(plain, PreparedModel(delivery_rung="verbatim"), request)
    without = chooser.walks[-1]
    adapted = chooser.choose(declared, PreparedModel(delivery_rung="verbatim"), request)
    with_adapters = chooser.walks[-1]
    assert (
        with_adapters["rungs"][0]["needs_bytes"] - without["rungs"][0]["needs_bytes"]
        == geometry.device_bytes + geometry.scratch_bytes
    )
    assert adapted.headroom_bytes == geometry.scratch_bytes
    assert dict(adapted.scope_headroom_bytes)["forward"] == geometry.scratch_bytes
    assert declared.group_weight_bytes() == 4096 + geometry.device_bytes + geometry.scratch_bytes


def test_zero_strength_source_is_charged_before_any_host_allocation(tmp_path: Path) -> None:
    root, base, ref, model, _, _ = _fixture(tmp_path)
    selection = adapters.AdapterSelection("dit", ref, "adapter", "0")
    geometry = adapters.inspect(root, [selection], {"dit": base})
    assert geometry.device_bytes == geometry.scratch_bytes == 0
    controller = AdapterModel({"dit": model}, {"dit": base}, geometry.compatibility, str(root))
    with pytest.raises(AdapterRefusal) as error:
        AdapterAttempt(
            {"model": controller},
            {"model": [selection.document()]},
            device="cpu",
            check_cancelled=lambda: None,
            check_capacity=lambda _: None,
            host_available=lambda: 0,
        )
    assert error.value.code == "adapter_host_shortfall"
    assert controller.linear._active.get() is None


def test_group_host_preflight_prices_every_rank_before_any_copy(tmp_path: Path) -> None:
    from cozy_runtime.internal.executor_adapters import check_host_capacity

    root, base, ref, _, _, _ = _fixture(tmp_path)
    selection = adapters.AdapterSelection("dit", ref, "adapter", "0")
    geometry = replace(
        adapters.inspect(root, [selection], {"dit": base}),
        source_bytes=8 << 30,
        host_scratch_bytes=0,
    )
    assert (
        check_host_capacity([geometry], device="cuda", copies=2, available=lambda: 20 << 30)
        == 16 << 30
    )
    with pytest.raises(AdapterRefusal) as error:
        check_host_capacity([geometry], device="cuda", copies=4, available=lambda: 20 << 30)
    assert error.value.code == "adapter_host_shortfall"


def test_prepare_frame_carries_refs_and_digest_instead_of_target_table(tmp_path: Path) -> None:
    from cozy_runtime.internal import canonical
    from cozy_runtime.internal.seam import MAX_FRAME
    from cozy_runtime.internal.worker.plan import DeclaredBinding
    from cozy_runtime.internal.worker.session import executor_prepare_command

    root, base, ref, _, _, _ = _fixture(tmp_path)
    selection = adapters.AdapterSelection("dit", ref, "adapter", "1")
    geometry = adapters.inspect(root, [selection], {"dit": base})
    targets = tuple(
        {**geometry.compatibility[0], "target": f"blocks.{index}.projection"}
        for index in range(1000)
    )
    member = ModelBinding(
        "Model",
        "serve.models.model",
        "model",
        str(root),
        "",
        base,
        ("dit",),
        {"dit": base},
        adapters=(selection,),
        adapter_compatibility=targets,
    )
    binding = DeclaredBinding(
        "sha256:" + "a" * 64,
        "serve",
        "Model",
        "serve.models.model",
        "model",
        "release",
        models=(member,),
    )
    command = executor_prepare_command(binding, devices="0", authorized_device_limit_bytes=1 << 30)
    assert len(canonical.write(command)) < MAX_FRAME
    assert len(canonical.write(list(targets))) > MAX_FRAME
    selected = command["binding"]
    assert "adapter_compatibility" not in selected
    assert selected["adapter_compatibility_digest"] == canonical.digest(list(targets))
    assert selected["adapter_selections"] == [selection.document()]
