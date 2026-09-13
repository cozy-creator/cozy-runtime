"""The placement cross preserves each model's routes and hashes only decisions.

Native prepare/planner/invoke is exercised by test_parallel_calls. These cases
isolate decode-cost and identity rules without making a GPU execution claim.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    PlanChooser,
    PlanRefusal,
    PreparedModel,
    PreparedRequest,
)


def choose(models: dict[str, dict[str, Any]], *, staged: bool = False) -> Any:
    ledger = Ledger()
    ledger.observe_construction(
        {
            "filled_bytes": 32,
            "filled": 2,
            "allocator_bytes": 32,
            "reserved_bytes": 32,
            "resident": {"base/dit": 24, "lora/factors": 8},
            "device_free_bytes": 4 << 30,
            "device_total_bytes": 8 << 30,
            "parked": ["base/dit"] if staged else [],
            "declared_scopes": {"sample": ["base/dit", "lora/factors"]},
        }
    )
    binding = DeclaredBinding(
        entrypoint_binding_digest="sha256:binding",
        entrypoint="generate",
        model_class="Models",
        model_binding_path="generate.models.base",
        model_parameter_name="base",
        release="fixture",
        model_weight_bytes=32,
    )
    return PlanChooser(ledger).choose(
        binding,
        PreparedModel(
            constructed_model_digest="sha256:constructed",
            model_deliveries=tuple(sorted(models.items())),
        ),
        PreparedRequest.unresolved("generate", {}),
    )


def deliveries(*, decode: bool = False) -> dict[str, dict[str, Any]]:
    return {
        "base": {
            "rung": "aot_decode" if decode else "encoded_gemm",
            "variant": "fp8",
            "plan_digest": "sha256:base",
            "routes": {"encoded_gemm": 3, **({"decoded_float": 2} if decode else {})},
            "kernels": {"encoded_gemm": 3, **({"library_float": 2} if decode else {})},
            "calibrated": False,
            "confession": "not measured",
            "leaf_speedup_x": 0.0,
        },
        "lora": {
            "rung": "verbatim",
            "variant": "plain",
            "plan_digest": "sha256:lora",
            "routes": {"verbatim": 4},
            "kernels": {"library_float": 4},
        },
    }


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("staged", [False, True])
def test_each_model_keeps_its_kernel_and_decode_cost(decode: bool, staged: bool) -> None:
    source = deliveries(decode=decode)
    plan = choose(source, staged=staged)
    rows = plan.document()["model_deliveries"]
    assert plan.delivery == ("float" if decode else "native")
    assert plan.materialization == (("staged_decode" if staged else "aot_decode") if decode else "")
    assert rows["base"]["kernels"] == source["base"]["kernels"]
    assert rows["base"]["route_costs"]["encoded_gemm"] == "none"
    assert rows["lora"]["route_costs"] == {"verbatim": "none"}
    if decode:
        assert rows["base"]["route_costs"]["decoded_float"] == (
            "per_transition" if staged else "per_generation"
        )


def test_observational_fields_do_not_change_attempt_identity() -> None:
    source = deliveries()
    before = choose(source).digest()
    reported = copy.deepcopy(source)
    reported["base"].update(
        calibrated=True, confession="measured", leaf_speedup_x=1.25, objective="latency"
    )
    assert choose(reported).digest() == before
    reported["lora"]["plan_digest"] = "sha256:other-prepared-model"
    assert choose(reported).digest() != before


def test_unknown_model_route_is_a_typed_admission_refusal() -> None:
    source = deliveries()
    source["base"]["routes"]["invented"] = 1
    with pytest.raises(PlanRefusal, match="model 'base'") as refused:
        choose(source)
    assert refused.value.code == "unknown_route"
