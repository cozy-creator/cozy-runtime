"""Changing consumer observations cannot choose another kernel implementation policy."""

from __future__ import annotations

import pytest

from cozy_runtime.internal import probe
from cozy_runtime.internal.encoding import Encoded, RolePart, launch_providers, params_of
from cozy_runtime.internal.fill import PlanRow, dtype_name
from cozy_runtime.internal.planfacts import PlanFacts, RunFact
from cozy_runtime.internal.resolution import ConstructionFacts, Variant, resolve
from test_leaf_speedup_record import sm86


@pytest.mark.parametrize("objective", ["latency", "footprint", "kernel_numeric_error"])
def test_local_timing_and_memory_history_never_choose_kernel_policy(objective: str) -> None:
    # The recorded real offline table provides capabilities; test history below is deliberately
    # contradictory input data, not a performance measurement or a mocked device execution.
    table = sm86()
    shape = (64, 256)
    parts = (RolePart("weight", "i8", shape, 64 * 256), RolePart("scale", "f32", (64,), 64 * 4))
    row = PlanRow(
        "model.weight",
        "weight",
        "model",
        "bf16",
        shape,
        64 * 256 * 2,
        Encoded(
            "int8.convrot", params_of({"rotation": "kron_regular_h4", "rotation_size": 256}), parts
        ),
    )
    facts = PlanFacts()

    def choose(consent: str):
        return resolve(
            variants=[Variant("lane", "store", "sha256:" + "1" * 64, {}, True)],
            rows_for={"lane": [row]},
            providers=launch_providers(),
            capabilities=probe.admission_table(table),
            device=table.device,
            runtime=table.runtime,
            construction=ConstructionFacts("test", "test:Model", ("model",)),
            facts=facts,
            encoded_leaves=consent,
            dtype_name=dtype_name,
            objective=objective,
        )

    original = choose("accept")
    floor = choose("refuse")
    assert floor.tensors[0].route == "decoded_float"
    if objective == "latency":
        assert original.tensors[0].route == "encoded_gemm"
    for sign in (1, -1):
        for policy in (original, floor):
            fast = (policy.policy == "decode_floor") == (sign == 1)
            facts.bank(
                RunFact(
                    policy.digest(),
                    f"contradictory-{sign}-{policy.policy}",
                    fill_ms=0 if fast else 1_000_000,
                    step_ms=0 if fast else 1_000_000,
                    resident_bytes=1 if fast else 1 << 40,
                    scratch_bytes=0 if fast else 1 << 40,
                )
            )
        selected = choose("accept")
        assert selected.policy == original.policy and selected.digest() == original.digest()
        assert selected.measurements is not None  # observations are retained, not a chooser
        assert choose("refuse").tensors[0].route == "decoded_float"
