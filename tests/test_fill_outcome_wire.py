"""CPU transport boundary: real measurement conversion and DTO, without Torch/CUDA imports."""

from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from cozy_runtime.internal import request_memory
from cozy_runtime.internal.executor_replies import Memory


def test_fill_outcome_survives_measurement_and_the_typed_reply() -> None:
    fill = dict(phase="allocate", outcome="refused", error="DEVICE_FULL", io_runs=0)
    event = dict(sequence=4, request="r", step=1, page=65, gpu=0, rank=0,
                 kind="fill-refused", reason="demand", bytes=16, capacity=64,
                 resident=32, orphan_bytes=0, **{"class": "working"},
                 fill=fill)
    # Structural stand-in for the native RequestStats record: only the actual
    # converter/serialization is under test, not simulated allocation behavior.
    facts: Any = SimpleNamespace(steps=(), leftover=(), copied=0, retries=0, evictions=(),
                                 peaks=(), plan=(), copies=(), trace=(event,), proofs=())
    measured = request_memory.measured(facts, [3], rank=2)
    assert measured is not None
    decoded = msgspec.json.decode(msgspec.json.encode(measured), type=Memory)
    wire = msgspec.to_builtins(decoded.trace[0])
    assert wire["gpu"] == 3 and wire["rank"] == 2
    assert wire["fill"] == event["fill"]
    legacy = {key: value for key, value in event.items() if key != "fill"}
    facts.trace = (legacy,)
    legacy_measured = request_memory.measured(facts, [3])
    assert legacy_measured is not None and legacy_measured.trace[0].fill is None
    facts.trace = ({**event, "fill": {**fill, "io_runs": "zero"}},)
    with pytest.raises(msgspec.ValidationError, match="int"):
        request_memory.measured(facts, [3])
