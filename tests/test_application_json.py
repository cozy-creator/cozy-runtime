from __future__ import annotations

import math
from pathlib import Path

import msgspec
import pytest

from cozy_runtime import _application_json as application_json
from cozy_runtime import canonical_json
from cozy_runtime.author import App, Invocation, attempt
from cozy_runtime.internal.executor import _serialize_result


class SpoolRequest(msgspec.Struct):
    pass


class SpoolValue(msgspec.Struct):
    seed: int
    number: int | float


def test_application_values_preserve_unsigned_seeds_number_kind_and_array_order() -> None:
    value = application_json.decode(
        b'{ "seed": 18446744073709551615, "number": 1e0,'
        b' "ordered": [9007199254740993, 9007199254740992] }'
    )
    assert value == {
        "seed": (1 << 64) - 1,
        "number": 1.0,
        "ordered": [9007199254740993, 9007199254740992],
    }
    assert type(value["seed"]) is int and type(value["number"]) is float
    encoded = application_json.encode(value)
    ordered = b"[9007199254740993,9007199254740992]"
    assert encoded == b'{"number":1.0,"ordered":' + ordered + b',"seed":18446744073709551615}'
    for changed in (
        {**value, "seed": value["seed"] - 1},
        {**value, "number": 1},
        {**value, "ordered": list(reversed(value["ordered"]))},
    ):
        assert application_json.encode(changed) != encoded
    with pytest.raises(ValueError, match="number_range"):
        canonical_json.encode(value)
    assert canonical_json.encode({"number": 1.0}) == b'{"number":1}'


def test_application_values_keep_strict_structural_and_finite_number_bounds() -> None:
    for bad in (b'{"a":1,"a":2}', b"{} {}", b'{"a":1e999}', b"[" * 34 + b"0" + b"]" * 34):
        with pytest.raises(ValueError):
            application_json.decode(bad)


def test_typed_job_result_spool_preserves_exact_seed_and_float_kind(tmp_path: Path) -> None:
    app = App()

    @app.job(accelerator=False)
    def echo(payload: SpoolRequest) -> SpoolValue:
        return SpoolValue((1 << 64) - 1, 1.0)

    spool = tmp_path / "spool"
    result, outcome, _ = attempt(app.get("echo"), {}, Invocation("seed-proof", spool, math.inf))
    assert outcome.terminal == "succeeded", outcome
    reference, outputs, oversize = _serialize_result(result, spool)
    assert reference is not None and not outputs and oversize == 0
    decoded = application_json.decode((spool / "result.canonical").read_bytes())
    assert decoded["seed"] == (1 << 64) - 1 and type(decoded["number"]) is float
