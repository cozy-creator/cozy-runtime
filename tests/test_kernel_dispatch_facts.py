"""A real offline CPU numerical verdict is consumed without any consumer probe or write."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import int8_linear, verify  # noqa: E402


def test_offline_verdict_round_trip_and_missing_identity_refuses(tmp_path: Path) -> None:
    assert not torch.cuda.is_initialized()
    artifact = "sha256:" + hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest()
    calls: list[str] = []

    def numeric_probe() -> verify.Verdict:
        calls.append("cpu-mm")
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
        weight = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 48
        return verify.within(
            {
                "float32 CPU matmul": verify.error(
                    torch, x @ weight.T, x.double() @ weight.double().T
                )
            },
            1e-6,
        )

    def forbidden() -> Any:
        raise AssertionError("consumer invoked the development oracle")

    # A missing release table cannot start probing or create a machine-local verdict file.
    missing = tmp_path / "missing"
    assert not verify.admitted("cpu-mm", artifact, 0, forbidden, tables=missing).ok
    assert not missing.exists()
    with verify.measuring() as records:
        measured = verify.admitted("cpu-mm", artifact, 0, numeric_probe)
    assert measured.ok and calls == ["cpu-mm"]
    # The successful development cache is insufficient: only explicitly published facts count.
    assert not verify.admitted("cpu-mm", artifact, 0, forbidden, tables=missing).ok
    paths = verify.publish(tmp_path, records)
    assert len(paths) == 1
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert verify.admitted("cpu-mm", artifact, 0, forbidden, tables=tmp_path) == measured
    assert not verify.admitted("cpu-mm", artifact + "-different", 0, forbidden, tables=tmp_path).ok
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before
    # Altering the record's identity never transfers the measured fact to another runtime.
    document = json.loads(paths[0].read_bytes())
    document["runtime"]["torch_build"] = "different"
    paths[0].write_text(json.dumps(document))
    assert not verify.admitted("cpu-mm", artifact, 0, forbidden, tables=tmp_path).ok
    assert calls == ["cpu-mm"]


def test_only_offline_tooling_can_call_the_machine_local_probe_cache() -> None:
    root = Path(verify.__file__).parents[1]
    callers = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "verify"
                and node.func.attr == "verified"
            ):
                callers.append(str(path.relative_to(root)))
    assert not callers, f"consumer bypassed shipped kernel facts: {callers}"


def test_unqualified_optional_kitchen_never_loads_during_consumer_dispatch() -> None:
    before = dict(int8_linear._KITCHEN)
    assert not verify.in_oracle()
    assert int8_linear.kitchen(torch, 0) is None
    assert before == int8_linear._KITCHEN and not torch.cuda.is_initialized()
