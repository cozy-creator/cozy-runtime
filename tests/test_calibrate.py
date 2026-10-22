"""The oracle's verdict, quantizer law and workload-rule derivation; CPU only."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.internal import calibrate, prebuilt, verify
from cozy_runtime.internal.calibrate import DEFAULT, Candidate, Measured, Quantizer
from cozy_runtime.internal.encoding.formats import hadamard256

BUDGET = {"bfloat16": 2.2e-3, "float16": 2.8e-4}
SMS = 142  # an L40S: its rules hold for 128 < SMs <= 256


def timed(choice: int | str, median: float, spread: float = 0.01) -> Candidate:
    quartile = median * spread / 2
    return Candidate(
        choice,
        True,
        {"bfloat16": 1.1e-3, "float16": 1.4e-4},
        BUDGET,
        warm_us=(median - quartile, median - quartile, median, median + quartile, median),
    )


def shape(m: int, n: int, k: int, *candidates: Candidate, layers: int = 1) -> Measured:
    return Measured(
        "int8", m, n, k, verify.shape_class(n, k), layers, ["test"], 2, "seed",
        Quantizer(0.0, 0, 0), 1.0, 1.0, list(candidates),
    )  # fmt: skip


def test_a_rule_ships_only_where_it_beats_the_default_beyond_the_spread() -> None:
    evidence = [
        # (2048, 4096] square: tile 4 wins by 20% over a 1% spread at both shapes.
        shape(4096, 1280, 1280, timed(DEFAULT, 120.0), timed(4, 100.0), timed(0, 130.0)),
        shape(3000, 3072, 3072, timed(DEFAULT, 360.0), timed(4, 300.0), timed(0, 330.0)),
        # (4096, 8192] up: the best is within its own 30% spread of the default.
        shape(8192, 10240, 1280, timed(DEFAULT, 1000.0, 0.3), timed(5, 900.0, 0.3)),
        # (0, 1] square: the GEMV is the default; no other candidate beats it.
        shape(1, 4096, 4096, timed(DEFAULT, 10.0), timed(verify.GEMV, 10.0), timed(3, 40.0)),
    ]
    assert calibrate._rules(evidence, SMS) == (verify.Rule(2048, 4096, "square", 4, 128, 256),)


def test_a_rule_regresses_no_shape_of_its_cell() -> None:
    # Tile 4 wins the cell's weighted time but loses 10% at the second shape: no rule. Tile 0 is
    # slower at the first shape but never regresses, yet gains too little to ship.
    evidence = [
        shape(8192, 1280, 1280, timed(DEFAULT, 100.0), timed(4, 50.0), timed(0, 100.5), layers=60),
        shape(4608, 3072, 3072, timed(DEFAULT, 100.0), timed(4, 110.0), timed(0, 100.0), layers=1),
    ]
    assert calibrate._rules(evidence, SMS) == ()
    refused = Candidate(4, True, note="not supported: tile 4 cannot take 2900x3072")
    evidence = [
        shape(4096, 1280, 1280, timed(DEFAULT, 120.0), timed(4, 100.0), timed(0, 110.0)),
        shape(2900, 3072, 3072, timed(DEFAULT, 360.0), refused, timed(0, 300.0)),
    ]
    assert calibrate._rules(evidence, 24) == (verify.Rule(2048, 4096, "square", 0, 16, 32),)


def test_only_the_default_and_the_quantizer_fail_the_verdict() -> None:
    good = [shape(154, 1280, 2048, timed(DEFAULT, 9.0), timed(2, 8.0))]
    verdict = calibrate._verdict(good)
    assert verdict.ok and "default worst" in verdict.detail and "budget 2.20e-03" in verdict.detail
    wrong = Candidate(6, False, {"bfloat16": 0.25, "float16": -1.0}, BUDGET, note="failed")
    excluded = calibrate._verdict([shape(154, 1280, 2048, timed(DEFAULT, 9.0), wrong)])
    assert excluded.ok and "1 other candidates excluded" in excluded.detail
    broken = Candidate(DEFAULT, False, {"bfloat16": 0.25, "float16": 0.3}, BUDGET, note="failed")
    assert not calibrate._verdict([shape(154, 1280, 2048, broken)]).ok
    lawless = shape(154, 1280, 2048, timed(DEFAULT, 9.0))
    lawless.quantizer = Quantizer(3.0, 7, 7)
    assert "broke its law" in calibrate._verdict([lawless]).detail
    assert not calibrate._verdict([]).ok


def test_the_evidence_round_trips_as_strict_json() -> None:
    record = shape(1, 4096, 4096, timed(DEFAULT, 10.0), timed(verify.GEMV, 9.0))
    assert msgspec.json.decode(msgspec.json.encode(record), type=Measured) == record


def test_the_packages_shapes_are_legal_for_their_formats() -> None:
    shapes = msgspec.json.decode(calibrate.SHAPES.read_bytes(), type=list[calibrate.Shape])
    assert {one.format for one in shapes} == {"int8", "fp8"}
    assert {one.model for one in shapes} == {"sdxl", "qwen-image", "klein", "describer", "h3"}
    # int8 rotates 256-column groups; fp8 rows are 32-column blocks; N is a multiple of 8.
    assert all(one.k % (256 if one.format == "int8" else 32) == 0 for one in shapes)
    assert all(one.n % 8 == 0 and one.m > 0 and one.layers > 0 for one in shapes)


def test_every_16_row_block_has_a_reference_row() -> None:
    rows = calibrate._rows(154, random.Random(1))
    assert rows[-1] == 153 and {row // 16 for row in rows} == set(range(10))
    assert calibrate._rows(3, random.Random(1))[-1] == 2


def test_the_reference_and_the_quantizer_law_on_the_cpu() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    m, n, k = 300, 64, 512
    x = torch.randn(m, k, generator=generator).to(torch.bfloat16)
    rotated = hadamard256(torch, x.double()) / 16
    row = (rotated.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-12)
    q = torch.round(rotated / row).clamp(-127, 127).to(torch.int8)
    data = torch.randint(-127, 128, (n, k), generator=generator, dtype=torch.int8)
    scale, bias = torch.rand(n, generator=generator) * 1e-3, torch.randn(n).to(torch.bfloat16)
    one = calibrate._Set(x, q, row.view(-1).float(), data, scale, bias)
    rows = torch.tensor(calibrate._rows(m, random.Random(2)))
    law, ties = calibrate._law(torch, prebuilt.INT8, one, rows)
    assert law == Quantizer(0.0, 0, 0) and ties.shape == (len(rows), k)
    want = (q[rows].double() @ data.double().t()) * row[rows].float().double()
    want = want * scale.double() + bias.double()
    assert torch.equal(calibrate._exact(torch, one, rows), want)
    column = int((~ties[0]).nonzero()[0])
    q[rows[0], column] += 1 if q[rows[0], column] < 0 else -1  # one step, but at no tie
    assert calibrate._law(torch, prebuilt.INT8, one, rows)[0] == Quantizer(1.0, 1, 1)


def test_only_the_offline_oracle_imports_the_calibration() -> None:
    root = Path(calibrate.__file__).parents[1]
    importers = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "calibrate" for alias in node.names
            ):
                importers.append(str(path.relative_to(root)))
    assert importers == ["cli/qualify.py"]
