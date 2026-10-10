"""CPU controls for diagnostic indexing and byte/error accounting; no CUDA calls."""

import importlib.util
import math
from itertools import pairwise
from pathlib import Path

import torch

source = Path(__file__).parent / "rowwise_partition_probe/probe.py"
spec = importlib.util.spec_from_file_location("partition_probe", source)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_partitions_cover_ragged_and_full_sequences():
    for rows in (129, 1031, 109225):
        for degree in (2, 4, 7, 8):
            parts = probe.partitions(rows, degree)
            assert len(parts) == degree
            assert parts[0][0] == 0 and parts[-1][1] == rows
            assert all(a[1] == b[0] for a, b in pairwise(parts))
            lengths = [b - a for a, b in parts]
            assert max(lengths) - min(lengths) <= 1


def test_metric_detects_numeric_change_and_signed_zero_bits():
    original = torch.tensor([[0.0, -0.0], [1.0, 2.0]], dtype=torch.bfloat16)
    changed = original.clone()
    changed[0, 0] = -0.0
    changed[1, 1] = 3.0
    result = probe.compared(torch, changed, original)
    assert result["different_elements"] == 2
    assert result["first_different_element"] == 0
    assert result["max_absolute_difference"] == 1.0
    assert result["squared_error_sum"] == 1.0
    assert math.isclose(result["relative_l2"], math.sqrt(1 / 5))
    assert not result["bytes_equal"]


def test_accumulated_chunks_keep_global_difference_offset():
    original = torch.arange(24, dtype=torch.float32).reshape(8, 3).to(torch.bfloat16)
    changed = original.clone()
    changed[6, 2] += 1
    result = probe.new_metric()
    for start, stop in probe.partitions(8, 7):
        probe.accumulate(torch, result, changed[start:stop], original[start:stop])
    result = probe.finish(result)
    assert result["different_elements"] == 1
    assert result["first_different_element"] == 20
    assert result["elements"] == 24


def test_same_codes_and_nonfinite_accounting_are_explicit():
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    assert probe.compared(torch, original.clone(), original)["bytes_equal"]
    nonfinite = torch.tensor([[float("inf"), 1.0]], dtype=torch.bfloat16)
    result = probe.compared(torch, nonfinite.clone(), nonfinite)
    assert result["bytes_equal"]
    assert result["nonfinite_elements"] == 1


def test_metric_offsets_across_internal_slices():
    original = torch.zeros(1300, 3, dtype=torch.bfloat16)
    changed = original.clone()
    changed[1100, 2] = 1
    result = probe.compared(torch, changed, original)
    assert result["different_elements"] == 1
    assert result["first_different_element"] == 3302
    assert result["elements"] == 3900


def test_quantizer_route_observation_does_not_compile():
    from types import SimpleNamespace

    def kernel_factory():
        raise AssertionError("Observation must not compile or change the quantizer")

    leaves = SimpleNamespace(_triton_rowwise_kernel=kernel_factory)
    assert probe.quantizer_route(leaves) == "not resolved yet"
    kernel_factory._cached = object()
    assert probe.quantizer_route(leaves) == "Triton cached"
    kernel_factory._cached = None
    assert probe.quantizer_route(leaves) == "eager fallback"
