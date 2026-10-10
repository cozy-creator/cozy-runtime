"""CPU layout/custody and reuse controls, not attention GPU arithmetic claims."""

import ast
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from head_partition_probe import metrics, probe  # noqa: E402


def test_partitions_cover_exact_head_order_without_padding():
    for degree, count in ((2, 28), (4, 14), (7, 8), (8, 7)):
        spans = probe.ranges(degree)
        assert len(spans) == degree and all(stop - start == count for start, stop in spans)
        assert [i for start, stop in spans for i in range(start, stop)] == list(range(56))


def test_fresh_canonical_group_layout_preserves_input_bytes_and_order():
    original = torch.arange(1 * 129 * 56 * 128, dtype=torch.float32).reshape(1, 129, 56, 128)
    original_bytes = original.clone()
    reference = probe.canonical(original)
    for degree in probe.DEGREES:
        rows = []
        for start, stop in probe.ranges(degree):
            piece = probe.fresh([original], start, stop)[0]
            assert piece.is_contiguous() and piece.storage_offset() == 0
            assert piece.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
            rows.append(probe.canonical(piece))
        assert torch.equal(torch.cat(rows), reference)
        assert torch.equal(original, original_bytes)


def test_metric_functions_are_ast_identical_to_gpu_qualified_rowwise_probe():
    root = Path(__file__).parents[1]
    left = ast.parse(
        (root / "h3-rowwise-partition-probe/rowwise_partition_probe/probe.py").read_text()
    )
    right = ast.parse(Path(metrics.__file__).read_text())
    original = {n.name: ast.dump(n) for n in left.body if isinstance(n, ast.FunctionDef)}
    for node in right.body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node) == original[node.name]


def test_head_major_first_difference_survives_group_accumulation():
    original = torch.zeros(1, 3, 56, 128, dtype=torch.bfloat16)
    changed = original.clone()
    changed[0, 2, 41, 7] = 1
    result = metrics.new_metric()
    for start, stop in probe.ranges(4):
        a = probe.canonical(changed[:, :, start:stop])
        b = probe.canonical(original[:, :, start:stop])
        metrics.accumulate(torch, result, probe.metric_rows(a), probe.metric_rows(b))
    result = metrics.finish(result)
    assert result["different_elements"] == 1
    assert result["first_different_element"] == (41 * 3 + 2) * 128 + 7
