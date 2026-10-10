"""CPU layout/custody and reuse controls, not attention GPU arithmetic claims."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from sage2_mode_partition_probe import metrics, preprocess, probe  # noqa: E402


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


@pytest.mark.parametrize("mode", ["accurate", "per-warp", "fp16pv"])
def test_preprocessing_uses_the_pinned_upstream_call_arguments(mode):
    # Call-surface proof only. The helper below does not simulate GPU quantization.
    contract = json.loads(Path(__file__).with_name("upstream-call-contract.json").read_text())[mode]
    assert contract["core_sha256"] == preprocess.SOURCE_HASHES["core.py"]
    q, k = (torch.randn(1, 129, 14, 128, dtype=torch.bfloat16) for _ in range(2))
    calls = []

    def helper(*args, **options):
        calls.append((args, options))
        n, h = q.shape[1:3]
        q_count = -(-n // options["BLKQ"]) * (options["BLKQ"] // options["WARPQ"])
        k_count = -(-n // options["BLKK"])
        if "WARPK" in options:
            q_count *= 8
            k_count *= options["BLKK"] // options["WARPK"] * 4
        return (
            q.to(torch.int8),
            torch.ones(1, h, q_count),
            k.to(torch.int8),
            torch.ones(1, h, k_count),
        )

    outputs, evidence = preprocess.run(
        torch, SimpleNamespace(**{contract["helper"]: helper}), mode, q, k
    )
    namespace = {
        "q": q,
        "k": k,
        "seq_dim": 1,
        "tensor_layout": "NHD",
        "pv_accum_dtype": "fp32" if mode == "fp16pv" else "fp32+fp32",
        contract["helper"]: helper,
    }
    namespace["km"] = eval(contract["key_mean"], {"__builtins__": {}}, namespace)
    eval(contract["quantizer_call"], {"__builtins__": {}}, namespace)
    actual, expected = calls
    assert actual[1] == expected[1]
    assert actual[0][0] is expected[0][0] and actual[0][1] is expected[0][1]
    assert torch.equal(actual[0][2], expected[0][2])
    assert torch.equal(outputs["k_mean"], namespace["km"][0].transpose(0, 1).contiguous())
    assert evidence["helper"] == contract["helper"]


@pytest.mark.parametrize("tokens", [1, 7, 8, 16, 129, 109225])
@pytest.mark.parametrize("block,lanes", [(16, 8), (32, 8), (128, 4), (64, 4), (16, 1), (128, 1)])
def test_compared_scale_slots_are_exactly_those_consumed_by_live_tokens(tokens, block, lanes):
    # Enumerate actual token ownership, independently of the first-token mask.
    expected = sorted(
        {
            token // block * lanes + (token % 8 // 2 if lanes == 4 else token % lanes)
            for token in range(tokens)
        }
    )
    padded = -(-tokens // block) * lanes + 2 * lanes
    scale = torch.empty(1, 3, padded)
    selected = preprocess.active_scale_indices(torch, scale, tokens, block, lanes)
    assert selected.tolist() == expected


def test_unreviewed_runtime_is_refused_before_source_or_gpu_work(tmp_path):
    with pytest.raises(RuntimeError, match=r"not yet been frozen|Unsupported diagnostic cohort"):
        probe.verify_cohort({"cozy-runtime": "0.0.0-unreviewed", "tensorfs": "0.6.1"}, tmp_path)
