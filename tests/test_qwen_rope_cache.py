"""The real upstream RoPE tables are a counted, reclaimable derived cache."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
upstream = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")

from test_graph_cache_gpu import loaded_model, residency  # noqa: E402


@pytest.mark.parametrize("axes", [[16, 56, 56], [8, 8, 8]])
def test_upstream_rope_tables_are_counted_reclaimed_and_recreated(
    tmp_path: Path, axes: list[int]
) -> None:
    root, checkpoint, memory, model = loaded_model(tmp_path, "cpu")
    rope = upstream.QwenImage21Rope(theta=10000, axes_dim=axes)
    root.add_module("pos_embed", rope)
    placed = residency(memory, model)
    reference = upstream.QwenImage21Rope(theta=10000, axes_dim=axes)
    mask = torch.tensor([False, True, True, True, True, True, True, False])
    shapes = [(1, 2, 3)]
    expected = reference(shapes, mask, torch.device("cpu"))
    try:
        assert len(memory.caches) == 1, "the upstream tensor-list owner has no declared cache"
        cache = next(iter(memory.caches))
        assert cache.nbytes == 0, "construction must preserve lazy device placement"
        for _ in range(3):
            actual = rope(shapes, mask, torch.device("cpu"))
            assert torch.equal(actual, expected)
            assert actual.dtype == torch.complex64
            size = 4_718_592 if axes == [16, 56, 56] else 884_736
            assert cache.nbytes == size
            old = [weakref.ref(value) for value in rope.freqs]
            assert cache.shrink(size) == size
            assert cache.nbytes == 0
            assert not rope.freqs
            gc.collect()
            assert all(value() is None for value in old), "reclaimed table storage is still owned"
            # The returned embedding is independent of the reclaimed source tables.
            assert torch.equal(actual, expected)
    finally:
        # The generation normally closes these through its Backend.
        for owner in getattr(placed, "derived", ()):
            owner.close()
        memory.close()
        checkpoint.close()


def test_an_unrelated_tensor_list_is_not_registered(tmp_path: Path) -> None:
    root, checkpoint, memory, model = loaded_model(tmp_path, "cpu")
    unrelated: Any = torch.nn.Module()
    unrelated.freqs = [torch.ones(4)]
    root.add_module("unrelated", unrelated)
    placed = residency(memory, model)
    try:
        assert not memory.caches
        assert unrelated.freqs[0].tolist() == [1.0] * 4
        assert placed.models["net"] is model
    finally:
        memory.close()
        checkpoint.close()
