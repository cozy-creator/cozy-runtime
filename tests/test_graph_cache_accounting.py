"""Allocator-block selection for graph-owned pools and static-input storage."""

from __future__ import annotations

from typing import Any

from cozy_runtime.graphs import _allocated_bytes


def test_owned_allocations_use_rounded_block_bytes_and_count_overlap_once() -> None:
    segments: list[dict[str, Any]] = [
        {
            "device": 0,
            "segment_pool_id": (0, 0),
            "blocks": [
                {"address": 10, "size": 512, "requested_size": 1, "state": "active_allocated"},
                {"address": 20, "size": 1024, "state": "active_allocated"},
                {"address": 30, "size": 2048, "state": "inactive"},
                {"address": 40, "size": 4096, "state": "active_awaiting_free"},
            ],
        },
        {
            "device": 0,
            "segment_pool_id": (7, 1),
            "blocks": [
                {"address": 100, "size": 2048, "state": "active_allocated"},
                {"address": 110, "size": 1024, "state": "active_allocated"},
                {"address": 120, "size": 8192, "state": "inactive"},
            ],
        },
        {
            "device": 1,
            "segment_pool_id": (0, 0),
            "blocks": [{"address": 10, "size": 4096, "state": "active_allocated"}],
        },
    ]
    pools = {(7, 1)}
    inputs = {(0, 10), (0, 30), (0, 40), (0, 100)}
    assert _allocated_bytes(segments, set(), inputs) == 512 + 2048
    assert _allocated_bytes(segments, pools, set()) == 2048 + 1024
    assert _allocated_bytes(segments, pools, inputs) == 512 + 2048 + 1024
    for segment in segments:
        for block in segment["blocks"]:
            block["state"] = "inactive"
    assert _allocated_bytes(segments, pools, inputs) == 0
