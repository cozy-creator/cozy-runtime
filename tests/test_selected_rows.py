"""Exact CPU lookup from real file descriptors, including rows split across objects."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal.selected_rows import SelectedRows  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ids", [[], [7, 1, 7, 0], [[2, 3], [4, 2]]])
def test_lookup_is_exact_with_bounded_buffers_and_cross_object_rows(
    tmp_path: Path, dtype: object, ids: object
) -> None:
    torch.set_num_threads(2)
    weight = torch.arange(11 * 13, dtype=torch.float32).reshape(11, 13).to(dtype)
    weight[0, 0] = -0.0
    payload = weight.view(torch.uint8).numpy().tobytes()
    split = 3 * 26 + 7  # one requested row crosses the object boundary
    files, items = [], []
    position = 0
    for index, part in enumerate((payload[:split], payload[split:])):
        path = tmp_path / str(index)
        path.write_bytes(b"prefix" + part + b"suffix")
        fd = os.open(path, os.O_RDONLY)
        files.append((str(index), path.stat().st_size, fd))
        items.append(
            {
                "source": "object",
                "object": str(index),
                "object_length": path.stat().st_size,
                "dest_off": 99 + position,
                "off": 6,
                "len": len(part),
            }
        )
        position += len(part)
    rows = SelectedRows((11, 13), "bf16" if dtype == torch.bfloat16 else "f16", items, files)
    for _, _, fd in files:
        os.close(fd)  # the reader owns its lifetime, not the caller's borrowed handles
    indices = torch.tensor(ids, dtype=torch.long)
    try:
        result = rows.lookup(torch, indices, torch.device("cpu"), dtype, 2 * (26 + 64))
        reference = torch.nn.functional.embedding(indices, weight)
        assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
        assert rows.read_bytes == indices.numel() * 26
        assert rows.peak_host_bytes <= 2 * (26 + 64)
        assert rows.h2d_bytes == 0
    finally:
        rows.close()


def test_source_coverage_refuses_gaps_before_any_lookup(tmp_path: Path) -> None:
    path = tmp_path / "weight"
    path.write_bytes(bytes(100))
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="exact stored table"):
            SelectedRows(
                (10, 10),
                "bf16",
                [
                    {
                        "source": "object",
                        "object": "a",
                        "object_length": 100,
                        "dest_off": 0,
                        "off": 0,
                        "len": 100,
                    }
                ],
                [("a", 100, fd)],
            )
    finally:
        os.close(fd)


def test_lookup_preserves_dtype_conversion_and_rejects_bad_indices(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
    path = tmp_path / "weight"
    path.write_bytes(weight.view(torch.uint8).numpy().tobytes())
    fd = os.open(path, os.O_RDONLY)
    try:
        reader = SelectedRows(
            (3, 4),
            "bf16",
            [
                {
                    "source": "object",
                    "object": "a",
                    "object_length": 24,
                    "dest_off": 0,
                    "off": 0,
                    "len": 24,
                }
            ],
            [("a", 24, fd)],
        )
        indices = torch.tensor([2, 0], dtype=torch.int32)
        actual = reader.lookup(torch, indices, torch.device("cpu"), torch.float32, 0)
        assert torch.equal(actual, torch.nn.functional.embedding(indices, weight.float()))
        assert reader.peak_host_bytes <= 4 * 2 + 64
        for invalid in (-1, 3):
            with pytest.raises(IndexError):
                reader.lookup(torch, torch.tensor([invalid]), "cpu", weight.dtype, 0)
        reader.close()
        with pytest.raises(RuntimeError, match="closed"):
            reader.lookup(torch, indices, "cpu", weight.dtype, 0)
    finally:
        os.close(fd)


@pytest.mark.real_gpu
def test_h3_shape_lookup_under_a_one_gib_cuda_allocator_ceiling(tmp_path: Path) -> None:
    """Diagnostic operator gate only: no full table allocation or artificial GPU ballast.

    Run on the coordinated rental; ordinary machine/CLI selection is a separate gate.
    """
    if not torch.cuda.is_available():
        pytest.skip("a coordinated CUDA rental is required")
    torch.set_num_threads(2)
    shape = (151936, 5120)
    row_bytes, length = shape[1] * 2, shape[0] * shape[1] * 2
    indices = torch.tensor([[0, 151935, 12, 12]], dtype=torch.int64)
    values = torch.arange(4 * shape[1], dtype=torch.float32).reshape(4, shape[1])
    values = (values / 16384).to(torch.bfloat16)
    values[3] = values[2]
    path = tmp_path / "sparse-h3-shape-table"
    with path.open("wb") as source:
        source.truncate(length)
        for index, row in zip(indices.flatten().tolist(), values, strict=True):
            source.seek(index * row_bytes)
            source.write(row.view(torch.uint8).numpy().tobytes())
    fd = os.open(path, os.O_RDONLY)
    reader = SelectedRows(
        shape,
        "bf16",
        [
            {
                "source": "object",
                "object": "a",
                "object_length": length,
                "dest_off": 0,
                "off": 0,
                "len": length,
            }
        ],
        [("a", length, fd)],
    )
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = torch.cuda.get_per_process_memory_fraction(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, (1 << 30) / total), 0)
    try:
        torch.cuda.reset_peak_memory_stats(0)
        result = reader.lookup(
            torch, indices.to("cuda"), torch.device("cuda", 0), torch.bfloat16, 1 << 20
        )
        assert torch.equal(
            result.cpu().view(torch.int16), values.reshape(1, 4, -1).view(torch.int16)
        )
        assert torch.cuda.max_memory_allocated(0) < 1 << 30
        assert reader.peak_host_bytes <= 1 << 20
        assert reader.read_bytes == indices.numel() * row_bytes
        assert reader.h2d_bytes == indices.numel() * row_bytes
    finally:
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        reader.close()
        os.close(fd)
