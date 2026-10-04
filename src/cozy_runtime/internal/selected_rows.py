"""Exact embedding lookup from immutable object descriptors, without a full table copy.

The machine verifies and grants the descriptors. This reader owns duplicated handles,
checks range coverage, and keeps one bounded row buffer. It opens no TensorFS store.
"""

from __future__ import annotations

import bisect
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True, slots=True)
class Segment:
    start: int
    end: int
    fd: int
    offset: int


class SelectedRows:
    def __init__(
        self,
        shape: tuple[int, int],
        dtype: str,
        items: Sequence[Mapping[str, Any]],
        files: Sequence[tuple[str, int, int]],
    ) -> None:
        if dtype not in ("bf16", "f16") or min(shape) <= 0:
            raise ValueError("selected-row lookup requires a plain bf16/f16 embedding")
        self.shape, self.dtype = shape, dtype
        self.row_bytes = shape[1] * 2
        self.segments: list[Segment] = []
        self.handles: dict[str, int] = {}
        self.closed = False
        self.read_bytes = self.h2d_bytes = self.copy_ns = self.peak_host_bytes = 0
        granted = {sha: (length, fd) for sha, length, fd in files}
        ordered = sorted(items, key=lambda item: int(item["dest_off"]))
        origin = int(ordered[0]["dest_off"]) if ordered else 0
        position = 0
        try:
            for item in ordered:
                if item.get("source") != "object":
                    raise ValueError("selected-row lookup requires verified object-file ranges")
                sha = str(item["object"]).removeprefix("sha256:")
                length, source = granted[sha]
                start = int(item["dest_off"]) - origin
                count, offset = int(item["len"]), int(item["off"])
                if (
                    start != position
                    or count <= 0
                    or offset < 0
                    or offset + count > length
                    or int(item["object_length"]) != length
                    or os.fstat(source).st_size != length
                ):
                    raise ValueError("embedding source ranges have a gap or leave their object")
                if sha not in self.handles:
                    self.handles[sha] = os.dup(source)
                self.segments.append(Segment(start, start + count, self.handles[sha], offset))
                position += count
            if position != shape[0] * self.row_bytes:
                raise ValueError("embedding source ranges do not cover the exact stored table")
        except BaseException:
            self.close()
            raise
        self.starts = [segment.start for segment in self.segments]

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            for fd in self.handles.values():
                os.close(fd)
            self.handles.clear()

    def _read(self, offset: int, into: memoryview) -> None:
        index = bisect.bisect_right(self.starts, offset) - 1
        written = 0
        while written < len(into):
            segment = self.segments[index]
            count = min(len(into) - written, segment.end - offset)
            done = 0
            while done < count:
                got = os.preadv(
                    segment.fd,
                    [into[written + done : written + count]],
                    segment.offset + offset - segment.start + done,
                )
                if got == 0:
                    raise EOFError("verified embedding source ended inside a requested row")
                done += got
            self.read_bytes += count
            offset += count
            written += count
            index += 1

    def lookup(
        self,
        torch: ModuleType,
        indices: Any,
        device: Any,
        dtype: Any,
        host_bytes: int,
    ) -> Any:
        if self.closed:
            raise RuntimeError("embedding source descriptors are closed")
        if indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("embedding indices must be int32 or int64")
        output = torch.empty((*indices.shape, self.shape[1]), device=device, dtype=dtype)
        flat, target = indices.reshape(-1), output.reshape(-1, self.shape[1])
        # Include a conservative per-index allowance beside the source-row bytes. A zero
        # optional pinned-tier grant still permits one pageable row, never a full table.
        batch = max(1, min(max(host_bytes, 0), 1 << 20) // (self.row_bytes + 64))
        stored = torch.bfloat16 if self.dtype == "bf16" else torch.float16
        for start in range(0, flat.numel(), batch):
            ids = flat[start : start + batch].to("cpu").tolist()
            if any(index < 0 or index >= self.shape[0] for index in ids):
                raise IndexError("embedding index out of range")
            buffer = bytearray(len(ids) * self.row_bytes)
            self.peak_host_bytes = max(self.peak_host_bytes, len(buffer) + len(ids) * 64)
            view = memoryview(buffer)
            for row, index in enumerate(ids):
                self._read(
                    index * self.row_bytes, view[row * self.row_bytes : (row + 1) * self.row_bytes]
                )
            values = torch.frombuffer(buffer, dtype=stored).reshape(len(ids), self.shape[1])
            began = time.perf_counter_ns()
            target[start : start + len(ids)].copy_(values)
            if str(device).startswith("cuda"):
                self.copy_ns += time.perf_counter_ns() - began
                self.h2d_bytes += len(ids) * self.shape[1] * output.element_size()
        return output
