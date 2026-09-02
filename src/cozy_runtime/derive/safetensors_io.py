"""safetensors read/write with NO framework dependency — numpy and the struct module.

Why this exists rather than `safetensors.torch`: the two quant kinds in this module's
family is pure integer/float32 arithmetic with no
GEMM in them, so requiring torch would make their arms unrunnable anywhere torch is not
installed — which is every control-plane host, including the one their author works on.
A quantizer that cannot be exercised where it is written gets its evidence from a pod, and
that is how #517's exponent defect survived a review pass.

The format is a JSON header behind an 8-byte little-endian length, then a tightly packed
data region. Two rules this writer holds that the framework writers do not promise:

  * keys are emitted SORTED and the header is the compact JSON spelling, so the produced
    bytes are a pure function of the tensor contents — bit-reproducibility across runs and
    across hosts is a property of the writer, not a hope about dict ordering;
  * the data region starts 8-byte aligned (space padding after the header), which is what
    every mmap reader assumes and no reader checks.

Dtypes are carried as NAMES, never as numpy dtypes: F8_E4M3, U8 and BF16 have no numpy
type, and inventing one would mean this module decides what a byte means. It does not —
it moves bytes and reports the name the header carries.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from cozy_runtime.author import UnsupportedInput

#: Bytes per element, by header dtype name. A name absent here is not refused by this
#: module — it is carried verbatim as opaque bytes, because a passthrough tensor's dtype
#: is the SOURCE's fact and a converter that must understand every dtype it copies is a
#: converter that cannot carry an artifact it did not design.
ITEMSIZE: dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

#: The largest header this module will parse. A safetensors header is metadata; a
#: 100 MB one is a claim about the file, not a header, and it refuses before allocation.
MAX_HEADER = 100 << 20


def numel(shape: list[int]) -> int:
    total = 1
    for d in shape:
        total *= int(d)
    return total


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """(header, data_region_offset). A BOUNDED read: the data region is never touched."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise UnsupportedInput(f"{path.name}: shorter than a safetensors length prefix")
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_HEADER:
            raise UnsupportedInput(f"{path.name}: {length}-byte header is not a header")
        try:
            doc = json.loads(handle.read(length))
        except json.JSONDecodeError as exc:
            raise UnsupportedInput(f"{path.name}: header is not JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise UnsupportedInput(f"{path.name}: header is not an object")
    doc.pop("__metadata__", None)
    return doc, 8 + length


def read_raw(path: Path, header: dict[str, Any], base: int, key: str) -> bytes:
    """The exact stored bytes of one tensor. Seek + one read — no whole-file load."""
    row = header[key]
    start, end = (int(x) for x in row["data_offsets"])
    with path.open("rb") as handle:
        handle.seek(base + start)
        raw = handle.read(end - start)
    if len(raw) != end - start:
        raise UnsupportedInput(f"{path.name}: {key!r} is truncated in the data region")
    return raw


def to_f32(raw: bytes | bytearray | memoryview, dtype: str, shape: list[int]) -> np.ndarray:
    """Decode a FLOAT tensor's stored bytes to float32. Refuses anything else, loudly:
    a quantizer that silently reinterprets an int tensor as floats produces plausible
    numbers from bytes that never were numbers."""
    count = numel(shape)
    if dtype == "F32":
        out = np.frombuffer(raw, dtype="<f4", count=count)
    elif dtype == "F16":
        out = np.frombuffer(raw, dtype="<f2", count=count).astype(np.float32)
    elif dtype == "BF16":
        # bf16 is the high half of an f32: widen by placing the 16 bits in the top half.
        wide = np.zeros(count * 2, dtype=np.uint16)
        wide[1::2] = np.frombuffer(raw, dtype="<u2", count=count)
        out = wide.view("<f4")
    else:
        raise UnsupportedInput(
            f"{dtype} is not a float carrier this quantizer reads — it quantizes float "
            "weights, and reinterpreting other bytes as floats is how a plausible wrong "
            "artifact gets made"
        )
    return np.ascontiguousarray(out, dtype=np.float32).reshape(shape)


def from_f32(values: np.ndarray, dtype: str) -> bytes:
    """Encode float32 back to a stored float carrier, round-to-nearest-EVEN throughout."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if dtype == "F32":
        return flat.tobytes()
    if dtype == "F16":
        return flat.astype("<f2").tobytes()
    if dtype == "BF16":
        bits = flat.view(np.uint32)
        # RNE on the truncated 16 low bits: add the rounding bias then take the top half.
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return rounded.astype("<u2").tobytes()
    raise UnsupportedInput(f"{dtype} is not a float carrier this writer emits")


class Writer:
    """A BUFFERING writer for fixture-scale files: every tensor's bytes are retained
    until `write()`. Say it plainly — #553d convicted this class's old docstring of
    claiming largest-tensor bounds it never had. Arms and small evidence files use this;
    a PRODUCTION carrier goes through `StreamingWriter`, which is bounded for real."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, str, list[int], bytes]] = []

    def add(self, key: str, dtype: str, shape: list[int], raw: bytes) -> None:
        want = ITEMSIZE.get(dtype)
        if want is not None and len(raw) != numel(shape) * want:
            raise UnsupportedInput(
                f"{key!r}: {len(raw)} bytes for a {dtype} {shape} tensor, which needs "
                f"{numel(shape) * want} — the producer and its own header disagree"
            )
        self._rows.append((key, dtype, list(shape), raw))

    def header(self) -> dict[str, Any]:
        """The header these tensors WILL produce — available before any byte is written,
        which is what makes a header prediction checkable against the real thing."""
        out: dict[str, Any] = {}
        offset = 0
        for key, dtype, shape, raw in sorted(self._rows, key=lambda r: r[0]):
            out[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
            offset += len(raw)
        return out

    def write(self, path: Path) -> int:
        blob = json.dumps(self.header(), separators=(",", ":"), sort_keys=True).encode()
        blob += b" " * ((-len(blob)) % 8)
        path.parent.mkdir(parents=True, exist_ok=True)
        total = 8 + len(blob)
        with path.open("wb") as handle:
            handle.write(struct.pack("<Q", len(blob)))
            handle.write(blob)
            for _, _, _, raw in sorted(self._rows, key=lambda r: r[0]):
                handle.write(raw)
                total += len(raw)
        return total


class StreamingWriter:
    """The production writer (#553d): memory bounded by ONE tensor, proven at model
    shape, not asserted.

    Two phases, which is what a PREDICTED header makes possible: every tensor is
    DECLARED first (metadata only), the complete header and every final offset are
    computed, the destination is created at full size — then each tensor's bytes are
    written directly to their final offset, in any order, and freed. Passthrough tensors
    range-copy from the source file in bounded chunks and are never materialized.

    Transactional: bytes land in `<path>.tmp` and `close()` renames onto the final path
    only after every declared tensor was written — a kill at any point leaves either
    nothing or a `.tmp` no consumer looks at, never a torn final. `close()` with an
    unwritten tensor refuses, naming it."""

    _CHUNK = 64 << 20

    def __init__(self) -> None:
        self._declared: dict[str, tuple[str, list[int], int]] = {}
        self._offsets: dict[str, tuple[int, int]] = {}
        self._written: set[str] = set()
        self._handle: Any = None
        self._path: Path | None = None
        self._data_base = 0

    def declare(self, key: str, dtype: str, shape: list[int]) -> None:
        if self._handle is not None:
            raise UnsupportedInput("declare after open: the header is already on disk")
        want = ITEMSIZE.get(dtype)
        if want is None:
            raise UnsupportedInput(f"{key!r}: no itemsize for dtype {dtype!r}")
        self._declared[key] = (dtype, list(shape), numel(shape) * want)

    def header(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        offset = 0
        for key in sorted(self._declared):
            dtype, shape, length = self._declared[key]
            out[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
            offset += length
        return out

    def open(self, path: Path) -> None:
        header = self.header()
        blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
        blob += b" " * ((-len(blob)) % 8)
        self._data_base = 8 + len(blob)
        total = self._data_base + sum(length for _, _, length in self._declared.values())
        for key, row in header.items():
            self._offsets[key] = (int(row["data_offsets"][0]), int(row["data_offsets"][1]))
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._handle = (path.with_suffix(path.suffix + ".tmp")).open("wb")
        self._handle.write(struct.pack("<Q", len(blob)))
        self._handle.write(blob)
        self._handle.truncate(total)

    def put(self, key: str, raw: bytes) -> None:
        start, _end = self._require(key, len(raw))
        self._handle.seek(self._data_base + start)
        self._handle.write(raw)
        self._written.add(key)

    def put_from_file(self, key: str, src: Path, src_offset: int, length: int) -> None:
        """Range-copy a tensor from its source without materializing it (#553d)."""
        start, _ = self._require(key, length)
        self._handle.seek(self._data_base + start)
        with src.open("rb") as reader:
            reader.seek(src_offset)
            left = length
            while left:
                chunk = reader.read(min(self._CHUNK, left))
                if not chunk:
                    raise UnsupportedInput(f"{key!r}: source truncated mid range-copy")
                self._handle.write(chunk)
                left -= len(chunk)
        self._written.add(key)

    def close(self) -> int:
        missing = sorted(set(self._declared) - self._written)
        if missing:
            self._handle.close()
            raise UnsupportedInput(
                f"close with {len(missing)} declared tensors unwritten "
                f"({missing[:3]}…) — a torn carrier must not exist at the final path"
            )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        assert self._path is not None
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        os.replace(tmp, self._path)
        return self._data_base + sum(length for _, _, length in self._declared.values())

    def _require(self, key: str, length: int) -> tuple[int, int]:
        if self._handle is None:
            raise UnsupportedInput("put before open")
        if key not in self._declared:
            raise UnsupportedInput(f"{key!r} was never declared")
        start, end = self._offsets[key]
        if end - start != length:
            raise UnsupportedInput(
                f"{key!r}: {length} bytes for a declaration of {end - start} — the "
                "producer and its own header disagree"
            )
        if key in self._written:
            raise UnsupportedInput(f"{key!r} written twice")
        return start, end


def shape_only(header: dict[str, Any]) -> dict[str, Any]:
    """A header stripped to what a PREDICTION can assert: names, dtypes, shapes. Offsets
    are a packing fact and belong to the writer, not to the plan."""
    return {
        k: {"dtype": v["dtype"], "shape": [int(d) for d in v["shape"]]} for k, v in header.items()
    }


def row_chunks(rows: int, cols: int, budget_elements: int = 1 << 22) -> Iterator[slice]:
    """Row slices whose working set stays inside `budget_elements`. Every encoder in this
    family is row-independent, so chunking changes no number — it only decides whether a
    [28672, 5376] tensor needs 600 MB of float32 scratch or 16 MB of it."""
    per = max(1, budget_elements // max(1, cols))
    for start in range(0, rows, per):
        yield slice(start, min(start + per, rows))
