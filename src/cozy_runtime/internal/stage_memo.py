"""Memoized Model methods, executor side (tracker #298, design pipeline-memoization.md).

One general mechanism: every method marked `@uses_components(..., memoize=True)` takes this
path, and nothing here names a package, model or component. The key binds the method's code
and declared helpers, the installation's third-party environment, the weights of the
components it declares, the checkpoint's non-tensor assets, the call's arguments and the
numerics it runs under. The lookup happens before the scope opens, so a hit stages nothing.

Tier 0 is this process's LRU. Tier 1 is the Worker's machine store, reached over the
attempt's durable exchange when the Worker offers it. Writing there is a value decision made
per call from measurements: a result goes to disk only when it is small and producing it
(staging plus run) clearly costs more than reading it back.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib.metadata
import inspect
import logging
import os
import struct
import sys
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

import msgspec
import numpy as np
from PIL import Image

from cozy_runtime.author._assets import Asset
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._executor_requests import (
    Answer,
    MemoEntry,
    MemoStored,
    Request,
    StageMemoLookup,
    StageMemoStore,
)
from cozy_runtime.author._stage_memo import exactness
from cozy_runtime.internal import attention
from cozy_runtime.internal.executor_commands import MemoSettings
from cozy_runtime.internal.hostfacts import HostFacts
from cozy_runtime.internal.memo_implementation import describe
from cozy_runtime.internal.numerical_environment import fingerprint

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_LOG = logging.getLogger(__name__)
SCHEMA = "cozy.stage-memo/1"
CAPABILITY = "stage/1"
#: A written entry must cost this many times its measured read to produce.
VALUE_RATIO = 2.0
#: The newest measurement's weight in each moving average.
SMOOTHING = 0.3
_EXCLUDED = frozenset({"cozy-runtime", "tensorfs"})
#: Safetensors dtype names, by torch attribute. A result in another dtype is not stored.
_DTYPES = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}

type Exchange = Callable[[Request, type[Answer]], Answer]


class Tensor(Protocol):
    """The torch.Tensor surface this module touches; torch itself is never imported here."""

    dtype: object
    shape: Sequence[int]
    device: object
    is_sparse: bool

    def detach(self) -> Tensor: ...
    def contiguous(self) -> Tensor: ...
    def reshape(self, *shape: int) -> Tensor: ...
    def view(self, dtype: object) -> Tensor: ...
    def cpu(self) -> Tensor: ...
    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]: ...
    def to(self, device: object, *, copy: bool) -> Tensor: ...


class Unmemoizable(Exception):
    """This call cannot be keyed or stored; it runs as if unmarked. `str()` is the reason."""


type Leaf = Tensor | np.ndarray[tuple[int, ...], np.dtype[np.generic]] | Image.Image


def _raw(leaf: Leaf, torch: ModuleType) -> memoryview:
    """A leaf's bytes, without a copy where it is already contiguous on the host."""
    if isinstance(leaf, np.ndarray):
        return memoryview(np.ascontiguousarray(leaf)).cast("B")
    if isinstance(leaf, Image.Image):
        return memoryview(leaf.tobytes())
    flat = leaf.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return memoryview(flat)


def _flatten(
    value: object, torch: ModuleType, leaves: dict[str, Leaf], host: bool, depth: int = 0
) -> object:
    """`value` as a JSON structure with its tensors, arrays and images pulled out into
    `leaves`. Arguments (`host`) admit only host tensors, so keying one never waits for a
    device; results admit tensors anywhere. Anything else is `Unmemoizable`.

    A plain recursive function, never a closure: a closure that names itself is a reference
    cycle holding `leaves`, which would keep a result's device tensors allocated until the
    collector next ran, past the end of the attempt."""

    def leaf(item: Leaf) -> str:
        leaves[name := f"t{len(leaves)}"] = item
        return name

    def walk(item: object) -> object:
        return _flatten(item, torch, leaves, host, depth + 1)

    if depth > 16:
        raise Unmemoizable("nesting")
    if isinstance(value, torch.Tensor):
        tensor = cast(Tensor, value)
        if (
            type(value) is not torch.Tensor
            or tensor.is_sparse
            or (host and str(tensor.device) != "cpu")
        ):
            raise Unmemoizable("tensor not a plain host tensor" if host else "tensor kind")
        if str(tensor.dtype).removeprefix("torch.") not in _DTYPES:
            raise Unmemoizable(f"dtype {tensor.dtype}")
        return ["t", leaf(tensor.contiguous())]
    if isinstance(value, Image.Image):
        return ["pil", value.mode, list(value.size), leaf(value)]
    if isinstance(value, np.ndarray):
        return ["np", value.dtype.str, list(value.shape), leaf(value)]
    if isinstance(value, enum.Enum):
        return ["e", _qualname(value), walk(value.value)]
    if value is None or isinstance(value, (bool, int, str)):
        return ["v", value]
    if isinstance(value, float):
        return ["f", value.hex()]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["b", bytes(value).hex()]
    if isinstance(value, Asset) and host and value.digest:
        return ["a", type(value).__name__, value.digest]
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return ["n", _qualname(value), [walk(v) for v in value]]
    if type(value) in (tuple, list):
        items = cast(Sequence[object], value)
        return [type(value).__name__, [walk(v) for v in items]]
    if type(value) is dict and all(isinstance(k, str) for k in value):
        return ["dict", [[k, walk(v)] for k, v in value.items()]]
    if isinstance(value, msgspec.Struct) or dataclasses.is_dataclass(value):
        fields = (
            value.__struct_fields__
            if isinstance(value, msgspec.Struct)
            else [f.name for f in dataclasses.fields(cast("DataclassInstance", value))]
        )
        return ["r", _qualname(value), [[f, walk(getattr(value, f))] for f in fields]]
    raise Unmemoizable(f"{'argument' if host else 'result'} of type {type(value).__name__}")


def _qualname(value: object) -> str:
    return f"{type(value).__module__}:{type(value).__qualname__}"


# ---------------------------------------------------------------------------------- entries


class _Meta(msgspec.Struct, frozen=True):
    schema: str
    key: str
    sha256: str
    structure: object
    devices: dict[str, str]


class _Row(msgspec.Struct, frozen=True):
    dtype: str
    shape: list[int]
    data_offsets: tuple[int, int]


@dataclasses.dataclass(slots=True)
class Entry:
    """One computed result: a safetensors payload whose metadata carries its structure. The
    payload is a function of the result alone, so two computations agree byte for byte."""

    payload: bytearray
    meta: _Meta
    rows: dict[str, _Row]
    base: int
    cost_s: float = 0.0

    @property
    def nbytes(self) -> int:
        return len(self.payload)

    @classmethod
    def parse(cls, raw: bytes, key: str) -> Entry:
        """A stored entry, refused unless it is exactly the entry for `key`."""
        size = struct.unpack("<Q", raw[:8])[0] if len(raw) >= 8 else -1
        if not 0 < size <= min(len(raw) - 8, 1 << 24):
            raise Unmemoizable("entry header")
        header = msgspec.json.decode(raw[8 : 8 + size], type=dict[str, msgspec.Raw])
        meta = msgspec.json.decode(
            msgspec.json.decode(header.pop("__metadata__"), type=dict[str, str])[SCHEMA],
            type=_Meta,
        )
        rows = {name: msgspec.json.decode(row, type=_Row) for name, row in header.items()}
        entry = cls(bytearray(raw), meta, rows, 8 + size)
        if meta.schema != SCHEMA or meta.key != key or entry.digest() != meta.sha256:
            raise Unmemoizable("entry names another key or differs from its digest")
        return entry

    def digest(self) -> str:
        """The result's digest: structure and tensors, independent of the key."""
        digest = hashlib.sha256(msgspec.json.encode(self.meta.structure, order="sorted"))
        for name in sorted(self.rows):
            row = self.rows[name]
            digest.update(f"{name}:{row.dtype}:{row.shape}".encode())
            digest.update(self._data(row))
        return "sha256:" + digest.hexdigest()

    def _data(self, row: _Row) -> memoryview:
        start, end = row.data_offsets
        return memoryview(self.payload)[self.base + start : self.base + end]

    def materialize(self, torch: ModuleType) -> object:
        """Fresh tensors on their recorded devices: the cache keeps its own bytes."""
        tensors: dict[str, object] = {}
        for name, row in self.rows.items():
            dtype = getattr(torch, next(k for k, v in _DTYPES.items() if v == row.dtype))
            data = self._data(row)
            flat: Tensor = (
                torch.frombuffer(data, dtype=torch.uint8).view(dtype)
                if len(data)
                else torch.empty(0, dtype=dtype)
            )
            tensors[name] = flat.reshape(*row.shape).to(self.meta.devices[name], copy=True)
        return _rebuild(self.meta.structure, tensors)


def _pack(structure: object, leaves: Mapping[str, Leaf], key: str, torch: ModuleType) -> Entry:
    rows: dict[str, _Row] = {}
    chunks: list[memoryview] = []
    devices: dict[str, str] = {}
    offset = 0
    for name, item in leaves.items():
        chunks.append(_raw(item, torch))
        size = offset + len(chunks[-1])
        if isinstance(item, (np.ndarray, Image.Image)):
            rows[name], devices[name] = _Row("U8", [len(chunks[-1])], (offset, size)), "cpu"
        else:
            dtype = _DTYPES[str(item.dtype).removeprefix("torch.")]
            rows[name], devices[name] = (
                _Row(dtype, list(item.shape), (offset, size)),
                str(item.device),
            )
        offset = size
    probe = Entry(bytearray(b"".join(chunks)), _Meta(SCHEMA, key, "", structure, devices), rows, 0)
    meta = _Meta(SCHEMA, key, probe.digest(), structure, devices)
    head = msgspec.json.encode(
        {"__metadata__": {SCHEMA: msgspec.json.encode(meta, order="sorted").decode()}, **rows},
        order="sorted",
    )
    head += b" " * (-(len(head) + 8) % 8)
    payload = bytearray(struct.pack("<Q", len(head)) + head + probe.payload)
    return Entry(payload, meta, rows, 8 + len(head))


def encode(result: object, key: str, torch: ModuleType) -> tuple[Entry, object]:
    """The entry for `result`, and what the caller receives: the same structure with every
    tensor contiguous, so a hit and a miss hand back one form. A non-finite result is not
    kept: a hit must never replay a diverged computation."""
    leaves: dict[str, Leaf] = {}
    structure = _flatten(result, torch, leaves, host=False)
    floating = [t for t in leaves.values() if isinstance(t, torch.Tensor) and t.is_floating_point()]
    if not all(bool(torch.isfinite(t.float()).all()) for t in floating):
        raise Unmemoizable("non-finite result")
    return _pack(structure, leaves, key, torch), _rebuild(structure, leaves)


def arguments_digest(arguments: object, torch: ModuleType) -> str:
    """The canonical digest of a call's bound arguments, hashed in place: what the key binds."""
    leaves: dict[str, Leaf] = {}
    structure = _flatten(arguments, torch, leaves, host=True)
    digest = hashlib.sha256(msgspec.json.encode(structure, order="sorted"))
    for name, item in leaves.items():
        shape = item.size if isinstance(item, Image.Image) else tuple(item.shape)
        digest.update(f"{name}:{getattr(item, 'dtype', '')}:{shape}".encode())
        digest.update(_raw(item, torch))
    return "sha256:" + digest.hexdigest()


def _type(qualname: str) -> Callable[..., object]:
    """A result record's class, from the already-imported module that defined it."""
    module, _, path = qualname.partition(":")
    found: object = sys.modules.get(module)
    for part in path.split("."):
        found = getattr(found, part, None)
    if not isinstance(found, type):
        raise Unmemoizable(f"result type {qualname} is not importable here")
    return found


def _rebuild(node: object, tensors: Mapping[str, object]) -> object:
    parts = cast(list[object], node)
    kind, first = parts[0], parts[1]
    if kind == "t":
        return tensors[str(first)]
    if kind in ("np", "pil"):
        if isinstance(stored := tensors[str(parts[3])], (np.ndarray, Image.Image)):
            return stored  # a miss hands back its own value; the entry holds its bytes
        raw = cast(Tensor, stored).numpy().tobytes()
        shape = tuple(cast(list[int], parts[2]))
        if kind == "pil":
            return Image.frombytes(str(first), (shape[0], shape[1]), raw)
        return np.frombuffer(raw, dtype=str(first)).reshape(shape).copy()
    if kind == "v":
        return first
    if kind == "f":
        return float.fromhex(str(first))
    if kind == "b":
        return bytes.fromhex(str(first))
    if kind == "e":
        return _type(str(first))(_rebuild(parts[2], tensors))
    if kind in ("tuple", "list"):
        items = [_rebuild(item, tensors) for item in cast(list[object], first)]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {str(k): _rebuild(v, tensors) for k, v in cast(list[list[object]], first)}
    rest = cast(list[object], parts[2])
    if kind == "n":
        return _type(str(first))(*(_rebuild(item, tensors) for item in rest))
    pairs = cast(list[list[object]], rest)
    return _type(str(first))(**{str(k): _rebuild(v, tensors) for k, v in pairs})


# --------------------------------------------------------------------------------- identity


@functools.cache
def environment(model_module: str) -> str:
    """The installation's third-party distributions. The Runtime, TensorFS and the package
    itself are excluded: the method's own code enters the key through `describe`."""

    def spelled(name: str) -> str:
        return name.lower().replace("_", "-")

    owned = importlib.metadata.packages_distributions().get(model_module.split(".")[0], [])
    excluded = _EXCLUDED | {spelled(name) for name in owned}
    rows = sorted(
        f"{name}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if (name := spelled(str(dist.metadata["Name"] or ""))) and name not in excluded
    )
    return _hash(rows)


def _hash(document: object) -> str:
    return "sha256:" + hashlib.sha256(msgspec.json.encode(document, order="sorted")).hexdigest()


# ---------------------------------------------------------------------------------- engine


@dataclasses.dataclass(slots=True)
class MethodStats:
    hits: int = 0
    machine_hits: int = 0
    misses: int = 0
    skips: Counter[str] = dataclasses.field(default_factory=Counter)
    stored_bytes: int = 0
    saved_s: float = 0.0
    produce_s: float = 0.0
    read_s: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class Call:
    stage: str
    outcome: str
    reason: str = ""
    sha256: str = ""
    bytes: int = 0
    ms: float = 0.0


def _ewma(previous: float, sample: float) -> float:
    return sample if previous <= 0 else previous + SMOOTHING * (sample - previous)


def _default_process_bytes() -> int:
    """1 GiB, or a sixteenth of the host's memory where that is less."""
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return min(1 << 30, physical // 16) if physical > 0 else 1 << 30


class Engine:
    """This process's tier 0, its statistics and its link to the Worker's machine tier."""

    def __init__(self) -> None:
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.held = 0
        self.stats: dict[str, MethodStats] = {}
        self.calls: list[Call] = []
        self.disputed: set[tuple[str, str]] = set()
        self.exchange: Exchange | None = None
        self.spool: Path | None = None
        self._stages: dict[Callable[..., object], tuple[str | None, inspect.Signature]] = {}
        self._keys: dict[str, tuple[threading.Lock, list[int]]] = {}
        self._lock = threading.RLock()
        self.configure(MemoSettings())

    def configure(self, settings: MemoSettings) -> None:
        with self._lock:
            self.settings = settings
            self.process_bytes = (
                settings.process_bytes if settings.process_bytes >= 0 else _default_process_bytes()
            )
            self._trim()

    def open(self, exchange: Exchange | None, spool: Path | None) -> None:
        """An attempt starts: the Worker's machine tier is reachable only inside one."""
        self.exchange = exchange if self.settings.entry_bytes > 0 else None
        self.spool = spool

    def close(self, sink: Callable[..., object]) -> None:
        """The attempt ends: its calls and its methods' running totals become observations."""
        self.exchange, self.spool = None, None
        with self._lock:
            calls, self.calls = self.calls, []
        for call in calls:
            sink("metric", "memo.call", round(call.ms, 3), **dataclasses.asdict(call))
        for stage in sorted({call.stage for call in calls}):
            row = dataclasses.asdict(self.stats[stage])
            row["skips"] = ",".join(f"{k}={v}" for k, v in sorted(self.stats[stage].skips.items()))
            sink("metric", "memo.method", round(row.pop("saved_s") * 1000, 3), stage=stage, **row)

    def stage(self, fn: Callable[..., object]) -> tuple[str | None, inspect.Signature]:
        found = self._stages.get(fn)
        if found is None:
            found = self._stages[fn] = (
                describe(fn).get("operation_identity"),
                inspect.signature(fn),
            )
        return found

    def get(self, key: str) -> Entry | None:
        with self._lock:
            if (entry := self.entries.get(key)) is not None:
                self.entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: Entry) -> None:
        with self._lock:
            if entry.nbytes <= self.process_bytes and key not in self.entries:
                self.entries[key] = entry
                self.held += entry.nbytes
                self._trim()

    def _trim(self) -> None:
        while self.entries and self.held > self.process_bytes:
            self.held -= self.entries.popitem(last=False)[1].nbytes

    @contextmanager
    def single(self, key: str) -> Iterator[None]:
        """One computation per key in this process; a concurrent caller waits for it."""
        with self._lock:
            lock, users = self._keys.setdefault(key, (threading.Lock(), [0]))
            users[0] += 1
        try:
            with lock:
                yield
        finally:
            with self._lock:
                users[0] -= 1
                if not users[0]:
                    del self._keys[key]

    def fetch(self, key: str, stage: str, numerics: str) -> Entry | None:
        if self.exchange is None:
            return None
        answer = self.exchange(StageMemoLookup(key=key, stage=stage, numerics=numerics), MemoEntry)
        if not isinstance(answer, MemoEntry) or not answer.ok:
            return None
        if answer.disputed:
            self.disputed.add((stage, numerics))
        if not answer.local:
            return None
        path = Path(answer.local)
        try:
            entry = Entry.parse(path.read_bytes(), key)
        except (OSError, ValueError, KeyError, msgspec.ValidationError, Unmemoizable):
            return None
        finally:
            path.unlink(missing_ok=True)
        entry.cost_s = answer.cost_ms / 1000
        return entry

    def store(self, key: str, stage: str, numerics: str, entry: Entry | None, reason: str) -> str:
        """Hand `entry` to the Worker, or end the key's claim saying why there is none."""
        if self.exchange is None:
            return reason
        local = ""
        if entry is not None and self.spool is not None:
            target = self.spool / f"stage-memo-{key[:16]}-{time.monotonic_ns()}.safetensors"
            target.write_bytes(entry.payload)
            local = str(target)
        answer = self.exchange(
            StageMemoStore(
                key=key,
                stage=stage,
                numerics=numerics,
                local=local,
                sha256=entry.meta.sha256 if entry is not None else "",
                length=entry.nbytes if entry is not None else 0,
                cost_ms=round(entry.cost_s * 1000, 3) if entry is not None else 0.0,
                reason=reason,
            ),
            MemoStored,
        )
        stored = isinstance(answer, MemoStored) and answer.ok and answer.stored
        if isinstance(answer, MemoStored) and answer.disputed:
            self.disputed.add((stage, numerics))
        if not local:
            return reason
        Path(local).unlink(missing_ok=True)  # the Worker keeps its own copy
        return "" if stored else getattr(answer, "reason", "") or "worker_declined"

    def worth_storing(self, stats: MethodStats, entry: Entry) -> str:
        """Why not to write this entry to disk, or "" to write it."""
        if entry.nbytes > self.settings.entry_bytes:
            return "too_large"
        read = stats.read_s or max((s.read_s for s in self.stats.values()), default=0.0)
        return "cheap" if read > 0 and entry.cost_s < VALUE_RATIO * read else ""

    def record(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)


ENGINE = Engine()


# ------------------------------------------------------------------------------ generation


@dataclasses.dataclass(slots=True)
class GenerationMemo:
    """The `StageMemo` one constructed generation carries (`Model._cozy_memo`)."""

    torch: ModuleType
    environment: str
    components: Mapping[str, str]
    assets: str
    #: process facts that change arithmetic: numerical fingerprint, GPU, fused kernels
    device: str
    #: per component, the attention processors whose selected kernel is read at call time
    sites: Mapping[str, tuple[object, ...]]
    engine: Engine = dataclasses.field(default_factory=lambda: ENGINE)

    def numerics(self, components: tuple[str, ...]) -> str:
        backends = self.torch.backends
        matmul = backends.cuda.matmul
        return _hash(
            [
                self.device,
                bool(matmul.allow_tf32),
                bool(backends.cudnn.allow_tf32),
                bool(matmul.allow_fp16_reduced_precision_reduction),
                bool(matmul.allow_bf16_reduced_precision_reduction),
                bool(self.torch.are_deterministic_algorithms_enabled()),
                bool(backends.cudnn.deterministic),
                str(self.torch.get_float32_matmul_precision()),
                [
                    [str(getattr(p, "_attention_backend", "")) for p in self.sites.get(name, ())]
                    for name in components
                ],
            ]
        )

    def call(
        self,
        model: object,
        fn: Callable[..., object],
        components: tuple[str, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        run: Callable[[], object],
    ) -> object:
        started = time.perf_counter()
        engine, name = self.engine, f"{type(model).__qualname__}.{fn.__name__}"
        stats = engine.stats.setdefault(name, MethodStats())
        stage, signature = engine.stage(fn)
        try:
            if stage is None:
                raise Unmemoizable("method source unavailable")
            bound = signature.bind(model, *args, **kwargs)
            bound.apply_defaults()
            arguments = arguments_digest(dict(list(bound.arguments.items())[1:]), self.torch)
            numerics = self.numerics(components)
            if (stage, numerics) in engine.disputed:
                raise Unmemoizable("disputed")
        except Unmemoizable as exc:
            result = run()
            stats.skips[str(exc)] += 1
            engine.record(Call(name, "skip", str(exc), ms=(time.perf_counter() - started) * 1e3))
            return result
        key = _hash(
            [
                SCHEMA,
                stage,
                self.environment,
                [self.components.get(c, "") for c in components],
                self.assets,
                arguments,
                numerics,
            ]
        )[7:]
        with engine.single(key):
            tier, entry = "memory", engine.get(key)
            if entry is None and (entry := engine.fetch(key, stage, numerics)) is not None:
                tier = "machine"
                engine.put(key, entry)
            if entry is None:
                return self._miss(name, stats, key, stage, numerics, run, started)
            result = entry.materialize(self.torch)
            spent = time.perf_counter() - started
            stats.hits += 1
            if tier == "machine":
                stats.machine_hits += 1
                stats.read_s = _ewma(stats.read_s, spent)
            stats.saved_s += max(0.0, max(stats.produce_s, entry.cost_s) - spent)
            engine.record(
                Call(name, f"hit:{tier}", "", entry.meta.sha256, entry.nbytes, spent * 1e3)
            )
            return result

    def _miss(
        self,
        name: str,
        stats: MethodStats,
        key: str,
        stage: str,
        numerics: str,
        run: Callable[[], object],
        started: float,
    ) -> object:
        engine, before = self.engine, _rng(self.torch)
        try:
            with exactness() as exact:
                result = run()
        except BaseException:
            engine.store(key, stage, numerics, None, "failed")
            raise
        if _rng(self.torch) != before:
            engine.store(key, stage, numerics, None, "memo_rng")
            raise ConformanceError(
                f"{name} drew from the default random generator: a memoized method's result "
                "must be a function of its arguments, so take an explicit seed argument",
                code="memo_rng",
            )
        cost = time.perf_counter() - started
        stats.misses += 1
        stats.produce_s = _ewma(stats.produce_s, cost)
        try:
            entry, normalized = encode(result, key, self.torch)
        except Unmemoizable as exc:
            engine.store(key, stage, numerics, None, "result")
            stats.skips[str(exc)] += 1
            engine.record(Call(name, "skip", str(exc), ms=cost * 1e3))
            return result
        entry.cost_s = cost
        reason = "" if exact[0] else "inexact"
        if not reason:
            engine.put(key, entry)
            if engine.exchange is not None:
                reason = engine.worth_storing(stats, entry)
        reason = engine.store(key, stage, numerics, None if reason else entry, reason)
        if reason:
            stats.skips[reason] += 1
        elif engine.exchange is not None:
            stats.stored_bytes += entry.nbytes
        engine.record(Call(name, "miss", reason, entry.meta.sha256, entry.nbytes, cost * 1e3))
        return normalized


def _rng(torch: ModuleType) -> tuple[bytes, ...]:
    states: list[Tensor] = [torch.random.get_rng_state()]
    if torch.cuda.is_initialized():
        states.append(torch.cuda.get_rng_state())
    return tuple(state.numpy().tobytes() for state in states)


def install(
    model: object,
    *,
    torch: ModuleType,
    headers: Mapping[str, Mapping[str, object]],
    assets: Mapping[str, object],
    plan_rows: Iterable[Mapping[str, object]],
    adapters: Iterable[Mapping[str, object]],
    roots: Mapping[str, object],
    device: object,
    world: int,
    sealed: Mapping[str, str],
    fusion: object,
) -> None:
    """Give one constructed generation its memo, once per load. `headers` maps each component
    to the checkpoint header it fills from; `assets` is the header carrying assets and configs.
    In a group only GPU 0 runs package code and commands the others' module calls, so a hit
    there skips the call on every GPU; the group's width is part of the numerics. A
    generation whose identity cannot be read serves unmemoized, never refused."""
    try:
        memo = _generation(
            model,
            torch,
            headers,
            assets,
            list(plan_rows),
            list(adapters),
            roots,
            device,
            world,
            sealed,
            fusion,
        )
    except Exception as exc:  # the memo is an optimization: this load serves without it
        _LOG.warning("stage memo off for %s: %s", type(model).__qualname__, exc)
        return
    object.__setattr__(model, "_cozy_memo", memo)


def _generation(
    model: object,
    torch: ModuleType,
    headers: Mapping[str, Mapping[str, object]],
    assets: Mapping[str, object],
    rows: list[Mapping[str, object]],
    adapters: list[Mapping[str, object]],
    roots: Mapping[str, object],
    device: object,
    world: int,
    sealed: Mapping[str, str],
    fusion: object,
) -> GenerationMemo:
    kind, sm, gpu = (getattr(device, name) for name in ("kind", "sm", "name"))
    numerical = fingerprint(
        HostFacts(backend="" if kind == "cpu" else str(kind), gpu_sm=int(sm)),
        threads=int(torch.get_num_threads()),
        inherited=sealed,
    ).hex()
    processors: dict[str, list[object]] = {}
    for component, _module, processor in attention.sites(roots):
        processors.setdefault(component, []).append(processor)

    def weights(name: str) -> str:
        """One component's weights as served: its stored tensors and config, the delivery
        the plan resolved for each tensor, and any adapter composed onto it."""
        header = headers.get(name, {})
        tables, configs = header.get("components"), header.get("configs")
        return _hash(
            [
                tables.get(name) if isinstance(tables, Mapping) else None,
                configs.get(name) if isinstance(configs, Mapping) else None,
                sorted((r for r in rows if r.get("component") == name), key=str),
                [r for r in adapters if r.get("component") == name],
            ]
        )

    return GenerationMemo(
        torch=torch,
        environment=environment(type(model).__module__),
        components={name: weights(name) for name in roots},
        assets=_hash([assets.get("assets"), assets.get("configs")]),
        device=_hash([numerical, str(gpu), fusion, world]),
        sites={name: tuple(found) for name, found in processors.items()},
    )
