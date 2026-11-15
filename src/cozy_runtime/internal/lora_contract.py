"""The typed prepared LoRA graph shared by metadata readers and model executors."""

from __future__ import annotations

import math

import msgspec

from cozy_runtime.author._context import AdapterRef
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal.model_config import ADAPTERS

GRAPH_CONFIG = ADAPTERS
FORMAT = "cozy.model.lora/1"
#: A graph with a DoRA layer (TensorD's `DORA_FORMAT`): a Runtime reading only `FORMAT` refuses it.
DORA_FORMAT = "cozy.model.lora/2"
#: A graph with a LoKr layer, DoRAs or not (TensorD's `KRON_FORMAT`): older Runtimes refuse it.
KRON_FORMAT = "cozy.model.lora/3"
#: A LoKr's factor roles: `W1` and `W2`, each whole or an `_a @ _b` pair.
KRON_ROLES = ("w1", "w2", "w1_a", "w1_b", "w2_a", "w2_b")
CAPABILITY = "peft_graph/1"
_DTYPES = {"F16", "BF16", "F32"}


def _refuse(code: str, detail: str) -> CapabilityError:
    return CapabilityError(detail, code="adapter_" + code)


class Kron(msgspec.Struct, frozen=True):
    """A LoKr layer's update (LyCORIS' Kronecker adapter): `kron(W1, W2)`, each factor grafted as
    `<target>.lokr_<role>.<adapter>.weight` with its `[rows, cols]` (`factors`), and the runs
    `[start, stop)` of its rows the target takes, in order (`rows`; none: all of them)."""

    factors: dict[str, tuple[int, int]]
    rows: tuple[tuple[int, int], ...] = ()

    def dims(self) -> tuple[int, int, int, int]:
        """`W1` `[a, b]` and `W2` `[c, d]` as `(a, b, c, d)`."""
        f = self.factors
        w1 = f["w1"] if "w1" in f else (f["w1_a"][0], f["w1_b"][1])
        w2 = f["w2"] if "w2" in f else (f["w2_a"][0], f["w2_b"][1])
        return (*w1, *w2)


class Layer(msgspec.Struct, frozen=True):
    """One PEFT layer's (a linear's or a 2-D convolution's) construction geometry and ordered
    scaling. A DoRA layer (`dora`, a linear's) also holds its magnitude `m` `[out]`, grafted as
    `<target>.lora_magnitude_vector.<adapter>.weight`. A LoKr layer (`kron`, a linear's) holds
    its Kronecker factors in place of `lora_A`/`lora_B`, scaled by `alpha / rank` too."""

    component: str
    target: str
    adapter: str
    rank: int
    alpha: float
    strength: float
    dtype: str
    dora: bool = False
    kron: Kron | None = None


class Graph(msgspec.Struct, frozen=True):
    format: str
    layers: tuple[Layer, ...]
    adapters: tuple[AdapterRef, ...] = ()


def read(data: bytes) -> Graph:
    """Read a prepared graph's semantic version; added metadata remains harmless."""
    try:
        graph = msgspec.json.decode(data, type=Graph)
    except msgspec.DecodeError as exc:
        raise _refuse("graph", "malformed prepared LoRA graph") from exc
    if graph.format not in (FORMAT, DORA_FORMAT, KRON_FORMAT):
        raise _refuse("graph", f"unsupported prepared LoRA graph {graph.format!r}")
    kron = any(row.kron is not None for row in graph.layers)
    want = KRON_FORMAT if kron else DORA_FORMAT if any(row.dora for row in graph.layers) else FORMAT
    if graph.format != want:
        raise _refuse(
            "graph", f"{graph.format}: DoRAs come in {DORA_FORMAT}, LoKr layers in {KRON_FORMAT}"
        )
    expected = {f"adapter_{index}": reference for index, reference in enumerate(graph.adapters)}
    used: set[str] = set()
    for row in graph.layers:
        reference = expected.get(row.adapter)
        if reference is None or (reference.component, reference.scale) != (
            row.component,
            row.strength,
        ):
            raise _refuse("graph", "prepared layer differs from the ordered adapter selection")
        used.add(row.adapter)
        if (
            not row.component
            or not row.target
            or not row.adapter
            or row.rank <= 0
            or row.dtype not in _DTYPES
            or not math.isfinite(row.alpha)
            or not math.isfinite(row.strength)
            or not math.isfinite(row.alpha / row.rank * row.strength)
            or (row.kron is not None and not _kron_valid(row))
        ):
            raise _refuse("graph", "invalid prepared LoRA layer")
    if used != {name for name, reference in expected.items() if reference.scale != 0}:
        raise _refuse("graph", "prepared graph omits an active adapter")
    return graph


def _kron_valid(row: Layer) -> bool:
    """A LoKr layer whose factors are W1 and W2, each whole or a pair of one inner size, and whose
    rows are runs inside `kron(W1, W2)`'s; never a DoRA."""
    kron = row.kron
    assert kron is not None
    roles = set(kron.factors)
    pairs = all(
        (w in roles) != ({f"{w}_a", f"{w}_b"} <= roles)
        and not (w in roles and roles & {f"{w}_a", f"{w}_b"})
        for w in ("w1", "w2")
    )
    if row.dora or not roles <= set(KRON_ROLES) or not pairs:
        return False
    if any(len(shape) != 2 or min(shape) <= 0 for shape in kron.factors.values()):
        return False
    for w in ("w1", "w2"):
        if f"{w}_a" in roles and kron.factors[f"{w}_a"][1] != kron.factors[f"{w}_b"][0]:
            return False
    a, _, c, _ = kron.dims()
    return all(0 <= start < stop <= a * c for start, stop in kron.rows)
