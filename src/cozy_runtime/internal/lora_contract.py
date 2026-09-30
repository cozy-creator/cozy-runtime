"""The typed prepared LoRA graph shared by metadata readers and model executors."""

from __future__ import annotations

import math
import msgspec

from cozy_runtime.author import AdapterRef, CapabilityError
from cozy_runtime.internal.model_config import ADAPTERS

GRAPH_CONFIG = ADAPTERS
FORMAT = "cozy.model.lora/1"
CAPABILITY = "peft_graph/1"
_DTYPES = {"f16", "bf16", "f32"}


def _refuse(code: str, detail: str) -> CapabilityError:
    return CapabilityError(detail, code="adapter_" + code)


class Linear(msgspec.Struct, frozen=True):
    """One ordinary PEFT layer's construction geometry and ordered scaling."""

    component: str
    target: str
    adapter: str
    rank: int
    alpha: float
    strength: float
    dtype: str


class Graph(msgspec.Struct, frozen=True):
    format: str
    layers: tuple[Linear, ...]
    adapters: tuple[AdapterRef, ...] = ()


def read(data: bytes) -> Graph:
    """Read a prepared graph's semantic version; added metadata remains harmless."""
    try:
        graph = msgspec.json.decode(data, type=Graph)
    except msgspec.DecodeError as exc:
        raise _refuse("graph", "malformed prepared LoRA graph") from exc
    if graph.format != FORMAT:
        raise _refuse("graph", f"unsupported prepared LoRA graph {graph.format!r}")
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
        ):
            raise _refuse("graph", "invalid prepared LoRA layer")
    if used != {name for name, reference in expected.items() if reference.scale != 0}:
        raise _refuse("graph", "prepared graph omits an active adapter")
    return graph
