"""Resolve generic LoRA target containment and lease ordinary TensorFS snapshots."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

from cozy_runtime.internal.encoding import SPEC_PLAIN, TORCH_DTYPES
from cozy_runtime.internal.fill import Checkpoint, PlanRow
from cozy_runtime.internal.lora import TILE_ROWS, AdapterInput, AdapterRefusal
from cozy_runtime.protocol import documents

_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?\Z")
_SUFFIXES = ((".lora_A.weight", "a"), (".lora_B.weight", "b"), (".alpha", "alpha"))


@cache
def implementation_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256(b"cozy.runtime.linear-adapters/1\0")
    for name in ("adapters.py", "executor_adapters.py", "lora.py", "fusion_install.py"):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return "sha256:" + digest.hexdigest()


def scale_string(value: float) -> str:
    """The finite shortest decimal spelling used by Go FormatFloat(g, -1, 64)."""
    if not math.isfinite(value):
        raise AdapterRefusal("adapter_scale", "adapter strength must be finite")
    number = Decimal(repr(value)).normalize()
    sign, digits, exponent = number.as_tuple()
    assert isinstance(exponent, int)
    text = "".join(str(n) for n in digits)
    prefix = "-" if sign else ""
    if not number:
        return "0"
    power = len(text) + exponent - 1
    if power < -4 or power >= 6:
        fraction = "." + text[1:] if len(text) > 1 else ""
        return f"{prefix}{text[0]}{fraction}e{power:+03d}"
    return format(number, "f")


def parse_scale(text: str) -> float:
    if not isinstance(text, str) or len(text) > 32 or not _DECIMAL.fullmatch(text):
        raise AdapterRefusal("adapter_scale", "adapter strength must be a canonical decimal")
    value = float(text)
    if scale_string(value) != text:
        raise AdapterRefusal("adapter_scale", "adapter strength is not canonically spelled")
    return value


@dataclass(frozen=True, slots=True)
class AdapterSelection:
    component: str
    manifest: str
    source_component: str
    scale: str

    def document(self) -> dict[str, str]:
        return {
            "component": self.component,
            "manifest": self.manifest,
            "source_component": self.source_component,
            "scale": self.scale,
        }

    @classmethod
    def read(cls, row: Mapping[str, Any]) -> AdapterSelection:
        if set(row) != {"component", "manifest", "source_component", "scale"} or any(
            not isinstance(value, str) or not value for value in row.values()
        ):
            raise AdapterRefusal("adapter_selection", "adapter selection fields are incomplete")
        documents.raw(row["manifest"])
        parse_scale(row["scale"])
        return cls(**row)


@dataclass(frozen=True, slots=True)
class AdapterGeometry:
    selections: tuple[AdapterSelection, ...]
    targets: Mapping[str, tuple[str, ...]]
    compatibility: tuple[Mapping[str, Any], ...]
    source_bytes: int
    device_bytes: int
    scratch_bytes: int


def inspect(
    store: str | Path,
    selections: Sequence[AdapterSelection],
    snapshots: Mapping[str, str],
) -> AdapterGeometry:
    """Header-only containment before construction/device allocation; never infer a task.

    The selected base snapshot/component and logical target geometry join construction
    identity. Adapter payloads, ranks and strengths remain request binding facts.
    """
    bases: dict[str, dict[str, PlanRow]] = {}
    targets: dict[str, set[str]] = {}
    compatible: dict[tuple[str, str], dict[str, Any]] = {}
    source_bytes = device_bytes = scratch_bytes = 0
    for selection in selections:
        parse_scale(selection.scale)
        component = selection.component
        if component not in snapshots:
            raise AdapterRefusal("adapter_component", f"unknown selected component {component!r}")
        if component not in bases:
            base = Checkpoint(store, snapshots[component])
            bases[component] = {row.name: row for row in base.rows(component)}
        adapter = Checkpoint(store, selection.manifest)
        if adapter.header.get("configs") or adapter.header.get("assets"):
            raise AdapterRefusal(
                "adapter_contract_unsupported",
                "adapter has an unsupported configuration or inference contract",
            )
        rows = adapter.rows(selection.source_component)
        if not rows:
            raise AdapterRefusal("adapter_empty", f"{selection.manifest}: no adapter tensors")
        pairs: dict[str, dict[str, PlanRow]] = {}
        for row in rows:
            if (
                row.encoded.encoding != SPEC_PLAIN
                or row.encoded.roles != {"value"}
                or row.dtype not in {"f16", "bf16", "f32"}
                or row.encoded.part("value").shape != row.shape
                or row.encoded.part("value").dtype != row.dtype
            ):
                raise AdapterRefusal(
                    "adapter_tensor", f"{row.key}: factors must be plain floating tensors"
                )
            source_bytes += row.stored_nbytes
            for suffix, role in _SUFFIXES:
                if row.name.endswith(suffix):
                    pairs.setdefault(row.name[: -len(suffix)], {})[role] = row
                    break
            else:
                raise AdapterRefusal("adapter_key", f"unrecognized factor {row.key}")
        for path, pair in pairs.items():
            base_row = bases[component].get(path + ".weight")
            if base_row is None:
                raise AdapterRefusal(
                    "adapter_target_missing", f"{component}.{path}.weight is absent"
                )
            if not {"a", "b"} <= pair.keys():
                raise AdapterRefusal("adapter_pair", f"{component}.{path}: incomplete A/B pair")
            a, b = pair["a"], pair["b"]
            if (
                len(base_row.shape) != 2
                or len(a.shape) != 2
                or len(b.shape) != 2
                or a.shape[0] <= 0
                or a.shape[0] != b.shape[1]
                or (b.shape[0], a.shape[1]) != base_row.shape
                or a.dtype != b.dtype
                or base_row.dtype not in {"f16", "bf16", "f32"}
            ):
                raise AdapterRefusal(
                    "adapter_shape", f"{component}.{path}: factor geometry differs from base"
                )
            if "alpha" in pair and pair["alpha"].shape != ():
                raise AdapterRefusal("adapter_alpha", f"{component}.{path}: alpha is not scalar")
            targets.setdefault(component, set()).add(path)
            compatible[component, path] = {
                "component": component,
                "base_snapshot": snapshots[component],
                "target": path,
                "shape": list(base_row.shape),
                "dtype": base_row.dtype,
            }
            if parse_scale(selection.scale) != 0:
                elements = math.prod(a.shape) + math.prod(b.shape)
                width = 4 if base_row.dtype == "f32" else 2
                device_bytes += elements * width
                scratch_bytes = max(
                    scratch_bytes, TILE_ROWS * (a.shape[1] + a.shape[0] + b.shape[0]) * width
                )
    return AdapterGeometry(
        tuple(selections),
        {component: tuple(sorted(paths)) for component, paths in sorted(targets.items())},
        tuple(compatible[key] for key in sorted(compatible)),
        source_bytes,
        device_bytes,
        scratch_bytes,
    )


@contextmanager
def load(
    store: str | Path,
    selections: Sequence[AdapterSelection],
    snapshots: Mapping[str, str],
    *,
    check_cancelled: Callable[[], None],
) -> Iterator[tuple[AdapterInput, ...]]:
    """Copy plain factors under verified TensorFS leases held for the whole attempt."""
    import torch

    loaded: list[AdapterInput] = []
    values: dict[str, Any] = {}
    tensor = None
    with ExitStack() as stack:
        try:
            for selection in selections:
                check_cancelled()
                checkpoint = Checkpoint(store, selection.manifest)
                lease = stack.enter_context(checkpoint.acquire())
                values = {}
                for row in checkpoint.rows(selection.source_component):
                    check_cancelled()
                    if row.encoded.encoding != SPEC_PLAIN or row.encoded.roles != {"value"}:
                        raise AdapterRefusal("adapter_tensor", f"{row.key}: encoded adapter factor")
                    tensor = torch.empty(row.shape, dtype=getattr(torch, TORCH_DTYPES[row.dtype]))
                    lease.read_part_into(
                        checkpoint.header_bytes,
                        selection.source_component,
                        row.name,
                        "value",
                        0,
                        tensor.reshape(-1).view(torch.uint8).numpy(),
                    )
                    values[row.name] = tensor
                loaded.append(
                    AdapterInput(
                        selection.manifest,
                        selection.component,
                        snapshots[selection.component],
                        values,
                        parse_scale(selection.scale),
                    )
                )
            yield tuple(loaded)
        finally:
            for adapter in loaded:
                # The mapping is ours, not caller-provided; erase CPU copies even when
                # retained error diagnostics keep the AdapterInput object reachable.
                assert isinstance(adapter.tensors, dict)
                adapter.tensors.clear()
            loaded.clear()
            values.clear()
            tensor = None
