"""Executor-side adapter lifecycle, shared by the leader and follower ranks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from typing import Any

from cozy_runtime.author._context import AdapterRef
from cozy_runtime.internal import adapters, canonical, proctree
from cozy_runtime.internal.lora import AdapterRefusal, LinearAdapters


class AdapterModel:
    def __init__(
        self,
        components: Mapping[str, Any],
        snapshots: Mapping[str, str],
        compatibility: Sequence[Mapping[str, Any]],
        store: str,
    ) -> None:
        self.store = store
        self.snapshots = dict(snapshots)
        self.compatibility = tuple(compatibility)
        targets: dict[str, list[str]] = {}
        for row in compatibility:
            if (
                set(row) != {"component", "base_snapshot", "target", "shape", "dtype"}
                or snapshots.get(row["component"]) != row["base_snapshot"]
            ):
                raise AdapterRefusal("adapter_base_mismatch", "adapter target binding changed")
            targets.setdefault(row["component"], []).append(row["target"])
        self.linear = LinearAdapters(components, snapshots, targets)
        try:
            for row in compatibility:
                incoming, outgoing, dtype = self.linear._sites[row["component"], row["target"]]
                if [outgoing, incoming] != row["shape"] or str(dtype).removeprefix("torch.") != {
                    "f16": "float16",
                    "bf16": "bfloat16",
                    "f32": "float32",
                }.get(row["dtype"]):
                    raise AdapterRefusal(
                        "adapter_shape", "constructed target differs from adapter fit"
                    )
        except BaseException:
            self.linear.close()
            raise

    def inspect(self, selections: Sequence[Mapping[str, Any]]) -> adapters.AdapterGeometry:
        self.linear.check_ready()
        geometry = adapters.inspect(
            self.store,
            tuple(adapters.AdapterSelection.read(row) for row in selections),
            self.snapshots,
        )
        if geometry.compatibility != self.compatibility:
            raise AdapterRefusal(
                "adapter_target_mismatch", "adapter target set differs from construction"
            )
        return geometry


def validate_slots(models: Mapping[str, AdapterModel], selections: Mapping[str, Any]) -> None:
    if set(selections) - set(models):
        raise AdapterRefusal("adapter_slot", "adapter request names an unprepared model slot")
    for name in selections:
        if any(other != name and models[other] is models[name] for other in models):
            raise AdapterRefusal(
                "adapter_slot_alias",
                "two active model slots share one constructed root; "
                "adapter slots need independent roots",
            )


def check_host_capacity(
    geometry: Iterable[adapters.AdapterGeometry],
    *,
    device: Any,
    copies: int = 1,
    available: Callable[[], int] = proctree.available_host_bytes,
) -> int:
    if copies < 1:
        raise AdapterRefusal("adapter_host_shortfall", "rank source-copy count must be positive")
    rows = tuple(geometry)
    required = sum(row.source_bytes for row in rows) + max(
        (row.host_scratch_bytes for row in rows), default=0
    )
    if str(device).split(":", 1)[0] == "cpu":
        required += sum(row.device_bytes for row in rows) + max(
            (row.scratch_bytes for row in rows), default=0
        )
    required *= copies
    if required == 0:
        return 0
    capacity = available()
    if capacity < 0 or required > capacity:
        raise AdapterRefusal(
            "adapter_host_shortfall",
            f"{copies} rank source/validation copies need {required} bytes; "
            f"effective host availability is {capacity}",
        )
    return required


class AdapterAttempt:
    """Own all slot contexts and source leases for exactly one executor attempt."""

    def __init__(
        self,
        models: Mapping[str, AdapterModel],
        selections: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        device: Any,
        check_cancelled: Callable[[], None],
        check_capacity: Callable[[int], None],
        host_available: Callable[[], int] = proctree.available_host_bytes,
    ) -> None:
        self.stack = ExitStack()
        self.refs: dict[str, tuple[AdapterRef, ...]] = {}
        self.digest = canonical.digest(selections) if selections else ""
        self.source_bytes = self.device_bytes = 0
        try:
            validate_slots(models, selections)
            geometry = {name: models[name].inspect(rows) for name, rows in selections.items()}
            self.source_bytes = sum(row.source_bytes for row in geometry.values())
            self.device_bytes = sum(row.device_bytes for row in geometry.values())
            self.host_bytes = check_host_capacity(
                geometry.values(), device=device, available=host_available
            )
            check_capacity(
                self.device_bytes + max((row.scratch_bytes for row in geometry.values()), default=0)
            )
            for name, row in geometry.items():
                model = models[name]
                loaded = self.stack.enter_context(
                    adapters.load(
                        model.store,
                        row.selections,
                        model.snapshots,
                        check_cancelled=check_cancelled,
                    )
                )
                self.refs[name] = self.stack.enter_context(
                    model.linear.request(
                        loaded,
                        device=device,
                        max_device_bytes=row.device_bytes,
                        check_cancelled=check_cancelled,
                        check_capacity=check_capacity,
                    )
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            self.stack.close()
        finally:
            self.refs.clear()
