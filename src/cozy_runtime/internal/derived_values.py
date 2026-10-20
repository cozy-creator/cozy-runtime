"""Explicit owners for upstream derived values that are not registered buffers.

This is an exact adapter, not a scan of arbitrary tensor-valued attributes. Diffusers'
QwenImage21Rope keeps its three immutable, constructor-derived tables in `freqs` and lazily
moves them in forward. The normal cache policy can count and reclaim those device bytes;
the unchanged upstream constructor recreates them when another call needs them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensorfs.varena.torch import Memory
    from torch.nn import Module


class DerivedValues:
    def __init__(
        self,
        memory: Memory,
        root: Module,
        component: str,
        torch: ModuleType,
        owners: list[Any],
        retire: Callable[[], None],
    ) -> None:
        self.memory, self.torch, self.owners, self.retire = memory, torch, owners, retire
        self.component = component
        self.cache = memory.cache(f"derived:{component}", self.shed)
        self.lock = threading.RLock()
        self.handles: list[Any] = []
        # Root leases protect eager component calls. Residency pins additionally lease
        # these bytes during replay, which bypasses forward hooks. Inner leases also
        # cover a direct call to the known upstream helper.
        seen: set[int] = set()
        for module in (root, *owners):
            if id(module) in seen:
                continue
            seen.add(id(module))
            self._hooks(module)

    @classmethod
    def install(
        cls,
        memory: Memory,
        root: Module,
        component: str,
        torch: ModuleType,
        retire: Callable[[], None],
    ) -> DerivedValues | None:
        owners = [
            module
            for module in root.modules()
            if (type(module).__module__, type(module).__name__)
            == ("diffusers.models.transformers.transformer_qwenimage21", "QwenImage21Rope")
        ]
        return cls(memory, root, component, torch, owners, retire) if owners else None

    def _hooks(self, module: Module) -> None:
        local = threading.local()
        helper = any(module is owner for owner in self.owners)

        def start(*_: object) -> None:
            calls: list[Any] = local.__dict__.setdefault("calls", [])
            calls.append(None)
            lease = self.cache.serving()
            lease.__enter__()
            if helper:
                self.lock.acquire()
            calls[-1] = lease
            with self.lock:
                for owner in self.owners:
                    if not owner.freqs:
                        # No checkpoint supplies these values. Run the original constructor,
                        # on the host as at load; upstream forward retains its lazy .to(device).
                        with self.torch.device("cpu"):
                            fresh = type(owner)(theta=owner.theta, axes_dim=owner.axes_dim)
                        owner.freqs = fresh.freqs

        def land(*_: object) -> None:
            calls = getattr(local, "calls", [])
            if calls and (lease := calls.pop()) is not None:
                try:
                    self._publish()
                finally:
                    if helper:
                        self.lock.release()
                    lease.__exit__(None, None, None)

        self.handles.append(module.register_forward_pre_hook(start, prepend=True))
        self.handles.append(module.register_forward_hook(land, always_call=True))

    def _publish(self) -> None:
        with self.lock:
            nbytes = 0
            for owner in self.owners:
                for value in owner.freqs:
                    if value.device == self.memory.device:
                        if self.memory.cuda:
                            # The last Python call can return before its CUDA reads finish.
                            # Protect the exact tables used, before another helper call can
                            # replace their list, including reads on a side stream.
                            value.record_stream(self.torch.cuda.current_stream(self.memory.device))
                        nbytes += value.numel() * value.element_size()
            self.cache.nbytes = nbytes

    @contextmanager
    def serving(self) -> Iterator[None]:
        """A graph's persistent pin also reads these addresses when Python is bypassed."""
        with self.cache.serving():
            try:
                yield
            finally:
                self._publish()

    def shed(self, _: int) -> int:
        nbytes = self.cache.nbytes
        # A recording can read the same table addresses without running Python. Drop its
        # existing residency pin before replacing them; active root leases exclude this path.
        self.retire()
        for owner in self.owners:
            owner.freqs = []
        self.cache.nbytes = 0
        return nbytes

    def close(self) -> None:
        if not self.cache.closed:
            self.shed(self.cache.nbytes)
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            self.cache.close()
