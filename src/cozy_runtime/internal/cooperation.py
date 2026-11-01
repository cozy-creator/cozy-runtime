"""Runtime lifecycle adapter for native donor service, with no memory byte policy.

Imported by CPU-only Hello: no Torch import, CUDA query, participant or thread starts here.
"""
from __future__ import annotations

import contextvars
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorfs import varena


def capable() -> bool:
    owner = getattr(varena, "Cooperation", None)
    return owner is not None and getattr(owner, "capability", lambda: 0)() == 1


def initialize() -> None:
    """Metadata-only atfork setup before the import-only parent can start fork producers."""
    if capable():
        varena.Cooperation.initialize()


@dataclass(frozen=True)
class Config:
    uuid: str
    directory: Path
    shared: bool
    priority: int

    @classmethod
    def from_wire(cls, value: Any) -> Config | None:
        if value is None:
            return None
        directory = Path(value.directory)
        if not directory.is_absolute():
            raise ValueError("cooperation requires an explicit absolute shared tmpfs directory")
        return cls(value.uuid, directory, value.shared, value.priority)


class Failure(RuntimeError):
    # A background coordination failure cannot replay partially executed package code.
    varena_no_retry = True


class Owner:
    """One guarded native participant, plus a mechanical wake/service loop for its Memory."""

    def __init__(
        self, device: int, config: Config, failed: Callable[[BaseException], None]
    ) -> None:
        if not capable():
            raise Failure("the selected cooperative device requires the native owner capability")
        self.config = config
        self.pid = os.getpid()
        self._peer: Any = varena.Cooperation(
            device, config.uuid, config.directory, shared=config.shared, priority=config.priority,
        )
        self._failed = failed
        self._changed = threading.Condition()
        self._generation = 1
        self._used_at = 0
        self._stopping = False
        self._closed = False
        self._error: BaseException | None = None
        self._published = self._serviced = self._deferred = 0
        self._physical = 0
        self._last_donation: dict[str, Any] | None = None
        self._thread = threading.Thread(target=contextvars.Context().run, args=(self._run,),
                                        name="varena-owner", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._peer.close()
            raise

    def _process(self) -> None:
        if self.pid != os.getpid():
            raise Failure("a CUDA cooperation owner cannot be used after fork")

    def check(self) -> None:
        self._process()
        with self._changed:
            error = self._error
            closed = self._stopping or self._closed
        if error is not None:
            raise Failure(f"native memory cooperation failed: {error}") from error
        if closed:
            raise Failure("native memory cooperation is closed")

    def wake(self, *, used: bool = False) -> None:
        self._process()
        with self._changed:
            if self._stopping:
                return
            if used:
                self._used_at = time.monotonic_ns()
            self._generation += 1
            self._changed.notify_all()

    def facts(self) -> dict[str, Any]:
        self._process()
        with self._changed:
            return {"active": not self._stopping and not self._closed and self._error is None,
                    "published": self._published, "serviced": self._serviced,
                    "deferred": self._deferred, "last_donation": self._last_donation,
                    "physical_bytes": self._physical,
                    "failed": None if self._error is None else str(self._error)}

    def _run(self) -> None:
        generation = 0
        while True:
            with self._changed:
                def changed(seen: int = generation) -> bool:
                    return self._stopping or self._generation != seen

                self._changed.wait_for(changed, timeout=0.5)
                if self._stopping:
                    return
                generation, used_at = self._generation, self._used_at
                self._used_at = 0
            # All Python owner locks are gone. Native bindings detach the GIL, and native
            # service contains no Python callback or context/event wait.
            try:
                published = self._peer.publish(last_use_ns=used_at)
                with self._changed:
                    self._published += 1
                    self._physical = int(published["physical"]["total"])
                    self._deferred += int(not published["refreshed"])
                # This use timestamp was published even if the following service defers.
                used_at = 0
                serviced = self._peer.service()
                with self._changed:
                    self._serviced += 1
                    if serviced["round"] is not None:
                        self._last_donation = serviced
                    self._deferred += int(serviced["deferred"])
                    if serviced["accounting"] is not None:
                        self._physical = int(serviced["accounting"]["physical"]["total"])
            except varena.CooperationDeferredError:
                # Fresh native observation next tick; never relabel old numbers with a new
                # sample generation. Preserve an actual-use notification until it publishes.
                with self._changed:
                    self._used_at = max(self._used_at, used_at)
                    self._deferred += 1
            except BaseException as error:
                with self._changed:
                    self._error = error
                    self._stopping = True
                    self._changed.notify_all()
                self._failed(error)  # Runtime-owned metadata/error reporting, not a cache callback.
                return

    def close(self) -> None:
        self._process()
        with self._changed:
            if self._closed:
                return
            self._stopping = True
            self._changed.notify_all()
        # No Memory/Order/native/shared lock is held; join releases the Python GIL. This
        # pump never needs a command-order ticket or Python owner action to finish.
        self._thread.join()
        try:
            self._peer.close()
        except varena.CooperationDeferredError:
            # The native Python object's existing cleanup queue retains/retries its intact
            # owner through capture. Never force an unsafe teardown to finish shutdown.
            self._peer = None
        with self._changed:
            self._closed = True
