"""Runtime lifecycle adapter for native donor service, with no memory byte policy.

Imported by CPU-only Hello: no Torch import, CUDA query, participant or thread starts here.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorfs import varena


def capable() -> bool:
    owner = getattr(varena, "CooperationOwner", None)
    return owner is not None and getattr(owner, "capability", lambda: 0)() == 1


def initialize() -> None:
    """Metadata-only atfork setup before the import-only parent can start fork producers."""
    if capable():
        varena.CooperationOwner.initialize()


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
    """Runtime lifecycle adapter; the native library owns donor progress and retirement."""

    def __init__(
        self, device: int, config: Config, failed: Callable[[BaseException], None]
    ) -> None:
        if not capable():
            raise Failure("the selected cooperative device requires the native owner service")
        self.config = config
        self.pid = os.getpid()
        self._peer = varena.CooperationOwner(
            device, config.uuid, config.directory, shared=config.shared, priority=config.priority,
        )
        self._failed = failed
        self._reported = False
        self._closed = False

    def _process(self) -> None:
        if self.pid != os.getpid():
            raise Failure("a CUDA cooperation owner cannot be used after fork")

    def check(self) -> None:
        self._process()
        try:
            facts = self.facts()
        except Exception as cause:
            error = self._failure(str(cause))
            raise error from cause
        if facts["failed"] is not None:
            raise self._failure(str(facts["failed"]))
        if not facts["active"]:
            raise Failure("native memory cooperation is closed")

    def _failure(self, reason: str) -> Failure:
        error = Failure(f"native memory cooperation failed: {reason}")
        if not self._reported:
            self._reported = True
            self._failed(error)
        return error

    def wake(self, *, used: bool = False, check: bool = True) -> None:
        self._process()
        if self._closed:
            return
        try:
            self._peer.wake(used=used)
        except Exception as cause:
            error = self._failure(str(cause))
            if check:
                raise error from cause

    def facts(self) -> dict[str, Any]:
        self._process()
        return self._peer.facts()

    def close(self) -> None:
        self._process()
        if self._closed:
            return
        self._closed = True
        with suppress(varena.CooperationDeferredError):
            # Detaches the GIL; no Memory/Order/native lock is held. The native worker owns
            # final retirement and retains its intact Peer if capture currently excludes it.
            self._peer.close()
