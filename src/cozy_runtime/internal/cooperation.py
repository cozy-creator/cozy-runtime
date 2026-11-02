"""Runtime lifecycle adapter for native donor service, with no memory byte policy.

Imported by CPU-only Hello: no Torch import, CUDA query, participant or thread starts here.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorfs.varena import _runtime as varena


def capable() -> bool:
    return (
        varena.CooperationOwner.capability() == 1
        and varena.CooperationOwner.pressure_capability() == 1
        and varena.CooperationOwner.custody_capability() == 1
    )


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

    @classmethod
    def for_rank(cls, values: Sequence[Any], rank: int, degree: int) -> Config | None:
        """Validate the complete rank inventory before any device or follower starts."""
        if not values:
            return None
        if degree < 1 or len(values) != degree or not 0 <= rank < degree:
            raise ValueError("cooperation must name exactly one owner per GPU rank")
        configs = tuple(cls.from_wire(value) for value in values)
        if any(config is None for config in configs):
            raise ValueError("cooperation cannot omit a GPU rank")
        identities = [config.uuid.casefold() for config in configs if config is not None]
        if len(set(identities)) != degree:
            raise ValueError("cooperation GPU identities must be distinct")
        return configs[rank]


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
        self.config = config  # Negotiated Start intent, not a library creation default.
        self.device = device
        self.pid = os.getpid()
        peer = varena.CooperationOwner(
            device, config.uuid, config.directory, shared=config.shared, priority=config.priority,
        )
        try:
            table = peer.snapshot()
            facts = peer.facts()
            row = next((row for row in table["rows"] if row["id"] == table["owner"]), None)
            if (
                row is None or row["pid"] != self.pid or facts["owner"] != table["owner"]
                or not facts["active"] or facts["closed"]
            ):
                raise Failure("native cooperation did not retain this process's live service row")
            if (
                row["sharing"] != ("Shared" if config.shared else "Dedicated")
                or row["priority"] != config.priority
            ):
                raise Failure(
                    "existing native cooperation configuration differs from negotiated Start: "
                    f"row={row['id']} actual={row['sharing']}/{row['priority']} "
                    f"requested={'Shared' if config.shared else 'Dedicated'}/{config.priority}"
                )
            peer.retain_working_lifetime()
            retained = peer.facts()
            if not retained["active"] or retained["closed"] or retained["passive_holds"] == 0:
                raise Failure("native cooperation did not retain Working lifetime")
        except BaseException as error:
            # The constructor acquired only one client lease. Never overwrite a surviving
            # consumer's settings or publish a partially initialized Runtime Memory.
            try:
                peer.close()
            except varena.CooperationDeferredError:
                pass  # The native worker retains the final retirement ticket/resources.
            except Exception as cleanup:
                error.add_note(f"releasing the new cooperation lease also failed: {cleanup}")
            raise
        self._peer = peer
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

    def request_boundary(self, *, begin: bool, check: bool = True) -> None:
        self._process()
        if self._closed:
            return
        if self._peer.pressure_capability() != 1:
            raise Failure("native cooperation has an unsupported pressure protocol")
        try:
            self._peer.request_boundary(begin=begin)
        except Exception as cause:
            error = self._failure(str(cause))
            if check:
                raise error from cause

    def close(self) -> None:
        self._process()
        if self._closed:
            return
        self._closed = True
        with suppress(varena.CooperationDeferredError):
            # Detaches the GIL; no Memory/Order/native lock is held. The native worker owns
            # final retirement and retains its intact Peer if capture currently excludes it.
            self._peer.close()
