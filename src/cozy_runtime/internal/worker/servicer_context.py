"""The part of `grpc.ServicerContext` the worker's servicers use; grpcio ships no types."""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn, Protocol

import grpc


class ServicerContext(Protocol):
    def abort(self, code: grpc.StatusCode, details: str) -> NoReturn: ...

    def is_active(self) -> bool: ...

    def set_trailing_metadata(self, metadata: tuple[tuple[str, str], ...]) -> None: ...

    def add_callback(self, callback: Callable[[], None]) -> bool:
        """False when the RPC already ended: the callback will never run."""
