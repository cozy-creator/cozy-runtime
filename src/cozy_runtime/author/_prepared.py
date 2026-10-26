"""Pure declarations for prepared computation submitted through Context.plan.

The implementation belongs to the public cozy_runtime.compute facade, not the author plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cozy_runtime.author._context import Context


@dataclass(frozen=True, slots=True)
class Progress:
    completed_steps: int
    canceled: bool
    done: bool


class Execution(Protocol):
    def completion_proven(self) -> bool: ...
    def poll(self) -> Progress: ...
    def cancel(self) -> None: ...
    def wait(self) -> tuple[object, ...]: ...
    def close(self) -> None: ...


@runtime_checkable
class PreparedWork(Protocol):
    def submit(self, context: Context | None = None) -> Execution: ...
