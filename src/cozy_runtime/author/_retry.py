"""What a package states so the Runtime can run one op again after an out-of-memory error.

The Runtime restores an op's arguments, random states and the plain attributes of its
modules. Anything else a hook or forward keeps is invisible to it, so the op is not retried
unless the package states that the hook keeps nothing, or hands over its own save/restore.
"""

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def pure(fn: F) -> F:
    """Mark a forward hook or forward replacement as keeping no state a second run would see
    twice: the op it belongs to may be run again after an out-of-memory error."""
    fn.__cozy_pure__ = True  # type: ignore[attr-defined]
    return fn


def retry_state(module: Any, save: Callable[[], Any], restore: Callable[[Any], None]) -> None:
    """State of `module` that an op run again must find as it was: `save()` runs before each
    op of it, `restore(saved)` before a retry."""
    module.__dict__["_cozy_retry_state"] = (save, restore)


def read_whole(module: Any) -> None:
    """The package reads this component's tensors itself, outside its forward: it stays whole
    on the device while a method declaring it runs."""
    module.__dict__["_cozy_read_whole"] = True
