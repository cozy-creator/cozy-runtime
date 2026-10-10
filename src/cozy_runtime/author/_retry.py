"""Explicit contracts for replaying one operation after an allocation failure.

These are author promises, not purity inferred from Python attributes. Undeclared
operations still run, but the Runtime cannot replay them or promise reuse after failure.
"""

from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any
from weakref import ReferenceType, ref


def runtime_owned(value: object) -> bool:
    """Internal provenance predicate shared with Runtime's adapter lifecycle."""
    owner = getattr(value, "__cozy_op__", None)
    return isinstance(owner, ReferenceType) and owner() is value


def original_callable(value: object) -> object:
    """Unwrap only Runtime adapters, never arbitrary ``__wrapped__`` attributes."""
    while runtime_owned(value) and hasattr(value, "__cozy_bound__"):
        value = value.__cozy_bound__
    return value


def original_forward(module: Any) -> Any:
    """Read a module's forward beneath Runtime adapters, retaining foreign wrappers.

    An author composing a known forward replacement can verify its original before
    claiming ownership. This accessor does not itself declare purity or retry safety.
    """
    return original_callable(getattr(module, "forward", None))


def pure[F: Callable[..., Any]](fn: F) -> F:
    """Declare a callable (including a module) read-only apart from Runtime-managed RNG.

    For a module this covers its body and arguments; its hooks must also be pure or
    covered by ``retry_state``. Mutable containers and tensors passed as arguments are
    read-only under this promise. The marker never identifies a Runtime wrapper.
    """
    target: Any = fn.__func__ if isinstance(fn, MethodType) else fn
    target.__cozy_pure__ = ref(target)
    return fn


def is_declared_pure(value: object) -> bool:
    """Whether this callable itself carries an author purity declaration.

    This checks the declaration's identity, not the function's behavior. Bound methods
    use their underlying function. A wrapper copying metadata with functools.wraps
    does not inherit the original's declaration and must declare its own behavior.
    """
    target = value.__func__ if isinstance(value, MethodType) else value
    marker = getattr(target, "__cozy_pure__", None)
    return isinstance(marker, ReferenceType) and marker() is target


def plain_forward(module: Any) -> bool:
    """A known class implementation or explicitly declared replacement, below our adapters."""
    forward = original_forward(module)
    return is_declared_pure(forward) or (
        getattr(forward, "__self__", None) is module
        and getattr(forward, "__func__", None) is getattr(type(module), "forward", None)
    )


def retry_state(
    module: Any,
    save: Callable[[], Any],
    restore: Callable[[Any], None],
    *,
    tensors: Callable[[], Iterable[Any]] = lambda: (),
    generators: Callable[[], Iterable[Any]] = lambda: (),
    hooks: Iterable[object] = (),
    forwards: Iterable[object] = (),
) -> None:
    """Declare all mutable state of a module operation, including its hooks and caches.

    ``save`` runs once, after placement and before any hook; ``restore`` runs after every
    failed attempt, including terminal failure. They own Python state and container
    identity; the Runtime does not serialize object graphs. Both callbacks must be
    exception-safe and must not allocate device tensors. ``save`` must not mutate state.

    ``tensors`` names fixed-geometry mutable tensor state. The Runtime saves each distinct
    backing storage on the host and restores its bytes without replacing views. Tensor
    resizing or rebinding is outside this contract. Registered state pins its weight unit
    while the component lives, since checkpoint bytes cannot reconstruct mutable state.
    ``generators`` names RNGs held outside the call's arguments. All other state and
    arguments are read-only. A faulty restore makes the executor unusable.

    ``hooks`` explicitly names the hook callbacks or Diffusers registry hook objects
    whose state these callbacks restore. ``forwards`` names owned instance-forward
    replacements. These claims are captured at registration, never inferred from
    whatever hooks happen to exist later. Unclaimed mutations prevent retry even when
    the root has a custom contract; Runtime adapters and separately pure hooks remain
    eligible. A composing owner must preserve previous claims and add only its own.
    """
    module.__dict__["_cozy_retry_state"] = (save, restore)
    module.__dict__["_cozy_retry_tensors"] = tensors
    module.__dict__["_cozy_retry_generators"] = generators
    module.__dict__["_cozy_retry_hooks"] = tuple(hooks)
    module.__dict__["_cozy_retry_forwards"] = tuple(forwards)


def read_whole(module: Any) -> None:
    """The package reads this component's tensors outside its forward: keep it whole
    on the device while a method declaring it runs."""
    module.__dict__["_cozy_read_whole"] = True
