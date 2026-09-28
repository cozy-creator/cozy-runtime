"""Overlap independent Model calls where the group's placement lets them run at once."""

from __future__ import annotations

import contextvars
import functools
import threading
from collections.abc import Callable
from typing import Any

from cozy_runtime.author._model import Model


def concurrently(*calls: Callable[[], Any]) -> tuple[Any, ...]:
    """Run independent calls; the results, in call order, once every call has finished.

    The author's statement: no call reads what another writes before this returns, and each
    returns its result rather than publishing it through shared state. A call that is a
    public Model method (bound, or under `functools.partial`) whose declared components the
    runtime keeps on other GPUs of the group holds nothing on this device, so it runs on its
    own thread beside the rest. Every other call runs here, in the order given. Where nothing
    is hosted elsewhere, that is exactly the plain sequence of calls.

    A failure stops the calls not yet started here; the running ones finish, and the first
    failure in call order is raised.
    """
    results: list[Any] = [None] * len(calls)
    failures: list[BaseException | None] = [None] * len(calls)
    beside: list[threading.Thread] = []
    local: list[tuple[int, Callable[[], Any]]] = []
    for index, call in enumerate(calls):
        carried = _remote(call)
        if carried is None:
            local.append((index, call))
            continue
        thread = threading.Thread(
            target=contextvars.copy_context().run,
            args=(_capture, carried, results, failures, index),
            name=f"cozy-concurrent-{index}",
            daemon=True,
        )
        thread.start()
        beside.append(thread)
    try:
        for index, call in local:
            _capture(call, results, failures, index)
            if failures[index] is not None:
                break
    finally:
        for thread in beside:
            thread.join()
    failure = next((failure for failure in failures if failure is not None), None)
    if failure is not None:
        raise failure
    return tuple(results)


def _capture(
    call: Callable[[], Any], results: list[Any], failures: list[BaseException | None], index: int
) -> None:
    try:
        results[index] = call()
    except BaseException as exc:
        failures[index] = exc


def _remote(call: Callable[[], Any]) -> Callable[[], Any] | None:
    """`call` carried to another thread when it is a Model method whose scope is remote."""
    func: Any = call
    while isinstance(func, functools.partial):
        func = func.func
    owner = getattr(func, "__self__", None)
    declared = getattr(func, "__uses_components__", None)
    if not isinstance(owner, Model) or not isinstance(declared, tuple):
        return None
    placement = owner._cozy_placement
    if placement is None or not placement.remote(declared):
        return None
    return placement.carry(call, declared)
