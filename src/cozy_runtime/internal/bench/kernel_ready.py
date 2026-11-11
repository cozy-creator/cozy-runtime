"""Kernel readiness publication under two real openers of one kernel (`cozy_runtime.bench.kernel_ready`; the release
gate's `kernel-ready-*` cells). In a request `ready` opens a kernel on a background thread, and an explicit `wait` may
open the same slot at once (attention's `_opened` beside `_waited`). CPU only: each opener is a real Torch matrix
product, a failure a real read of a file that does not exist, and CUDA stays uninitialized.

- `late-success`: a wait makes the kernel serve, then the older background opener succeeds: the same kernel still
  serves and the epoch does not move.
- `late-error`: as above, but the background opener's read fails: the kernel stops serving, `failed()` is that
  `FileNotFoundError`, the epoch moves once.
- `dropped-slot`: the waiter that opened the slot fails and drops it while a second waiter succeeds: the second's
  kernel serves, the epoch moves once.

This proves registry publication only: not CUDA module lifetime, and not the cause of any GPU crash."""

from __future__ import annotations

import tempfile
import threading
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import torch

import cozy_runtime
from cozy_runtime.author import _request_path as request_path
from cozy_runtime.internal import kernel_ready

MODES = ("late-success", "late-error", "dropped-slot")


class Facts(TypedDict, total=False):
    """What a mode saw."""

    openers: int
    ready_before: bool
    ready_after: bool
    same_kernel_after: bool
    failure: str | None
    first_waiter_failure: str | None
    epoch_before: int
    epoch_after: int
    mode: str
    runtime: str
    torch: str
    exact: bool
    cuda_initialized: bool


def _late(name: str, mode: str, product: Callable[[], torch.Tensor], missing: Path) -> Facts:
    """A background opener finishes after an explicit wait made the kernel serve."""
    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    background: list[threading.Thread] = []

    def opener(wait: bool) -> torch.Tensor:
        value = product()
        if threading.get_ident() != owner:
            background.append(threading.current_thread())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("the background opener was never released")
            if mode == "late-error":
                missing.read_bytes()
        return value

    try:
        if kernel_ready.ready(name, opener) is not None:
            raise RuntimeError("in a request it opens in the background")
        if not entered.wait(5):
            raise RuntimeError("the background opener did not start")
        explicit = kernel_ready.wait(name, opener)
        if cast("torch.Tensor | None", kernel_ready.ready(name)) is not explicit:
            raise RuntimeError("the explicit wait's kernel does not serve")
        before = kernel_ready.epoch()
        release.set()
        background[0].join(5)
        if background[0].is_alive():
            raise RuntimeError("the background opener did not finish")
        after: torch.Tensor | None
        after, failure = kernel_ready.ready(name), kernel_ready.failed(name)
        return {
            "openers": 2,
            "ready_before": True,
            "ready_after": after is not None,
            "same_kernel_after": after is explicit,
            "failure": type(failure).__name__ if failure else None,
            "epoch_before": before,
            "epoch_after": kernel_ready.epoch(),
        }
    finally:
        release.set()


def _dropped(name: str, product: Callable[[], torch.Tensor], missing: Path) -> Facts:
    """The waiter that opened the slot fails while a second waiter of it succeeds."""
    entered, fail = threading.Event(), threading.Event()
    first: list[BaseException] = []

    def failing(wait: bool) -> torch.Tensor:
        product()
        entered.set()
        if not fail.wait(5):
            raise RuntimeError("the first waiter was never let fail")
        missing.read_bytes()
        return product()

    def first_waiter() -> None:
        try:
            kernel_ready.wait(name, failing)
        except FileNotFoundError as exc:
            first.append(exc)

    thread = threading.Thread(target=first_waiter, name="bench-kernel-ready first waiter")
    thread.start()

    def opener(wait: bool) -> torch.Tensor:
        value = product()
        fail.set()
        thread.join(5)
        if thread.is_alive():
            raise RuntimeError("the first waiter did not fail")
        return value

    try:
        if not entered.wait(5):
            raise RuntimeError("the first waiter did not start")
        before = kernel_ready.epoch()
        served = kernel_ready.wait(name, opener)
        after: torch.Tensor | None
        after, failure = kernel_ready.ready(name), kernel_ready.failed(name)
        return {
            "openers": 2,
            "first_waiter_failure": type(first[0]).__name__ if first else None,
            "ready_after": after is not None,
            "same_kernel_after": after is served,
            "failure": type(failure).__name__ if failure else None,
            "epoch_before": before,
            "epoch_after": kernel_ready.epoch(),
        }
    finally:
        fail.set()
        thread.join(5)


def check(mode: str, seed: int) -> Facts:
    """`mode` in this request; raises naming what went wrong."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {MODES}")
    if not request_path.in_flight():
        raise RuntimeError("needs the real Cozy request scope")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left, right = (torch.randn(64, 64, generator=generator) for _ in range(2))
    expected = torch.mm(left, right)
    name = f"bench-kernel-ready {mode} {seed} {uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="cozy-kernel-ready-") as tmp:
        missing = Path(tmp) / "never-written"
        product = partial(torch.mm, left, right)
        facts = _dropped(name, product, missing) if mode == "dropped-slot" else _late(name, mode, product, missing)
    served: torch.Tensor | None = kernel_ready.ready(name)
    facts.update({
        "mode": mode,
        "runtime": cozy_runtime.__version__,
        "torch": str(torch.__version__),
        "exact": served is not None and torch.equal(served, expected),
        "cuda_initialized": torch.cuda.is_initialized(),
    })  # fmt: skip
    grew = facts["epoch_after"] - facts["epoch_before"]
    if facts["cuda_initialized"]:
        ok, why = False, "CUDA was initialized"
    elif mode == "late-success":
        ok = facts["same_kernel_after"] and facts["exact"] and grew == 0 and not facts["failure"]
        why = "an older background success replaced or demoted the serving kernel"
    elif mode == "late-error":
        ok = not facts["ready_after"] and facts["failure"] == "FileNotFoundError" and grew == 1
        why = "a late real opener failure did not stop the kernel serving, or the epoch missed it"
    else:
        ok = (
            facts["first_waiter_failure"] == "FileNotFoundError"
            and facts["same_kernel_after"]
            and facts["exact"]
            and grew == 1
            and not facts["failure"]
        )
        why = "a waiter's verified kernel did not serve after the slot's opener failed"
    if not ok:
        raise RuntimeError(f"{why}: {facts}")
    return facts
