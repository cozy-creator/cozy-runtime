"""Immutable H3 window geometry, invocation order and output assembly.

This module makes no admission decision and grants no execution capability. Native
policy chooses one declared geometry. The coordinator supplies real queried plans,
physical buffer owners, fresh stage admissions and checked completion/retirement.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

GEOMETRY = (7, 16, 16)
TOKENS, WIDTH, BLOCKS = 1797, 2048, 36
PIXEL_SHAPE = (3, 28, 256, 256)


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def window_sizes(batch: int) -> tuple[int, ...]:
    """Legal balanced geometry only; no room, allocator or byte policy is consulted."""
    _positive(batch, "batch")
    sizes = [batch]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    return tuple(sizes)


@dataclass(frozen=True)
class Variant:
    batch: int
    outer: int
    inner: int

    def __post_init__(self) -> None:
        for value, name in ((self.batch, "batch"), (self.outer, "outer"), (self.inner, "inner")):
            _positive(value, name)
        if self.outer > self.batch or self.inner > self.outer:
            raise ValueError("window geometry exceeds its declared tile domain")

    @property
    def identity(self) -> str:
        return f"outer:{self.outer}/inner:{self.inner}/tail:{self.batch % self.outer}"

    def windows(self) -> tuple[Window, ...]:
        return tuple(Window(start, min(self.outer, self.batch - start), self.inner)
                     for start in range(0, self.batch, self.outer))


def variants(batch: int) -> tuple[Variant, ...]:
    """Full B first, then legal smaller windows. Native policy owns the ranking result."""
    return tuple(Variant(batch, outer, inner)
                 for outer in window_sizes(batch) for inner in window_sizes(outer))


def resolve(declarations: tuple[Variant, ...], identity: str) -> Variant:
    """Resolve an exact declared ID; this does not select/admit a geometry."""
    matches = tuple(variant for variant in declarations if variant.identity == identity)
    if len(matches) != 1:
        raise ValueError("native selection must name exactly one declared geometry")
    return matches[0]


@dataclass(frozen=True)
class Window:
    start: int
    count: int
    inner: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("window start must be a nonnegative integer")
        _positive(self.count, "window count")
        _positive(self.inner, "inner window")

    @property
    def stop(self) -> int:
        return self.start + self.count

    @property
    def block_chunk(self) -> int:
        # Ragged geometry is fixed in the declaration, not chosen from memory pressure.
        return min(self.inner, self.count)

    @property
    def block_tail(self) -> int:
        return self.count % self.block_chunk


@dataclass(frozen=True)
class Operation:
    window: Window
    kind: str
    block: int | None
    input_owner: str
    output_owner: str
    phase_role: str

    @property
    def identity(self) -> tuple[int, int, str, int | None]:
        return self.window.start, self.window.count, self.kind, self.block


def operations(variant: Variant) -> tuple[Operation, ...]:
    """One body per logical invocation; full-B geometry is strictly blockwise."""
    result = []
    for window in variant.windows():
        result.append(Operation(window, "post_quant", None, "input", "post_result", "edge"))
        result.append(Operation(window, "prefix", None, "post_result", "hidden16", "edge"))
        current = "hidden16"
        for index in range(BLOCKS):
            target = "hidden32_a" if index % 2 == 0 else "hidden32_b"
            result.append(Operation(window, "block", index, current, target, "supplied_block"))
            current = target
        result.append(Operation(window, "tail", None, current, "pixels", "tail_functional"))
    return tuple(result)


@dataclass(frozen=True)
class Lifetime:
    owner: str
    introduced: str
    last_use: str
    retire: str


def lifetimes() -> tuple[Lifetime, ...]:
    """Per-window local lifetimes plus whole-call input/output and query owners.

    Recreating retired locals for the next window needs fresh native admission of
    the same fixed geometry; it is not a new geometry choice or a replay.
    """
    return (
        Lifetime("input", "caller", "post_quant", "caller-owned; retain through its consuming stream"),
        Lifetime("pixels", "native allocation bootstrap", "output handoff", "last downstream owner"),
        Lifetime("post_result", "window allocation preparation", "prefix", "checked prefix completion"),
        Lifetime("hidden16", "window allocation preparation", "block:0", "checked block:0 completion"),
        Lifetime("rotary_cos_sin", "window allocation preparation", "block:35", "checked block:35 completion"),
        Lifetime("hidden32_a", "window allocation preparation", "block:35", "checked block:35 completion before tail admission"),
        Lifetime("hidden32_b", "window allocation preparation", "tail", "checked tail completion"),
        Lifetime("block_scratch", "window allocation preparation", "block:35", "checked block:35 completion before tail admission"),
        Lifetime("tail_norm_y_mean_rstd", "functional LayerNorm inside tail", "tail", "checked tail completion"),
        Lifetime("source_views", "one stage's ordinary source hooks/binding", "that stage", "existing checked source-use lifetime; never pin all 36"),
        Lifetime("query_bias_index_context_plans", "real provider preparation", "last borrowing invocation", "checked completion and final borrower release"),
    )


def tail_functional_allocations(window: Window) -> tuple[tuple[str, int], ...]:
    """Exact real functional outputs, independently admitted at the tail phase."""
    rows = window.count * TOKENS
    return (("norm_output", rows * WIDTH * 4), ("norm_mean", rows * 4), ("norm_rstd", rows * 4))


def pixel_allocation(batch: int) -> int:
    _positive(batch, "batch")
    return batch * math.prod(PIXEL_SHAPE) * 2


class OutputAssembly:
    """Views into the actual fixed returned owner; no allocation or hidden join copy."""
    def __init__(self, output: Tensor, batch: int) -> None:
        _positive(batch, "batch")
        if (type(output) is not Tensor or output.layout is not torch.strided
                or output.dtype is not torch.float16 or tuple(output.shape) != (batch, *PIXEL_SHAPE)
                or not output.is_contiguous() or output.is_conj() or output.is_neg()):
            raise ValueError("sequence output must be its declared contiguous FP16 owner")
        self.output, self.batch = output, batch

    def view(self, window: Window) -> Tensor:
        if window.stop > self.batch:
            raise ValueError("window exceeds the fixed output owner")
        return self.output.narrow(0, window.start, window.count)

    def validate_result(self, window: Window, result: Tensor) -> None:
        target = self.view(window)
        if (type(result) is not Tensor or result.device != target.device or result.dtype != target.dtype
                or tuple(result.shape) != tuple(target.shape) or tuple(result.stride()) != tuple(target.stride())
                or result.is_conj() or result.is_neg() or result.data_ptr() != target.data_ptr()):
            raise ValueError("tail result is not its exact assigned output view")


@dataclass(frozen=True)
class Ticket:
    operation: Operation
    position: int
    owner: object


class Cursor:
    """Host submission order only; never a CUDA completion or admission certificate."""
    def __init__(self, variant: Variant) -> None:
        self.operations = operations(variant)
        self.position = 0
        self.active: Ticket | None = None
        self.failure: BaseException | None = None
        self._owner = object()

    def claim(self) -> Ticket:
        if self.failure is not None or self.active is not None or self.position == len(self.operations):
            raise RuntimeError("sequence cannot repeat, overlap or resume a failed invocation")
        self.active = Ticket(self.operations[self.position], self.position, self._owner)
        return self.active

    def _check(self, ticket: Ticket) -> None:
        if ticket.owner is not self._owner or ticket is not self.active:
            raise RuntimeError("sequence ticket is not the current invocation")

    def submitted(self, ticket: Ticket) -> None:
        self._check(ticket)
        self.position += 1
        self.active = None

    def failed(self, ticket: Ticket, error: BaseException) -> None:
        self._check(ticket)
        self.failure = error
        self.active = None
        error.varena_no_retry = True  # type: ignore[attr-defined]

    @property
    def host_done(self) -> bool:
        return self.failure is None and self.active is None and self.position == len(self.operations)


@dataclass(frozen=True)
class EdgeAllocations:
    """Snapshot of a real prepared provider, not a substitute for its live owners."""
    stage: str
    preparation: tuple[tuple[str, int], ...]
    supplied: tuple[tuple[str, int], ...]
    during_execute: tuple[tuple[str, int], ...]
    workspace_bytes: int

    @property
    def all_execution_allocations(self) -> tuple[tuple[str, int], ...]:
        return self.supplied + self.during_execute


def edge_allocations(stage: str, prepared: Any) -> EdgeAllocations:
    """Require the actual queried edge preparation; unknown workspace is never zero."""
    from cozy_runtime.internal.execution import (
        native_vae_post_quant as post, native_vae_prefix as prefix, native_vae_output as tail,
    )
    types = {"post_quant": post.Prepared, "prefix": prefix.Prepared, "tail": tail.Prepared}
    if stage not in types or type(prepared) is not types[stage]:
        raise TypeError("sequence allocation metadata requires the actual prepared edge")
    if prepared.pid != os.getpid():
        raise RuntimeError("prepared edge metadata cannot be inherited after fork")
    from cozy_runtime._kernels import _dense
    if type(prepared.plan) is not _dense.Plan or type(prepared.context) is not _dense.Context:
        raise TypeError("sequence metadata requires a real native query and Context")
    info = prepared.plan.info()
    if info != prepared.info or info["bias_address"] != prepared.bias.data_ptr():
        raise ValueError("prepared edge query or bias owner changed")
    workspace = info["workspace_bytes"]
    description = prepared.description
    if stage == "tail":
        supplied = description.preallocations(workspace)
        functional = description.functional_allocations
    else:
        supplied, functional = description.allocations(workspace), ()
    preparation = (("bias", prepared.bias.numel() * prepared.bias.element_size()),)
    return EdgeAllocations(stage, preparation, supplied, functional, workspace)
