"""Step caching for a diffusion transformer's block stack: a forward whose stack would change
little reuses the stack's cached residual instead of running its blocks.

It is approximate by construction, so a package turns it on per call, for a preset that trades
quality for speed, never by default. Declare the stack at module scope, install it in the
execution plan, and name the policy where the denoise loop runs:

    STACK = stepcache.Stack(MyTransformer, blocks="transformer_blocks", carried=(0,))

    def execution_plan(self, plan):
        return stepcache.install(plan, STACK, self.transformer)

    denoise_loop(step, steps, state=(latents,), cache=stepcache.FirstBlockCache(0.08))

`carried` names the block arguments its output returns, in order: a positional index or a
keyword (`(0,)` for a single-stream block, `("encoder_hidden_states", "hidden_states")` for a
dual-stream one). The stack runs as one `for` over its list, in the transformer's forward or in
the step; a skipped forward's iteration ends after block 0, so not inside `zip(..., strict=True)`.
Embeddings and the final layer run every forward; a skipped forward runs none of the stack's
blocks after its first, so their weights, AdaLN and carried state stay as the last computed
forward left them. State starts with every call. Each forward's verdict comes from fixed-order
reductions on the device, so the same inputs and policy give the same verdicts and bytes. The
loop's record carries `cache` (its forwards, computed and skipped).

- `FirstBlockCache(t)` (NVIDIA sol-engine `first_block_cache.py`): block 0 runs, `r0 = h0 - x`;
  the rest run when there is no reference yet or `Σ|r0 - r0_ref| > t·Σ|r0_ref|`, and then
  `r0_ref = r0`, `R = y - h0`; else `y = h0 + R`.
- `EasyCache(t, retain, cooldown, max_hits)` (sol-engine `easycache.py`): a forward computes
  while `i < retain`, from `N - cooldown` (`N` the call's last forward), before its rate `k` exists
  and after `max_hits` skips in a row; else it adds `k·mean|x - x_prev| / mean|Y_c|` to `acc` and
  computes once `acc ≥ t`. A computed forward sets `k = mean|y - Y_c| / mean|x - X_c|`,
  `X_c = x`, `Y_c = y`, `R = y - x` and clears `acc`; a skip is `y = x + R`.

Its state lives in device buffers updated in place. An eager step reads each forward's verdict on
the host once; a recorded step keeps it on the device: the step's graph holds both branches, the
rest of the stack and the cached residual, and the verdict picks one (`internal.loop`'s cut). Both
run the same operations, so they give the same verdicts and bytes. Each step's `plans` line names
its verdict when the step ran eagerly; the loop's `cache` counts every step's, read after the loop.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.author.kernels import Plan

try:
    import torch
except ImportError:  # pragma: no cover - the base package is torch-free
    torch = None


@dataclass(frozen=True, slots=True)
class FirstBlockCache:
    threshold: float

    def __post_init__(self) -> None:
        if not self.threshold > 0:
            raise ValueError(f"FirstBlockCache threshold {self.threshold} must be positive")

    def document(self) -> dict[str, object]:
        return {"method": "first-block", "threshold": self.threshold}


@dataclass(frozen=True, slots=True)
class EasyCache:
    threshold: float
    retain: int = 0
    cooldown: int = 0
    max_hits: int = 1

    def __post_init__(self) -> None:
        if not self.threshold > 0 or min(self.retain, self.cooldown) < 0 or self.max_hits < 1:
            raise ValueError(
                f"EasyCache {self}: a positive threshold and max_hits, retain/cooldown >= 0"
            )

    def document(self) -> dict[str, object]:
        return {
            "method": "easy",
            "threshold": self.threshold,
            "retain": self.retain,
            "cooldown": self.cooldown,
            "max_hits": self.max_hits,
        }


Policy = FirstBlockCache | EasyCache


@dataclass(frozen=True, slots=True)
class Stack:
    """A transformer's block stack: `blocks` its `nn.ModuleList`, `carried` the block arguments
    (index or keyword) its output returns, in that order."""

    model: type
    blocks: str = "transformer_blocks"
    carried: tuple[int | str, ...] = (0,)

    @property
    def name(self) -> str:
        return f"stepcache.{self.model.__name__.lower()}"


@dataclass(slots=True)
class _Installed:
    stack: Stack
    #: per carried argument: (its positional index, its keyword)
    places: tuple[tuple[int, str], ...]

    def inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(kwargs[name] if name in kwargs else args[index] for index, name in self.places)

    def outputs(self, out: Any) -> tuple[Any, ...]:
        found = (out,) if len(self.places) == 1 and isinstance(out, torch.Tensor) else tuple(out)
        if len(found) != len(self.places):
            raise ValueError(
                f"{self.stack.name}: a block returned {len(found)} tensors, the stack carries "
                f"{len(self.places)}"
            )
        return found

    def pack(self, like: Any, tensors: tuple[Any, ...]) -> Any:
        return tensors[0] if isinstance(like, torch.Tensor) else type(like)(tensors)


def install(plan: Plan, stack: Stack, transformer: Any) -> str:
    """Make `transformer`'s block stack cacheable on `plan`'s device: the plan's identity, or ""
    when it does not hold the declared stack (reported). Nothing changes until a loop names a
    policy."""
    blocks: Any = getattr(transformer, stack.blocks, None)
    why = ""
    if torch is None or not isinstance(transformer, stack.model):
        why = f"not a {stack.model.__name__}"
    elif not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 2:
        why = f"{stack.blocks} is not a list of two or more blocks"
    if why:
        plan.report(stack.name, {"applied": False, "reason": why})
        return ""
    parameters = _parameters(type(blocks[0]))
    places = []
    for argument in stack.carried:
        if isinstance(argument, str) and argument not in parameters:
            plan.report(stack.name, {"applied": False, "reason": f"a block takes no {argument}"})
            return ""
        index = argument if isinstance(argument, int) else parameters.index(argument)
        places.append((index, parameters[index] if index < len(parameters) else ""))
    blocks.__class__ = _listed(type(blocks))
    blocks._cozy_installed = _Installed(stack, tuple(places))
    plan.report(stack.name, {"applied": True, "blocks": len(blocks)})
    return f"{stack.name}:{stack.blocks}:{','.join(map(str, stack.carried))}"


def _parameters(block: type) -> list[str]:
    """The block's forward parameters after `self`: the nearest class's that names them (a fused
    or region subclass's forward takes `*args, **kwargs`)."""
    for cls in block.__mro__:
        forward = cls.__dict__.get("forward")
        if forward is None:
            continue
        named = list(inspect.signature(forward).parameters.values())[1:]
        if named and all(p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in named[:1]):
            return [p.name for p in named]
    return []


_LISTED: dict[type, type] = {}


def _listed(base: type) -> type:
    """The stack's list: under a cached loop, its iterations go through the call's cache."""
    if base not in _LISTED:

        def __iter__(self: Any) -> Iterator[Any]:
            run = _RUN.get()
            if run is None:
                return iter(self._modules.values())
            return run.forward(self, self._cozy_installed)

        _LISTED[base] = type(f"StepCached{base.__name__}", (base,), {"__iter__": __iter__})
    return _LISTED[base]


@dataclass(slots=True)
class _State:
    """One slot's references (a stack forward's place within its step), in device buffers updated
    in place, so a recorded step's graph reads what the last forward left. They begin unset, and
    count only once its flags say so: `has` FirstBlockCache's has-reference, EasyCache's
    has-computed and has-rate; `hits` EasyCache's skips in a row. The flags are a row of the
    call's (`_Run.flags`), zeroed with every call."""

    has: Any
    hits: Any
    ref: tuple[Any, ...] = ()
    residual: tuple[Any, ...] = ()
    previous: tuple[Any, ...] = ()
    inputs_c: tuple[Any, ...] = ()
    outputs_c: tuple[Any, ...] = ()
    rate: Any = None
    acc: Any = None
    like: Any = None


def _sum(tensors: Sequence[Any]) -> Any:
    """Σ|t| over every element, in float32, a fixed-order reduction."""
    return torch.stack([t.abs().sum(dtype=torch.float32) for t in tensors]).sum()


def _mean(tensors: Sequence[Any]) -> Any:
    return _sum(tensors) / sum(t.numel() for t in tensors)


def _diff(a: Sequence[Any], b: Sequence[Any], mean: bool = False) -> Any:
    deltas = [x - y for x, y in zip(a, b, strict=True)]
    return _mean(deltas) if mean else _sum(deltas)


def _like(tensors: Sequence[Any]) -> tuple[Any, ...]:
    """Unset buffers: an allocation alone, which a recording does not replay."""
    return tuple(torch.empty_like(t) for t in tensors)


#: The forwards a step's stacks hold state for (`_Run.flags`): every slot's row is the call's.
SLOTS = 64


@dataclass(slots=True)
class _Run:
    """One loop call's cache: its policy, its slots' state and its verdicts. `computes` and
    `forwards` count each step's computed and all forwards on the device, read once after the
    loop; `flags` holds each slot's. While a step's graph records, `cut` splits it at each
    forward's verdict (`_Graph.cut`): the rest of the stack in one branch, the cached residual in
    the other."""

    policy: Policy
    steps: range
    computes: Any
    forwards: Any
    flags: Any
    index: Any = None
    step: int | None = None
    cut: Any = None
    slot: int = 0
    states: dict[tuple[int, int], _State] = field(default_factory=dict)
    verdicts: list[tuple[int, bool]] = field(default_factory=list)

    def begin(self, index: Any, step: int | None = None, cut: Any = None) -> None:
        """A step: its device index, its host index when it runs eagerly, the recording's cut."""
        self.index, self.step, self.cut, self.slot = index, step, cut, 0

    def forward(self, blocks: Any, installed: _Installed) -> Iterator[Any]:
        """An iteration of the stack. Its first item stands for block 0 (attribute reads pass
        through); an iteration that calls it is the stack's forward, any other (a read of each
        block's processor) sees the blocks as they are."""
        modules = list(blocks._modules.values())
        gate = _Gate(self, blocks, installed, modules[0])
        yield gate
        if gate.state is None:
            yield from modules[1:]
        elif gate.computed:
            yield from modules[1:-1]
            yield _Last(self, gate.state, installed, modules[-1], gate)

    def claim(self, blocks: Any) -> _State:
        """The state of the stack's next forward in this step (its slot)."""
        key = (id(blocks), self.slot)
        self.slot += 1
        if key not in self.states:
            if len(self.states) == SLOTS:
                raise ValueError(f"a step cache holds at most {SLOTS} forwards a step")
            row = self.flags[len(self.states)]
            self.states[key] = _State(row[:3], row[3:])
        return self.states[key]

    def counted(self) -> tuple[Any, Any, Any]:
        """What a recorded program's graphs write and a later call zeroes (`adopt`)."""
        return self.computes, self.forwards, self.flags

    def adopt(self, counted: tuple[Any, Any, Any]) -> None:
        """A kept program's counters and flags, which its graphs write: this call's."""
        self.computes, self.forwards, self.flags = counted
        for tensor in counted:
            tensor.zero_()

    def verdict(self) -> str:
        if self.step is None:
            return ""
        mine = [computed for step, computed in self.verdicts if step == self.step]
        return "" if not mine else "cache:compute" if any(mine) else "cache:skip"

    def document(self) -> dict[str, object]:
        computes, forwards = self.computes.tolist(), self.forwards.tolist()
        return {
            **self.policy.document(),
            "forwards": sum(forwards),
            "computed": sum(computes),
            "skipped_steps": [
                self.steps.start + step
                for step, (done, total) in enumerate(zip(computes, forwards, strict=True))
                for _ in range(total - done)
            ],
        }


class _Gate:
    """The stack's first call each forward: the signal, the verdict, block 0 when it runs."""

    def __init__(self, run: _Run, blocks: Any, installed: _Installed, first: Any) -> None:
        self.run, self.blocks, self.installed, self.first = run, blocks, installed, first
        self.state: _State | None = None
        self.computed = True
        self.inputs: tuple[Any, ...] = ()
        self.head: tuple[Any, ...] = ()
        self.r0: tuple[Any, ...] = ()
        self.merged: tuple[Any, ...] = ()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["first"], name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        run, installed = self.run, self.installed
        self.inputs = xs = installed.inputs(args, kwargs)
        self.state = state = run.claim(self.blocks)
        relative = run.index - run.steps.start
        out = None
        if isinstance(run.policy, FirstBlockCache):
            out = self.first(*args, **kwargs)
            self.head = installed.outputs(out)
            self.r0 = tuple(h - x for h, x in zip(self.head, xs, strict=True))
            if not state.ref:
                state.ref, state.residual = _like(self.r0), _like(self.head)
            computed = (state.has[0] == 0) | (
                _diff(self.r0, state.ref) > run.policy.threshold * _sum(state.ref).clamp_min(1e-8)
            )
        else:
            computed = self._easy(run.policy, state, xs, relative)
        run.computes.index_add_(0, relative, computed.int().reshape(1))
        run.forwards.index_add_(0, relative, torch.ones_like(run.forwards[:1]))
        if run.cut is not None:  # recording: both branches, the device decides
            self.merged = _like(self.head or xs)
            run.cut(computed.int())
        else:
            self.computed = bool(computed.item())
            run.verdicts.append((run.step, self.computed))
            if not self.computed:
                skipped = self.skip()
                return installed.pack(out if out is not None else state.like, skipped)
        return out if out is not None else self.first(*args, **kwargs)

    def _easy(self, policy: EasyCache, state: _State, xs: tuple[Any, ...], relative: Any) -> Any:
        if not state.previous:
            state.previous, state.inputs_c, state.outputs_c, state.residual = (
                _like(xs), _like(xs), _like(xs), _like(xs)
            )
            state.rate, state.acc = (torch.empty(1, device=xs[0].device) for _ in range(2))
        last = len(self.run.steps) - 1
        forced = (
            (relative < policy.retain)
            | (relative >= last - policy.cooldown)
            | (state.has[2] == 0)
            | (state.hits >= policy.max_hits)
        )
        change = _diff(xs, state.previous, mean=True) / _mean(state.outputs_c).clamp_min(1e-8)
        accumulated = state.acc + state.rate * change
        state.acc.copy_(torch.where(forced, state.acc, accumulated))
        for previous, x in zip(state.previous, xs, strict=True):
            previous.copy_(x)
        return forced | (accumulated >= policy.threshold)

    def skip(self) -> tuple[Any, ...]:
        """The cached forward: block 0's output (FirstBlockCache) or its input plus the
        residual."""
        state = self.state
        assert state is not None
        if isinstance(self.run.policy, FirstBlockCache):
            return tuple(h + r for h, r in zip(self.head, state.residual, strict=True))
        state.hits.add_(1)
        return tuple(x + r for x, r in zip(self.inputs, state.residual, strict=True))


class _Last:
    """The stack's last block on a computed forward: its output, then the references. Recording,
    the skipped branch follows: the stack's output is `merged` either way."""

    def __init__(
        self, run: _Run, state: _State, installed: _Installed, last: Any, gate: _Gate
    ) -> None:
        self.run, self.state, self.installed, self.last, self.gate = (
            run,
            state,
            installed,
            last,
            gate,
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["last"], name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.last(*args, **kwargs)
        ys, state, gate = self.installed.outputs(out), self.state, self.gate
        if isinstance(self.run.policy, FirstBlockCache):
            for ref, r0 in zip(state.ref, gate.r0, strict=True):
                ref.copy_(r0)
            for residual, y, h in zip(state.residual, ys, gate.head, strict=True):
                residual.copy_(y - h)
            state.has[0].fill_(1)
        else:
            xs = gate.inputs
            rate = _diff(ys, state.outputs_c, mean=True) / _diff(
                xs, state.inputs_c, mean=True
            ).clamp_min(1e-8)
            state.rate.copy_(torch.where(state.has[1] != 0, rate, state.rate))
            state.has[2].copy_(state.has[2] | state.has[1])
            pairs = (*zip(state.inputs_c, xs, strict=True), *zip(state.outputs_c, ys, strict=True))
            for kept, value in pairs:
                kept.copy_(value)
            for residual, y, x in zip(state.residual, ys, xs, strict=True):
                residual.copy_(y - x)
            state.acc.zero_()
            state.hits.zero_()
            state.has[1].fill_(1)
            state.like = out
        if self.run.cut is None:
            return out
        for merged, y in zip(gate.merged, ys, strict=True):
            merged.copy_(y)
        self.run.cut(None)  # the skipped branch
        for merged, value in zip(gate.merged, gate.skip(), strict=True):
            merged.copy_(value)
        self.run.cut(None)  # after the branches
        return self.installed.pack(out, gate.merged)


_RUN: ContextVar[_Run | None] = ContextVar("cozy_stepcache", default=None)


@contextmanager
def running(policy: Policy, steps: range, device: Any) -> Iterator[_Run]:
    """A loop call under `policy` (`author.denoise_loop(cache=…)`): its stacks' forwards cache."""
    counted = (torch.zeros(len(steps), dtype=torch.int32, device=device) for _ in range(2))
    run = _Run(policy, steps, *counted, torch.zeros(SLOTS, 4, dtype=torch.int32, device=device))
    token = _RUN.set(run)
    try:
        yield run
    finally:
        _RUN.reset(token)


def verdict() -> str:
    """The current step's verdict for its `plans` line, "" outside a cached loop."""
    run = _RUN.get()
    return "" if run is None else run.verdict()
