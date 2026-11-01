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
loop's record carries `cache` (its forwards, computed and skipped), and each step's `plans` line
ends `cache:compute` or `cache:skip`.

- `FirstBlockCache(t)` (NVIDIA sol-engine `first_block_cache.py`): block 0 runs, `r0 = h0 - x`;
  the rest run when there is no reference yet or `Σ|r0 - r0_ref| > t·Σ|r0_ref|`, and then
  `r0_ref = r0`, `R = y - h0`; else `y = h0 + R`.
- `EasyCache(t, retain, cooldown, max_hits)` (sol-engine `easycache.py`): a forward computes
  while `i < retain`, from `N - cooldown` (`N` the call's last forward), before its rate `k` exists
  and after `max_hits` skips in a row; else it adds `k·mean|x - x_prev| / mean|Y_c|` to `acc` and
  computes once `acc ≥ t`. A computed forward sets `k = mean|y - Y_c| / mean|x - X_c|`,
  `X_c = x`, `Y_c = y`, `R = y - x` and clears `acc`; a skip is `y = x + R`.

A loop with a cache runs eagerly: each forward reads its verdict on the host once.
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
    """One slot's references (a stack forward's place within its step), on the device."""

    ref: tuple[Any, ...] = ()
    residual: tuple[Any, ...] = ()
    previous: tuple[Any, ...] = ()
    inputs_c: tuple[Any, ...] = ()
    outputs_c: tuple[Any, ...] = ()
    rate: Any = None
    acc: Any = None
    hits: int = 0
    like: Any = None


def _sum(tensors: Sequence[Any]) -> Any:
    """Σ|t| over every element, in float32, a fixed-order reduction."""
    return torch.stack([t.abs().sum(dtype=torch.float32) for t in tensors]).sum()


def _mean(tensors: Sequence[Any]) -> Any:
    return _sum(tensors) / sum(t.numel() for t in tensors)


def _diff(a: Sequence[Any], b: Sequence[Any], mean: bool = False) -> Any:
    deltas = [x - y for x, y in zip(a, b, strict=True)]
    return _mean(deltas) if mean else _sum(deltas)


@dataclass(slots=True)
class _Run:
    """One loop call's cache: its policy, its slots' state and its verdicts."""

    policy: Policy
    forwards: int
    step: int = 0
    slot: int = 0
    states: dict[tuple[int, int], _State] = field(default_factory=dict)
    verdicts: list[tuple[int, bool]] = field(default_factory=list)

    def begin(self, step: int) -> None:
        self.step, self.slot = step, 0

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
        state = self.states.setdefault((id(blocks), self.slot), _State())
        self.slot += 1
        return state

    def verdict(self) -> str:
        mine = [computed for step, computed in self.verdicts if step == self.step]
        return "" if not mine else "cache:compute" if any(mine) else "cache:skip"

    def document(self) -> dict[str, object]:
        skipped = [step for step, computed in self.verdicts if not computed]
        return {
            **self.policy.document(),
            "forwards": len(self.verdicts),
            "computed": len(self.verdicts) - len(skipped),
            "skipped_steps": skipped,
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

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["first"], name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        run, installed = self.run, self.installed
        self.state = state = run.claim(self.blocks)
        policy = run.policy
        self.inputs = xs = installed.inputs(args, kwargs)
        if isinstance(policy, FirstBlockCache):
            out = self.first(*args, **kwargs)
            self.head = installed.outputs(out)
            self.r0 = tuple(h - x for h, x in zip(self.head, xs, strict=True))
            if state.ref:
                verdict = _diff(self.r0, state.ref) > policy.threshold * _sum(state.ref).clamp_min(
                    1e-8
                )
                self.computed = bool(verdict.item())
            run.verdicts.append((run.step, self.computed))
            if self.computed:  # the old references go now: the tail holds r0 and h0 alone
                state.ref = state.residual = ()
                return out
            return installed.pack(
                out, tuple(h + r for h, r in zip(self.head, state.residual, strict=True))
            )
        self.computed = self._easy(policy, xs)
        run.verdicts.append((run.step, self.computed))
        state.previous = tuple(x.clone() for x in xs)
        if self.computed:
            return self.first(*args, **kwargs)
        state.hits += 1
        return installed.pack(
            state.like, tuple(x + r for x, r in zip(xs, state.residual, strict=True))
        )

    def _easy(self, policy: EasyCache, xs: tuple[Any, ...]) -> bool:
        state, index = self.state, self.run.step
        assert state is not None
        last = self.run.forwards - 1
        if (
            index < policy.retain
            or index >= last - policy.cooldown
            or state.rate is None
            or state.hits >= policy.max_hits
        ):
            return True
        change = _diff(xs, state.previous, mean=True) / _mean(state.outputs_c).clamp_min(1e-8)
        state.acc = state.acc + state.rate * change
        return bool((state.acc >= policy.threshold).item())


class _Last:
    """The stack's last block on a computed forward: its output, then the references."""

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
            state.ref = gate.r0
            state.residual = tuple(y - h for y, h in zip(ys, gate.head, strict=True))
            return out
        xs = gate.inputs
        if state.inputs_c:
            state.rate = _diff(ys, state.outputs_c, mean=True) / _diff(
                xs, state.inputs_c, mean=True
            ).clamp_min(1e-8)
        state.inputs_c = tuple(x.clone() for x in xs)
        state.outputs_c = tuple(y.clone() for y in ys)
        state.residual = tuple(y - x for y, x in zip(ys, xs, strict=True))
        state.acc = torch.zeros((), dtype=torch.float32, device=ys[0].device)
        state.hits = 0
        state.like = out
        return out


_RUN: ContextVar[_Run | None] = ContextVar("cozy_stepcache", default=None)


@contextmanager
def running(policy: Policy, forwards: int) -> Iterator[_Run]:
    """A loop call under `policy` (`author.denoise_loop(cache=…)`): its stacks' forwards cache."""
    run = _Run(policy, forwards)
    token = _RUN.set(run)
    try:
        yield run
    finally:
        _RUN.reset(token)


def verdict() -> str:
    """The current step's verdict for its `plans` line, "" outside a cached loop."""
    run = _RUN.get()
    return "" if run is None else run.verdict()
