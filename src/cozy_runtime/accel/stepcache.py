"""Step caching for a diffusion transformer's block stack: a forward whose stack would change
little reuses the stack's cached residual instead of running its blocks.

It is approximate by construction, so a package turns it on per call, for a preset that trades
quality for speed, never by default. Declare the stack at module scope, install it in the
execution plan, and run the denoise loop under the policy:

    STACK = stepcache.Stack(MyTransformer, blocks="transformer_blocks", carried=(0,))

    def execution_plan(self, plan):
        return stepcache.install(plan, STACK, self.transformer)

    with stepcache.running(stepcache.FirstBlockCache(0.08), n, device) as run:
        for i in range(n):
            run.step(i)
            ...

`carried` names the block arguments its output returns, in order: a positional index or a
keyword (`(0,)` for a single-stream block, `("encoder_hidden_states", "hidden_states")` for a
dual-stream one). The stack runs as one `for` over its list, in the transformer's forward or in
the step; a skipped forward's iteration ends after block 0, so not inside `zip(..., strict=True)`.
Embeddings and the final layer run every forward; a skipped forward runs none of the stack's
blocks after its first, so their weights, AdaLN and carried state stay as the last computed
forward left them. State starts with every `running`. Each forward's verdict comes from
fixed-order reductions on the device, so the same inputs and policy give the same verdicts and
bytes. `run.document()` counts its forwards, computed and skipped.

- `FirstBlockCache(t)` (NVIDIA sol-engine `first_block_cache.py`): block 0 runs, `r0 = h0 - x`;
  the rest run when there is no reference yet or `Σ|r0 - r0_ref| > t·Σ|r0_ref|`, and then
  `r0_ref = r0`, `R = y - h0`; else `y = h0 + R`.
- `EasyCache(t, retain, cooldown, max_hits)` (sol-engine `easycache.py`): a forward computes
  while `i < retain`, from `N - cooldown` (`N` the call's last forward), before its rate `k` exists
  and after `max_hits` skips in a row; else it adds `k·mean|x - x_prev| / mean|Y_c|` to `acc` and
  computes once `acc ≥ t`. A computed forward sets `k = mean|y - Y_c| / mean|x - X_c|`,
  `X_c = x`, `Y_c = y`, `R = y - x` and clears `acc`; a skip is `y = x + R`.

Its state lives in device buffers updated in place; each forward reads its verdict on the host
once, so component calls under `running` run eagerly, never from a CUDA graph. Each step's `plans`
line names its verdict.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author._model import WARNINGS, WarningRow, recovered

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn
else:
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

    model: type[nn.Module]
    blocks: str = "transformer_blocks"
    carried: tuple[int | str, ...] = (0,)

    @property
    def name(self) -> str:
        return f"stepcache.{self.model.__name__.lower()}"


#: What a block returns: its hidden states, or a tuple of them (the stack's `carried`).
type BlockOutput = Tensor | tuple[Tensor, ...]


@dataclass(slots=True)
class _Installed:
    stack: Stack
    #: per carried argument: (its positional index, its keyword)
    places: tuple[tuple[int, str], ...]

    def inputs(self, args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[Tensor, ...]:
        found = (kwargs[name] if name in kwargs else args[index] for index, name in self.places)
        return cast("tuple[Tensor, ...]", tuple(found))  # the Stack's carried hidden states

    def outputs(self, out: BlockOutput) -> tuple[Tensor, ...]:
        found = (out,) if len(self.places) == 1 and isinstance(out, torch.Tensor) else tuple(out)
        if len(found) != len(self.places):
            raise ValueError(
                f"{self.stack.name}: a block returned {len(found)} tensors, the stack carries "
                f"{len(self.places)}"
            )
        return found

    def pack(self, like: BlockOutput, tensors: tuple[Tensor, ...]) -> BlockOutput:
        return tensors[0] if isinstance(like, torch.Tensor) else type(like)(tensors)


class _Listed(Protocol):
    """A stack's `nn.ModuleList` once installed (`_listed`'s class)."""

    _modules: dict[str, nn.Module | None]
    _cozy_installed: _Installed


def install(plan: Plan, stack: Stack, transformer: nn.Module) -> str:
    """Make `transformer`'s block stack cacheable on `plan`'s device: the plan's identity, or ""
    when it does not hold the declared stack (reported). Nothing changes until a loop names a
    policy."""
    found: object = getattr(transformer, stack.blocks, None)
    why = ""
    if torch is None or not isinstance(transformer, stack.model):
        why = f"not a {stack.model.__name__}"
    elif not isinstance(found, torch.nn.ModuleList) or len(found) < 2:
        why = f"{stack.blocks} is not a list of two or more blocks"
    if why:
        plan.report(stack.name, {"applied": False, "reason": why})
        return ""
    blocks = cast("nn.ModuleList", found)  # the elif above found it one
    parameters = _parameters(type(blocks[0]))
    places = []
    for argument in stack.carried:
        if isinstance(argument, str) and argument not in parameters:
            plan.report(stack.name, {"applied": False, "reason": f"a block takes no {argument}"})
            return ""
        index = argument if isinstance(argument, int) else parameters.index(argument)
        places.append((index, parameters[index] if index < len(parameters) else ""))
    blocks.__class__ = _listed(type(blocks))
    cast("_Listed", blocks)._cozy_installed = _Installed(stack, tuple(places))
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


_LISTED: dict[type[nn.ModuleList], type[nn.ModuleList]] = {}


def _listed(base: type[nn.ModuleList]) -> type[nn.ModuleList]:
    """The stack's list: under a cached loop, its iterations go through the call's cache."""
    if base not in _LISTED:

        def __iter__(self: _Listed) -> Iterator[_Block | None]:
            run = _RUN.get()
            if run is None:
                return iter(self._modules.values())
            return run.forward(self, self._cozy_installed)

        listed = type(f"StepCached{base.__name__}", (base,), {"__iter__": __iter__})
        _LISTED[base] = cast("type[nn.ModuleList]", listed)  # a subclass of `base`
    return _LISTED[base]


@dataclass(slots=True)
class _State:
    """One slot's references (a stack forward's place within its step), in device buffers updated
    in place. They begin unset, and
    count only once its flags say so: `has` FirstBlockCache's has-reference, EasyCache's
    has-computed and has-rate; `hits` EasyCache's skips in a row. The flags are a row of the
    run's (`Run.flags`), zeroed with every `running`."""

    has: Tensor
    hits: Tensor
    ref: tuple[Tensor, ...] = ()
    residual: tuple[Tensor, ...] = ()
    previous: tuple[Tensor, ...] = ()
    inputs_c: tuple[Tensor, ...] = ()
    outputs_c: tuple[Tensor, ...] = ()
    rate: Tensor | None = None
    acc: Tensor | None = None
    like: BlockOutput | None = None


def _sum(tensors: Sequence[Tensor]) -> Tensor:
    """Σ|t| over every element, in float32, a fixed-order reduction."""
    return torch.stack([t.abs().sum(dtype=torch.float32) for t in tensors]).sum()


def _mean(tensors: Sequence[Tensor]) -> Tensor:
    return _sum(tensors) / sum(t.numel() for t in tensors)


def _diff(a: Sequence[Tensor], b: Sequence[Tensor], mean: bool = False) -> Tensor:
    deltas = [x - y for x, y in zip(a, b, strict=True)]
    return _mean(deltas) if mean else _sum(deltas)


def _like(tensors: Sequence[Tensor]) -> tuple[Tensor, ...]:
    """Unset buffers: an allocation alone, which a recording does not replay."""
    return tuple(torch.empty_like(t) for t in tensors)


#: The forwards a step's stacks hold state for (`Run.flags`): every slot's row is the run's.
SLOTS = 64


@dataclass(slots=True)
class Run:
    """One loop's cache (`running`): its policy, its slots' state (`flags`) and its verdicts."""

    policy: Policy
    steps: int
    flags: torch.Tensor
    index: torch.Tensor | None = None
    current: int | None = None
    slot: int = 0
    states: dict[tuple[int, int], _State] = field(default_factory=dict)
    verdicts: list[tuple[int, bool]] = field(default_factory=list)

    def step(self, i: int) -> None:
        """Step `i` (0-based, of `steps`) begins: call before its model calls."""
        if not 0 <= i < self.steps:
            raise ValueError(f"step {i} is outside the cached loop's {self.steps} steps")
        self.current, self.slot = i, 0
        self.index = torch.full((1,), i, dtype=torch.int64, device=self.flags.device)

    def forward(self, blocks: _Listed, installed: _Installed) -> Iterator[_Block]:
        """An iteration of the stack. Its first item stands for block 0 (attribute reads pass
        through); an iteration that calls it is the stack's forward, any other (a read of each
        block's processor) sees the blocks as they are."""
        modules = cast("list[nn.Module]", list(blocks._modules.values()))  # a list holds no None
        gate = _Gate(self, blocks, installed, modules[0])
        yield gate
        if gate.state is None:
            yield from modules[1:]
        elif gate.computed:
            yield from modules[1:-1]
            yield _Last(self, gate.state, installed, modules[-1], gate)

    def claim(self, blocks: _Listed) -> _State:
        """The state of the stack's next forward in this step (its slot)."""
        key = (id(blocks), self.slot)
        self.slot += 1
        if key not in self.states:
            if len(self.states) == SLOTS:
                raise ValueError(f"a step cache holds at most {SLOTS} forwards a step")
            row = self.flags[len(self.states)]
            self.states[key] = _State(row[:3], row[3:])
        return self.states[key]

    def verdict(self) -> str:
        if self.current is None:
            return ""
        mine = [computed for step, computed in self.verdicts if step == self.current]
        return "" if not mine else "cache:compute" if any(mine) else "cache:skip"

    def document(self) -> dict[str, object]:
        """The policy, and its forwards: computed and skipped, by step."""
        return {
            **self.policy.document(),
            "forwards": len(self.verdicts),
            "computed": sum(computed for _, computed in self.verdicts),
            "skipped_steps": [step for step, computed in self.verdicts if not computed],
        }


class _Gate:
    """The stack's first call each forward: the signal, the verdict, block 0 when it runs."""

    def __init__(self, run: Run, blocks: _Listed, installed: _Installed, first: nn.Module) -> None:
        self.run, self.blocks, self.installed, self.first = run, blocks, installed, first
        self.state: _State | None = None
        self.computed = True
        self.inputs: tuple[Tensor, ...] = ()
        self.head: tuple[Tensor, ...] = ()
        self.r0: tuple[Tensor, ...] = ()

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["first"], name)

    def __call__(self, *args: object, **kwargs: object) -> BlockOutput:
        run, installed = self.run, self.installed
        if run.current is None:
            raise ValueError("a step-cached loop calls `run.step(i)` before each step's forwards")
        self.inputs = xs = installed.inputs(args, kwargs)
        self.state = state = run.claim(self.blocks)
        out: BlockOutput | None = None
        if isinstance(run.policy, FirstBlockCache):
            out = first = cast("BlockOutput", self.first(*args, **kwargs))  # a block of the Stack
            self.head = installed.outputs(first)
            self.r0 = tuple(h - x for h, x in zip(self.head, xs, strict=True))
            if not state.ref:
                state.ref, state.residual = _like(self.r0), _like(self.head)
            computed = (state.has[0] == 0) | (
                _diff(self.r0, state.ref) > run.policy.threshold * _sum(state.ref).clamp_min(1e-8)
            )
        else:
            # `run.step(i)` set the index with the current step checked above.
            computed = self._easy(run.policy, state, xs, cast("Tensor", run.index))
        self.computed = bool(computed.item())
        run.verdicts.append((run.current, self.computed))
        if not self.computed:
            skipped = self.skip()
            # EasyCache skips only after a computed forward, which kept its output (`like`).
            like = out if out is not None else cast("BlockOutput", state.like)
            return installed.pack(like, skipped)
        if out is not None:
            return out
        computed_out: BlockOutput = self.first(*args, **kwargs)
        return computed_out

    def _easy(
        self, policy: EasyCache, state: _State, xs: tuple[Tensor, ...], relative: Tensor
    ) -> Tensor:
        if not state.previous:
            state.previous, state.inputs_c, state.outputs_c, state.residual = (
                _like(xs), _like(xs), _like(xs), _like(xs)
            )
            state.rate, state.acc = (torch.empty(1, device=xs[0].device) for _ in range(2))
        rate, acc = cast("Tensor", state.rate), cast("Tensor", state.acc)  # made just above
        last = self.run.steps - 1
        forced = (
            (relative < policy.retain)
            | (relative >= last - policy.cooldown)
            | (state.has[2] == 0)
            | (state.hits >= policy.max_hits)
        )
        change = _diff(xs, state.previous, mean=True) / _mean(state.outputs_c).clamp_min(1e-8)
        accumulated = acc + rate * change
        acc.copy_(torch.where(forced, acc, accumulated))
        for previous, x in zip(state.previous, xs, strict=True):
            previous.copy_(x)
        return forced | (accumulated >= policy.threshold)

    def skip(self) -> tuple[Tensor, ...]:
        """The cached forward: block 0's output (FirstBlockCache) or its input plus the
        residual."""
        state = self.state
        assert state is not None
        if isinstance(self.run.policy, FirstBlockCache):
            return tuple(h + r for h, r in zip(self.head, state.residual, strict=True))
        state.hits.add_(1)
        return tuple(x + r for x, r in zip(self.inputs, state.residual, strict=True))


class _Last:
    """The stack's last block on a computed forward: its output, then the references."""

    def __init__(
        self, run: Run, state: _State, installed: _Installed, last: nn.Module, gate: _Gate
    ) -> None:
        self.run, self.state, self.installed, self.last, self.gate = (
            run,
            state,
            installed,
            last,
            gate,
        )

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["last"], name)

    def __call__(self, *args: object, **kwargs: object) -> BlockOutput:
        out: BlockOutput = self.last(*args, **kwargs)
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
            kept_rate = cast("Tensor", state.rate)  # `_Gate._easy` made it this forward
            kept_rate.copy_(torch.where(state.has[1] != 0, rate, kept_rate))
            state.has[2].copy_(state.has[2] | state.has[1])
            pairs = (*zip(state.inputs_c, xs, strict=True), *zip(state.outputs_c, ys, strict=True))
            for kept, value in pairs:
                kept.copy_(value)
            for residual, y, x in zip(state.residual, ys, xs, strict=True):
                residual.copy_(y - x)
            cast("Tensor", state.acc).zero_()  # `_Gate._easy` made it this forward
            state.hits.zero_()
            state.has[1].fill_(1)
            state.like = out
        return out


type _Block = nn.Module | _Gate | _Last

_RUN: ContextVar[Run | None] = ContextVar("cozy_stepcache", default=None)


@contextmanager
def running(policy: Policy, steps: int, device: torch.device | str) -> Iterator[Run]:
    """A loop of `steps` steps under `policy`: its installed stacks' forwards cache. Call
    `run.step(i)` as each step begins. Component calls inside run eagerly (no CUDA graph).
    Across the GPUs of a group the cache is off: every step computes."""
    if steps < 1:
        raise ValueError("a step-cached loop has at least one step")
    run = Run(policy, steps, torch.zeros(SLOTS, 4, dtype=torch.int32, device=device))
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        # Each rank's verdict would differ (its sums are not all-reduced): every step computes.
        row: WarningRow = recovered("step cache: off across ranks")
        if row not in WARNINGS:
            WARNINGS.append(row)
        yield run
        return
    token = _RUN.set(run)
    try:
        yield run
    finally:
        _RUN.reset(token)


def verdict() -> str:
    """The current step's verdict for its `plans` line, "" outside a cached loop."""
    run = _RUN.get()
    return "" if run is None else run.verdict()
