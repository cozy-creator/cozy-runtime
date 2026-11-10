"""Every GPU of a group runs the attempt's handler, as torchrun runs a script on every rank.

The same request, seeds and random streams make every GPU compute the same per-step state with
the same code; a `_cp_plan` component splits and gathers inside its own call (Diffusers' hooks).
What one GPU alone decides or holds crosses here, always at the same point of every GPU's
program: a store meeting (`Ranks.meet`) that also says which GPUs are not at the same call, then
one broadcast per tensor from the GPU that holds it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json
import random
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, cast

import msgspec

from cozy_runtime.author import _group
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.parallel import wire
from cozy_runtime.internal.parallel.group import gather
from cozy_runtime.internal.parallel.plan import GroupRefusal

if TYPE_CHECKING:
    import torch
    from torch.distributed import ProcessGroup
    from torch.utils.hooks import RemovableHandle

#: One GPU's row at a meeting: what it is at, and what it states there.
type Row = dict[str, Json]


class UngatedShardedForward(GroupRefusal):
    """A sharded or hosted component called outside a group attempt."""

    default_code = "ungated_sharded_forward"


class Ranks:
    """This GPU's part of one group attempt: who it is among the group, and the meetings and
    crossings that keep every GPU on one program."""

    def __init__(
        self,
        *,
        rank: int,
        world: int,
        pg: ProcessGroup,
        device: torch.device,
        tag: str,
        gpu: Callable[[int], str],
    ) -> None:
        import torch.distributed as dist
        from torch._C._distributed_c10d import PrefixStore
        from torch.distributed.distributed_c10d import _get_process_group_store

        self.rank, self.world, self.pg, self.device, self.gpu = rank, world, pg, device, gpu
        self.store = PrefixStore(f"cozy.spmd/{tag}/", _get_process_group_store(pg))
        self.sequence = 0
        self.lock = threading.Lock()
        self.nccl = dist.get_backend(pg) == "nccl"

    def meet(self, value: Mapping[str, Json], what: str) -> list[Row]:
        """Every GPU's `value` at this point of the program. GPUs at different points raise,
        naming each; a GPU whose handler already ended is named at once (`gather`)."""
        with self.lock:
            self.sequence += 1
            n = self.sequence
            keys = [f"{n}/{rank}" for rank in range(self.world)]
            self.store.set(keys[self.rank], json.dumps({"what": what, **value}))
            ended = [f"end/{rank}" for rank in range(self.world)]
            rows: list[Row] = [json.loads(raw) for raw in gather(self.store, keys, what, ended)]
            if n > 1:
                self.store.delete_key(f"{n - 1}/{self.rank}")
        if len({row["what"] for row in rows}) > 1:
            raise GroupRefusal(
                "the GPUs are not at one call: "
                + ", ".join(f"{self.gpu(rank)} at {row['what']}" for rank, row in enumerate(rows)),
                code="group_diverged",
            )
        return rows

    def end(self, ok: bool) -> None:
        """This GPU's handler returned or raised: a GPU still waiting at a meeting for it learns
        so at once."""
        self.store.set(f"end/{self.rank}", "ok" if ok else "failed")

    def clear(self, meetings: Mapping[int, int]) -> None:
        """GPU 0, once every GPU answered: the attempt's last keys go."""
        keys = [f"end/{rank}" for rank in range(self.world)]
        keys += [f"{n}/{rank}" for rank, n in meetings.items() if n]
        for key in keys:
            with contextlib.suppress(Exception):
                self.store.delete_key(key)

    def all(self, flag: bool, what: str) -> bool:
        return all(row["flag"] for row in self.meet({"flag": bool(flag)}, what))

    def told(self, value: Mapping[str, Json] | None, what: str) -> Row:
        """GPU 0's `value` on every GPU."""
        told = self.meet({"told": dict(value or {}) if self.rank == 0 else None}, what)[0]["told"]
        if not isinstance(told, dict):
            raise GroupRefusal(f"{what}: GPU 0 told no value", code="group_diverged")
        return told

    def cross(self, value: object, owner: int, what: str) -> object:
        """`value` as GPU `owner` holds it, on every GPU: its structure through the store, each
        tensor by one broadcast from `owner`, onto this GPU (host tensors stay on the host)."""
        tensors: list[torch.Tensor] = []
        structure = wire.marshal(value, tensors) if self.rank == owner else None
        rows = self.meet({"structure": structure}, what)
        structure = rows[owner]["structure"]
        received = [
            self._broadcast(tensors[spec.v] if self.rank == owner else None, spec, owner)
            for spec in wire.specs(structure)
        ]
        return wire.unmarshal(structure, received)

    def spread[*Ts, R](
        self, method: Callable[[*Ts], R], arguments: Iterable[tuple[*Ts]]
    ) -> Iterator[R]:
        """Call n runs on GPU n mod K; a round's K results cross from their GPUs to every GPU,
        and come out in call order."""
        name = getattr(method, "__qualname__", getattr(method, "__name__", "spread"))
        calls, at = iter(arguments), 0
        while batch := list(itertools.islice(calls, self.world)):
            mine = method(*batch[self.rank]) if self.rank < len(batch) else None
            sent = wire.one_tensor(mine) if self.rank < len(batch) else None
            tensors: list[torch.Tensor] = []
            head = wire.marshal(sent, tensors)
            rows = self.meet({"head": head}, f"spread {name} {at}-{at + len(batch) - 1}")
            for owner in range(len(batch)):
                spec = msgspec.convert(rows[owner]["head"], wire._Tensor)
                received = self._broadcast(sent if owner == self.rank else None, spec, owner)
                # every call returns one tensor (`author.spread`): GPU `owner`'s is that tensor
                yield cast(R, mine if owner == self.rank else received)
            at += len(batch)
            if len(batch) < self.world:
                return

    def _broadcast(
        self, tensor: torch.Tensor | None, spec: wire._Tensor, owner: int
    ) -> torch.Tensor:
        import torch
        import torch.distributed as dist
        from torch.utils._python_dispatch import _disable_current_modes

        dtype = getattr(torch, spec.dtype)
        wire_device = self.device if self.nccl else torch.device("cpu")
        with _disable_current_modes():
            if tensor is not None:
                buffer = tensor.detach().resolve_conj().resolve_neg().contiguous()
                buffer = buffer.to(wire_device)
            else:
                buffer = torch.empty(spec.shape, dtype=dtype, device=wire_device)
            if buffer.numel():
                dist.broadcast(buffer.reshape(-1).view(torch.uint8), src=owner, group=self.pg)
        if tensor is not None:
            return tensor
        return buffer.cpu() if spec.host else buffer.to(self.device)


def current() -> Ranks | None:
    group = _group.CURRENT[0]
    return group if isinstance(group, Ranks) else None


@contextlib.contextmanager
def attempt(ranks: Ranks, entropy: int) -> Iterator[Ranks]:
    """`ranks` serves this process's attempt; every GPU's global random streams start from the
    attempt's entropy, so code drawing from them draws the same numbers everywhere."""
    seed(entropy)
    _group.CURRENT[0] = ranks
    try:
        yield ranks
    finally:
        _group.CURRENT[0] = None


def seed(entropy: int) -> None:
    import torch

    random.seed(entropy)
    torch.manual_seed(entropy & ((1 << 63) - 1))
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(entropy & 0xFFFFFFFF)


def install_call_check(component: torch.nn.Module, name: str) -> None:
    """Before each call of a sharded root, every GPU meets at it: a GPU at another call, or one
    whose handler already returned, is named instead of hanging a collective. Arguments may
    differ: a GPU's own rows of a table the plan splits are its own."""
    if getattr(component, "_cozy_spmd_check", None) is not None:
        return

    def check(module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        ranks = current()
        if ranks is None:
            raise UngatedShardedForward(
                f"sharded component {name!r} was called outside a group attempt: every GPU "
                "of its group joins its collectives, and none is running one"
            )
        ranks.meet({}, f"{name} call")

    handle: RemovableHandle = component.register_forward_pre_hook(check, with_kwargs=True)
    with contextlib.suppress(Exception):
        object.__setattr__(component, "_cozy_spmd_check", handle)


_hosting = threading.local()


def host_component(component: torch.nn.Module, *, name: str, home: int, rank: int) -> None:
    """Every module call into `component`, its root or any submodule, runs on GPU `home`, which
    alone holds its bytes; its result crosses to every GPU (`Ranks.cross`)."""
    for path, module in component.named_modules():
        what = f"{name}.{path}" if path else name
        object.__setattr__(module, "_call_impl", _hosted(module._call_impl, what, home, rank))


def _hosted[**P](call: Callable[P, object], what: str, home: int, rank: int) -> Callable[P, object]:
    def hosted(*args: P.args, **kwargs: P.kwargs) -> object:
        if getattr(_hosting, "inside", False):
            return call(*args, **kwargs)
        ranks = current()
        if ranks is None:
            if rank == home:
                return call(*args, **kwargs)
            raise UngatedShardedForward(
                f"{what} lives on GPU {home}'s process and was called outside a group attempt"
            )
        if ranks.rank != home:
            return ranks.cross(None, home, f"hosted {what}")
        _hosting.inside = True
        try:
            result = call(*args, **kwargs)
        finally:
            _hosting.inside = False
        return ranks.cross(result, home, f"hosted {what}")

    return hosted


class Placement:
    """This GPU's `author.Placement`: a scope whose components all live on other GPUs stages
    nothing here (their calls run on their home GPUs, `host_component`)."""

    def __init__(self, hosted: Mapping[str, int], rank: int) -> None:
        self.hosted, self.rank = dict(hosted), rank

    def remote(self, components: tuple[str, ...]) -> bool:
        return bool(components) and all(
            self.hosted.get(name, self.rank) != self.rank for name in components
        )


def signature(value: object, depth: int = 0) -> str:
    """`value`'s shape: tensor shapes and dtypes, files' kinds, containers' structure, and
    other values' types (numbers and text may differ between GPUs; their places may not)."""
    import torch

    if depth > 12:
        return "…"
    if isinstance(value, torch.Tensor):
        return f"{str(value.dtype).removeprefix('torch.')}{list(value.shape)}"
    if isinstance(value, msgspec.Struct):
        return _record(value, value.__struct_fields__, depth)
    kind = getattr(value, "kind", None)
    if isinstance(kind, str) and hasattr(value, "media_type"):
        return f"file:{kind}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(signature(v, depth + 1) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{signature(v, depth + 1)}" for k, v in value.items()) + "}"
    if dataclasses.is_dataclass(value):
        return _record(value, tuple(f.name for f in dataclasses.fields(value)), depth)
    return type(value).__name__


def _record(value: object, fields: tuple[str, ...], depth: int) -> str:
    inner = ",".join(f"{f}={signature(getattr(value, f), depth + 1)}" for f in fields)
    return f"{type(value).__name__}({inner})"
