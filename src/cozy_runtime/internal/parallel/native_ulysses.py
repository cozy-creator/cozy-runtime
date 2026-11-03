"""Native-selected Ulysses geometry with host agreement and supplied crossings.

Store messages carry metadata only. Every rank admits and physically allocates its
fixed choice before any tensor collective. Native selection is not a distributed
reservation: later foreign/library failures remain terminal and abort the group.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import tensorfs.varena.torch as vt
import torch
import torch.distributed as dist
from torch import Tensor

from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.parallel import copy_engine, native_attention
from cozy_runtime.internal.residency import _REQUEST

_UNPROVEN: list[Any] = []
_STATES: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


class _Control:
    def __init__(self, group: Any) -> None:
        from torch.distributed.distributed_c10d import _get_process_group_store

        self.store = dist.PrefixStore(
            f"cozy.native-ulysses/1/{group.group_name}/", _get_process_group_store(group)
        )
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.sequence = 0
        self.lock = threading.Lock()

    def gather(self, sequence: int, stage: str, value: Any) -> list[Any]:
        prefix = f"{sequence}/{stage}/"
        keys = [prefix + str(rank) for rank in range(self.world)]
        self.store.set(keys[self.rank], json.dumps(value, sort_keys=True, allow_nan=False))
        values = [json.loads(data) for data in self.store.multi_get(keys)]
        # Delete only after every rank has actually read this stage. Keys cannot
        # survive into a reused pass or grow with the lifetime of a long-lived group.
        done = [prefix + f"read/{rank}" for rank in range(self.world)]
        self.store.set(done[self.rank], b"1")
        if self.rank == 0:
            self.store.multi_get(done)
            for key in (*keys, *done):
                self.store.delete_key(key)
        return values


def _control(group: Any) -> _Control:
    with _LOCK:
        if group not in _STATES:
            _STATES[group] = _Control(group)
        return _STATES[group]


def _semantic(value: object) -> str:
    encoded = phase.configuration(value)
    if encoded is None:
        raise ValueError("native Ulysses requires immutable semantic metadata")
    return hashlib.sha256(encoded.encode()).hexdigest()


def _failure(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__qualname__, "message": str(error)}


def _require(rows: list[Any], stage: str) -> None:
    failed = [(rank, row["error"]) for rank, row in enumerate(rows) if row.get("error")]
    if failed:
        raise RuntimeError(f"native Ulysses {stage} refused before tensor collectives: {failed}")


@dataclass(frozen=True)
class Geometry:
    step: int
    rows: tuple[int, ...]
    heads: int
    dtype: torch.dtype
    local: native_attention.Prepared

    @property
    def identity(self) -> str:
        return f"heads:{self.step}/local:{self.step // len(self.rows)}"

    def specs(self, rank: int) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        world, rows, local = len(self.rows), self.rows[rank], self.step // len(self.rows)
        total, padded = sum(self.rows), self.heads + (-self.heads % world)
        values = [("output", (1, rows, padded, 128), self.dtype)]
        for letter in "qkv":
            values.extend(
                (
                    (f"send.{letter}", (world * rows, 1, local, 128), self.dtype),
                    (f"recv.{letter}", (total, 1, local, 128), self.dtype),
                    (f"land.{letter}", (1, total, local, 128), self.dtype),
                )
            )
        values.extend(
            (
                ("attended", (1, total, local, 128), self.dtype),
                ("send.back", (total, 1, local, 128), self.dtype),
                ("recv.back", (world * rows, 1, local, 128), self.dtype),
            )
        )
        values.extend(
            (f"attention.{name}", shape, dtype) for name, shape, dtype in self.local.specs()
        )
        return tuple(values)

    def allocations(self, rank: int) -> tuple[int, ...]:
        return tuple(
            math.prod(shape) * dtype.itemsize
            for _, shape, dtype in self.specs(rank)
            if math.prod(shape)
        )


def validate_transport(inputs: tuple[Tensor, ...], owners: dict[str, Tensor]) -> None:
    """Reject physical aliasing before packing or crossing writes.

    Read-only Q/K/V may be strided views. Compare their actual Storage allocation
    spans, not synthetic tensor bounds inferred from noncontiguous strides.
    """
    device = inputs[0].device
    written: list[Tensor] = []
    for target in owners.values():
        if (
            type(target) is not Tensor
            or target.device != device
            or not target.is_contiguous()
            or target.requires_grad
            or target.is_conj()
            or target.is_neg()
        ):
            raise ValueError("native Ulysses transport writes require ordinary contiguous owners")
        if not target.numel():
            continue
        begin, end = target.data_ptr(), target.data_ptr() + target.numel() * target.element_size()
        for source in (*inputs, *written):
            storage = source.untyped_storage()
            start, stop = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
            if torch._C._is_alias_of(target, source) or max(begin, start) < min(end, stop):
                raise ValueError(
                    "native Ulysses transport owner overlaps an actual live storage span"
                )
        written.append(target)


def _cross(
    group: Any,
    engines: Any,
    sends: list[Tensor],
    receives: list[Tensor],
    matrix: list[list[int]],
    rank: int,
) -> None:
    if engines is not None:
        engines.all_to_all_into(sends, matrix, receives)
        return
    works = [
        dist.all_to_all_single(
            out,
            value,
            output_split_sizes=[row[rank] for row in matrix],
            input_split_sizes=matrix[rank],
            group=group,
            async_op=True,
        )
        for value, out in zip(sends, receives, strict=True)
    ]
    for work in works:
        work.wait()


def pack_heads(source: Tensor, target: Tensor, start: int, world: int) -> None:
    """Strided source -> supplied head-first send pack, including declared head padding."""
    rows, local = source.shape[1], target.shape[2]
    heads = source.shape[2]
    grouped = target.view(world, rows, 1, local, 128)
    if start + world * local > heads:
        grouped.zero_()
    for peer in range(world):
        first = start + peer * local
        count = min(local, max(heads - first, 0))
        if count:
            grouped[peer, :, :, :count].permute(1, 0, 2, 3).copy_(
                source[:, :, first : first + count]
            )


def unpack_heads(source: Tensor, output: Tensor, start: int, world: int) -> None:
    """Supplied returned rank blocks -> exact strided slice of the fixed output owner."""
    rows, local = output.shape[1], source.shape[2]
    target = output[:, :, start : start + world * local].view(1, rows, world, local, 128)
    target.copy_(source.view(world, rows, 1, local, 128).permute(2, 1, 0, 3, 4))


def _execute(
    geometry: Geometry,
    values: dict[str, Tensor],
    inputs: tuple[Tensor, ...],
    group: Any,
    engines: Any,
    rank: int,
    record: dict[str, Any],
) -> Tensor:
    world = len(geometry.rows)
    step = geometry.step
    output = values["output"]
    sent = [[n] * world for n in geometry.rows]
    back = [list(geometry.rows) for _ in range(world)]
    scratch = {
        name.removeprefix("attention."): value
        for name, value in values.items()
        if name.startswith("attention.")
    }
    for start in range(0, output.shape[2], step):
        for letter, source in zip("qkv", inputs, strict=True):
            pack_heads(source, values[f"send.{letter}"], start, world)
        record["tensor_crossing_attempted"] = True
        _cross(
            group,
            engines,
            [values[f"send.{c}"] for c in "qkv"],
            [values[f"recv.{c}"] for c in "qkv"],
            sent,
            rank,
        )
        for c in "qkv":
            values[f"land.{c}"].copy_(values[f"recv.{c}"].permute(1, 0, 2, 3))
        geometry.local.execute(*(values[f"land.{c}"] for c in "qkv"), values["attended"], scratch)
        values["send.back"].copy_(values["attended"].permute(1, 0, 2, 3))
        _cross(group, engines, [values["send.back"]], [values["recv.back"]], back, rank)
        unpack_heads(values["recv.back"], output, start, world)
        record["completed_head_groups"] += 1
    return output[:, :, : geometry.heads]


def exchange(
    query: Tensor, key: Tensor, value: Tensor, group: Any, *, backend: str, options: dict[str, Any]
) -> Tensor:
    """One exact grouped exchange, admitted by every rank before the first crossing."""
    control = _control(group)
    # One model pass owns its group's collective order. Concurrent calls cannot
    # interleave this rank's sequence identities or native leaves.
    with control.lock:
        sequence = control.sequence
        control.sequence += 1
        try:
            return _exchange(control, sequence, query, key, value, group, backend, options)
        except BaseException as error:
            error.varena_no_retry = True
            raise


def _exchange(
    control: _Control,
    sequence: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    group: Any,
    backend: str,
    options: dict[str, Any],
) -> Tensor:
    request = _REQUEST.get()
    error: BaseException | None = None
    metadata: dict[str, Any] = {}
    try:
        if (
            request is None
            or request.idle
            or not request.memory.cuda
            or torch.is_grad_enabled()
            or catch.uncommitted_scope()
            or query.device != request.memory.device
            or query.dtype not in (torch.float16, torch.bfloat16)
            or query.dim() != 4
            or query.shape[0] != 1
            or query.shape[-1] != 128
            or query.shape[2] < 1
            or torch.cuda.is_current_stream_capturing()
            or any(
                type(t) is not Tensor
                or t.shape != query.shape
                or t.dtype != query.dtype
                or t.device != query.device
                or t.requires_grad
                or t.is_conj()
                or t.is_neg()
                or t.stride(-1) != 1
                for t in (query, key, value)
            )
        ):
            raise ValueError(
                "native Ulysses requires independent eager B1/D128 half-precision Q/K/V"
            )
        metadata = {
            "shape": list(query.shape),
            "dtype": str(query.dtype),
            "backend": backend,
            "scale": options.get("scale"),
            "mask": options.get("attn_mask") is not None,
            "causal": options.get("is_causal", False),
            "dropout": options.get("dropout_p", 0),
            "gqa": options.get("enable_gqa", False),
            "transport": "nccl" if copy_engine.for_group(group) is None else "copy_engines",
        }
    except BaseException as exc:
        error = exc
    rows = control.gather(
        sequence,
        "metadata",
        {"metadata": metadata, "error": None if error is None else _failure(error)},
    )
    _require(rows, "metadata")
    facts = [r["metadata"] for r in rows]
    common = [{**r, "shape": [r["shape"][0], None, *r["shape"][2:]]} for r in facts]
    if any(r != common[0] for r in common[1:]):
        raise ValueError("native Ulysses ranks disagree on immutable geometry or selected recipe")
    sizes = tuple(r["shape"][1] for r in facts)
    device, stream = query.device.index or 0, torch.cuda.current_stream(query.device)
    rank, world = control.rank, control.world
    shapes = (
        sizes,
        tuple(query.shape[:1]),
        int(query.shape[2]),
        str(query.dtype),
        backend,
        metadata["transport"],
        group.group_name,
    )
    phase_id = _semantic(("native-ulysses/1", shapes))
    declarations: tuple[Geometry, ...] = ()
    selection = None
    try:
        selected = native_attention.recipe(backend, options, sum(sizes), device)
        padded = query.shape[2] + (-query.shape[2] % world)
        groups = sorted(
            {padded, *(n for n in range(world, padded, world) if padded % n == 0)}, reverse=True
        )
        declarations = tuple(
            Geometry(
                step,
                sizes,
                query.shape[2],
                query.dtype,
                native_attention.prepare(
                    selected,
                    sum(sizes),
                    step // world,
                    query.dtype,
                    query.device,
                    stream.cuda_stream,
                ),
            )
            for step in groups
        )
        menu = tuple((g.identity, g.allocations(rank)) for g in declarations)
        local_key = "runtime.ulysses.local/1:" + _semantic(
            (phase_id, rank, menu, tuple(g.local.identity for g in declarations))
        )
        request.memory.remember_working_phase(local_key, (request,))
        selection = vt.split_select(
            device, local_key, menu, preferred=menu[0][0], stream=stream.cuda_stream
        )
        error = None
    except BaseException as exc:
        error = exc
    proposals = control.gather(
        sequence,
        phase_id + "/eligible",
        {
            "identity": None
            if not declarations
            else _semantic(tuple(g.local.identity for g in declarations)),
            "eligible": [] if selection is None else selection["eligible"],
            "error": None if error is None else _failure(error),
        },
    )
    _require(proposals, "native selection")
    if any(row["identity"] != proposals[0]["identity"] for row in proposals[1:]):
        raise ValueError(
            "native Ulysses ranks disagree on actual provider recipe/artifact identity"
        )
    eligible = set(proposals[0]["eligible"]).intersection(
        *(set(p["eligible"]) for p in proposals[1:])
    )
    common_menu = tuple(
        (g.identity, g.allocations(rank)) for g in declarations if g.identity in eligible
    )
    chosen = None
    error = None
    if rank == 0:
        try:
            if not common_menu:
                raise ValueError("native Ulysses ranks have no common legal geometry")
            common_key = "runtime.ulysses.common/1:" + _semantic(
                (phase_id, common_menu, proposals[0]["identity"])
            )
            request.memory.remember_working_phase(common_key, (request,))
            chosen = vt.split_select(
                device,
                common_key,
                common_menu,
                preferred=common_menu[0][0],
                stream=stream.cuda_stream,
            )["choice_id"]
        except BaseException as exc:
            error = exc
    agreed = control.gather(
        sequence,
        phase_id + "/choice",
        {"choice": chosen, "error": None if error is None else _failure(error)},
    )
    _require(agreed, "common native choice")
    identity = agreed[0]["choice"]
    if identity not in eligible:
        raise RuntimeError(
            "native Ulysses agreed choice is not every rank's declared legal geometry"
        )
    geometry = next(g for g in declarations if g.identity == identity)
    fixed = ((identity, geometry.allocations(rank)),)
    fixed_key = "runtime.ulysses.fixed/1:" + _semantic(
        (phase_id, rank, fixed, geometry.local.identity)
    )
    request.memory.remember_working_phase(fixed_key, (request,))
    token, owners = None, {}
    submitted = False
    committed = False
    record = {
        "site": "native-ulysses",
        "sequence": sequence,
        "shape_identity": phase_id,
        "geometry": identity,
        "rank": rank,
        "stream": int(stream.cuda_stream),
        "provider_identity": geometry.local.identity,
        "provider_sol_query": dict(geometry.local.sol_plan),
        "buffer_inventory": tuple(
            (name, shape, str(dtype)) for name, shape, dtype in geometry.specs(rank)
        ),
        "rows": sizes,
        "eligible": selection["eligible"],
        "all_rank_eligible": [p["eligible"] for p in proposals],
        "additional_allocations": geometry.allocations(rank),
        "library_holdings": "ProcessGroup/copy-engine owners; library pressure is not reserved",
        "tensor_crossing_attempted": False,
        "submission_started": False,
        "completed_head_groups": 0,
        "checked_stream_completion": False,
    }
    request.working_splits[fixed_key] = record
    custody = [query, key, value, geometry, owners, group, stream]
    try:
        error = None
        final = None
        try:
            final = vt.split_select(
                device, fixed_key, fixed, preferred=identity, stream=stream.cuda_stream
            )
            record["selection"] = final
            phase.choice(("native-ulysses", fixed_key, identity))
            token = phase.begin_split(device, fixed_key, identity, final, stream=stream.cuda_stream)
            for name, shape, dtype in geometry.specs(rank):
                owners[name] = torch.empty(shape, dtype=dtype, device=query.device)
            scratch = {
                name.removeprefix("attention."): tensor
                for name, tensor in owners.items()
                if name.startswith("attention.")
            }
            validate_transport((query, key, value), owners)
            geometry.local.validate(
                *(owners[f"land.{c}"] for c in "qkv"), owners["attended"], scratch
            )
        except BaseException as exc:
            error = exc
        ready = control.gather(
            sequence,
            phase_id + "/" + identity + "/ready",
            {
                "choice": identity,
                "physical_owners_allocated": error is None,
                "native_selection": final,
                "error": None if error is None else _failure(error),
            },
        )
        record["all_rank_readiness"] = ready
        _require(ready, "fixed admission/allocation")
        committed = True
        engines = copy_engine.for_group(group)
        custody.append(engines)
        record["transport"] = "nccl" if engines is None else "copy_engines"
        record["submission_started"] = True
        result = _execute(geometry, owners, (query, key, value), group, engines, rank, record)
        stream.synchronize()
        record["checked_stream_completion"] = True
        submitted = True
        return result
    except BaseException as exc:
        exc.varena_no_retry = True
        record["failure"] = _failure(exc)
        if committed:
            from cozy_runtime.internal.parallel.group import abort_process_group

            abort_process_group()
        try:
            stream.synchronize()
        except BaseException:
            _UNPROVEN.append(custody)
            raise
        raise
    finally:
        if token is not None:
            record["submission"] = vt.split_end(device, token, submitted=submitted)
