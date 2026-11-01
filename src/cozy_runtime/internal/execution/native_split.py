"""One closed native split consumer; shape geometry stays beside its actual kernel."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor

from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author.kernels import Launch, TritonEntry, read
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.residency import _REQUEST


def _layout(value: Tensor | None) -> object:
    if value is None:
        return None
    return tuple(value.shape), tuple(value.stride()), str(value.dtype), value.data_ptr() % 16


def _divisors(batch: int) -> tuple[int, ...]:
    found = {1, batch}
    for part in range(2, math.isqrt(batch) + 1):
        if batch % part == 0:
            found.update((part, batch // part))
    return tuple(sorted(found, reverse=True))


def channel_norm(
    site: str, launch: Launch, entry: TritonEntry, x: Tensor, weight: Tensor,
    bias: Tensor | None, scale: float, eps: float,
) -> Tensor | None:
    """Split only a noncontiguous, batched fused norm with one bounded input copy.

    None means this layout/context is outside this adapter, not a capacity refusal.
    Selection and execution exhaustion propagate; no started operation is retried.
    """
    request = _REQUEST.get()
    if (
        request is None or request.idle or not request.memory.cuda
        or type(x) is not Tensor or x.layout is not torch.strided or not x.is_cuda
        or x.device != request.memory.device
        or x.dim() < 3 or x.shape[0] < 2 or x.is_contiguous()
        or torch.is_grad_enabled() or catch.uncommitted_scope()
        or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        return None
    batch = int(x.shape[0])
    output_bytes = x.numel() * x.element_size()
    sample_bytes = output_bytes // batch
    if not sample_bytes or sample_bytes % 16:
        return None
    from cozy_runtime.internal.triton_kernels import _Launcher

    if type(launch) is not _Launcher:
        return None
    device = x.device.index or 0
    if not vt.split_context(device)["eligible"]:
        return None
    source = getattr(entry.kernel, "cache_key", None)
    if not isinstance(source, str):
        return None
    # The compiled launcher normally binds lazy weights at launch. Prepare these inputs
    # before the leaf interval, so no source/decode allocation hides inside its menu.
    weight, bias = read(weight), read(bias)
    chunks = _divisors(batch)
    geometry = {f"batch:{chunk}": chunk for chunk in chunks}
    choices: list[tuple[str, tuple[int, ...]]] = []
    for choice_id, chunk in geometry.items():
        # These are metadata-only views. Equal chunks have one allocation class; a
        # same-stream copy buffer is released before the next equal-size copy starts.
        copy = any(
            not (part := x.narrow(0, at, chunk)).is_contiguous() or part.data_ptr() % 16
            for at in range(0, batch, chunk)
        )
        allocations = (output_bytes, chunk * sample_bytes) if copy else (output_bytes,)
        choices.append((choice_id, allocations))
    with request.memory.lock:
        models = tuple(
            model for model in request.memory.models.values() if model.name in request.current
        )
    identities = tuple((model.content, model.wanted_identity()) for model in models)
    if any(identity is None for _, identity in identities):
        return None
    semantic = phase.configuration((
        "channel-norm-batch/1", site,
        tuple(sorted(identities)),
        _layout(x), _layout(weight), _layout(bias),
        ("contiguous-output", tuple(x.shape), str(x.dtype)),
        source, launch.key, entry.key, tuple(sorted(entry.signature.items())),
        tuple(sorted(entry.constexprs.items())), tuple(sorted(entry.options.items())),
        entry.aligned,
        float(scale), float(eps), tuple(choices),
    ))
    if semantic is None:
        return None
    key = "runtime.split.channel_norm/1:" + hashlib.sha256(semantic.encode()).hexdigest()
    # The request owns a componentless menu only until it ends. Reusable model owners
    # retain their exact records even when the enclosing phase key itself is unknown.
    request.memory.remember_working_phase(key, models or (request,))
    stream = torch.cuda.current_stream(device)
    stream_id = int(stream.cuda_stream)
    choice = vt.split_select(
        device, key, choices, preferred=f"batch:{batch}", stream=stream_id,
    )
    choice_id = str(choice["choice_id"])
    chunk = geometry[choice_id]
    record: dict[str, Any] = request.working_splits.setdefault(
        key, {"site": site, "key": key, "selections": 0},
    )
    record.update(selection=dict(choice), batch_chunk=chunk)
    record["selections"] += 1
    from cozy_runtime.fusion import _kernels

    observation_id = vt.split_begin(
        device, key, choice_id, stream=stream_id,
        completion_owner=phase.completion_owner(device, stream_id),
    )
    try:
        with torch.cuda.device(device), torch.cuda.stream(stream):
            out = torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
            for at in range(0, batch, chunk):
                catch.checkpoint()
                _kernels.channel_norm(
                    launch, entry, x.narrow(0, at, chunk), weight, bias, scale, eps,
                    out=out.narrow(0, at, chunk),
                )
    except BaseException as error:
        # The private output may already contain earlier chunks. Preserve the original
        # error and prohibit an enclosing legacy cut from replaying this operation.
        if fault(error) is Fault.OOM:
            error.varena_no_retry = True  # type: ignore[attr-defined]
        try:
            record["submission"] = vt.split_end(device, observation_id, submitted=False)
        except BaseException as cleanup:
            raise cleanup from error
        raise
    record["submission"] = vt.split_end(device, observation_id, submitted=True)
    # Python return is not completion. The matching phase's existing stream proof is
    # the only publication point; outside it this is declaration-only execution.
    return out
