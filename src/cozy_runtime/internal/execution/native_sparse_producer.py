"""Complete selected Kitchen producer: exact queries, owned scratch, one final mean commit."""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import Tensor

from cozy_runtime import _kernels
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author._split import Cut
from cozy_runtime.author.kernels import _LINEAR_SPLIT, _RECENT, _SHARED, SparseProjectedHeads, read
from cozy_runtime.internal import attention, int8_linear, kernel_ready, prebuilt
from cozy_runtime.internal.execution import catch, phase
from cozy_runtime.internal.execution.native_heads import (
    _binary_identity,
    _head_chunks,
    _Projection,
    _Quantized,
    _RowInvocation,
)
from cozy_runtime.internal.execution.native_linear_split import _Dispatch, _layout
from cozy_runtime.internal.execution.native_residual_ffn import (
    _entry_identity,
    _leaf,
    _overlaps,
    _plain,
)
from cozy_runtime.internal.residency import _REQUEST
from cozy_runtime.rows import rows as projection_rows

CHUNK = 4096


@dataclass(frozen=True)
class _Geometry:
    heads: int
    projections: tuple[tuple[int, int, bool, _Dispatch], ...]
    sol: tuple[tuple[int, tuple[tuple[str, int], ...]], ...]
    quantized_rows: int
    gemm_bytes: int
    sol_bytes: int

    @property
    def identity(self) -> str:
        calls = "/".join(
            f"{m}:{h}:{int(packed)}:{d.kind}:{d.config}" for m, h, packed, d in self.projections
        )
        return f"heads:{self.heads}/chunks:{CHUNK}/{calls}"

    def dispatch(self, rows: int, heads: int, packed: bool) -> _Dispatch:
        return next(d for m, h, p, d in self.projections if (m, h, p) == (rows, heads, packed))


def _kernel(backend: object, device: int) -> tuple[Any, tuple[Any, ...]] | None:
    name = getattr(backend, "value", str(backend))
    candidate = next((c for c in attention.known().values() if c.backend == name), None)
    original = attention.BY_NAME["kitchen-sol-int4qk"]
    if (
        candidate is None
        or candidate.producer is None
        or candidate.producer.base != original.name
        or candidate.producer.context_parallel
        or candidate.dense != original.dense
        or candidate.local.forward is not original.local.forward
        or candidate.local.impl is not original.local.impl
    ):
        return None
    ready = next(
        (
            r
            for r in kernel_ready.served(f"attention cuda:{device} ")
            if isinstance(r, attention.Ready) and r.candidate is candidate
        ),
        None,
    )
    extension = getattr(sys.modules.get("comfy_kitchen.backends.cuda"), "_C", None)
    filename = getattr(extension, "__file__", None)
    if (
        ready is None
        or not ready.artifact
        or not filename
        or not all(
            callable(getattr(extension, method, None))
            for method in (
                "sol_attn_plan",
                "sol_producer_begin",
                "sol_producer_chunk",
                "sol_attn_core",
            )
        )
    ):
        return None
    path = Path(filename)
    if not path.is_file():
        return None
    stat = path.stat()
    return extension, (
        "kitchen-sol-producer/1",
        candidate.name,
        tuple(sorted(ready.artifact.items())),
        _binary_identity(str(path), stat.st_size, stat.st_mtime_ns),
    )


def _entries(intent: SparseProjectedHeads, device: int) -> tuple[Any, ...] | None:
    from cozy_runtime.author.kernels import _SETS
    from cozy_runtime.internal.triton_kernels import _LOADED, _Launcher

    launch = intent.launch
    if type(launch) is not _Launcher or not launch.key:
        return None
    sets = tuple(
        _SETS[name]
        for (name, index), loaded in _LOADED.items()
        if loaded is launch and index == device and name in _SETS
    )
    expected = (
        (
            intent.value_absmax,
            {
                "value_ptr": "*bf16",
                "maximum_ptr": "*fp32",
                "rows": "i32",
                "channels": "i32",
                "stride_t": "i32",
                "BLOCK_T": "constexpr",
                "BLOCK_C": "constexpr",
            },
            {"BLOCK_T": 64, "BLOCK_C": 32},
        ),
        (
            intent.key_sum,
            {
                "key_ptr": "*fp32",
                "out_ptr": "*fp32",
                "tokens": "i32",
                "stride_t": "i32",
                "HEAD_DIM": "constexpr",
                "BLOCK_T": "constexpr",
            },
            {"HEAD_DIM": 128, "BLOCK_T": 64},
        ),
    )
    for entry, signature, constants in expected:
        if (
            dict(entry.signature) != signature
            or dict(entry.constexprs) != constants
            or not any(entry in declared.entries for declared in sets)
        ):
            return None
    return (
        launch.key,
        tuple((s.name, s.identity) for s in sets),
        tuple(_entry_identity(entry) for entry, _, _ in expected),
    )


def _capsule(value: Tensor) -> Any:
    # Preparation already rejects grad-enabled execution. Match the pinned Kitchen
    # inference boundary: Parameter metadata may still require grad under no_grad.
    if value.requires_grad:
        value = value.detach()  # owning storage alias, no copy or original-flag mutation
    return value.__dlpack__(stream=-1)


@dataclass
class _Plan:
    intent: SparseProjectedHeads
    request: Any
    library: Any
    kernel: Any
    kernel_identity: tuple[Any, ...]
    projections: tuple[_Projection, ...]
    norm_q: Tensor
    norm_k: Tensor
    rotary: Tensor
    prior: Tensor | None
    destination: Tensor
    geometries: dict[str, _Geometry]
    choices: list[tuple[str, tuple[int, ...]]]
    key: str
    stream: Any

    def run(self) -> Tensor:
        device, stream = self.intent.hidden.device.index or 0, int(self.stream.cuda_stream)
        selection = vt.split_select(
            device, self.key, self.choices, preferred=self.choices[0][0], stream=stream
        )
        geometry = self.geometries[str(selection["choice_id"])]
        record = self.request.working_splits.setdefault(
            self.key,
            {"site": "sparse-projected-heads", "key": self.key, "selections": 0},
        )
        record.update(
            selection=dict(selection),
            head_chunk=geometry.heads,
            statistics="bootstrap" if self.prior is None else "rolling",
            kernel_identity=self.kernel_identity,
            choices=self.choices,
            token_chunk=CHUNK,
            sol_workspace_bytes=geometry.sol_bytes,
            gemm_workspace_bytes=geometry.gemm_bytes,
        )
        record["selections"] += 1
        phase.choice(("native-split", self.key, geometry.identity))
        observation = phase.begin_split(
            device, self.key, geometry.identity, selection, stream=stream
        )
        submitted = False
        try:
            result = self.execute(geometry)
            submitted = True
            return result
        except BaseException as error:
            if fault(error) is Fault.OOM:
                error.varena_no_retry = True  # type: ignore[attr-defined]
            raise
        finally:
            record["submission"] = vt.split_end(device, observation, submitted=submitted)

    def execute(self, geometry: _Geometry) -> Tensor:
        intent, x = self.intent, self.intent.hidden
        tokens, width, count = int(x.shape[1]), int(x.shape[2]), geometry.heads
        device, stream = x.device.index or 0, int(self.stream.cuda_stream)
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            # Allocate the entire accepted menu before projection or any statistic write.
            output = torch.empty((1, tokens, intent.heads, 128), dtype=x.dtype, device=x.device)
            next_mean = torch.empty((intent.heads, 128), dtype=torch.float32, device=x.device)
            packed = torch.empty(
                (min(CHUNK, tokens), 3 * count * 128), dtype=x.dtype, device=x.device
            )
            window = (
                torch.empty((1, tokens, count, 128), dtype=x.dtype, device=x.device)
                if count < intent.heads
                else None
            )
            q = (
                torch.empty((geometry.quantized_rows, width), dtype=torch.int8, device=x.device)
                if geometry.quantized_rows
                else None
            )
            scales = (
                torch.empty(geometry.quantized_rows, dtype=torch.float32, device=x.device)
                if geometry.quantized_rows
                else None
            )
            gemm = (
                torch.empty(geometry.gemm_bytes, dtype=torch.uint8, device=x.device)
                if geometry.gemm_bytes
                else None
            )
            workspace = torch.empty(geometry.sol_bytes, dtype=torch.uint8, device=x.device)
            vscale = torch.empty((count, 128), dtype=torch.float32, device=x.device)
            vamax = torch.empty_like(vscale)
            bootstrap_mean = torch.empty_like(vscale) if self.prior is None else None
            denominator = (
                torch.empty((), dtype=torch.float32, device=x.device)
                if self.prior is None
                else None
            )
            if denominator is not None:
                denominator.fill_(tokens)
            for first in range(0, intent.heads, count):
                catch.checkpoint()
                heads = min(count, intent.heads - first)
                channels = heads * 128
                layout = dict(next(p for h, p in geometry.sol if h == heads))
                target = (
                    output
                    if window is None
                    else window.view(-1)[: tokens * channels].view(1, tokens, heads, 128)
                )
                scale, peak = vscale[:heads], vamax[:heads]
                if bootstrap_mean is not None:
                    prior = bootstrap_mean[:heads]
                else:
                    assert self.prior is not None
                    prior = self.prior[first : first + heads]

                def project(
                    index: int,
                    part: Tensor,
                    out: Tensor,
                    dispatch: _Dispatch,
                    quantization: _Quantized,
                    *,
                    first: int = first,
                    heads: int = heads,
                ) -> None:
                    _RowInvocation(
                        self.projections[index].module,
                        self.projections[index],
                        part,
                        out,
                        first * 128,
                        (first + heads) * 128,
                        dispatch,
                        q,
                        scales,
                        gemm,
                        quantization,
                        self.library,
                        stream,
                    ).call()

                if self.prior is not None:
                    scale.zero_()
                    for start in range(0, tokens, CHUNK):
                        part = x[:, start : start + CHUNK]
                        rows = int(part.shape[1])
                        values = target[:, start : start + rows].flatten(2, 3)
                        project(
                            2, part, values, geometry.dispatch(rows, heads, False), _Quantized()
                        )
                        intent.launch(
                            intent.value_absmax.kernel,
                            ((rows + 63) // 64, (channels + 31) // 32, 1),
                            values,
                            scale,
                            rows,
                            channels,
                            values.stride(-2),
                            64,
                            32,
                        )
                    scale.div_(127.0).mul_(1.1).clamp_min_(1e-8)
                else:
                    prior.zero_()
                    scale.fill_(1)

                # Bootstrap measures first, then repeats with its real statistics.
                # Rolling reads already-projected current-call V and prior key means.
                for bootstrap in (True, False) if self.prior is None else (False,):
                    self.kernel.sol_producer_begin(
                        _capsule(workspace), 1, tokens, heads, stream, token_aug=0
                    )
                    for start in range(0, tokens, CHUNK):
                        part = x[:, start : start + CHUNK]
                        rows = int(part.shape[1])
                        chunk = packed.view(-1)[: rows * 3 * channels].view(1, rows, 3 * channels)
                        dispatch, shared = geometry.dispatch(rows, heads, True), _Quantized()
                        project(0, part, chunk[..., :channels], dispatch, shared)
                        project(1, part, chunk[..., channels : 2 * channels], dispatch, shared)
                        if self.prior is None:
                            project(2, part, chunk[..., 2 * channels :], dispatch, shared)
                        else:
                            chunk[..., 2 * channels :].copy_(
                                target[:, start : start + rows].flatten(2, 3)
                            )
                        self.kernel.sol_producer_chunk(
                            _capsule(workspace),
                            _capsule(chunk),
                            _capsule(self.rotary),
                            _capsule(self.norm_q),
                            _capsule(self.norm_k),
                            _capsule(prior),
                            _capsule(scale),
                            intent.eps,
                            96,
                            start,
                            rows,
                            1,
                            tokens,
                            heads,
                            stream,
                            block_len=None,
                            token_aug=0,
                        )
                    if bootstrap:
                        assert denominator is not None
                        sums = workspace[
                            layout["scratch"] : layout["scratch"] + heads * layout["NPAD"] * 128 * 4
                        ]
                        sums = sums.view(torch.float32).view(heads, layout["NPAD"], 128)
                        for head in range(heads):
                            intent.launch(
                                intent.key_sum.kernel,
                                (1, 1, 1),
                                sums[head],
                                prior[head],
                                layout["NTB"],
                                128,
                                128,
                                64,
                            )
                        prior.div_(denominator)
                        scale.copy_(
                            workspace[layout["statsV"] : layout["statsV"] + heads * 128 * 4]
                            .view(torch.float32)
                            .view(heads, 128)
                        )
                        scale.div_(127.0).mul_(1.1).clamp_min_(1e-8)
                sink = (intent.protected_prefix + 63) // 64
                self.kernel.sol_attn_core(
                    _capsule(workspace),
                    _capsule(target),
                    _capsule(scale),
                    _capsule(next_mean[first : first + heads]),
                    _capsule(peak),
                    1,
                    tokens,
                    heads,
                    intent.tau,
                    128**-0.5,
                    0,
                    sink,
                    0,
                    sink,
                    stream,
                    threshold=None,
                    block_len=None,
                    tail=True,
                    token_aug=0,
                )
                if window is not None:
                    output[:, :, first : first + heads].copy_(target)
            # All prior head means have been consumed. Only this final same-stream copy
            # changes caller state; refusal or a failed head window cannot partially commit it.
            self.destination.copy_(next_mean)
            return output


def prepare_sparse_projected_heads(
    intent: SparseProjectedHeads,
    output: Any,
    cuts: tuple[Cut, ...],
) -> _Plan | None:
    request = _REQUEST.get()
    if (
        type(intent) is not SparseProjectedHeads
        or request is None
        or request.idle
        or not request.memory.cuda
    ):
        return None
    x = intent.hidden
    if (
        output is not None
        or type(x) is not Tensor
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.device != request.memory.device
        or x.dim() != 3
        or x.shape[0] != 1
        or not x.is_contiguous()
        or not x.numel()
        or x.data_ptr() % 16
        or cuts != (Cut(2),)
        or type(intent.heads) is not int
        or intent.heads < 1
        or type(intent.protected_prefix) is not int
        or not 0 <= intent.protected_prefix <= x.shape[1]
        or type(intent.eps) not in (int, float)
        or not math.isfinite(intent.eps)
        or intent.eps <= 0
        or type(intent.tau) not in (int, float)
        or not math.isfinite(intent.tau)
        or intent.tau < 0
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or catch.uncommitted_scope()
        or _LINEAR_SPLIT.get() is not None
        or torch.cuda.is_current_stream_capturing()
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
        or torch.nn.modules.module._global_forward_pre_hooks
        or torch.nn.modules.module._global_forward_hooks
        or any(not _leaf(module, x.dtype) for module in (intent.query, intent.key, intent.value))
    ):
        return None
    tokens, width, device = int(x.shape[1]), int(x.shape[2]), x.device.index or 0
    if (
        tokens * intent.heads * 128 > 2**31 - 1
        or (tokens + 63) // 64 > 65535
        or int8_linear.serving(device) != int8_linear.OP
        or not vt.split_context(device)["eligible"]
    ):
        return None
    held = getattr(_RECENT, "held", None)
    if any(shared.tensor is x for shared in _SHARED.get()) or (
        held is not None and held.tensor is not None and held.tensor() is x
    ):
        return None
    kernel, entries = _kernel(intent.backend, device), _entries(intent, device)
    library = int8_linear.library(device)
    if kernel is None or entries is None or library.family != "mma.sync":
        return None
    extension, kernel_identity = kernel
    norm_q, norm_k, rotary, prior, destination = (
        read(v)
        for v in (intent.norm_q, intent.norm_k, intent.rotary, intent.prior_mean, intent.next_mean)
    )
    if any(
        not _plain(v, x.device, x.dtype) or tuple(v.shape) != (128,) or not v.is_contiguous()
        for v in (norm_q, norm_k)
    ):
        return None
    if (
        not _plain(rotary, x.device, torch.float32)
        or tuple(rotary.shape) != (tokens, 96, 2)
        or not rotary.is_contiguous()
    ):
        return None
    for tensor in (destination,) if prior is None else (destination, prior):
        if (
            type(tensor) is not Tensor
            or not _plain(tensor, x.device, torch.float32)
            or tuple(tensor.shape) != (intent.heads, 128)
            or not tensor.is_contiguous()
        ):
            return None
    projections = []
    for module in (intent.query, intent.key, intent.value):
        data, scale, bias = read(module.data), read(module.scale), read(module.bias)
        if (
            tuple(data.shape) != (intent.heads * 128, width)
            or scale.data_ptr() % 16
            or (
                bias is not None
                and (bias.dtype != x.dtype or not bias.is_contiguous() or bias.data_ptr() % 16)
            )
        ):
            return None
        library._weight(prebuilt.INT8, data, scale, bias, width, x.dtype)
        call = projection_rows(module)
        if call is None:
            return None
        projections.append(_Projection(module, data, scale, bias, call))
    readonly = (
        x,
        norm_q,
        norm_k,
        rotary,
        *(v for p in projections for v in (p.data, p.scale, p.bias) if v is not None),
    )
    if any(_overlaps(destination, v) for v in readonly):
        return None
    queried, sol = {}, {}
    row_counts = tuple(dict.fromkeys((min(CHUNK, tokens), tokens % CHUNK or min(CHUNK, tokens))))

    def query(rows: int, heads: int, packed: bool) -> _Dispatch | None:
        key = (rows, heads, packed)
        if key not in queried:
            channels = heads * 128
            info = _kernels._gemm.int8_plan(
                device, False, rows, channels, width, channels * (3 if packed else 1)
            )
            if info["status"] == _kernels._gemm.NOT_SUPPORTED:
                return None
            library._check(info["status"], "the sparse projection plan")
            queried[key] = _Dispatch(rows, info["kind"], info["config"], info["workspace_bytes"])
        return queried[key]

    geometries, choices = {}, []
    for chunk in _head_chunks(intent.heads):
        head_counts = tuple(dict.fromkeys((chunk, intent.heads % chunk or chunk)))
        dispatches = []
        for heads in head_counts:
            if heads not in sol:
                sol[heads] = extension.sol_attn_plan(1, tokens, heads, token_aug=0)
            for rows in row_counts:
                for packed in (True,) if prior is None else (False, True):
                    dispatch = query(rows, heads, packed)
                    if dispatch is None:
                        return None
                    dispatches.append((rows, heads, packed, dispatch))
        quantized_rows = max((d.rows for _, _, _, d in dispatches if d.kind == 0), default=0)
        gemm_bytes = max(d.workspace for _, _, _, d in dispatches)
        sol_bytes = max(sol[h]["total"] for h in head_counts)
        geometry = _Geometry(
            chunk,
            tuple(dispatches),
            tuple((h, tuple(sorted(sol[h].items()))) for h in head_counts),
            quantized_rows,
            gemm_bytes,
            sol_bytes,
        )
        allocations = [
            2 * tokens * intent.heads * 128,
            4 * intent.heads * 128,
            2 * min(CHUNK, tokens) * 3 * chunk * 128,
        ]
        if chunk < intent.heads:
            allocations.append(2 * tokens * chunk * 128)
        if quantized_rows:
            allocations.extend((quantized_rows * width, quantized_rows * 4))
        if gemm_bytes:
            allocations.append(gemm_bytes)
        allocations.extend((sol_bytes, 4 * chunk * 128, 4 * chunk * 128))
        if prior is None:
            allocations.extend((4 * chunk * 128, 4))
        geometries[geometry.identity] = geometry
        choices.append((geometry.identity, tuple(allocations)))
    with request.memory.lock:
        models = tuple(
            model for model in request.memory.models.values() if model.name in request.current
        )
    identities = tuple((model.content, model.wanted_identity()) for model in models)
    if any(identity is None for _, identity in identities):
        return None
    semantic = phase.configuration(
        (
            "sparse-projected-heads/1",
            tuple(sorted(identities)),
            tuple(_layout(v) for v in readonly),
            _layout(prior),
            _layout(destination),
            intent.heads,
            intent.eps,
            intent.tau,
            intent.protected_prefix,
            "bootstrap" if prior is None else "rolling",
            CHUNK,
            kernel_identity,
            entries,
            library.identity,
            library.family,
            tuple(p.module.encoding for p in projections),
            tuple(choices),
            tuple(geometries.values()),
        )
    )
    if semantic is None:
        return None
    key = "runtime.split.sparse_projected_heads/1:" + hashlib.sha256(semantic.encode()).hexdigest()
    request.memory.remember_working_phase(key, models or (request,))
    return _Plan(
        intent,
        request,
        library,
        extension,
        kernel_identity,
        tuple(projections),
        norm_q,
        norm_k,
        rotary,
        prior,
        destination,
        geometries,
        choices,
        key,
        torch.cuda.current_stream(device),
    )


def split_sparse_projected_heads(
    intent: SparseProjectedHeads,
    output: Any,
    cuts: tuple[Cut, ...],
) -> Tensor | None:
    """Matched sparse control: decline before native admission or execution."""
    return None
