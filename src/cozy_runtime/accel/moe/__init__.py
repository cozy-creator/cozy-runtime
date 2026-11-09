"""Mixture-of-experts layers on Runtime's weight paging: every routed expert is its own block.

    from cozy_runtime.accel import moe

    experts = [moe.RoutedExpert(4096, 3072, gate_first=False) for _ in range(64)]
    self.experts = moe.Experts(experts, layer=3)
    out = self.experts(hidden, top_k_index, top_k_weights)

An expert is two Linears, `gate_and_up_proj` [2I, H] and `down_proj` [H, I], so every lane's leaf
serves it (int8.convrot, w4a8, nvfp4) and varena pages it on its own (`RoutedExpert` in the model's
`_no_split_modules`). `split` turns Transformers' fused experts (`gate_up_proj` [E, 2I, H],
`down_proj` [E, H, I]) into these; the checkpoint stores them per expert (TensorFS splits fused
uploads at ingest).

A few rows (decode, a verify pass) run as one grouped GEMV per projection over every (token,
expert) pair; more (a prefill, a denoise step) as one grouped GEMM per projection over the pairs
sorted by expert on the device. Each reads its expert's weight through a table of addresses
(`install`, bf16 and int8.convrot): routing stays on the device, so a decode or denoise step
records as one graph. They serve only while every expert is bound (an all-resident model, its
blocks leased by a loop); otherwise each hit expert runs its own Linears after one stable sort,
the host reading the counts, and a capture refuses there, typed.

`Routing` records which experts each forward used and replays expert caches over that sequence:
least recently used, and next use (Belady's MIN, the fewest bytes any cache of that size moves).
"""

from __future__ import annotations

import heapq
import statistics
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import partial

import torch
from torch import nn

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._model import Building
from cozy_runtime.accel.kernels import (
    INT8_CONVROT,
    KernelsUnavailable,
    Launch,
    Plan,
    TritonKernels,
)
from cozy_runtime.graphs import say
from cozy_runtime.internal import int8_linear
from cozy_runtime.accel.moe import _kernels

#: Routed-expert caches a `Routing` replays, in experts.
CACHES = (256, 512, 1024)
#: The most (token, expert) pairs a layer runs as a grouped GEMV per projection: decode's and a
#: verify pass's largest batches (16 rows of 8 experts). More run as a grouped GEMM.
GROUPED = 128
KERNELS = TritonKernels("cozy-moe-experts", _kernels.entries(), origin=__name__)
#: While the grouped kernels build, a capture waits for them (`Building`).
COMPILING = "compiling: it serves from the next request"
#: The grouped kernels' launcher once they serve, else why they do not.
_LAUNCH: list[Launch | str] = ["the model's execution plan installs no grouped kernels"]


class RoutedExpert(nn.Module):  # type: ignore[misc]
    """One routed SwiGLU expert. `gate_first`: `act(gate) * up` with the gate the first half of
    `gate_and_up_proj`'s rows (Qwen's), else `first * act(second)` (Hunyuan's)."""

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        *,
        gate_first: bool,
        act: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate_and_up_proj = nn.Linear(
            hidden, 2 * intermediate, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)
        self.gate_first, self.act = gate_first, act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.gate_and_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act(a) * b if self.gate_first else a * self.act(b))


class Experts(nn.ModuleList):  # type: ignore[misc]
    """A layer's routed experts. `forward(hidden [T, H], index [T, k], weights [T, k])`: each
    (token, k) pair's expert output, summed by `weights` in the outputs' dtype.

    While every expert is bound and of one kind the grouped kernels serve (`install`), routed on
    the device so a step records as a graph: a GEMV per projection for up to `GROUPED` pairs,
    else a GEMM over the pairs sorted by expert. Otherwise each hit expert multiplies its rows,
    gathered by one stable sort of the flat top-k index (ascending (token, k)), experts in index
    order, and the outputs return to their (token, k) slots: the host waits once a layer, for
    the counts, so a capture refuses that path."""

    #: Set to `trace(layer, counts)` to observe routing: tokens per expert, every forward. The
    #: counts are read on the host, so a traced layer records no graph.
    trace: Callable[[int, list[int]], None] | None = None

    def __init__(self, experts: Iterable[RoutedExpert], *, layer: int) -> None:
        super().__init__(experts)
        self.layer = layer
        #: The grouped kernels' address table on the device, the addresses it holds, and whether
        #: they and the experts' widths take the GEMM's 16-byte reads.
        self._table: torch.Tensor | None = None
        self._addresses: list[int] = []
        self._aligned = False

    def forward(
        self, hidden: torch.Tensor, index: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        tokens, k = index.shape
        out = self._grouped(hidden, index)
        if isinstance(out, str):
            self._uncaptured(hidden, out)
            out = self._sorted(hidden, index)
        elif self.trace is not None:
            self._uncaptured(hidden, "a routing trace reads the counts on the host")
            self.trace(self.layer, torch.bincount(index.reshape(-1), minlength=len(self)).tolist())
        return (out.view(tokens, k, -1) * weights.unsqueeze(-1)).sum(dim=1)

    def _uncaptured(self, hidden: torch.Tensor, why: str) -> None:
        """Inside a CUDA graph capture, the host read `why` names refused before it runs:
        `Building` while the grouped kernels compile, else a refusal no retry changes."""
        if not (hidden.is_cuda and torch.cuda.is_current_stream_capturing()):
            return
        detail = f"moe layer {self.layer}: {why}"
        if why == COMPILING:
            raise Building(detail)
        raise CapabilityError(detail, code="prepared_graph_ineligible")

    def _sorted(self, hidden: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        k = index.shape[1]
        flat = index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=len(self)).tolist()
        if self.trace is not None:
            self.trace(self.layer, counts)
        selected = hidden[order // k]
        outputs, start = [], 0
        for expert, count in zip(self, counts, strict=True):
            if count:
                outputs.append(expert(selected[start : start + count]))
            start += count
        out = torch.empty_like(selected, dtype=outputs[0].dtype)
        out[order] = torch.cat(outputs)
        return out

    def _grouped(self, hidden: torch.Tensor, index: torch.Tensor) -> torch.Tensor | str:
        """Each (token, expert) pair's expert output, `[T * k, H]`, by the grouped kernels; else
        why they do not serve (not built, not bf16 on a GPU, an expert not bound or of another
        kind, the GEMM's reads not aligned)."""
        launch = _LAUNCH[0]
        if isinstance(launch, str):
            return launch
        if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
            return f"the grouped kernels take bf16 on a GPU, not {hidden.dtype} on {hidden.device}"
        found = self._operands(hidden.device)
        if isinstance(found, str):
            return found
        if index.numel() > GROUPED and not self._aligned:
            return "an expert's operand or width is not 16-byte aligned, as the grouped GEMM reads"
        return self._run(launch, *found, hidden, index)

    def _run(
        self,
        launch: Launch,
        kind: str,
        table: torch.Tensor,
        hidden: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Two grouped launches with the gate between them as `RoutedExpert.forward` computes it:
        GEMVs for up to `GROUPED` pairs, else GEMMs over the pairs sorted by expert."""
        first = self[0]
        size, inner = first.down_proj.out_features, first.down_proj.in_features
        if index.numel() <= GROUPED:
            project = partial(self._gemv, launch, kind, table, index.reshape(-1).to(torch.int64))
        else:
            project = partial(self._gemm, launch, kind, table, *self._tiles(index))
        up = project(0, hidden.reshape(-1, size), 2 * inner, index.shape[1])
        a, b = up.chunk(2, dim=-1)
        gated = first.act(a) * b if first.gate_first else a * first.act(b)
        return project(1, gated, size, 1)

    def _gemv(
        self,
        launch: Launch,
        kind: str,
        table: torch.Tensor,
        experts: torch.Tensor,
        which: int,
        x: torch.Tensor,
        n: int,
        div: int,
    ) -> torch.Tensor:
        """`[pairs, n]`: projection `which` (0 gate-and-up, 1 down) of input row `p // div` by
        pair p's expert."""
        x = x.contiguous()
        pairs, width = int(experts.numel()), int(x.shape[-1])
        out = torch.empty((pairs, n), dtype=torch.bfloat16, device=x.device)
        grid = (pairs, -(-n // _kernels.BLOCK_N), 1)
        blocks = (_kernels.BLOCK_N, _kernels.BLOCK_K)
        if kind == "bf16":
            launch(_kernels.experts_bf16_kernel, grid, x, table[which], experts, out, n, width,
                   div, *blocks)  # fmt: skip
        else:
            rows, scales = int8_linear.quantize_rows(torch, x)
            launch(_kernels.experts_int8_kernel, grid, rows, scales, table[2 * which],
                   table[2 * which + 1], experts, out, n, width, div, *blocks)  # fmt: skip
        return out

    def _gemm(
        self,
        launch: Launch,
        kind: str,
        table: torch.Tensor,
        order: torch.Tensor,
        spans: torch.Tensor,
        tiles: int,
        which: int,
        x: torch.Tensor,
        n: int,
        div: int,
    ) -> torch.Tensor:
        """`[pairs, n]`: projection `which` of the pairs sorted by expert (`_tiles`), a program
        per (row tile, `GEMM_N` outputs). Gate-and-up reads token row `order[r] // div` and
        writes sorted row r; down reads sorted row r and writes pair `order[r]`'s row. int8 rows
        quantize as an int8.convrot leaf's own forward does: the op's kernel where it serves."""
        x = x.contiguous()
        if kind == "bf16" and x.data_ptr() % 16:  # read in 16-byte vectors
            x = x.clone()
        pairs, width = int(order.numel()), int(x.shape[-1])
        out = torch.empty((pairs, n), dtype=torch.bfloat16, device=x.device)
        grid = (tiles * -(-n // _kernels.GEMM_N), 1, 1)
        rest = (order, spans, out, n, width, div, tiles, _kernels.GEMM_M, _kernels.GEMM_N,
                _kernels.GEMM_K[kind], _kernels.GROUP, 1 - which)  # fmt: skip
        if kind == "bf16":
            launch(_kernels.grouped_bf16_kernel, grid, x, table[which], *rest)
        else:
            found = int8_linear.quantize(torch, x)
            if found is None:
                rows, scales = int8_linear.quantize_rows(torch, x)
            else:
                rows, scales = found.payload, found.scale
            launch(_kernels.grouped_int8_kernel, grid, rows, scales, table[2 * which],
                   table[2 * which + 1], *rest)  # fmt: skip
        return out

    def _tiles(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The GEMM's routing on the device, no count read: each sorted pair's flat (token, k)
        slot (one stable sort: ascending (token, k) within an expert, experts in index order);
        the `[3, tiles]` spans of its `GEMM_M`-row tiles (expert, or -1 past the routing's tiles;
        first sorted row; its expert's end); and `tiles`, the most any routing of these pairs
        needs: the full tiles and one ragged per expert hit."""
        experts, pairs, m = len(self), index.numel(), _kernels.GEMM_M
        keys, order = torch.sort(index.reshape(-1), stable=True)
        edges = torch.arange(experts + 1, device=keys.device, dtype=keys.dtype)
        starts = torch.searchsorted(keys, edges)  # each expert's first sorted pair; then all
        per = (starts.diff() + (m - 1)).div(m, rounding_mode="floor")  # each expert's tiles
        ends = per.cumsum(0)
        tiles = -(-pairs // m) + min(experts, pairs)
        tile = torch.arange(tiles, device=keys.device)
        owner = torch.searchsorted(ends, tile, right=True)  # `experts`: past the last tile
        expert = owner.clamp(max=experts - 1)
        first = starts[expert] + (tile - ends[expert] + per[expert]) * m
        spans = torch.stack((torch.where(owner < experts, expert, -1), first, starts[expert + 1]))
        return order, spans, tiles

    def _operands(self, device: torch.device) -> tuple[str, torch.Tensor] | str:
        """The kind ("bf16": plain Linears; "int8": int8.convrot leaves) and the address table
        (bf16: weights of gate-and-up, of down; int8: data and scale of each), when every
        expert's operands are bound, contiguous and of that kind on `device`; else why not.
        Outside a capture a changed address set is written to the table; inside one it is a
        refusal."""
        head = self[0].gate_and_up_proj
        kind = _kind(head)
        if kind is None:
            name = getattr(head, "provider", None) or type(head).__name__
            return f"its experts are {name}: the grouped kernels take bf16 and int8.convrot"
        rows: list[list[int]] = [[] for _ in range(2 if kind == "bf16" else 4)]
        for expert in self:
            for which, linear in enumerate((expert.gate_and_up_proj, expert.down_proj)):
                if _kind(linear) != kind:
                    return "its experts are not all of one kind"
                operands = (linear.weight,) if kind == "bf16" else (linear.data, linear.scale)
                for role, tensor in enumerate(operands):
                    address = _address(tensor, device)
                    if not address:
                        return f"an expert's operand is not bound, contiguous and on {device}"
                    rows[len(operands) * which + role].append(address)
        addresses = [a for row in rows for a in row]
        if addresses != self._addresses:
            if torch.cuda.is_current_stream_capturing():
                return "an expert's operands moved inside a capture"
            self._table = torch.tensor(rows, dtype=torch.int64, device=device)
            self._addresses = addresses
            widths = (self[0].down_proj.in_features, self[0].down_proj.out_features)
            self._aligned = all(a % 16 == 0 for a in (*addresses, *widths))
        assert self._table is not None
        return kind, self._table


def install(plan: Plan, model: nn.Module) -> str:
    """The grouped expert kernels for `model`'s `Experts`. Call from `Model.execution_plan`; it
    never waits for the compile (the kernels serve from the first request after it lands).
    Returns the plan's identity, "" when nothing applies (no experts, a CPU, no Triton). Either
    way the outcome is in the run's receipts (`plan_reports`)."""
    if not any(isinstance(module, Experts) for module in model.modules()):
        return ""
    if _kernels.TRITON_ABSENT:
        return _report(plan, _kernels.TRITON_ABSENT)
    if (device := torch.device(plan.device)).type != "cuda":
        return _report(plan, f"its kernels serve CUDA, not {device}")

    def activate(launch: Launch) -> None:
        _LAUNCH[0] = launch
        _report(plan)

    try:
        if plan.when_ready is None:
            activate(plan.load(KERNELS))
        elif not plan.when_ready(KERNELS, activate):
            say(KERNELS.name, f"[moe] {KERNELS.name} {COMPILING}")
            _report(plan, COMPILING)
    except KernelsUnavailable as exc:
        say(KERNELS.name, f"[moe] each expert runs its own Linears: {exc.code}: {exc}")
        return _report(plan, f"{exc.code}: {exc}")
    return f"{KERNELS.name}:{KERNELS.identity}"


def _report(plan: Plan, reason: str = "") -> str:
    """The plan's outcome under its kernel set's name, as `fusion` reports it: applied, or why
    each expert runs its own Linears, which a capture's refusal names until a launcher serves.
    Returns the plan's identity when refused: ""."""
    plan.report(KERNELS.name, {"applied": not reason, **({"reason": reason} if reason else {})})
    if reason and isinstance(_LAUNCH[0], str):
        _LAUNCH[0] = reason
    return ""


def _kind(linear: nn.Module) -> str | None:
    """ "bf16" for a plain bias-free bf16 Linear, "int8" for an int8.convrot leaf, else None."""
    if type(linear) is nn.Linear:
        return "bf16" if linear.bias is None and linear.weight.dtype == torch.bfloat16 else None
    if getattr(linear, "provider", None) == INT8_CONVROT and getattr(linear, "bias", 0) is None:
        return "int8"
    return None


def _address(tensor: torch.Tensor, device: torch.device) -> int:
    """`tensor`'s address when it is bound (a placeholder has none), contiguous and on `device`;
    else 0."""
    if tensor.device != device or not tensor.is_contiguous():
        return 0
    try:
        return int(tensor.data_ptr())
    except RuntimeError:
        return 0


def split(model: nn.Module) -> int:
    """Every Transformers fused-experts module of `model` (3-D `gate_up_proj` [E, 2I, H] and
    `down_proj` [E, H, I], `act(gate) * up`) replaced in place by `Experts` of `RoutedExpert`s on
    its device and dtype, numbered in module order; `RoutedExpert` joins the model's
    `_no_split_modules`. Returns how many layers were split."""
    fused = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "gate_up_proj", None), nn.Parameter)
        and isinstance(getattr(module, "down_proj", None), nn.Parameter)
        and module.gate_up_proj.dim() == 3
    ]
    for layer, (name, module) in enumerate(fused):
        count, double, hidden = module.gate_up_proj.shape
        if getattr(module, "is_transposed", False) or not getattr(module, "is_concatenated", True):
            raise ValueError(f"{name}: only [E, 2I, H] gate-then-up experts are split")
        like = {"device": module.gate_up_proj.device, "dtype": module.gate_up_proj.dtype}
        experts = Experts(
            (
                RoutedExpert(hidden, double // 2, gate_first=True, act=module.act_fn, **like)
                for _ in range(count)
            ),
            layer=layer,
        )
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent), attr, experts)
    if fused:
        model._no_split_modules = [
            *(getattr(model, "_no_split_modules", None) or ()),
            RoutedExpert.__name__,
        ]
    return len(fused)


class Routing:
    """Called as `trace(layer, counts)` by every MoE layer of every forward, in dispatch order."""

    def __init__(self, experts: int = 64, caches: tuple[int, ...] = CACHES) -> None:
        self.experts, self.caches = experts, caches
        self.forwards = 0
        self.hit_per_layer = array("I")
        self.tokens_per_hit = array("I")
        #: `layer * experts + expert` of every expert a forward ran, in the order it ran them.
        self.sequence = array("I")

    def __call__(self, layer: int, counts: list[int]) -> None:
        self.forwards += layer == 0
        hit = 0
        for expert, tokens in enumerate(counts):
            if tokens:
                hit += 1
                self.tokens_per_hit.append(tokens)
                self.sequence.append(layer * self.experts + expert)
        self.hit_per_layer.append(hit)

    def summary(self) -> dict[str, float]:
        """Per layer and forward means; `working_set` counts the distinct experts of the stage;
        per cache size the hit rate of LRU and of next-use eviction, and the latter's misses per
        forward (experts that must move)."""
        if not self.sequence:
            return {}
        found: dict[str, float] = {
            "forwards": self.forwards,
            "experts_hit_per_layer_mean": round(statistics.fmean(self.hit_per_layer), 2),
            "experts_hit_per_layer_min": min(self.hit_per_layer),
            "tokens_per_hit_expert_min": min(self.tokens_per_hit),
            "tokens_per_hit_expert_median": statistics.median(self.tokens_per_hit),
            "tokens_per_hit_expert_max": max(self.tokens_per_hit),
            "working_set": len(set(self.sequence)),
        }
        for size in self.caches:
            found[f"lru_{size}_hit_rate"] = round(
                lru_hits(self.sequence, size) / len(self.sequence), 4
            )
            hits = min_hits(self.sequence, size)
            found[f"min_{size}_hit_rate"] = round(hits / len(self.sequence), 4)
            found[f"min_{size}_misses_per_forward"] = round(
                (len(self.sequence) - hits) / max(self.forwards, 1), 1
            )
        return found


def lru_hits(sequence: array[int] | list[int], size: int) -> int:
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for key in sequence:
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            cache[key] = None
            if len(cache) > size:
                cache.popitem(last=False)
    return hits


def min_hits(sequence: array[int] | list[int], size: int) -> int:
    """Belady's MIN: on a miss with the cache full, evict the expert used furthest in the future."""
    never = len(sequence)
    upcoming = [never] * len(sequence)
    seen: dict[int, int] = {}
    for i in range(len(sequence) - 1, -1, -1):
        upcoming[i] = seen.get(sequence[i], never)
        seen[sequence[i]] = i
    cached: dict[int, int] = {}  # key -> its next use
    furthest: list[tuple[int, int]] = []  # (-next use, key); stale entries are skipped
    hits = 0
    for i, key in enumerate(sequence):
        if key in cached:
            hits += 1
        elif len(cached) >= size:
            while True:
                use, victim = heapq.heappop(furthest)
                if cached.get(victim) == -use:
                    del cached[victim]
                    break
        cached[key] = upcoming[i]
        heapq.heappush(furthest, (-upcoming[i], key))
    return hits
