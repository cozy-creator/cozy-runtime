"""Mixture-of-experts layers on Runtime's weight paging: every routed expert is its own block.

    from cozy_runtime import moe

    experts = [moe.RoutedExpert(4096, 3072, gate_first=False) for _ in range(64)]
    self.experts = moe.Experts(experts, layer=3)
    out = self.experts(hidden, top_k_index, top_k_weights)

An expert is two Linears, `gate_and_up_proj` [2I, H] and `down_proj` [H, I], so every lane's leaf
serves it (int8.convrot, w4a8, nvfp4) and varena pages it on its own (`RoutedExpert` in the model's
`_no_split_modules`). `split` turns Transformers' fused experts (`gate_up_proj` [E, 2I, H],
`down_proj` [E, H, I]) into these; the checkpoint stores them per expert (TensorFS splits fused
uploads at ingest).

`Routing` records which experts each forward used and replays expert caches over that sequence:
least recently used, and next use (Belady's MIN, the fewest bytes any cache of that size moves).
"""

from __future__ import annotations

import heapq
import statistics
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable

import torch
from torch import nn

#: Routed-expert caches a `Routing` replays, in experts.
CACHES = (256, 512, 1024)


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
    expert multiplies its rows, gathered by one stable sort of the flat top-k index (ascending
    (token, k)), experts in index order; the outputs return to their (token, k) slots and are
    summed by `weights` in the outputs' dtype. The host waits once a layer, for the counts."""

    #: Set to `trace(layer, counts)` to observe routing: tokens per expert, every forward.
    trace: Callable[[int, list[int]], None] | None = None

    def __init__(self, experts: Iterable[RoutedExpert], *, layer: int) -> None:
        super().__init__(experts)
        self.layer = layer

    def forward(
        self, hidden: torch.Tensor, index: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        tokens, k = index.shape
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
        return (out.view(tokens, k, -1) * weights.unsqueeze(-1)).sum(dim=1)


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
