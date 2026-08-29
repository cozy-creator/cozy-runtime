"""The issue's explicitly named planted red arms, as real package code.

`devices.py` already carries the CPU-stray pair and `scripts/derive-verify.py` carries the
missing-destination and orphan-part/scale arms (they are MATCH-direction facts, which need
a candidate table). What lives here is the rest of the named list: a hidden cache, and a
broken tie — both of which are census facts a match can never catch, because the checkpoint
never disagrees with them.

Dropout is deliberately absent and this is the reason: `nn.Dropout` holds no tensor at all,
so it cannot be planted AS one. What it could do is change the census by existing, and that
is covered from the other side — a module the walk does not reach is a census disagreement
against the fill plane (`check_agreement`), which is set equality in both directions.
"""

from __future__ import annotations

import torch
from torch import nn

from corpus.harness import Case, Config, Pipeline


def _hidden_cache(config: Config) -> Pipeline:
    """A GB-scale non-persistent buffer: no checkpoint fills it and no ledger meters it.

    The authoring rule is that a GB-scale tensor is a NAMED saved component, not a derived
    table (§1.1) — scale decides, not provenance. A rope table is fine at kilobytes and is
    a hidden 4 GiB allocation at this size.
    """
    width = config.as_int("width", 16)

    class Hoarder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            self.register_buffer(
                "precomputed",
                torch.zeros(1 << 14, 1 << 14, dtype=torch.bfloat16),
                persistent=False,
            )

    return Pipeline({"transformer": Hoarder()})


def _sized_cache(config: Config) -> Pipeline:
    """The same idiom at a legitimate size — the arm that keeps the cap from being a ban."""
    width = config.as_int("width", 16)

    class Ordinary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            self.register_buffer(
                "inv_freq", torch.zeros(width // 2, dtype=torch.float32), persistent=False
            )

    return Pipeline({"transformer": Ordinary()})


def _tied(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, width, dtype=torch.bfloat16)
            self.head = nn.Linear(width, 32, bias=False, dtype=torch.bfloat16)
            self.head.weight = self.embed.weight

    return Pipeline({"text_encoder": Tied()})


def _broken_tie(config: Config) -> Pipeline:
    """The tie ESTABLISHED and then silently broken — the shape v1's `init_empty_weights`
    produced by accident, which is why its skeleton had to retie twice.

    The two names now hold two independent tensors of the same shape. No checkpoint can
    ever notice: the keys and shapes are identical either way. Only an alias census that
    reads OBJECT/STORAGE identity sees the difference, which is exactly why the contract
    records alias groups and why `_tied_weights_keys` (a list of names a class MIGHT tie)
    is never the answer.
    """
    width = config.as_int("width", 16)

    class Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, width, dtype=torch.bfloat16)
            self.head = nn.Linear(width, 32, bias=False, dtype=torch.bfloat16)
            self.head.weight = self.embed.weight
            # …and then a later line rebinds it. Same key, same shape, different tensor.
            self.head.weight = nn.Parameter(
                torch.zeros(32, width, dtype=torch.bfloat16), requires_grad=False
            )

    return Pipeline({"text_encoder": Broken()})


CASES = (
    Case(
        "hidden_cache",
        "a GB-scale non-persistent buffer no checkpoint fills",
        _hidden_cache,
        "typed-refusal",
        names="hidden cache",
        ground_truth=False,
        note="the planted hidden-cache red arm; ground truth would allocate 512 MiB for real",
    ),
    Case(
        "sized_derived_table",
        "the same idiom at a legitimate size",
        _sized_cache,
        "derived-correct",
        note="the cap must bound the idiom, not ban it",
    ),
    Case(
        "live_tie",
        "a weight tie that is still live at census time",
        _tied,
        "derived-correct",
        note="the baseline the broken-tie arm is compared against",
    ),
    Case(
        "broken_tie",
        "a tie established in __init__ and silently rebound after",
        _broken_tie,
        "derived-correct",
        note="derives correctly AND differently from live_tie — no checkpoint could tell "
        "them apart, which is why the alias census exists",
    ),
)
