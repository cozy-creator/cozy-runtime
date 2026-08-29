"""Constructs that MUST derive correctly. If the harness cannot do these it can do nothing.

Every case here is an ordinary shape real packages use, and each one is a place a naive
census gets a subtly wrong answer: a tie counted twice as two destinations, a shared
submodule counted once, a non-persistent buffer demanded from a checkpoint that has no
such key, an fp32 island flattened to bf16.
"""

from __future__ import annotations

import torch
from torch import nn

from corpus.harness import Case, Config, Pipeline, linear_block


def _plain(config: Config) -> Pipeline:
    width = config.as_int("width", 16)
    return Pipeline({"unet": linear_block(width), "vae": linear_block(width // 2)})


def _tied(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, width)
            self.head = nn.Linear(width, 32, bias=False)
            self.head.weight = self.embed.weight  # the tie, established by assignment
            self.to(torch.bfloat16)  # and it must SURVIVE the dtype cast

    return Pipeline({"text_encoder": Tied()})


def _shared_submodule(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Shared(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(width, width, bias=False)
            self.down = shared
            self.up = shared  # ONE object reached by TWO names
            self.to(torch.bfloat16)

    return Pipeline({"transformer": Shared()})


def _buffers(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Buffers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            # persistent: a checkpoint carries it, so it IS a destination
            self.register_buffer("pos", torch.zeros(8, width, dtype=torch.bfloat16))
            # non-persistent: the rope table. No checkpoint has this key and demanding one
            # would refuse every real artifact — pgw#1644 died here.
            self.register_buffer("inv_freq", torch.zeros(width // 2), persistent=False)
            # a plain attribute reaches no state_dict at all and is not a tensor fact
            self.scale = torch.ones(1)

    return Pipeline({"transformer": Buffers()})


def _dtype_islands(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Islands(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Linear(width, width).to(torch.bfloat16)
            # an fp32 island: NaN-prone VAEs pin this, and the dtype-diet rule must carry
            # the deviation while staying silent about the bf16 majority
            self.head = nn.Linear(width, width).to(torch.float32)
            self.register_buffer("index", torch.zeros(4, dtype=torch.int64))
            self.register_buffer("mask", torch.zeros(4, dtype=torch.bool))

    return Pipeline({"vae": Islands()})


def _containers(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Containers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
            self.named = nn.ModuleDict({"a": nn.Linear(width, width)})
            self.params = nn.ParameterList([nn.Parameter(torch.zeros(width))])
            self.to(torch.bfloat16)

    return Pipeline({"transformer": Containers()})


def _reflection(config: Config) -> Pipeline:
    """Construction driven by config DATA, not by literal Python — the shape a
    `model_index.json`-driven pipeline builds itself with."""
    width = config.as_int("width", 16)
    spec = config.section("blocks")

    class Reflected(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in spec.keys():  # noqa: SIM118 — Config is not a Mapping
                kind = spec.as_str(name)
                built = {"linear": nn.Linear, "embedding": nn.Embedding}[kind](width, width)
                self.add_module(name, built)
            self.to(torch.bfloat16)

    return Pipeline({"transformer": Reflected()})


def _deep(config: Config) -> Pipeline:
    width = config.as_int("width", 8)

    class Leaf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(width, width, bias=False)
            self.k = nn.Linear(width, width, bias=False)

    class Level(nn.Module):
        def __init__(self, depth: int) -> None:
            super().__init__()
            self.attn = Leaf()
            if depth:
                self.inner = Level(depth - 1)

    root = Level(5).to(torch.bfloat16)
    return Pipeline({"transformer": root})


CASES = (
    Case("plain", "ordinary Linear/LayerNorm leaves", _plain, "derived-correct"),
    Case(
        "tie_survives_cast",
        "a weight tie established by assignment, then .to(dtype)",
        _tied,
        "derived-correct",
        note="both names must appear as destinations AND share one alias group",
    ),
    Case(
        "shared_submodule",
        "one nn.Module object reached by two attribute names",
        _shared_submodule,
        "derived-correct",
    ),
    Case(
        "non_persistent_buffer",
        "a rope table registered persistent=False",
        _buffers,
        "derived-correct",
        note="inv_freq is derived, never a destination; pos is a destination",
    ),
    Case(
        "dtype_islands",
        "an fp32 island and int64/bool buffers beside bf16",
        _dtype_islands,
        "derived-correct",
    ),
    Case("containers", "ModuleList / ModuleDict / ParameterList", _containers, "derived-correct"),
    Case(
        "reflection_driven",
        "submodules built from config data by add_module",
        _reflection,
        "derived-correct",
        config={"width": 16, "blocks": {"alpha": "linear", "beta": "embedding"}},
    ),
    Case(
        "deep_recursion", "six levels of recursively constructed blocks", _deep, "derived-correct"
    ),
)
