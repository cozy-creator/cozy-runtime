"""Placement hostility: strays, moves, host transfers, pins.

There are no developer-facing placement knobs (§1). Every one of these constructs decides
placement inside `__init__`, which is exactly the fact the runtime owns — so each must be
caught, and caught by NAME, not by a downstream shape mismatch three frames later.
"""

from __future__ import annotations

import torch
from torch import nn

from corpus.harness import Case, Config, Pipeline, linear_block


def _cpu_stray(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Stray(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            # the planted CPU-stray buffer: an explicit device= pin that survives every
            # later `.to()` the runtime performs, so it serves from host memory forever
            self.register_buffer("bias_table", torch.zeros(width, device="cpu"))

    return Pipeline({"transformer": Stray()})


def _cpu_stray_nonpersistent(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Stray(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            self.register_buffer("rope", torch.zeros(width, device="cpu"), persistent=False)

    return Pipeline({"transformer": Stray()})


def _cuda_move(config: Config) -> Pipeline:
    width = config.as_int("width", 16)
    return Pipeline({"transformer": linear_block(width).to("cuda")})


def _cuda_allocate(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class OnCuda(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, device="cuda")

    return Pipeline({"transformer": OnCuda()})


def _host_transfer(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Egress(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            # reading a parameter's contents during construction: there are no contents
            self.register_buffer("norm", self.proj.weight.detach().cpu().clone())

    return Pipeline({"transformer": Egress()})


def _pinned(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Pinned(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            self.register_buffer("staging", torch.zeros(width).pin_memory())

    return Pipeline({"transformer": Pinned()})


CASES = (
    Case(
        "cpu_stray_buffer",
        "a persistent buffer pinned to device='cpu'",
        _cpu_stray,
        "typed-refusal",
        names="device pin",
        note="the planted CPU-stray buffer red arm",
    ),
    Case(
        "cpu_stray_derived",
        "a NON-persistent buffer pinned to device='cpu'",
        _cpu_stray_nonpersistent,
        "typed-refusal",
        names="device pin",
        note="no checkpoint fills it, so only the census can catch it",
    ),
    Case(
        "cuda_module_move",
        "module.to('cuda') inside the factory",
        _cuda_move,
        "typed-refusal",
        names="device",
        ground_truth=False,
        note="the fence is CUDA_VISIBLE_DEVICES='', so torch itself has no device",
    ),
    Case(
        "cuda_allocation",
        "a leaf constructed with device='cuda'",
        _cuda_allocate,
        "typed-refusal",
        names="device pin",
        ground_truth=False,
    ),
    Case(
        "host_transfer",
        "reading a parameter through .cpu() during construction",
        _host_transfer,
        "typed-refusal",
        names="device move",
    ),
    Case(
        "pinned_host_buffer",
        "a pin_memory() staging buffer built at construction",
        _pinned,
        "typed-refusal",
        names="device move",
    ),
)
