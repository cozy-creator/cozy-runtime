"""Data-dependent shapes and value-dependent control flow inside `__init__`.

`__init__` must be meta-device safe: allocate shapes and dtypes, never VALUES (§1.1). Each
case here reaches for a value that does not exist until the fill, and each must be refused
by NAME rather than producing a plausible-looking tensor schema built from whatever a
zero-filled tensor happens to say. v1 had NO handling for this class at all — its
`ShapeEnv` is created and never queried — so this is new ground, not a port.
"""

from __future__ import annotations

import torch
from torch import nn

from corpus.harness import Case, Config, Pipeline


def _shape_from_value(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Dependent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            probe = torch.zeros(width, dtype=torch.bfloat16)
            # the layer's WIDTH depends on a tensor's contents — under any weightless
            # derivation this reads zero and builds a graph the checkpoint never had
            size = int(probe.sum().item()) + width
            self.proj = nn.Linear(size, size, dtype=torch.bfloat16)

    return Pipeline({"transformer": Dependent()})


def _branch_on_value(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Branching(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            gate = torch.zeros(4, dtype=torch.bfloat16)
            if bool((gate > 0).any()):
                self.proj = nn.Linear(width, width * 2, dtype=torch.bfloat16)
            else:
                self.proj = nn.Linear(width, width, dtype=torch.bfloat16)

    return Pipeline({"transformer": Branching()})


def _nonzero_shape(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Sparse(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mask = torch.zeros(width, dtype=torch.bool)
            keep = torch.nonzero(mask).numel() + width
            self.proj = nn.Linear(keep, keep, dtype=torch.bfloat16)

    return Pipeline({"transformer": Sparse()})


def _numpy_protocol(config: Config) -> Pipeline:
    """The numpy PROTOCOL path, not `.numpy()`. v1 shipped a host-egress shim listing only
    `Tensor.numpy` and missed `Tensor.__array__` entirely — which is the path diffusers'
    `EulerDiscreteScheduler.set_timesteps` actually takes through `np.array(sigmas)`."""
    width = config.as_int("width", 16)
    import numpy as np

    class ViaProtocol(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            sigmas = self.proj.weight.detach()
            self.register_buffer(
                "schedule", torch.tensor(np.array(sigmas).sum(), dtype=torch.bfloat16)
            )

    return Pipeline({"transformer": ViaProtocol()})


def _tolist(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class ToList(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            self.sizes = self.proj.bias.detach().tolist()

    return Pipeline({"transformer": ToList()})


def _config_shapes(config: Config) -> Pipeline:
    """The LEGAL twin of every case above: shapes derived from CONFIG, which is exactly
    what a typed read-only config capability is for. This must derive correctly."""
    width = config.as_int("width", 16)
    layers = config.as_int("layers", 3)

    class FromConfig(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [nn.Linear(width * (i + 1), width, dtype=torch.bfloat16) for i in range(layers)]
            )
            # a derived table that is a pure function of config: register_buffer, no
            # device pin — the shape v1's MetaMaterializationError prescribes
            self.register_buffer(
                "inv_freq", torch.zeros(width // 2, dtype=torch.float32), persistent=False
            )

    return Pipeline({"transformer": FromConfig()})


CASES = (
    Case(
        "data_dependent_shape",
        "a layer width read from a tensor's contents",
        _shape_from_value,
        "typed-refusal",
        names="host egress",
    ),
    Case(
        "value_branch",
        "an `if bool(tensor.any())` structural branch in __init__",
        _branch_on_value,
        "typed-refusal",
        names="host egress",
    ),
    Case(
        "nonzero_shape",
        "a shape derived from torch.nonzero",
        _nonzero_shape,
        "typed-refusal",
        names="data-dependent shape",
    ),
    Case(
        "numpy_protocol",
        "host egress through np.array(), i.e. Tensor.__array__",
        _numpy_protocol,
        "typed-refusal",
        names="host egress",
    ),
    Case(
        "tolist_egress",
        "reading a parameter through .tolist()",
        _tolist,
        "typed-refusal",
        names="host egress",
    ),
    Case(
        "config_driven_shapes",
        "shapes and a derived table from CONFIG only — the legal twin of every case above",
        _config_shapes,
        "derived-correct",
        config={"width": 16, "layers": 3},
    ),
)
