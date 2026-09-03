"""Weight mutation during load, and the parametrization machinery that hides it.

The one-fill-transform rule (§1.1): a `load()`-time weight transform is a conformance
failure. It desynchronizes derive from serve and it is STORAGE territory — a registered
encoding with a runtime provider, never author repacking. Marlin-class pre-transposed
layouts are the canonical example of what belongs on the other side of that line.

The distinction this file draws, and which the harness must draw mechanically: ordinary
initialization INSIDE a leaf's own `__init__` (`kaiming_uniform_`, `zero_`) writes values
that every fill overwrites, so it is invisible to the tensor schema and harmless. A mutation
AFTER the factory has returned its object is a transform the fill cannot see.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize

from corpus.harness import Case, Config, Pipeline


def _ordinary_init(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Initialized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width)
            # every leaf torch ships does exactly this; it must NOT be a refusal
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.to(torch.bfloat16)

    return Pipeline({"transformer": Initialized()})


def _repack(config: Config) -> Pipeline:
    """A Marlin-class weight repack performed at load. Storage territory, refused."""
    width = config.as_int("width", 16)

    class Repacked(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)

        def repack(self) -> None:
            with torch.no_grad():
                self.proj.weight.copy_(self.proj.weight.t().contiguous())

    built = Repacked()
    built.repack()  # AFTER construction — the transform the fill cannot see
    return Pipeline({"transformer": built})


def _weight_norm(config: Config) -> Pipeline:
    """`weight_norm` DELETES the `weight` Parameter and installs a plain attribute computed
    from `weight_g`/`weight_v`. Every DAC/BigVGAN audio VAE does this, so the census must
    report the two real destinations and NOT the synthesized attribute."""
    width = config.as_int("width", 16)

    class Normed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.utils.parametrizations.weight_norm(
                nn.Conv1d(width, width, 3, dtype=torch.bfloat16)
            )

    return Pipeline({"vae": Normed()})


def _parametrized(config: Config) -> Pipeline:
    """A `torch.nn.utils.parametrize` registration re-homes `weight` under
    `parametrizations.weight.original`, changing every checkpoint key the module demands."""
    width = config.as_int("width", 16)

    class Symmetric(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.triu() + x.triu(1).transpose(-1, -2)

    class Parametrized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
            parametrize.register_parametrization(self.proj, "weight", Symmetric())

    return Pipeline({"transformer": Parametrized()})


def _hooks(config: Config) -> Pipeline:
    """A load-state-dict pre-hook that rewrites keys — v1's `keymap` territory. It must not
    change the derived tensor schema, because derivation never calls `load_state_dict`."""
    width = config.as_int("width", 16)

    class Hooked(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            self._register_load_state_dict_pre_hook(self._rename)

        @staticmethod
        def _rename(state_dict: dict[str, object], prefix: str, *args: object) -> None:
            for key in list(state_dict):
                if key.endswith("gamma"):
                    state_dict[key.replace("gamma", "weight")] = state_dict.pop(key)

    return Pipeline({"transformer": Hooked()})


CASES = (
    Case(
        "ordinary_initialization",
        "nn.init.* inside a leaf's own __init__",
        _ordinary_init,
        "derived-correct",
        note="values every fill overwrites — must NOT be refused",
    ),
    Case(
        "post_construction_repack",
        "a Marlin-class weight repack after the factory returns",
        _repack,
        "typed-refusal",
        names="in-place weight transform",
    ),
    Case(
        "weight_norm",
        "weight_norm replacing `weight` with a computed attribute",
        _weight_norm,
        "derived-correct",
        note="the census must report the real parametrization destinations",
    ),
    Case(
        "parametrization",
        "register_parametrization re-homing weight under parametrizations.weight.original",
        _parametrized,
        "derived-correct",
        note="the key MOVES, and the derived tensor schema must say so",
    ),
    Case(
        "load_state_dict_hook",
        "a key-renaming load_state_dict pre-hook",
        _hooks,
        "derived-correct",
        note="derivation never calls load_state_dict, so the hook must not move a key",
    ),
)
