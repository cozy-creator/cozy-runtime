"""Construction-time lifecycle hostility: compile, laziness, module swaps, capability
gates, and the two capabilities a derivation must never have.

The last three cases are the fences themselves. A fence that has never been attempted is a
comment, so each one is attempted here by real package code and observed refusing.
"""

from __future__ import annotations

import torch
from torch import nn

from corpus.harness import Case, Config, Pipeline, linear_block


def _compiled(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Compiled(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.compile(linear_block(width))

    return Pipeline({"vae": Compiled()})


def _compiled_method(config: Config) -> Pipeline:
    width = config.as_int("width", 16)
    block = linear_block(width)
    block.compile()  # the in-place spelling, which returns None
    return Pipeline({"vae": block})


def _lazy_module(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Lazy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.LazyLinear(width)

    return Pipeline({"transformer": Lazy()})


def _lazy_components(config: Config) -> Pipeline:
    """`ModularPipeline.from_pretrained` returns a pipeline whose components are all None
    and defers the build to whoever holds the object. A component the pipeline SAYS it can
    build and then has not built must be named and refused — the lazy build swallows
    per-component failures into a log line (v1's tcg#65 postcondition)."""
    width = config.as_int("width", 16)
    built: dict[str, object] = {"transformer": linear_block(width), "vae": None}
    return Pipeline(built)


def _quantized_swap(config: Config) -> Pipeline:
    """An ENCODING-AWARE module choice: the provider variant replaces the leaf class, so
    the constructed graph differs per variant and the contract must record which one."""
    width = config.as_int("width", 16)
    variant = config.as_str("variant", "plain")

    class PackedLinear(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.register_parameter(
                "packed",
                nn.Parameter(torch.zeros(size, size // 2, dtype=torch.uint8), requires_grad=False),
            )
            self.register_parameter(
                "scales", nn.Parameter(torch.zeros(size, dtype=torch.bfloat16), requires_grad=False)
            )

    class Swapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = (
                PackedLinear(width)
                if variant == "w4a16"
                else nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
            )

    return Pipeline({"transformer": Swapped()})


def _capability_gate(config: Config) -> Pipeline:
    """sm89 gets the packed kernel path, sm90 the wide one — v1's real sm89/sm90 gate."""
    width = config.as_int("width", 16)
    major, _ = torch.cuda.get_device_capability()

    class Gated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if major >= 9:
                self.attn = nn.Linear(width, width * 2, bias=False, dtype=torch.bfloat16)
            else:
                self.attn = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)

    return Pipeline({"transformer": Gated()})


def _kernel_fetch(config: Config) -> Pipeline:
    """v1 downloaded digest-pinned kernels at boot. v2 forbids it: kernel binaries are
    pinned artifacts and image inputs, so a construction-time fetch has no legal shape."""
    width = config.as_int("width", 16)
    import socket

    class Fetching(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            socket.create_connection(("127.0.0.1", 9), timeout=1)

    return Pipeline({"transformer": Fetching()})


def _weight_read(config: Config) -> Pipeline:
    """A factory reaching for checkpoint bytes. There is no path in the config capability,
    so the only way to try is to name one — and the fence answers by extension."""
    width = config.as_int("width", 16)

    class Reading(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            with open("/tmp/cozy-corpus-model.safetensors", "rb") as handle:
                handle.read(8)

    return Pipeline({"transformer": Reading()})


def _write_output(config: Config) -> Pipeline:
    width = config.as_int("width", 16)

    class Writing(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width, dtype=torch.bfloat16)
            with open("/tmp/cozy-corpus-escape.txt", "w") as handle:
                handle.write("derivation wrote to the filesystem")

    return Pipeline({"transformer": Writing()})


CASES = (
    Case(
        "torch_compile",
        "torch.compile(module) inside the factory",
        _compiled,
        "typed-refusal",
        names="torch.compile",
        note="it prefixes every destination with `_orig_mod.`; v1 neutered it to eager, "
        "v2 has no consumer that needs the call to survive",
    ),
    Case(
        "torch_compile_method",
        "the in-place Module.compile() spelling",
        _compiled_method,
        "typed-refusal",
        names="torch.compile",
    ),
    Case(
        "lazy_module",
        "nn.LazyLinear — no shape until the first forward",
        _lazy_module,
        "typed-refusal",
        names="LazyModuleMixin",
    ),
    Case(
        "lazy_component_none",
        "a pipeline that declares a component and leaves it None",
        _lazy_components,
        "typed-refusal",
        names="unbuilt component",
    ),
    Case(
        "encoding_aware_swap",
        "a w4a16 provider replacing the leaf class",
        _quantized_swap,
        "derived-correct",
        config={"width": 16, "variant": "w4a16"},
        note="the packed/scales pair IS the derived tensor schema for that variant",
    ),
    Case(
        "hardware_gate",
        "construction branching on device capability — tensor requirements are a function of "
        "the config document alone (model-code-fit D3)",
        _capability_gate,
        "typed-refusal",
        names="hardware-conditional",
        ground_truth=False,
    ),
    Case(
        "boot_kernel_fetch",
        "a digest-pinned kernel download during construction",
        _kernel_fetch,
        "typed-refusal",
        names="no_network",
        ground_truth=False,
    ),
    Case(
        "weight_byte_read",
        "opening a .safetensors file from the factory",
        _weight_read,
        "typed-refusal",
        names="no_weight_bytes",
        ground_truth=False,
    ),
    Case(
        "filesystem_write",
        "writing a file from the factory",
        _write_output,
        "typed-refusal",
        names="no_filesystem_writes",
        ground_truth=False,
    ),
)
