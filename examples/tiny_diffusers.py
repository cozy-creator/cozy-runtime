"""A stand-in for the model libraries an endpoint brings as an ordinary dependency.

cozy-runtime defines NO model architectures (§1.1): T is whatever the endpoint imports —
upstream diffusers where it exists, the author's own module where it doesn't. This file
plays that part for the fixtures: two pipeline families with real component graphs, real
`state_dict()` topologies, and deterministic weightless compute, so every construction,
census, match, fill, guard and scope in `cozy_runtime.author` runs for real.

It also plays the OTHER part a real library plays: it REGISTERS the exact construction
adapter for its whole-pipeline class, which is what makes `loader.load(T)` legal for it.
The runtime ships that registration for diffusers outside the author fence; nothing here
is duck-typed, and `from_pretrained` is deliberately present-and-unused to prove the
reflective fallback is gone.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from cozy_runtime.author import Config, register_adapter

Task = Literal["fl2va", "ref2va"]


class TinyTensor:
    """A logical tensor destination. `shape`/`dtype` are all the topology walk reads, and
    the fill path writes through it — exactly the surface `torch.Tensor` presents."""

    __slots__ = ("dtype", "filled", "seed", "shape")

    def __init__(self, *shape: int, dtype: str = "bf16", seed: int = 0) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.filled = False
        self.seed = seed

    def fill_(self, nbytes: int) -> None:
        """cr-005 replaces this with the TensorFS byte path; the SEAM is the same call."""
        self.filled = True
        self.seed = nbytes

    @property
    def nbytes(self) -> int:
        size = 2 if self.dtype == "bf16" else 4
        for dim in self.shape:
            size *= dim
        return size


class TinyModule:
    """A component root. The topology walk can only see what `state_dict()` reaches."""

    def __init__(self, name: str, tensors: Mapping[str, TinyTensor]) -> None:
        self.name = name
        self._tensors = dict(tensors)

    def state_dict(self) -> Mapping[str, TinyTensor]:
        return self._tensors

    def __call__(self, *args: object, **kwargs: object) -> int:
        """Deterministic weightless 'compute': a stable digest over the argument text."""
        del kwargs
        return (sum(len(repr(a)) * 131 for a in args) + len(self.name) * 17) & 0xFFFF


class TinyScheduler:
    """A scheduler family instance: `view.make_scheduler(name)` clones one per request."""

    def __init__(self, family: str, *, shift: float = 3.0, sampler: str | None = None) -> None:
        self.family = family
        self.shift = shift
        self.sampler = sampler
        self.steps: tuple[float, ...] = ()

    @property
    def config(self) -> Mapping[str, object]:
        return {"family": self.family, "shift": self.shift, "sampler": self.sampler}

    @property
    def compatibles(self) -> tuple[str, ...]:
        return ("flow_match", "euler", "dpm")

    def clone(self, sampler: str | None = None) -> TinyScheduler:
        return TinyScheduler(self.family, shift=self.shift, sampler=sampler or self.sampler)

    def set_timesteps(self, steps: int) -> tuple[float, ...]:
        self.steps = tuple(round(1.0 - i / steps, 4) for i in range(steps))
        return self.steps


# --------------------------------------------------------------------- SDXL-shaped


def _linear(rows: int, cols: int, *, prefix: str) -> dict[str, TinyTensor]:
    return {f"{prefix}.weight": TinyTensor(rows, cols), f"{prefix}.bias": TinyTensor(rows)}


class TinySdxlPipeline:
    """The whole-pipeline path: one `__call__`, four heavyweight components.

    SDXL has TWO text encoders — the second is load-bearing, and an endpoint that declares
    three components poisons at first touch of the fourth.
    """

    def __init__(self, config: Config) -> None:
        width = config.as_int("hidden_size", 32)
        self.components: dict[str, object] = {
            "text_encoder": TinyModule("text_encoder", _linear(width, width, prefix="proj")),
            "text_encoder_2": TinyModule(
                "text_encoder_2", _linear(width * 2, width, prefix="proj")
            ),
            "unet": TinyModule(
                "unet",
                {
                    **_linear(width, width, prefix="down.0"),
                    **_linear(width, width, prefix="mid.attn.q"),
                    **_linear(width, width, prefix="up.0"),
                },
            ),
            "vae": TinyModule("vae", _linear(3, width, prefix="decoder.out")),
            "scheduler": TinyScheduler("euler_discrete", shift=1.0),
        }

    @classmethod
    def from_pretrained(cls, path: str) -> TinySdxlPipeline:
        """Present, and never reached: `load(T)` uses the REGISTERED adapter below. A
        reflective fallback would hand third-party code a filesystem path (§1.1)."""
        raise RuntimeError(f"from_pretrained({path!r}) performed I/O")

    def _use(self, name: str) -> Callable[..., int]:
        """Every component is reached BY NAME through the table — which is exactly where
        the runtime installs its guarded roots."""
        return cast("Callable[..., int]", self.components[name])

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        generator: random.Random,
    ) -> ImageFrameData:
        conditioning = self._use("text_encoder")(prompt)
        # CFG-free execution skips the negative branch entirely — a VALUE-plane gate
        if guidance > 1.0:
            conditioning ^= self._use("text_encoder_2")(negative_prompt or "")
        latents = conditioning
        for step in range(steps):
            latents = self._use("unet")(latents, step, guidance)
        tint = self._use("vae")(latents)
        return ImageFrameData(width, height, tint & 0xFF, generator.randbytes(3))


@dataclass(frozen=True, slots=True)
class ImageFrameData:
    width: int
    height: int
    tint: int
    jitter: bytes


def build_sdxl_pipeline(config: Config) -> TinySdxlPipeline:
    """The registered EXACT construction adapter — typed config in, whole object out."""
    return TinySdxlPipeline(config)


register_adapter(TinySdxlPipeline, build_sdxl_pipeline)


# --------------------------------------------------------------------- H3-shaped


@dataclass(frozen=True, slots=True)
class H3Topology:
    """A CONSTRUCTION fact, fixed before load: exactly one modulation route exists."""

    baked: bool


class H3Blocks:
    """The modular pipeline's blocks. They resolve every component BY NAME through the
    pipeline's component table, which is precisely where the guarded roots live."""

    def __init__(self, components: Mapping[str, object]) -> None:
        self._components = components

    def _use(self, name: str) -> Callable[..., int]:
        return cast("Callable[..., int]", self._components[name])

    def build_presentation(
        self, prompt: str, references: Sequence[object], keyframes: Sequence[object]
    ) -> str:
        labels = "".join(f"<{type(r).__name__} {i}>" for i, r in enumerate(references))
        anchors = "".join(f"<Picture {i}>" for i, _ in enumerate(keyframes))
        return f"{anchors}{labels}{prompt}"

    def text_encoder(self, presentation: str) -> tuple[int, int]:
        hidden = self._use("text_encoder")(presentation)
        return hidden, len(presentation)

    def encode_visual(self, source: object, *, fps: float | None = None, clock: int = 0) -> int:
        return self._use("visual_vae")(source, fps, clock)

    def encode_audio(self, source: object, *, sample_rate: int | None = None) -> int:
        return self._use("audio_vae")(source, sample_rate)

    def prepare_latents(self, *, frames: int, generator: random.Random) -> int:
        return (frames * 1009 + generator.randrange(1 << 16)) & 0xFFFFFF

    def build_modulation_tables(self, plan: object) -> tuple[int, ...]:
        """Computed under the ACTIVE transformer lease on a full topology (§1.1.1)."""
        block = self._use(
            "fl2va_transformer" if self._has("fl2va_transformer") else "ref2va_transformer"
        )
        return tuple(block("adaln", plan, i) for i in range(4))

    def baked_tables(self, plan: object) -> tuple[int, ...]:
        """Ordinary checkpoint components: coverage was proven at admission (§1.1.1)."""
        block = self._use(
            "fl2va_transformer" if self._has("fl2va_transformer") else "ref2va_transformer"
        )
        return tuple(block("baked", plan, i) for i in range(4))

    def _has(self, name: str) -> bool:
        return name in self._components

    def step(
        self,
        transformer: str,
        latents: int,
        layout: object,
        t_v: float,
        t_a: float,
        modulation: int,
    ) -> int:
        return self._use(transformer)(latents, layout, t_v, t_a, modulation)

    def decode_video(self, latents: int) -> int:
        return self._use("visual_vae")("decode", latents)

    def decode_audio(self, latents: int) -> tuple[int, int]:
        return self._use("audio_vae")("decode", latents), 48_000


class MiniMaxH3ModularPipeline:
    """The component-staged path: real semantic operations over named components."""

    def __init__(self, config: Config, *, task: Task | None, both: bool = False) -> None:
        width = config.as_int("hidden_size", 32)
        baked = config.as_bool("baked", False)
        components: dict[str, object] = {
            "text_encoder": TinyModule(
                "text_encoder", _linear(width * 4, width, prefix="qwen.layers.0.attn.q")
            ),
            "visual_vae": TinyModule("visual_vae", _linear(width, width, prefix="enc.conv_in")),
            "audio_vae": TinyModule("audio_vae", _linear(width // 2, width, prefix="enc.conv_in")),
            "video_scheduler": TinyScheduler("flow_match_video"),
            "audio_scheduler": TinyScheduler("flow_match_audio", shift=5.0),
        }
        roles = ("fl2va", "ref2va") if both else ((task,) if task is not None else ())
        for role in roles:
            components[f"{role}_transformer"] = _transformer(f"{role}_transformer", width, baked)
        self.components = components
        self.blocks = H3Blocks(components)
        self.topology = H3Topology(baked=baked)


def _transformer(name: str, width: int, baked: bool) -> TinyModule:
    tensors = {**_linear(width, width, prefix="blocks.0.attn.q")}
    if baked:
        # A baked artifact CARRIES exact modulation tables and OMITS the projections.
        tensors["blocks.0.modulation_table"] = TinyTensor(4, width)
    else:
        tensors.update(_linear(width * 6, width, prefix="blocks.0.adaln.proj"))
        tensors.update(_linear(width * 6, width, prefix="time_embed.proj"))
    return TinyModule(name, tensors)


def build_h3_pipeline(config: Config, *, task: Task) -> MiniMaxH3ModularPipeline:
    """The ROLE-SCOPED family factory: the constructed graph contains exactly the
    components the calling class declares (§1.1)."""
    return MiniMaxH3ModularPipeline(config, task=task)


def build_h3_both(config: Config) -> MiniMaxH3ModularPipeline:
    """The planted violation for the dual-artifact red arm: one class constructing BOTH
    transformers could never pass exact binding against a narrow artifact."""
    return MiniMaxH3ModularPipeline(config, task=None, both=True)
