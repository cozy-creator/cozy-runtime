"""Qwen-Image-Edit-2511's DiT (its int8 lane) with the Lightning LoRA's `lora.Hook` on every site, for the release
gate's LoRA overlay exercisers (`cozy_runtime.bench.overlay_replay`). Torch-free until a model loads: the Runtime's
App, which registers the exercisers, imports this module.

`replay`: three calls in one run, unarmed twice, then armed before the loop as Qwen-Image's package arms; the armed
latents equal the same call with graphs bypassed, byte for byte, and differ from the unarmed ones. `inside` arms in the
step instead: the latents equal those armed before the loop. Was the gate package `execution/overlay-replay` (xs-174:
the graphs bypass and the LoRA bake kernels are the Runtime's)."""

from __future__ import annotations

import time
from functools import partial
from typing import TYPE_CHECKING, cast

import msgspec

from cozy_runtime.author import Config, Context, Loader, Model, Telemetry, uses_components

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from torch import nn

    from cozy_runtime.internal.bench.qwen_overlay_factors import Factors

SITES = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj",
    "attn.add_v_proj", "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
)  # fmt: skip
WIDTH, HIDDEN, BLOCKS, RANK = 3072, 12288, 60, 64
LORA_COMPONENT = "model"
#: latent side in tokens, text rows, and DiT forwards a call
SIDE, TEXT, STEPS = 32, 64, 3
#: The forward's own option that carries an overlay armed per forward (H3: `OVERLAY_KWARG`).
OVERLAY = "bench_overlay"


class Inputs(msgspec.Struct, frozen=True):
    """A refill call: its seed, square side in pixels (16 a token) and DiT forwards."""

    seed: int
    size: int
    steps: int


def _arm(dit: nn.Module, overlay: nn.Module | None) -> None:
    """Arms `dit` with `overlay`, or disarms it (None)."""
    dit._bench_arming = overlay  # type: ignore[assignment]  # torch types Module.__setattr__'s value Tensor | Module; it keeps None too


def _armed(dit: nn.Module, path: str) -> Factors | None:
    from torch import nn

    arming = dit._bench_arming  # torch types Module.__getattr__ Tensor | Module: here it is `_arm`'s module or None
    # LoRAGraph puts a Factors at every site.
    return cast("Factors", arming.get_submodule(path)) if isinstance(arming, nn.Module) else None


def _construct[M: ConfigMixin](cls: type[M], config: dict[str, object]) -> M:
    """`cls.from_config(config)`, typed: it returns a tuple only with `return_unused_kwargs=True`."""
    return cast("M", cls.from_config(config))


class DiTGraph:
    """The DiT in bf16 with a permanent `lora.Hook` per site. A step loop arms it before calling its blocks
    (`QwenDiT.denoise`); a forward of the whole DiT is armed by its pre-hook from its own `attention_kwargs`
    (`QwenDiT.forward_denoise`, as H3 arms its Turbo overlay), and disarmed after."""

    def __init__(self, config: Config) -> None:
        import torch
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

        from cozy_runtime.accel import lora

        dit = _construct(QwenImageTransformer2DModel, config.section("transformer").mapping())
        dit = cast("QwenImageTransformer2DModel", dit.to(dtype=torch.bfloat16))
        _arm(dit, None)

        def arm(_: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[tuple[object, ...], dict[str, object]]:
            options = dict(cast("dict[str, object] | None", kwargs.get("attention_kwargs")) or {})  # diffusers' attention options
            _arm(dit, cast("nn.Module | None", options.pop(OVERLAY, None)))
            return args, {**kwargs, "attention_kwargs": options}

        def disarm(*_: object) -> None:
            _arm(dit, None)

        dit.register_forward_pre_hook(arm, with_kwargs=True)
        dit.register_forward_hook(disarm, always_call=True)
        for index in range(len(dit.transformer_blocks)):
            for site in SITES:
                path = f"transformer_blocks.{index}.{site}"
                dit.get_submodule(path).register_forward_hook(lora.Hook(partial(_armed, dit, path)))
        self.components: dict[str, QwenImageTransformer2DModel] = {"transformer": dit}


class LoRAGraph:
    """The Lightning LoRA's factors at every site of every block, in the checkpoint's stored dtypes."""

    def __init__(self, config: Config) -> None:
        import torch
        from torch import nn

        from cozy_runtime.internal.bench.qwen_overlay_factors import Factors

        dtypes = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16, "i64": torch.int64}

        def dtype(name: str) -> torch.dtype:
            return dtypes[config.tensor_dtype(f"{LORA_COMPONENT}.transformer_blocks.0.{SITES[0]}.{name}", default="f32")]

        weights, alpha = dtype("lora_A.weight"), dtype("alpha")
        root = nn.Module()
        root.transformer_blocks = nn.ModuleList()
        for _ in range(BLOCKS):
            block = nn.Module()
            for site in SITES:
                parent = block
                *path, leaf = site.split(".")
                for name in path:
                    if not hasattr(parent, name):
                        parent.add_module(name, nn.Module())
                    parent = getattr(parent, name)
                sizes = ((WIDTH, HIDDEN) if site.endswith("net.0.proj") else (HIDDEN, WIDTH)
                         if site.endswith("net.2") else (WIDTH, WIDTH))  # fmt: skip
                parent.add_module(leaf, Factors(*sizes, RANK, weights, alpha))
            root.transformer_blocks.append(block)
        self.components: dict[str, nn.Module] = {LORA_COMPONENT: root}


class QwenDiT(Model[DiTGraph]):
    graph: DiTGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(DiTGraph)

    @uses_components("transformer")
    def denoise(self, seed: int, overlay: nn.Module | None, inside: bool = False) -> torch.Tensor:
        """`STEPS` Euler steps from seeded noise and text rows in a plain loop, the DiT armed with `overlay` (or none)
        before it; the step reads its sigma at the step index, its rotary table and modulation rows made once."""
        import torch

        from cozy_runtime import rng

        dit = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        latents = rng.noise((1, SIDE * SIDE, 64), seed, dtype=torch.bfloat16).to(device)
        text = rng.noise((1, TEXT, 3584), seed + 1, dtype=torch.bfloat16).to(device) * 0.1
        sigmas = torch.linspace(1.0, 1.0 / (STEPS + 1), STEPS + 1, device=device)
        rotary = dit.pos_embed([[(1, SIDE, SIDE)]], max_txt_seq_len=TEXT, device=device)
        modulate = torch.zeros((1, SIDE * SIDE), dtype=torch.int, device=device) if dit.zero_cond_t else None

        def step(i: torch.Tensor, latents: torch.Tensor, text: torch.Tensor, sigmas: torch.Tensor) -> None:
            if inside:
                _arm(dit, overlay)
            sigma = sigmas.index_select(0, i)
            hidden = dit.img_in(latents)
            timestep = sigma.to(hidden.dtype)
            if dit.zero_cond_t:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
            encoder = dit.txt_in(dit.txt_norm(text))
            temb = dit.time_text_embed(timestep, hidden)
            for block in dit.transformer_blocks:
                encoder, hidden = block(hidden_states=hidden, encoder_hidden_states=encoder,
                                        encoder_hidden_states_mask=None, temb=temb, image_rotary_emb=rotary,
                                        joint_attention_kwargs=None, modulate_index=modulate)  # fmt: skip
            if dit.zero_cond_t:
                temb = temb.chunk(2, dim=0)[0]
            velocity = dit.proj_out(dit.norm_out(hidden, temb))
            latents.copy_(latents + (sigmas.index_select(0, i + 1) - sigma).to(latents.dtype) * velocity)

        _arm(dit, None if inside else overlay)
        try:
            index = torch.arange(STEPS, device=device)
            for i in range(STEPS):
                step(index[i : i + 1], latents, text, sigmas)
        finally:
            _arm(dit, None)
        return latents.float().cpu()

    @uses_components("transformer")
    def forward_denoise(self, inputs: Inputs, overlay: nn.Module | None, after: Callable[[int], None]) -> torch.Tensor:
        """`inputs.steps` Euler steps through whole DiT forwards from seeded noise and text rows, each armed with
        `overlay` through its `attention_kwargs` (or a base forward); `after(step)` runs after each forward."""
        from itertools import pairwise

        import torch

        from cozy_runtime import rng

        dit = self.graph.components["transformer"]
        device = torch.device(str(self.device))
        side = inputs.size // 16
        latents = rng.noise((1, side * side, 64), inputs.seed, dtype=torch.bfloat16).to(device)
        text = rng.noise((1, 64, 3584), inputs.seed + 1, dtype=torch.bfloat16).to(device) * 0.1
        sigmas = torch.linspace(1.0, 1.0 / (inputs.steps + 1), inputs.steps + 1).tolist()
        with torch.inference_mode():
            for step, (sigma, following) in enumerate(pairwise(sigmas)):
                options = {} if overlay is None else {OVERLAY: overlay}
                velocity = dit(
                    hidden_states=latents,
                    timestep=torch.full((1,), sigma, dtype=torch.bfloat16, device=device),
                    encoder_hidden_states=text,
                    encoder_hidden_states_mask=None,
                    img_shapes=[[(1, side, side)]],
                    attention_kwargs=options,
                    return_dict=False,
                )[0]
                latents = latents + (following - sigma) * velocity
                after(step)
        return latents.float().cpu()


class Lightning(Model[LoRAGraph]):
    graph: LoRAGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(LoRAGraph)

    @uses_components(LORA_COMPONENT)
    def denoise(self, base: QwenDiT, seed: int, inside: bool = False) -> torch.Tensor:
        return base.denoise(seed, self.graph.components[LORA_COMPONENT], inside)

    @uses_components(LORA_COMPONENT)
    def forward_denoise(self, base: QwenDiT, inputs: Inputs, after: Callable[[int], None]) -> torch.Tensor:
        """`base.forward_denoise` with this LoRA handed to each forward, as H3's `_sample` hands its overlay through
        `attention_kwargs` while its own scope holds the factors."""
        return base.forward_denoise(inputs, self.graph.components[LORA_COMPONENT], after)


def _bake_kernels(device: torch.device, ctx: Context, tel: Telemetry) -> None:
    """The LoRA bake kernels serving before the calls, so a bake never builds inside them."""
    import torch

    from cozy_runtime.internal import kernel_ready, lora_bake

    kernel = f"{lora_bake.KERNELS.name} cuda:{device.index or 0}"
    began = time.monotonic()
    while (slot := kernel_ready._SLOTS.get(kernel)) is not None and slot.state == "opening":
        ctx.raise_if_cancelled()
        if time.monotonic() - began > 900:
            raise RuntimeError("the LoRA bake kernels did not build in 15 minutes")
        time.sleep(1)
    try:
        opener = partial(kernel_ready.TRITON[0], lora_bake.KERNELS, partial(lora_bake.probe, torch, device), device=device)
        kernel_ready.wait(kernel, opener)
    except Exception as error:  # noqa: BLE001 - the bake then runs eagerly, and the run says why
        tel.log("the LoRA bake kernels failed; baking eagerly", reason=str(error)[:300])


def replay(model: QwenDiT, turbo_lora: Lightning, seed: int, variant: str, ctx: Context, tel: Telemetry) -> dict[str, bool]:
    """`variant` (replay or inside): its checks; raises when one fails."""
    import torch

    from cozy_runtime import graphs

    _bake_kernels(torch.device(str(model.device)), ctx, tel)
    if variant == "inside":
        before = turbo_lora.denoise(model, seed)
        inside = turbo_lora.denoise(model, seed, inside=True)
        checks = {"inside_equals_before": bool(torch.equal(inside, before))}
    elif variant == "replay":
        model.denoise(seed, None)
        model.denoise(seed + 1, None)
        armed = turbo_lora.denoise(model, seed + 2)
        eager = graphs.BYPASS.set(True)
        try:
            reference = turbo_lora.denoise(model, seed + 2)
            base = model.denoise(seed + 2, None)
        finally:
            graphs.BYPASS.reset(eager)
        checks = {
            "armed_equals_eager": bool(torch.equal(armed, reference)),
            "lora_changes_it": not torch.equal(reference, base),
        }
    else:
        raise ValueError(f"unknown variant {variant!r}; replay or inside")
    if not all(checks.values()):
        raise RuntimeError(f"overlay {variant}: {checks}")
    return checks
