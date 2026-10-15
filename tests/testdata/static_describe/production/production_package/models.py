"""Layouts the `Weights[M]` arguments name (sdxl, anima, birefnet)."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Loader, Model, uses_components


class SdxlModel(Model[object], encoded_leaves="accept"):
    def load(self, loader: Loader) -> None:
        return None

    @uses_components("text_encoder", "text_encoder_2")
    def encode(self, prompt: str) -> Any:
        return None

    @uses_components("unet")
    def denoise(self, latents: Any) -> Any:
        return None

    @uses_components("vae")
    def decode(self, latents: Any) -> Any:
        return None


class AnimaModel(Model[object], encoded_leaves="accept"):
    def load(self, loader: Loader) -> None:
        return None

    @uses_components("transformer")
    def denoise(self, latents: Any) -> Any:
        return None


class BackgroundRemovalModel(Model[object]):
    def load(self, loader: Loader) -> None:
        return None

    @uses_components("model")
    def matte(self, image: Any) -> Any:
        return None
