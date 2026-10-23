from __future__ import annotations

from typing import Any

from cozy_runtime.author import Loader, Model, uses_components


class Tiny(Model[object]):
    def load(self, loader: Loader) -> None:
        return None

    @uses_components("unet")
    def denoise(self, state: Any) -> Any:
        return state


class Speaker(Model[object]):
    def load(self, loader: Loader) -> None:
        return None

    @uses_components("text")
    def encode(self, state: Any) -> Any:
        return state
