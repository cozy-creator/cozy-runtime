"""Observed codec facts for the shared image preparation profile.

Diagnostics only: any codec version is accepted, and asset fingerprinting remains the
only content-digest authority.
"""

from __future__ import annotations

from PIL import __version__ as pillow_version
from PIL import features

from cozy_runtime.author._markers import IMAGE_PREPARATION_PROFILE


def _version(value: object) -> str:
    return value if isinstance(value, str) and 0 < len(value) <= 64 else ""


def image_preparation_profile() -> dict[str, object]:
    """Report the still-image libraries in use, without requiring an audio/video codec."""
    return {
        "profile": IMAGE_PREPARATION_PROFILE,
        "codecs": {
            "pillow": pillow_version,
            "webp": _version(features.version("webp")),
            "zlib": _version(features.version("zlib")),
        },
    }
