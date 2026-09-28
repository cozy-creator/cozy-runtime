"""Observed codec facts for the shared image preparation profile.

Diagnostics only: any codec version is accepted, and asset fingerprinting remains the
only content-digest authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import __version__ as pillow_version
from PIL import features

from cozy_runtime.author._markers import IMAGE_PREPARATION_PROFILE


def _version(value: object) -> str:
    return value if isinstance(value, str) and 0 < len(value) <= 64 else ""


def _library(value: object) -> str:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or any(type(part) is not int or not 0 <= part <= 99999 for part in value)
    ):
        return ""
    return ".".join(str(part) for part in value)


def image_preparation_profile(codec: Any) -> dict[str, object]:
    """Return bounded observed codec facts for diagnostics; any codec version is accepted."""
    libraries = getattr(codec, "library_versions", {})
    if not isinstance(libraries, Mapping):
        libraries = {}
    facts = {
        "pyav": _version(getattr(codec, "__version__", "")),
        "pillow": pillow_version,
        "libavcodec": _library(libraries.get("libavcodec")),
        "libavformat": _library(libraries.get("libavformat")),
        "libavutil": _library(libraries.get("libavutil")),
        "libswscale": _library(libraries.get("libswscale")),
        "webp": _version(features.version("webp")),
        "zlib": _version(features.version("zlib")),
    }
    return {"profile": IMAGE_PREPARATION_PROFILE, "codecs": facts}
