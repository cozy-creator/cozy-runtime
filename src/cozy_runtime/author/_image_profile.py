"""One observed codec qualification for the shared image preparation profile.

This is a software capability check, not an input identity. Caller/worker asset
fingerprinting remains the only content-digest authority. The decoder passes its
already loaded PyAV module, so this module neither imports another codec nor creates
an import cycle with the shared image core.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from PIL import __version__ as pillow_version
from PIL import features

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._markers import IMAGE_PREPARATION_PROFILE

# Qualified by full-pixel client/core round trips on the exact released media wheels.
# Changing this closure requires the same preparation parity proof; wire compatibility
# alone does not establish image-codec behavior.
_QUALIFIED_CODECS: Mapping[str, str] = MappingProxyType(
    {
        "pyav": "18.1.0",
        "pillow": "12.3.0",
        "libavcodec": "62.28.102",
        "libavformat": "62.12.102",
        "libavutil": "60.26.102",
        "libswscale": "9.5.102",
        "webp": "1.6.0",
        "zlib": "1.3",
    }
)


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
    """Return bounded observed facts; an unsupported closure is explicitly unqualified."""
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
    return {
        "profile": IMAGE_PREPARATION_PROFILE,
        "qualified": facts == _QUALIFIED_CODECS,
        "codecs": facts,
    }


def require_image_preparation_profile(codec: Any) -> dict[str, object]:
    """Admit preparation only on the single qualified decode/resize/encode closure."""
    profile = image_preparation_profile(codec)
    if not profile["qualified"]:
        raise CapabilityError(
            "image preparation requires the qualified image codec closure",
            code="image_preparation_profile",
        )
    return profile
