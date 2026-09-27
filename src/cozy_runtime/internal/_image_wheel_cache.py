"""The worker image's immutable uv cache, seeded at image build and read by uv at install.

Symlinks avoid Docker overlayfs copying a whole lower-layer Torch payload up when the
first private environment is created. uv owns resolution, hashes and cache locking.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path("/opt/cozy/dependency-seed")
INDEXES = (
    "https://pypi.org/simple",
    "https://download.pytorch.org/whl/cpu",
    "https://download.pytorch.org/whl/cu130",
)


def command(uv: Path | str, python: Path | str, requirements: Path) -> list[str]:
    """The image build's offline, hash-required install that populates the seed cache."""
    return [
        str(uv),
        "pip",
        "install",
        "--no-config",
        "--offline",
        "--no-deps",
        "--no-build",
        "--require-hashes",
        "--link-mode",
        "symlink",
        "--cache-dir",
        str(ROOT / "uv-cache"),
        "--index-url",
        INDEXES[0],
        "--index-strategy",
        "unsafe-best-match",
        *(part for index in INDEXES[1:] for part in ("--extra-index-url", index)),
        "--python",
        str(python),
        "--requirements",
        str(requirements),
    ]
