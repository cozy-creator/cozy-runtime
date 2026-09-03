#!/usr/bin/env python3
"""Stamp an exact Git commit into release artifacts without consulting Git at runtime."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "src/cozy_runtime/_build_provenance.py"


def main() -> int:
    if len(sys.argv) != 2 or re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]) is None:
        raise SystemExit("usage: stamp-build-provenance.py <40-lowercase-hex-commit>")
    TARGET.write_text(
        f'"""Exact source revision embedded by release automation."""\n\nCOMMIT = "{sys.argv[1]}"\n'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
