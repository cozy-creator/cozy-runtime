"""Isolated import probe executed by a materialized environment's own Python."""

from __future__ import annotations

import sys
from collections.abc import Sequence


class ProbeFailure(RuntimeError):
    """The installed fixture imports did not expose their exact expected values."""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or ())
    if args == ["cpu"]:
        import proof_environment_project
        import proof_environment_resolved
        import proof_environment_wheelhouse

        values = (
            proof_environment_project.VALUE,
            proof_environment_resolved.VALUE,
            proof_environment_wheelhouse.VALUE,
        )
        if values != ("project-1.0.0", "resolved-2.0.0", "wheelhouse-3.0.0"):
            raise ProbeFailure(f"unexpected installed fixture values: {values!r}")
        print("installed")
        return 0
    if args == ["cuda"]:
        import torch

        fact = (torch.version.cuda, torch.cuda.is_available())
        if fact != ("12.9", False):
            raise ProbeFailure(f"unexpected installed CUDA fact: {fact!r}")
        print("12.9 False")
        return 0
    raise ProbeFailure("usage: environment_imports.py cpu|cuda")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
