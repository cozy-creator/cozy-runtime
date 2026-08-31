"""Isolated import probe executed by a materialized environment's own Python."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.machinery import PathFinder


class ProbeFailure(RuntimeError):
    """The installed fixture imports did not expose their exact expected values."""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or ())
    if len(args) != 2:
        raise ProbeFailure("usage: environment_imports.py OVERLAY cpu|cuda")
    sys.path.append(args[0])
    if args[1] == "cpu":
        for metadata in ("package", "pyproject", "uv"):
            if PathFinder.find_spec(metadata, [args[0]]) is not None:
                raise ProbeFailure(f"project metadata {metadata!r} became an import root")
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
    if args[1] == "cuda":
        import torch

        fact = (torch.version.cuda, torch.cuda.is_available())
        if fact != ("13.0", False):
            raise ProbeFailure(f"unexpected installed CUDA fact: {fact!r}")
        print("13.0 False")
        return 0
    raise ProbeFailure("usage: environment_imports.py OVERLAY cpu|cuda")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
