"""Isolated import probe executed by a materialized environment's own Python."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.machinery import PathFinder


class ProbeFailure(RuntimeError):
    """The installed fixture imports did not expose their exact expected values."""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or ())
    if len(args) != 1:
        raise ProbeFailure("usage: environment_imports.py cpu|cuda")
    if args[0] == "cpu":
        for metadata in ("package", "pyproject", "uv"):
            if PathFinder.find_spec(metadata) is not None:
                raise ProbeFailure(f"project metadata {metadata!r} became an import root")
        import proof_environment_project
        import proof_environment_resolved

        values = (
            proof_environment_project.VALUE,
            proof_environment_resolved.VALUE,
        )
        if values != ("project-1.0.0", "resolved-2.0.0"):
            raise ProbeFailure(f"unexpected installed fixture values: {values!r}")
        print("installed")
        return 0
    raise ProbeFailure("usage: environment_imports.py cpu")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
