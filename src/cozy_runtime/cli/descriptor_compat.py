"""Process boundary for Runtime's PackageDescriptor compatibility classifier."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cozy_runtime.internal import descriptor
from cozy_runtime.internal.descriptor_compat import classify_bytes

USAGE = "python -m cozy_runtime.cli.descriptor_compat <previous.json> <candidate.json>"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    if len(args) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        previous = Path(args[0]).read_bytes()
        candidate = Path(args[1]).read_bytes()
        result = classify_bytes(previous, candidate)
    except (OSError, descriptor.StalePackageDescriptor) as exc:
        print(f"descriptor compatibility refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"classification": result.classification, "reasons": list(result.reasons)},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
