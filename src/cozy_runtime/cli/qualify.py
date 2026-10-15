"""A release gate's qualification table: `python -m cozy_runtime.cli.qualify <dir>` measures
the decode suite on this process's first card over `qualification.TABLE_DTYPES` and writes its
table into `<dir>` (the source tree's `qualification.SHIPPED`), replacing the stale tables of
the same card architecture and torch minor."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cozy_runtime import __version__
from cozy_runtime.internal import qualification
from cozy_runtime.internal.encoding import launch_providers, measure_device, measure_runtime
from cozy_runtime.internal.probe import qualify


def main(argv: Sequence[str]) -> int:
    import torch

    device = measure_device(torch, 0)
    runtime = measure_runtime(torch, __version__)
    dtypes = qualification.TABLE_DTYPES
    _, table = qualify(
        torch, launch_providers(), device, dtypes, release=__version__, runtime=runtime
    )
    directory, name = Path(argv[0]), qualification.table_name(device, runtime)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob(f"{name.rsplit('-', 2)[0]}-*.json"):
        stale.unlink()
    qualification.write_table(directory / f"{name}.json", table)
    passed = sum(one.evidence.passed for one in table.observations)
    print(json.dumps({"table": name, "observations": len(table.observations), "passed": passed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
