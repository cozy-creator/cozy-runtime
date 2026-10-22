"""A release gate's qualification table: `python -m cozy_runtime.cli.qualify <dir>` measures
the decode suite on this process's first card over `qualification.TABLE_DTYPES` and writes its
table into `<dir>` (the source tree's `qualification.SHIPPED`), replacing the stale tables of
the same card architecture and torch minor."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from cozy_runtime import __version__
from cozy_runtime.internal import arch, attention, prebuilt, qualification, verify
from cozy_runtime.internal.encoding import launch_providers, measure_device, measure_runtime
from cozy_runtime.internal.probe import qualify


def main(argv: Sequence[str]) -> int:
    import torch

    device = measure_device(torch, 0)
    runtime = measure_runtime(torch, __version__)
    dtypes = qualification.TABLE_DTYPES
    with verify.measuring() as verdicts:
        _, table = qualify(
            torch, launch_providers(), device, dtypes, release=__version__, runtime=runtime
        )
        if device.kind == "cuda":
            # The exact prebuilt arm and available optional attention arms are explicit
            # development probes. Missing builds remain refusals, never fabricated facts.
            prebuilt.library(torch, device.index, device.sm)
            if device.sm == 90:
                prebuilt.fa3(torch, device.index, device.sm)
            for name in attention.walk(arch.row(device.sm), 4):
                with suppress(attention.AttentionRefusal):
                    attention._resolve(attention.known()[name], device, wait=False)

    directory, name = Path(argv[0]), qualification.table_name(device, runtime)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob(f"{name.rsplit('-', 2)[0]}-*.json"):
        stale.unlink()
    qualification.write_table(directory / f"{name}.json", table)
    kernel_tables = verify.publish(directory, verdicts)
    passed = sum(one.evidence.passed for one in table.observations)
    print(
        json.dumps(
            {
                "table": name,
                "observations": len(table.observations),
                "passed": passed,
                "kernel_tables": [p.name for p in kernel_tables],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
