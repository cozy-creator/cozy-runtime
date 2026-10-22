"""A release gate's qualification table: `python -m cozy_runtime.cli.qualify <dir>` measures
the decode suite on this process's first card over `qualification.TABLE_DTYPES` and writes its
table and its kernel verdicts into `<dir>` (the source tree's `qualification.SHIPPED`), replacing
that card architecture's."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import msgspec

from cozy_runtime import __version__
from cozy_runtime.internal import arch, attention, calibrate, prebuilt, qualification, verify
from cozy_runtime.internal.encoding import launch_providers, measure_device, measure_runtime
from cozy_runtime.internal.probe import Observation, cells, qualify


@dataclass(slots=True)
class Progress:
    """`PROGRESS <done>/<total> <what>` lines: the hand-run oracle reports them as the run's
    progress, and its watch ends a run that stops printing them."""

    total: int
    done: int = 0

    def said(self, line: str) -> None:
        self.done += 1
        print(f"PROGRESS {self.done}/{self.total} {line}", flush=True)

    def decoded(self, one: Observation) -> None:
        self.said(f"decode {one.implementation} {one.output_dtype} passed={one.evidence.passed}")

    def calibrated(self, one: calibrate.Measured) -> None:
        runs = [c for c in one.candidates if c.warm_us]
        best = min(runs, key=lambda c: c.warm_us[2], default=None)
        self.said(
            f"gemm {one.format} {one.m}x{one.n}x{one.k}: {len(one.candidates)} candidates, "
            f"{sum(not c.ok for c in one.candidates)} failed, best "
            f"{best.choice if best else '-'} {best.warm_us[2] if best else 0:.1f} us"
        )


def main(argv: Sequence[str]) -> int:
    import torch

    device = measure_device(torch, 0)
    runtime = measure_runtime(torch, __version__)
    dtypes = qualification.TABLE_DTYPES
    cuda = device.kind == "cuda"
    family = arch.row(device.sm).family if cuda else ""
    provenance: dict[str, object] = {"device": device.document(), "runtime": runtime.document()}
    providers = launch_providers()
    progress = Progress(
        len(list(cells(providers, dtypes))) + (calibrate.shapes(device.sm) if family else 0)
    )
    calibrations: dict[str, calibrate.Calibration] = {}
    ops: prebuilt.Prebuilt | None = None
    with verify.measuring() as verdicts:
        _, table = qualify(
            torch,
            providers,
            device,
            dtypes,
            release=__version__,
            runtime=runtime,
            progress=progress.decoded,
        )
        if cuda:
            # The exact prebuilt arm and available optional attention arms are explicit
            # development probes. Missing builds remain refusals, never fabricated facts.
            ops, _ = prebuilt.library(torch, device.index, device.sm)
            if ops is not None:
                provenance["card"] = calibrate.card(torch, device.index, device.sm).facts()
                provenance["clocks_before"] = calibrate.clocks(device.index)
                calibrations = calibrate.run(torch, ops, device.sm, progress=progress.calibrated)
                provenance["clocks_after"] = calibrate.clocks(device.index)
            if device.sm == 90:
                prebuilt.fa3(torch, device.index, device.sm)
            for name in attention.walk(arch.row(device.sm), 4):
                with suppress(attention.AttentionRefusal):
                    attention._resolve(attention.known()[name], device, wait=False)

    # Each format's GEMM verdict holds its probe and its calibration at the packages' shapes.
    rules = {}
    for name, found in calibrations.items():
        if ops is None:
            break
        identity = prebuilt.gemm_kernel(name, ops, device.sm)
        probed = verdicts.get((identity, device.sm), verify.Verdict(False, "never probed"))
        verdicts[identity, device.sm] = verify.Verdict(
            probed.ok and found.verdict.ok,
            f"probe: {probed.detail}; calibration: {found.verdict.detail}",
        )
        rules[identity] = found.rules
    directory, name = Path(argv[0]), qualification.table_name(device, runtime)
    directory.mkdir(parents=True, exist_ok=True)
    qualification.write_table(directory / f"{name}.json", table)
    evidence = {f: msgspec.to_builtins(found.evidence) for f, found in calibrations.items()}
    kernels = verify.publish(directory, device.sm, verdicts, rules, provenance, evidence)
    print(
        json.dumps(
            {
                "table": name,
                "observations": len(table.observations),
                "passed": sum(one.evidence.passed for one in table.observations),
                "kernels": kernels.name,
                "verdicts": {key.name: v.ok for (key, _), v in verdicts.items()},
                "rules": {key.name: len(found) for key, found in rules.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
