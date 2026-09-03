"""Run ONE corpus case inside the bounded sandbox and report its verdict as JSON.

A child per case is not an optimization: `sys.addaudithook` cannot be removed, and one
case's fence must not become the next case's ambient condition. It also means a case that
kills its interpreter (v1's pgw#1659 killed a CUDA context process-wide and the drive
finished green against the corpse) costs exactly one verdict instead of every verdict after
it.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from typing import Any

from corpus.harness import Case, diff, ground_truth_census, model_for, weights_for
from cozy_runtime.internal import sandbox
from cozy_runtime.internal.derive import ConstructionFault, HarnessGap, derive

MEMORY_CAP = 24 << 30
CPU_CAP = 120


def find(module_name: str, case_name: str) -> Case:
    module = importlib.import_module(module_name)
    for case in module.CASES:
        if case.name == case_name:
            return case
    raise SystemExit(f"{module_name} has no case {case_name!r}")


def run(case: Case) -> dict[str, Any]:
    out: dict[str, Any] = {"name": case.name, "construct": case.construct, "expect": case.expect}
    truth = None
    if case.ground_truth:
        try:
            truth = ground_truth_census(case)
            out["real_destinations"] = len(truth.logical)
        except Exception as exc:
            out["ground_truth_error"] = f"{type(exc).__name__}: {exc}"

    artifact = weights_for(case, truth) if truth is not None else _bare(case)

    # Torch defers a lot of its own import graph (`torch._dynamo` lands the first time an
    # in-place init runs). Warm it BEFORE the bounds, or the address-space cap turns a
    # lazy import into a MemoryError attributed to the author's line.
    import torch._dynamo
    import torch._refs  # noqa: F401

    sandbox.install(allow_write_prefixes=())
    sandbox.bound(address_bytes=MEMORY_CAP, cpu_seconds=CPU_CAP)

    started = time.perf_counter()
    try:
        result = derive(model_for(case), artifact, component_use=case.component_use)
    except HarnessGap as exc:
        out.update(verdict="typed-harness-gap", named=exc.construct, detail=exc.message[:200])
    except ConstructionFault as exc:
        out.update(verdict="typed-refusal", named=exc.construct, detail=exc.message[:200])
    except sandbox.FenceViolation as exc:
        out.update(verdict="typed-refusal", named=exc.fence, detail=exc.detail[:200])
    except Exception as exc:  # an UNTYPED escape is itself a harness defect
        out.update(
            verdict="untyped-error",
            named=type(exc).__name__,
            detail=f"{exc}"[:200],
            traceback=traceback.format_exc()[-1200:],
        )
    else:
        out["destinations"] = len(result.logical_tensors)
        out["derived_tensors"] = len(result.derived)
        out["components"] = list(result.components)
        if truth is None:
            out["verdict"] = "derived-correct"
            out["unverified"] = "no ground truth on this box"
        else:
            divergence = diff(result.logical_tensors, truth.logical) + diff(
                result.derived, truth.derived
            )
            out["verdict"] = "derived-correct" if not divergence else "silent-wrong"
            if divergence:
                out["divergence"] = divergence[:12]
    out["ms"] = round((time.perf_counter() - started) * 1000, 1)
    return out


def _bare(case: Case) -> Any:
    from cozy_runtime.author import Artifact, Config

    return Artifact(f"corpus/{case.name}@1", {}, Config(dict(case.config)))


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("run_case must be spawned with CUDA_VISIBLE_DEVICES='' (no GPU)")
    case = find(sys.argv[1], sys.argv[2])
    print(json.dumps(run(case)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
