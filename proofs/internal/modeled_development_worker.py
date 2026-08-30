"""Live cardless refusal arm through the modeled development worker and attempt kernel."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "proofs" / "fixtures" / "modeled-development-package"
sys.path.insert(0, str(ROOT / "src"))

from cozy_runtime.internal import descriptor  # noqa: E402
from cozy_runtime.internal.config import read_config  # noqa: E402
from cozy_runtime.internal.discovery import discover  # noqa: E402
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest, run_slice  # noqa: E402
from cozy_runtime.internal.worker.session import WorkerOptions  # noqa: E402
from cozy_runtime.protocol import worker_pb2 as pb  # noqa: E402

SOURCE = "sha256:" + "7" * 64


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("home", type=Path)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args(argv)
    environment = dict(os.environ)
    environment["COZY_HOME"] = str(args.home.resolve())
    config = read_config(environment)
    found = discover(FIXTURE)
    captured: dict[str, Any] = {}

    def hook(owner: LocalRecordOwner, _worker: Any) -> None:
        captured["owner"] = owner

    with contextlib.chdir(FIXTURE):
        outcome = run_slice(
            config,
            LocalRequest(entrypoint="render", payload={"value": 3}, request_id="modeled-red"),
            workspace=args.workspace.resolve(),
            package_revision_digest=SOURCE,
            package_descriptor=descriptor.build(found),
            found=found,
            development_artifact_store=args.home.resolve() / "cas",
            options=WorkerOptions(root=args.workspace.resolve() / "worker", no_warm=True),
            record_owner_hook=hook,
        )
    owner = captured["owner"]
    assert isinstance(owner, LocalRecordOwner)
    assert outcome.ended == "worker_error"
    assert "development_modeled_accelerator_unavailable" in outcome.error
    reports = [report for report in owner.reports if report.placements]
    assert reports
    status = reports[-1].placements[0]
    assert status.materialization == pb.MaterializationState.MATERIALIZATION_STATE_FAILED
    assert status.serving == pb.ServingState.SERVING_STATE_OFFLINE
    assert any(
        fault.reason == "development_modeled_accelerator_unavailable"
        for fault in reports[-1].faults
    )
    print(
        "modeled development worker: cardless prepare refused typed, published FAILED/OFFLINE, "
        "and the one-shot owner exited without a readiness timeout"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
