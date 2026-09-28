"""Cooperative, request-scoped cancellation (the flat 3 s grace is gone).

The REAL worker and a REAL executor process serve `corpus/queue`'s stepped entrypoint over
the real Claim/Offer/CancelAttempt conversation; the group arm drives a real degree-2 gloo
group over the real seam. A cancel stops the attempt at its next safe point and the same
process serves the next request; only an executor whose frames stop against its own measured
cadence is killed, and the kill carries that measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import liveness
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.protocol import worker_pb2 as pb
from local_owner import LocalRequest
from test_lane_staging import QueueOwner, _request, _serve, _workspace, needs_executor

#: longer than the flat grace this replaces, so the old watchdog would have killed the step
STEP_SECONDS = 5.0
CANCELED = pb.OutcomeStatus.OUTCOME_STATUS_CANCELED
SUCCEEDED = pb.OutcomeStatus.OUTCOME_STATUS_SUCCEEDED


def _steps(tag: str, *, steps: int, step_seconds: float, hang_at: int = -1) -> LocalRequest:
    return LocalRequest(
        entrypoint="steps",
        payload={"tag": tag, "steps": steps, "step_seconds": step_seconds, "hang_at": hang_at},
        outputs=(),
        request_id=f"cancel-{tag}",
    )


def _cancel_when(
    target: LocalRequest, ready: Callable[[AttemptRecord], bool], seen: dict[str, Any]
) -> Callable[[QueueOwner, Worker], None]:
    """Send CancelAttempt once the running attempt satisfies `ready`; keep every note and
    every GPU release the worker makes."""

    def hook(owner: QueueOwner, worker: Worker) -> None:
        seen["notes"], seen["released"] = [], {}
        narrate, release = worker.note, worker.gpu.release

        def note(kind: str, step: str) -> None:
            seen["notes"].append(step)
            narrate(kind, step)

        def released(key: str, **facts: Any) -> None:
            seen["released"][key] = facts
            release(key, **facts)

        worker.note = note  # type: ignore[method-assign]
        worker.gpu.release = released  # type: ignore[method-assign]

        def cancel() -> None:
            while not owner.done.is_set():
                held = worker.engine.live.get(target.request_id)
                if held is not None and held.state == "running" and ready(held):
                    break
                time.sleep(0.02)
            else:
                return
            seen["cancelled_at"] = time.perf_counter()
            owner._send(
                pb.RecordOwnerFrame(
                    cancel_attempt=owner._env(
                        pb.CancelAttempt(
                            request_id=target.request_id,
                            attempt_ordinal=1,
                            invocation_spec_digest=held.digest,
                            reason=pb.CancelReason.CANCEL_REASON_CLIENT,
                        )
                    )
                )
            )

        threading.Thread(target=cancel, name="test-cancel", daemon=True).start()

    return hook


@needs_executor
def test_a_cancel_mid_step_stops_at_the_boundary_and_the_same_executor_serves_next() -> None:
    stepped, following = _steps("mid-step", steps=3, step_seconds=STEP_SECONDS), _request("next", 0)
    seen: dict[str, Any] = {}

    def half_a_second_into_step_one(held: AttemptRecord) -> bool:
        return held.moves >= 1 and time.monotonic() - held.pace.moved_at >= 0.5

    with _workspace() as root:
        owner, worker = _serve(
            root, [stepped, following], _cancel_when(stepped, half_a_second_into_step_one, seen)
        )
    assert owner.ended == "terminal", (owner.ended, owner.error)
    assert int(owner.outcomes[stepped.request_id]["status"]) == CANCELED
    assert int(owner.outcomes[following.request_id]["status"]) == SUCCEEDED
    cancelled = worker.engine.history[(stepped.request_id, 1)]
    served = worker.engine.history[(following.request_id, 1)]
    # It ran on past the old 3 s grace to its next step boundary, and stopped there.
    waited = cancelled.resulted_at - seen["cancelled_at"]
    assert 3.0 < waited < STEP_SECONDS + 2.0, waited
    assert not any("forceful" in step for step in seen["notes"]), seen["notes"]
    # The warm executor served the next request.
    assert cancelled.executor_pid == served.executor_pid != 0
    assert cancelled.executor_epoch == served.executor_epoch
    # The device release carries the attempt's rank record.
    ranks = seen["released"][f"{stepped.request_id}#1"]["ranks"]
    assert [(row["rank"], row["pid"]) for row in ranks] == [(0, cancelled.executor_pid)], ranks


@needs_executor
def test_an_executor_that_stops_moving_is_killed_after_its_own_measured_window() -> None:
    step = 4.5  # 8 x this is above the noise floor, so the window is derived, not floored
    hung = _steps("hung", steps=2, step_seconds=step, hang_at=1)
    following = _request("after-kill", 0)
    seen: dict[str, Any] = {}
    with _workspace() as root:
        owner, worker = _serve(
            root, [hung, following], _cancel_when(hung, lambda held: held.moves >= 1, seen)
        )
    assert owner.ended == "terminal", (owner.ended, owner.error)
    assert int(owner.outcomes[hung.request_id]["status"]) == CANCELED
    assert int(owner.outcomes[following.request_id]["status"]) == SUCCEEDED
    killed = worker.engine.history[(hung.request_id, 1)]
    served = worker.engine.history[(following.request_id, 1)]
    assert any("no measured progress" in note for note in seen["notes"]), seen["notes"]
    forceful = [n for n in seen["notes"] if n.startswith("forceful: ")]
    assert len(forceful) == 1, seen["notes"]
    facts = json.loads(forceful[0].removeprefix("forceful: "))
    assert facts["killed"] == 1, facts
    assert facts["worst_gap_s"] >= step - 0.5, facts
    derived = liveness.STILL_FACTOR * facts["worst_gap_s"]
    assert facts["patience_s"] == pytest.approx(derived, abs=0.01), facts
    assert facts["patience_s"] > liveness.noise_floor(), facts
    assert facts["still_s"] > facts["patience_s"], facts
    # Killed no sooner than its own window after the cancel, and replaced.
    assert killed.resulted_at - seen["cancelled_at"] >= facts["patience_s"] - 0.5
    assert served.executor_pid not in (0, killed.executor_pid)


GROUP = Path(__file__).parent / "testdata" / "cancel_group.py"


def _torch_stack() -> str:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        return f"the gloo group arm needs torch and diffusers: {exc}"
    return ""


@pytest.mark.skipif(bool(_torch_stack()), reason=_torch_stack() or "")
def test_a_two_rank_group_stops_at_one_boundary_and_is_reused(tmp_path: Path) -> None:
    """Cancelled during a step's wait, rank 0 raises before commanding the next mirrored call:
    the follower runs nothing after the cancel, neither rank is poisoned, and the same two
    processes serve the next request with the same result."""
    seconds = 4.0
    result = subprocess.run(
        [
            "nice",
            "-n",
            "19",
            sys.executable,
            str(GROUP),
            "--root",
            str(tmp_path),
            "--seconds",
            str(seconds),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    cancelled = facts["cancelled"]
    assert cancelled["outcome"]["terminal"] == "canceled", cancelled
    assert not cancelled["poisoned"], cancelled
    waited = cancelled["replied_at"] - cancelled["marked"]["at"]
    assert 3.0 < waited < 2 * seconds, waited
    ranks = {row["rank"]: row for row in cancelled["ranks"]}
    assert sorted(ranks) == [0, 1], ranks
    assert ranks[1]["end_us"] < cancelled["marked"]["us"], ranks
    assert facts["after"]["outcome"]["terminal"] == "succeeded", facts["after"]
    assert facts["after"]["result"] == facts["before"]["result"], facts
    assert len({tuple(pids) for pids in facts["pids"]}) == 1, facts["pids"]
