"""One oversized request must not refuse every later request of its placement (cr-190).

Measured live on Qwen Image 2.1 (RTX 3090 and L40S): a ten-reference edit died of a device
OOM, and every later request of that placement -- a one-reference edit that had succeeded
bit-exactly minutes earlier included -- was refused `device_shortfall` without running,
until the rental ended. The retry rule (cr-097) remembered the OOM per PLACEMENT, so it
answered "the same request asked twice" for requests that were not the same.

This drives the real worker, executors, fill plane and lane arbitration on device 0 with
`corpus/tenant` plus one `burst` entrypoint that allocates request-sized device scratch.
Nothing in the runtime is patched: the OOM is the allocator's own answer.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cozy_runtime.internal import accel
from cozy_runtime.internal.local import LocalRequest
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_shared_lane import (
    DEVICE,
    PACKAGE,
    TenantStore,
    Turn,
    TwoTenantOwner,
    _drive,
    _workspace,
    needs_tenant,
    write_tenant_store,
)

BURST = '''

class BurstInput(msgspec.Struct, forbid_unknown_fields=True):
    scratch_bytes: int = 0


@app.entrypoint
def burst(ctx: Context, payload: BurstInput, model: TenantModel, tel: Telemetry) -> TouchOutput:
    """Touch the resident model, then hold request-sized device scratch."""
    import torch

    view = model.for_request(ctx, seed=1)
    checksum = model.touch(view._seed)
    scratch = torch.empty(payload.scratch_bytes, dtype=torch.uint8, device="cuda")
    scratch.fill_(1)
    torch.cuda.synchronize()
    del scratch
    return TouchOutput(checksum=checksum, components=sorted(model.pipe.components))
'''

BINDING = """
[bindings."burst.models.model"]
model = "cozytest/tenant"
release = "1"
lane = "bf16"
"""


def burst_package(root: Path) -> Path:
    package = root / "burst-package"
    shutil.copytree(PACKAGE, package)
    # Its own module name: `tiny_tenant` is already imported by other tests in this process.
    source = package / "burst_tenant.py"
    (package / "tiny_tenant.py").rename(source)
    source.write_text(source.read_text() + BURST)
    manifest = package / "package.toml"
    manifest.write_text(
        manifest.read_text().replace("tiny_tenant:app", "burst_tenant:app") + BINDING
    )
    return package


class RecordingOwner(TwoTenantOwner):
    """The shared-lane owner, except a `device_shortfall` refusal is the turn's outcome.

    The base owner re-dispatches every refusal, which is right for transient ones (an ack
    in flight, a stale admission epoch) and would loop forever on a typed capacity answer.
    """

    def _on_outcome(self, outcome: pb.AttemptOutcome) -> None:
        body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        refused = int(body.get("status", 0)) == pb.OutcomeStatus.OUTCOME_STATUS_REFUSED
        if not refused or "device_shortfall" not in str(body.get("safe_message", "")):
            super()._on_outcome(outcome)
            return
        self._send(
            pb.RecordOwnerFrame(
                outcome_ack=self._env(
                    pb.AttemptOutcomeAck(
                        request_id=outcome.request_id,
                        attempt_ordinal=outcome.attempt_ordinal,
                        invocation_spec_digest=outcome.invocation_spec_digest,
                        outcome_id=outcome.outcome_id,
                        outcome_digest=outcome.outcome_digest,
                    )
                )
            )
        )
        turn = self.current
        assert turn is not None
        turn.outcome = body
        self.turns.append(turn)
        self.current = None
        if self.pending:
            self._next()
            return
        self.ended = "terminal"
        self.done.set()
        self.host.close()


def _burst(request_id: str, scratch_bytes: int) -> Turn:
    return Turn(
        "plc-burst",
        LocalRequest(
            entrypoint="burst",
            payload={"scratch_bytes": scratch_bytes},
            outputs=(),
            request_id=request_id,
        ),
    )


def _verdict(turn: Turn) -> str:
    outcome = turn.outcome
    if int(outcome.get("status", 0)) == pb.OutcomeStatus.OUTCOME_STATUS_SUCCEEDED:
        return "succeeded"
    text = f"{outcome.get('safe_message', '')} {outcome.get('cause', '')}"
    for code in ("device_shortfall", "device_out_of_memory"):
        if code in text:
            return code
    return f"other: {text[:300]}"


@needs_tenant
def test_an_oversized_request_refuses_only_its_own_repeats(tmp_path: Path) -> None:
    total = accel.device_memory(DEVICE, accel.host_backend_family()).total_bytes
    store: TenantStore = write_tenant_store(tmp_path / "store", blocks=2, scale=4)
    oversized = total  # more than the whole card: the allocator's answer is certain
    turns = [
        _burst("big-1", oversized),
        _burst("big-2", oversized),
        _burst("big-3", oversized),
        _burst("small-1", 16 << 20),
        _burst("small-2", 32 << 20),
        _burst("big-4", oversized),
    ]
    with _workspace() as root:
        package = burst_package(root)
        owner, _worker, notes = _drive(store, turns, root, package, RecordingOwner)
    verdicts = [(turn.request.request_id, _verdict(turn)) for turn in owner.turns]
    print(verdicts, flush=True)
    assert owner.ended == "terminal", (owner.ended, owner.error, owner.refusals, verdicts)
    by_request = dict(verdicts)
    assert by_request["big-1"] == "device_out_of_memory", verdicts
    # THE DEFECT: smaller, different requests ran into the placement-wide memo.
    assert by_request["small-1"] == "succeeded", (verdicts, notes[-12:])
    assert by_request["small-2"] == "succeeded", verdicts
    # THE RULE IT KEEPS (cr-097): the oversized invocation, under a lane that has not
    # changed, is refused typed instead of spending the handler again -- and a different
    # request's success does not clear that.
    repeats = [by_request[name] for name in ("big-2", "big-3", "big-4")]
    assert "succeeded" not in repeats, verdicts
    assert repeats[-1] == "device_shortfall", verdicts
    assert all(
        not turn.outcome.get("execution_started")
        for turn in owner.turns
        if _verdict(turn) == "device_shortfall"
    ), [turn.outcome for turn in owner.turns]
