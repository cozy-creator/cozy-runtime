"""A request's attention selection is an execution parameter of ONE construction (cr-191).

The worker arm drives the real worker, attempts, executor process and TensorFS fill on a
real card, one request after another on one warm placement, and reads which kernel
Diffusers' own dispatch actually ran from each attempt's per-rank execution record. Run
1188 failed `construction_not_loaded` because a pin loaded a second construction the
request then did not name; here a pin never loads anything.

The group arm runs the executor program with a real gloo follower on CPU: every rank
takes the request's selection, reports what it holds and what it dispatched, and restores.
CPU offers one kernel (`sdpa`), so a distinct-kernel swap across ranks is proven only on a
multi-GPU rental, not here.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tensorfs = pytest.importorskip("tensorfs")
pytest.importorskip("diffusers")

from tensorfs.derived import Config as NativeConfig  # noqa: E402
from tensorfs.derived import Derivation, Part, Target, Tensor  # noqa: E402

from cozy_runtime.internal import attention, canonical, child_env, package_interface  # noqa: E402
from cozy_runtime.internal.config import Credentials, RuntimeConfig  # noqa: E402
from cozy_runtime.internal.discovery import discover  # noqa: E402
from cozy_runtime.internal.encoding import SPEC_PLAIN, DeviceFacts  # noqa: E402
from cozy_runtime.internal.local import (  # noqa: E402
    LocalRecordOwner,
    LocalRequest,
    stage_development_placement,
)
from cozy_runtime.internal.weights_sink import weights_transaction_id  # noqa: E402
from cozy_runtime.internal.worker import triage  # noqa: E402
from cozy_runtime.internal.worker.session import Worker, WorkerOptions  # noqa: E402
from cozy_runtime.protocol import documents  # noqa: E402
from cozy_runtime.protocol import worker_pb2 as pb  # noqa: E402
from test_end_to_end import NO_EXECUTOR  # noqa: E402

TESTDATA = Path(__file__).parent / "testdata"
PACKAGE = TESTDATA / "request_attention"
sys.path[:0] = [str(PACKAGE), str(TESTDATA)]
from request_attention import site  # type: ignore[import-not-found]  # noqa: E402
from request_attention_group import ranks  # type: ignore[import-not-found]  # noqa: E402


def _card() -> str:
    """Why this host cannot run the worker arm, or "" when it can."""
    if NO_EXECUTOR:
        return NO_EXECUTOR
    if not torch.cuda.is_available():
        return "the worker arm swaps kernels on a real card; no CUDA device is visible"
    major, minor = torch.cuda.get_device_capability(0)
    device = DeviceFacts(
        kind="cuda",
        name=torch.cuda.get_device_name(0),
        sm=major * 10 + minor,
        driver="",
        configuration="",
        index=0,
    )
    try:
        attention.pinned("cudnn", device)
    except attention.AttentionRefusal as exc:
        return f"the pinned arm needs cuDNN attention here: {exc}"
    return ""


NO_CARD = _card()
SDPA = {"flash", "efficient", "cudnn", "math"}  # torch's own choice behind `sdpa`


def _publish(store: Path) -> None:
    """`test/attention@1`: the fixture's one attention site, in bf16, as a TensorFS release."""
    torch.manual_seed(7)
    rows = {
        key: value.to(torch.bfloat16).contiguous() for key, value in site().state_dict().items()
    }
    opened = tensorfs.Store.ensure(str(store))
    shapes = {key: tuple(value.shape) for key, value in rows.items()}
    transaction = opened.begin_derived(
        weights_transaction_id("request-attention", "attention", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={
                "dit": Target(
                    add={
                        key: Tensor("bf16", shape, SPEC_PLAIN, {"value": Part("bf16", shape)})
                        for key, shape in shapes.items()
                    }
                )
            },
            configs={"model": NativeConfig("add")},
            order=tuple(("dit", key) for key in rows),
        ).native_arguments(sum(v.numel() * 2 for v in rows.values()) + 1_000_000),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for key, value in rows.items():
        transaction.add_part(
            "dit", key, "value", io.BytesIO(value.view(torch.int16).numpy().tobytes())
        )
    transaction.add_config("model", io.BytesIO(canonical.write({})))
    receipt = transaction.commit()
    manifest, length = "sha256:" + receipt["manifest"]["sha256"], int(receipt["manifest"]["length"])
    operation = opened.begin_operation("publish-attention", "test", "attention")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1", "main", manifest, length)


class Sequence(LocalRecordOwner):
    """`LocalRecordOwner`'s conversation, one request after another on its one placement."""

    def __init__(self, requests: list[LocalRequest], *args: Any, **kwargs: Any) -> None:
        super().__init__(requests[0], *args, **kwargs)
        self.pending = requests[1:]
        self.outcomes: list[dict[str, Any]] = []

    def _on_outcome(self, outcome: pb.AttemptOutcome) -> None:
        self.terminal = outcome
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
        self.outcomes.append(documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody))
        if self.pending:
            self.request, self.pending = self.pending[0], self.pending[1:]
            self.accepted, self.dispatched = None, False  # the next ready report dispatches it
            return
        self.ended = "terminal"
        self.done.set()
        self.host.close()


def _serve(root: Path, requests: list[LocalRequest]) -> tuple[Sequence, Worker, list[str]]:
    base = {
        k: v
        for k, v in os.environ.items()
        if not child_env.erased(k) and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
    }
    config = RuntimeConfig(
        cozy_home=root / "home",
        credentials=Credentials(),
        child_base_env=tuple(sorted(base.items())),
    )
    _publish(root / "store")
    workspace = root / "run"
    staged = stage_development_placement(
        config.cozy_home,
        package_interface.build(discover(PACKAGE)),
        package="local/development-package",
        release="0.0.0",
        installation_id="local-request-attention",
        artifact_cache=workspace / "artifact-cache",
        tensorfs_root=root / "store",
        project=PACKAGE,
    )
    owner = Sequence(
        requests,
        dict(staged.bindings),
        workspace,
        package_installation_id=staged.placement.development.installation_id,
        placement=staged.placement,
    )
    options = WorkerOptions(
        root=workspace / "worker",
        devices="0",
        instance_id="ins-request-attention",
        tensorfs_root=staged.tensorfs_root,
        artifact_cache=staged.artifact_cache,
        development_project=PACKAGE,
        development_package=staged.placement.development.package,
        development_release=staged.placement.development.release,
        development_installation_id=staged.placement.development.installation_id,
    )
    worker = Worker(config, options, owner.host)
    notes: list[str] = []
    narrate = worker.note

    def note(kind: str, step: str) -> None:
        notes.append(step)
        narrate(kind, step)

    worker.note = note  # type: ignore[method-assign]
    owner.native_workspace = worker.workspace
    owner.watch(worker)
    with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
        worker.run()
    return owner, worker, notes


def _ranks(worker: Worker, outcome: dict[str, Any]) -> dict[int, dict[str, str]]:
    """rank -> requested, observed and dispatched attention, from the triage bundle."""
    subject = outcome["triage_bundle"]["subject_id"]
    bundle = triage.BundleStore.read(worker.root / "triage", subject, worker.records)
    return ranks(bundle["measurements"].get("execution") or {"ranks": []})


@pytest.mark.skipif(bool(NO_CARD), reason=NO_CARD or "")
def test_pins_are_request_scoped_on_one_warm_construction() -> None:
    """Pinned first on a fresh executor, then ordinary, pinned, unsupported and ordinary
    again: each request runs its own kernel on the ONE construction and executor."""
    arms = [
        ("pinned-first", "dit=cudnn"),
        ("ordinary", ""),
        ("pinned", "dit=cudnn"),
        ("unsupported", "dit=flashinfer-bf16-fp8"),
        ("ordinary-after", ""),
    ]
    requests = [
        LocalRequest("generate", {"value": 1}, (), attention_kernel=pin, request_id=name)
        for name, pin in arms
    ]
    with tempfile.TemporaryDirectory(prefix="cz-attn.", dir="/tmp") as raw:
        owner, worker, notes = _serve(Path(raw), requests)
        assert owner.ended == "terminal", (owner.ended, owner.error)
        outcomes = {row["request_id"]: row for row in owner.outcomes}
        served = {name: _ranks(worker, outcomes[name]) for name, _ in arms}
    status = {name: pb.OutcomeStatus.Name(outcomes[name]["status"]) for name, _ in arms}
    assert status == {
        "pinned-first": "OUTCOME_STATUS_SUCCEEDED",
        "ordinary": "OUTCOME_STATUS_SUCCEEDED",
        "pinned": "OUTCOME_STATUS_SUCCEEDED",
        "unsupported": "OUTCOME_STATUS_REFUSED",
        "ordinary-after": "OUTCOME_STATUS_SUCCEEDED",
    }, {name: outcomes[name].get("safe_message") for name, _ in arms}
    assert "attention_kernel_unsupported" in outcomes["unsupported"]["safe_message"]
    # Each request dispatched the kernel it asked for, and only that one.
    for name, pin in arms:
        if name == "unsupported":
            assert served[name] == {}, served[name]
        elif pin:
            assert served[name] == {0: {"requested": pin, "observed": "cudnn", "impl": "cudnn"}}
        else:
            row = served[name][0]
            assert (row["requested"], row["observed"]) == ("", "sdpa"), served[name]
            assert row["impl"] in SDPA, served[name]
    # One construction was built, once, and one executor served every request.
    assert sum(" loaded in " in step for step in notes) == 1, notes
    entered = [row.body for row in worker.records.records if row.kind == "entered"]
    assert [row["request_id"] for row in entered] == [
        "pinned-first",
        "ordinary",
        "pinned",
        "ordinary-after",
    ]
    assert len({(row["executor_epoch"], row["executor_pid"]) for row in entered}) == 1, entered


GROUP = TESTDATA / "request_attention_group.py"


def test_every_rank_takes_reports_and_restores_the_request_selection(tmp_path: Path) -> None:
    """A degree-2 gloo group: base, unsupported pin, pinned and base again on the same
    ranks, each rank's held and dispatched kernels reported per attempt."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        ["nice", "-n", "19", sys.executable, str(GROUP), "--root", str(tmp_path)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    assert facts["refused"] == "attention_kernel_unsupported", facts
    for name, pin in (("base", ""), ("pinned", "base/dit=sdpa"), ("base-again", "")):
        held = facts["attention"][name]
        assert sorted(held) == ["0", "1"], held  # JSON object keys
        for row in held.values():
            assert (row["requested"], row["observed"]) == (pin, "sdpa"), held
            assert row["impl"] in SDPA, held  # every rank dispatched it
    assert len({tuple(pids) for pids in facts["pids"]}) == 1, facts["pids"]
