"""An upload converts and publishes a model larger than the machine's disk.

A loopback supervisor admits writes against a disk smaller than the model's source alone;
a sampler proves the store never allocated more. The upload is killed mid-download and
resumed by the run's next attempt in a fresh process: committed conversion and custody
are never redone, and at most one source member moves twice. A machine with room
publishes the same checkpoint, evicts nothing, and reuses it all without moving a byte.
"""

from __future__ import annotations

import hashlib
import queue
import threading
from pathlib import Path
from weakref import WeakValueDictionary

import pytest
import tensorfs

import upload_fixture as fixture
from cozy_runtime import canonical_json
from cozy_runtime.internal import source_interfaces, storage_admission
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.source_calls import SourceCalls
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from upload_standins import Hub, HuggingFace, Supervisor

MIB = 1 << 20
DESTINATION = "example/bounded"
REGISTRY = (Path(__file__).parent / "testdata/native_source/sharded-registry.json").read_bytes()
REQUEST = {
    "repository": fixture.REPOSITORY,
    "revision": fixture.REVISION,
    "destination": DESTINATION,
    "carriers": [fixture.INDEX],
    "profiles": [fixture.PROFILE],
}


def upload(workspace: Workspace, parent: str, attempt: int = 1) -> pb.NativeSourceCommand:
    raw = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="upload-fixture", job_descriptor_id="sha256:" + "44" * 32
            )
        )
    )
    spec = hashlib.sha256(raw).digest()
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id=parent,
            attempt_ordinal=attempt,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=raw,
        ),
    )
    workspace.mark_running(
        "owner",
        pb.AttemptAccepted(request_id=parent, attempt_ordinal=attempt, invocation_spec_digest=spec),
    )
    intent = canonical_json.encode(
        {"module": source_interfaces.MODULE, "export": "upload_huggingface", "request": REQUEST}
    )
    command = pb.NativeSourceCommand(
        service_id=workspace_sources.identity("owner", parent, 0),
        operation=pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
        parent_call=pb.ChildCallRequest(
            parent_request_id=parent,
            parent_attempt_ordinal=attempt,
            parent_invocation_spec_digest=spec,
            call_index=0,
            module=source_interfaces.MODULE,
            export="upload_huggingface",
            intent_digest=hashlib.sha256(intent).digest(),
            request_canonical_bytes=canonical_json.encode(REQUEST),
        ),
    )
    workspace_sources.accepted(workspace, "owner", command)
    return command


def run(
    calls: SourceCalls,
    messages: queue.Queue[pb.NativeSourceStatus],
    command: pb.NativeSourceCommand,
    started: threading.Event | None = None,
) -> pb.NativeSourceStatus:
    command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
    calls.handle(command)
    resolved = messages.get(timeout=120)
    assert resolved.state == pb.NATIVE_SOURCE_STATE_RESOLVED, resolved.safe_code
    command.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
    command.selection.CopyFrom(resolved.selection)
    calls.handle(command)
    if started is not None:
        started.set()
    return messages.get(timeout=900)


@pytest.fixture(autouse=True)
def isolated_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Worker leaked by another test must not become this store's admission owner.
    monkeypatch.setattr(storage_admission, "_reclaimers", WeakValueDictionary())


def test_upload_larger_than_the_disk_resumes_and_matches_an_unbounded_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = fixture.files()
    source = sum(len(body) for member, body in files.items() if member != fixture.INDEX)
    shard = max(len(body) for body in files.values())
    capacity, reserve = 448 * MIB, 8 * MIB
    assert capacity < source, "the disk must be smaller than the source alone"

    origin, hub = HuggingFace(files), Hub()
    root = tmp_path / "bounded"
    supervisor = Supervisor(root, capacity, reserve)
    with origin.running() as hf, hub.running(), supervisor.running() as channel:
        monkeypatch.setattr(storage_admission, "_channel", channel)
        workspace = Workspace(root)
        messages: queue.Queue[pb.NativeSourceStatus] = queue.Queue()

        def calls() -> SourceCalls:
            return SourceCalls(
                workspace,
                lambda: "owner",
                messages.put,
                lambda _: True,
                endpoints={"huggingface": hf},
                native_registry=REGISTRY,
                publication=lambda *_: hub.client(),
            )

        # Kill the machine's native work once three shards' bytes have moved.
        first, crashed = calls(), threading.Event()
        interrupted = upload(workspace, "crashing-script")

        def crash(total: int) -> None:
            if total >= 3 * shard and not crashed.is_set():
                crashed.set()
                with first.lock:
                    process = first.processes.get(interrupted.service_id)
                if process is not None:
                    process.kill()

        origin.watch = crash
        stopped = run(first, messages, interrupted)
        assert crashed.is_set(), (stopped.state, stopped.safe_code, stopped.safe_detail)
        assert stopped.state == pb.NATIVE_SOURCE_STATE_FAILED
        origin.watch = lambda _total: None
        before_resume = origin.total()
        uploaded_before = set(hub.uploads)
        assert uploaded_before, "custody must overlap conversion before the crash"

        # The same run's next attempt, in a fresh process, resumes from committed progress.
        resumed = upload(workspace, "crashing-script", attempt=2)
        finished = run(calls(), messages, resumed)
        assert finished.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, finished.safe_code
        checkpoint = canonical_json.decode(finished.result_canonical_bytes)
        assert checkpoint["destination"] == DESTINATION
        assert checkpoint["observation"] == "acknowledged"

        refetched = origin.total() - source
        assert 0 <= refetched <= shard, f"{refetched} B moved twice"
        assert before_resume < source
        moved_once = [m for m, n in origin.served.items() if n == len(files[m])]
        assert len(moved_once) >= len(files) - 2, "only the interrupted shard may move twice"
        assert all(count == 1 for count in hub.uploads.values()), "an object was uploaded twice"
        assert supervisor.peak <= capacity, f"allocated {supervisor.peak} B on {capacity} B"
        assert supervisor.refusals, "the disk never constrained the upload"

        assert checkpoint["checkpoint"] in hub.checkpoints
        published = hub.publications[resumed.service_id]["objects"]
        assert all(key in hub.verified for key in published)
        # Every converted object reached the Hub through a custody publication; the
        # final publication moved only the composed documents.
        assert sum(published.values()) > source - len(files[fixture.INDEX])
        print(
            f"bounded upload: disk {capacity} B, source {source} B, "
            f"output {sum(published.values())} B, peak {supervisor.peak} B, "
            f"{supervisor.refusals} refusals, {refetched} B moved twice, "
            f"{len(hub.uploads)} objects uploaded once each"
        )
        # Progress holds are dropped once the checkpoint retains every object.
        progress = {op for op in hub.publications if op.startswith("ingest-")}
        assert progress and progress <= hub.abandoned

        # The same call replays its acknowledged checkpoint without moving a byte.
        moved = origin.total(), hub.uploaded_bytes
        replay = calls()
        resumed.phase = pb.NATIVE_SOURCE_PHASE_EXECUTE
        replay.handle(resumed)
        again = messages.get(timeout=60)
        assert again.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED
        assert again.result_canonical_bytes == finished.result_canonical_bytes
        assert (origin.total(), hub.uploaded_bytes) == moved

    # A disk with room converts the same bytes into the same checkpoint and evicts
    # nothing: the source and the published model stay local, so a later call on this
    # machine reuses them without moving a byte.
    roomy = tmp_path / "roomy"
    wide, other = HuggingFace(files), Hub()
    room = Supervisor(roomy, 4 * source, reserve)
    with wide.running() as hf, other.running(), room.running() as channel:
        monkeypatch.setattr(storage_admission, "_channel", channel)
        workspace = Workspace(roomy)
        messages = queue.Queue()

        def roomy_calls() -> SourceCalls:
            return SourceCalls(
                workspace,
                lambda: "owner",
                messages.put,
                lambda _: True,
                endpoints={"huggingface": hf},
                native_registry=REGISTRY,
                publication=lambda *_: other.client(),
            )

        result = run(roomy_calls(), messages, upload(workspace, "roomy-script"))
        assert result.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, result.safe_code
        published = canonical_json.decode(result.result_canonical_bytes)["checkpoint"]
        assert published == checkpoint["checkpoint"]
        assert wide.total() == source
        assert not room.refusals, "a disk with room refused a pass"
        kept = tensorfs.Store.open(str(roomy))
        for member, body in files.items():
            if member != fixture.INDEX:  # the index plans; conversion never needs its body
                assert kept.contains(hashlib.sha256(body).hexdigest()), "source evicted with room"
        for row in kept.walk(published):
            assert kept.contains(str(row["id"])[7:]), "published model evicted with room"
        moved = wide.total(), other.uploaded_bytes

        # A later upload of the same model on this machine moves nothing.
        again = run(roomy_calls(), messages, upload(workspace, "reuse-script"))
        assert again.state == pb.NATIVE_SOURCE_STATE_SUCCEEDED, again.safe_code
        assert canonical_json.decode(again.result_canonical_bytes)["checkpoint"] == published
        assert (wide.total(), other.uploaded_bytes) == moved
