"""A function's weights output is the attempt's grant, and the executor writes it itself: straight
into the store under its replay key, shown to the run as a checkpoint item, replayed by the
same work. Nothing about weights crosses the seam to TensorD but that item."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec
import numpy as np
import pytest
import tensorfs

from cozy_runtime import weights_store
from cozy_runtime.author import CapabilityError
from cozy_runtime.author._executor_requests import Answer, CheckpointItem, Request, refuse
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from session import session
from store_fixtures import Shown, checkpoint, context, home

PACKAGE = """
from typing import Any

import msgspec

from cozy_runtime.author import App, CapabilityError, Context, Weights, WeightsOutput
from cozy_runtime.weights_store import Derivation

app = App()


class Request(msgspec.Struct):
    use_output: bool


class Result(msgspec.Struct):
    code: str
    checkpoint: str = ""
    transaction: str = ""


@app.entrypoint(accelerator=False)
def idle(payload: Request) -> Result:
    return Result("no-storage-needed")


@app.entrypoint(accelerator=False, weights=(WeightsOutput("model", max_new_bytes=4096),))
def compute(payload: Request, ctx: Context, source: Weights[Any]) -> Result:
    if not payload.use_output:
        return Result("no-storage-needed")
    try:
        base = ctx.tensorfs_source(source)
        with ctx.output("model", Derivation(base)) as out:
            out.put("unet", "extra", "F32", [2], bytes(8))
            artifact = ctx.adopt_model(out)
    except CapabilityError as exc:
        return Result(exc.code)
    return Result("granted", artifact.checkpoint, artifact.transaction)
"""


@pytest.mark.parametrize("granted", [False, True])
def test_a_granted_output_is_written_into_the_store_shown_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, granted: bool
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    source = checkpoint(store, {"unet": {"w": np.ones(2, np.float32)}}, configs={"unet": {"in": 2}})
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="storage_grant_fn:app"\n')
    (project / "storage_grant_fn.py").write_text(PACKAGE)
    shown: list[CheckpointItem] = []

    def machine(request: Request) -> Answer:
        if not isinstance(request, CheckpointItem):
            return refuse("no_durable_exchange", type(request).__name__)
        shown.append(request)
        return Answer(ok=True)

    results = []
    with session(tmp_path, discover(project)) as host:
        host.handler = machine
        # One run twice (an attempt again), then another run of the same operation.
        for spool, use_output, invoke in (
            ("first", False, {}),
            ("run", True, {}),
            ("run", True, {}),
            ("again", True, {"operation": "sha256:" + "a" * 64, "actor": "me"}),
            ("later", True, {"operation": "sha256:" + "a" * 64, "actor": "me"}),
        ):
            reply = host.run(
                "compute",
                {"use_output": use_output},
                tmp_path / spool,
                sources={"source": source},
                weights={"model": 4096} if granted else {},
                actor=invoke.get("actor", ""),
                operation=invoke.get("operation", ""),
            )
            assert reply["outcome"]["terminal"] == "succeeded", reply
            results.append(json.loads((tmp_path / spool / RESULT_DOCUMENT).read_bytes()))
    assert results[0] == {"code": "no-storage-needed", "checkpoint": "", "transaction": ""}
    if not granted:
        assert {r["code"] for r in results[1:]} == {"weights_output_ungranted"} and not shown
        return
    assert {r["code"] for r in results[1:]} == {"granted"}
    assert results[1] == results[2], "the same run's attempt replays its committed output"
    assert results[3]["checkpoint"] == results[4]["checkpoint"] == results[1]["checkpoint"]
    assert results[3]["transaction"] == results[4]["transaction"] != results[1]["transaction"]
    assert [(item.output, item.repo_id) for item in shown] == [("model", weights_store.DERIVED)] * 4
    out = tensorfs.open(shown[0].revision, cache_dir=store, local_files_only=True)
    assert out.id == results[1]["checkpoint"]
    unet = out.components["unet"]
    assert set(unet.tensors) == {"w", "extra"} and unet.config == {"in": 2}


def test_concurrent_requests_keep_their_output_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    source = checkpoint(store, {"unet": {"w": np.ones(2, np.float32)}})
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="storage_isolation_fn:app"\n')
    (project / "storage_isolation_fn.py").write_text(PACKAGE)
    shown: list[CheckpointItem] = []

    def machine(request: Request) -> Answer:
        assert isinstance(request, CheckpointItem)
        shown.append(request)
        return Answer(ok=True)

    with session(tmp_path, discover(project)) as host:
        host.handler = machine

        def run(actor: str, bound: int) -> dict[str, str]:
            spool = tmp_path / actor
            reply = host.run(
                "compute", {"use_output": True}, spool, sources={"source": source},
                weights={"model": bound}, actor=actor, operation="same-operation",
            )  # fmt: skip
            assert reply["outcome"]["terminal"] == "succeeded", reply
            return msgspec.json.decode((spool / RESULT_DOCUMENT).read_bytes(), type=dict[str, str])

        with ThreadPoolExecutor(max_workers=2) as pool:
            alice = pool.submit(run, "alice", 4096)
            bob = pool.submit(run, "bob", 4)
            assert alice.result()["code"] == "granted"
            assert bob.result()["code"] == "weights_output_bound"
    assert len(shown) == 1
    ck = tensorfs.open(shown[0].revision, cache_dir=store, local_files_only=True)
    assert set(ck.components["unet"].tensors) == {"w", "extra"}


def test_a_finished_derivation_replays_in_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    ref = checkpoint(store, {"unet": {"w": np.ones(2, np.float32)}})
    base = tensorfs.open(ref, cache_dir=store, local_files_only=True)
    ctx = context(store)
    with ctx.output("model", weights_store.Derivation(base)) as out:
        out.put("unet", "extra", "F32", [2], bytes(8))
        first = ctx.adopt_model(out)
    script = """
import sys
from pathlib import Path
import tensorfs
from cozy_runtime.weights_store import Derivation, Outputs
store = Path(sys.argv[1])
outputs = Outputs(store, request_id='test', actor='tester', code='test',
                  bounds={'model': 64 << 20}, show=lambda *args: None)
base = tensorfs.open(sys.argv[2], cache_dir=store, local_files_only=True)
with outputs.open('model', Derivation(base)) as out:
    assert out.done is not None, 'another process finds the committed replay'
    assert out.written == 0 and out.has('unet', 'extra')
    print(outputs.adopt(out).checkpoint)
"""
    replay = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "tensorfs"), ref],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert replay.stdout.strip() == first.checkpoint


def test_an_upload_failure_keeps_the_committed_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    source = checkpoint(store, {"unet": {"w": np.ones(2, np.float32)}})
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.toml").write_text('[application]\nobject="storage_failure_fn:app"\n')
    (project / "storage_failure_fn.py").write_text(PACKAGE)
    shown: list[CheckpointItem] = []

    def machine(request: Request) -> Answer:
        assert isinstance(request, CheckpointItem)
        shown.append(request)
        return Answer(ok=True)

    with session(tmp_path, discover(project)) as host:
        host.handler = machine
        reply = host.run(
            "compute", {"use_output": True}, tmp_path / "failed", sources={"source": source},
            weights={"model": 4096}, destination="alice/model",
            tensorfs={"endpoint": "file:///not-a-distribution-endpoint"},
        )  # fmt: skip
    assert reply["outcome"]["terminal"] == "failed", reply
    assert reply["outcome"]["code"] == "upload_failed", reply
    (local,) = shown
    assert local.repo_id == weights_store.DERIVED
    kept = tensorfs.open(local.revision, cache_dir=store, local_files_only=True)
    assert set(kept.components["unet"].tensors) == {"w", "extra"}
    assert not kept.missing()


def test_a_fresh_output_creates_its_component_and_replays_without_a_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    derivation = weights_store.Derivation(None, order=(("model", "table"),))
    shown: Shown = []
    ctx = context(store, "fresh", shown=shown)
    with ctx.output("model", derivation) as output:
        output.put("model", "table", "U8", [4], b"data")
        first = ctx.adopt_model(output)
    ck = tensorfs.open(first.checkpoint, cache_dir=store, local_files_only=True)
    assert ck.components["model"].cls == "" and ck.components["model"].config == {}
    data = bytearray(4)
    ck.components["model"].tensors.read_into("table", data, role="")
    assert data == b"data"
    with ctx.output("model", derivation) as replay:
        assert replay.has("model", "table")
        second = ctx.adopt_model(replay)
        assert replay.written == 0
    assert second == first
    assert len(shown) == 2 and shown[0] == shown[1]
    with ctx.output("model", weights_store.Derivation(ck)) as inherited:
        with pytest.raises(CapabilityError) as refused:
            inherited.put("missing", "table", "U8", [4], b"data")
        assert refused.value.code == "weights_output_component"


def test_a_closed_uncommitted_writer_resumes_its_saved_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    base = tensorfs.open(
        checkpoint(store, {"unet": {"w": np.ones(2, np.float32)}}),
        cache_dir=store, local_files_only=True,
    )  # fmt: skip
    with context(store, "resume").output("model", weights_store.Derivation(base)) as output:
        output.put("unet", "first", "U8", [3], b"one")
    shown: Shown = []
    ctx = context(store, "resume", shown=shown)
    with ctx.output("model", weights_store.Derivation(base)) as resumed:
        assert resumed.has("unet", "first")
        resumed.put("unet", "second", "U8", [3], b"two")
        artifact = ctx.adopt_model(resumed)
    ck = tensorfs.open(artifact.checkpoint, cache_dir=store, local_files_only=True)
    tensors = ck.components["unet"].tensors
    first, second = tensors.read("first"), tensors.read("second")
    assert isinstance(first, tensorfs.Array) and isinstance(second, tensorfs.Array)
    assert first.tobytes() == b"one"
    assert second.tobytes() == b"two"


def test_a_raw_read_fills_only_the_requested_tensor_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    ck = tensorfs.open(
        checkpoint(store, {"unet": {"w": np.arange(8, dtype=np.uint8)}}),
        cache_dir=store, local_files_only=True,
    )  # fmt: skip
    into = bytearray(4)
    weights_store.read_into(ck, "unet", "w", "", 3, into)
    assert into == bytes([3, 4, 5, 6])
    with pytest.raises(CapabilityError) as past:
        weights_store.read_into(ck, "unet", "w", "", 5, into)
    assert past.value.code == "weights_reader_bounds"
    with pytest.raises(CapabilityError) as readonly:
        weights_store.read_into(ck, "unet", "w", "", 0, bytes(4))
    assert readonly.value.code == "weights_reader_buffer"


@pytest.mark.parametrize("previous_release", [None, "1", "2"])
def test_local_destinations_replace_the_alias_and_preserve_other_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous_release: str | None
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    endpoint = tensorfs.Endpoint("local", cache_dir=store)
    source = checkpoint(store, {"model": {"w": np.array([9], np.uint8)}})
    prior = ""
    with tensorfs.open(source, cache_dir=store, local_files_only=True) as seed:
        seed_folder = tmp_path / "seed-portable"
        tensorfs.materialize(seed, local_dir=seed_folder, format="cozytensors")
        if previous_release is not None:
            prior = tensorfs.upload_folder(
                repo_id="local/output", revision=previous_release,
                folder_path=seed_folder, cache_dir=store,
            )
        draft = tensorfs.upload_folder(
            repo_id="local/output", folder_path=seed_folder, cache_dir=store,
        )
        tensorfs.upload_folder(
            repo_id="local/unrelated", revision="7", folder_path=seed_folder, cache_dir=store,
        )
    unrelated = endpoint.repo("local/unrelated")
    shown: Shown = []

    def adopt(request: str, value: bytes) -> tuple[str, str]:
        outputs = weights_store.Outputs(
            store, request_id=request, actor="tester", code=request,
            bounds={"model": 4}, show=lambda *item: shown.append(item),
            destination="local/output",
        )
        with outputs.open("model", weights_store.Derivation(None)) as output:
            output.put("model", "w", "U8", [len(value)], value)
            artifact = outputs.adopt(output)
        assert shown[-1][1] == "local/output"
        with tensorfs.open("local/output", cache_dir=store) as named:
            assert named.id == artifact.checkpoint
            array = named.components["model"].tensors.read("w")
            assert isinstance(array, tensorfs.Array) and array.tobytes() == value
        return artifact.checkpoint, shown[-1][2]

    first, first_tree = adopt("first", b"one")
    second, second_tree = adopt("second", b"two")
    assert first != second and first_tree != second_tree
    with tensorfs.open(first_tree, cache_dir=store, local_files_only=True) as immutable:
        assert immutable.id == first
        array = immutable.components["model"].tensors.read("w")
        assert isinstance(array, tensorfs.Array) and array.tobytes() == b"one"
    if previous_release == "2":
        with tensorfs.open("local/output", revision="2", cache_dir=store) as old:
            assert old.tree == prior and old.id == source
    result = endpoint.repo("local/output")
    assert result is not None
    repo, _etag = result
    releases = repo["releases"]
    assert isinstance(releases, list) and releases[-1]["version"] == "1"
    assert releases[-1].get("revision", 0) == (2 if previous_release == "1" else 1)
    with tensorfs.open(draft, cache_dir=store, local_files_only=True) as draft_checkpoint:
        assert draft_checkpoint.tree_ref is not None
        draft_ref = draft_checkpoint.tree_ref
    assert repo["checkpoints"] == [{"sha256": draft_ref[0], "length": draft_ref[1]}]
    assert endpoint.repo("local/unrelated") == unrelated


def test_a_remote_destination_still_adds_an_unreleased_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = home(tmp_path / "tensorfs", monkeypatch)
    origin = (tmp_path / "origin").as_uri()
    shown: Shown = []
    outputs = weights_store.Outputs(
        store, request_id="remote", actor="tester", code="remote", bounds={"model": 4},
        show=lambda *item: shown.append(item), destination="alice/output",
        connection=weights_store.Connection(endpoint=origin),
    )
    with outputs.open("model", weights_store.Derivation(None)) as output:
        output.put("model", "w", "U8", [4], b"data")
        artifact = outputs.adopt(output)
    assert shown[-1][1] == "alice/output"
    result = tensorfs.Endpoint(origin).repo("alice/output")
    assert result is not None
    repo, _etag = result
    assert repo["releases"] == []
    checkpoints = repo["checkpoints"]
    assert isinstance(checkpoints, list) and len(checkpoints) == 1
    with tensorfs.open(shown[-1][2], cache_dir=store, local_files_only=True) as uploaded:
        assert uploaded.id == artifact.checkpoint
