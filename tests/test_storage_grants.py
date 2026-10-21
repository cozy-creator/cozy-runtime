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
    assert all(item.tag.startswith("replay-") for item in shown)
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
    assert tuple(ck.components["model"].cls) == ("", "", "") and ck.components["model"].config == {}
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


@pytest.mark.parametrize("destination", ["local/output", "alice/output", "alice/output@main"])
def test_a_destination_commits_each_output_under_its_private_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    """The tag the destination names (`org/name@tag`), else the output's replay tag."""
    store = home(tmp_path / "tensorfs", monkeypatch)
    origin = None if destination.startswith("local/") else (tmp_path / "origin").as_uri()
    repo, _, named = destination.partition("@")
    shown: Shown = []
    trees = []
    for value in (b"one", b"two"):
        outputs = weights_store.Outputs(
            store,
            request_id=value.decode(),
            actor="tester",
            code=value.decode(),
            bounds={"model": 4},
            show=lambda *item: shown.append(item),
            destination=destination,
            connection=weights_store.Connection(endpoint=origin),
        )
        with outputs.open("model", weights_store.Derivation(None)) as output:
            output.put("model", "w", "U8", [len(value)], value)
            artifact = outputs.adopt(output)
        _, repo_id, tree, moved, _ = shown[-1]
        assert repo_id == repo and tree.startswith("sha256:")
        assert moved == (named or weights_store.tag(artifact.transaction))
        trees.append(tree)
        # A tag resolves through its endpoint: what it names now is the endpoint's to say.
        with tensorfs.open(repo, revision=moved, endpoint=origin, cache_dir=store) as tagged:
            assert tagged.id == artifact.checkpoint
        with tensorfs.open(
            repo, revision=tree, endpoint=origin, cache_dir=store, local_files_only=True
        ) as committed:
            assert committed.id == artifact.checkpoint
            array = committed.components["model"].tensors.read("w")
            assert isinstance(array, tensorfs.Array) and array.tobytes() == value
            with weights_store._thin(
                store, committed
            ) as folder:  # the header only; no chunk copies
                files = sorted(str(f.relative_to(folder)) for f in folder.rglob("*") if f.is_file())
                assert len(files) == 1 and files[0].endswith(".cozytensors"), files
        if origin is not None:  # the chunks reached the endpoint from the store
            fresh = tmp_path / f"fresh-{value.decode()}"
            with tensorfs.open(repo, revision=tree, endpoint=origin, cache_dir=fresh) as got:
                array = got.components["model"].tensors.read("w")
                assert isinstance(array, tensorfs.Array) and array.tobytes() == value
    assert trees[0] != trees[1]
    # A new tag is private: no public tag names a default for the destination.
    with pytest.raises(tensorfs.NotFoundError):
        tensorfs.open(repo, endpoint=origin, cache_dir=tmp_path / "default")
