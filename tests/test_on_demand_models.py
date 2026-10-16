"""An on-demand model (`on_demand=`) is built, and its weights fetched, only when a request
reads it; every other model keeps its early, overlapped fetch (one-request-kind §2.6).

Real: a loopback hub counting each object's GETs, a Store holding only the checkpoint's
structure (as TensorD's prepare leaves it), TensorFS's own fetch-channel owner, the machine's
sealed host layouts filled from objects it waits for (as TensorD's readers do), and one
executor driven Start, Load, Activate, then PrepareRequest and Invoke per attempt."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

import tensorfs  # noqa: E402
from tensorfs import Store  # noqa: E402
from tensorfs import plane as tfs_plane  # noqa: E402
from tensorfs.fetch import Fetch  # noqa: E402

from cozy_runtime.author._errors import ConformanceError  # noqa: E402
from cozy_runtime.author._executor_requests import SealedPlan  # noqa: E402
from cozy_runtime.internal import (  # noqa: E402
    canonical,
    package_interface,
    plane,
    static_interface,
)
from cozy_runtime.internal.discovery import discover  # noqa: E402
from cozy_runtime.internal.executor import _SEALED, Executor  # noqa: E402
from cozy_runtime.internal.executor_commands import (  # noqa: E402
    Activate,
    Invoke,
    Load,
    PrepareRequest,
    Start,
)
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel  # noqa: E402
from fetch_hub import COMPONENTS, Hub, objects_of, served  # noqa: E402

MODULE = "on_demand_fixture"
SOURCE = """
import msgspec
import torch
from cozy_runtime.author import App, Loader, Model, uses_components

HIDDEN = 256


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.out = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.norm = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))


class Small(torch.nn.Module):
    _no_split_modules = ("Block",)

    def __init__(self, blocks):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))
        self.blocks = torch.nn.ModuleList([Block() for _ in range(blocks)])


class PipeGraph:
    def __init__(self, config):
        self.components = {"encoder": Small(1), "decoder": Small(2)}


class SpareGraph:
    def __init__(self, config):
        self.components = {"spare": Small(2)}


class Pipe(Model[PipeGraph]):
    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(PipeGraph, factory=PipeGraph)

    @uses_components("encoder")
    def encode(self) -> float:
        return float(self.graph.components["encoder"].embed.float().sum())

    @uses_components("decoder")
    def decode(self, value: float) -> float:
        return value + float(self.graph.components["decoder"].embed.float().sum())


class Refiner(Model[SpareGraph]):
    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(SpareGraph, factory=SpareGraph)

    @uses_components("spare")
    def refine(self, value: float) -> float:
        return value + float(self.graph.components["spare"].embed.float().sum())


class Ask(msgspec.Struct):
    refine: bool


class Answer(msgspec.Struct):
    value: float


app = App()


@app.entrypoint(on_demand=("refiner",))
def run(payload: Ask, model: Pipe, refiner: Refiner) -> Answer:
    value = model.encode()
    if payload.refine:
        value = refiner.refine(value)
    return Answer(model.decode(value))
"""

PIPE = ("decoder", "encoder")


def project(root: Path, source: str = SOURCE) -> Path:
    path = root / "project"
    path.mkdir()
    (path / f"{MODULE}.py").write_text(source)
    (path / "package.toml").write_text(f'[application]\nobject = "{MODULE}:app"\n')
    return path


def test_on_demand_reaches_the_interface_from_both_readers(tmp_path: Path) -> None:
    path = project(tmp_path)
    try:
        imported = package_interface.build(discover(path))
    finally:
        sys.modules.pop(MODULE, None)
    assert imported == static_interface.build(path)
    slots = {slot["path"]: slot for slot in imported["entrypoints"][0]["models"]}
    assert slots["run.models.refiner"]["on_demand"] is True
    assert "on_demand" not in slots["run.models.model"]
    parsed = package_interface.parse(package_interface.canonical_bytes(imported), "interface")
    assert [slot.on_demand for slot in parsed.entrypoints[0].models] == [False, True]


@pytest.mark.parametrize("named", ["absent", "source"])
def test_on_demand_names_a_constructed_model_argument(tmp_path: Path, named: str) -> None:
    source = SOURCE.replace(
        "def run(payload: Ask, model: Pipe, refiner: Refiner)",
        "def run(payload: Ask, model: Pipe, refiner: Refiner, source: Weights[Pipe])",
    ).replace("from cozy_runtime.author import ", "from cozy_runtime.author import Weights, ")
    path = project(tmp_path, source.replace('on_demand=("refiner",)', f'on_demand=("{named}",)'))
    try:
        with pytest.raises(ConformanceError) as refused:
            discover(path)
    finally:
        sys.modules.pop(MODULE, None)
    assert refused.value.code == "model_on_demand"
    with pytest.raises(ConformanceError, match="on_demand"):
        static_interface.build(path)


def owner(channel: Channel, hub: Hub, store: Store, sealed: list[str]) -> None:
    """The machine's half: fetch channels, the checkpoint's header and assets, and each sealed
    host layout, filled once its objects are here (waited for at Now, as TensorD's readers do)."""
    held: dict[str, int] = {}
    while (frame := channel.recv()) is not None:
        if frame.get("event") != "request":
            continue
        answer = {"event": "answer", "seq": frame["seq"], "ok": True}
        kind = frame["kind"]
        if kind == "fetch_channel":
            channel.send(answer | {"descriptor": True})
            with socket.socket(fileno=hub.channel(store)) as handed:
                channel.send_descriptor(handed)
            continue
        if kind == "model_source":
            header = store.manifest(frame["manifest"])["header"]
            assert header is not None
            body = bytes(header)
            if frame.get("name"):
                with store.acquire_manifest(frame["manifest"]) as lease:
                    body = bytes(lease.read_asset(header, frame["name"], max_bytes=1 << 20))
            memfd = plane.sealed_memfd("model-source", body)
            facts = {"sha256": hashlib.sha256(body).hexdigest(), "length": len(body)}
            channel.send(answer | {"held": True, "descriptor": True} | facts)
            channel.send_memfd(memfd)
            os.close(memfd)
            continue
        memfd = channel.recv_memfd() if frame.get("descriptor") else -1
        if kind != "sealed_tier":  # a budget cell: this machine keeps none
            if memfd >= 0:
                os.close(memfd)
            channel.send(answer | {"ok": False, "code": "unknown_durable_request"})
            continue
        body = os.pread(memfd, frame["length"], 0)
        os.close(memfd)
        doc = msgspec.convert(canonical.parse(body), SealedPlan)
        if doc.name not in held:
            header = store.manifest(doc.manifest)["header"]
            assert header is not None
            read = tensorfs.plan(header, list(doc.traversal), list(doc.components), doc.window)
            refs = sorted(
                {
                    item["object"].removeprefix("sha256:"): item["object_length"]
                    for item in read.items()
                    if item["source"] == "object"
                }.items()
            )
            wire = [{"sha256": sha, "length": length} for sha, length in refs]
            with Fetch.from_fd(hub.channel(store)) as fetch:
                fetch.wait(doc.manifest, wire)
            lease = store.acquire(doc.manifest, [(f"sha256:{sha}", n) for sha, n in refs])
            regions = [list(region) for region in doc.regions]
            held[doc.name], _ = tfs_plane.fill_sealed(store, lease, read, regions, doc.name, 2)
            sealed.append(doc.name)
        granted = os.open(f"/proc/self/fd/{held[doc.name]}", os.O_RDONLY | os.O_CLOEXEC)
        channel.send(answer | {"held": True, "descriptor": True})
        channel.send_memfd(granted)
        os.close(granted)
    for fd in held.values():
        os.close(fd)


def binding(
    interface: Path, manifest: str, parameter: str, cls: str, components: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "application": f"{MODULE}:app",
        "package_interface": str(interface),
        "model_class": cls,
        "model_binding_path": f"run.models.{parameter}",
        "model_parameter_name": parameter,
        "component": components[0],
        "components": list(components),
        "snapshot": manifest,
        "snapshots": dict.fromkeys(components, manifest),
        "release": f"{MODULE}@1",
    }


def attempt(executor: Executor, request_id: str, refine: bool) -> dict[str, Any]:
    """One request through PrepareRequest and Invoke; its result."""
    prepared = executor.serve_request(
        PrepareRequest(request_id=request_id, entrypoint="run", payload={"refine": refine})
    )
    assert prepared["ok"], prepared
    spool = executor.root / request_id
    spool.mkdir()
    reply = executor.serve_invoke(
        Invoke(request_id=request_id, entrypoint="run", spool=str(spool), deadline_s=60)
    )
    assert reply["outcome"]["terminal"] == "succeeded", reply
    return dict(json.loads((spool / RESULT_DOCUMENT).read_bytes()))


class Machine:
    """One executor's seam to `owner`, over the fixture's release and the hub's checkpoint."""

    def __init__(self, root: Path, hub: Hub, store: Store, path: Path) -> None:
        found = discover(path)
        self.application = found.application
        self.interface = root / "package-interface.json"
        self.interface.write_bytes(
            package_interface.canonical_bytes(package_interface.build(found))
        )
        self.hub, self.store = hub, store
        self.sealed: list[str] = []
        self.ours, self.theirs = socket.socketpair()
        self.thread = threading.Thread(
            target=owner, args=(Channel(self.theirs), hub, store, self.sealed), daemon=True
        )
        self.thread.start()
        (root / "root").mkdir()
        self.executor = Executor(Channel(self.ours), root / "root")

    def start(self) -> None:
        rows = [
            {"path": "run.models.model", "manifest": self.hub.manifest, "components": list(PIPE)},
            {"path": "run.models.refiner", "manifest": self.hub.manifest, "components": ["spare"]},
        ]
        start = {
            "application": self.application,
            "package_interface": str(self.interface),
            "devices": _SEALED.get("CUDA_VISIBLE_DEVICES", ""),
            "constructions": rows,
        }
        assert self.executor.start(msgspec.convert(start, Start))["ok"]

    def close(self) -> None:
        self.ours.close()
        self.thread.join(timeout=30)
        self.theirs.close()


@pytest.fixture
def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    path = project(tmp_path)
    monkeypatch.syspath_prepend(str(path))
    with served(tmp_path) as (hub, store):
        held = Machine(tmp_path, hub, store, path)
        try:
            yield held
        finally:
            held.close()
            sys.modules.pop(MODULE, None)


def test_the_start_pass_wants_every_model_but_an_on_demand_one(machine: Machine) -> None:
    machine.start()
    planner = machine.executor.fetch
    assert planner is not None
    store, manifest = machine.store, machine.hub.manifest
    wanted = sorted(ref for _, refs in planner.groups for ref in refs)
    assert wanted == sorted({ref for name in PIPE for ref in objects_of(store, manifest, name)})
    for _, refs in planner.groups:
        planner.wait(manifest, refs)
    spare = objects_of(store, manifest, "spare")
    assert machine.hub.count(spare) == [0] * len(spare)
    planner.end()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a model-bearing load needs a GPU")
def test_an_on_demand_model_is_built_and_fetched_only_by_a_request_that_reads_it(
    machine: Machine,
) -> None:
    hub, store, executor = machine.hub, machine.store, machine.executor
    pipe = sorted({ref for name in PIPE for ref in objects_of(store, hub.manifest, name)})
    spare = objects_of(store, hub.manifest, "spare")
    machine.start()
    models = [
        {"binding": binding(machine.interface, hub.manifest, "model", "Pipe", PIPE)},
        {"binding": binding(machine.interface, hub.manifest, "refiner", "Refiner", ("spare",))},
    ]
    load = {
        "construction": "c",
        "devices": _SEALED.get("CUDA_VISIBLE_DEVICES", ""),
        "authorized_device_limit_bytes": 1 << 30,
        "models": models,
        "model_sources": True,
        "sealed_tiers": True,
    }
    loaded = executor.load(msgspec.convert(load, Load))
    assert loaded["ok"], loaded
    assert set(loaded["facts"]["models"]) == {"model"}, "the refiner is not built"
    assert executor.activate(Activate(construction="c"))["ok"]

    plain = attempt(executor, "plain", refine=False)
    assert hub.count(pipe) == [1] * len(pipe)
    assert hub.count(spare) == [0] * len(spare), "an unused on-demand model fetched"
    assert len(machine.sealed) == len(PIPE), "an unused on-demand model was built"

    refined = attempt(executor, "refined", refine=True)
    assert hub.count(spare) == [1] * len(spare), "its first use fetches it, once"
    assert hub.count(pipe) == [1] * len(pipe)
    embed = torch.from_numpy(COMPONENTS["spare"]["embed"]).float().sum().item()
    assert refined["value"] == pytest.approx(plain["value"] + embed, rel=1e-3)

    # Built once: a later request reuses it and its weights, and fetches nothing again.
    again = attempt(executor, "again", refine=True)
    assert again["value"] == refined["value"]
    assert hub.count(spare) == [1] * len(spare)
    assert len(machine.sealed) == len(PIPE) + 1
