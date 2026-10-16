"""Runtime on an Apple GPU (MPS, the unified memory class), on a real Mac: the accelerator
boundary, and models served through the executor as TensorD drives one (Start, Load, Activate,
then PrepareRequest and Invoke) with weights fetched by TensorFS into a real store and read
from the machine's object files. `test_sdxl_on_the_apple_gpu` is the SDXL spike: the published
`fidika/sdxl` package and weights from tensorhub.com, one 1024px image.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("an Apple GPU", allow_module_level=True)

import tensorfs  # noqa: E402
from tensorfs import Store, _ext  # noqa: E402
from tensorfs.fetch import Fetch  # noqa: E402

from cozy_runtime.author._errors import is_device_oom  # noqa: E402
from cozy_runtime.author._executor_requests import SealedPlan  # noqa: E402
from cozy_runtime.internal import accel, canonical, executor, package_interface, plane  # noqa: E402
from cozy_runtime.internal.discovery import discover, discover_installed  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.executor_commands import (  # noqa: E402
    Activate,
    Invoke,
    Load,
    PrepareRequest,
    Start,
)
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel  # noqa: E402
from fetch_hub import COMPONENTS, Hub, write  # noqa: E402

pytestmark = pytest.mark.real_gpu
OUT = Path(os.environ.get("RUNNER_TEMP", "/tmp"), "mps-results")


def test_the_accelerator_boundary_on_the_apple_gpu() -> None:
    accel.initialize(torch, "mps")
    assert accel.present(torch, "mps") and accel.memory_class("mps") == "unified"
    memory = accel.allocation(torch, "mps")
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert 0 < memory["driver_total_bytes"] <= min(torch.mps.recommended_max_memory(), ram)
    held = torch.ones(64 << 20, dtype=torch.uint8, device="mps")
    accel.synchronize(torch, "mps")
    assert accel.allocated(torch, "mps") >= held.numel()
    assert accel.completion_event(torch, "mps").query() in (True, False)
    del held
    accel.release_cached(torch, "mps")
    identity = accel.device_identity(torch, "mps")
    assert identity["name"].startswith("Apple") and identity["driver_api"].startswith("macOS")
    # MPS refuses past its pool limit and past Metal's largest buffer with plain RuntimeErrors;
    # both are a device out-of-memory to Runtime's recovery.
    refused: list[str] = []
    for nbytes in (1 << 36, int(torch.mps.recommended_max_memory() * 1.9)):
        with pytest.raises(RuntimeError) as raised:
            torch.empty(nbytes, dtype=torch.uint8, device="mps")
        assert is_device_oom(raised.value), raised.value
        refused.append(f"{type(raised.value).__name__}: {str(raised.value)[:120]}")
    _record("oom", {"refusals": refused})


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

    def forward(self, x):
        return x + torch.tanh(self.out(torch.tanh(self.proj(x * self.norm))))


class Small(torch.nn.Module):
    _no_split_modules = ("Block",)

    def __init__(self, blocks):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))
        self.blocks = torch.nn.ModuleList([Block() for _ in range(blocks)])

    def forward(self, x):
        x = x * self.embed
        for block in self.blocks:
            x = block(x)
        return x


class Graph:
    def __init__(self, config):
        self.components = {"denoiser": Small(4)}


class Pipe(Model[Graph]):
    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=Graph)

    @uses_components("denoiser")
    def denoise(self, x):
        net = self.graph.components["denoiser"]
        return net(x.to(net.embed.device)).float().cpu(), net.embed.device.type


class Ask(msgspec.Struct):
    seed: int


class Answer(msgspec.Struct):
    rows: list[float]
    device: str


app = App()


@app.entrypoint()
def run(payload: Ask, model: Pipe) -> Answer:
    generator = torch.Generator().manual_seed(payload.seed)
    x = torch.randn(4, HIDDEN, generator=generator).half()
    y, device = model.denoise(x)
    return Answer(y.sum(dim=1).tolist(), device)
"""


def test_a_model_serves_on_the_apple_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "apple_fixture.py").write_text(SOURCE)
    (project / "package.toml").write_text('[application]\nobject = "apple_fixture:app"\n')
    monkeypatch.syspath_prepend(str(project))
    producer, manifest = write(tmp_path / "producer", {"denoiser": COMPONENTS["denoiser"]})
    hub = Hub(producer, manifest)
    store = Store.init(tmp_path / "store")
    try:
        with served(tmp_path, discover(project), store, hub.grant()) as machine:
            answer = machine.run(manifest, "Pipe", ["denoiser"], {"seed": 7}, "run")
    finally:
        hub.close()
        sys.modules.pop("apple_fixture", None)
    assert answer["device"] == "mps"
    weights = {k: torch.from_numpy(v).float() for k, v in COMPONENTS["denoiser"].items()}
    x = torch.randn(4, 256, generator=torch.Generator().manual_seed(7)).half().float()
    x = x * weights["embed"]
    for i in range(4):
        h = torch.tanh((x * weights[f"blocks.{i}.norm"]) @ weights[f"blocks.{i}.proj.weight"].t())
        x = x + torch.tanh(h @ weights[f"blocks.{i}.out.weight"].t())
    assert answer["rows"] == pytest.approx(x.sum(dim=1).tolist(), rel=2e-2, abs=0.5)
    warned = " ".join(json.dumps(frame) for frame in machine.frames)
    for feature, _ in accel.unavailable("mps"):
        assert f"{feature} unavailable on mps" in warned


SDXL = "fidika/sdxl"


@pytest.mark.parametrize("size", [1024])
def test_sdxl_on_the_apple_gpu(tmp_path: Path, size: int) -> None:
    """The spike: SDXL at 16 bits (its f16 lane), one image, timed, kept for a look."""
    closure = _post(
        "https://tensorhub.com/v1/tensorfs/closure",
        {"ref": f"{SDXL}@1.0.0", "lane": "bf16", "limit": 1},
    )
    manifest = "sha256:" + closure["manifest"]["sha256"]
    origins = [
        {"hub": "https://tensorhub.com", "ref": f"{SDXL}@1.0.0", "lane": closure["lane"]}
        | {"manifest": manifest, "manifest_length": closure["manifest"]["length"]}
    ]
    store = Store.init(tmp_path / "store")
    with served(tmp_path, discover_installed("sdxl:app"), store, origins) as machine:
        started = time.perf_counter()
        first = machine.run(
            manifest,
            "SdxlModel",
            ["text_encoder", "text_encoder_2", "unet", "vae"],
            {"prompt": "a red fox in fresh snow at dawn, photograph", "seed": 1801},
            "generate",
        )
        cold = time.perf_counter() - started
        started = time.perf_counter()
        second = machine.attempt(
            "generate", {"prompt": "a lighthouse on a cliff in a storm, oil painting", "seed": 77}
        )
        warm = time.perf_counter() - started
    assert (first["width"], first["height"]) == (size, size)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, answer in (("fox", first), ("lighthouse", second)):
        (OUT / f"sdxl-{name}.webp").write_bytes(Path(answer["_image_path"]).read_bytes())
    _record(
        "sdxl",
        {
            "cold_s": round(cold, 1),
            "warm_s": round(warm, 1),
            "steps": first["steps"],
            "chip": accel.apple_chip(),
            "budget_bytes": accel.unified_budget(torch),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
            "ram_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
            "digests": [first["digest"], second["digest"]],
            "plane": machine.plane,
        },
    )


# ---------------------------------------------------------------------------- the machine


class Machine:
    """The machine's half of an Apple executor's seam, as TensorD serves one on unified
    memory: model files, fetch channels into its store, and the verified object files behind
    a plan. It keeps no host layouts: the page cache is the host tier."""

    def __init__(self, root: Path, found: Any, store: Store, origins: list[dict[str, Any]]):
        self.root, self.store, self.origins = root, store, origins
        self.application = found.application
        self.interface = root / "package-interface.json"
        body = package_interface.build(found, installed=found.project == Path("/"))
        self.interface.write_bytes(package_interface.canonical_bytes(body))
        self.frames: list[dict[str, Any]] = []
        self.plane: dict[str, Any] = {}
        ours, self.theirs = socket.socketpair()
        self.thread = threading.Thread(target=self.serve, args=(Channel(self.theirs),), daemon=True)
        self.thread.start()
        (root / "executor").mkdir()
        self.executor = Executor(Channel(ours), root / "executor")
        self.ours = ours

    def serve(self, channel: Channel) -> None:
        while (frame := channel.recv()) is not None:
            if frame.get("event") != "request":
                self.frames.append(frame)
                continue
            answer = {"event": "answer", "seq": frame["seq"]}
            kind = frame["kind"]
            if kind == "fetch_channel":
                channel.send(answer | {"ok": True, "descriptor": True})
                with socket.socket(fileno=self._channel()) as handed:
                    channel.send_descriptor(handed)
                continue
            if kind == "model_source":
                body = self.source(frame["manifest"], frame.get("name", ""))
                fd = plane.sealed_memfd("model-source", body)
                channel.send(
                    answer
                    | {"ok": True, "held": True, "descriptor": True, "length": len(body)}
                    | {"sha256": hashlib.sha256(body).hexdigest()}
                )
                channel.send_memfd(fd)
                os.close(fd)
                continue
            memfd = channel.recv_memfd() if frame.get("descriptor") else -1
            if kind == "object_files":
                body = os.pread(memfd, frame["length"], 0)
                os.close(memfd)
                files = self.object_files(msgspec.convert(canonical.parse(body), SealedPlan))
                listed = "".join(f"{sha}:{n}\n" for sha, n, _ in files).encode()
                channel.send(
                    answer
                    | {"ok": True, "objects_sha256": hashlib.sha256(listed).hexdigest()}
                    | {"descriptors": len(files)}
                )
                for _, _, fd in files:
                    channel._send_fd(fd, "object file")
                    os.close(fd)
                continue
            if memfd >= 0:
                os.close(memfd)  # a budget cell: this machine keeps none
            channel.send(answer | {"ok": False, "code": "unknown_durable_request"})

    def source(self, manifest: str, name: str) -> bytes:
        header = self.store.manifest(manifest)["header"]
        if header is None:  # its structure lands first, as TensorD's prepare does
            with Fetch.from_fd(self._channel()) as fetch:
                fetch.wait(manifest, [])
            header = self.store.manifest(manifest)["header"]
        assert header is not None
        if not name:
            return bytes(header)
        with self.store.acquire_cozytensors(manifest) as lease:
            return bytes(lease.read_asset(header, name, max_bytes=64 << 20))

    def _channel(self) -> int:
        owner, executor_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        threading.Thread(
            target=_ext.serve_fetch,
            args=(self.store, owner.detach()),
            kwargs={"origins": self.origins, "run": 1},
            daemon=True,
        ).start()
        return executor_end.detach()

    def object_files(self, doc: SealedPlan) -> list[tuple[str, int, int]]:
        header = self.store.manifest(doc.manifest)["header"]
        assert header is not None
        read = tensorfs.plan(header, list(doc.traversal), list(doc.components), window=doc.window)
        if doc.parts:
            read = read.select(list(doc.parts))
        objects = sorted(
            {
                i["object"].removeprefix("sha256:"): i["object_length"]
                for i in read.items()
                if i["source"] == "object"
            }.items()
        )
        root = Path(self.store.root, "blobs")
        return [
            (sha, n, os.open(root / sha[:2] / sha[2:4] / sha, os.O_RDONLY | os.O_CLOEXEC))
            for sha, n in objects
        ]

    def run(
        self,
        manifest: str,
        model_class: str,
        components: list[str],
        payload: dict[str, Any],
        entrypoint: str,
    ) -> dict[str, Any]:
        """Start, Load and Activate one construction, then one attempt."""
        path = f"{entrypoint}.models.model"
        start = {
            "application": self.application,
            "package_interface": str(self.interface),
            "devices": "0",
            "constructions": [{"path": path, "manifest": manifest, "components": components}],
        }
        started = self.executor.start(msgspec.convert(start, Start))
        assert started["ok"], started
        binding = {
            "application": self.application,
            "package_interface": str(self.interface),
            "model_class": model_class,
            "model_binding_path": path,
            "model_parameter_name": "model",
            "component": components[0],
            "components": components,
            "snapshot": manifest,
            "snapshots": dict.fromkeys(components, manifest),
            "release": f"{self.application}@1",
        }
        load = {
            "construction": "c",
            "devices": "0",
            "authorized_device_limit_bytes": accel.unified_budget(torch),
            "models": [{"binding": binding}],
            "model_sources": True,
        }
        loaded = self.executor.load(msgspec.convert(load, Load))
        assert loaded["ok"], loaded
        assert self.executor.activate(Activate(construction="c"))["ok"]
        return self.attempt(entrypoint, payload)

    def attempt(self, entrypoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:12]
        prepared = self.executor.serve_request(
            PrepareRequest(request_id=request, entrypoint=entrypoint, payload=payload)
        )
        assert prepared["ok"], prepared
        spool = self.executor.root / request
        spool.mkdir()
        reply = self.executor.serve_invoke(
            Invoke(request_id=request, entrypoint=entrypoint, spool=str(spool), deadline_s=None)
        )
        assert reply["outcome"]["terminal"] == "succeeded", reply
        self.plane = self.executor._plane_facts() or {}
        result = dict(json.loads((spool / RESULT_DOCUMENT).read_bytes()))
        images = sorted(spool.rglob("*.webp"))
        if images:
            result["_image_path"] = str(images[0])
        return result

    def close(self) -> None:
        self.ours.close()
        self.thread.join(timeout=30)
        self.theirs.close()


@pytest.fixture(autouse=True)
def _apple_seal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lane's one device, as the machine seals an Apple executor."""
    monkeypatch.setitem(executor._SEALED, "CUDA_VISIBLE_DEVICES", "0")


@contextlib.contextmanager
def served(
    root: Path, found: Any, store: Store, origins: list[dict[str, Any]]
) -> Iterator[Machine]:
    machine = Machine(root, found, store, origins)
    try:
        yield machine
    finally:
        machine.close()


def _post(url: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(url, json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _record(name: str, facts: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(facts, indent=2, default=str))
    print(f"[mps] {name}: {json.dumps(facts, default=str)}")
