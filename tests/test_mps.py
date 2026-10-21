"""Runtime on an Apple GPU (MPS, the unified memory class), on a real Mac: the accelerator
boundary, and models served through the executor as TensorD drives one (Start, Load, Activate,
then PrepareRequest and Invoke) with weights fetched by TensorFS into a real store and read
from the machine's object files. `test_sdxl_on_the_apple_gpu` is the SDXL spike: the published
`fidika/sdxl` package and weights from tensorhub.com, one 1024px image.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "pending: weights register on cozytensors/1 through varena (xs-045 U8) and cr-203 U11b",
    allow_module_level=True,
)

import contextlib
import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("an Apple GPU", allow_module_level=True)

import tensorfs  # noqa: E402
from tensorfs import Store, _ext  # noqa: E402
from tensorfs.fetch import Fetch  # noqa: E402
from tensorfs.varena import cuts  # noqa: E402

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
    storage = torch.ones(1 << 20, dtype=torch.uint8, device="mps").untyped_storage()
    try:
        storage.resize_(0)
        _record("storage_resize", {"ok": storage.nbytes() == 0})
    except Exception as exc:
        _record("storage_resize", {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    _record(
        "budget",
        {"budget": memory["driver_total_bytes"], "available": accel._available(), "ram": ram}
        | {"recommended": int(torch.mps.recommended_max_memory())},
    )
    identity = accel.device_identity(torch, "mps")
    assert identity["name"].startswith("Apple") and identity["driver_api"].startswith("macOS")
    # MPS refuses one buffer larger than Metal allows, and a pool past its limit, with plain
    # RuntimeErrors; both are a device out-of-memory to Runtime's recovery.
    refused: list[str] = []
    with pytest.raises(RuntimeError) as raised:
        torch.empty(1 << 36, dtype=torch.uint8, device="mps")
    refused.append(str(raised.value)[:200])
    torch.mps.set_per_process_memory_fraction(0.05)
    try:
        with pytest.raises(RuntimeError) as pool:
            chunks = [torch.empty(256 << 20, dtype=torch.uint8, device="mps") for _ in range(64)]
            del chunks
    finally:
        accel.hold_pool(torch, -1)
        accel.release_cached(torch, "mps")
    refused.append(f"{type(pool.value).__name__}: {str(pool.value)[:200]}")
    assert is_device_oom(raised.value) and is_device_oom(pool.value), refused
    del pool, raised  # their tracebacks hold the refused leg's frames, and with them its pool
    gc.collect()
    accel.release_cached(torch, "mps")
    _record("oom", {"refusals": refused})


def test_attention_runs_inside_the_catch_layer() -> None:
    """Every op within a `use` runs inside varena's out-of-memory catch (a dispatch mode),
    where SDPA asks for its kernel through an op MPS lacks: it gets MPS's own attention."""
    q, k, v = (torch.randn(2, 10, 4096, 64, device="mps", dtype=torch.float16) for _ in range(3))
    plain = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    cuts.enter()
    try:
        recovered = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    finally:
        cuts.leave()
    assert torch.equal(plain, recovered)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.cpu().float(), k.cpu().float(), v.cpu().float()
    )
    assert (recovered.cpu().float() - reference).norm() / reference.norm() < 2e-3


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


PAGED = """
import msgspec
import torch
from cozy_runtime.author import App, Loader, Model, uses_components

WIDE = 2048


class Stack(torch.nn.Module):
    _no_split_modules = ("Linear",)

    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(WIDE, WIDE, bias=False, dtype=torch.float16) for _ in range(blocks)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = torch.tanh(block(x))
        return x


class Graph:
    def __init__(self, config):
        self.components = {"encoder": Stack(8), "decoder": Stack(1)}


class Pipe(Model[Graph]):
    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=Graph)

    @uses_components("encoder")
    def encode(self, x):
        net = self.graph.components["encoder"]
        return net(x.to(net.blocks[0].weight.device))

    @uses_components("decoder")
    def decode(self, x, scratch):
        y = self.graph.components["decoder"](x)
        if scratch:  # activations the pool has room for only once idle weights leave
            y = y + torch.ones(scratch // 2, dtype=torch.float16, device=y.device)[:1].float()
        torch.mps.synchronize()
        return float(y.float().sum()), torch.mps.current_allocated_memory()


class Ask(msgspec.Struct):
    seed: int
    scratch: int = 0


class Answer(msgspec.Struct):
    value: float
    allocated: int


app = App()


@app.entrypoint()
def run(payload: Ask, model: Pipe) -> Answer:
    x = torch.randn(64, 2048, generator=torch.Generator().manual_seed(payload.seed)).half()
    value, allocated = model.decode(model.encode(x), payload.scratch)
    return Answer(value, allocated)
"""


def test_a_stage_pages_weights_out_and_recovers_from_running_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unified memory pages as a discrete card does: an unmeasured decoder stage plans its own
    weights only, so the encoder's 64 MiB leave the pool while it runs, and the next request
    reads them back from the page cache."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "apple_paged.py").write_text(PAGED)
    (project / "package.toml").write_text('[application]\nobject = "apple_paged:app"\n')
    monkeypatch.syspath_prepend(str(project))
    rng = np.random.default_rng(3)

    def stack(blocks: int) -> dict[str, Any]:
        return {
            f"blocks.{i}.weight": (rng.normal(size=(2048, 2048)) / 45).astype(np.float16)
            for i in range(blocks)
        }

    producer, manifest = write(tmp_path / "producer", {"encoder": stack(8), "decoder": stack(1)})
    encoder, decoder = 8 * 2048 * 2048 * 2, 2048 * 2048 * 2
    hub = Hub(producer, manifest)
    try:
        with served(tmp_path, discover(project), Store.init(tmp_path / "store"), hub.grant()) as m:
            first = m.run(manifest, "Pipe", ["decoder", "encoder"], {"seed": 1}, "run")
            again = m.attempt("run", {"seed": 2})
            # Under an owner's cap MPS refuses past it with a plain RuntimeError: Runtime's
            # recovery gives the idle encoder's weights back and runs the allocation again. The
            # cap holds this process, both components and 24 MiB: not the decoder's 40 MiB.
            assert m.plane is not None
            evictions = m.plane["evictions"]
            gc.collect()
            accel.release_cached(torch, "mps")
            facts = m.executor._plane_facts()
            assert facts is not None
            mapped = facts["committed_bytes"]
            cap = torch.mps.driver_allocated_memory() - mapped + encoder + decoder + (24 << 20)
            try:
                tight = m.attempt("run", {"seed": 3, "scratch": 40 << 20}, cap_bytes=cap)
            finally:
                accel.hold_pool(torch, -1)
    finally:
        hub.close()
        sys.modules.pop("apple_paged", None)
    assert first["allocated"] < encoder, (first, m.plane)
    # The second request read the encoder back from the page cache; with room now measured
    # for each stage's activations, it keeps both components hot.
    assert again["value"] != first["value"]
    assert evictions > 0 and again["allocated"] > encoder
    assert tight["value"] != again["value"] and m.plane["evictions"] > evictions, m.plane


def test_weight_only_int8_against_fp16() -> None:
    """The Mac's 8-bit lane is weight-only int8 only where it beats fp16: SDXL UNet linears at
    a 1024px CFG batch, `_weight_int8pack_mm` against an fp16 GEMM, and the 256-wide rotation
    a `convrot256` lane adds in front of it."""
    hadamard = torch.ones(1, 1)
    while hadamard.shape[0] < 256:
        hadamard = torch.cat(
            [torch.cat([hadamard, hadamard], 1), torch.cat([hadamard, -hadamard], 1)]
        )
    hadamard = (hadamard / 16).half().to("mps")
    shapes = ((8192, 640, 640), (8192, 640, 5120), (2048, 1280, 1280), (2048, 1280, 10240))
    rows = [_int8_row(m, k, n, hadamard) for m, k, n in (*shapes, (2048, 5120, 1280))]
    _record("int8", {"rows": rows})


def _int8_row(m: int, k: int, n: int, hadamard: Any) -> dict[str, Any]:
    x = torch.randn(m, k, device="mps", dtype=torch.float16)
    w = torch.randn(n, k, device="mps", dtype=torch.float16) * 0.02
    scale = w.float().abs().amax(dim=1) / 127
    q = torch.round(w.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    half = scale.half()
    reference = x.float() @ (q.float() * scale[:, None]).t()
    int8 = torch._weight_int8pack_mm(x, q, half)
    rotate = (lambda: (x.view(m, k // 256, 256) @ hadamard).view(m, k)) if k % 256 == 0 else None
    return {
        "m": m,
        "k": k,
        "n": n,
        "fp16_ms": _ms(lambda: x @ w.t()),
        "int8_ms": _ms(lambda: torch._weight_int8pack_mm(x, q, half)),
        "rotate_ms": _ms(rotate) if rotate else None,
        "int8_rel": float((int8.float() - reference).norm() / reference.norm()),
    }


def _ms(fn: Any, reps: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return round((time.perf_counter() - started) * 1000 / reps, 3)


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
        # what the Hub's grant names as its object hosts: its presigned reads are R2's
        | {"allowed_hosts": ["tensorhub.com", ".r2.cloudflarestorage.com"]}
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
        planes = [machine.plane]
        started = time.perf_counter()
        second = machine.attempt(
            "generate", {"prompt": "a lighthouse on a cliff in a storm, oil painting", "seed": 77}
        )
        warm = time.perf_counter() - started
        planes.append(machine.plane)
    assert (first["width"], first["height"]) == (size, size)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sdxl-frames.json").write_text(json.dumps(machine.frames, default=str))
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
            "stages_ms": _stages(machine.frames),
            "planes": [
                {k: p.get(k) for k in ("evictions", "disk_copy_bytes", "disk_gbps", "resident")}
                for p in planes
            ],
            # what the store's objects read at, twice: the disk, then what the page cache kept
            "read_gbps": [_read_gbps(Path(store.root, "blobs")) for _ in range(2)],
            "swap": subprocess.run(
                ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
            ).stdout.strip(),
        },
    )


def _stages(frames: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Each attempt's stage durations, from its telemetry."""
    out: dict[str, dict[str, float]] = {}
    for frame in frames:
        if frame.get("kind") == "stage":
            out.setdefault(frame["request_id"], {})[frame["name"]] = frame["value"]
    return out


def _read_gbps(root: Path) -> float:
    started, total = time.perf_counter(), 0
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb", buffering=0) as handle:
                while chunk := handle.read(8 << 20):
                    total += len(chunk)
    return round(total / (time.perf_counter() - started) / 1e9, 3)


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
        self.errors: list[str] = []
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
            memfd = channel.recv_memfd() if frame.get("descriptor") else -1
            try:
                self.answer(channel, frame, memfd)
            except Exception as exc:  # answered, so the executor never waits on a dead machine
                self.errors.append(f"{frame['kind']}: {type(exc).__name__}: {exc}")
                channel.send(
                    {"event": "answer", "seq": frame["seq"], "ok": False}
                    | {"code": "request_failed", "detail": self.errors[-1][:400]}
                )

    def answer(self, channel: Channel, frame: dict[str, Any], memfd: int) -> None:
        answer = {"event": "answer", "seq": frame["seq"]}
        kind = frame["kind"]
        if kind == "fetch_channel":
            handed = socket.socket(fileno=self._channel())
            channel.send(answer | {"ok": True, "descriptor": True})
            with handed:
                channel.send_descriptor(handed)
        elif kind == "model_source":
            body = self.source(frame["manifest"], frame.get("name", ""))
            fd = plane.sealed_memfd("model-source", body)
            channel.send(
                answer
                | {"ok": True, "held": True, "descriptor": True, "length": len(body)}
                | {"sha256": hashlib.sha256(body).hexdigest()}
            )
            channel.send_memfd(fd)
            os.close(fd)
        elif kind == "object_files":
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
        else:
            if memfd >= 0:
                os.close(memfd)  # a budget cell: this machine keeps none
            channel.send(answer | {"ok": False, "code": "unknown_durable_request"})

    def structure(self, manifest: str) -> None:
        """Land the checkpoint's structure (manifest and header), as TensorD's prepare does."""
        with Fetch.from_fd(self._channel()) as fetch:
            fetch.wait(manifest, [])

    def source(self, manifest: str, name: str) -> bytes:
        header = self.store.manifest(manifest)["header"]
        assert header is not None
        if not name:
            return bytes(header)
        with self.store.acquire_cozytensors(manifest) as lease:
            asset = lease.read_asset(header, name, max_bytes=64 << 20)
        return bytes(asset)

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
        self.structure(manifest)
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
        }
        loaded = self.executor.load(msgspec.convert(load, Load))
        assert loaded["ok"], loaded
        assert self.executor.activate(Activate(construction="c"))["ok"]
        return self.attempt(entrypoint, payload)

    def attempt(self, entrypoint: str, payload: dict[str, Any], **invoke: Any) -> dict[str, Any]:
        request = hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:12]
        prepared = self.executor.serve_request(
            PrepareRequest(request_id=request, entrypoint=entrypoint, payload=payload)
        )
        assert prepared["ok"], prepared
        spool = self.executor.root / request
        spool.mkdir()
        reply = self.executor.serve_invoke(
            msgspec.convert(
                {"request_id": request, "entrypoint": entrypoint, "spool": str(spool)}
                | {"deadline_s": None, **invoke},
                Invoke,
            )
        )
        assert reply["outcome"]["terminal"] == "succeeded", json.dumps(
            {"outcome": reply["outcome"], "logs": self.frames[-12:]}, default=str
        )[:6000]
        self.plane = self.executor._plane_facts() or {}
        result = dict(json.loads((spool / RESULT_DOCUMENT).read_bytes()))
        images = sorted(spool.glob("image-*"))
        if images:
            result["_image_path"] = str(images[0])
        return result

    def close(self) -> None:
        self.ours.close()
        self.thread.join(timeout=30)
        self.theirs.close()
        assert not self.errors, self.errors


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
