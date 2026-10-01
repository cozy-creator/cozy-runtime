"""On this machine's real GPU (`--real-gpu`, under the GPU lock): an import-only prespawn creates
no CUDA context, even beside another process's context on the card; the grant's adoption creates
it, serves a real model call, and its construction takes the fused glue's build the prespawn
submitted.

`test_startup_warm.py` proves the same flow on cards no driver knows; here the card, the driver's
process table and the context are real. The package environment's guard only records its CUDA
driver calls (`GUARD`, `refuse=False`); nothing on the device path is faked. The machine's boot
compile of the attention ladder is switched off: it is not what this test measures.

Run: `flock <locks>/gpu.lock .venv/bin/pytest tests/test_startup_warm_gpu.py --real-gpu -s`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal import accel, fusion, kernel_cache, kernel_compile, machine_kernels
from cozy_runtime.internal import package_interface as interfaces
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor_commands import Probe
from cozy_runtime.internal.worker.workspace_executions import TERMINAL
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_startup_warm import OWNER, Origin, Warm, _attempts, _call, _install, _wait

pytestmark = [
    pytest.mark.real_gpu,
    pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or ""),
]

TINY = "release-startup-warm-gpu"
#: One model whose one component is the fixture checkpoint's `unet` (a single f32 weight), and a
#: class that consents to the fused glue, so its start submits that build.
TINY_SOURCE = """import msgspec
import torch
from cozy_runtime.author import App, Config, Loader, Model, uses_components


class Request(msgspec.Struct):
    n: int = 1


class Result(msgspec.Struct):
    device: str
    value: float


class Scalar(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.components = {"unet": Scalar()}


class Tiny(Model[Pipeline], fusion="accept"):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=Pipeline)

    @uses_components("unet")
    def measure(self, n: int) -> tuple[str, float]:
        weight = self.pipe.components["unet"].weight
        return str(weight.device), float(weight.sum()) * n


app = App()


@app.entrypoint
def generate(payload: Request, model: Tiny) -> Result:
    device, value = model.measure(payload.n)
    return Result(device=device, value=value)
"""
#: A second process's context on the card, held until its stdin closes.
HOLDER = """import sys, torch
held = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
print("held", flush=True)
sys.stdin.read()
"""
DIGEST = "sha256:" + hashlib.sha256(b"startup-warm-gpu").hexdigest()


def _tree(pid: int) -> set[int]:
    """`pid` and every live descendant: the executor and anything it started (its builders)."""
    found, todo = set(), [pid]
    while todo:
        current = todo.pop()
        found.add(current)
        for task in Path(f"/proc/{current}/task").glob("*/children"):
            try:
                todo += [int(child) for child in task.read_text().split()]
            except OSError:
                continue
    return found


def _on_card(pids: set[int]) -> set[int]:
    """The pids the driver lists with a context on a card, by NVML and by nvidia-smi."""
    listed = {
        pid for pid, row in accel.process_memories(pids, "cuda").items() if row.state == "present"
    }
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return listed | {int(row) for row in smi.stdout.split() if row.isdigit() and int(row) in pids}


def _package(root: Path) -> tuple[Path, bytes]:
    project = root / "tiny-project"
    (project / "tiny_warm_gpu").mkdir(parents=True)
    (project / "tiny_warm_gpu" / "__init__.py").write_text(TINY_SOURCE)
    (project / "package.toml").write_text('[application]\nobject = "tiny_warm_gpu:app"\n')
    return project, interfaces.canonical_bytes(interfaces.build(discover(project)))


def _serve(machine: Warm, name: str, interface: bytes, checkpoint: dict[str, Any]) -> None:
    """A direct serving root of the tiny package with its payload, as Creator submits one."""
    placement_id = "tmpl-" + TINY
    placement: dict[str, Any] = {
        "placement_id": placement_id,
        "installation_id": TINY,
        "package_interface": base64.b64encode(interface).decode(),
        "bindings_digest": DIGEST,
        "package": {"package": "t/tiny", "release": "1"},
        "models": [{"id": "model", "repo": "proof/model", "manifest": checkpoint}],
        "entrypoints": [
            {
                "name": "generate",
                "entrypoint_binding_digest": DIGEST,
                "slots": [
                    {
                        "slot": "model",
                        "reference_model_id": "model",
                        "components": [{"component": "unet", "model_id": "model"}],
                    }
                ],
            }
        ],
    }
    raw_set = canonical_json.encode(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [placement]}
    )
    state = pb.DesiredWorkerState(
        wire_minor=WIRE_MINOR,
        posture=pb.POSTURE_ACCEPTING,
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=hashlib.sha256(raw_set).digest(),
            placement_set_canonical_bytes=raw_set,
        ),
    )
    payload = canonical_json.encode({"n": 3})
    payload_digest = documents.spell(documents.digest_of(payload))
    spec, spec_digest = documents.identity(
        pb.InvocationSpec(
            installation_id=TINY,
            payload_digest=payload_digest,
            inputs=[
                pb.InputBinding(
                    input_id="payload",
                    digest=payload_digest,
                    length=len(payload),
                    kind_mime="application/json",
                )
            ],
            serving=pb.ServingInvocationSpec(
                entrypoint_binding_digest=DIGEST,
                attempt_binding_id=DIGEST,
                bindings_digest=DIGEST,
            ),
        )
    )
    offer = pb.AttemptOffer(
        request_id=name,
        attempt_ordinal=1,
        placement_id=placement_id,
        invocation_spec_canonical_bytes=spec,
        invocation_spec_digest=spec_digest,
        grant=pb.DeliveryGrant(
            invocation_spec_digest=spec_digest,
            inputs=[
                pb.InputAccess(
                    input_id="payload",
                    url="data:application/json;base64," + base64.b64encode(payload).decode(),
                )
            ],
        ),
    )
    preparation = {
        "installations": {},
        "state": base64.b64encode(state.SerializeToString()).decode(),
    }
    machine.executions.submit(
        OWNER,
        name,
        hashlib.sha256(name.encode()).digest(),
        offer,
        expected_execution_workspace_id=machine.executions.workspace_id,
        worker_boot=machine.worker.fence.worker_boot_id,
        preparation=canonical_json.encode(preparation),
        worker_id=machine.worker.options.worker_id,
    )
    machine.worker.execution(OWNER, name)


def test_a_real_card_holds_no_context_before_the_grant_and_serves_after_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    arch = accel.device_capability(card)
    assert arch, f"NVML does not know card {card!r}"
    monkeypatch.setattr(machine_kernels, "boot", lambda *args, **kwargs: None)
    seen: dict[str, Any] = {"card": card, "arch": arch}
    holder: subprocess.Popen[str] | None = None
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-gpu-warm.", dir="/tmp") as raw:
        root = Path(raw)
        kernels = root / "kernels"
        machine = Warm(root, card, kernel_cache=kernels)
        origin = Origin(Path(tempfile.mkdtemp(prefix="cz-origin.", dir="/tmp")))
        try:
            project, interface = _package(root)
            install = machine.worker.options.install_root or root
            _install(install, TINY, refuse=False, package=project)

            # The checkpoint lands in the store first (the Host's preparation).
            machine.captured_root("A", origin)
            origin.release.set()
            call, request = _call("A")
            serving = machine.worker.machine_calls.serving  # type: ignore[union-attr]
            landing = serving._shared(
                OWNER, call, machine.target(origin.checkpoint), request, {}, speculative=False
            )
            assert landing is not None and landing.result(300)
            landed = list(machine.worker.prespawns.slots.values())
            for slot in landed:
                machine.worker.prespawns.forget(slot.installation)
            _wait(lambda: all(s.supervision._closed for s in landed), "the landing's prespawn gone")

            # A holds the card; another process holds a context on it.
            a1 = machine.child("A", "qwen")
            machine.tick()
            assert machine.granted(a1) == [0]
            holder = subprocess.Popen(
                [sys.executable, "-c", HOLDER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
            assert _on_card({holder.pid}) == {holder.pid}, "the driver's table misses a context"

            # D waits; its executor spawns and imports on the card's seal with no context.
            _serve(machine, "D", interface, origin.checkpoint)
            slot = _wait(lambda: machine.worker.prespawns.slots.get((TINY, (0,))), "D's prespawn")
            assert slot.started.wait(300) and slot.ready, slot.frames
            executor = slot.executor
            assert executor is not None and executor.alive() and not executor.started
            assert executor.call(Probe())["torch"] is False
            tree = _tree(executor.pid)
            seen["import_only_on_card"] = sorted(_on_card(tree))
            seen["import_only_attempts"] = _attempts(machine, TINY)
            prepared = _wait(
                lambda: machine.phases("D").get("Preparing model executor"), "D's prespawn phase"
            )
            seen["prespawn_detail"] = prepared["detail"]
            assert seen["import_only_on_card"] == [], seen
            assert seen["import_only_attempts"] == [], seen
            assert f"fusion sm{arch}" in prepared["detail"], prepared
            assert _on_card({holder.pid}) == {holder.pid}
            print(json.dumps({"stage": "import-only", **seen}), flush=True)
            holder.stdin.close()  # type: ignore[union-attr]
            holder.wait(60)

            # The fused glue's build the start submitted, keyed as the executor keys it.
            own = kernel_cache.namespace(kernels, os.geteuid())
            monkeypatch.setenv("TRITON_CACHE_DIR", str(own / "triton"))
            try:
                store, found = kernel_cache.Store(own), fusion.job(arch)
            except fusion.FusionUnavailable as exc:
                # This environment has no usable Triton: the start reported it; adoption is
                # still what this test proves.
                seen["fusion_before_grant"] = f"unavailable: {exc.code}"
            else:
                deadline = time.monotonic() + 240  # the slot's patience, not a product rule
                while kernel_compile.status(store, found).state not in ("ready", "failed"):
                    if time.monotonic() > deadline:
                        break
                    time.sleep(1)
                seen["fusion_before_grant"] = kernel_compile.status(store, found).line()

            # The grant adopts that process: its device start makes the context, and it serves.
            machine.finish(a1)
            machine.finish("A")
            _wait(lambda: machine.executions.status(OWNER, "D").state in TERMINAL, "D's outcome")
            outcome = machine.executions.collect(OWNER, "D")
            body = documents.parse(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            assert body.status == pb.OUTCOME_STATUS_SUCCEEDED, body
            result = json.loads(body.result.inline_result)
            seen["result"] = result
            seen["adopted_on_card"] = sorted(_on_card({executor.pid}))
            seen["attempts_after"] = _attempts(machine, TINY)[:3]
            facts = [row.facts.get("execution_fusion") for row in executor.loaded.values()]
            seen["execution_fusion"] = facts
            print(json.dumps({"stage": "served", **seen}), flush=True)
            assert result["device"].startswith("cuda"), result
            assert slot.supervision.spawns == 1, "the grant spawned another process"
            assert seen["adopted_on_card"] == [executor.pid], seen
            assert _attempts(machine, TINY)[0] == (executor.pid, "_cuda_init"), seen
            # The construction consented to the fused glue; a ready build loads, and the tiny
            # model then has no transformer to fuse. A build still compiling is reported.
            (fused,) = facts
            assert fused is not None, facts
            if seen["fusion_before_grant"].startswith("ready"):
                assert fused["code"] == "fusion_no_transformer", fused
        finally:
            if holder is not None and holder.poll() is None:
                holder.kill()
                holder.wait(30)
            origin.close()
            machine.worker.shutdown()
