"""Real ordinary invocation capture: reader compatibility, RNG, hooks, and refusals."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import (  # noqa: E402
    ActivationCapture,
    App,
    Context,
    Invocation,
    Loader,
    Model,
    Telemetry,
    attempt,
    uses_components,
)
from cozy_runtime.author._errors import InvalidRequest  # noqa: E402
from cozy_runtime.probe._capture import CaptureSession, validate_components  # noqa: E402
from durable_seam import wire  # noqa: E402


class Denoiser(torch.nn.Module):  # type: ignore[name-defined,misc]
    def __init__(self) -> None:
        super().__init__()
        self.second = torch.nn.Linear(4, 4)
        self.first = torch.nn.Linear(4, 4)

    def forward(self, value: Any) -> Any:
        # Firing order intentionally differs from module declaration order.
        return self.second(torch.relu(self.first(value)))


class Tiny(Model[object]):
    denoiser: Any

    def load(self, loader: Loader) -> None:
        self.denoiser = loader.construct(Denoiser, factory=lambda config: Denoiser())

    @uses_components("denoiser")
    def generate(self, tel: Telemetry, fail: bool = False) -> float:
        on_step = tel.step_callback(3, stage="denoise")
        value = torch.randn(1, 4)
        for step in range(3):
            value = self.denoiser(value)
            if fail and step == 1:
                raise RuntimeError("interrupted")
            on_step(step)
        return float(value.detach().sum())


class Request(msgspec.Struct):
    fail: bool = False


class Result(msgspec.Struct):
    value: float


app = App()


@app.entrypoint
async def render(ctx: Context, payload: Request, model: Tiny, tel: Telemetry) -> Result:
    return Result(model.generate(tel, payload.fail))


def run(model: Tiny, root: Path, *, fail: bool = False) -> Any:
    root.mkdir()
    return attempt(
        app.get("render"),
        {"fail": fail},
        Invocation(
            request_id="capture-proof",
            spool=root,
            deadline=time.monotonic() + 60,
            models={"model": model},
        ),
    )


def test_capture_observes_the_ordinary_render_without_rng_or_tensor_changes(tmp_path: Path) -> None:
    reader = pytest.importorskip("cozy_eval.capture")
    module = Denoiser()
    model = Tiny.for_test(denoiser=module)
    weights = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    torch.manual_seed(42)
    expected, outcome, _ = run(model, tmp_path / "plain")
    assert outcome.terminal == "succeeded"
    rng_after = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    capture = CaptureSession(ActivationCapture(("denoiser",), (0, 2)), {"denoiser": module})
    with capture:
        actual, outcome, _ = run(model, tmp_path / "captured")
    assert outcome.terminal == "succeeded"
    assert actual.result == expected.result
    assert torch.equal(torch.random.get_rng_state(), rng_after)
    assert all(torch.equal(weights[name], value) for name, value in module.state_dict().items())
    assert not any(member._forward_hooks for member in module.modules())
    written = capture.write(tmp_path / "captured")
    observed = reader.read(written["root"])
    assert observed.identity == written["content_digest"]
    assert observed.manifest.steps == (0, 2)
    assert [tap.path for tap in observed.manifest.taps] == ["denoiser.first", "denoiser.second"]
    assert len(observed.manifest.rows) == 4
    assert observed.sketches.shape == (4, 4096)


def test_failure_removes_hooks_and_a_second_render_is_unaffected(tmp_path: Path) -> None:
    module = Denoiser()
    model = Tiny.for_test(denoiser=module)
    with CaptureSession(ActivationCapture(("denoiser",)), {"denoiser": module}):
        _, outcome, _ = run(model, tmp_path / "failed", fail=True)
    assert outcome.terminal == "failed"
    assert not any(member._forward_hooks for member in module.modules())
    _, outcome, _ = run(model, tmp_path / "retry")
    assert outcome.terminal == "succeeded"


def test_nonfinite_output_is_counted_and_sketch_remains_readable(tmp_path: Path) -> None:
    reader = pytest.importorskip("cozy_eval.capture")
    module = torch.nn.Linear(4, 4)
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
    capture = CaptureSession(ActivationCapture(("denoiser",)), {"denoiser": module})
    with capture:
        module(torch.ones(1, 4))
    observed = reader.read(capture.write(tmp_path)["root"])
    assert observed.manifest.rows[0].nan == 1


def test_request_validation_happens_before_any_component_execution() -> None:
    with pytest.raises(InvalidRequest, match="capture_component_undeclared"):
        validate_components(ActivationCapture(("missing",)), [Tiny])
    for steps in ((1,), (0, 0), (0, -1), (0, 2, 1)):
        with pytest.raises(InvalidRequest, match="capture_steps"):
            ActivationCapture(("denoiser",), steps)
    module = Denoiser()
    with (
        CaptureSession(ActivationCapture(("denoiser",), (0, 3)), {"denoiser": module}) as capture,
        pytest.raises(InvalidRequest, match="capture_steps"),
    ):
        capture.schedule(3)
    assert not any(member._forward_hooks for member in module.modules())


def test_cooperative_cancellation_removes_hooks_and_never_writes_partial_capture(
    tmp_path: Path,
) -> None:
    from cozy_runtime.author import Device

    module = Denoiser()
    model = Tiny.for_test(denoiser=module)
    canceled = False

    def progress(frame: Any) -> None:
        nonlocal canceled
        if getattr(frame, "position", 0) == 1:
            canceled = True

    with CaptureSession(ActivationCapture(("denoiser",)), {"denoiser": module}):
        _, outcome, _ = attempt(
            app.get("render"),
            {},
            Invocation(
                "cancel-proof",
                tmp_path,
                time.monotonic() + 60,
                device=Device(),
                models={"model": model},
                cancel=lambda: canceled,
                progress=progress,
            ),
        )
    assert outcome.terminal == "canceled", outcome
    assert not (tmp_path / "runtime.capture").exists()
    assert not any(member._forward_hooks for member in module.modules())


def test_pending_call_keeps_the_result_type_and_observes_capture_separately(tmp_path: Path) -> None:
    import asyncio
    import json
    from concurrent.futures import ThreadPoolExecutor

    from cozy_runtime import __version__
    from cozy_runtime.author._calls import _Broker, _CallType
    from cozy_runtime.internal.capture_observation import document
    from cozy_runtime.protocol import worker_pb2 as pb

    module = Denoiser()
    model = Tiny.for_test(denoiser=module)
    answer: dict[str, Any] = {}

    def exchange(kind: str, frame: dict[str, Any]) -> dict[str, Any]:
        if kind == "child_call":
            assert frame["capture"] == {"components": ["denoiser"], "steps": [0, 2]}
            assert json.loads(frame["payload"]) == {"fail": False}
            capture = CaptureSession(ActivationCapture(("denoiser",), (0, 2)), {"denoiser": module})

            def child() -> Any:
                with capture:
                    return run(model, tmp_path / "child")

            with ThreadPoolExecutor(max_workers=1) as pool:
                result, outcome, _ = pool.submit(child).result()
            assert outcome.terminal == "succeeded", outcome
            capture_fact = capture.write(tmp_path / "child")
            wire = pb.ExecutionObservation(
                environment=pb.ExecutionEnvironment(runtime_version=__version__, accelerator="CPU"),
                capture=pb.ActivationCaptureResult(
                    output_id="runtime.capture",
                    content_digest=bytes.fromhex(capture_fact["content_digest"][7:]),
                ),
            )
            import hashlib

            from cozy_runtime import canonical_json

            members = [
                {
                    "path": name,
                    "kind": "file",
                    "blob": {
                        "sha256": hashlib.sha256(
                            (Path(capture_fact["root"]) / name).read_bytes()
                        ).hexdigest(),
                        "length": (Path(capture_fact["root"]) / name).stat().st_size,
                    },
                }
                for name in ("capture.json", "sketches.f32")
            ]
            manifest = canonical_json.encode({"entries": members})
            answer.update(
                {
                    "ok": True,
                    "state": "succeeded",
                    "result": msgspec.json.encode(result.result).decode(),
                    "observation": document(wire),
                    "byte_grants": [
                        {
                            "output_id": "runtime.capture",
                            "kind": "tree",
                            "digest": "sha256:" + hashlib.sha256(manifest).hexdigest(),
                            "length": len(manifest),
                            "content_bytes": capture_fact["length"],
                            "local": capture_fact["root"],
                            "media_type": "application/vnd.cozy.tree-manifest",
                        }
                    ],
                }
            )
            return {"ok": True}
        if kind == "child_forget":
            # A settled call releases its runtime handle.
            return {"ok": True}
        assert kind == "child_poll"
        return answer

    broker = _Broker(
        "parent",
        {
            ("tiny", "render"): _CallType(
                "sha256:" + "01" * 32,
                "tiny",
                "render",
                Request,
                Result,
            )
        },
        wire(exchange),
    )
    broker.bind(Context("parent", time.monotonic() + 60))
    pending = broker.reserve("tiny", "render", {"fail": False})
    pending.capture = ActivationCapture(("denoiser",), (0, 2))
    before = pending.observation
    assert before is None

    async def invoke_child() -> Any:
        return await pending

    result = asyncio.run(invoke_child())
    assert isinstance(result, Result)
    assert pending.observation is not None
    assert pending.observation.environment.runtime_version == __version__
    assert pending.observation.environment.accelerator == "CPU"
    assert pending.observation.environment.kernel_symbol == ""
    assert pending.observation.capture is not None
    assert pending.observation.capture.output_id == "runtime.capture"
    assert pending.observation.capture.tree is not None
    assert (pending.observation.capture.tree.path / "capture.json").is_file()
    broker.finish()


def test_job_capture_refuses_before_any_executor_device_or_job_work(tmp_path: Path) -> None:
    import socket

    from cozy_runtime.internal.executor import Executor
    from cozy_runtime.internal.executor_commands import RunJob
    from cozy_runtime.internal.seam import Channel

    left, right = socket.socketpair()
    try:
        executor = Executor(Channel(left), tmp_path)
        response = executor.run_job(
            RunJob(
                request_id="job",
                spool=str(tmp_path / "spool"),
                deadline_s=None,
                job="job",
                payload={},
                application="app",
                package_interface="interface.json",
                capture={"components": ["denoiser"], "steps": [0]},
            )
        )
        assert response["code"] == "capture_mode" and response["terminal"] == "refused"
        assert executor.torch is None and not executor.models
        assert list(tmp_path.iterdir()) == []
    finally:
        left.close()
        right.close()
