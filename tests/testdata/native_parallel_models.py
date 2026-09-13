"""Full native multi-Model prepare with real CPU/Gloo; host RAM supplies CPU accounting.

Only the hardware accounting boundary is adapted from CUDA to measured host memory.
TensorFS, model loading, plan qualification, group preparation, scopes and forwards run
through their actual implementations. No claim about GPU/NCCL capacity follows.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import tensorfs
import torch

from cozy_runtime.author import WeightsConfig, WeightsPart, WeightsTarget, WeightsTensor
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal import accel, canonical, package_interface, spawn
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.executor import main as executor_main
from cozy_runtime.internal.local import development_source_digest, stage_development_placement
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import PlanChooser, PreparedModel, PreparedRequest
from cozy_runtime.internal.worker.session import executor_prepare_command

PACKAGE = Path(__file__).parent / "dual_model_package"
sys.path.insert(0, str(PACKAGE))
from dual_model_package import (  # type: ignore[import-not-found]  # noqa: E402
    CONFIG,
    BasePipe,
    OverlayPipe,
    batch,
)

from cozy_runtime.author import Config  # noqa: E402


class _CpuEvent:
    def record(self, _stream: Any = None) -> None:
        return None

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class _CpuRing:
    """Synchronous byte transport for this CPU/Gloo fixture only."""

    def __init__(
        self,
        torch_module: Any,
        _device: Any,
        *,
        window_bytes: int,
        slots: int,
        inflight: int,
        **_: Any,
    ) -> None:
        if slots < 2 or inflight >= slots:
            raise ValueError("CPU ring requires slots > inflight")
        self.torch, self.device = torch_module, torch_module.device("cpu")
        self.window_bytes, self.keep = int(window_bytes), int(inflight)
        self.registered_bytes = 0  # CPU slots are ordinary host memory, not pinned CUDA memory.
        self.in_flight_peak = 0
        self.waits = 0
        self.buffers = [
            torch_module.frombuffer(bytearray(self.window_bytes), dtype=torch_module.uint8)
            for _ in range(slots)
        ]
        self.views = [memoryview(buffer.numpy()) for buffer in self.buffers]
        self.inflight: list[tuple[_CpuEvent, Any]] = []
        self.stream = object()

    def track(self, _tensor: Any) -> None:
        return None

    def hold(self, event: _CpuEvent, batch: Any) -> None:
        self.inflight.append((event, batch))
        self.in_flight_peak = max(self.in_flight_peak, len(self.inflight))

    def drain(self, _keep: int) -> None:
        while self.inflight:
            self.inflight.pop(0)[1].release()

    def finish(self) -> None:
        self.drain(0)

    def close(self) -> None:
        self.inflight.clear()
        self.buffers.clear()
        self.views.clear()


def install_cpu_transport() -> None:
    import cozy_runtime.internal.fill as fill_module

    setattr(fill_module, "StagingRing", _CpuRing)  # noqa: B010
    setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())  # noqa: B010
    setattr(torch.cuda, "Event", lambda *args, **kwargs: _CpuEvent())  # noqa: B010
    setattr(torch.cuda, "current_stream", lambda *args, **kwargs: _CpuEvent())  # noqa: B010
    setattr(torch.cuda, "default_stream", lambda *args, **kwargs: _CpuEvent())  # noqa: B010


def cpu_accounting() -> None:
    original_allocation, original_allocated = accel.allocation, accel.allocated
    original_identity = accel.device_identity

    def rss_bytes() -> int:
        return int(Path("/proc/self/statm").read_text().split()[1]) * int(os.sysconf("SC_PAGESIZE"))

    def allocation(torch_module: Any, kind: str, device: Any = None) -> dict[str, int]:
        if kind != "cpu":
            return original_allocation(torch_module, kind, device)
        page_size = int(os.sysconf("SC_PAGESIZE"))
        total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
        rss = rss_bytes()
        return {
            "allocated_bytes": rss,
            "reserved_bytes": rss,
            "driver_free_bytes": int(os.sysconf("SC_AVPHYS_PAGES")) * page_size,
            "driver_total_bytes": total,
        }

    def allocated(torch_module: Any, kind: str, device: Any = None) -> int:
        return rss_bytes() if kind == "cpu" else original_allocated(torch_module, kind, device)

    def identity(torch_module: Any, kind: str, index: int = 0) -> dict[str, Any]:
        if kind != "cpu":
            return original_identity(torch_module, kind, index)
        return {"name": "CPU fixture", "sm": 0, "driver_api": 0, "cuda_runtime": ""}

    accel.allocation, accel.allocated = cast(Any, allocation), cast(Any, allocated)
    accel.device_identity = cast(Any, identity)


def native_model(root: Path, name: str, components: dict[str, Any], config: dict[str, Any]) -> None:
    store = tensorfs.Store.ensure(str(root))
    rows = {
        (component, key): tensor.detach().cpu().contiguous()
        for component, module in components.items()
        for key, tensor in module.state_dict().items()
    }
    max_bytes = sum(t.numel() * t.element_size() for t in rows.values()) + 1_000_000
    host = WeightsTransactionHost(
        store=store,
        owner_scope="native-cp-test",
        request_id=name,
        invocation_spec_digest="sha256:" + "1" * 64,
        work_fingerprint="sha256:" + "2" * 64,
        writer_session_id=1,
        allowed_sources={},
        output_bounds={"model": max_bytes},
    )
    raw_config = canonical.write(config)
    transaction = host.open(
        WeightsCommit(
            output_slot="model",
            sources={},
            targets={
                component: WeightsTarget(
                    add={
                        key: WeightsTensor(
                            "f32",
                            tuple(value.shape),
                            SPEC_PLAIN,
                            {"value": WeightsPart("f32", tuple(value.shape))},
                        )
                        for (owner, key), value in rows.items()
                        if owner == component
                    }
                )
                for component in components
            },
            configs={"model": WeightsConfig(data=raw_config)},
            order=tuple(rows),
            max_new_bytes=max_bytes,
        )
    )
    for (component, key), value in rows.items():
        transaction.add_part(component, key, "value", value.numpy().tobytes())
    transaction.add_config("model", raw_config, len(raw_config))
    receipt = json.loads(transaction.commit().tensorfs_receipt)
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    length = int(receipt["manifest"]["length"])
    operation = store.begin_operation("publish-" + name, "test", name)
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1", "main", manifest, length)


def follower(args: Any) -> None:
    # Run the actual follower command loop. The test only adds a read-only probe
    # of model ownership and maps accelerator accounting to measured host RAM.
    original_probe = Executor.probe

    def probe(self: Executor) -> dict[str, Any]:
        models = {key[0]: model for key, (model, _root) in self.group_components.items()}
        return {
            **original_probe(self),
            "rank": self.rank,
            "active": {key: model._cozy_active for key, model in models.items()},
            "models": sorted(models),
            "filled": [child.facts.get("filled_bytes") for child in self.model_executors],
        }

    Executor.probe = probe  # type: ignore[method-assign]
    accel.host_backend_family = lambda: "cpu"  # type: ignore[assignment,return-value]
    install_cpu_transport()
    raise SystemExit(executor_main(sys.argv[1:]))


def leader(root: Path, degree: int, backend: str = "cpu") -> None:
    root.mkdir(parents=True, exist_ok=True)
    store = root / "store"
    torch.manual_seed(231)
    base = BasePipe(Config({"dit": CONFIG}))
    overlay = OverlayPipe(Config({}))
    with torch.no_grad():
        overlay.components["projection"].weight.copy_(torch.tensor([[1.0, 0.25], [-0.5, 1.25]]))
        expected = base.components["dit"](
            **batch(), attention_kwargs={"projection": overlay.components["projection"]}
        )
        expected_plain = base.components["dit"](**batch())
    native_model(store, "base", base.components, {"dit": CONFIG})
    native_model(store, "overlay", overlay.components, {})
    found = discover(PACKAGE)
    interface = package_interface.build(found)
    staged = stage_development_placement(
        root / "home",
        interface,
        package="test/dual-model-package",
        release="0.0.1",
        source_digest=development_source_digest(PACKAGE),
        artifact_cache=root / "cache",
        tensorfs_root=store,
        project=PACKAGE,
        binding_overrides={
            "generate.models.base": "test/base@1",
            "generate.models.overlay": "test/overlay@1",
        },
    )
    binding = next(iter(staged.declared_bindings.values()))
    command = executor_prepare_command(
        binding,
        devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        sequence_parallel_degree=degree,
        authorized_device_limit_bytes=2 << 30,
    )
    # Tiny weights do not need the production 256 MiB staging ring on every rank.
    for row in command["models"]:
        row["binding"].update(window_bytes=1 << 20, slots=2, readers=1, inflight=1)
    # The test's explicit CPU variant is qualified on CPU; no CUDA claim is substituted.
    if backend == "cpu":
        for row in command["models"]:
            row["binding"]["variant"] = "cpu"
    ours, theirs = socket.socketpair()
    # Thread caps are sealed process inputs; set them before leader and followers snapshot the seal.
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    executor = Executor(Channel(ours), root / "executor", world=degree)
    if backend == "cpu":
        install_cpu_transport()
    executor.device_kind = cast(Any, backend)
    _capture_seal()

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(Path(__file__)), *module_argv[2:]],
            inherit_fd=fd,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    group = (
        RankGroup(degree=degree, backend="cpu:gloo", root=str(root / "group"), launch=launch)
        if degree > 1
        else None
    )
    executor.group = group
    try:
        if group is not None:
            group.spawn()
            group.dial_back(
                __import__("cozy_runtime.internal.executor", fromlist=["_SEALED"])._SEALED
            )
        reply = executor.prepare(command)
        assert reply.get("ok"), reply
        assert len(executor.model_executors) == 2
        assert set(executor.models) == {"base", "overlay"}
        assert all(child.facts["filled_bytes"] > 0 for child in executor.model_executors)
        # Cross the real admission boundary before calling either model directly.
        # The former fixture skipped the chooser and could not expose the invented
        # aggregate delivery rung that refused real turbo serving requests.
        resolved = PreparedModel.from_reply(reply)
        ledger = Ledger()
        ledger.observe_construction(reply["facts"])
        chooser = PlanChooser(ledger, degree=degree)
        request_id = "native-multi-model-invocation"
        request = executor.prepare_request(
            {"request_id": request_id, "entrypoint": "generate", "payload": {"value": 9}}
        )
        assert request["ok"], request
        prepared = PreparedRequest(
            request_id=request_id,
            entrypoint="generate",
            features=request["features"],
            features_digest=request["features_digest"],
            overlay_digest=request["overlay_digest"],
            facts_digest=request["facts_digest"],
        )
        attempt_plan = chooser.choose(binding, resolved, prepared)
        deliveries = attempt_plan.document()["model_deliveries"]
        assert set(deliveries) == {"base", "overlay"}, deliveries
        assert {name: row["routes"] for name, row in deliveries.items()} == {
            name: row["delivery"]["routes"] for name, row in reply["facts"]["models"].items()
        }
        served = executor.invoke(
            {
                "request_id": request_id,
                "entrypoint": "generate",
                "deadline_s": None,
                "spool": str(root / "invocation"),
                "placement": attempt_plan.placement,
                "headroom_bytes": attempt_plan.headroom_bytes,
                "scope_headroom_bytes": dict(attempt_plan.scope_headroom_bytes),
            }
        )
        assert served["ok"] and served["outcome"]["terminal"] == "succeeded", served
        assert not executor.poisoned
        executor._attempt_spool = root / "attempt"
        scope_headrooms = {"base.sample": 11, "overlay.sample": 17}
        executor.residency.open_attempt("all_resident", 0, scope_headrooms)
        device = next(executor.models["base"].pipe.components["dit"].parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch().items()}
        values = SimpleNamespace(values=inputs, opaque=lambda: None)
        callbacks: list[int] = []
        for index in range(2):
            got = executor.models["base"].sample(
                executor.models["overlay"], values, lambda index=index: callbacks.append(index)
            )
            torch.testing.assert_close(tuple(t.cpu() for t in got), expected, rtol=2e-5, atol=2e-6)
        plain = executor.models["base"].plain(inputs)
        torch.testing.assert_close(
            tuple(t.cpu() for t in plain), expected_plain, rtol=2e-5, atol=2e-6
        )
        assert callbacks == [0, 1]
        for remote in group.call({"cmd": "probe"}) if group is not None else []:
            assert remote["ok"] and len(remote["models"]) == 2, remote
            assert not any(remote["active"].values()), remote
            assert all(value > 0 for value in remote["filled"]), remote
        report = executor.residency.document()
        assert report["scope_headroom_bytes"] == scope_headrooms, report
        assert set(report["attempt_activation_peaks"]) == {
            "base.sample",
            "base.plain",
            "overlay.sample",
        }, report
        for child in executor.model_executors:
            for captured in child.backend.derived.values():
                for (module, name), retained in captured.items():
                    current = module._buffers[name]
                    if current.device == retained.device and current.numel():
                        assert (
                            current.untyped_storage().data_ptr()
                            != retained.untyped_storage().data_ptr()
                        )
        if backend == "cuda":
            torch.cuda.synchronize()
        vacated = executor.vacate()
        assert vacated["ok"], vacated
        assert set(vacated["vacated"]["freed"]) == {"base/dit", "overlay/projection"}, vacated
        restored = executor.restore(names=("overlay/projection",))
        assert restored["ok"], restored
        assert restored["restored"]["restored"] == {"overlay/projection": 16}, restored
        assert restored["restored"]["held"] == {}, restored
        assert set(executor.residency.document()["resident"]) == {"overlay/projection"}
        restored = executor.restore(names=("base/dit",))
        assert restored["ok"] and set(restored["restored"]["restored"]) == {"base/dit"}, restored
        got = executor.models["base"].sample(
            executor.models["overlay"], values, lambda: callbacks.append(2)
        )
        torch.testing.assert_close(tuple(t.cpu() for t in got), expected, rtol=2e-5, atol=2e-6)
        assert callbacks == [0, 1, 2]
        print(
            json.dumps(
                {
                    "degree": degree,
                    "models": sorted(executor.models),
                    "callbacks": callbacks,
                    "filled_bytes": [
                        child.facts["filled_bytes"] for child in executor.model_executors
                    ],
                    "follower_pids": group.pids() if group is not None else [],
                    "source": "full native Executor.prepare",
                }
            ),
            flush=True,
        )
    finally:
        if group is not None:
            group.close()
        for child in executor.model_executors:
            if child.backend is not None:
                child.backend.close()
        ours.close()
        theirs.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--rank-fd", type=int)
    parser.add_argument("--leader")
    args = parser.parse_args()
    if args.backend == "cuda" and args.degree > 1:
        parser.error("this fixture uses one CUDA GPU; multi-GPU qualification uses Creator")
    torch.set_num_threads(1)
    if args.backend == "cpu":
        cpu_accounting()
    if args.rank is not None:
        follower(args)
    else:
        leader(args.root, args.degree, args.backend)


if __name__ == "__main__":
    main()
