"""A prepare with or without the release's interface installs with uv and describes once,
only without metadata."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest

from cozy_runtime.internal import (
    canonical,
    package_installation,
    package_interface,
    proctree,
    static_interface,
)
from cozy_runtime.internal.worker.child import Executor, ExecutorSupervision
from cozy_runtime.internal.worker.control import InMemoryControlHost, _PreparationServicer
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal
from cozy_runtime.internal.worker.session import Worker, WorkerOptions, read_placement_set
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_device_lanes import _config, _workspace, needs_executor
from test_publish_derivation import _closure_rows, _image_python, _SimpleIndex
from test_unpublished_exact_dependencies import _wheel

ROOT = Path(__file__).resolve().parent.parent
SOURCE = b"""import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct): pass
class Result(msgspec.Struct): pass
@app.entrypoint
def serve(ctx: Context, payload: Request) -> Result:
    return Result()
"""


class ObservedWorker(Worker):
    descriptions: list[str]
    describe_executors: list[Executor]
    describe_supervisions: list[ExecutorSupervision]

    def _new_preparation_supervision(self) -> ExecutorSupervision:
        supervision = super()._new_preparation_supervision()
        self.describe_supervisions.append(supervision)

        def observe(executor: Executor | None) -> None:
            if executor is not None:
                self.describe_executors.append(executor)

        supervision.on_change = observe
        return supervision

    def _describe_in_slot(
        self,
        installed: package_installation.InstalledEnvironment,
        distribution: str,
        *,
        application: str = "",
    ) -> bytes:
        self.descriptions.append(installed.installation_id)
        return super()._describe_in_slot(installed, distribution, application=application)

    def assert_describes_reaped(self) -> None:
        for executor in self.describe_executors:
            assert executor.exit_status is not None
            assert proctree.process_state(executor.process) is None  # no live process or zombie
        for supervision in self.describe_supervisions:
            assert supervision.current is None and supervision._read_owner() is None
            assert not Path(supervision.socket_path).exists()


def _placement(result: pb.PreparePackageSetResult) -> pb.Placement:
    (placement,) = read_placement_set(result.placement_set)
    assert placement.installation_id == result.installed_package.installation_id
    placement.ClearField("installation_id")
    return placement


@needs_executor
@pytest.mark.parametrize("metadata", [False, True])
def test_absent_and_present_interface_share_preparation(metadata: bool) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    with _workspace() as root:
        wheels = root / "wheels"
        wheels.mkdir()
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(wheels), str(ROOT)],
            check=True,
            capture_output=True,
        )
        runtime = next(wheels.glob("cozy_runtime-*.whl"))
        source = root / "source"
        source.mkdir()
        (source / "compat_package.py").write_bytes(SOURCE)
        (source / "package.toml").write_text('[application]\nobject="compat_package:app"\n')
        interface = package_interface.canonical_bytes(static_interface.build(source))
        info = "compat_package-1.0.0.dist-info/"
        resources = {info + "entry_points.txt": b"[cozy.application]\ndefault=compat_package:app\n"}
        if metadata:
            resources[info + "package-interface.json"] = interface
        project = _wheel(
            wheels,
            "compat_package",
            ("cozy-runtime>=0.18", "msgspec>=0.19"),
            SOURCE,
            resources=resources,
            version="1.0.0",
        )
        python = _image_python(uv, root, runtime)
        index = _SimpleIndex(root / "index", [runtime, project])
        config = _config(root / "home")
        worker = ObservedWorker(
            config,
            WorkerOptions(
                root=root / "worker",
                python=str(python),
                artifact_cache=root / "artifacts",
                install_root=root / "installs",
                tensorfs_root=root / "store",
            ),
            InMemoryControlHost(),
        )
        worker.descriptions = []
        worker.describe_executors = []
        worker.describe_supervisions = []
        rows = sorted(
            [
                *_closure_rows(uv),
                *(
                    f"{wheel.name.split('-')[0].replace('_', '-')}=={wheel.name.split('-')[1]} "
                    f"--hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}"
                    for wheel in (runtime, project)
                ),
            ]
        )
        request = pb.PreparePackageSetRequest(
            install_root=str(root / "installs"),
            application="compat_package:app",
            package_interface=interface,
            download_delegation=canonical.write(
                {
                    "format": "cozy.worker.v1.DownloadDelegation/1",
                    "packages": [{"package": "proof/compat-package", "release": "1.0.0"}],
                    "models": [],
                }
            ),
            image_inventory=pb.ImageInventory(
                profile="proof", python=worker.base.python_full_version
            ),
            locked_requirements=(
                "--index-url https://pypi.org/simple\n"
                f"--extra-index-url {index.url}\n" + "\n".join(rows) + "\n"
            ).encode(),
        )
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                server = grpc.server(pool)
                rpc.add_RuntimePreparationServicer_to_server(
                    _PreparationServicer(worker.prepare_package_set, None, None, None), server
                )
                port = server.add_insecure_port("127.0.0.1:0")
                server.start()
                try:
                    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                        client = rpc.RuntimePreparationStub(channel)
                        described = 0 if metadata else 1
                        modern = client.PreparePackageSet(request)
                        assert len(worker.descriptions) == described
                        # Without the release's interface Runtime describes the install itself.
                        request.ClearField("package_interface")
                        legacy = client.PreparePackageSet(request)
                        assert _placement(legacy) == _placement(modern)
                        # One immutable installation is described once per worker.
                        assert len(worker.descriptions) == described
                        worker.assert_describes_reaped()
                        # One release and lock is one reused environment.
                        installations = (root / "installs/installations").glob("release-*")
                        assert len(list(installations)) == 1
                        (placement,) = read_placement_set(legacy.placement_set)
                        assert placement.entrypoints[0].name == "serve"
                        # Caller copies of installed facts are advisory.
                        request.application = "compat_package:wrong"
                        request.model_slot_paths.append("serve.models.unexpected")
                        request.package_interface = b"{}"
                        assert _placement(client.PreparePackageSet(request)) == _placement(modern)
                        assert len(worker.descriptions) == described
                        # Failure also returns only after the dedicated executor is reaped.
                        installed = next(iter(worker.prepared_installations.values())).installed
                        assert installed is not None
                        with pytest.raises(PreparationRefusal, match="interface_failed"):
                            worker._describe_installed(installed, "absent-distribution")
                        worker.assert_describes_reaped()
                finally:
                    server.stop(0).wait()
        finally:
            worker.shutdown()
            index.close()
