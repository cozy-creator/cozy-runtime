"""Both supported prepare shapes use one immutable install and one optional sandbox import."""

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
    package_environment,
    package_interface,
    static_interface,
)
from cozy_runtime.internal.worker.control import InMemoryControlHost, _PreparationServicer
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

    def _describe_installed(
        self, installed: package_environment.InstalledEnvironment, distribution: str
    ) -> bytes:
        self.descriptions.append(installed.content_digest)
        return super()._describe_installed(installed, distribution)


@needs_executor
@pytest.mark.parametrize("metadata", [False, True])
def test_wire60_absent_interface_and_wire61_interface_share_preparation(metadata: bool) -> None:
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
                        modern = client.PreparePackageSet(request)
                        assert not worker.descriptions
                        # Field 12 did not exist at minor 60; this is its actual wire shape.
                        request.ClearField("package_interface")
                        compatibility = client.CheckPackageSetCompatibility(request)
                        assert not compatibility.refusal_code and not compatibility.refusal_detail
                        legacy = client.PreparePackageSet(request)
                        repeated = client.PreparePackageSet(request)
                        assert legacy == repeated == modern
                        assert len(worker.descriptions) == (0 if metadata else 1)
                        assert len(list((root / "installs/contents").iterdir())) == 1
                        assert len(list((root / "artifacts/installed-interfaces").iterdir())) == 1
                        (placement,) = read_placement_set(legacy.placement_set)
                        assert placement["entrypoints"][0]["name"] == "serve"
                        request.application = "compat_package:wrong"
                        refused = client.CheckPackageSetCompatibility(request)
                        assert refused.refusal_code == "package_prepare_application_disagrees"
                        assert "compat_package:wrong" in refused.refusal_detail
                        with pytest.raises(grpc.RpcError, match="application_disagrees"):
                            client.PreparePackageSet(request)
                        assert len(worker.descriptions) == (0 if metadata else 1)
                        request.application = "compat_package:app"
                        request.model_slot_paths.append("serve.models.unexpected")
                        with pytest.raises(grpc.RpcError, match="model_slots_disagree"):
                            client.PreparePackageSet(request)
                        request.ClearField("model_slot_paths")
                        request.package_interface = b"{}"
                        refused = client.CheckPackageSetCompatibility(request)
                        assert refused.refusal_code == "package_prepare_interface_invalid"
                        with pytest.raises(grpc.RpcError, match="interface_invalid"):
                            client.PreparePackageSet(request)
                        assert len(worker.descriptions) == (0 if metadata else 1)
                finally:
                    server.stop(0).wait()
        finally:
            worker.shutdown()
            index.close()
