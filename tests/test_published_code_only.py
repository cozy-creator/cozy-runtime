"""A real published environment can be retained before its caller binds a model."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, package_installation, package_interface
from cozy_runtime.internal import package_environment as environments
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    landing_models,
    prepare_package_set,
)
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import worker_pb2 as pb
from test_publish_derivation import _SimpleIndex
from test_unpublished_exact_dependencies import _wheel


@pytest.mark.parametrize("declared", [True, False])
def test_published_serving_code_prepares_without_models_or_executable_bindings(
    tmp_path: Path, declared: bool
) -> None:
    slots = ["generate.models.model"] if declared else []
    interface = canonical.write(
        {
            "format": package_interface.SCHEMA,
            "application": "serving_child:app",
            "entrypoints": [
                {
                    "name": "generate",
                    "models": [{"class": "Model", "path": slots[0], "component_use": {}}],
                    "request": {"fields": []},
                    "result": {"fields": []},
                }
            ]
            if declared
            else [],
            "jobs": [],
        }
    )
    wheel = _wheel(
        tmp_path,
        "serving_child",
        ("cozy-runtime>=0.1",),
        b"VALUE=42\n",
        version="1.0.0",
        resources={"serving_child-1.0.0.dist-info/package-interface.json": interface},
    )
    runtime = _wheel(tmp_path, "cozy_runtime", (), b"VALUE=1\n", version="1.0.0")
    index = _SimpleIndex(tmp_path / "index", [wheel, runtime])
    base = environments.observe_base()
    root = tmp_path / "installs"
    retained: list[package_installation.InstalledEnvironment] = []

    try:
        locked = environments.read_locked_requirements(
            (
                f"--index-url {index.url}\n"
                + "\n".join(
                    sorted(
                        f"{item.name.split('-')[0].replace('_', '-')}=={item.name.split('-')[1]} "
                        f"--hash=sha256:{hashlib.sha256(item.read_bytes()).hexdigest()}"
                        for item in [wheel, runtime]
                    )
                )
                + "\n"
            ).encode()
        )
        request = pb.PreparePackageSetRequest(
            install_root=str(root),
            application="serving_child:app",
            package_interface=interface,
            model_slot_paths=slots,
            download_delegation=canonical.write(
                {
                    "format": "cozy.worker.v1.DownloadDelegation/1",
                    "packages": [{"package": "proof/serving-child", "release": "1.0.0"}],
                    "models": [],
                }
            ),
            image_inventory=pb.ImageInventory(
                profile="proof",
                python=base.python_full_version,
                distributions=[
                    pb.ImageDistribution(distribution=n, version=v) for n, v in base.distributions
                ],
            ),
            locked_requirements=(
                f"--index-url {index.url}\n" + "\n".join(row.text for row in locked.rows) + "\n"
            ).encode(),
        )

        def prepare() -> pb.PreparePackageSetResult:
            return prepare_package_set(
                request,
                artifact_cache=tmp_path / "artifacts",
                tensorfs_root=tmp_path / "store",
                install_root=root,
                python=Path(sys.executable),
                verified=lambda *args: None,
                base=base,
                materialized=retained.append,
            )

        if not declared:
            with pytest.raises(PreparationRefusal, match="package_prepare_callable_absent"):
                prepare()
            assert retained == []
            return
        result = prepare()
        (placement,) = read_placement_set(result.placement_set)
        assert not placement.entrypoints and not placement.models
        assert placement.package == pb.PackageSelection(
            package="proof/serving-child", release="1.0.0"
        )
        assert len(retained) == 1
        assert placement.installation_id == retained[0].installation_id
        assert (retained[0].site_packages / "serving_child.py").is_file()
        reopened = package_installation.open_installation(root, retained[0].installation_id)
        assert reopened.site_packages == retained[0].site_packages
        again = read_placement_set(prepare().placement_set)[0]
        # The same release and lock reuse one environment.
        assert again.installation_id == placement.installation_id
        assert again.placement_id == placement.placement_id
    finally:
        index.close()


def test_real_worker_retains_published_serving_only_child_without_loading_model() -> None:
    import os
    import shutil
    import tempfile

    from conftest import image_python
    from cozy_runtime.internal import child_env
    from cozy_runtime.internal.config import Credentials, RuntimeConfig
    from cozy_runtime.internal.worker.acquire import Acquirer
    from cozy_runtime.internal.worker.control import InMemoryControlHost
    from cozy_runtime.internal.worker.session import Worker, WorkerOptions
    from test_end_to_end import NO_EXECUTOR
    from test_publish_derivation import _closure_rows

    if NO_EXECUTOR:
        pytest.skip(NO_EXECUTOR)

    python = image_python()
    runtime = next(python.parent.parent.parent.glob("wheels/cozy_runtime-*.whl"))
    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cozy-pubcode.") as raw:
        root = Path(raw)
        source = b"""import msgspec
from cozy_runtime.author import App, Model
app = App()
class Request(msgspec.Struct):
    text: str
class Reply(msgspec.Struct):
    text: str
class Pipeline:
    pass
class Reference(Model[Pipeline]):
    def load(self, loader):
        raise AssertionError("code preparation must not load a model")
@app.entrypoint
async def generate(payload: Request, model: Reference) -> Reply:
    raise AssertionError("code preparation must not execute a callable")
"""
        wheel = _wheel(
            root,
            "serving_child",
            ("cozy-runtime>=0.18.24",),
            source,
            version="1.0.0",
            resources={
                "serving_child-1.0.0.dist-info/entry_points.txt": (
                    b"[cozy.application]\napp = serving_child:app\n"
                )
            },
        )
        index = _SimpleIndex(root / "index", [runtime, wheel])
        runtime_version = runtime.name.split("-")[1]
        rows = sorted(
            [
                *_closure_rows(uv),
                f"cozy-runtime=={runtime_version} "
                f"--hash=sha256:{hashlib.sha256(runtime.read_bytes()).hexdigest()}",
                "serving-child==1.0.0 "
                f"--hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
            ]
        )
        locked = (
            "--index-url https://pypi.org/simple\n"
            + f"--extra-index-url {index.url}\n"
            + "\n".join(rows)
            + "\n"
        ).encode()
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                child_base_env=tuple(
                    sorted(
                        (key, value)
                        for key, value in os.environ.items()
                        if not child_env.erased(key) and key != "PYTHONPATH"
                    )
                ),
            ),
            WorkerOptions(
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(python),
                install_root=root / "installs",
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
            ),
            InMemoryControlHost(),
        )
        try:
            request = pb.PreparePackageSetRequest(
                install_root=str(root / "installs"),
                application="serving_child:app",
                package_interface=canonical.write(
                    {
                        "format": package_interface.SCHEMA,
                        "application": "serving_child:app",
                        "entrypoints": [
                            {
                                "name": "generate",
                                "models": [
                                    {
                                        "class": "Reference",
                                        "path": "generate.models.model",
                                        "component_use": {},
                                    }
                                ],
                                "request": {"fields": [{"name": "text", "type": "str"}]},
                                "result": {"fields": [{"name": "text", "type": "str"}]},
                            }
                        ],
                        "jobs": [],
                    }
                ),
                model_slot_paths=["generate.models.model"],
                download_delegation=canonical.write(
                    {
                        "format": "cozy.worker.v1.DownloadDelegation/1",
                        "packages": [{"package": "proof/serving-child", "release": "1.0.0"}],
                        "models": [],
                    }
                ),
                image_inventory=pb.ImageInventory(
                    profile="proof",
                    python=worker.base.python_full_version,
                    distributions=[
                        pb.ImageDistribution(distribution=n, version=v)
                        for n, v in worker.base.distributions
                    ],
                ),
                locked_requirements=locked,
            )
            result = worker.prepare_package_set(request)
            (placement,) = read_placement_set(result.placement_set)
            retained = next(
                value
                for value in worker.prepared_installations.values()
                if value.document == placement
            )
            assert retained.installed is not None
            assert not retained.document.models and not retained.document.entrypoints
            assert (retained.installed.site_packages / "serving_child.py").read_bytes() == source
            reopened = package_installation.open_installation(
                root / "installs", retained.installed.installation_id
            )
            assert reopened.site_packages == retained.installed.site_packages
            # The worker described the installed code; the request's copy is advisory.
            interface = package_interface.read_bytes(placement.package_interface)
            assert interface["jobs"] == [] and interface["entrypoints"][0]["name"] == "generate"
            acquirer = Acquirer(
                cache_root=root / "artifacts",
                install_root=root / "installs",
                selection_root=root / "selection",
                tensorfs_root=root / "store",
                python=python,
                base=worker.base,
            )
            bindings, count = acquirer._bindings(retained.document)
            assert bindings == {} and count == 1
            (repeated,) = read_placement_set(worker.prepare_package_set(request).placement_set)
            assert repeated.placement_id == placement.placement_id
            assert not repeated.entrypoints and not repeated.models
        finally:
            worker.shutdown()
            index.close()


def test_a_landing_preparation_installs_the_release_and_reads_no_model(tmp_path: Path) -> None:
    """h3a-089: while the Host is still landing the models, preparation installs the release
    and answers its installation; the selected model is not in the Store and nothing reads
    it. The same selection without `models_landing` reads the Store and refuses."""
    slot = {"class": "Model", "path": "generate.models.model", "component_use": {}}
    slot["sequence_parallel"] = {"degrees": [2, 4]}
    interface = canonical.write(
        {
            "format": package_interface.SCHEMA,
            "application": "serving_child:app",
            "entrypoints": [
                {
                    "name": "generate",
                    "models": [slot],
                    "request": {"fields": []},
                    "result": {"fields": []},
                },
                {
                    "name": "other",
                    "models": [
                        {"class": "Model", "path": "other.models.model", "component_use": {}}
                    ],
                    "request": {"fields": []},
                    "result": {"fields": []},
                },
            ],
            "jobs": [],
        }
    )
    wheel = _wheel(
        tmp_path,
        "serving_child",
        ("cozy-runtime>=0.1",),
        b"VALUE=42\n",
        version="1.0.0",
        resources={"serving_child-1.0.0.dist-info/package-interface.json": interface},
    )
    runtime = _wheel(tmp_path, "cozy_runtime", (), b"VALUE=1\n", version="1.0.0")
    index = _SimpleIndex(tmp_path / "index", [wheel, runtime])
    base = environments.observe_base()
    root = tmp_path / "installs"
    retained: list[package_installation.InstalledEnvironment] = []
    try:
        locked = "\n".join(
            sorted(
                f"{item.name.split('-')[0].replace('_', '-')}=={item.name.split('-')[1]} "
                f"--hash=sha256:{hashlib.sha256(item.read_bytes()).hexdigest()}"
                for item in [wheel, runtime]
            )
        )
        request = pb.PreparePackageSetRequest(
            install_root=str(root),
            application="serving_child:app",
            package_interface=interface,
            model_slot_paths=["generate.models.model", "other.models.model"],
            download_delegation=canonical.write(
                {
                    "format": "cozy.worker.v1.DownloadDelegation/1",
                    "packages": [{"package": "proof/serving-child", "release": "1.0.0"}],
                    "models": [
                        {
                            "package": "proof/serving-child",
                            "slot": "generate.models.model",
                            "model": "proof/landing",
                            "release": "1.0.0",
                            "lane": "bf16",
                            "manifest": "sha256:" + "ab" * 32,
                        }
                    ],
                }
            ),
            locked_requirements=f"--index-url {index.url}\n{locked}\n".encode(),
            models_landing=True,
        )

        def prepare(selected: pb.PreparePackageSetRequest) -> pb.PreparePackageSetResult:
            return prepare_package_set(
                selected,
                artifact_cache=tmp_path / "artifacts",
                tensorfs_root=tmp_path / "store",
                install_root=root,
                python=Path(sys.executable),
                verified=lambda *args: None,
                base=base,
                materialized=retained.append,
            )

        landed = prepare(request)
        assert not (tmp_path / "store").exists(), "a landing preparation opened the Store"
        (placement,) = read_placement_set(landed.placement_set)
        assert not placement.models and not placement.entrypoints
        assert placement.installation_id == retained[0].installation_id
        assert (retained[0].site_packages / "serving_child.py").is_file()
        # The executor starts at the width of what the selection binds whole: `generate`.
        assert landing_models(request, bytes(landed.installed_package.package_interface)) == [slot]
        request.models_landing = False
        with pytest.raises(PreparationRefusal, match="package_prepare_model"):
            prepare(request)
        # Both calls named one environment; the second installed nothing again.
        assert {row.installation_id for row in retained} == {retained[0].installation_id}
    finally:
        index.close()
