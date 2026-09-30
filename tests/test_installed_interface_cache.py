"""Real uv environments and preparation executors exercise interface reuse and repair."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cozy_runtime.internal import package_installation, package_interface, static_interface
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config, _workspace, needs_executor
from test_publish_derivation import _closure_rows, _image_python
from test_unpublished_exact_dependencies import _wheel

ROOT = Path(__file__).resolve().parent.parent


@needs_executor
@pytest.mark.parametrize("metadata,memoized", [(False, False), (True, False), (True, True)])
def test_real_private_prepares_reuse_and_rebuild_interface(metadata: bool, memoized: bool) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    with _workspace() as root:
        wheels = root / "wheels"
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(wheels), str(ROOT)],
            check=True,
            capture_output=True,
        )
        runtime = next(wheels.glob("cozy_runtime-*.whl"))
        python = _image_python(uv, root, runtime)
        counter = root / "imports"
        # Observe the actual process alarm. Supervised preparation must not arm
        # a fixed deadline; builder discovery remains explicitly bounded.
        source = (
            "import signal\nfrom pathlib import Path\nfrom cozy_runtime.author import App\n"
            f"with Path({str(counter)!r}).open('a') as stream: stream.write('imported\\n')\n"
            "assert signal.getitimer(signal.ITIMER_REAL)[0] == 0\n"
            "app = App()\nalternate = App()\n"
        ).encode()
        if memoized:
            source += (
                b"from cozy_runtime.author import Context, invocable\n"
                b"import msgspec\nclass Result(msgspec.Struct):\n    value: int\n"
                b"@invocable(memoize=True)\n"
                b"async def measure(ctx: Context, *, value: int) -> Result:\n"
                b"    return Result(value)\n"
                b"app.job(measure)\n"
            )
        resources = {
            "cache_probe-1.0.dist-info/entry_points.txt": (
                b"[cozy.application]\ndefault=cache_probe:app\n"
            )
        }
        if metadata:
            authored = root / "source"
            authored.mkdir()
            (authored / "cache_probe.py").write_bytes(source)
            (authored / "package.toml").write_text('[application]\nobject="cache_probe:app"\n')
            resources["cache_probe-1.0.dist-info/package-interface.json"] = (
                package_interface.canonical_bytes(static_interface.build(authored))
            )
        project = _wheel(
            wheels,
            "cache_probe",
            ("cozy-runtime>=0.18",),
            source,
            resources=resources,
        )
        requirements = ("\n".join(_closure_rows(uv)) + "\n").encode()
        install_root = root / "installs"

        def install() -> package_installation.InstalledEnvironment:
            return package_installation.install(
                install_root,
                installation_id="captured",
                package="local/cache-probe",
                release="1.0",
                python=python,
                wheels=(runtime, project),
                requirements=requirements,
            )

        installed = install()
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(
                root=root / "worker",
                python=str(python),
                artifact_cache=root / "artifacts",
                install_root=install_root,
                tensorfs_root=root / "store",
            ),
            InMemoryControlHost(),
        )
        request = pb.PrepareLocalPackageRequest(
            install_root=str(install_root),
            operation_id="cache-replay",
            files=[pb.LocalPackageFile(filename=project.name)],
            package=pb.DevelopmentPackage(
                package=installed.package,
                release=installed.release,
                installation_id=installed.installation_id,
            ),
        )

        def prepare(_: int = 0) -> bytes:
            return bytes(worker.prepare_local_package(request).installed_package.package_interface)

        def imports() -> int:
            return len(counter.read_text().splitlines()) if counter.exists() else 0

        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(pool.map(prepare, range(4)))
            assert len(set(replies)) == 1
            assert package_interface.parse(replies[0]).application == "cache_probe:app"
            described = 0 if metadata and not memoized else 1
            assert imports() == described
            if memoized:
                assert json.loads(replies[0])["jobs"][0]["invocable"]["operation_identity"]
            alternate = worker._describe_installed(
                installed, "cache-probe", application="cache_probe:alternate"
            )
            assert package_interface.parse(alternate).application == "cache_probe:alternate"
            assert imports() == described + 1
            # A genuine discovery failure is never cached.
            for _ in range(2):
                with pytest.raises(PreparationRefusal, match="interface_failed"):
                    worker._describe_installed(installed, "missing-distribution")
            assert len(worker._installed_interfaces) == described + 1
            record_path = install_root / "installations/captured/installation.json"
            record_path.unlink()
            repaired = install()
            assert repaired.generation == installed.generation
            assert repaired.incarnation != installed.incarnation
            assert prepare() == replies[0]
            assert imports() == 2 * described + 1
            # Real uv rebuilds revisit the same generation path after an SDK record
            # changes. Its replaced record must invalidate the earlier metadata.
            generations = []
            incarnations = []
            for _ in range(2):
                record = json.loads(record_path.read_bytes())
                record["machine"] = {"cozy-runtime": "0.0.0-old"}
                record_path.write_text(json.dumps(record))
                refreshed = package_installation.refresh(install_root, "captured")
                generations.append(refreshed.generation)
                incarnations.append(refreshed.incarnation)
                assert prepare() == replies[0]
            assert generations[0] == generations[1]
            assert incarnations[0] != incarnations[1]
            assert imports() == 4 * described + 1
            # Adopted environments are mutable, so each description imports.
            adopted = package_installation.retain_environment(
                install_root, refreshed.python, package=installed.package, release=installed.release
            )
            assert adopted.incarnation is None
            request.package.installation_id = adopted.installation_id
            for _ in range(2):
                actual = prepare()
                assert package_interface.parse(actual).application == "cache_probe:app"
            assert imports() == 4 * described + 3
        finally:
            worker.shutdown()


def test_explicit_builder_deadline_still_bounds_import(tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text(
        "import time\ntime.sleep(1)\nfrom cozy_runtime.author import App\napp=App()\n"
    )
    (tmp_path / "package.toml").write_text('[application]\nobject="slow:app"\n')
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from cozy_runtime.internal.discovery import discover; "
            f"discover(Path({str(tmp_path)!r}), seconds=0.01)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0
    assert "import_timeout" in done.stderr or "0.01" in done.stderr
