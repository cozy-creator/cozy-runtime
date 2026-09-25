"""Actual package preparation and cached-child imports for the isolated dev-worker proof.

The wheel directory contains ordinary built app/helper wheels and public Runtime0.2.24
and TensorFS0.3.10 wheels. Framework pins stay old while the installed base changes.
Run each phase in a fresh process, preserving only this test root and the package cache.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import shutil
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path
from typing import Any

from cozy_runtime.internal import (
    base_observation,
    canonical,
    package_environment,
    package_interface,
    static_interface,
)
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, prepare_package_set
from cozy_runtime.protocol import worker_pb2 as pb


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root: Path, wheels: Path, closure: Path, install: Path) -> None:
        self.root, self.install = root, install
        root.mkdir(parents=True, exist_ok=True)
        self.base = base_observation.current()
        versions = self.base.distribution_versions
        heavy = {
            key: value for key, value in versions.items() if key not in {"cozy-runtime", "tensorfs"}
        }
        heavy_path = root / "heavy.json"
        if heavy_path.exists():
            assert json.loads(heavy_path.read_text()) == heavy, "immutable heavy base changed"
        else:
            heavy_path.write_text(json.dumps(heavy, sort_keys=True))
        pins = []
        index = root / "index"
        for wheel in sorted(wheels.glob("*.whl")):
            name, version = wheel.name.split("-")[:2]
            name = name.replace("_", "-")
            project = index / name
            project.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wheel, project / wheel.name)
            (project / "index.html").write_text(
                f'<a href="{wheel.name}#sha256={digest(wheel)}">wheel</a>'
            )
            pins.append(f"{name}=={version} --hash=sha256:{digest(wheel)}")
        assert len(pins) == 4, "need exact app, helper and two old public framework wheels"
        port_path = root / "port"
        port = int(port_path.read_text()) if port_path.exists() else 0
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(index))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        port = self.server.server_address[1]
        port_path.write_text(str(port))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        exported = package_environment.read_locked_requirements(
            b"--index-url https://pypi.org/simple\n" + closure.read_bytes()
        )
        rows = sorted(
            [row.text for row in exported.rows] + pins, key=lambda row: row.split("==")[0]
        )
        locked = (f"--index-url http://127.0.0.1:{port}/\n" + "\n".join(rows) + "\n").encode()
        self.request = pb.PreparePackageSetRequest(
            download_delegation=canonical.write(
                {
                    "format": "cozy.worker.v1.DownloadDelegation/1",
                    "packages": [{"package": "proof/platform-provider-app", "release": "1.0.0"}],
                    "models": [],
                }
            ),
            install_root=str(install),
            application="platform_provider_app:app",
            model_slot_paths=[],
            image_inventory=pb.ImageInventory(
                profile="development-heavy-proof",
                python=".".join(str(part) for part in sys.version_info[:3]),
                distributions=[
                    pb.ImageDistribution(distribution=k, version=v)
                    for k, v in sorted(heavy.items())
                ],
            ),
            locked_requirements=locked,
            package_interface=package_interface.canonical_bytes(
                static_interface.build(Path(__file__).resolve().parent / "app")
            ),
        )

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def prepare(self) -> pb.PreparePackageSetResult:
        for name in ("artifacts", "store"):
            (self.root / name).mkdir(exist_ok=True)
        return prepare_package_set(
            self.request,
            artifact_cache=self.root / "artifacts",
            tensorfs_root=self.root / "store",
            install_root=self.install,
            python=Path(sys.executable),
            verified=lambda _digest, _length, _path: None,
            base=self.base,
        )

    def readback(self, prepared: pb.PreparePackageSetResult) -> dict[str, Any]:
        placement = json.loads(prepared.placement_set.placement_set_canonical_bytes)["placements"][
            0
        ]
        candidates = list((self.install / "contents").glob("*/bin/python"))
        assert len(candidates) == 1
        python = candidates[0]
        code = """import hashlib, importlib.metadata as m, json, pathlib, tempfile
import cozy_runtime, tensorfs, platform_provider_app as app, platform_provider_helper as helper
reply = app.versions(app.Request(text="proof"))
with tempfile.TemporaryDirectory(prefix="platform-native-") as root:
    store = tensorfs.Store.ensure(root)
    native_ops = store.model_source_operations()
so = next(pathlib.Path(tensorfs.__file__).parent.glob("*.so"))
print(json.dumps({"runtime": reply.runtime, "tensorfs": reply.tensorfs,
"helper": reply.helper, "runtime_module": cozy_runtime.__file__,
"tensorfs_module": tensorfs.__file__,
"native_sha256": hashlib.sha256(so.read_bytes()).hexdigest(), "native_operations": native_ops,
"helper_sha256": hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest(),
"helper_version": m.version("platform-provider-helper")}))"""
        result = subprocess.run(
            [str(python), "-I", "-c", code], cwd="/", capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        value: dict[str, Any] = json.loads(result.stdout)
        versions = self.base.distribution_versions
        assert value["runtime"] == versions["cozy-runtime"], value
        assert value["tensorfs"] == versions["tensorfs"], value
        assert value["helper"] == "locked-helper-1.0.0" and value["helper_version"] == "1.0.0"
        assert not Path(value["runtime_module"]).is_relative_to(self.install)
        assert not Path(value["tensorfs_module"]).is_relative_to(self.install)
        value.update(
            environment_digest=placement["environment_digest"],
            python=str(python),
            generation_inode=python.parent.parent.stat().st_ino,
        )
        before_path = self.root / "before.json"
        if before_path.exists():
            before = json.loads(before_path.read_text())
            for field in ("environment_digest", "python", "generation_inode", "helper_sha256"):
                assert before[field] == value[field], (field, before, value)
            value["cached_environment_reused"] = True
        else:
            before_path.write_text(json.dumps(value, sort_keys=True))
            value["cached_environment_reused"] = False
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("wheels", type=Path)
    parser.add_argument("closure", type=Path)
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--expect-refusal")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    fixture = Fixture(
        args.root, args.wheels, args.closure, args.install_root or args.root / "environments"
    )
    try:
        try:
            prepared = fixture.prepare()
        except PreparationRefusal as exc:
            if args.expect_refusal != exc.code:
                raise
            print(json.dumps({"refusal": exc.code}))
            return
        assert not args.expect_refusal, "expected preparation refusal"
        if args.prepare_only:
            copies = [
                str(path)
                for pattern in ("cozy_runtime-*.dist-info", "tensorfs-*.dist-info")
                for path in fixture.install.glob("contents/*/lib/python*/site-packages/" + pattern)
            ]
            print(json.dumps({"prepared": True, "local_framework_metadata": sorted(copies)}))
            return
        print(json.dumps(fixture.readback(prepared), sort_keys=True))
    finally:
        fixture.close()


if __name__ == "__main__":
    main()
