"""RUN 1271, AS A TEST. Replicas of one installation activate on their own threads (#725),
and each publishes that installation's package-interface.json for its executor. One of three
Qwen replicas on a 4xH100 pod was refused `package_interface_invalid: malformed_json:
Expecting value at offset 0`: it read the file while a sibling had truncated it.
"""

from __future__ import annotations

import threading
from pathlib import Path

from cozy_runtime.internal import base_observation, canonical, package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

REPLICAS = 4
ROUNDS = 200
ENTRYPOINTS = tuple(f"generate_{index}" for index in range(32))


def _placement() -> pb.Placement:
    schema: dict[str, Json] = {"fields": []}
    interface = canonical.write(
        {
            "application": "replicas:app",
            "entrypoints": [
                {"name": name, "request": schema, "result": schema} for name in ENTRYPOINTS
            ],
            "format": package_interface.SCHEMA,
            "jobs": [],
        }
    )
    return pb.Placement(
        placement_id="replica",
        package=pb.PackageSelection(package="proof/replicas", release="1.0.0"),
        installation_id="install-replicas",
        package_interface=interface,
        bindings_digest=documents.digest_of(canonical.write({"entrypoints": [], "models": []})),
        entrypoints=[
            pb.Entrypoint(
                name=name,
                entrypoint_binding_digest=documents.digest_of(
                    canonical.write({"entrypoint": name})
                ),
            )
            for name in ENTRYPOINTS
        ],
    )


def test_sibling_replicas_never_observe_a_partial_interface(tmp_path: Path) -> None:
    placement = _placement()
    path = tmp_path / "selection" / placement.installation_id / "package-interface.json"
    failures: list[str] = []
    done = threading.Event()

    def executor_reader() -> None:
        while not done.is_set():
            try:
                package_interface.read_bytes(path.read_bytes(), str(path))
            except FileNotFoundError:
                continue
            except package_interface.StalePackageInterface as exc:
                failures.append(f"executor: {exc}")

    def activate(barrier: threading.Barrier) -> None:
        acquirer = Acquirer(
            cache_root=tmp_path / "cache",
            install_root=tmp_path / "environment",
            selection_root=tmp_path / "selection",
            tensorfs_root=tmp_path / "store",
            base=base_observation.current(),
            python=Path("/usr/bin/python3"),
        )
        barrier.wait()
        try:
            bindings, _ = acquirer._bindings(placement)
        except AcquisitionRefusal as exc:
            failures.append(f"activation: {exc}")
            return
        if {binding.interface_path for binding in bindings.values()} != {str(path)}:
            failures.append("activation published another interface path")

    reader = threading.Thread(target=executor_reader)
    reader.start()
    try:
        for _ in range(ROUNDS):
            barrier = threading.Barrier(REPLICAS)
            replicas = [threading.Thread(target=activate, args=(barrier,)) for _ in range(REPLICAS)]
            for thread in replicas:
                thread.start()
            for thread in replicas:
                thread.join()
    finally:
        done.set()
        reader.join()

    assert failures == []
    assert path.read_bytes() == placement.package_interface
    # The executor reads this path as its own uid.
    assert path.stat().st_mode & 0o777 == 0o644
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]
