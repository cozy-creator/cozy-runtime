"""Private file capacity is checked before interpreter probing or carrier reads."""

from pathlib import Path

import pytest

from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, prepare_package
from cozy_runtime.protocol import worker_pb2 as pb


@pytest.mark.parametrize("count", [0, 130])
def test_unpublished_revision_count_refused_before_io(count: int, tmp_path: Path) -> None:
    with pytest.raises(PreparationRefusal) as fault:
        prepare_package(
            package_name="local/capacity",
            release="1.0.0",
            files=[pb.LocalPackageFile()] * count,
            wheel_root=tmp_path / "absent",
            installation_id="local-capacity",
            artifact_cache=tmp_path / "cache",
            install_root=tmp_path / "installs",
            python=tmp_path / "no-python",
            describe=lambda *_: b"",
            verified=lambda *_: None,
            development=pb.DevelopmentPackage(
                package="local/capacity", release="1.0.0", installation_id="local-capacity"
            ),
        )
    assert fault.value.code == "package_prepare_file_count"
    assert list(tmp_path.iterdir()) == []
