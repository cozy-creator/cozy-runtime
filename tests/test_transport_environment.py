"""Transport setup uses the one environment authority and restores temporary values."""

import os

import pytest

from cozy_runtime.internal.config import inherited_environment, short_socket_directory
from cozy_runtime.internal.parallel import group


@pytest.mark.parametrize("previous", [None, "/long/executor/scratch"])
def test_rendezvous_directory_restores_environment_after_failure(
    monkeypatch: pytest.MonkeyPatch, previous: str | None
) -> None:
    if previous is None:
        monkeypatch.delenv("TMPDIR", raising=False)
    else:
        monkeypatch.setenv("TMPDIR", previous)
    before = inherited_environment()
    with pytest.raises(RuntimeError, match="rendezvous failed"):
        with short_socket_directory():
            assert os.environ["TMPDIR"] in ("/tmp", "/dev/shm")
            raise RuntimeError("rendezvous failed")
    assert inherited_environment() == before


def test_nccl_settings_are_applied_before_communicator_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
    monkeypatch.setenv("NCCL_P2P_LEVEL", "NVL")
    monkeypatch.setattr(group, "ROUTED", {})
    group.abortable("gloo")
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "3"
    group.abortable("nccl")
    group.route({"NCCL_P2P_LEVEL": "SYS"})
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "0"
    assert os.environ["NCCL_P2P_LEVEL"] == "SYS"
    assert group.ROUTED == {"NCCL_P2P_LEVEL": "SYS"}
