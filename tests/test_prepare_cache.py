"""RUN 188, AS A TEST. The prepare cache is the worker's own, and its documents cross a uid
boundary.

th-128 deleted the pod-global artifact CAS, and the worker image stopped creating
`/var/lib/cozy/tmp/transfers` with it. The SERVING half never noticed, because
`acquire._stage` made its own root; the PREPARING half did not, and a rented pod
refused with

    [Errno 2] No such file or directory:
    '/var/lib/cozy/tmp/transfers/.prepare-20de39121003468bba96af6147eb1836'

The second half of the same lesson is the mode. A staged interface's PATH is recorded into
job plans and handed to the executor, which runs as its own uid, so a root-owned 0400
document is the same class of refusal one layer further in -- which is exactly why
`acquire._stage` wrote 0444.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from cozy_runtime.internal import host_paths
from cozy_runtime.internal.worker.package_prepare import _stage

#: The pod's executor identity (`cli/runtime_worker.py`).
EXECUTOR_UID = 65533
EXECUTOR_GID = 65533

root_only = pytest.mark.skipif(
    os.geteuid() != 0,
    reason="a uid boundary can only be crossed by a process that may hand a child another uid",
)


def _read_as_executor(path: Path) -> str:
    """Read `path` as the executor uid does in `_discover_provisioned`. Returns "" on
    success or the refusal, from a forked child so this process keeps its identity."""

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        outcome = ""
        try:
            os.close(read_fd)
            os.setgid(EXECUTOR_GID)
            os.setuid(EXECUTOR_UID)
            path.read_bytes()
        except BaseException as exc:  # the refusal IS the result this proof collects
            outcome = f"{type(exc).__name__}: {exc}"
        try:
            os.write(write_fd, outcome.encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as stream:
        outcome = stream.read().decode()
    os.waitpid(pid, 0)
    return outcome


def test_staging_creates_its_own_root(tmp_path: Path) -> None:
    """The image creates no cache directory, so the worker must. This is run 188."""

    root = tmp_path / "prepare-cache"
    assert not root.exists()
    digest, length = _stage(root, b"{}")
    staged = root / digest.removeprefix("sha256:")
    assert staged.read_bytes() == b"{}"
    assert length == 2


def test_a_staged_document_is_readable_by_another_uid(tmp_path: Path) -> None:
    """The mode is a uid decision, made here and asserted here: the executor opens this
    path by name, and nothing chmods it afterwards."""

    root = tmp_path / "prepare-cache"
    digest, _ = _stage(root, b"{}")
    staged = root / digest.removeprefix("sha256:")
    assert stat.S_IMODE(staged.stat().st_mode) == 0o444
    assert stat.S_IMODE(root.stat().st_mode) == 0o755


def test_restaging_the_same_bytes_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "prepare-cache"
    first = _stage(root, b'{"a":1}')
    second = _stage(root, b'{"a":1}')
    assert first == second
    assert [p.name for p in root.iterdir()] == [first[0].removeprefix("sha256:")]


def test_the_cache_root_is_not_the_install_root() -> None:
    """`artifact_cache` holds content-addressed CONTROL DOCUMENTS; the install root holds
    package environments. Collapsing them would scatter digest-named files among installs."""

    layout = host_paths.Layout()
    assert layout.prepare_cache != layout.install_root
    assert layout.install_root not in layout.prepare_cache.parents


@root_only
def test_the_executor_uid_can_read_a_staged_document(tmp_path: Path) -> None:
    root = tmp_path / "prepare-cache"
    for parent in (tmp_path, *tmp_path.parents):
        if parent == Path("/"):
            break
        os.chmod(parent, os.stat(parent).st_mode | 0o755)
    digest, _ = _stage(root, b"{}")
    staged = root / digest.removeprefix("sha256:")

    refusal = _read_as_executor(staged)
    assert refusal == "", f"the executor could not read the interface it is handed: {refusal}"


@root_only
def test_a_root_only_document_is_refused_by_the_executor(tmp_path: Path) -> None:
    """The negative control: the mode this staging used to leave behind. It is what the
    executor would have hit one layer past run 188."""

    root = tmp_path / "prepare-cache"
    for parent in (tmp_path, *tmp_path.parents):
        if parent == Path("/"):
            break
        os.chmod(parent, os.stat(parent).st_mode | 0o755)
    digest, _ = _stage(root, b"{}")
    staged = root / digest.removeprefix("sha256:")
    staged.chmod(0o400)

    refusal = _read_as_executor(staged)
    assert "PermissionError" in refusal, (
        "a root-owned 0400 document was readable by the executor, so this proof no longer "
        f"demonstrates why staging writes 0444: {refusal!r}"
    )
