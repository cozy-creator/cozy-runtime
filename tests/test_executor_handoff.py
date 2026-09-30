"""RUN 187, AS A TEST. The executor handoff is a UID BOUNDARY, and the grant covers exactly
one directory.

`ExecutorSupervision.handoff` hands its directory to the current executor -- `chown` to the
executor uid, `chmod 0700` -- and that directory is the WHOLE granted surface. The two
callers used to append `metadata/` and `mkdir` it themselves, as the ROOT worker, inside the
directory just handed over; the unprivileged executor then could not create its document
there. Run 187 died on exactly that, on a rented GPU, after a 92 GiB fetch had already
succeeded:

    PermissionError: [Errno 13] Permission denied:
    '/run/cozy/worker/handoff/metadata/package-interface.json'

Nothing had covered it because nothing tests the isolated path: `executor_uid` appears in no
other test in this tree, and an unprivileged developer machine runs the executor as itself,
where every one of these opens succeeds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.child import (
    INTERFACE_DOCUMENT,
    ExecutorGone,
    ExecutorSupervision,
)

#: The pod's executor identity (`cli/runtime_worker.py`). Nothing on a developer machine
#: may hand a child another uid, so these proofs run where the pod runs: as root, in the
#: worker image.
EXECUTOR_UID = 65533
EXECUTOR_GID = 65533

root_only = pytest.mark.skipif(
    os.geteuid() != 0,
    reason="a uid boundary can only be crossed by a process that may hand a child another uid",
)


@pytest.fixture(autouse=True)
def traversable_fixture_root(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """The real machine root is traversable; pytest's private parents default to 0700."""
    if os.geteuid() == 0:
        base = tmp_path_factory.getbasetemp()
        paths = [tmp_path, base]
        if base.name.startswith("pytest-") and base.parent.name.startswith("pytest-of-"):
            paths.append(base.parent)
        for path in paths:
            path.chmod(path.stat().st_mode | 0o111)


class _Epoch:
    """The one thing `grant_directory` reads off the current executor."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch


def _supervision(tmp_path: Path, *, isolated: bool) -> ExecutorSupervision:
    supervision = ExecutorSupervision(
        root=tmp_path / "worker",
        python=sys.executable,
        base_env=(),
        cozy_home=tmp_path / "home",
        executor_uid=EXECUTOR_UID if isolated else -1,
        executor_gid=EXECUTOR_GID if isolated else -1,
    )
    supervision.current = _Epoch(7)  # type: ignore[assignment]
    return supervision


def _create_as_executor(path: Path) -> str:
    """Create `path` exactly as `describe_installed` does, as the executor uid. Returns ""
    on success or the refusal, from a forked child so this process keeps its identity."""

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        outcome = ""
        try:
            os.close(read_fd)
            os.setgid(EXECUTOR_GID)
            os.setuid(EXECUTOR_UID)
            with path.open("xb") as stream:
                stream.write(b"{}")
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


@root_only
def test_the_executor_can_write_the_document_the_handoff_returns(tmp_path: Path) -> None:
    supervision = _supervision(tmp_path, isolated=True)
    document = supervision.handoff(7, INTERFACE_DOCUMENT)

    refusal = _create_as_executor(document)
    assert refusal == "", f"the executor could not write its own handoff document: {refusal}"
    # And the root worker reads it back, which is the other half of the handoff.
    assert document.read_bytes() == b"{}"


@root_only
def test_a_directory_beneath_the_grant_is_not_granted(tmp_path: Path) -> None:
    """The negative control: the shape run 187 refused. A subdirectory the root worker digs
    under the granted directory belongs to root, and the executor cannot create in it."""

    supervision = _supervision(tmp_path, isolated=True)
    document = supervision.handoff(7, INTERFACE_DOCUMENT)
    ungranted = document.parent / "metadata"
    ungranted.mkdir()

    refusal = _create_as_executor(ungranted / INTERFACE_DOCUMENT)
    assert "PermissionError" in refusal, (
        "a root-created subdirectory beneath the grant was writable by the executor, so "
        f"this proof no longer demonstrates why the document is not nested: {refusal!r}"
    )


@root_only
def test_a_stale_document_does_not_refuse_the_next_preparation(tmp_path: Path) -> None:
    """`describe_installed` creates the document exclusively, so one left behind by an
    interrupted preparation would refuse the re-issue -- and a preparation whose fetch or
    executor died is re-issued by design."""

    supervision = _supervision(tmp_path, isolated=True)
    first = supervision.handoff(7, INTERFACE_DOCUMENT)
    assert _create_as_executor(first) == ""

    second = supervision.handoff(7, INTERFACE_DOCUMENT)
    refusal = _create_as_executor(second)
    assert refusal == "", f"a stale handoff document refused the next preparation: {refusal}"


def test_the_handoff_takes_one_file_name(tmp_path: Path) -> None:
    """The returned path is granted; a path a caller builds beneath it is not, so the
    document may not carry a directory of its own."""

    supervision = _supervision(tmp_path, isolated=False)
    assert supervision.handoff(7, INTERFACE_DOCUMENT).name == INTERFACE_DOCUMENT
    for bad in ("metadata/package-interface.json", "..", "", "."):
        with pytest.raises(ExecutorGone):
            supervision.handoff(7, bad)
