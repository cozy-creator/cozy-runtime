"""A group installs what one GPU would, on every GPU at one boundary: a real follower process
and rank 0's executor over the control seam (`testdata/group_installs.py`)."""

from __future__ import annotations

import os
import runpy
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cozy_runtime.internal import attention, attention_sol, spawn  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.parallel.group import RankGroup  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402

FIXTURE = Path(__file__).parent / "testdata" / "group_installs.py"
HELD = runpy.run_path(str(FIXTURE), run_name="group_installs")


@contextmanager
def group_of_two(root: Path) -> Iterator[Any]:
    """Rank 0's construction (an `Executor`), with a real follower process holding the same
    one."""

    def launch(argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=[str(FIXTURE), *argv[2:]],
            inherit_fd=fd,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )

    group = RankGroup(degree=2, backend="cpu:gloo", root=str(root), launch=launch)
    left, right = socket.socketpair()
    parent: Any = Executor(Channel(left), root, world=2)
    parent.group, parent.torch, parent.device_kind = group, torch, "cpu"
    group.spawn()
    try:
        # Its construction is held once it answers: markers written later are read later.
        assert all(reply["ok"] for reply in group.call({"cmd": "hello"}))
        yield HELD["construction"](parent, root)
    finally:
        group.close()
        left.close()
        right.close()


def test_a_group_installs_its_kernels_only_once_every_gpu_has_them(tmp_path: Path) -> None:
    with group_of_two(tmp_path) as leader:
        assert leader._install_ready_kernels() is None
        (tmp_path / "built-0").touch()  # GPU 0's build is done, GPU 1's still compiling
        assert leader._install_ready_kernels() is None
        assert not list(tmp_path.glob("installed-*")), "no GPU installs while one cannot"
        assert [row[0] for row in leader._installs] == ["test-group-glue"]
        (tmp_path / "built-1").touch()
        assert leader._install_ready_kernels() is None
        assert sorted(p.name for p in tmp_path.glob("installed-*")) == [
            "installed-0",
            "installed-1",
        ]
        assert leader._installs == [] and leader._install_ready_kernels() is None
        assert not leader.poisoned


def test_a_set_one_gpu_cannot_build_is_dropped_on_every_gpu(tmp_path: Path) -> None:
    with group_of_two(tmp_path) as leader:
        (tmp_path / "built-0").touch()
        (tmp_path / "failed-1").touch()
        assert leader._install_ready_kernels() is None
        assert leader._installs == [] and not list(tmp_path.glob("installed-*"))
        assert not leader.poisoned


def test_an_install_only_a_follower_refuses_is_a_divergent_group(tmp_path: Path) -> None:
    with group_of_two(tmp_path) as leader:
        for name in ("built-0", "built-1", "refuse-1"):
            (tmp_path / name).touch()
        refused = leader._install_ready_kernels()
        assert refused is not None and refused["code"] == "gpu_divergence"
        assert "GPU 1" in str(refused["detail"]) and leader.poisoned == "group: gpu_divergence"


def test_an_install_rank_0_refuses_is_dropped_and_the_group_stays_whole(tmp_path: Path) -> None:
    with group_of_two(tmp_path) as leader:
        for name in ("built-0", "built-1", "refuse-0"):
            (tmp_path / name).touch()
        assert leader._install_ready_kernels() is None
        assert leader._installs == [] and not list(tmp_path.glob("installed-*"))
        assert not leader.poisoned


def test_a_follower_serves_rank_0s_whole_selection(tmp_path: Path) -> None:
    with group_of_two(tmp_path) as leader:
        roots = leader.backend.roots
        # GPU 1 starts on flash-attn3; GPU 0 serves SDPA, which GPU 1 then serves too.
        assert attention.chosen(roots).sites == (("dit", "", "sdpa"),)
        assert leader._agree_attention() is None
        assert not leader.poisoned


def test_a_kernel_one_gpu_lacks_keeps_the_group_on_what_every_gpu_has(tmp_path: Path) -> None:
    with group_of_two(tmp_path) as leader:
        roots = leader.backend.roots
        assert leader._agree_attention() is None
        # GPU 0 moves to a kernel GPU 1 cannot serve now: GPU 1 says so instead of waiting
        # for a build, and the group stays on SDPA, which every GPU has.
        roots["dit"].processor._attention_backend = attention._member(
            attention.known()["flash-attn3"]
        )
        assert leader._agree_attention(poison=False) is None
        assert attention.chosen(roots).sites == (("dit", "", "sdpa"),)
        assert "GPU 1" in leader.attention.skipped["flash-attn3"]
        assert not leader.poisoned


def test_the_selection_carries_sols_dense_reference_and_restores_it() -> None:
    site = HELD["Attention"](query_dim=32, heads=2, dim_head=16)
    roots = {"dit": site}
    attention.adopt(roots)
    site.processor._attention_backend = attention._member(attention.known()["sol-attn"])
    held = attention_sol.binding()

    def dense(**_: Any) -> Any:
        raise AssertionError("not called")

    try:
        attention_sol.bind_dense(dense, "sageattention; exact=sdpa", dense)
        prepared = attention.chosen(roots)
        assert (prepared.dense, prepared.exact) == ("sageattention", "sdpa")
        defaults = attention.snapshot(roots)
        attention_sol.bind_dense(dense, "sdpa")  # a later selection moved Sol's dense rows
        moved = attention.chosen(roots)
        assert moved.sites == prepared.sites
        assert attention.difference(prepared, moved).startswith(
            "Sol's dense reference sageattention"
        )
        defaults.restore()
        assert attention.chosen(roots) == prepared
    finally:
        attention_sol.rebind(held)
