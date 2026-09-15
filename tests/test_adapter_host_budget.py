"""Host admission observes ancestor limits before allocating adapter source tensors."""

from pathlib import Path

from cozy_runtime.internal.proctree import _v2_host_available


def test_unlimited_child_respects_tight_ancestor_and_host(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    parent = root / "pod"
    child = parent / "executor"
    child.mkdir(parents=True)
    meminfo, membership = tmp_path / "meminfo", tmp_path / "membership"
    meminfo.write_text("MemAvailable: 1000 kB\n")
    membership.write_text("0::/pod/executor\n")
    (child / "memory.max").write_text("max\n")
    (parent / "memory.max").write_text("8192\n")
    (parent / "memory.current").write_text("4096\n")
    assert _v2_host_available(meminfo, root, membership) == 4096
    (child / "memory.max").write_text("1024\n")
    (child / "memory.current").write_text("512\n")
    assert _v2_host_available(meminfo, root, membership) == 512
    meminfo.write_text("MemAvailable: 0 kB\n")
    assert _v2_host_available(meminfo, root, membership) == 0


def test_unreadable_applicable_limit_never_falls_back_to_host(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text("4096")
    meminfo, membership = tmp_path / "meminfo", tmp_path / "membership"
    meminfo.write_text("MemAvailable: 1000000 kB\n")
    membership.write_text("0::/\n")
    assert _v2_host_available(meminfo, root, membership) == -1
