"""How a group's measured links choose its route: pure decisions over real measurements."""

from __future__ import annotations

from cozy_runtime.internal.parallel.links import Links, Pair, Transport

GPUS = ("GPU-a", "GPU-b", "GPU-c", "GPU-d")

#: lieselotte (4x H100 NVL): NVLink within pairs 0-1 and 2-3, peer copies across the bridge
NVL = tuple(
    Pair(a, b, True, 244.0 if {a, b} in ({0, 1}, {2, 3}) else 27.5, 3.5, True)
    for a in range(4)
    for b in range(4)
    if a != b
)
FORMED = {"nccl_peer": True, "copy_engines": True, "level": "SYS", "detail": "", "nccl_ms": 4.2}


def test_direct_pairs_and_a_formed_process_group_take_peer_memory_and_copy_engines() -> None:
    links = Links(NVL, FORMED)
    assert links.nccl() == {"NCCL_P2P_LEVEL": "SYS"} and links.copy_engines
    assert links.why() == "every GPU pair copies directly faster than through host memory"
    assert Links.among(links.document(GPUS), GPUS) == links


def test_one_pair_slower_direct_than_staged_keeps_nccl_on_nvlink() -> None:
    # ruijerd (4x RTX A5000): GPU 2 to GPU 3 copied directly at 0.68 GB/s, staged at 7.7
    slow = tuple(
        Pair(p.src, p.dst, True, 0.68 if (p.src, p.dst) == (2, 3) else 10.5, 7.7, True) for p in NVL
    )
    unformed = Links(slow, {})
    assert unformed.nccl() == {"NCCL_P2P_LEVEL": "NVL"} and not unformed.copy_engines
    record = Transport(4, [4, 5, 6, 7], "nccl", unformed.nccl(), unformed).document()
    assert record["reason"] == (
        "GPU 6 to GPU 7 copies directly at 0.7 GB/s, through host memory at 7.7 GB/s"
    )
    # Peer copies all land correctly and a process group mapped them: NCCL keeps NVLink-only
    # peer memory, and the copy engines are still timed against it (masayuki, 4x H100 NVL).
    formed = Links(slow, FORMED)
    assert formed.nccl() == {"NCCL_P2P_LEVEL": "NVL"} and formed.copy_engines
    assert formed.why().endswith("copy engines are timed against it")
    assert [p["route"] for p in record["pairs"] if (p["a"], p["b"]) == (6, 7)] == ["host"]


def test_a_corrupting_peer_copy_keeps_nccl_and_copy_engines_off() -> None:
    # garrod (2x RTX A5000): GPU 1 to GPU 0 direct copies landed different bytes
    broken = (Pair(0, 1, True, 0.8, 13.1, True), Pair(1, 0, True, 0.0, 13.0, False, "landed"))
    links = Links(broken, FORMED)
    assert links.nccl() == {"NCCL_P2P_LEVEL": "NVL"} and not links.copy_engines


def test_a_process_group_that_failed_peer_memory_keeps_nvlink_only_and_says_why() -> None:
    failed = {"nccl_peer": False, "copy_engines": False, "detail": "DistBackendError: ncclProxy"}
    links = Links(NVL, failed)
    assert links.nccl() == {"NCCL_P2P_LEVEL": "NVL"} and not links.copy_engines
    assert links.why() == "peer memory between processes failed: DistBackendError: ncclProxy"


def test_nccl_peer_without_copy_engines_keeps_peer_memory_only() -> None:
    links = Links(NVL, {**FORMED, "copy_engines": False, "detail": "RuntimeError: bind"})
    assert links.nccl() == {"NCCL_P2P_LEVEL": "SYS"} and not links.copy_engines
    assert links.why().startswith("copy engines between processes failed")


def test_a_group_of_some_of_the_machines_gpus_reads_their_pairs_from_its_boot_measurement() -> None:
    machine = Links(NVL, FORMED).document(GPUS)
    # GPU-d then GPU-b: positions 3 and 1 of the machine, 0 and 1 of the group
    group = Links.among(machine, ("GPU-d", "GPU-b"))
    assert group is not None and group.nccl() == {"NCCL_P2P_LEVEL": "SYS"} and group.copy_engines
    assert [(p.src, p.dst, p.direct_gbps) for p in group.pairs] == [(1, 0, 27.5), (0, 1, 27.5)]
    assert Links.among(machine, ("GPU-a", "GPU-e")) is None


def test_a_group_whose_route_the_machines_process_group_did_not_try_measures_itself() -> None:
    # ruijerd: the machine formed at NVL for its slow pair; GPUs a and b alone would take SYS
    slow = tuple(
        Pair(p.src, p.dst, True, 0.68 if (p.src, p.dst) == (2, 3) else 10.5, 7.7, True) for p in NVL
    )
    machine = Links(slow, {**FORMED, "level": "NVL"}).document(GPUS)
    assert Links.among(machine, ("GPU-a", "GPU-b")) is None
    group = Links.among(machine, ("GPU-c", "GPU-d"))
    assert group is not None and group.nccl() == {"NCCL_P2P_LEVEL": "NVL"} and group.copy_engines


def test_a_boot_probe_that_stopped_at_a_broken_pair_keeps_every_group_conservative() -> None:
    # The probe gave up on GPU 0 to GPU 2 and measured nothing after it.
    wedged = Pair(0, 2, True, 0.0, 3.5, False, "its direct copy did not finish")
    reason = "GPU 0 to GPU 2: its direct copy did not finish"
    stopped = Links((NVL[0], wedged), {}, reason).document(GPUS)
    group = Links.among(stopped, ("GPU-a", "GPU-b"))
    assert group is not None and group.nccl() == {"NCCL_P2P_LEVEL": "NVL"}
    assert not group.copy_engines and group.why() == reason
