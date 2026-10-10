"""How a group's measured links choose its route: pure decisions over real measurements."""

from __future__ import annotations

from cozy_runtime.internal.parallel.links import Links, Pair, Transport

#: lieselotte (4x H100 NVL): NVLink within pairs 0-1 and 2-3, peer copies across the bridge
NVL = tuple(
    Pair(a, b, True, 244.0 if {a, b} in ({0, 1}, {2, 3}) else 27.5, 3.5, True)
    for a in range(4)
    for b in range(4)
    if a != b
)
FORMED = {"nccl_peer": True, "copy_engines": True, "detail": "", "nccl_ms": 4.2}


def test_direct_pairs_and_a_formed_process_group_take_peer_memory_and_copy_engines() -> None:
    links = Links(NVL, FORMED)
    assert links.nccl() == {"NCCL_P2P_LEVEL": "SYS"} and links.copy_engines
    assert links.why() == "every GPU pair copies directly faster than through host memory"
    assert Links.read(links.document()) == links


def test_one_pair_slower_direct_than_staged_keeps_nvlink_only() -> None:
    # ruijerd (4x RTX A5000): GPU 2 to GPU 3 copied directly at 0.68 GB/s, staged at 7.7
    slow = tuple(
        Pair(p.src, p.dst, True, 0.68 if (p.src, p.dst) == (2, 3) else 10.5, 7.7, True) for p in NVL
    )
    links = Links(slow, FORMED)
    assert links.nccl() == {"NCCL_P2P_LEVEL": "NVL"} and not links.copy_engines
    record = Transport(4, [4, 5, 6, 7], "nccl", links.nccl(), links).document()
    assert record["reason"] == (
        "GPU 6 to GPU 7 copies directly at 0.7 GB/s, through host memory at 7.7 GB/s"
    )
    assert [p["route"] for p in record["pairs"] if (p["a"], p["b"]) == (6, 7)] == ["host"]


def test_a_process_group_that_failed_peer_memory_keeps_nvlink_only_and_says_why() -> None:
    failed = {"nccl_peer": False, "copy_engines": False, "detail": "DistBackendError: ncclProxy"}
    links = Links(NVL, failed)
    assert links.nccl() == {"NCCL_P2P_LEVEL": "NVL"} and not links.copy_engines
    assert links.why() == "peer memory between processes failed: DistBackendError: ncclProxy"


def test_nccl_peer_without_copy_engines_keeps_peer_memory_only() -> None:
    links = Links(NVL, {**FORMED, "copy_engines": False, "detail": "RuntimeError: bind"})
    assert links.nccl() == {"NCCL_P2P_LEVEL": "SYS"} and not links.copy_engines
    assert links.why().startswith("copy engines between processes failed")
