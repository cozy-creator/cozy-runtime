"""`cozy-runtime attention-qualify`: every attention kernel under real Ulysses on this host.

One process per rank (NCCL on CUDA, gloo on CPU). Each rank places the image's kernel site,
resolves each kernel through preparation's own path (its per-rank numeric probe included),
then runs the registered backend at degree N on identical seeded global Q/K/V and compares
its sequence shard with the same backend at degree 1. Ulysses hands each rank the whole
sequence of its heads, so a head-local kernel must reproduce one GPU.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import msgspec

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.author._numeric import within
from cozy_runtime.internal import attention, attention_sol, attention_ulysses, kernel_site
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.kernel_site import KernelSite

# torch, torch.distributed, their tensors and Diffusers' parallel config are `Any`: neither
# torch nor diffusers is installed in the check venv.

#: One H3 rank group's packed 10 s document (unequal shards) and its 56 heads.
ROWS, HEADS = 73_439, 56
#: Degree N against degree 1: the same per-head arithmetic, so beyond one BF16 rounding the
#: wrapper, not the kernel, is wrong.
EQUIVALENT_REL_L2 = 2e-3
#: H3's measured protected prefix (text + reference + audio rows) for Sol's sparse arm.
SOL_PREFIX = 1254


class Verdict(msgspec.Struct, frozen=True, omit_defaults=True):
    """One kernel on one rank: its refusal, or its degree-N shard against degree 1. `probe` is
    the kernel's per-rank numeric probe, carried through as it was measured."""

    kernel: str
    rank: int
    status: str
    detail: str = ""
    impl: str | None = None
    probe: object = None
    shape: list[int] | None = None
    byte_equal: bool | None = None
    rel_l2_to_degree1: float | None = None
    rel_l2_max: float | None = None


class RankReport(msgspec.Struct, frozen=True):
    rank: int
    device: str
    sm: int
    kernel_site: KernelSite
    kernels: list[Verdict]


class Report(msgspec.Struct, frozen=True):
    degree: int
    device: str
    shape: list[int]
    ok: bool
    verdicts: dict[str, list[str]]
    ranks: list[RankReport]


def _device(torch: Any, kind: str, rank: int) -> DeviceFacts:
    if kind == "cpu":
        return DeviceFacts("cpu", "CPU", 0, "", "", 0)
    major, minor = torch.cuda.get_device_capability(rank)
    return DeviceFacts("cuda", torch.cuda.get_device_name(rank), major * 10 + minor, "", "", rank)


def _agreed(torch: Any, dist: Any, ok: bool, device: Any) -> bool:
    """Every rank must enter a collective together; one rank's refusal skips it for all."""
    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _kernel(
    torch: Any,
    dist: Any,
    name: str,
    rank: int,
    degree: int,
    facts: DeviceFacts,
    config: Any,
    shape: tuple[int, int, int, int],
) -> Verdict:
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    device = torch.device("cuda", rank) if facts.kind == "cuda" else torch.device("cpu")
    status, detail, ready = "skipped: another rank refused this kernel", "", None
    try:
        ready = attention.pinned(name, facts)[0]
    except Exception as exc:
        status = exc.code if isinstance(exc, attention.AttentionRefusal) else type(exc).__name__
        detail = str(exc)[:600]
    if not _agreed(torch, dist, ready is not None, device):
        return Verdict(name, rank, status, detail)
    assert ready is not None
    backend = _AttentionBackendRegistry._backends[ready.member]
    dtype = torch.bfloat16 if facts.kind == "cuda" else torch.float32
    if ready.candidate.dtypes and "bfloat16" not in ready.candidate.dtypes:
        dtype = torch.float16
    seed = torch.Generator(device=device).manual_seed(0x51D3)
    q, k, v = (torch.randn(shape, generator=seed, device=device).to(dtype) for _ in range(3))
    shards = [t.tensor_split(degree, dim=1)[rank].contiguous() for t in (q, k, v)]
    rows = shape[1]
    layout = AttentionLayout(rows, min(SOL_PREFIX, rows // 8), step=4, dense_until_step=4)
    token = attention_sol._SITE.set(attention_sol.Site("dit", "transformer_blocks.2.attn"))
    try:
        with torch.no_grad(), attention_scope(layout), attention_ulysses.observing() as seen:
            got = backend(query=shards[0], key=shards[1], value=shards[2], _parallel_config=config)
            impl = attention.implementations(seen)
            one = backend(query=q, key=k, value=v).tensor_split(degree, dim=1)[rank]
    finally:
        attention_sol._SITE.reset(token)
    difference = float((got.double() - one.double()).norm() / one.double().norm())
    return Verdict(
        name,
        rank,
        "ready" if within(difference, EQUIVALENT_REL_L2) else "not_equivalent",
        impl=impl,
        probe=ready.admission,
        shape=list(shape),
        byte_equal=bool(
            torch.equal(got.contiguous().view(torch.uint8), one.contiguous().view(torch.uint8))
        ),
        rel_l2_to_degree1=difference,
        rel_l2_max=EQUIVALENT_REL_L2,
    )


def _rank(
    rank: int,
    degree: int,
    root: str,
    kind: str,
    kernels: list[str],
    shape: tuple[int, int, int, int],
) -> None:
    kernel_site.activate()
    import torch
    import torch.distributed as dist
    from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig

    from cozy_runtime.internal.parallel.cp import _GroupMesh

    torch.set_num_threads(1)
    if kind == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if kind == "cuda" else "cpu:gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=degree,
    )
    try:
        facts = _device(torch, kind, rank)
        device = torch.device("cuda", rank) if kind == "cuda" else torch.device("cpu")
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(
                ulysses_degree=degree, ulysses_anything=True
            )
        )
        mesh: Any = _GroupMesh(dist.group.WORLD, degree, rank)
        config.setup(rank, degree, device, mesh=mesh)
        names = kernels or [
            candidate.name
            for candidate in attention.BY_NAME.values()
            if not attention._device_admits(candidate, facts)
        ]
        rows = [_kernel(torch, dist, name, rank, degree, facts, config, shape) for name in names]
        report = RankReport(rank, facts.name, facts.sm, kernel_site.current(), rows)
        (Path(root) / f"rank-{rank}.json").write_bytes(msgspec.json.encode(report))
    finally:
        dist.destroy_process_group()


def qualify(
    degree: int, kernels: list[str], kind: str, rows: int = ROWS, heads: int = HEADS
) -> Report:
    """Spawn `degree` ranks, qualify `kernels` on each, and return the merged report."""
    from torch.multiprocessing.spawn import spawn as start

    spawn: Any = start
    unknown = sorted(set(kernels) - set(attention.BY_NAME))
    if unknown:
        raise ValueError(f"unknown attention kernels {unknown}; known {sorted(attention.BY_NAME)}")
    with TemporaryDirectory(prefix="cozy-attention-qualify-") as root:
        spawn(
            _rank,
            args=(degree, root, kind, kernels, (1, rows, heads, 128)),
            nprocs=degree,
            join=True,
        )
        ranks = [
            msgspec.json.decode((Path(root) / f"rank-{rank}.json").read_bytes(), type=RankReport)
            for rank in range(degree)
        ]
    verdicts: dict[str, list[str]] = {}
    for rank in ranks:
        for row in rank.kernels:
            statuses = verdicts.setdefault(row.kernel, [])
            if row.status not in statuses:
                statuses.append(row.status)
    return Report(
        degree,
        kind,
        [1, rows, heads, 128],
        bool(verdicts) and all(statuses == ["ready"] for statuses in verdicts.values()),
        verdicts,
        ranks,
    )
