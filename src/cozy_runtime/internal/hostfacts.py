"""Host facts, measured once, tri-state: present / absent / UNREADABLE.

ONE owner for "what is this machine". The worker's `Register` and the `doctor` verb are
two consumers of the same measurement, not two probes that could disagree — a wedged driver
must read the same to both. Anything that cannot be read WITHOUT a CUDA context is NAMED in
`unreadable` rather than reported as zero, because a wedged driver is not a CPU host, and
every probe is a bounded file read or a bounded subprocess (a canary that hangs is worse
than one that fails).
"""

from __future__ import annotations

import ast
import os
import platform
import re
import subprocess
import sys
import sysconfig
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cozy_runtime.internal.config import sees_no_gpu

TORCH_VERSION_MAX_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class HostFacts:
    gpu_name: str = ""
    gpu_count: int = 0
    """HOW MANY cards this host holds, counted from the driver's own rows — never
    inferred from the presence of a name. A rental is billed by width, and the paid
    width is read back against this number, so a wide pod that reports one card is
    refused as a billing fault (Creator `rental.accelerator_count_mismatch`)."""
    gpu_sm: int = 0
    vram_total_bytes: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    platform: str = ""
    host_ram_total_bytes: int = 0
    vcpu_count: int = 0
    unreadable: tuple[str, ...] = field(default_factory=tuple)
    backend: str = ""
    """`cuda` / `mps` / "" — which accelerator FAMILY is present. Registration reports an
    accelerator PRESENT-BUT-UNQUALIFIED (#446): the torch-free worker can see that a
    device family exists, and only the executor's probe (#430) qualifies what it can DO —
    the worker never reports a capability it did not measure."""

    def state(self, name: str) -> str:
        """`present` / `absent` / `unreadable` for one fact — the tri-state, spelled."""
        if name in self.unreadable:
            return "unreadable"
        return "present" if getattr(self, name, "") else "absent"


def measure(expected_backend: str = "") -> HostFacts:
    """Measure host facts, using a baked CPU profile to distinguish absence from driver failure."""

    if expected_backend not in {"", "cuda", "none"}:
        raise ValueError(f"unsupported expected backend {expected_backend!r}")
    unreadable: list[str] = []
    gpu_name, gpu_sm, vram, driver, backend = "", 0, 0, "", ""
    gpu_count = 0
    if expected_backend == "none" or sees_no_gpu():
        backend = "none"
    elif sys.platform == "darwin" and platform.machine() == "arm64":
        # Apple silicon: the FAMILY is present and that is all a torch-free process can
        # honestly say — present-but-unqualified (#446). The CUDA facts are ABSENT here,
        # not unreadable, and the device facts are the executor's to measure (#430).
        backend = "mps"
    else:
        try:
            rows = [
                tuple(part.strip() for part in line.split(","))
                for line in subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version,compute_cap",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                .stdout.strip()
                .splitlines()
                if line.strip()
            ]
            if not rows:
                raise IndexError("nvidia-smi named no device")
            # ONE ROW PER CARD. The pair the protocol carries is (device_name,
            # device_count), which describes a homogeneous host and nothing else, so a
            # host whose cards disagree is UNREADABLE rather than reported by its first
            # card — a width the renter is billed for must not be a guess.
            if len({row[0] for row in rows}) != 1:
                raise ValueError(f"mixed accelerators: {sorted({row[0] for row in rows})}")
            name, total, drv, cap = rows[0]
            gpu_name, vram, driver = name, int(float(total)) * 1024 * 1024, drv
            gpu_sm = int(float(cap) * 10)
            gpu_count = len(rows)
            backend = "cuda"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            unreadable += ["gpu_name", "gpu_sm", "vram_total_bytes", "driver_version"]
    try:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        pages = 0
        unreadable.append("host_ram_total_bytes")
    cuda_version, cuda_unreadable = _installed_cuda_version()
    if cuda_unreadable:
        unreadable.append("backend_version")
    return HostFacts(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_sm=gpu_sm,
        vram_total_bytes=vram,
        driver_version=driver,
        cuda_version=cuda_version,
        platform=platform.platform()[:120],
        host_ram_total_bytes=pages,
        vcpu_count=os.cpu_count() or 0,
        unreadable=tuple(sorted(unreadable)),
        backend=backend,
    )


def _installed_cuda_version(paths: Mapping[str, str] | None = None) -> tuple[str, bool]:
    """Installed torch toolkit literal and whether that evidence was unreadable.

    The worker must not import torch or create a CUDA context to describe its image. Python's
    own install scheme already locates the distribution, and torch's generated ``version.py``
    records the toolkit it was built against as one top-level literal. Read only that bounded
    file and parse it as syntax; the driver's maximum supported CUDA version is a different
    fact and is never a fallback.
    """

    install_paths = sysconfig.get_paths() if paths is None else paths
    candidates: list[Path] = []
    for key in ("purelib", "platlib"):
        root = install_paths.get(key, "")
        candidate = Path(root) / "torch" / "version.py" if root else None
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)

    documents: list[bytes] = []
    for candidate in candidates:
        try:
            with candidate.open("rb") as handle:
                raw = handle.read(TORCH_VERSION_MAX_BYTES + 1)
        except FileNotFoundError:
            continue
        except OSError:
            return "", True
        if len(raw) > TORCH_VERSION_MAX_BYTES:
            return "", True
        documents.append(raw)
    if not documents:
        return "", True

    versions: list[str] = []
    for raw in documents:
        try:
            tree = ast.parse(raw.decode("utf-8"), filename="torch/version.py")
        except (UnicodeDecodeError, SyntaxError):
            return "", True
        assignments: list[ast.expr] = []
        for node in tree.body:
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "cuda" for target in node.targets
            ):
                value = node.value
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "cuda"
                and node.value is not None
            ):
                value = node.value
            if value is not None:
                assignments.append(value)
        if len(assignments) != 1 or not isinstance(assignments[0], ast.Constant):
            return "", True
        literal = assignments[0].value
        if literal is None:
            versions.append("")  # a well-formed CPU-only torch build: measured absent
        elif isinstance(literal, str) and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", literal):
            versions.append(literal)
        else:
            return "", True
    return (versions[0], False) if len(set(versions)) == 1 else ("", True)
