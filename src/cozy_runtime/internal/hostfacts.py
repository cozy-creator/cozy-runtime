"""Host facts, measured once, tri-state: present / absent / UNREADABLE.

ONE owner for "what is this machine". The supervisor's `Register` and the `doctor` verb are
two consumers of the same measurement, not two probes that could disagree — a wedged driver
must read the same to both. Anything that cannot be read WITHOUT a CUDA context is NAMED in
`unreadable` rather than reported as zero, because a wedged driver is not a CPU host, and
every probe is a bounded file read or a bounded subprocess (a canary that hangs is worse
than one that fails).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HostFacts:
    gpu_name: str = ""
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
    accelerator PRESENT-BUT-UNQUALIFIED (#446): the torch-free supervisor can see that a
    device family exists, and only the executor's probe (#430) qualifies what it can DO —
    the supervisor never reports a capability it did not measure."""

    @property
    def gpu_count(self) -> int:
        return 1 if self.gpu_name else 0

    def state(self, name: str) -> str:
        """`present` / `absent` / `unreadable` for one fact — the tri-state, spelled."""
        if name in self.unreadable:
            return "unreadable"
        return "present" if getattr(self, name, "") else "absent"


def measure() -> HostFacts:
    unreadable: list[str] = []
    gpu_name, gpu_sm, vram, driver, backend = "", 0, 0, "", ""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        # Apple silicon: the FAMILY is present and that is all a torch-free process can
        # honestly say — present-but-unqualified (#446). The CUDA facts are ABSENT here,
        # not unreadable, and the device facts are the executor's to measure (#430).
        backend = "mps"
    else:
        try:
            out = (
                subprocess.run(
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
                .splitlines()[0]
            )
            name, total, drv, cap = (part.strip() for part in out.split(","))
            gpu_name, vram, driver = name, int(float(total)) * 1024 * 1024, drv
            gpu_sm = int(float(cap) * 10)
            backend = "cuda"
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            unreadable += ["gpu_name", "gpu_sm", "vram_total_bytes", "driver_version"]
    try:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        pages = 0
        unreadable.append("host_ram_total_bytes")
    return HostFacts(
        gpu_name=gpu_name,
        gpu_sm=gpu_sm,
        vram_total_bytes=vram,
        driver_version=driver,
        cuda_version=_cuda_version(),
        platform=platform.platform()[:120],
        host_ram_total_bytes=pages,
        vcpu_count=os.cpu_count() or 0,
        unreadable=tuple(sorted(unreadable)),
        backend=backend,
    )


def _cuda_version() -> str:
    """The driver's CUDA version, read WITHOUT importing torch or creating a context."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query", "--display=COMPUTE"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in out.splitlines():
        if "CUDA Version" in line:
            return line.split(":")[-1].strip()
    return ""
