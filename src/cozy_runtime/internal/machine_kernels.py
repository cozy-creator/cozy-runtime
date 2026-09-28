"""The machine's attention kernels, compiled at worker boot on the cards present, best first,
while the first weights download.

The worker compiles, for each distinct card architecture, the artifacts of `LADDER` the card
admits, in its order, through the compile manager (`kernel_compile.in_order`): one build at a
time, each as wide as the machine measures, niced. They build into the worker's own store
namespace, which every executor on the machine reads: a newer executor finds each one ready,
or building and claimed, and never builds it twice; an older one takes it from the image
kernel site (`kernel_site`) at its next construction. The builders are the worker's children
and outlive any executor. A request never waits on any of it: selection serves the best
kernel ready at the time and takes a better one at a later construction or request.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cozy_runtime.internal import (
    accel,
    attention,
    jit_cache,
    kernel_cache,
    kernel_compile,
    kernel_site,
    kernel_sources,
)
from cozy_runtime.internal.encoding import DeviceFacts

#: The best attention first, as the installed models rank it: H3's DiT ladder (Sol,
#: SageAttention2, FA3 BF16; SDPA needs no compile). Models without a preference use the
#: ranked table, whose only compiled kernel is FA3.
LADDER: tuple[str, ...] = ("sol-attn", "sageattention", "flash-attn3")
#: The trees an executor before boot compile imports by distribution.
SITE_ARTIFACTS = frozenset({kernel_sources.PYTHON, "sageattention"})


@dataclass
class MachineKernels:
    store: kernel_cache.Store
    #: (kernel, job) in build order
    jobs: Sequence[tuple[str, kernel_compile.Job]]
    #: the image kernel site, and why it is not written (empty when it is)
    site_root: Path = kernel_site.ROOT
    site: str = ""
    thread: threading.Thread | None = None
    published: list[str] = field(default_factory=list)

    def line(self) -> str:
        """Each boot kernel's state now, for the warm's phase row."""
        states = [
            f"{name} {kernel_compile.status(self.store, job).line()}" for name, job in self.jobs
        ]
        site = f"; image kernel site not written: {self.site}" if self.site else ""
        return "machine: " + ", ".join(states) + site if states else ""

    def _done(self, job: kernel_compile.Job, state: kernel_compile.State) -> None:
        if state.state != "ready" or self.site or job.key.kernel not in SITE_ARTIFACTS:
            return
        try:
            kernel_site.publish(Path(state.path), job.key, self.site_root)
        except OSError as exc:
            self.site = f"publishing {job.key.kernel} failed: {exc}"
            return
        self.published.append(job.key.kernel)


def boot(
    kernel_root: Path | None,
    devices: str,
    environment: Mapping[str, str],
    site_root: Path = kernel_site.ROOT,
) -> MachineKernels | None:
    """Start compiling this machine's ladder for its cards; returns at once. None when there
    is no store, no NVIDIA card or no kernel source to build from."""
    cards = {
        sm: device
        for device in reversed([d.strip() for d in devices.split(",") if d.strip()])
        if (sm := accel.device_capability(device))
    }
    if kernel_root is None or not cards:
        return None
    jobs: dict[str, tuple[str, kernel_compile.Job]] = {}
    for sm, device in sorted(cards.items()):
        facts = DeviceFacts("cuda", "", sm, "", "")
        for name in LADDER:
            candidate = attention.BY_NAME[name]
            if attention._device_admits(candidate, facts):
                continue
            try:
                job = kernel_sources.job(candidate.artifact, sm)
            except kernel_sources.Absent:
                continue
            # Upstream's setup.py builds for the cards it sees: this one.
            job = dataclasses.replace(job, env={"CUDA_VISIBLE_DEVICES": device})
            jobs.setdefault(job.key.digest, (name, job))
    if not jobs:
        return None
    try:
        jit_cache.machine(kernel_root)
        store = kernel_cache.Store(kernel_cache.namespace(kernel_root, os.geteuid()))
        kernel_compile.machine_slot(store.own, create=True)
    except (OSError, jit_cache.JITCacheRefusal):
        return None  # executors then compile their own, as without a machine build
    machine = MachineKernels(
        store, list(jobs.values()), site_root, kernel_site.open_site(site_root)
    )
    builders = {**environment, "TMPDIR": tempfile.gettempdir()}
    machine.thread = kernel_compile.in_order(
        store, [job for _, job in machine.jobs], builders, machine._done
    )
    return machine
