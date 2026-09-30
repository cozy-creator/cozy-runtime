"""An executor from before boot compile (a published cozy-runtime, 0.18.51 through 0.18.69)
takes the machine's kernels from the image kernel site. Run by `tests/test_kernel_boot.py`
under its own uid in a venv that installs that release: it places the site at entry, as its
executor does, warms H3's ladder while the site lacks SageAttention2, waits for `argv[1]` (the
machine publishing it), and warms again, as its next construction would. Prints one JSON line.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable, Sequence
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import torch

from cozy_runtime.internal.encoding.selection import measure_device


class _OldPlacement(Protocol):
    def document(self) -> dict[str, object]: ...


class _OldKernelSite(Protocol):
    def activate(self) -> _OldPlacement: ...


class _OldProbe(Protocol):
    kernel: str
    status: str
    rel_l2_by_rows: object


class _OldEvidence(Protocol):
    probes: Sequence[_OldProbe]


class _OldAttention(Protocol):
    def warm(self, device: object, preferred: Sequence[str]) -> object: ...
    def evidence(self, served: Iterable[str]) -> _OldEvidence: ...


# This subprocess deliberately imports the published old SDK, not the source
# tree mypy checks. These protocols describe that fixture's historical API only.
kernel_site = cast(_OldKernelSite, import_module("cozy_runtime.internal.kernel_site"))
attention = cast(_OldAttention, import_module("cozy_runtime.internal.attention"))

#: `cozy_runtime.models.minimax_h3.model.DIT_ATTENTION` in these releases
H3 = ("sol-attn", "sageattention", "flash-attn3", "sdpa")


def main() -> None:
    published = Path(sys.argv[1])
    placed = kernel_site.activate()
    device = measure_device(torch, 0)
    before = attention.warm(device, H3)
    published.with_suffix(".waiting").write_text("")
    while not published.exists():
        time.sleep(0.2)
    after = attention.warm(device, H3)
    print(
        json.dumps(
            {
                "runtime": version("cozy-runtime"),
                "site": placed.document(),
                "before": before,
                "after": after,
                "sageattention": version("sageattention"),
                "probes": [
                    {"kernel": row.kernel, "status": row.status, "rel_l2": row.rel_l2_by_rows}
                    for row in attention.evidence({"sageattention"}).probes
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
