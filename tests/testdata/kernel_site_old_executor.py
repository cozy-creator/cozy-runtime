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
from importlib.metadata import version
from pathlib import Path

import torch

from cozy_runtime.internal import attention, kernel_site
from cozy_runtime.internal.encoding.selection import measure_device

#: `cozy_runtime.models.minimax_h3.model.DIT_ATTENTION` in these releases
H3 = ("sol-attn", "sageattention", "flash-attn3", "sdpa")


def main() -> None:
    published = Path(sys.argv[1])
    # `activate`, `warm` and `evidence(...).probes` are those releases' API, not this tree's.
    warm, evidence = getattr(attention, "warm"), getattr(attention, "evidence")  # noqa: B009
    placed = getattr(kernel_site, "activate")()  # noqa: B009
    device = measure_device(torch, 0)
    before = warm(device, H3)
    published.with_suffix(".waiting").write_text("")
    while not published.exists():
        time.sleep(0.2)
    after = warm(device, H3)
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
                    for row in evidence({"sageattention"}).probes
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
