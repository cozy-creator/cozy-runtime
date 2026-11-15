"""Single-GPU storage/event check; deliberately not a native multi-GPU transport proof."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPU already has compute processes: {active}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import torch

    from cozy_runtime.internal.parallel import ulysses

    torch.cuda.init()
    transport = torch.cuda.Stream()
    cases = []
    for dtype in (torch.bfloat16, torch.float16):
        for rows in (65, 4103):
            for iteration in range(3):
                values = torch.empty((3, rows, 1, 2, 128), device="cuda", dtype=dtype)
                packer = ulysses._Packer(values, False)
                allocated = torch.cuda.Event()
                allocated.record()
                ready = torch.cuda.Event()
                with torch.cuda.stream(transport):
                    transport.wait_event(allocated)
                    # Delay the producer enough that missing inbound ordering can expose stale
                    # data. No synchronization occurs between this write and consumer launch.
                    torch.cuda._sleep(1_000_000)
                    values.fill_(iteration + 1)
                    ready.record()

                def wait(ready=ready) -> None:
                    torch.cuda.current_stream().wait_event(ready)

                arriving = ulysses._Arriving(wait, values, None, None, packer.mark())
                receive_ref = weakref.ref(values)
                pointer = values.untyped_storage().data_ptr()
                operands = ulysses._unpack(packer, arriving)
                del arriving, values
                gc.collect()
                assert receive_ref() is not None
                assert all(x.untyped_storage().data_ptr() == pointer for x in operands)
                consumed = [x.clone() for x in operands]
                del operands
                # Encourage reuse while the earlier reads are still queued on the caller stream.
                churn = torch.empty((3, rows, 1, 2, 128), device="cuda", dtype=dtype)
                churn.zero_()
                torch.cuda.synchronize()
                assert all(torch.all(x == iteration + 1).item() for x in consumed)
                packer.settle()
                cases.append(dict(dtype=str(dtype), rows=rows, iteration=iteration, ordered=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                scope=(
                    "Single GPU, stock Torch storage, synthetic transport event; "
                    "no NCCL/Varena qualification"
                ),
                pid=os.getpid(),
                torch=torch.__version__,
                gpu=str(torch.cuda.get_device_properties(0)),
                cases=cases,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"PASS {len(cases)} local CUDA receive-view/event cases")


if __name__ == "__main__":
    main()
