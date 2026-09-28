"""Create the tiny native H3 checkpoints the CPU content/forward qualification reads."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import tensorfs
import torch
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import Config
from cozy_runtime.internal import canonical
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.weights_sink import weights_transaction_id

PACKAGE = Path(__file__).parent / "h3_group_warm"
sys.path[:0] = [str(PACKAGE), str(Path(__file__).parents[1])]
from h3_group_warm import (  # type: ignore[import-not-found]  # noqa: E402
    CONFIG,
    Base,
    BasePipe,
    OverlayPipe,
)


def checkpoint(
    store_root: Path, name: str, components: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    store = tensorfs.Store.ensure(str(store_root))
    rows = {
        (owner, key): tensor.detach().cpu().contiguous()
        for owner, module in components.items()
        for key, tensor in module.state_dict().items()
    }
    dtypes = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int64: "i64"}
    total = sum(t.numel() * t.element_size() for t in rows.values())
    writer = store.begin_derived(
        weights_transaction_id("h3a093", name, "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets={
                owner: Target(
                    add={
                        key: Tensor(
                            dtypes[tensor.dtype],
                            tuple(tensor.shape),
                            SPEC_PLAIN,
                            {"value": Part(dtypes[tensor.dtype], tuple(tensor.shape))},
                        )
                        for (component, key), tensor in rows.items()
                        if component == owner
                    }
                )
                for owner in components
            },
            configs={"model": NativeConfig("add")},
            order=tuple(rows),
        ).native_arguments(total + 1_000_000),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for (owner, key), tensor in rows.items():
        writer.add_part(owner, key, "value", io.BytesIO(tensor.view(torch.uint8).numpy().tobytes()))
    writer.add_config("model", io.BytesIO(canonical.write(config)))
    receipt = writer.commit()
    manifest = "sha256:" + receipt["manifest"]["sha256"]
    length = int(receipt["manifest"]["length"])
    operation = store.begin_operation("publish-" + name, "test", name)
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1", "main", manifest, length)
    return {"name": name, "manifest": manifest, "length": length, "weight_bytes": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--degree", type=int, choices=(1, *Base.__sequence_parallel__), default=4)
    parser.add_argument("--attention", choices=("sdpa", "flash-attn3", "sol-attn"), default="sdpa")
    parser.add_argument(
        "--allocator",
        choices=("expandable_segments:True", "expandable_segments:False"),
        default="expandable_segments:True",
    )
    parser.add_argument("--prepare-only", action="store_true")  # the only mode
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o755)
    torch.set_num_threads(1)
    torch.manual_seed(231)
    base = BasePipe(Config({"dit": CONFIG}))
    overlay = OverlayPipe(Config({}))
    store = root / "store"
    receipts = [
        checkpoint(store, "h3-group-warm-base", base.components, {"dit": CONFIG}),
        checkpoint(store, "h3-group-warm-overlay", overlay.components, {}),
    ]
    (root / "checkpoints.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(
        json.dumps(
            {"checkpoint_bytes": sum(r["weight_bytes"] for r in receipts), "receipts": receipts}
        ),
        flush=True,
    )
    del base, overlay


if __name__ == "__main__":
    main()
