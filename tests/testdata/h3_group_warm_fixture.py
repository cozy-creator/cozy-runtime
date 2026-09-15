"""Create a tiny checkpoint and run the real Worker prepare/warm under the pod UID.

No fake CUDA transport or accountancy is installed. This local RecordOwner pins
its own placement to the requested devices; all worker/executor behavior is real.
Run as root on an explicitly owned GPU host. --prepare-only uses CPU and no worker.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import tensorfs
import torch
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import Config
from cozy_runtime.internal import canonical, child_env, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.local import (
    LOCAL_PLACEMENT,
    LocalRequest,
    development_source_digest,
    run_slice,
)
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker.session import WorkerOptions

PACKAGE = Path(__file__).parent / "h3_group_warm"
sys.path.insert(0, str(PACKAGE))
from h3_group_warm import (  # type: ignore[import-not-found]  # noqa: E402
    CONFIG,
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


def preflight_uid(root: Path) -> None:
    """Prove actual UID65533 can traverse the interpreter and immutable fixture paths."""
    import cozy_runtime

    paths = [sys.executable, str(PACKAGE), str(Path(__file__).resolve()), cozy_runtime.__file__]
    code = (
        "import json,os,sys; from pathlib import Path; "
        "rows=[{'path':p,'readable':os.access(p,os.R_OK),"
        "'parents_searchable':all(os.access(x,os.X_OK) for x in Path(p).parents)} "
        "for p in sys.argv[1:]]; print(json.dumps({'uid':os.getuid(),'paths':rows})); "
        "sys.exit(0 if all(r['readable'] and r['parents_searchable'] for r in rows) else 1)"
    )
    checked = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, *paths],
        user=65533,
        group=65533,
        extra_groups=[],
        env={},
        text=True,
        capture_output=True,
        check=False,
    )
    (root / "uid-preflight.stdout").write_text(checked.stdout)
    (root / "uid-preflight.stderr").write_text(checked.stderr)
    if checked.returncode:
        raise RuntimeError(
            "fixture/interpreter paths are not readable and traversable by UID65533; "
            "stage them under a traversable owned test directory before GPU execution"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--degree", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--attention", choices=("sdpa", "flash-attn3", "sol-attn"), default="sdpa")
    parser.add_argument(
        "--allocator",
        choices=("expandable_segments:True", "expandable_segments:False"),
        default="expandable_segments:True",
    )
    parser.add_argument("--prepare-only", action="store_true")
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
    if args.prepare_only:
        return
    if os.geteuid() != 0:
        parser.error("actual pod executor UID qualification requires root")
    # TensorFS owns its lease/recovery path policy for separate reader UIDs.
    tensorfs.Store.ensure(str(store)).prepare_readers()
    preflight_uid(root)
    config = RuntimeConfig(
        cozy_home=root / "home",
        credentials=Credentials(),
        child_base_env=tuple(
            sorted((key, value) for key, value in os.environ.items() if not child_env.erased(key))
        ),
    )
    found = discover(PACKAGE)
    body = package_interface.build(found)
    digest = development_source_digest(PACKAGE)

    def pin_devices(owner: Any, _worker: Any) -> None:
        original = owner._desired_state

        def desired() -> Any:
            directive = original()
            directive.placement_set.device_pins.add(
                placement_id=LOCAL_PLACEMENT, device_ordinals=range(args.degree)
            )
            return directive

        owner._desired_state = desired

    with contextlib.chdir(PACKAGE):
        outcome = run_slice(
            config,
            LocalRequest("generate", {"value": 1}, outputs=(), request_id="h3a093"),
            workspace=root / "run",
            package_code_digest=digest,
            package_interface_document=body,
            project=PACKAGE,
            development_tensorfs_root=store,
            options=WorkerOptions(
                root=root / "run" / "worker",
                release_id=digest,
                devices=",".join(str(i) for i in range(args.degree)),
                accelerator_backend="cuda",
                executor_uid=65533,
                executor_gid=65533,
                alloc_conf=args.allocator,
                attention_pin="base/fl2va_dit=" + args.attention,
            ),
            delete_workspace=False,
            record_owner_hook=pin_devices,
        )
    report = {
        "status": outcome.status,
        "ended": outcome.ended,
        "error": outcome.error,
        "exit_code": outcome.exit_code,
        "faults": outcome.faults,
        "result": outcome.result,
        "accepted": outcome.accepted,
        "executions": outcome.executions,
    }
    (root / "outcome.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if outcome.exit_code or outcome.status != "OUTCOME_STATUS_SUCCEEDED":
        raise SystemExit(outcome.exit_code or 1)


if __name__ == "__main__":
    main()
