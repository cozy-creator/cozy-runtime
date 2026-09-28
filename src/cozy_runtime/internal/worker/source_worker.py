"""One privileged native source operation, launched on an inherited stdin channel.

No credentials or source URLs enter argv, environment, logs, or persistent state.
The parent owns acceptance and memoization; this child owns only bounded native work.
"""

from __future__ import annotations

import base64
import faulthandler
import hashlib
import os
import sys
import threading
import traceback
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill

MAX_COMMAND = 2 << 20
_STDOUT = threading.Lock()


def report(stage: str, position: int, total: int) -> None:
    """One progress line ahead of the final result line; the parent owns throttling."""
    line = canonical_json.encode(
        {"progress": {"stage": stage, "position": position, "total": total}}
    )
    with _STDOUT:
        sys.stdout.buffer.write(line + b"\n")
        sys.stdout.buffer.flush()


def materialize(store: Any, command: dict[str, Any]) -> dict[str, Any]:
    selected = command["selection"]
    members = [tuple(member) for member in selected["members"]]
    produced: dict[str, Any] = store.materialize_source(
        command["native_owner"],
        members,
        allowed_hosts=selected["allowed_hosts"],
        credential_hosts=selected["credential_hosts"],
        credential=command.get("credential", ""),
        allow_local=command.get("allow_local", False),
        progress=lambda present, whole: report("download", present, whole),
    )
    return produced


class ConversionRefusal(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def profiles_of(arguments: dict[str, Any]) -> tuple[str, ...]:
    """The reviewed profiles one conversion composes, in their declared order."""
    single, several = arguments.get("profile", ""), tuple(arguments.get("profiles", ()))
    if bool(single) == bool(several) or len(set(several)) != len(several):
        raise ConversionRefusal(
            "model_source_profiles_invalid", "name one profile or distinct profiles"
        )
    return (single,) if single else several


def execute(command: dict[str, Any]) -> dict[str, Any]:
    store = fill.store(command["store"])
    operation = command["operation"]
    args = command["arguments"]
    if operation == "commit_file":
        from cozy_runtime.internal.worker.commit_files import execute as commit_file

        return commit_file(store, command)
    if operation == "source_files":
        from cozy_runtime.author.sources import SOURCE_FILES_MAX_BYTES
        from cozy_runtime.internal.worker.workspace_byte_outputs import manifest_members

        source = store.tree_root(command["source_owner"])
        manifest = command["view_manifest"]
        if (
            not source
            or not source["complete"]
            or source["released"]
            or source["manifest_digest"] != manifest["digest"]
            or source["manifest_length"] != manifest["length"]
            or source["receipt_digest"] != args["source"]["tensorfs_receipt_digest"]
        ):
            raise ValueError("source view lost its exact native source")
        body = store.manifest(manifest["digest"])["manifest"]
        members = manifest_members(body, SOURCE_FILES_MAX_BYTES)
        size = sum(member.blob.length for member in members)
        if size != command["view_bytes"]:
            raise ValueError("source view content bound changed")
        produced = store.create_tree_root(
            command["view_owner"], manifest["digest"], manifest["length"]
        )
        return {
            "result": {
                "files": {
                    "asset_ref": manifest["digest"],
                    "kind": "tree",
                    "digest": manifest["digest"],
                    "size_bytes": size,
                }
            },
            "native_receipt": base64.b64encode(produced["receipt"]).decode(),
        }
    if command["phase"] == "resolve":
        if operation == "download_huggingface":
            uri = f"hf://{args['repository']}@{args['revision']}"
            carriers, profiles = args.get("carriers", []), args.get("profiles", [])
            files = args.get("files", [])
        elif operation == "download_civitai":
            uri = f"civitai://{args['version']}"
            carriers, profiles = [args["file"]] if args.get("file") else [], []
            files = []
        else:
            raise ValueError("conversion does not resolve a foreign source")
        selection = store.resolve_source(
            uri,
            carriers,
            profiles=profiles,
            files=files,
            credential=command.get("credential", ""),
            huggingface=command.get("huggingface"),
            civitai=command.get("civitai"),
            allow_local=command.get("allow_local", False),
        )
        return {"selection": selection, "implementation": store.implementation_sha256}
    if operation in ("download_huggingface", "download_civitai"):
        materialized = materialize(store, command)
        root = materialized["artifact"]
        receipt = root["receipt"]
        result = {
            "producer_request_id": command["service_id"],
            "output_slot": "source",
            "manifest": {"digest": root["manifest_digest"], "length": root["manifest_length"]},
            "tensorfs_receipt_digest": root["receipt_digest"],
        }
        return {
            "result": result,
            "native_receipt": base64.b64encode(receipt).decode(),
            "transferred_bytes": materialized["transferred_bytes"],
        }
    if operation != "convert_cozytensors":
        raise ValueError("unknown fixed source operation")
    conversion_id = command["service_id"]
    transaction = command["native_owner"]
    observed = store.derived_lookup(transaction)
    if observed.get("state") == "committed":
        receipt = observed["receipt"]
    else:
        # One reviewed profile per slot; several profiles compose one model whose
        # components are the union of theirs.
        slots = [(f"part{index}", profile) for index, profile in enumerate(profiles_of(args))]
        if len(slots) == 1:
            slots = [("model", slots[0][1])]
        checkpoints: list[tuple[str, str, int]] = []
        adopt = command.get("adopt_operation")
        registry = (
            base64.b64decode(command["registry"], validate=True)
            if command.get("registry")
            else None
        )
        retained = store.tree_root(command["source_owner"])
        if retained is None:
            raise ValueError("conversion source has no retained native root")
        total = sum(int(row["length"]) for row in store.walk(retained["manifest_digest"]))
        written = 0
        report("convert", 0, total)
        while True:
            prepared = store.prepare_source_artifact(
                command["source_owner"],
                conversion_id,
                slots,
                checkpoints=checkpoints,
                adopt_from_operation_id=adopt,
                registry=registry,
            )
            adopt = None
            written += int(prepared["converted_bytes"])
            if prepared["complete"]:
                report("convert", total, total)
                break
            # Checkpoint heads carry resumed work; this process's passes carry the rest.
            durable = sum(int(row["bytes"]) for row in prepared["checkpoints"])
            report("convert", min(max(durable, written), total), total)
            checkpoints = [
                (row["slot"], row["head"], row["head_length"]) for row in prepared["checkpoints"]
            ]
        converted = {row["slot"]: row for row in prepared["sources"]}
        if epoch := observed.get("writer_session_id"):
            store.derived_fence(transaction, epoch)
        sources: dict[str, tuple[str, int]] = {}
        targets: dict[str, dict[str, Any]] = {}
        configs: dict[str, dict[str, str]] = {}
        order: list[tuple[str, str]] = []
        assets: dict[str, list[str]] = {}
        for slot, _ in slots:
            part = converted[slot]
            sources[slot] = (part["manifest_digest"], part["manifest_length"])
            header = fill.tensorfs_module().parse_header(
                store.manifest(part["manifest_digest"])["header"]
            )
            for name, tensors in header["components"].items():
                if name in targets:
                    raise ConversionRefusal(
                        "model_source_components_overlap",
                        f"source profiles both produce component {name}",
                    )
                targets[name] = {"source": slot, "source_component": name, "drop": [], "add": {}}
                order.extend((name, key) for key in tensors)
            for name in header["configs"]:
                if name in configs:
                    raise ConversionRefusal(
                        "model_source_configs_overlap",
                        f"source profiles both carry config {name}",
                    )
                configs[name] = {"kind": "copy", "source": slot, "source_config": name}
            if header["assets"]:
                assets[slot] = sorted(header["assets"])
        writer = store.begin_derived(
            transaction,
            int(command["writer_epoch"]),
            sources,
            targets,
            configs,
            order,
            0,
            work_fingerprint=command["computation_digest"],
        )
        receipt = writer.commit()
        for slot, names in assets.items():
            inherited = store.inherit_model_assets(
                receipt["manifest"]["sha256"],
                receipt["manifest"]["length"],
                sources[slot][0],
                sources[slot][1],
                names,
            )
            receipt["manifest"] = {
                "sha256": str(inherited["manifest_id"]).removeprefix("sha256:"),
                "length": inherited["manifest_length"],
            }
    native_receipt = canonical_json.encode(receipt)
    result = {
        "producer_request_id": command["service_id"],
        "output_slot": "model",
        "manifest": {
            "digest": "sha256:" + receipt["manifest"]["sha256"],
            "length": receipt["manifest"]["length"],
        },
        "tensorfs_receipt_digest": "sha256:" + hashlib.sha256(native_receipt).hexdigest(),
    }
    # The genuine derived result now holds every inherited model byte independently.
    store.release_model_source(conversion_id)
    return {
        "result": result,
        "native_receipt": base64.b64encode(native_receipt).decode(),
    }


def main() -> None:
    # A native crash leaves every thread's stack in the stderr tail the parent keeps.
    faulthandler.enable()
    try:
        command = canonical_json.decode(sys.stdin.buffer.read(MAX_COMMAND + 1))
        if not isinstance(command, dict):
            raise ValueError("source command must be an object")
        parent = command.get("parent_pid")
        if type(parent) is not int or parent <= 0:
            raise ValueError("native source execution needs its process owner")

        def parent_lifetime() -> None:
            tick = threading.Event()
            while os.getppid() == parent:
                tick.wait(0.25)
            os._exit(70)

        threading.Thread(target=parent_lifetime, daemon=True, name="source-parent").start()
        result = {"ok": True, **execute(command)}
    except Exception as exc:
        # Only the fixed code is the result. The parent redacts this stderr tail into the
        # failure's diagnosis, never into the result.
        traceback.print_exc()
        code = getattr(exc, "code", "native_source_failed")
        result = {"ok": False, "code": code if isinstance(code, str) else "native_source_failed"}
    body = canonical_json.encode(result)
    if len(body) > MAX_COMMAND:
        body = canonical_json.encode({"ok": False, "code": "source_result_bound"})
    with _STDOUT:
        sys.stdout.buffer.write(body + b"\n")
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
