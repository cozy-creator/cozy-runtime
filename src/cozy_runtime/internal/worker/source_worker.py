"""One privileged native source operation, launched on an inherited stdin channel.

No credentials or source URLs enter argv, environment, logs, or persistent state.
The parent owns acceptance and memoization; this child owns only bounded native work.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
import threading
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill

MAX_COMMAND = 2 << 20


def execute(command: dict[str, Any]) -> dict[str, Any]:
    store = fill.tensorfs_module().Store.open(command["store"])
    operation = command["operation"]
    args = command["arguments"]
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
        body = bytes(store.manifest(manifest["digest"])["manifest"])
        members = manifest_members(body, SOURCE_FILES_MAX_BYTES)
        size = sum(member["blob"]["length"] for member in members)
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
        selected = command["selection"]
        produced = store.materialize_source(
            command["native_owner"],
            [tuple(member) for member in selected["members"]],
            allowed_hosts=selected["allowed_hosts"],
            credential_hosts=selected["credential_hosts"],
            credential=command.get("credential", ""),
            allow_local=command.get("allow_local", False),
        )
        root = produced["artifact"]
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
            "transferred_bytes": produced["transferred_bytes"],
        }
    if operation != "convert_cozytensors":
        raise ValueError("unknown fixed source operation")
    conversion_id = command["service_id"]
    transaction = command["native_owner"]
    observed = store.derived_lookup(transaction)
    if observed.get("state") == "committed":
        receipt = observed["receipt"]
    else:
        checkpoints: list[tuple[str, str, int]] = []
        adopt = command.get("adopt_operation")
        registry = (
            base64.b64decode(command["registry"], validate=True)
            if command.get("registry")
            else None
        )
        while True:
            prepared = store.prepare_source_artifact(
                command["source_owner"],
                conversion_id,
                [("model", args["profile"])],
                checkpoints=checkpoints,
                adopt_from_operation_id=adopt,
                registry=registry,
            )
            adopt = None
            if prepared["complete"]:
                break
            checkpoints = [
                (row["slot"], row["head"], row["head_length"]) for row in prepared["checkpoints"]
            ]
        converted = prepared["sources"][0]
        if epoch := observed.get("writer_session_id"):
            store.derived_fence(transaction, epoch)
        header = fill.tensorfs_module().parse_header(
            store.manifest(converted["manifest_digest"])["header"]
        )
        if header["assets"]:
            raise ValueError("metadata-only model adoption needs explicit asset inheritance")
        components = header["components"]
        targets = {
            name: {"source": "converted", "source_component": name, "drop": [], "add": {}}
            for name in components
        }
        configs = {
            name: {"kind": "copy", "source": "converted", "source_config": name}
            for name in header["configs"]
        }
        order = [(name, key) for name, tensors in components.items() for key in tensors]
        writer = store.begin_derived(
            transaction,
            int(command["writer_epoch"]),
            {"converted": (converted["manifest_digest"], converted["manifest_length"])},
            targets,
            configs,
            order,
            0,
            work_fingerprint=command["computation_digest"],
        )
        receipt = writer.commit()
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
        # Exception text may include an input URL. Only the fixed native code leaves.
        code = getattr(exc, "code", "native_source_failed")
        result = {"ok": False, "code": code if isinstance(code, str) else "native_source_failed"}
    body = canonical_json.encode(result)
    if len(body) > MAX_COMMAND:
        body = canonical_json.encode({"ok": False, "code": "source_result_bound"})
    sys.stdout.buffer.write(body)


if __name__ == "__main__":
    main()
