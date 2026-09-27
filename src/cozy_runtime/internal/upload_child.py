"""One bounded native step of an upload, launched on an inherited stdin channel.

A step resolves the provider selection, reads member headers, or runs one pass: land a
window of source members and convert what they unlock within the admitted budget. The
pass that completes the conversion composes the published model. The parent owns
admission, publication and eviction; no credential or URL leaves this process.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

from tensorfs.derived import Config, Derivation, Source, SourceInspection, Target

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.internal.worker.upload_plan import (
    UploadRefusal,
    recipe_of,
    source_uri,
)
from cozy_runtime.models.qwen_image21.ingestion import preparation

MAX_COMMAND = 2 << 20
# A predecessor that is absent, complete, live, or already adopted has no prefix to give.
_NOT_ADOPTABLE = frozenset(
    {
        "ROOT_ABSENT",
        "TRANSACTION_CLOSED",
        "TRANSACTION_CONFLICT",
        "STORE_BUSY",
        "DURABILITY_UNPROVEN",
    }
)


def _endpoints(command: dict[str, Any]) -> dict[str, Any]:
    return {
        "credential": command.get("credential", ""),
        "huggingface": command.get("huggingface"),
        "civitai": command.get("civitai"),
        "allow_local": command.get("allow_local", False),
    }


#: The document a Hugging Face Diffusers repository names its pipeline components in.
DIFFUSERS_INDEXES = ("model_index.json", "modular_model_index.json")


def diffusers_configs(index: bytes) -> dict[str, str]:
    """Each Diffusers component's construction config file, by component name.

    A model is built from its folder's ``config.json`` and a scheduler from its
    ``scheduler_config.json``; tokenizers and processors are assets, not constructors.
    Classic indexes map a component to ``[library, class]``, modular ones add a spec whose
    ``subfolder`` and ``type_hint`` name the folder and class.
    """
    document = json.loads(index)
    found: dict[str, str] = {}
    for name, entry in document.items() if isinstance(document, dict) else ():
        if name.startswith("_") or not isinstance(entry, list) or len(entry) < 2:
            continue
        kind = entry[1]
        spec = entry[2] if len(entry) > 2 and isinstance(entry[2], dict) else {}
        hint = spec.get("type_hint")
        if not isinstance(kind, str) and isinstance(hint, list) and len(hint) == 2:
            kind = hint[1]
        folder = spec.get("subfolder") or name
        if not isinstance(kind, str) or not isinstance(folder, str):
            continue
        if kind.endswith("Scheduler"):
            found[name] = f"{folder}/scheduler_config.json"
        elif not any(word in kind for word in ("Tokenizer", "Processor", "FeatureExtractor")):
            found[name] = f"{folder}/config.json"
    return found


def _merge(selection: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Small metadata rides in the same pin as the carriers, never as a carrier."""
    names = {row[0] for row in selection["members"]}
    selection["members"] = sorted(
        [*selection["members"], *(row for row in metadata["members"] if row[0] not in names)]
    )
    for key in ("allowed_hosts", "credential_hosts"):
        selection[key] = sorted({*selection[key], *metadata[key]})


def _pinned(store: Any, uri: str, names: list[str], command: dict[str, Any]) -> dict[str, Any]:
    """The files of ``names`` the pinned revision holds; one it lacks is not selected."""
    try:
        return dict(store.resolve_source(uri, files=names, **_endpoints(command)))
    except Exception as exc:
        if getattr(exc, "code", None) != "MISSING_FIELD":
            raise
    found: dict[str, Any] = {"members": [], "allowed_hosts": [], "credential_hosts": []}
    for name in names if len(names) > 1 else ():
        _merge(found, _pinned(store, uri, [name], command))
    return found


def _diffusers_members(store: Any, uri: str, command: dict[str, Any]) -> dict[str, Any]:
    """A Diffusers repository's index and each component's config, in the same pin."""
    for index in DIFFUSERS_INDEXES:
        pinned = _pinned(store, uri, [index], command)
        if pinned["members"]:
            break
    else:
        return pinned
    member = tuple(pinned["members"][0])
    ((_, body),) = fill.tensorfs_module().read_source_heads(uri, [member], **_endpoints(command))
    return _pinned(store, uri, [index, *sorted(set(diffusers_configs(body).values()))], command)


def resolve(store: Any, command: dict[str, Any]) -> dict[str, Any]:
    operation, arguments = command["operation"], command["arguments"]
    recipe = recipe_of(arguments, operation)
    uri = source_uri(operation, arguments)
    if operation == "upload_civitai":
        carriers = [arguments["file"]] if arguments["file"] else []
        selection = store.resolve_source(uri, carriers, **_endpoints(command))
    elif recipe is not None:
        selection = store.resolve_source(uri, list(recipe.carriers), **_endpoints(command))
    elif arguments["carriers"]:
        selection = store.resolve_source(uri, list(arguments["carriers"]), **_endpoints(command))
    else:
        selection = store.resolve_source(
            uri, profiles=list(arguments["profiles"]), **_endpoints(command)
        )
    if recipe is not None:
        # Reviewed metadata rides in the same pin; its bytes are named by the recipe.
        _merge(
            selection,
            store.resolve_source(uri, files=sorted(recipe.metadata), **_endpoints(command)),
        )
    elif operation == "upload_huggingface":
        # A Diffusers repository's components are constructed from its own configs.
        _merge(selection, _diffusers_members(store, uri, command))
    return {"selection": selection}


def heads(command: dict[str, Any]) -> dict[str, Any]:
    tensorfs = fill.tensorfs_module()
    rows = tensorfs.read_source_heads(
        source_uri(command["operation"], command["arguments"]),
        [tuple(row) for row in command["members"]],
        **_endpoints(command),
    )
    body = canonical_json.encode({member: base64.b64encode(head).decode() for member, head in rows})
    path = Path(command["heads_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged.write(body)
        staged.flush()
        os.fsync(staged.fileno())
    os.replace(staged.name, path)
    return {"heads": "sha256:" + hashlib.sha256(body).hexdigest()}


def _materialize(store: Any, owner: str, rows: list[Any], command: dict[str, Any]) -> None:
    selection = command["selection"]
    store.materialize_source(
        owner,
        [tuple(row) for row in rows],
        allowed_hosts=selection["allowed_hosts"],
        credential_hosts=selection["credential_hosts"],
        credential=command.get("credential", ""),
        allow_local=command.get("allow_local", False),
    )


def advance(store: Any, command: dict[str, Any]) -> dict[str, Any]:
    tensorfs = fill.tensorfs_module()
    members = {row[0]: row for row in command["selection"]["members"]}
    for name in command["window"]:
        owner = command["owners"][name]
        for predecessor in command["adopt"].get(name, ()):
            # A stopped upload of this conversion keeps its verified prefix for this one.
            try:
                store.adopt_source_progress(predecessor, owner)
            except Exception as exc:
                if getattr(exc, "code", None) not in _NOT_ADOPTABLE:
                    raise
        _materialize(store, owner, [members[name]], command)
    if command.get("metadata"):
        _materialize(store, command["metadata_owner"], command["metadata"], command)
    recorded = canonical_json.decode(Path(command["heads_path"]).read_bytes())
    registry = (
        base64.b64decode(command["registry"], validate=True) if command.get("registry") else None
    )
    facts = tensorfs.prepare_selected_source(
        store,
        command["conversion"],
        command["source_selection_digest"],
        [tuple(slot) for slot in command["slots"]],
        [
            (name, members[name][1], members[name][2], base64.b64decode(recorded[name]))
            for name in command["carriers"]
        ],
        registry=registry,
        write_budget_bytes=command["budget"],
    )
    result: dict[str, Any] = {"facts": facts}
    if facts["complete"]:
        result["model"] = compose(store, command, facts["sources"])
    return result


def _header(store: Any, manifest: str) -> dict[str, Any]:
    return dict(fill.tensorfs_module().parse_header(store.manifest(manifest)["header"]))


def compose(store: Any, command: dict[str, Any], prepared: list[dict[str, Any]]) -> dict[str, Any]:
    """The published model: the prepared profiles' union, or the recipe's derivation."""
    transaction = command["native_owner"]
    observed = store.derived_lookup(transaction)
    if observed.get("state") != "committed":
        if epoch := observed.get("writer_session_id"):
            store.derived_fence(transaction, epoch)
        writer = (
            _recipe_writer(store, command, prepared)
            if command.get("recipe")
            else _union_writer(store, command, prepared)
        )
        receipt = writer.commit()
    else:
        receipt = observed["receipt"]
    manifest = receipt["manifest"]
    if not command.get("recipe"):
        for row in prepared:
            names = sorted(_header(store, row["manifest_digest"])["assets"])
            if names:
                inherited = store.inherit_model_assets(
                    "sha256:" + manifest["sha256"],
                    manifest["length"],
                    row["manifest_digest"],
                    row["manifest_length"],
                    names,
                )
                manifest = {
                    "sha256": inherited["manifest_id"].removeprefix("sha256:"),
                    "length": inherited["manifest_length"],
                }
    return {"digest": "sha256:" + manifest["sha256"], "length": manifest["length"]}


def _added_configs(store: Any, command: dict[str, Any]) -> dict[str, bytes]:
    """The Diffusers configs landed beside the carriers, named by their component."""
    files = {
        name: bytes(store.document(digest.removeprefix("sha256:"), size))
        for name, digest, size, _ in command.get("metadata", [])
    }
    index = next((files[name] for name in DIFFUSERS_INDEXES if name in files), None)
    if index is None:
        return {}
    return {name: files[path] for name, path in diffusers_configs(index).items() if path in files}


def _union_writer(store: Any, command: dict[str, Any], prepared: list[dict[str, Any]]) -> Any:
    sources: dict[str, tuple[str, int]] = {}
    targets: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    for row in sorted(prepared, key=lambda row: row["slot"]):
        slot = row["slot"]
        sources[slot] = (row["manifest_digest"], row["manifest_length"])
        header = _header(store, row["manifest_digest"])
        for name, tensors in header["components"].items():
            if name in targets:
                raise UploadRefusal(
                    "model_source_components_overlap", f"profiles both produce component {name}"
                )
            targets[name] = {"source": slot, "source_component": name, "drop": [], "add": {}}
            order.extend((name, key) for key in tensors)
        for name in header["configs"]:
            if name in configs:
                raise UploadRefusal(
                    "model_source_configs_overlap", f"profiles both carry config {name}"
                )
            configs[name] = {"kind": "copy", "source": slot, "source_config": name}
    added = _added_configs(store, command)
    for name in added:
        if name in configs:
            raise UploadRefusal("model_source_configs_overlap", f"config {name} is carried twice")
        configs[name] = {"kind": "add"}
    writer = store.begin_derived(
        command["native_owner"],
        int(command["writer_epoch"]),
        sources,
        targets,
        configs,
        order,
        sum(len(body) for body in added.values()),
        work_fingerprint=command["computation_digest"],
    )
    for name, body in sorted(added.items()):
        writer.add_config(name, io.BytesIO(body))
    return writer


def _recipe_writer(store: Any, command: dict[str, Any], prepared: list[dict[str, Any]]) -> Any:
    (row,) = prepared
    manifest, length = row["manifest_digest"], row["manifest_length"]
    components = sorted(_header(store, manifest)["components"])
    inspection = SourceInspection.from_native(
        store.inspect_derived_source(manifest, length, components, [])
    )
    with tempfile.TemporaryDirectory(dir=Path(command["store"]) / "tmp") as scratch:
        root = Path(scratch)
        for name, digest, size, _ in command["metadata"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(store.document(digest.removeprefix("sha256:"), size)))
        config, order, files = preparation(inspection, root)
    body = canonical_json.encode(config)
    arguments = Derivation(
        sources={"source": Source(manifest, length)},
        targets={name: Target(source="source", source_component=name) for name in components},
        configs={"model": Config("add")},
        order=order,
        files=files,
    ).native_arguments(len(body) + sum(len(data) for data in files.values()))
    writer = store.begin_derived(
        command["native_owner"],
        int(command["writer_epoch"]),
        *arguments,
        work_fingerprint=command["computation_digest"],
    )
    writer.add_config("model", io.BytesIO(body))
    return writer


def execute(command: dict[str, Any]) -> dict[str, Any]:
    store = fill.store(command["store"])
    step = command["step"]
    if step == "resolve":
        return resolve(store, command)
    if step == "heads":
        return heads(command)
    if step == "advance":
        return advance(store, command)
    raise ValueError("unknown upload step")


def main() -> None:
    try:
        command = canonical_json.decode(sys.stdin.buffer.read(MAX_COMMAND + 1))
        if not isinstance(command, dict):
            raise ValueError("upload command must be an object")
        parent = command.get("parent_pid")
        if type(parent) is not int or parent <= 0:
            raise ValueError("an upload step needs its process owner")

        def parent_lifetime() -> None:
            tick = threading.Event()
            while os.getppid() == parent:
                tick.wait(0.25)
            os._exit(70)

        threading.Thread(target=parent_lifetime, daemon=True, name="upload-parent").start()
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
    sys.stdout.buffer.write(body)


if __name__ == "__main__":
    main()
