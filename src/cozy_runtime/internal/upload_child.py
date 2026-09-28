"""One bounded native step of an upload, launched on an inherited stdin channel.

A step resolves the provider selection, reads member headers, or runs one pass: land a
window of source members and convert what they unlock within the admitted budget. The
pass that completes the conversion composes the published model. The parent owns
admission, publication and eviction; no credential or URL leaves this process.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import msgspec
import tensorfs
from tensorfs.derived import Config, Derivation, Source, SourceInspection, Target

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.internal import fill
from cozy_runtime.internal.worker.source_steps import (
    Advance,
    Advanced,
    Answer,
    Facts,
    Heads,
    Launch,
    Member,
    Pin,
    Plan,
    Prepared,
    Resolve,
    Resolved,
    Select,
    Selected,
    Selection,
    SourceRefusal,
    UploadStep,
    Written,
    serve,
)
from cozy_runtime.models.qwen_image21.ingestion import preparation

# A predecessor that is absent, complete, live, or already adopted has no prefix to give.
_NOT_ADOPTABLE = (
    tensorfs.errors.RootAbsent,
    tensorfs.errors.TransactionClosed,
    tensorfs.errors.TransactionConflict,
    tensorfs.errors.StoreBusy,
    tensorfs.errors.DurabilityUnproven,
)

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


def _merge[P: Pin](selection: P, metadata: Pin) -> P:
    """Small metadata rides in the same pin as the carriers, never as a carrier."""
    names = {row[0] for row in selection.members}
    return msgspec.structs.replace(
        selection,
        members=sorted(
            [*selection.members, *(row for row in metadata.members if row[0] not in names)]
        ),
        allowed_hosts=sorted({*selection.allowed_hosts, *metadata.allowed_hosts}),
        credential_hosts=sorted({*selection.credential_hosts, *metadata.credential_hosts}),
    )


def _pinned(store: tensorfs.Store, step: Resolve, names: list[str]) -> Pin:
    """The files of ``names`` the pinned revision holds; one it lacks is not selected."""
    try:
        return msgspec.convert(store.resolve_source(step.uri, files=names, **step.access), Pin)
    except tensorfs.errors.MissingField:
        found = Pin([], [], [])
        for name in names if len(names) > 1 else ():
            found = _merge(found, _pinned(store, step, [name]))
        return found


def _diffusers_members(store: tensorfs.Store, step: Resolve) -> Pin:
    """A Diffusers repository's index and each component's config, in the same pin."""
    for index in DIFFUSERS_INDEXES:
        pinned = _pinned(store, step, [index])
        if pinned.members:
            break
    else:
        return pinned
    ((_, body),) = tensorfs.read_source_heads(step.uri, pinned.members[:1], **step.access)
    return _pinned(store, step, [index, *sorted(set(diffusers_configs(body).values()))])


def resolve(store: tensorfs.Store, step: Resolve) -> Resolved:
    if step.carriers is None:
        # No profile named: TensorFS selects one from the provider headers and narrows.
        found = store.resolve_source(
            step.uri, profiles=step.profiles, registry=step.registry, **step.access
        )
    else:
        found = store.resolve_source(step.uri, step.carriers, **step.access)
    selection = msgspec.convert(found, Selection)
    if step.metadata:
        # Reviewed metadata rides in the same pin; its bytes are named by the recipe.
        metadata = store.resolve_source(step.uri, files=step.metadata, **step.access)
        selection = _merge(selection, msgspec.convert(metadata, Pin))
    elif step.diffusers:
        # A Diffusers repository's components are constructed from its own configs.
        selection = _merge(selection, _diffusers_members(store, step))
    return Resolved(selection=selection)


def select(store: tensorfs.Store, step: Select) -> Selected:
    """The profile the pinned carriers' headers select, on this machine's registry."""
    heads = dict(tensorfs.read_source_heads(step.uri, step.members, **step.access))
    rows = [(name, length, heads[name]) for name, _, length, _ in step.members]
    return Selected(profile=tensorfs.select_source_profile(store, rows, registry=step.registry))


def heads(step: Heads) -> Written:
    body = msgspec.json.encode(
        dict(tensorfs.read_source_heads(step.uri, step.members, **step.access))
    )
    path = Path(step.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        staged.write(body)
        staged.flush()
        os.fsync(staged.fileno())
    os.replace(staged.name, path)
    return Written()


def _materialize(store: tensorfs.Store, owner: str, rows: list[Member], plan: Plan) -> None:
    store.materialize_source(
        owner,
        rows,
        allowed_hosts=plan.pin.allowed_hosts,
        credential_hosts=plan.pin.credential_hosts,
        credential=plan.access["credential"],
        allow_local=plan.access["allow_local"],
    )


def advance(store: tensorfs.Store, step: Advance) -> Advanced:
    plan = step.plan
    members = {row[0]: row for row in plan.pin.members}
    for name in step.window:
        owner = plan.owners[name]
        for predecessor in step.adopt[name]:
            # A stopped upload of this conversion keeps its verified prefix for this one.
            with contextlib.suppress(*_NOT_ADOPTABLE):
                store.adopt_source_progress(predecessor, owner)
        _materialize(store, owner, [members[name]], plan)
    if plan.metadata:
        _materialize(store, plan.metadata_owner, plan.metadata, plan)
    recorded = msgspec.json.decode(Path(plan.heads_path).read_bytes(), type=dict[str, bytes])
    facts = msgspec.convert(
        tensorfs.prepare_selected_source(
            store,
            plan.conversion,
            plan.source_selection_digest,
            plan.slots,
            [(name, members[name][1], members[name][2], recorded[name]) for name in plan.carriers],
            registry=plan.registry,
            write_budget_bytes=step.budget,
        ),
        Facts,
    )
    return Advanced(
        facts=facts, model=compose(store, plan, facts.sources) if facts.complete else None
    )


def _header(store: tensorfs.Store, manifest: str) -> tensorfs.Header:
    header = store.manifest(manifest)["header"]
    if header is None:
        raise SourceRefusal("model_source_header_absent", f"{manifest} carries no header")
    return tensorfs.parse_header(header)


def compose(store: tensorfs.Store, plan: Plan, prepared: list[Prepared]) -> ObjectRef:
    """The published model: the prepared profiles' union, or the recipe's derivation."""
    transaction = plan.native_owner
    observed = store.derived_lookup(transaction)
    if observed.get("state") != "committed":
        if epoch := observed.get("writer_session_id"):
            store.derived_fence(transaction, epoch)
        writer = (
            _recipe_writer(store, plan, prepared)
            if plan.recipe
            else _union_writer(store, plan, prepared)
        )
        receipt = writer.commit()
    else:
        receipt = observed["receipt"]
    digest, length = "sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"]
    if not plan.recipe:
        for row in prepared:
            names = sorted(_header(store, row.manifest_digest)["assets"])
            if names:
                inherited = store.inherit_model_assets(
                    digest, length, row.manifest_digest, row.manifest_length, names
                )
                digest, length = str(inherited["manifest_id"]), int(inherited["manifest_length"])
    return ObjectRef(digest, length)


def _added_configs(store: tensorfs.Store, plan: Plan) -> dict[str, bytes]:
    """The Diffusers configs landed beside the carriers, named by their component."""
    files = {
        name: store.document(digest.removeprefix("sha256:"), size)
        for name, digest, size, _ in plan.metadata
    }
    index = next((files[name] for name in DIFFUSERS_INDEXES if name in files), None)
    if index is None:
        return {}
    return {name: files[path] for name, path in diffusers_configs(index).items() if path in files}


def _union_writer(
    store: tensorfs.Store, plan: Plan, prepared: list[Prepared]
) -> tensorfs.DerivedWriter:
    sources: dict[str, tuple[str, int]] = {}
    targets: dict[str, dict[str, object]] = {}
    configs: dict[str, dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    for row in sorted(prepared, key=lambda row: row.slot):
        sources[row.slot] = (row.manifest_digest, row.manifest_length)
        header = _header(store, row.manifest_digest)
        for name, tensors in header["components"].items():
            if name in targets:
                raise SourceRefusal(
                    "model_source_components_overlap", f"profiles both produce component {name}"
                )
            targets[name] = {"source": row.slot, "source_component": name, "drop": [], "add": {}}
            order.extend((name, key) for key in tensors)
        for name in header["configs"]:
            if name in configs:
                raise SourceRefusal(
                    "model_source_configs_overlap", f"profiles both carry config {name}"
                )
            configs[name] = {"kind": "copy", "source": row.slot, "source_config": name}
    added = _added_configs(store, plan)
    for name in added:
        if name in configs:
            raise SourceRefusal("model_source_configs_overlap", f"config {name} is carried twice")
        configs[name] = {"kind": "add"}
    writer = store.begin_derived(
        plan.native_owner,
        plan.writer_epoch,
        sources,
        targets,
        configs,
        order,
        sum(len(body) for body in added.values()),
        work_fingerprint=plan.computation_digest,
    )
    for name, body in sorted(added.items()):
        writer.add_config(name, io.BytesIO(body))
    return writer


def _recipe_writer(
    store: tensorfs.Store, plan: Plan, prepared: list[Prepared]
) -> tensorfs.DerivedWriter:
    (row,) = prepared
    components = sorted(_header(store, row.manifest_digest)["components"])
    inspection = SourceInspection.from_native(
        store.inspect_derived_source(row.manifest_digest, row.manifest_length, components, [])
    )
    with tempfile.TemporaryDirectory(dir=Path(store.root) / "tmp") as scratch:
        root = Path(scratch)
        for name, digest, size, _ in plan.metadata:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(store.document(digest.removeprefix("sha256:"), size))
        config, order, files = preparation(inspection, root)
    body = canonical_json.encode(config)
    arguments = Derivation(
        sources={"source": Source(row.manifest_digest, row.manifest_length)},
        targets={name: Target(source="source", source_component=name) for name in components},
        configs={"model": Config("add")},
        order=order,
        files=files,
    ).native_arguments(len(body) + sum(len(data) for data in files.values()))
    writer = store.begin_derived(
        plan.native_owner, plan.writer_epoch, *arguments, work_fingerprint=plan.computation_digest
    )
    writer.add_config("model", io.BytesIO(body))
    return writer


def execute(launch: Launch[UploadStep]) -> Answer:
    store = fill.store(launch.store)
    match launch.step:
        case Resolve() as step:
            return resolve(store, step)
        case Select() as step:
            return select(store, step)
        case Heads() as step:
            return heads(step)
        case Advance() as step:
            return advance(store, step)


if __name__ == "__main__":
    serve(Launch[UploadStep], execute)
