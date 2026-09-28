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
from cozy_runtime.author._loader import PATH_KEYS, PATH_SUFFIXES
from cozy_runtime.internal import fill
from cozy_runtime.internal.worker.source_steps import (
    Advance,
    Advanced,
    Answer,
    Composition,
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
from cozy_runtime.internal.worker.upload_plan import carrier
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


_ASSET_KINDS = ("Tokenizer", "Processor", "FeatureExtractor")


def _components(index: bytes) -> dict[str, tuple[str, str]]:
    """``{component: (class, folder)}`` from a classic or modular Diffusers index."""
    document = json.loads(index)
    found: dict[str, tuple[str, str]] = {}
    for name, entry in document.items() if isinstance(document, dict) else ():
        if name.startswith("_") or not isinstance(entry, list) or len(entry) < 2:
            continue
        kind = entry[1]
        spec = entry[2] if len(entry) > 2 and isinstance(entry[2], dict) else {}
        hint = spec.get("type_hint")
        if not isinstance(kind, str) and isinstance(hint, list) and len(hint) == 2:
            kind = hint[1]
        folder = spec.get("subfolder") or name
        if isinstance(kind, str) and isinstance(folder, str):
            found[name] = (kind, folder)
    return found


def diffusers_configs(index: bytes) -> dict[str, str]:
    """Each Diffusers component's construction config file, by component name.

    A model is built from its folder's ``config.json`` and a scheduler from its
    ``scheduler_config.json``; tokenizers and processors are assets, not constructors.
    Classic indexes map a component to ``[library, class]``, modular ones add a spec whose
    ``subfolder`` and ``type_hint`` name the folder and class.
    """
    return {
        name: f"{folder}/scheduler_config.json"
        if kind.endswith("Scheduler")
        else f"{folder}/config.json"
        for name, (kind, folder) in _components(index).items()
        if not any(word in kind for word in _ASSET_KINDS)
    }


def diffusers_assets(index: bytes) -> list[str]:
    """The folders of the index's tokenizers, processors and feature extractors: every
    ordinary file in them is a model asset the pipeline needs to run."""
    return sorted(
        f"{folder}/"
        for kind, folder in _components(index).values()
        if any(word in kind for word in _ASSET_KINDS)
    )


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
    """A Diffusers repository's index, each component's config and every tokenizer or
    processor file, in the same pin."""
    for index in DIFFUSERS_INDEXES:
        pinned = _pinned(store, step, [index])
        if pinned.members:
            break
    else:
        return pinned
    ((_, body),) = tensorfs.read_source_heads(step.uri, pinned.members[:1], **step.access)
    configs = sorted(set(diffusers_configs(body).values()))
    return _pinned(store, step, [index, *configs, *diffusers_assets(body)])


def reference_members(store: tensorfs.Store, step: Resolve, reference: str) -> Pin:
    """A converter reference's pipeline files; an unreachable one refuses by name."""
    try:
        pinned = _diffusers_members(store, msgspec.structs.replace(step, uri=reference))
    except Exception as exc:
        raise SourceRefusal(
            "model_reference_unavailable", f"converter reference {reference} is unavailable: {exc}"
        ) from exc
    if not pinned.members:
        raise SourceRefusal(
            "model_reference_unavailable", f"converter reference {reference} has no pipeline index"
        )
    return pinned


def references(store: tensorfs.Store, step: Resolve, selection: Pin) -> list[str]:
    """Pinned repositories whose configs and tokenizers the converters' output needs. A
    source no profile can plan pins none here; the conversion refuses it by its own code."""
    profiles = step.profiles
    if not profiles:
        members = [row for row in selection.members if carrier(row[0])]
        select_step = Select(
            uri=step.uri, members=members, registry=step.registry, access=step.access
        )
        try:
            profiles = [select(store, select_step).profile]
        except tensorfs.errors.Refusal:
            return []
    rows = tensorfs.source_profile_converters(store, profiles, registry=step.registry)
    return sorted({reference for _, reference in rows if reference})


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
    if not step.metadata and not any(row[0] in DIFFUSERS_INDEXES for row in selection.members):
        # A source with no pipeline index (a Civitai single file) takes it, the configs and
        # tokenizers from its converters' pinned reference, in the same pin.
        for reference in references(store, step, selection):
            selection = _merge(selection, reference_members(store, step, reference))
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
        facts=facts,
        model=compose(store, plan.composition(), facts.sources) if facts.complete else None,
    )


def _header(store: tensorfs.Store, manifest: str) -> tensorfs.Header:
    header = store.manifest(manifest)["header"]
    if header is None:
        raise SourceRefusal("model_source_header_absent", f"{manifest} carries no header")
    return tensorfs.parse_header(header)


def compose(store: tensorfs.Store, plan: Composition, prepared: list[Prepared]) -> ObjectRef:
    """The model: the prepared profiles' union with its pinned metadata, or the recipe's
    derivation."""
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
    return ObjectRef("sha256:" + receipt["manifest"]["sha256"], receipt["manifest"]["length"])


def _metadata_index(store: tensorfs.Store, metadata: list[Member]) -> tuple[str, bytes] | None:
    for name, digest, size, _ in metadata:
        if name in DIFFUSERS_INDEXES:
            return name, store.document(digest.removeprefix("sha256:"), size)
    return None


def _added_configs(store: tensorfs.Store, metadata: list[Member]) -> dict[str, bytes]:
    """The pipeline index and each component's Diffusers config landed beside the carriers,
    named by their component (the index by its own stem)."""
    found = _metadata_index(store, metadata)
    if found is None:
        return {}
    index_name, index = found
    paths = diffusers_configs(index)
    wanted = set(paths.values())
    files = {
        name: store.document(digest.removeprefix("sha256:"), size)
        for name, digest, size, _ in metadata
        if name in wanted
    }
    added = {name: _constructible(files[path]) for name, path in paths.items() if path in files}
    added[index_name.removesuffix(".json")] = index
    return added


def _constructible(config: bytes) -> bytes:
    """A saved Diffusers/Transformers config minus its private save-time carrier keys
    (`_name_or_path`): a construction config never names a source (the Loader refuses one)."""
    document = json.loads(config)
    private = {
        key
        for key in document
        if key.startswith("_") and (key.lower() in PATH_KEYS or key.lower().endswith(PATH_SUFFIXES))
    }
    if not private:
        return config
    return canonical_json.encode({k: v for k, v in document.items() if k not in private})


def _asset_files(store: tensorfs.Store, metadata: list[Member]) -> dict[str, tuple[str, int]]:
    """Every tokenizer/processor file landed beside the carriers: a model asset, named by
    its verified Store object."""
    found = _metadata_index(store, metadata)
    folders = tuple(diffusers_assets(found[1])) if found is not None else ()
    return {
        name: (digest, size)
        for name, digest, size, _ in metadata
        if folders and name.startswith(folders)
    }


def _union_writer(
    store: tensorfs.Store, plan: Composition, prepared: list[Prepared]
) -> tensorfs.DerivedWriter:
    sources: dict[str, tuple[str, int]] = {}
    targets: dict[str, dict[str, object]] = {}
    configs: dict[str, dict[str, str]] = {}
    bodies: dict[str, bytes] = {}
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
        for name, body in header["configs"].items():
            if name in configs and bodies[name] != body:
                raise SourceRefusal(
                    "model_source_configs_overlap", f"profiles both carry config {name}"
                )
            if name not in configs:
                configs[name] = {"kind": "copy", "source": row.slot, "source_config": name}
                bodies[name] = body
    added = _added_configs(store, plan.metadata)
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
        _asset_files(store, plan.metadata) or None,
        work_fingerprint=plan.computation_digest,
    )
    for name, body in sorted(added.items()):
        writer.add_config(name, io.BytesIO(body))
    return writer


def _recipe_writer(
    store: tensorfs.Store, plan: Composition, prepared: list[Prepared]
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
