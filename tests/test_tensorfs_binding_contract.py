"""The runtime↔tensorfs seam has no compiler; this is what stands in for one (xs-024 §0).

`tensorfs` is a compiled PyO3 extension in a SEPARATE REPOSITORY, resolved by version range
(`tensorfs>=0.3.1,<1`). Nothing in either tree checks that a call site still matches the
binding it calls:

- **mypy cannot.** tensorfs ships `py.typed`, but `_ext.pyi` is 41 lines whose operative
  content is `Store.__getattr__(self, name) -> Any` and a module-level `__getattr__ -> Any`.
  Every method on this page type-checks as `Any` under `strict = true`. The runtime's own
  `fill.tensorfs_module() -> Any` erases what little is left.
- **The wheel is a binary.** A parameter deleted in Rust changes no Python source here, so
  no review, diff or lint sees it.

That gap was not hypothetical. tensorfs `03c9713` (#64, 2026-09-02) deleted `source_uri` and
`declared_license` from `Store.prepare_model_source`; `worker/model_source_prepare.py` kept
passing them for two days, so every foreign-source preparation on a pod raised `TypeError`
before tensorfs was entered. Nothing went red, because nothing tested that file.

Two arms, and the second is the one that would have caught it:

1. **The frozen table.** Every tensorfs entry point cozy-runtime calls, with the exact
   signature it was written against. A rename, an added parameter, a reordering or a
   removal on the Rust side turns this red on the next `uv sync`.
2. **The real call.** `prepare_model_source`/`release_model_source` driven through the real
   runtime function against a real `Store` on disk. Arity is proved by binding, not by
   reading — the strongest form available without a compiler.

WHAT THIS CANNOT COVER, stated so nobody mistakes the coverage for more than it is:

- **Types and semantics.** `inspect.signature` sees names and defaults, never that
  `files` wants `list[tuple[str, str, int, str]]` or that a `str` now means a path. A Rust
  parameter that keeps its name and changes its meaning passes this test.
- **The other two mechanisms.** cozy-runtime also reaches tensorfs through the `tfs` CLI's
  TEXT stdout (pod-supervisor, the daemon, the hub) and through vendored artifacts. Neither
  is a Python signature and neither is checked here (xs-024 §1B, §1C).
- **Anything the runtime does not call.** ~110 exports, ~34 exercised. A change to the
  other two thirds is invisible here, correctly.
- **The version actually installed on a pod.** This checks the wheel in THIS environment.
  Worker images pin their own, and the startup handshake that would close that is xs-024 §1.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import tensorfs
import tensorfs.plane

from cozy_runtime.internal.worker import model_source_prepare
from cozy_runtime.protocol import worker_pb2 as pb

SRC = Path(__file__).resolve().parent.parent / "src" / "cozy_runtime"

#: Every callable tensorfs entry point cozy-runtime calls, against the signature it was
#: written for. Transcribed from the installed extension, not from the Rust source: the
#: wheel is what a pod runs. A diff here is a seam change and wants a human, not a re-freeze.
CALLED: dict[str, str] = {
    "pull": (
        "(store, hub, refspec, *, lane=Ellipsis, credential=Ellipsis, timeout=None, "
        "session=Ellipsis, streams=Ellipsis, streams_start=Ellipsis, allowed_hosts=Ellipsis, "
        "allow_local=False, sample_seconds=None, cancellation=None, progress=None)"
    ),
    "PullCancellation": "()",
    "transfer_streams": "(wanted)",
    "manifest_max_bytes": "()",
    "read_source_heads": (
        "(source_uri, members, *, credential=Ellipsis, huggingface=None, civitai=None, "
        "allow_local=False)"
    ),
    "select_source_profile": "(store, heads, *, registry=None)",
    "source_profile_converters": "(store, profiles, *, registry=None)",
    "prepare_selected_source": (
        "(store, operation_id, source_selection_digest, profiles, members, *, registry=None, "
        "write_budget_bytes=None)"
    ),
    "model_source_objects": "(store, operation_id)",
    "record_model_source_custody": "(store, operation_id, objects, evidence)",
    "PullCancellation.cancel": "(self, /)",
    "gc": "(root, dry_run=False, *, evict_cached=False, keep_manifests=Ellipsis)",
    "fit": (
        "(requirements: 'Sequence[TensorRequirement]', header: 'bytes', *, custody: 'str', "
        "encoded_leaves: 'bool', device: 'str | None' = None, "
        "observations: 'bytes | None' = None) -> 'dict[str, object]'"
    ),
    "plan": "(header, traversal, components=None, window=Ellipsis)",
    "parse_header": "(data: 'bytes', closure: 'Closure | None' = None) -> 'Header'",
    "parse_manifest": "(data)",
    "seed_digests": "()",
    "object_id": "(data)",
    "dtypes": "()",
    "TensorRequirement": (
        "(component: 'str', key: 'str', shape: 'Sequence[int]', "
        "logical_dtype: 'str | None' = None) -> None"
    ),
    "Store.open": "(root=None)",
    "Store.ensure": "(root=None)",
    "Store.resolve_source": (
        "(self, /, source_uri, carriers=Ellipsis, *, profiles=Ellipsis, files=Ellipsis, "
        "credential=Ellipsis, huggingface=None, civitai=None, allow_local=False, registry=None)"
    ),
    "Store.materialize_source": (
        "(self, /, owner, members, *, allowed_hosts, credential_hosts=Ellipsis, "
        "credential=Ellipsis, allow_local=False, progress=None)"
    ),
    "Store.prepare_source_artifact": (
        "(self, /, source_owner, operation_id, profiles, *, checkpoints=Ellipsis, "
        "adopt_from_operation_id=None, registry=None)"
    ),
    "Store.create_tree_root": "(self, /, owner, manifest_id, manifest_length)",
    "Store.import_tree": "(self, /, owner, manifest_bytes, files)",
    "Store.retain_tree_root": "(self, /, source_owner, owner)",
    "Store.tree_root": "(self, /, owner)",
    "Store.release_tree_root": "(self, /, owner)",
    "Store.release_tree_retention": "(self, /, source_owner, receipt_digest, owner)",
    "Store.adopt_source_progress": "(self, /, source_owner, owner)",
    "Store.prepare_readers": "(self, /)",
    "Store.manifest": "(self, /, hex)",
    "Store.put_file": "(self, /, path, expect=None, expect_length=None)",
    "Store.put_manifest": "(self, /, data, expect=None, expect_length=None)",
    "Store.acquire": "(self, /, manifest, objects)",
    "Store.acquire_manifest": "(self, /, hex)",
    "Store.acquire_cozytensors": "(self, /, hex)",
    "Store.walk": "(self, /, hex)",
    "Store.walk_cozytensors": "(self, /, hex)",
    "Store.resolve_release": "(self, /, org, name, version, lane)",
    "Store.complete_cozytensors_manifests": "(self, /)",
    "Store.begin_derived": (
        "(self, /, transaction_id, writer_session_id, sources, targets, configs, order, "
        "max_new_bytes, files=None, *, work_fingerprint, checkpoint=None)"
    ),
    "Store.derived_declaration": (
        "(self, /, sources, targets, configs, order, max_new_bytes, files=None, "
        "*, work_fingerprint)"
    ),
    "Store.inspect_derived_source": "(self, /, manifest, manifest_length, components, configs)",
    "Store.validate_derived_checkpoint": (
        "(self, /, transaction_id, declaration, head_id, head_length, *, "
        "operation_id, slot, plan_digest)"
    ),
    "Store.model_source_operations": "(self, /)",
    "Store.derived_lookup": "(self, /, transaction_id)",
    "Store.derived_fence": "(self, /, transaction_id, writer_session_id)",
    "Store.derived_abandon": "(self, /, transaction_id)",
    "Store.derived_dispose": "(self, /, transaction_id)",
    "Store.derived_adopt": "(self, /, transaction_id, scratch_root_id)",
    # The incremental roster/landed seam replaces the full-file barrier in xs-025.
    "Store.prepare_model_source": (
        "(self, /, operation_id, source_selection_digest, profiles, roster, landed, "
        "*, checkpoints=Ellipsis, adopt_from_operation_id=None, registry=None, "
        "write_budget_bytes=None)"
    ),
    "Store.release_model_source": "(self, /, operation_id)",
    "Store.checkpoint_page": (
        "(self, /, head_id, head_length, *, operation_id, slot, plan_digest, offset=0, limit=128)"
    ),
    "Store.checkpoint_push": (
        "(self, /, object_id, length, manifest, grant, *, allow_local=False)"
    ),
    "Store.checkpoint_fetch": (
        "(self, /, object_id, length, manifest, url, *, allow_local=False, derived_restore=None)"
    ),
    "ReadLease.stream": "(self, /, plan, slots, on_batch, readers=4)",
    "ReadLease.read_into": "(self, /, object, object_length, off, length, into)",
    "ReadLease.read_asset": "(self, /, header, name, max_bytes=67108864)",
    "ReadLease.read_part_into": "(self, /, header, component, key, role, off, into)",
    "ReadLease.release": "(self, /)",
    "DerivedWriter.add_part": "(self, /, component, key, role, reader)",
    "DerivedWriter.add_config": "(self, /, name, reader)",
    "DerivedWriter.commit": "(self, /)",
    "DerivedWriter.completed_parts": "(self, /)",
    "DerivedWriter.completed_configs": "(self, /)",
    "DerivedWriter.checkpoint": "(self, /, operation_id, slot, *, previous=None)",
    "DerivedWriter.fence": "(self, /)",
    "DerivedWriter.source_read_into": "(self, /, source, component, key, role, off, into)",
    "Batch.release": "(self, /)",
    "Batch.items": "(self, /)",
    "ReadPlan.items": "(self, /)",
    # The weight plane (`tensorfs.plane`, 0.3.84): what `internal/plane.py`, `internal/weights.py`
    # and `worker/host_tier.py` call.
    "plane.Plane": (
        "(devices, readers=8, copy_streams=1, slab_bytes=Ellipsis, staging_buffers=4, "
        "staging_bytes=Ellipsis, direct_io=True)"
    ),
    "plane.Plane.source": "(self, /, store, lease)",
    "plane.Plane.register": "(self, /, name, source, plan, regions, host_fd=None)",
    "plane.Plane.set_vram_budget": "(self, /, device, nbytes)",
    "plane.Plane.set_pinned_budget": "(self, /, nbytes)",
    "plane.Plane.want": "(self, /, ws, tier, regions=None, priority=0, pin=False)",
    "plane.Plane.prioritise": "(self, /, ws, tier, regions=None, priority=0, pin=None)",
    "plane.Plane.drop": "(self, /, ws, tier, regions=None)",
    "plane.Plane.acquire": "(self, /, ws, device, region, stream, priority=None)",
    "plane.Plane.stream": (
        "(self, /, ws, device, order, window, repeat=1, priority=0, tag='denoise', ring_bytes=None)"
    ),
    "plane.Plane.events": "(self, /)",
    "plane.Plane.stats": "(self, /)",
    "plane.WeightSet.close": "(self, /)",
    "plane.Ticket.wait": "(self, /)",
    "plane.Lease.release": "(self, /, stream=None)",
    "plane.Lease.view": "(self, /)",
    "plane.Cursor.acquire": "(self, /, region, stream)",
    "plane.Cursor.close": "(self, /)",
}

#: Bindings cozy-runtime uses only where the installed TensorFS has them: the call site
#: detects them and degrades without them, so absence is never a break and never a floor.
#: Where present they keep the signature they were written against. Keyed roots (the
#: stage memo's machine tier, `worker/stage_memo.py`) shipped in TensorFS 0.3.84.
OPTIONAL: dict[str, str] = {
    "Store.put_keyed_root": "(self, /, space, key, manifest_bytes, files)",
    "Store.keyed_roots": "(self, /, space)",
    "Store.drop_keyed_root": "(self, /, space, key)",
}

#: Non-callable crossings: read, never called, so they carry no signature to freeze. Their
#: DISAPPEARANCE is still a break, which is all this set asserts.
READ: frozenset[str] = frozenset(
    {
        "DTYPES",
        "errors.Refusal",
        "ReadLease.live",
        "Batch.slot",
        "Batch.nbytes",
        "ReadPlan.order",
        "ReadPlan.bytes",
    }
)

#: Module attributes reached by name that are not themselves called or read as values —
#: the types cozy-runtime names to get at the members above.
NAMED: frozenset[str] = frozenset({"Store", "ReadLease", "Header", "Tensor", "errors"})


def _resolve(dotted: str) -> object:
    value: object = tensorfs
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def _signature(dotted: str) -> str:
    return str(inspect.signature(_resolve(dotted)))  # type: ignore[arg-type]


def test_every_entry_point_cozy_runtime_calls_still_exists() -> None:
    """A DELETED binding is the loudest form of this break, so it gets its own arm."""

    missing = [name for name in sorted(CALLED | {n: "" for n in READ}) if not _exists(name)]
    assert not missing, (
        f"tensorfs {tensorfs.__version__} no longer exposes {missing}; cozy-runtime calls "
        "them. Repair the call sites, do not delete the entries."
    )


def _exists(dotted: str) -> bool:
    try:
        _resolve(dotted)
    except AttributeError:
        return False
    return True


def test_every_entry_point_cozy_runtime_calls_keeps_its_frozen_signature() -> None:
    """The arm that would have gone red on 2026-09-02 instead of on a rented pod."""

    drifted = {
        name: (frozen, _signature(name))
        for name, frozen in sorted(CALLED.items())
        if _signature(name) != frozen
    }
    assert not drifted, (
        f"tensorfs {tensorfs.__version__} changed a binding cozy-runtime calls:\n"
        + "\n".join(
            f"  {name}\n    written against: {was}\n    installed:       {now}"
            for name, (was, now) in drifted.items()
        )
        + "\nFix the call site FIRST; re-freeze only once the caller agrees."
    )


def test_optional_bindings_keep_their_signature_where_the_installed_tensorfs_has_them() -> None:
    """An older TensorFS lacks them and the Runtime runs without; a newer one must not have
    changed them under the call sites."""

    drifted = {
        name: (frozen, _signature(name))
        for name, frozen in sorted(OPTIONAL.items())
        if _exists(name) and _signature(name) != frozen
    }
    assert not drifted, f"tensorfs {tensorfs.__version__} changed an optional binding: {drifted}"


def test_the_signature_check_would_notice_a_deleted_parameter() -> None:
    """The guard's own red arm: prove the comparison detects the exact break it exists for."""

    installed = _signature("Store.prepare_model_source")
    missing_landed = installed.replace(", landed,", ",")
    assert missing_landed != installed
    assert CALLED["Store.prepare_model_source"] == installed


def test_the_frozen_table_names_every_tensorfs_attribute_the_source_reaches() -> None:
    """Keeps the table honest as call sites are ADDED, which is how a table normally rots.

    Module-level only. A method on a `Store` held in a local variable cannot be resolved
    statically without types, and tensorfs's stub supplies none — so `Store.*`, `ReadLease.*`
    and `DerivedWriter.*` above are maintained by hand and are the guard's soft edge.
    """

    covered = {name.split(".")[0] for name in CALLED} | {n.split(".")[0] for n in READ} | NAMED
    reached: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "tensorfs":
                for alias in node.names:
                    reached.setdefault(alias.name, f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Attribute) and _names_the_module(node.value):
                reached.setdefault(node.attr, f"{path.name}:{node.lineno}")
    uncovered = {name: where for name, where in sorted(reached.items()) if name not in covered}
    assert not uncovered, (
        f"these tensorfs attributes are reached by cozy-runtime but are not in this "
        f"module's frozen table, so a Rust-side change to them is unguarded: {uncovered}"
    )


def _names_the_module(node: ast.expr) -> bool:
    """`tensorfs.X` — whether the module came from `import tensorfs` or `tensorfs_module()`."""

    if isinstance(node, ast.Name):
        return node.id == "tensorfs"
    return isinstance(node, ast.Attribute) and node.attr == "tensorfs"


def _request(tmp_path: Path, *, operation_id: str) -> pb.PrepareModelSourceRequest:
    """One shape-valid request over a real file, so validation passes and the call crosses."""

    blob = tmp_path / "model.safetensors"
    blob.write_bytes(b"a real file that is not a real checkpoint" * 8)
    data = blob.read_bytes()
    return pb.PrepareModelSourceRequest(
        operation_id=operation_id,
        source_selection_digest=hashlib.sha256(b"selection").digest(),
        profiles=[pb.ModelSourceProfile(slot="unet", profile="sdxl/unet")],
        files=[
            pb.LocalModelSourceFile(
                member="model.safetensors",
                object_id="sha256:" + hashlib.sha256(data).hexdigest(),
                length=len(data),
                path=str(blob),
                header_path=str(blob),
                verified=True,
            )
        ],
    )


def test_prepare_model_source_reaches_tensorfs_and_comes_back_typed(tmp_path: Path) -> None:
    """The integration arm. A shape-valid request must REACH the binding.

    The file is deliberately not a checkpoint, so tensorfs answers with a typed refusal
    rather than a manifest — which is the point. A refusal proves the arguments BOUND. With
    the six-argument call this repair deleted, this test raises `TypeError` instead, because
    `prepare_model_source` re-raises anything that is not a tensorfs `Refusal`.
    """

    root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(root))

    result = model_source_prepare.prepare_model_source(
        _request(tmp_path, operation_id="op-contract-1"), tensorfs_root=root
    )

    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code.startswith("model_source_tensorfs_"), result.safe_code


def test_release_model_source_reaches_tensorfs_and_comes_back_typed(tmp_path: Path) -> None:
    """The same proof for the lifecycle half, which has its own arity."""

    root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(root))

    model_source_prepare.release_model_source(tensorfs_root=root, operation_id="op-contract-absent")
    model_source_prepare.release_model_source(tensorfs_root=root, operation_id="op-contract-absent")
    assert tensorfs.Store.open(str(root)).model_source_operations() == []


def test_prepare_model_source_refuses_a_relative_store_before_touching_tensorfs(
    tmp_path: Path,
) -> None:
    """The refusal paths of this module had no coverage at all; give them a floor."""

    result = model_source_prepare.prepare_model_source(
        _request(tmp_path, operation_id="op-contract-2"), tensorfs_root=Path("relative/store")
    )

    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code == "model_source_store_invalid"


def test_a_source_in_any_order_with_any_safe_member_name_reaches_tensorfs(tmp_path: Path) -> None:
    """Profiles and files arrive in the sender's order, a repeated file is one file, and a
    Hugging Face member such as `model (1)+ema.safetensors` is a safe relative member. Each
    used to refuse here, after the download, before TensorFS saw the source."""
    root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(root))
    request = _request(tmp_path, operation_id="op-contract-3")
    del request.profiles[:]
    request.profiles.extend(
        [
            pb.ModelSourceProfile(slot="vae", profile="sdxl/vae"),
            pb.ModelSourceProfile(slot="unet", profile="sdxl/unet"),
        ]
    )
    (row,) = request.files
    row.member = "model (1)+ema.safetensors"
    other = pb.LocalModelSourceFile()
    other.CopyFrom(row)
    other.member = "a/earlier.safetensors"
    request.files.extend([row, other])

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=root)

    assert result.safe_code.startswith("model_source_tensorfs_"), result.safe_code
    conflicting = pb.LocalModelSourceFile()
    conflicting.CopyFrom(other)
    conflicting.length += 1
    request.files.append(conflicting)
    result = model_source_prepare.prepare_model_source(request, tensorfs_root=root)
    assert result.safe_code == "model_source_file_order"


def test_two_members_carrying_one_object_share_one_path_and_are_admitted(
    tmp_path: Path,
) -> None:
    """A content-addressed Store makes member->path NON-injective, and that is correct.

    MiniMax-H3's `FL2VA/transformer/model.safetensors.index.json` and its `Ref2VA/`
    counterpart are byte-identical, so its 48 selected members admit 47 objects and two
    members name one file. Run 290 downloaded all 210.3 GB and then refused here, on
    `model_source_file_identity_invalid`, because the duplicate-path check read a shared
    path as a fault. It could not fire before tfs-060: those two members had never been
    fetched, so the pair had never reached this loop.
    """

    request = _request(tmp_path, operation_id="op-contract-shared")
    twin = pb.LocalModelSourceFile()
    twin.CopyFrom(request.files[0])
    twin.member = "z-second-member.safetensors"
    request.files.append(twin)

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    # It reaches TensorFS rather than refusing here. The store path is absent, so the
    # answer is a store refusal -- proof the identity gate passed the shared path.
    assert result.safe_code != "model_source_file_identity_invalid"


def test_a_tensorfs_refusal_carries_its_own_words(tmp_path: Path) -> None:
    """TensorFS says WHICH key it refused on; a fixed string threw that away.

    Run 294 refused `model_source_tensorfs_duplicate_key` after moving 210.3 GB and the
    operator's row read only "TensorFS refused source preparation" — so the duplicated
    path had to be found by reading tensorfs source. The code was preserved and the one
    fact that localises it was not.
    """

    request = _request(tmp_path, operation_id="op-contract-detail")
    # An absent store makes tensorfs refuse with a detail naming what it could not find.
    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "absent")

    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code.startswith("model_source_tensorfs_")
    # The detail is no longer the bare fixed string: TensorFS's own words survive.
    assert result.safe_detail != "TensorFS refused source preparation"
    assert len(result.safe_detail) > len("TensorFS refused source preparation")


def test_two_different_objects_claiming_one_path_are_still_refused(tmp_path: Path) -> None:
    """The invariant the duplicate-path check actually protects, kept.

    A shared path is legitimate only when the objects agree. Two DIFFERENT object ids on
    one file is a real fault and must stay a refusal, or the repair above would have
    widened the gate instead of correcting it.
    """

    request = _request(tmp_path, operation_id="op-contract-collide")
    twin = pb.LocalModelSourceFile()
    twin.CopyFrom(request.files[0])
    twin.member = "z-second-member.safetensors"
    twin.object_id = "sha256:" + "0" * 64
    request.files.append(twin)

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    assert result.safe_code == "model_source_file_identity_invalid"
    # Which member failed must be in the detail: 48 candidates and one static string is
    # how run 290 refused after moving 210.3 GB while naming nothing.
    assert "z-second-member.safetensors" in result.safe_detail


def test_prepare_model_source_refuses_a_file_whose_length_is_a_lie(tmp_path: Path) -> None:
    request = _request(tmp_path, operation_id="op-contract-4")
    request.files[0].length += 1

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    assert result.safe_code == "model_source_file_identity_invalid"


def test_unverified_payload_need_not_exist_before_header_preflight(tmp_path: Path) -> None:
    request = _request(tmp_path, operation_id="op-header-first")
    request.files[0].path = str(tmp_path / "not-downloaded.safetensors")
    request.files[0].verified = False

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    # The actual TensorFS binding, rather than Runtime path validation, rejects this
    # deliberately invalid header. No fabricated Store or downloaded payload is involved.
    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code.startswith("model_source_tensorfs_"), result.safe_code


def test_verified_payload_must_exist_before_conversion(tmp_path: Path) -> None:
    request = _request(tmp_path, operation_id="op-verified-missing")
    request.files[0].path = str(tmp_path / "not-downloaded.safetensors")

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    assert result.safe_code == "model_source_file_unavailable"


def test_header_bound_is_enforced_before_reading_a_model_sized_file(tmp_path: Path) -> None:
    from cozy_runtime.protocol import weights_limits

    request = _request(tmp_path, operation_id="op-header-bound")
    header = tmp_path / "oversized-header"
    with header.open("wb") as stream:
        stream.truncate(weights_limits.MAX_MODEL_SOURCE_HEADER_BYTES + 1)
    request.files[0].header_path = str(header)

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    assert result.safe_code == "model_source_header_invalid"


def test_incomplete_result_preserves_checkpoint_and_spent_carriers(tmp_path: Path) -> None:
    request = _request(tmp_path, operation_id="op-partial-result")
    result = model_source_prepare._result(
        request,
        {
            "complete": False,
            "replayed": False,
            "sources": [],
            "spent": [request.files[0].member],
            "checkpoints": [
                {
                    "slot": "unet",
                    "head": "sha256:" + "a" * 64,
                    "head_length": 512,
                    "plan_digest": "sha256:" + "b" * 64,
                    "index": 0,
                    "bytes": 4096,
                }
            ],
        },
    )
    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE
    assert not result.sources
    assert list(result.spent_members) == [request.files[0].member]
    assert result.checkpoints[0].head.digest == bytes.fromhex("a" * 64)
    assert result.checkpoints[0].head.length == 512
    assert result.checkpoints[0].plan_digest == bytes.fromhex("b" * 64)
    assert result.checkpoints[0].index == 0
    assert result.checkpoints[0].bytes == 4096
