"""Loopback conversion of a fixed source roster as verified carriers arrive.

Networking and credentials end in pod-supervisor. Runtime validates the small transport
shape, then hands the exact logical rows to TensorFS. It neither plans model formats nor
synthesizes checkpoint metadata.

No ADDRESS crosses this call. The request still carries `source_uri` and `declared_license`
for the supervisor's own bookkeeping, but TensorFS stopped taking them in `03c9713` and this
module stopped forwarding — or validating — them: a fact that reaches nobody is not this
module's to police. The seam itself is held by `tests/test_tensorfs_binding_contract.py`.
"""

from __future__ import annotations

import re
import stat
from functools import partial
from pathlib import Path

from cozy_runtime.internal import fill, storage_admission
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_PROFILE = re.compile(r"[a-z0-9][a-z0-9._+-]{0,63}(?:/[a-z0-9][a-z0-9._+-]{0,63})*")
_OBJECT_ID = re.compile(r"sha256:[0-9a-f]{64}")


class ModelSourceRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def prepare_model_source(
    request: pb.PrepareModelSourceRequest, *, tensorfs_root: Path
) -> pb.PrepareModelSourceResult:
    """Prepare one exact sorted profile set, returning only typed refs or a safe refusal."""

    try:
        if not tensorfs_root.is_absolute():
            raise ModelSourceRefusal(
                "model_source_store_invalid", "TensorFS store path is not absolute"
            )
        profiles, roster, landed, checkpoints = _request(request)
        tensorfs = fill.tensorfs_module()
        if request.adopt_from_operation_id and (
            _IDENTIFIER.fullmatch(request.adopt_from_operation_id) is None
            or request.adopt_from_operation_id == request.operation_id
            or landed
        ):
            raise ModelSourceRefusal(
                "model_source_adoption_invalid",
                "adoption requires an old operation and no active source conversion",
            )

        def write_bound(payload: int = 0) -> storage_admission.Write:
            bound = storage_admission.native_write(tensorfs_root, payload, len(roster))
            metadata = storage_admission.native_write(tensorfs_root)
            # Each profile can publish its own journal/header metadata. Sparse
            # carrier staging writes only the supplied headers, never file length.
            return storage_admission.Write(
                tensorfs_root,
                bound.bytes
                + metadata.bytes * max(0, len(profiles) - 1)
                + sum(len(row[4]) for row in roster),
                bound.inodes + metadata.inodes * len(profiles),
            )

        bounded = storage_admission.enabled()
        with storage_admission.admit(write_bound()):
            store = fill.ensure_store(tensorfs_root)
            advance = partial(
                store.prepare_model_source,
                request.operation_id,
                documents.spell(bytes(request.source_selection_digest)),
                profiles,
                roster,
                landed,
                checkpoints=checkpoints,
                adopt_from_operation_id=request.adopt_from_operation_id or None,
            )
            prepared = advance(write_budget_bytes=0) if bounded else advance()
        if bounded and (required := prepared["required_write_bytes"]) > 0:
            preferred = max(required, prepared["write_interval_bytes"])
            try:
                with storage_admission.admit(write_bound(preferred)):
                    prepared = advance(write_budget_bytes=preferred)
            except storage_admission.StorageRefusal:
                if preferred == required:
                    raise
                # Near capacity, one atomic op may still fit even when a full
                # preferred pass does not. TensorFS owns both byte counts.
                with storage_admission.admit(write_bound(required)):
                    prepared = advance(write_budget_bytes=required)
        return _result(request, prepared)
    except ModelSourceRefusal as exc:
        return _refused(exc.code, exc.detail)
    except storage_admission.StorageRefusal as exc:
        return _refused("model_source_insufficient_storage", exc.detail)
    except LaunchRefusal as exc:
        return _refused("model_source_tensorfs_unavailable", exc.code)
    except Exception as exc:
        tensorfs_refusal = getattr(getattr(locals().get("tensorfs"), "errors", None), "Refusal", ())
        if tensorfs_refusal and isinstance(exc, tensorfs_refusal):
            code = str(getattr(exc, "code", "refused")).lower()
            # TensorFS says WHICH key, path or member it refused on; a fixed string threw
            # that away. Run 294 refused `duplicate_key` after moving 210.3 GB and the
            # operator's row named nothing, so the cause was found by reading code rather
            # than by reading the error. The detail is TensorFS's own words, capped.
            detail = str(getattr(exc, "detail", "") or exc).strip()
            return _refused(
                f"model_source_tensorfs_{code}",
                f"TensorFS refused source preparation: {detail[:400]}"
                if detail
                else "TensorFS refused source preparation",
            )
        raise


def release_model_source(*, tensorfs_root: Path, operation_id: str) -> None:
    """Release the native operation through its existing lifecycle owner."""

    if _IDENTIFIER.fullmatch(operation_id) is None:
        raise ModelSourceRefusal("model_source_operation_invalid", "operation id is invalid")
    if not tensorfs_root.is_absolute():
        raise ModelSourceRefusal("model_source_store_invalid", "TensorFS store path is invalid")
    fill.store(tensorfs_root).release_model_source(operation_id)


def source_operations(*, tensorfs_root: Path) -> list[str]:
    """Read the native operation inventory; Runtime owns no filesystem scanner."""
    if not tensorfs_root.is_absolute():
        raise ModelSourceRefusal("model_source_store_invalid", "TensorFS store path is invalid")
    try:
        return list(fill.store(tensorfs_root).model_source_operations())
    except Exception as exc:
        raise LaunchRefusal(
            "model_source_inventory_unavailable",
            "the fixed pod requires a readable TensorFS source-operation inventory",
        ) from exc


def _request(
    request: pb.PrepareModelSourceRequest,
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str, int, str, bytes]],
    list[str],
    list[tuple[str, str, int]],
]:
    if (
        _IDENTIFIER.fullmatch(request.operation_id) is None
        or len(request.source_selection_digest) != 32
    ):
        raise ModelSourceRefusal(
            "model_source_prepare_identity_invalid", "source preparation identity is invalid"
        )

    # Order is the sender's; only one slot naming two profiles is ambiguous.
    profiles = sorted({(row.slot, row.profile) for row in request.profiles})
    if not 1 <= len(profiles) <= weights_limits.MAX_MODEL_SOURCE_PROFILES:
        raise ModelSourceRefusal("model_source_profile_count", "source profile count is invalid")
    if len({slot for slot, _ in profiles}) != len(profiles):
        raise ModelSourceRefusal("model_source_profile_order", "a source slot names two profiles")
    if any(
        _IDENTIFIER.fullmatch(slot) is None
        or len(profile.encode()) > weights_limits.MAX_MODEL_SOURCE_PROFILE_BYTES
        or _PROFILE.fullmatch(profile) is None
        for slot, profile in profiles
    ):
        raise ModelSourceRefusal("model_source_profile_invalid", "source profile is invalid")

    # Order is the sender's and a repeated identical row is one file; only one member naming
    # two different files is ambiguous.
    unique = {row.SerializeToString(deterministic=True): row for row in request.files}
    rows = sorted(unique.values(), key=lambda row: row.member)
    if not 1 <= len(rows) <= weights_limits.MAX_MODEL_SOURCE_FILES:
        raise ModelSourceRefusal("model_source_file_count", "source file count is invalid")
    if len({row.member for row in rows}) != len(rows):
        raise ModelSourceRefusal("model_source_file_order", "a source member names two files")

    roster: list[tuple[str, str, int, str, bytes]] = []
    landed: list[str] = []
    # A TensorFS Store is content addressed, so a path is an OBJECT's identity, never a
    # member's: two members carrying identical bytes resolve to one file and MUST. H3's
    # FL2VA and Ref2VA transformer indexes are byte-identical, so its 48 members admit 47
    # objects. What a duplicate path can still prove is a real fault -- two DIFFERENT
    # objects claiming one file -- so the path is checked against the object it carries.
    paths: dict[Path, str] = {}
    for row in rows:
        path = Path(row.path)
        if (
            not _member(row.member)
            or _OBJECT_ID.fullmatch(row.object_id) is None
            or row.length <= 0
            or not _printable(row.path, 4096, empty=False)
            or not path.is_absolute()
        ):
            raise ModelSourceRefusal("model_source_file_invalid", "source file fact is invalid")
        try:
            resolved = path.resolve(strict=row.verified)
            metadata = path.lstat() if row.verified else None
        except OSError as exc:
            raise ModelSourceRefusal(
                "model_source_file_unavailable", f"source file is unavailable for {row.member}"
            ) from exc
        if (
            resolved != path
            or (
                metadata is not None
                and (not stat.S_ISREG(metadata.st_mode) or metadata.st_size != row.length)
            )
            or paths.setdefault(path, row.object_id) != row.object_id
        ):
            # The member is named because this gate has 48 candidates on an H3 ingest and
            # one static string cannot say which failed: run 290 refused here after moving
            # 210.3 GB and the row identified nothing.
            raise ModelSourceRefusal(
                "model_source_file_identity_invalid",
                f"verified source path identity is invalid for {row.member}",
            )
        header = _header(row.header_path, row.member)
        roster.append((row.member, row.object_id, row.length, str(path), header))
        if row.verified:
            landed.append(row.member)
    slots = {slot for slot, _ in profiles}
    checkpoints: list[tuple[str, str, int]] = []
    prior = ""
    for checkpoint in request.checkpoints:
        if (
            checkpoint.slot <= prior
            or checkpoint.slot not in slots
            or len(checkpoint.head.digest) != 32
            or checkpoint.head.length <= 0
            or len(checkpoint.plan_digest) != 32
        ):
            raise ModelSourceRefusal(
                "model_source_checkpoint_invalid", "source restore checkpoint is invalid"
            )
        prior = checkpoint.slot
        checkpoints.append(
            (
                checkpoint.slot,
                documents.spell(bytes(checkpoint.head.digest)),
                checkpoint.head.length,
            )
        )
    return profiles, roster, landed, checkpoints


def _header(value: str, member: str) -> bytes:
    path = Path(value)
    try:
        if (
            not _printable(value, 4096, empty=False)
            or not path.is_absolute()
            or path.resolve(strict=True) != path
            or not stat.S_ISREG(path.lstat().st_mode)
        ):
            raise ValueError("invalid header path")
        with path.open("rb") as stream:
            header = stream.read(weights_limits.MAX_MODEL_SOURCE_HEADER_BYTES + 1)
        if not 0 < len(header) <= weights_limits.MAX_MODEL_SOURCE_HEADER_BYTES:
            raise ValueError("header exceeds its byte bound")
        return header
    except (OSError, ValueError) as exc:
        raise ModelSourceRefusal(
            "model_source_header_invalid", f"source header is unavailable or invalid for {member}"
        ) from exc


def _result(request: pb.PrepareModelSourceRequest, value: object) -> pb.PrepareModelSourceResult:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("replayed"), bool)
        or not isinstance(value.get("complete"), bool)
        or (value["replayed"] and not value["complete"])
    ):
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS returned an invalid result"
        )
    rows = value.get("sources")
    if (
        not isinstance(rows, list)
        or len(rows) > len(request.profiles)
        or (value["complete"] and len(rows) != len(request.profiles))
    ):
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS returned an incomplete result"
        )
    expected = {row.slot: row.profile for row in request.profiles}
    sources: list[pb.PreparedModelSource] = []
    prior = ""
    for row in rows:
        if not isinstance(row, dict):
            raise ModelSourceRefusal(
                "model_source_tensorfs_result_invalid", "TensorFS returned a malformed row"
            )
        slot = row.get("slot")
        profile = row.get("profile")
        manifest = row.get("manifest_digest")
        length = row.get("manifest_length")
        if (
            not isinstance(slot, str)
            or slot <= prior
            or expected.get(slot) != profile
            or not isinstance(manifest, str)
            or _OBJECT_ID.fullmatch(manifest) is None
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
        ):
            raise ModelSourceRefusal(
                "model_source_tensorfs_result_invalid", "TensorFS returned an invalid row"
            )
        prior = slot
        sources.append(
            pb.PreparedModelSource(
                slot=slot,
                profile=profile,
                manifest=pb.Ref(digest=documents.raw(manifest), length=length),
            )
        )
    checkpoints = _checkpoint_result(value.get("checkpoints"), expected)
    spent = value.get("spent")
    members = {row.member for row in request.files}
    if (
        not isinstance(spent, list)
        or any(not isinstance(member, str) or member not in members for member in spent)
        or spent != sorted(set(spent))
    ):
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS returned invalid spent carriers"
        )
    return pb.PrepareModelSourceResult(
        outcome=(
            pb.MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE
            if not value["complete"]
            else pb.MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED
            if value["replayed"]
            else pb.MODEL_SOURCE_PREPARE_OUTCOME_PREPARED
        ),
        sources=sources,
        checkpoints=checkpoints,
        spent_members=spent,
    )


def _checkpoint_result(rows: object, profiles: dict[str, str]) -> list[pb.ModelSourceCheckpoint]:
    if not isinstance(rows, list) or len(rows) > len(profiles):
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS returned invalid checkpoints"
        )
    checkpoints: list[pb.ModelSourceCheckpoint] = []
    prior = ""
    for row in rows:
        if not isinstance(row, dict):
            raise ModelSourceRefusal(
                "model_source_tensorfs_result_invalid", "TensorFS returned a malformed checkpoint"
            )
        slot, head, plan = row.get("slot"), row.get("head"), row.get("plan_digest")
        length, index, completed = row.get("head_length"), row.get("index"), row.get("bytes")
        if (
            not isinstance(slot, str)
            or slot <= prior
            or slot not in profiles
            or not isinstance(head, str)
            or _OBJECT_ID.fullmatch(head) is None
            or not isinstance(plan, str)
            or _OBJECT_ID.fullmatch(plan) is None
            or any(
                type(number) is not int or not 0 <= number < 1 << 64
                for number in (length, index, completed)
            )
            or length == 0
        ):
            raise ModelSourceRefusal(
                "model_source_tensorfs_result_invalid", "TensorFS returned invalid checkpoint facts"
            )
        prior = slot
        checkpoints.append(
            pb.ModelSourceCheckpoint(
                slot=slot,
                head=pb.Ref(digest=documents.raw(head), length=length),
                plan_digest=documents.raw(plan),
                index=index,
                bytes=completed,
            )
        )
    return checkpoints


def _refused(code: str, detail: str) -> pb.PrepareModelSourceResult:
    safe_code = re.sub(r"[^A-Za-z0-9._-]", "_", code)[:256] or "model_source_refused"
    safe_detail = "".join(character for character in detail if 0x20 <= ord(character) <= 0x7E)
    return pb.PrepareModelSourceResult(
        outcome=pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED,
        safe_code=safe_code,
        safe_detail=(safe_detail[:4096] or "model source preparation refused"),
    )


def _member(value: str) -> bool:
    """A safe relative member, as TensorFS admits one: `model (1).safetensors` is one."""
    return (
        value.isascii()
        and value.isprintable()
        and 0 < len(value) <= weights_limits.MAX_MODEL_SOURCE_MEMBER_BYTES
        and not value.startswith("/")
        and "\\" not in value
        and all(part not in ("", ".", "..") for part in value.split("/"))
    )


def _printable(value: str, limit: int, *, empty: bool) -> bool:
    if not isinstance(value, str) or (not empty and not value):
        return False
    try:
        encoded = value.encode("ascii")
    except UnicodeError:
        return False
    return len(encoded) <= limit and all(0x20 <= byte <= 0x7E for byte in encoded)
