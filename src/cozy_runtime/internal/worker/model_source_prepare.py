"""Loopback preparation of supervisor-verified foreign model source files.

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
from pathlib import Path

from cozy_runtime.internal import fill
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_MEMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
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
        profiles, files = _request(request)
        tensorfs = fill.tensorfs_module()
        store = tensorfs.Store.ensure(str(tensorfs_root))
        prepared = store.prepare_model_source(
            request.operation_id,
            documents.spell(bytes(request.source_selection_digest)),
            profiles,
            files,
        )
        return _result(request, prepared)
    except ModelSourceRefusal as exc:
        return _refused(exc.code, exc.detail)
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
    """Explicit local lifecycle cleanup; remote pod absence remains the correctness boundary."""

    if _IDENTIFIER.fullmatch(operation_id) is None:
        raise ModelSourceRefusal("model_source_operation_invalid", "operation id is invalid")
    if not tensorfs_root.is_absolute():
        raise ModelSourceRefusal("model_source_store_invalid", "TensorFS store path is invalid")
    tensorfs = fill.tensorfs_module()
    tensorfs.Store.open(str(tensorfs_root)).release_model_source(operation_id)


def _request(
    request: pb.PrepareModelSourceRequest,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, int, str]]]:
    if (
        _IDENTIFIER.fullmatch(request.operation_id) is None
        or len(request.source_selection_digest) != 32
    ):
        raise ModelSourceRefusal(
            "model_source_prepare_identity_invalid", "source preparation identity is invalid"
        )

    profiles = [(row.slot, row.profile) for row in request.profiles]
    if not 1 <= len(profiles) <= weights_limits.MAX_MODEL_SOURCE_PROFILES:
        raise ModelSourceRefusal("model_source_profile_count", "source profile count is invalid")
    if [slot for slot, _ in profiles] != sorted({slot for slot, _ in profiles}):
        raise ModelSourceRefusal("model_source_profile_order", "source profiles are not sorted")
    if any(
        _IDENTIFIER.fullmatch(slot) is None
        or len(profile.encode()) > weights_limits.MAX_MODEL_SOURCE_PROFILE_BYTES
        or _PROFILE.fullmatch(profile) is None
        for slot, profile in profiles
    ):
        raise ModelSourceRefusal("model_source_profile_invalid", "source profile is invalid")

    rows = list(request.files)
    if not 1 <= len(rows) <= weights_limits.MAX_MODEL_SOURCE_FILES:
        raise ModelSourceRefusal("model_source_file_count", "source file count is invalid")
    if [row.member for row in rows] != sorted({row.member for row in rows}):
        raise ModelSourceRefusal("model_source_file_order", "source files are not sorted")

    files: list[tuple[str, str, int, str]] = []
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
            resolved = path.resolve(strict=True)
            metadata = path.lstat()
        except OSError as exc:
            raise ModelSourceRefusal(
                "model_source_file_unavailable", "verified source file is unavailable"
            ) from exc
        if (
            resolved != path
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != row.length
            or paths.setdefault(path, row.object_id) != row.object_id
        ):
            # The member is named because this gate has 48 candidates on an H3 ingest and
            # one static string cannot say which failed: run 290 refused here after moving
            # 210.3 GB and the row identified nothing.
            raise ModelSourceRefusal(
                "model_source_file_identity_invalid",
                f"verified source path identity is invalid for {row.member}",
            )
        files.append((row.member, row.object_id, row.length, str(path)))
    return profiles, files


def _result(request: pb.PrepareModelSourceRequest, value: object) -> pb.PrepareModelSourceResult:
    if not isinstance(value, dict) or not isinstance(value.get("replayed"), bool):
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS returned an invalid result"
        )
    rows = value.get("sources")
    if not isinstance(rows, list) or len(rows) != len(request.profiles):
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
    return pb.PrepareModelSourceResult(
        outcome=(
            pb.MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED
            if value["replayed"]
            else pb.MODEL_SOURCE_PREPARE_OUTCOME_PREPARED
        ),
        sources=sources,
    )


def _refused(code: str, detail: str) -> pb.PrepareModelSourceResult:
    safe_code = re.sub(r"[^A-Za-z0-9._-]", "_", code)[:256] or "model_source_refused"
    safe_detail = "".join(character for character in detail if 0x20 <= ord(character) <= 0x7E)
    return pb.PrepareModelSourceResult(
        outcome=pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED,
        safe_code=safe_code,
        safe_detail=(safe_detail[:4096] or "model source preparation refused"),
    )


def _member(value: str) -> bool:
    try:
        encoded = value.encode("ascii")
    except UnicodeError:
        return False
    return (
        0 < len(encoded) <= weights_limits.MAX_MODEL_SOURCE_MEMBER_BYTES
        and _MEMBER.fullmatch(value) is not None
        and not value.startswith((".", "/"))
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
