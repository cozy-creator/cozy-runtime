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
#: TensorFS refuses a carrier header wider than this, so reading one is bounded.
_MAX_HEADER_BYTES = 16 * 1024 * 1024


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
        profiles, roster, landed = _request(request)
        tensorfs = fill.tensorfs_module()
        store = tensorfs.Store.ensure(str(tensorfs_root))
        prepared = store.prepare_model_source(
            request.operation_id,
            documents.spell(bytes(request.source_selection_digest)),
            profiles,
            roster,
            landed,
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
            return _refused(f"model_source_tensorfs_{code}", "TensorFS refused source preparation")
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
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str, int, str, bytes]],
    list[str],
]:
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

    rows = list(request.roster)
    if not 1 <= len(rows) <= weights_limits.MAX_MODEL_SOURCE_FILES:
        raise ModelSourceRefusal("model_source_file_count", "source file count is invalid")
    if [row.member for row in rows] != sorted({row.member for row in rows}):
        raise ModelSourceRefusal("model_source_file_order", "source files are not sorted")

    # The roster names every selected member; only the ones the supervisor says have LANDED
    # have bytes on disk. A member still in flight is a normal, expected row here — checking
    # its carrier would refuse the very ask this RPC exists to make.
    landed = list(request.landed)
    if len(landed) != len(set(landed)) or not set(landed) <= {row.member for row in rows}:
        raise ModelSourceRefusal(
            "model_source_landed_invalid", "landed members are not a distinct roster subset"
        )

    roster: list[tuple[str, str, int, str, bytes]] = []
    paths: set[Path] = set()
    for row in rows:
        path = Path(row.path)
        header_path = Path(row.header_path)
        if (
            not _member(row.member)
            or _OBJECT_ID.fullmatch(row.object_id) is None
            or row.length <= 0
            or not _printable(row.path, 4096, empty=False)
            or not path.is_absolute()
            or not _printable(row.header_path, 4096, empty=False)
            or not header_path.is_absolute()
        ):
            raise ModelSourceRefusal("model_source_file_invalid", "source file fact is invalid")
        if path in paths:
            raise ModelSourceRefusal(
                "model_source_file_identity_invalid", "verified source path identity is invalid"
            )
        paths.add(path)
        # The header is always required: a conversion plan is a pure function of the
        # headers, so every member's header must be readable before ANY carrier can be
        # converted. It is what the supervisor bought by range before the weights.
        roster.append((row.member, row.object_id, row.length, str(path), _header(header_path)))
        if row.member in landed:
            _landed_carrier(path, row.length)
    return profiles, roster, landed


def _header(path: Path) -> bytes:
    """One carrier's `u64le length || JSON` prefix, as the supervisor wrote it."""

    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise ModelSourceRefusal(
            "model_source_header_unavailable", "source header is unavailable"
        ) from exc
    if (
        resolved != path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or not 8 < metadata.st_size <= _MAX_HEADER_BYTES
    ):
        raise ModelSourceRefusal(
            "model_source_header_invalid", "source header path identity is invalid"
        )
    return path.read_bytes()


def _landed_carrier(path: Path, length: int) -> None:
    """A member the supervisor says has landed must really be there, at its exact length."""

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
        or metadata.st_size != length
    ):
        raise ModelSourceRefusal(
            "model_source_file_identity_invalid", "verified source path identity is invalid"
        )


def _result(request: pb.PrepareModelSourceRequest, value: object) -> pb.PrepareModelSourceResult:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("replayed"), bool)
        or not isinstance(value.get("complete"), bool)
    ):
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS returned an invalid result"
        )
    spent = _spent(request, value.get("spent"))
    if not value["complete"]:
        # ADVANCED is not a verdict. Work was converted and durably journalled, the artifact
        # is still owed carriers, and the supervisor asks again as more of them land. The
        # spent carriers it may drop now are the whole reason this answer exists.
        return pb.PrepareModelSourceResult(
            outcome=pb.MODEL_SOURCE_PREPARE_OUTCOME_ADVANCED, spent=spent
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
        spent=spent,
    )


def _spent(request: pb.PrepareModelSourceRequest, value: object) -> list[str]:
    """Roster members no remaining conversion op will read."""

    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS returned an invalid spent set"
        )
    spent = [name for name in value if isinstance(name, str)]
    if not set(spent) <= {row.member for row in request.roster}:
        raise ModelSourceRefusal(
            "model_source_tensorfs_result_invalid", "TensorFS named a spent member off-roster"
        )
    return sorted(spent)


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
