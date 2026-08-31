"""Portable private-package admission over Runtime's one overlay policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from packaging.utils import canonicalize_name

from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import descriptor as package_descriptor
from cozy_runtime.internal import overlay
from cozy_runtime.internal.exits import Exit
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_REQUEST_BYTES = 1 << 20
MAX_DESCRIPTOR_BYTES = 16 << 20
MAX_BASE_BYTES = 64 << 20


def run(args: list[str], _opts: Options) -> Result:
    if len(args) != 1:
        raise _error("usage", "pass one private preflight request JSON", Exit.usage)
    request_path = Path(args[0]).resolve()
    request = _json(request_path, MAX_REQUEST_BYTES, "private preflight request")
    if set(request) != {"base_manifests", "descriptor", "revision", "wheels"}:
        raise _error("private_preflight_request_invalid", "request fields are not closed")
    descriptor_path = _path(request.get("descriptor"), "descriptor")
    revision_path = _path(request.get("revision"), "revision")
    wheel_values = request.get("wheels")
    base_values = request.get("base_manifests")
    if (
        not isinstance(wheel_values, list)
        or not wheel_values
        or not isinstance(base_values, list)
        or not base_values
    ):
        raise _error("private_preflight_request_invalid", "wheels and bases must be nonempty")
    wheel_paths = [_path(value, "wheel") for value in wheel_values]
    if len(set(wheel_paths)) != len(wheel_paths):
        raise _error("private_preflight_request_invalid", "paths must be unique")

    revision_raw = _bytes(revision_path, MAX_REQUEST_BYTES, "private revision")
    try:
        revision = documents.read(revision_raw, pb.PrivatePackageRevision)
    except documents.DocumentError as exc:
        raise _error("private_revision_invalid", exc.detail) from exc
    required_revision = {
        "files",
        "format",
        "package",
        "package_descriptor",
        "release",
        "source_digest",
    }
    if set(revision) != required_revision:
        raise _error("private_revision_invalid", "required revision fields are absent")
    if not isinstance(revision["package"], str) or not isinstance(revision["release"], str):
        raise _error("private_revision_invalid", "package or release is malformed")
    try:
        documents.raw(str(revision["source_digest"]))
    except (documents.DocumentError, ValueError) as exc:
        raise _error("private_revision_invalid", "source_digest is malformed") from exc
    descriptor_raw = _bytes(descriptor_path, MAX_DESCRIPTOR_BYTES, "package descriptor")
    try:
        package_descriptor.read_bytes(descriptor_raw, str(descriptor_path))
    except package_descriptor.StalePackageDescriptor as exc:
        raise _error("private_revision_descriptor_invalid", str(exc)) from exc
    descriptor = revision.get("package_descriptor")
    expected_descriptor = {
        "digest": "sha256:" + hashlib.sha256(descriptor_raw).hexdigest(),
        "length": len(descriptor_raw),
    }
    if descriptor != expected_descriptor:
        raise _error("private_revision_descriptor_mismatch", str(descriptor))

    refs = revision.get("files")
    if not isinstance(refs, list) or len(refs) != len(wheel_paths):
        raise _error("private_revision_files_mismatch", "wheel count differs")
    digests: list[str] = []
    projects = 0
    expected: dict[str, dict[str, object]] = {}
    for row in refs:
        if not isinstance(row, dict) or set(row) != {"digest", "filename", "kind", "length"}:
            raise _error("private_revision_files_mismatch", "file row is not closed")
        digest = row.get("digest")
        filename = row.get("filename")
        kind = row.get("kind")
        length = row.get("length")
        if (
            not isinstance(digest, str)
            or not isinstance(filename, str)
            or not isinstance(kind, int)
            or not isinstance(length, int)
            or kind
            not in {
                pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
                pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL,
            }
            or filename in expected
        ):
            raise _error("private_revision_files_mismatch", "file row is invalid")
        digests.append(digest)
        expected[filename] = row
        projects += kind == pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL
    if digests != sorted(set(digests)) or projects != 1:
        raise _error("private_revision_files_mismatch", "files are not one sorted revision")

    actual = {path.name: path for path in wheel_paths}
    if set(actual) != set(expected):
        raise _error("private_revision_files_mismatch", "wheel filenames differ")
    project_path = next(
        actual[name]
        for name, row in expected.items()
        if row["kind"] == pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL
    )
    dependencies = tuple(
        actual[name]
        for name, row in expected.items()
        if row["kind"] == pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL
    )

    compatible: list[dict[str, object]] = []
    refused: list[dict[str, str]] = []
    measured_wheels: tuple[overlay.Wheel, ...] | None = None
    seen_bases: set[tuple[Path, str]] = set()
    bases: list[tuple[Path, str, bytes]] = []
    for value in base_values:
        if not isinstance(value, dict) or set(value) != {"digest", "path"}:
            raise _error("private_preflight_request_invalid", "base row is not exact")
        base_path = _path(value["path"], "base manifest")
        base_digest = value["digest"]
        if not isinstance(base_digest, str):
            raise _error("private_preflight_request_invalid", "base digest is absent")
        try:
            documents.raw(base_digest)
        except (documents.DocumentError, ValueError) as exc:
            raise _error("private_preflight_request_invalid", "base digest is malformed") from exc
        identity = (base_path, base_digest)
        if identity in seen_bases:
            raise _error("private_preflight_request_invalid", "base rows must be unique")
        seen_bases.add(identity)
        base_raw = _bytes(base_path, MAX_BASE_BYTES, "base manifest")
        if "sha256:" + hashlib.sha256(base_raw).hexdigest() != base_digest:
            raise _error("private_preflight_base_mismatch", str(base_path))
        bases.append((base_path, base_digest, base_raw))

    for base_path, base_digest, base_raw in bases:
        try:
            base, wheels = overlay.preflight_overlay(project_path, dependencies, base_raw)
        except overlay.OverlayRefusal as exc:
            refused.append({"base": base_digest, "code": exc.code, "detail": exc.detail})
            continue
        facts = {wheel.filename: wheel for wheel in wheels}
        if any(
            facts[name].digest != row["digest"] or facts[name].length != row["length"]
            for name, row in expected.items()
        ):
            raise _error("private_revision_files_mismatch", "wheel bytes differ from revision")
        measured_wheels = wheels
        compatible.append(
            {
                "base": base_digest,
                "path": str(base_path),
                "compatibility_profile": dict(base.compatibility_profile),
                "platform_target": dict(base.platform_target),
            }
        )
    if not compatible or measured_wheels is None:
        detail = refused[0]["detail"] if refused else "no approved base was compatible"
        raise _error("private_preflight_incompatible", detail)

    project = str(revision.get("package", ""))
    project_wheel = next(wheel for wheel in measured_wheels if wheel.owner == "project")
    if (
        project.count("/") != 1
        or project_wheel.distribution != canonicalize_name(project.split("/", 1)[1])
        or project_wheel.version != revision.get("release")
    ):
        raise _error("private_revision_project_mismatch", project_wheel.filename)
    revision_digest = documents.spell(documents.digest_of(revision_raw))
    return Result(
        rows=(("private_revision_digest", revision_digest), ("compatible", str(len(compatible)))),
        document={
            "private_revision_digest": revision_digest,
            "compatible": compatible,
            "refused": refused,
        },
    )


def _path(value: object, what: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _error("private_preflight_request_invalid", f"{what} path is absent")
    return Path(value).resolve()


def _bytes(path: Path, maximum: int, what: str) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _error(
            "private_preflight_input_unreadable", f"{what}: {path}", Exit.not_found
        ) from exc
    if not 0 < len(raw) <= maximum:
        raise _error("private_preflight_input_invalid", f"{what}: {len(raw)} bytes")
    return raw


def _json(path: Path, maximum: int, what: str) -> dict[str, object]:
    try:
        value = json.loads(_bytes(path, maximum, what))
    except json.JSONDecodeError as exc:
        raise _error("private_preflight_request_invalid", f"{what} is not JSON") from exc
    if not isinstance(value, dict):
        raise _error("private_preflight_request_invalid", f"{what} is not an object")
    return value


def _error(code: str, detail: str, exit_code: Exit = Exit.validation) -> CliError:
    return CliError(
        code,
        detail,
        "use the exact private revision wheels and an approved compatible base manifest",
        exit_code,
    )
