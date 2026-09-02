"""Author-side writer of the publish memo `cozy.package.code-topologies/1` (model-code-fit §3).

Runs on the author's machine inside the project venv (`uv run --locked`), at publish. Per
slot class it runs `derive` — `census(factory(config_document))` on the meta device, in the
one bounded child `cozy-runtime derive` uses (`internal/derive_child.py`) — and writes one
digest-addressed envelope: `result.json` (the memo index: one row per (slot, config_digest))
plus `code-topologies/<sha256>.json` sibling documents, each the canonical TensorFS
CodeTopology `tfs fit` reads, by atomic rename. The hub stores the memo and never grades it
(D2); it is the only artefact a future "which checkpoints fit this slot" search could join on.

No ModelConstructionContract, no hardware variants, no snapshots, no verdict. The
construction SEED is the config document of the slot's default binding, resolved by the
publisher; its checkpoint identity never enters the memo.

The console script is still named `cozy-model-contract-proof`, and the request still
carries `snapshot` and `hardware_variants` per slot: that is the verb cozy-creator's publish
invokes today, and both members are ignored here. cl-094 replaces the call with
`cozy-runtime derive` and this module is deleted with it.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import canonical, derive_child
from cozy_runtime.internal import descriptor as package_descriptor
from cozy_runtime.internal.descriptor import StalePackageDescriptor
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit

INDEX_NAME = "result.json"
TOPOLOGIES_DIR = "code-topologies"

INDEX_FORMAT = "cozy.package.code-topologies/1"

REQUEST_FIELDS = {
    "package_descriptor",
    "package",
    "project",
    "slots",
}
REF_FIELDS = {"digest", "length"}
LOCAL_REF_FIELDS = {"digest", "length", "path"}
SLOT_FIELDS = {"class", "config"}
#: Members of cozy-creator's request that named the deleted MCC plane; read and ignored.
RETIRED_SLOT_FIELDS = {"hardware_variants", "snapshot"}


class ProofRefusal(Exception):
    """One stable build-seat refusal, carried across the bounded child boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class DocumentRef:
    digest: str
    length: int

    def render(self) -> dict[str, Any]:
        return {"digest": self.digest, "length": self.length}


@dataclass(frozen=True, slots=True)
class LocalDocumentRef(DocumentRef):
    path: Path


@dataclass(frozen=True, slots=True)
class SlotRequest:
    """One slot class's construction seed: the config document the class's factory runs
    against on the meta device. Seed identity is an input, never a published fact."""

    class_key: str
    config: LocalDocumentRef


@dataclass(frozen=True, slots=True)
class ProofRequest:
    project: Path
    package: Mapping[str, Any]
    package_descriptor: LocalDocumentRef
    slots: tuple[SlotRequest, ...]

    @property
    def release_identity(self) -> str:
        return f"{self.package['package']}@{self.package['release']}"


def main(argv: Sequence[str] | None = None) -> int:
    """Derive the code-topologies memo from one already materialized release environment."""

    args = list(sys.argv[1:] if argv is None else argv)
    opts = Options()
    try:
        if len(args) != 2:
            raise CliError(
                "usage",
                "cozy-model-contract-proof requires REQUEST.json and a fresh OUTPUT_DIR",
                "cozy-model-contract-proof /absolute/request.json /absolute/output-dir",
                Exit.usage,
            )
        request_path = _cli_input_path(args[0])
        output = _cli_output_path(args[1])
        request = _read_request(request_path)
        if output.is_relative_to(request.project):
            raise CliError(
                "code_topologies_output_invalid",
                "the evidence output must be outside the project tree",
                "choose a separate fresh output directory",
                Exit.validation,
            )
        # The author's own venv is the derivation environment: the parent runs inside
        # `uv run --locked`, so sys.executable resolves the exact locked interpreter.
        index, topologies = _derive(Path(sys.executable), request)
        index_bytes = canonical.write(index)
        _publish(output, index_bytes, topologies)
        print(
            json.dumps(
                {
                    "format": INDEX_FORMAT,
                    "rows": len(index["rows"]),
                    "code_topologies": len(topologies),
                    "digest": _sha(index_bytes),
                    "length": len(index_bytes),
                    "output": str(output),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return int(Exit.ok)
    except CliError as err:
        emit_error(err, opts)
        return int(err.code)
    except (ProofRefusal, derive_child.DeriveRefusal) as err:
        emit_error(
            CliError(
                err.code,
                err.detail,
                "repair the exact release descriptor/config inputs and run a fresh proof",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except (ConformanceError, StalePackageDescriptor) as err:
        emit_error(
            CliError(
                str(getattr(err, "code", "code_topologies_conformance")),
                str(err),
                "repair the package source so its descriptor is the published one",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except OSError as err:
        emit_error(
            CliError(
                "code_topologies_io",
                f"{err.filename or 'local proof path'}: {err.strerror or type(err).__name__}",
                "repair the trusted build-seat paths and permissions",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except Exception as err:
        emit_error(
            CliError(
                "internal",
                f"unexpected code-topologies fault: {type(err).__name__}: {err}",
                "report this Runtime build and exact request",
                Exit.internal,
            ),
            opts,
        )
        return int(Exit.internal)


def _read_request(path: Path) -> ProofRequest:
    try:
        value = canonical.parse(_document_bytes(path, "request"))
    except canonical.CanonicalError as err:
        raise _request_error(str(err)) from err
    if not isinstance(value, dict) or set(value) != REQUEST_FIELDS:
        fields = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise _request_error(f"fields={fields} expected={sorted(REQUEST_FIELDS)}")
    slots_value = value["slots"]
    if not isinstance(slots_value, list) or not slots_value:
        raise _request_error("slots is not a non-empty array")
    slots = tuple(_slot(row, index) for index, row in enumerate(slots_value))
    classes = [row.class_key for row in slots]
    if classes != sorted(set(classes)):
        raise _request_error("slots must be sorted unique by class")
    return ProofRequest(
        project=_existing_directory(value["project"], "project"),
        package=_package(value["package"]),
        package_descriptor=_local_ref(value["package_descriptor"], "package_descriptor"),
        slots=slots,
    )


def _slot(value: object, index: int) -> SlotRequest:
    name = f"slots[{index}]"
    if not isinstance(value, dict) or set(value) - RETIRED_SLOT_FIELDS != SLOT_FIELDS:
        raise _request_error(f"{name} does not have the closed slot field set")
    class_key = value["class"]
    if (
        not isinstance(class_key, str)
        or not class_key
        or len(class_key) > 512
        or class_key.strip() != class_key
    ):
        raise _request_error(f"{name}.class is not one bounded non-empty string")
    return SlotRequest(class_key=class_key, config=_local_ref(value["config"], f"{name}.config"))


def _package(value: object) -> Mapping[str, Any]:
    # No release_digest: the memo is derived BEFORE finalize mints the manifest digest, so
    # it binds `<org>/<name>@<version>` (cr-067's circularity fix).
    fields = {"package", "project_wheel", "release"}
    if not isinstance(value, dict) or set(value) != fields:
        raise _request_error("package is not one closed package/release/project_wheel triple")
    identity = value.get("package")
    release = value.get("release")
    wheel = value.get("project_wheel")
    if (
        not isinstance(identity, str)
        or identity.count("/") != 1
        or not all(identity.split("/"))
        or not isinstance(release, str)
        or not release
        or not isinstance(wheel, dict)
        or not isinstance(wheel.get("ref"), dict)
    ):
        raise _request_error("package identity, release, or project wheel is invalid")
    _ref(wheel["ref"], "package.project_wheel.ref")
    if wheel.get("distribution") != identity.split("/", 1)[1] or wheel.get("version") != release:
        raise _request_error("project wheel does not match package release")
    return value


def _ref(value: object, name: str) -> DocumentRef:
    if not isinstance(value, dict) or set(value) != REF_FIELDS:
        raise _request_error(f"{name} is not one closed digest/length ref")
    length = value["length"]
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise _request_error(f"{name}.length is not a positive integer")
    return DocumentRef(_digest(value["digest"], f"{name}.digest"), length)


def _local_ref(value: object, name: str) -> LocalDocumentRef:
    if not isinstance(value, dict) or set(value) != LOCAL_REF_FIELDS:
        raise _request_error(f"{name} is not one closed path/digest/length ref")
    ref = _ref({key: value[key] for key in REF_FIELDS}, name)
    if ref.length > canonical.DOC_MAX_BYTES:
        raise _request_error(f"{name}.length exceeds the canonical document bound")
    return LocalDocumentRef(ref.digest, ref.length, _existing_file(value["path"], name))


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or canonical.DIGEST_RE.fullmatch(value) is None:
        raise _request_error(f"{name} is not sha256:<64 lowercase hex>")
    return value


def _request_error(detail: str) -> CliError:
    return CliError(
        "code_topologies_request_invalid",
        detail,
        "provide one closed cozy-model-contract-proof request",
        Exit.validation,
    )


def _existing_file(value: object, name: str) -> Path:
    path = _absolute_path(value, name)
    try:
        path = path.resolve(strict=True)
        info = path.lstat()
    except OSError as err:
        raise _request_error(f"{name} is unreadable: {type(err).__name__}") from err
    if not stat.S_ISREG(info.st_mode):
        raise _request_error(f"{name} is not a regular file")
    return path


def _existing_directory(value: object, name: str) -> Path:
    path = _absolute_path(value, name)
    try:
        path = path.resolve(strict=True)
        info = path.lstat()
    except OSError as err:
        raise _request_error(f"{name} is unreadable: {type(err).__name__}") from err
    if not stat.S_ISDIR(info.st_mode):
        raise _request_error(f"{name} is not a directory")
    return path


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _request_error(f"{name} is not a non-empty path string")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path != Path(os.path.normpath(value)):
        raise _request_error(f"{name} is not one normalized absolute non-root path")
    return path


def _cli_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return _existing_file(str(path.resolve(strict=False)), "request")


def _cli_output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.name:
        raise _request_error("output directory has no name")
    path = path.parent.resolve(strict=False) / path.name
    if os.path.lexists(path):
        raise CliError(
            "code_topologies_output_conflict",
            f"proof output already exists: {path}",
            "preserve it and choose a fresh output directory",
            Exit.conflict,
        )
    return path


def _document_bytes(path: Path, name: str) -> bytes:
    size = path.stat().st_size
    if size <= 0 or size > canonical.DOC_MAX_BYTES:
        raise _request_error(f"{name} has unbounded document length {size}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise _request_error(f"{name} changed while it was read")
    return raw


def _derive(python: Path, request: ProofRequest) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Derive one CodeTopology per requested slot class and fan it out to every slot path
    of that class: the memo rows are keyed (slot, config_digest), the derive is per class
    because one class has one factory.

    The publishing descriptor is byte-compared against the source first: the memo must
    describe the release it rides with, and a source that describes differently is refused
    here rather than shipped beside a descriptor it disagrees with.
    """
    descriptor_doc = _read_descriptor(request.package_descriptor)
    application = descriptor_doc.get("application")
    if not isinstance(application, str) or not application:
        raise ProofRefusal("code_topologies_package_descriptor_invalid", "application is absent")
    found = discover(request.project)
    package_descriptor.compare(
        descriptor_doc,
        package_descriptor.build(found),
        ("the publishing descriptor", "the project source"),
    )
    if found.application != application:
        raise ProofRefusal(
            "code_topologies_package_descriptor_invalid",
            "descriptor application does not name the discovered application",
        )
    paths_by_class: dict[str, list[str]] = {}
    for surface in found.surfaces:
        if surface.hidden:
            continue
        for binding in surface.model_bindings:
            paths_by_class.setdefault(binding.class_key, []).append(binding.path)
    unknown = sorted({row.class_key for row in request.slots} - set(paths_by_class))
    if unknown:
        raise ProofRefusal(
            "code_topologies_class_unknown",
            f"requested classes {unknown} are not model slot classes of this package "
            f"(have {sorted(paths_by_class)})",
        )
    for source in request.slots:
        _read_exact(source.config, f"config for {source.class_key}")
    slots = tuple(
        derive_child.SlotRequest(
            sorted(paths_by_class[source.class_key])[0],
            derive_child.ConfigRef(source.config.path, source.config.digest, source.config.length),
        )
        for source in request.slots
    )
    by_slot = {slot.slot: source for slot, source in zip(slots, request.slots, strict=True)}
    ordered = tuple(sorted(slots, key=lambda row: row.slot))
    scratch = Path(tempfile.mkdtemp(prefix="cozy-code-topologies-"))
    try:
        derived = derive_child.derive_in(
            python, derive_child.DeriveRequest(ordered, project=request.project), scratch
        )
    finally:
        _remove_tree(scratch)

    topologies: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for held in derived:
        source = by_slot[held.slot]
        previous = topologies.setdefault(held.digest, held.document)
        if previous != held.document:
            raise ProofRefusal("code_topologies_digest_collision", held.digest)
        for path in paths_by_class[source.class_key]:
            rows.append(
                {
                    "slot": path,
                    "class": source.class_key,
                    "config_digest": source.config.digest,
                    "code_topology": {"digest": held.digest, "length": len(held.document)},
                }
            )
    rows.sort(key=lambda row: (str(row["slot"]), str(row["config_digest"])))
    keys = [(str(row["slot"]), str(row["config_digest"])) for row in rows]
    if keys != sorted(set(keys)):
        raise ProofRefusal(
            "code_topologies_child_invalid", "rows are not sorted unique by (slot, config_digest)"
        )
    memo = {
        "format": INDEX_FORMAT,
        "package": request.package["package"],
        "release": request.package["release"],
        "descriptor_digest": request.package_descriptor.digest,
        "rows": rows,
    }
    return memo, topologies


def _read_exact(ref: LocalDocumentRef, name: str) -> bytes:
    raw = _document_bytes(ref.path, name)
    if len(raw) != ref.length or _sha(raw) != ref.digest:
        raise ProofRefusal(
            "code_topologies_input_identity_mismatch",
            f"{name} at {ref.path} does not match {ref.digest}/{ref.length}",
        )
    return raw


def _read_descriptor(ref: LocalDocumentRef) -> dict[str, Any]:
    """Read a review projection, then verify its semantic canonical ref."""
    raw = _document_bytes(ref.path, "package_descriptor")
    document = package_descriptor.read_bytes(raw, str(ref.path))
    semantic = package_descriptor.canonical_bytes(document)
    if len(semantic) != ref.length or _sha(semantic) != ref.digest:
        raise ProofRefusal(
            "code_topologies_input_identity_mismatch",
            f"package_descriptor at {ref.path} does not mean {ref.digest}/{ref.length}",
        )
    return document


def _publish(
    output: Path,
    index: bytes,
    topologies: Mapping[str, bytes],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not stat.S_ISDIR(output.parent.lstat().st_mode):
        raise ProofRefusal("code_topologies_output_invalid", "output parent is not a directory")
    temporary = output.parent / f".code-topologies-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o700)
        topologies_dir = temporary / TOPOLOGIES_DIR
        topologies_dir.mkdir(mode=0o700)
        for digest, raw in sorted(topologies.items()):
            _write_durable(topologies_dir / f"{digest.removeprefix('sha256:')}.json", raw)
        _write_durable(temporary / INDEX_NAME, index)
        topologies_dir.chmod(0o500)
        temporary.chmod(0o500)
        _fsync_dir(topologies_dir)
        _fsync_dir(temporary)
        try:
            os.rename(temporary, output)
        except OSError as err:
            if err.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise CliError(
                    "code_topologies_output_conflict",
                    f"proof output was concurrently claimed: {output}",
                    "preserve the winning output and choose a fresh directory",
                    Exit.conflict,
                ) from err
            raise
        _fsync_dir(output.parent)
    finally:
        if temporary.exists():
            _remove_tree(temporary)


def _write_durable(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o700)
            path.rmdir()
        else:
            path.chmod(0o600)
            path.unlink()
    root.chmod(0o700)
    root.rmdir()


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
