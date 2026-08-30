"""Trusted th-046 proof seat for one exact offline package environment."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import canonical, overlay
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.placement_materialization import package_overlay_input
from cozy_runtime.internal.refusal import LaunchRefusal

REQUEST_FIELDS = {
    "environment",
    "environment_root",
    "package",
    "python",
    "wheel_files",
    "wheelhouse_manifest",
}


@dataclass(frozen=True, slots=True)
class ProofRequest:
    package: Mapping[str, object]
    environment_path: Path
    wheelhouse_manifest: Path
    wheel_files: Mapping[str, Path]
    python: Path
    environment_root: Path
    environment: Mapping[str, object]
    environment_digest: str


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize once, publish the measured receipt bytes, and print their exact ref."""

    opts = Options()
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 2:
            raise CliError(
                "usage",
                "cozy-environment-proof requires REQUEST.json and RECEIPT.json",
                "cozy-environment-proof /absolute/request.json /absolute/receipt.json",
                Exit.usage,
            )
        request_path = _cli_input_path(args[0])
        receipt_path = _cli_output_path(args[1])
        if request_path == receipt_path:
            raise CliError(
                "environment_proof_output_invalid",
                "request and receipt name the same file",
                "choose a distinct receipt output",
                Exit.validation,
            )
        request = _read_request(request_path)
        if receipt_path.is_relative_to(request.environment_root):
            raise CliError(
                "environment_proof_output_invalid",
                "receipt output must be outside the environment root",
                "choose a separate absolute receipt path owned by the build seat",
                Exit.validation,
            )
        source = package_overlay_input(
            package=request.package,
            environment=request.environment,
            environment_digest=request.environment_digest,
            wheelhouse_manifest_bytes=_document_bytes(
                request.wheelhouse_manifest, "wheelhouse_manifest"
            ),
            wheel_paths=request.wheel_files,
        )
        installed = overlay.materialize(
            source,
            request.environment_root,
            python=request.python,
            base_manifest=request.wheelhouse_manifest,
            expected_receipt_digest="",
        )
        _publish_receipt(receipt_path, installed.receipt_bytes)
        print(
            json.dumps(
                {
                    "digest": installed.receipt_digest,
                    "generation": str(installed.generation),
                    "length": len(installed.receipt_bytes),
                    "reused": installed.reused,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return int(Exit.ok)
    except CliError as err:
        emit_error(err, opts)
        return int(err.code)
    except (LaunchRefusal, overlay.OverlayRefusal) as err:
        emit_error(
            CliError(
                err.code,
                err.detail,
                "repair the exact local release inputs and run a fresh proof seat",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except OSError as err:
        emit_error(
            CliError(
                "environment_proof_io",
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
                f"unexpected environment proof fault: {type(err).__name__}: {err}",
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
        raise CliError(
            "environment_proof_request_invalid",
            str(err),
            "provide one UTF-8 request object with no duplicate keys",
            Exit.validation,
        ) from err
    if not isinstance(value, dict) or set(value) != REQUEST_FIELDS:
        fields = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise _request_error(f"fields={fields} expected={sorted(REQUEST_FIELDS)}")
    wheels = value["wheel_files"]
    if not isinstance(wheels, dict) or not wheels:
        raise _request_error("wheel_files is not a non-empty digest-to-path object")
    wheel_files: dict[str, Path] = {}
    for digest, local_path in sorted(wheels.items()):
        if not isinstance(digest, str) or canonical.DIGEST_RE.fullmatch(digest) is None:
            raise _request_error(f"wheel_files key {digest!r} is not sha256:<64 lowercase hex>")
        wheel_files[digest] = _existing_file(local_path, f"wheel_files[{digest}]")
    environment_root = _fresh_root(value["environment_root"])
    environment_path = _existing_file(value["environment"], "environment")
    try:
        environment = canonical.parse_canonical(_document_bytes(environment_path, "environment"))
    except canonical.CanonicalError as err:
        raise _request_error(f"environment: {err}") from err
    if not isinstance(environment, dict) or not set(environment).issubset(
        {"wheelhouse_manifest", "wheels"}
    ):
        raise _request_error("environment is not one closed nested Environment object")
    environment_digest = (
        "sha256:"
        + hashlib.sha256(
            canonical.write({**environment, "format": "cozy.worker.v1.Environment/1"})
        ).hexdigest()
    )
    return ProofRequest(
        package=_package(value["package"]),
        environment_path=environment_path,
        wheelhouse_manifest=_existing_file(value["wheelhouse_manifest"], "wheelhouse_manifest"),
        wheel_files=wheel_files,
        python=_existing_executable(value["python"]),
        environment_root=environment_root,
        environment=environment,
        environment_digest=environment_digest,
    )


def _request_error(detail: str) -> CliError:
    return CliError(
        "environment_proof_request_invalid",
        detail,
        "provide the closed local-path request described by cozy-environment-proof --help",
        Exit.validation,
    )


def _package(value: object) -> Mapping[str, object]:
    fields = {"package", "project_wheel", "release", "release_digest"}
    if not isinstance(value, dict) or set(value) != fields:
        raise _request_error("package is not one closed PackageSelection object")
    return value


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


def _existing_executable(value: object) -> Path:
    path = _absolute_path(value, "python")
    try:
        target = path.resolve(strict=True)
    except OSError as err:
        raise _request_error(f"python is unreadable: {type(err).__name__}") from err
    if not target.is_file() or not os.access(target, os.X_OK):
        raise _request_error("python is not an executable file")
    return path


def _fresh_root(value: object) -> Path:
    path = _absolute_path(value, "environment_root").resolve(strict=False)
    if os.path.lexists(path):
        raise CliError(
            "environment_root_not_fresh",
            f"environment root already exists: {path}",
            "choose a new empty path; proof-seat retries never reuse unproven state",
            Exit.conflict,
        )
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
        raise _request_error("receipt output has no filename")
    path = path.parent.resolve(strict=False) / path.name
    if os.path.lexists(path):
        raise CliError(
            "environment_proof_receipt_conflict",
            f"receipt output already exists: {path}",
            "choose a fresh receipt output for this proof seat",
            Exit.conflict,
        )
    return path


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _request_error(f"{name} is not a non-empty path string")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path != Path(os.path.normpath(value)):
        raise _request_error(f"{name} is not one normalized absolute non-root path")
    return path


def _document_bytes(path: Path, name: str) -> bytes:
    size = path.stat().st_size
    if size <= 0 or size > canonical.DOC_MAX_BYTES:
        raise _request_error(f"{name} has unbounded document length {size}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise _request_error(f"{name} changed while it was read")
    return raw


def _publish_receipt(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise _request_error("receipt output parent is not a directory")
    temporary = path.parent / f".environment-receipt-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as err:
            raise CliError(
                "environment_proof_receipt_conflict",
                f"receipt output was concurrently claimed: {path}",
                "preserve the winning receipt and choose a new output for another proof",
                Exit.conflict,
            ) from err
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
