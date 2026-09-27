"""Authorized hardware proof for one already-materialized native package environment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import (
    base_observation,
    canonical,
    config,
    jit_cache,
    native_wheel,
    package_environment,
    package_installation,
    spawn,
)
from cozy_runtime.internal.exits import Exit

REQUEST_FIELDS = {
    "device_index",
    "installation_id",
    "install_root",
    "expected_result_digest",
    "fixture",
    "python",
}


@dataclass(frozen=True, slots=True)
class Request:
    device_index: int | None
    installation_id: str
    install_root: Path
    expected_result_digest: str
    fixture: str
    python: Path


def main(argv: Sequence[str] | None = None) -> int:
    opts = Options()
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 2:
            raise _error("usage: cozy-native-wheel-proof REQUEST.json EVIDENCE.json", Exit.usage)
        request_path = _existing_file(_absolute(args[0], "request"), "request")
        evidence_path = _fresh_output(_absolute(args[1], "evidence"))
        request = _request(request_path)
        base = package_environment.observe_base(request.python)
        installed = package_installation.open_installation(
            request.install_root, request.installation_id
        )
        inspection = native_wheel.inspect_installed(installed.site_packages)
        runtime_config = config.read_config()
        base_environment = spawn.seal(runtime_config.child_base_env, {})
        selected_seat = native_wheel.observe_seat(request.python, base_environment)
        running_seat = native_wheel.observe_seat(Path(sys.executable), base_environment)
        gpu = (
            native_wheel.observe_gpu(request.device_index)
            if request.device_index is not None
            else None
        )
        gpu_sm = gpu["sm"] if gpu is not None else None
        native_wheel.verify_seat(base, running_seat, gpu=gpu)
        native_wheel.verify_seat(base, selected_seat, gpu=gpu)
        if canonical.digest(running_seat) != canonical.digest(selected_seat):
            raise native_wheel.NativeWheelRefusal(
                "package_environment_native_python_seat_mismatch",
                "proof executable and requested base Python report different seats",
            )
        host = native_wheel.qualify_host(inspection, gpu_sm=gpu_sm)
        proof_pod_scope = f"native-proof-{uuid.uuid4().hex}"
        try:
            cache_scope = jit_cache.scope(
                runtime_config.cozy_home,
                installed.installation_id,
                pod_scope=proof_pod_scope,
            )
        except jit_cache.JITCacheRefusal as exc:
            raise native_wheel.NativeWheelRefusal(
                "package_environment_native_jit_cache_invalid", str(exc)
            ) from exc
        imposed = {
            **cache_scope.environment,
            "OMP_NUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "",
        }
        if request.device_index is not None:
            imposed["CUDA_VISIBLE_DEVICES"] = str(request.device_index)
        try:
            operator = native_wheel.qualify_operator(
                python=installed.python,
                fixture=request.fixture,
                expected_result_digest=request.expected_result_digest,
                environment=spawn.seal(runtime_config.child_base_env, imposed),
            )
        finally:
            try:
                jit_cache.remove_pod(runtime_config.cozy_home, proof_pod_scope)
            except jit_cache.JITCacheRefusal as exc:
                raise native_wheel.NativeWheelRefusal(
                    "package_environment_native_jit_cache_invalid", str(exc)
                ) from exc
        evidence = {
            "base_worker_profile": _profile_id(base),
            "installation_id": request.installation_id,
            "host_capability": {
                "gpu": gpu,
                "seat": running_seat,
            },
            "host_qualification": host,
            "operator_observation": operator,
            "package_environment_installation_id": installed.installation_id,
            "static_inspection": inspection.document(),
        }
        payload = canonical.write(evidence)
        _publish(evidence_path, payload)
        print(
            json.dumps(
                {"digest": _digest(payload), "length": len(payload)},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return int(Exit.ok)
    except CliError as exc:
        emit_error(exc, opts)
        return int(exc.code)
    except (native_wheel.NativeWheelRefusal, package_environment.EnvironmentRefusal) as exc:
        emit_error(
            CliError(
                exc.code,
                exc.detail,
                "repair the exact prebuilt wheel/base/hardware tuple and rerun qualification",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except OSError as exc:
        emit_error(
            CliError(
                "native_wheel_proof_io",
                f"{exc.filename or 'proof path'}: {exc.strerror or type(exc).__name__}",
                "repair the qualification-seat paths and ownership",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)


def _request(path: Path) -> Request:
    try:
        value = canonical.parse(_bounded_bytes(path, "request"))
    except canonical.CanonicalError as exc:
        raise _error(str(exc)) from exc
    if not isinstance(value, dict) or not REQUEST_FIELDS.issubset(value):
        raise _error("request is missing a field")
    digests = ("expected_result_digest",)
    if any(
        not isinstance(value[name], str) or canonical.DIGEST_RE.fullmatch(value[name]) is None
        for name in digests
    ):
        raise _error("request digest")
    fixture = value["fixture"]
    device = value["device_index"]
    if (
        not isinstance(fixture, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", fixture) is None
        or not (device is None or (isinstance(device, int) and not isinstance(device, bool)))
    ):
        raise _error("fixture or device")
    return Request(
        device,
        str(value["installation_id"]),
        _existing_directory(_absolute(value["install_root"], "install_root")),
        str(value["expected_result_digest"]),
        fixture,
        _existing_executable(_absolute(value["python"], "python")),
    )


def _profile_id(base: base_observation.Observation) -> str:
    return f"torch{base.torch_release}-{base.accelerator_build}-{base.python_abi}-linux-x86"


def _absolute(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _error(f"{name} is not a path")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path != Path(os.path.normpath(value)):
        raise _error(f"{name} is not one normalized absolute non-root path")
    return path


def _existing_file(path: Path, name: str) -> Path:
    resolved = path.resolve(strict=True)
    if not stat.S_ISREG(resolved.lstat().st_mode):
        raise _error(f"{name} is not a regular file")
    return resolved


def _existing_executable(path: Path) -> Path:
    try:
        target = path.resolve(strict=True)
    except OSError as exc:
        raise _error(f"python is unreadable: {type(exc).__name__}") from exc
    if not target.is_file() or not os.access(target, os.X_OK):
        raise _error("python is not an executable file")
    # Preserve the managed-venv executable path. Resolving its symlink to the base interpreter
    # silently discards pyvenv.cfg and is exactly how a local base could borrow OCI behavior.
    return path


def _existing_directory(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not stat.S_ISDIR(resolved.lstat().st_mode):
        raise _error("install_root is not a directory")
    return resolved


def _fresh_output(path: Path) -> Path:
    resolved = path.parent.resolve(strict=False) / path.name
    if os.path.lexists(resolved):
        raise _error("evidence output already exists", Exit.conflict)
    return resolved


def _bounded_bytes(path: Path, name: str) -> bytes:
    size = path.stat().st_size
    if not 0 < size <= canonical.DOC_MAX_BYTES:
        raise _error(f"{name} length is not bounded")
    raw = path.read_bytes()
    if len(raw) != size:
        raise _error(f"{name} changed while read")
    return raw


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".native-proof-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o400)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _error(detail: str, code: Exit = Exit.validation) -> CliError:
    return CliError(
        "native_wheel_proof_request_invalid",
        detail,
        "provide one closed cozy-native-wheel-proof request over an exact package environment",
        code,
    )
