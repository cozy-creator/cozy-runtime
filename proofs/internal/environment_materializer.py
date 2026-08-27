"""Private generation-reuse and tamper proof below the public proof-seat CLI.

The public environment proof deliberately requires a fresh root. Production replay reuses an
already published receipt-keyed generation, so these checks honestly exercise Runtime internals
without pretending they are part of the public CLI contract.
"""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cozy_runtime.internal import environment
from cozy_runtime.internal.materialize_launch import environment_input


class ProofFailure(RuntimeError):
    """Private materializer evidence did not match its contract."""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or ())
    if len(args) != 1:
        raise ProofFailure("usage: environment.py PROOF_ROOT")
    root = Path(args[0]).resolve()
    request = _object(root / "request-cold.json")
    wheel_files = _paths(request.get("wheel_files"))
    source = environment_input(
        environment_spec_bytes=_bytes(request, "endpoint_environment_spec"),
        endpoint_bundle_bytes=_bytes(request, "endpoint_bundle"),
        resolved_wheel_set_bytes=_bytes(request, "resolved_wheel_set"),
        wheelhouse_manifest_bytes=_bytes(request, "wheelhouse_manifest"),
        wheel_paths=wheel_files,
    )
    environment_root = _path(request, "environment_root")
    uv = _path(request, "uv")
    python = _path(request, "python")
    expected = (root / "receipt-cold.json").read_bytes()
    expected_digest = _digest(expected)

    raced = environment.materialize(source, environment_root, uv=uv, python=python)
    _require(raced.reused, "publish-race replay did not select the existing generation")
    _require(raced.receipt_bytes == expected, "publish-race replay changed receipt bytes")

    warm = environment.materialize(
        source,
        environment_root,
        uv=uv,
        python=python,
        expected_receipt_digest=expected_digest,
    )
    _require(warm.reused, "receipt-keyed replay did not report reuse")
    _require(warm.receipt_bytes == expected, "receipt-keyed replay changed receipt bytes")

    site = _site(warm.generation)
    site_mode = site.stat().st_mode
    site.chmod(site_mode | stat.S_IWUSR)
    planted = site / "planted_unrecorded.py"
    planted.write_text("not installed by a wheel\n")
    planted.chmod(planted.stat().st_mode & ~0o222)
    site.chmod(site_mode)
    _refusal(
        "installed_unrecorded_file",
        lambda: environment.verify_generation(
            warm.generation,
            receipt_digest=warm.receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
        ),
    )
    site.chmod(site_mode | stat.S_IWUSR)
    planted.unlink()
    site.chmod(site_mode)

    installed = site / "proof_environment_project" / "__init__.py"
    installed.chmod(installed.stat().st_mode | stat.S_IWUSR)
    installed.write_text("VALUE = 'tampered'\n")
    installed.chmod(installed.stat().st_mode & ~0o222)
    _refusal(
        "installed_record_mismatch",
        lambda: environment.verify_generation(
            warm.generation,
            receipt_digest=warm.receipt_digest,
            environment_spec_digest=source.environment_spec_digest,
        ),
    )
    print(
        "private environment materializer: publish-race and receipt-keyed reuse retained exact "
        "receipt bytes; unrecorded and tampered installed bytes refused"
    )
    return 0


def _refusal(code: str, call: Callable[[], object]) -> None:
    try:
        call()
    except environment.EnvironmentRefusal as exc:
        _require(exc.code == code, f"wanted {code}, got {exc.code}: {exc.detail}")
    else:
        raise ProofFailure(f"{code} refusal did not fire")


def _site(generation: Path) -> Path:
    sites = list((generation / "venv").glob("lib/python*/site-packages"))
    _require(len(sites) == 1, f"generation has {len(sites)} site-packages directories")
    return sites[0]


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ProofFailure(f"{path} is not a JSON object")
    return value


def _paths(value: object) -> Mapping[str, Path]:
    if not isinstance(value, dict):
        raise ProofFailure("wheel_files is not an object")
    result: dict[str, Path] = {}
    for digest, path in value.items():
        if not isinstance(digest, str) or not isinstance(path, str):
            raise ProofFailure("wheel_files is not a string-to-string object")
        result[digest] = Path(path)
    return result


def _bytes(request: Mapping[str, object], name: str) -> bytes:
    return _path(request, name).read_bytes()


def _path(request: Mapping[str, object], name: str) -> Path:
    value = request.get(name)
    if not isinstance(value, str):
        raise ProofFailure(f"{name} is not a path string")
    return Path(value)


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ProofFailure(detail)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
