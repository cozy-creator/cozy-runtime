"""Prepare and inspect exact inputs for the installed environment-proof CLI.

Shell owns process boundaries. This module owns deterministic document, wheel, and receipt
inspection that is awkward and error-prone in shell. It deliberately imports only Runtime's
public protocol package.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from proof_data import ProofFailure, Wheel
from proof_data import canonical as _canonical
from proof_data import digest as _digest
from proof_data import inventory as _inventory
from proof_data import json_file as _json
from proof_data import parse_canonical as _parse_canonical
from proof_data import read_json as _read_json
from proof_data import ref as _ref
from proof_data import require as _require
from proof_data import wheel as _wheel
from proof_data import write as _write

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

REQUEST_FORMAT = "cozy.runtime.EnvironmentProofRequest/1"
RECEIPT_FORMAT = "cozy.runtime.EndpointOverlayReceipt/1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("root", type=Path)
    prepare.add_argument("project", type=Path)
    prepare.add_argument("resolved", type=Path)
    prepare.add_argument("wheelhouse", type=Path)
    prepare.add_argument("torch", type=Path)
    prepare.add_argument("python", type=Path)

    sub.add_parser("verify").add_argument("root", type=Path)
    generation = sub.add_parser("generation")
    generation.add_argument("root", type=Path)
    generation.add_argument("name")
    refusal = sub.add_parser("verify-refusal")
    refusal.add_argument("root", type=Path)
    refusal.add_argument("name")
    refusal.add_argument("root_may_exist", choices=("yes", "no"))
    race = sub.add_parser("verify-race")
    race.add_argument("root", type=Path)
    race.add_argument("first_status", type=int)
    race.add_argument("second_status", type=int)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "verify":
        _verify(args.root)
    elif args.command == "generation":
        print(_observation(args.root / f"{args.name}.stdout")["generation"])
    elif args.command == "verify-refusal":
        _verify_refusal(args.root, args.name, root_may_exist=args.root_may_exist == "yes")
    else:
        _verify_race(args.root, args.first_status, args.second_status)
    return 0


def _prepare(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    inputs = root / "inputs"
    variants = root / "variants"
    inputs.mkdir()
    variants.mkdir()

    project_path = args.project.resolve()
    resolved_path = args.resolved.resolve()
    wheelhouse_path = args.wheelhouse.resolve()
    torch_path = args.torch.resolve()
    project = _inventory(project_path)
    resolved = _inventory(resolved_path)
    wheelhouse = _inventory(wheelhouse_path)
    torch = _inventory(torch_path)

    normal_paths = {
        project.digest: project_path,
        resolved.digest: resolved_path,
    }
    normal = _release(
        inputs / "normal",
        _target(),
        project,
        (resolved,),
        (wheelhouse,),
        normal_paths,
    )
    for name in ("cold", "rebuild", "race-first", "race-second", "inside"):
        _request(root, name, normal, root / f"environment-{name}", args.python)
    _request(root, "reuse", normal, root / "environment-cold", args.python)

    missing = dict(normal)
    missing["wheel_files"] = {
        digest: path
        for digest, path in _paths(normal["wheel_files"]).items()
        if digest != resolved.digest
    }
    _request(root, "missing", missing, root / "environment-missing", args.python)

    extra = dict(normal)
    extra["wheel_files"] = {**_paths(normal["wheel_files"]), torch.digest: torch_path}
    _request(root, "extra", extra, root / "environment-extra", args.python)

    corrupt_path = variants / "corrupt" / resolved.filename
    corrupt_path.parent.mkdir()
    corrupt = bytearray(resolved_path.read_bytes())
    corrupt[len(corrupt) // 2] ^= 1
    corrupt_path.write_bytes(corrupt)
    corrupt_request = dict(normal)
    corrupt_request["wheel_files"] = {
        **_paths(normal["wheel_files"]),
        resolved.digest: corrupt_path,
    }
    _request(root, "corrupt", corrupt_request, root / "environment-corrupt", args.python)

    renamed = _release(
        inputs / "renamed",
        _target(),
        project,
        (replace(resolved, filename="renamed-2.0.0-py3-none-any.whl"),),
        (wheelhouse,),
        normal_paths,
    )
    _request(root, "renamed", renamed, root / "environment-renamed", args.python)

    duplicate_path = variants / "duplicate" / resolved.filename
    duplicate_path.parent.mkdir()
    shutil.copyfile(resolved_path, duplicate_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("proof_environment_resolved/__init__.py", b"VALUE = 'duplicate'\n")
    duplicate = _inventory(duplicate_path)
    duplicate_request = _release(
        inputs / "duplicate",
        _target(),
        project,
        (duplicate,),
        (wheelhouse,),
        {
            project.digest: project_path,
            duplicate.digest: duplicate_path,
        },
    )
    _request(
        root,
        "duplicate",
        duplicate_request,
        root / "environment-duplicate",
        args.python,
    )

    expanded_path = variants / "expanded" / resolved.filename
    expanded_path.parent.mkdir()
    shutil.copyfile(resolved_path, expanded_path)
    with zipfile.ZipFile(expanded_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("expansion-payload", b"\0" * (2 << 20))
    expanded = _inventory(expanded_path)
    expanded_request = _release(
        inputs / "expanded",
        _target(),
        project,
        (expanded,),
        (wheelhouse,),
        {
            project.digest: project_path,
            expanded.digest: expanded_path,
        },
    )
    _request(
        root,
        "expanded",
        expanded_request,
        root / "environment-expanded",
        args.python,
    )

    forbidden = dict(normal)
    forbidden["expected_receipt"] = str(root / "receipt-cold.json")
    _request(
        root,
        "forbidden",
        forbidden,
        root / "environment-forbidden",
        args.python,
    )

    cuda_target = {**_target(), "accelerator_backend": "cuda", "accelerator_abi": "cu129"}
    cuda = _release(
        inputs / "cuda",
        cuda_target,
        project,
        (resolved,),
        (wheelhouse, torch),
        normal_paths,
    )
    _request(root, "cuda", cuda, root / "environment-cuda", args.python)


def _release(
    root: Path,
    target: Mapping[str, str],
    project: Wheel,
    resolved_wheels: Sequence[Wheel],
    wheelhouse_wheels: Sequence[Wheel],
    wheel_paths: Mapping[str, Path],
) -> dict[str, object]:
    root.mkdir()
    wheelhouse = _canonical(
        {
            "format": "WheelhouseManifest/1",
            "platform_target": dict(target),
            "wheels": [_wheel(wheel) for wheel in wheelhouse_wheels],
        }
    )
    resolved = _canonical(
        {
            "format": "ResolvedWheelSet/1",
            "lock_digest": _digest(b"proof-lock"),
            "platform_target": dict(target),
            "wheelhouse_manifest_digest": _digest(wheelhouse),
            "wheels": [_wheel(wheel) for wheel in resolved_wheels],
        }
    )
    descriptor = _canonical({"format": "proof.EndpointDescriptor/1"})
    evaluated = _canonical({"format": "proof.EvaluatedConfig/1"})
    bundle = _canonical(
        {
            "descriptor": _ref(descriptor),
            "endpoint_release_id": "proof-release",
            "evaluated_config": _ref(evaluated),
            "format": "tensorhub.endpoint_bundle/1",
            "project_wheel": _wheel(project),
            "resolved_wheel_set": _ref(resolved),
            "wheelhouse_manifest": _ref(wheelhouse),
        }
    )
    spec = documents.canonical_bytes(
        pb.EndpointEnvironmentSpec(
            platform_target=pb.PlatformTarget(**target),
            endpoint_bundle_digest=documents.digest_of(bundle),
            project_wheel_digest=documents.raw(project.digest),
            wheelhouse_manifest_digest=documents.digest_of(wheelhouse),
        )
    )
    return {
        "endpoint_environment_spec": _write(root / "environment.json", spec),
        "endpoint_bundle": _write(root / "bundle.json", bundle),
        "resolved_wheel_set": _write(root / "resolved.json", resolved),
        "wheelhouse_manifest": _write(root / "wheelhouse.json", wheelhouse),
        "wheel_files": dict(wheel_paths),
    }


def _request(
    root: Path,
    name: str,
    release: Mapping[str, object],
    environment_root: Path,
    python: Path,
) -> None:
    value = {
        **release,
        "environment_root": environment_root.resolve(),
        "format": REQUEST_FORMAT,
        "python": python.resolve(),
    }
    _json(root / f"request-{name}.json", value)


def _verify(root: Path) -> None:
    receipts = [_receipt(root, name) for name in ("cold", "rebuild")]
    _require(receipts[0] == receipts[1], "cold and rebuild receipt bytes differ")
    for name, raw in zip(("cold", "rebuild"), receipts, strict=True):
        observation = _success(root, name, raw)
        _require(observation["reused"] is False, f"{name} unexpectedly reported reuse")
    cuda = _receipt(root, "cuda")
    cuda_observation = _success(root, "cuda", cuda)
    _require(cuda_observation["reused"] is False, "CUDA proof unexpectedly reported reuse")
    _require(not list(root.glob(".environment-receipt-*")), "temporary receipt survived")


def _success(root: Path, name: str, receipt: bytes) -> dict[str, Any]:
    observation = _observation(root / f"{name}.stdout")
    _require(
        set(observation) == {"digest", "generation", "length", "reused"},
        f"{name} observation fields changed: {sorted(observation)}",
    )
    digest = _digest(receipt)
    _require(observation["digest"] == digest, f"{name} observation digest is not stored bytes")
    _require(
        observation["length"] == len(receipt), f"{name} observation length is not stored bytes"
    )
    generation = Path(str(observation["generation"]))
    _require(generation.name == digest.removeprefix("sha256:"), f"{name} generation is mis-keyed")
    _require(generation.is_dir(), f"{name} generation is absent")
    _require(
        not list((root / f"environment-{name}").glob(".staging-*")), f"{name} staging survived"
    )
    external = root / f"receipt-{name}.json"
    _require(external.read_bytes() == receipt, f"{name} external receipt changed while read")
    _require(external.stat().st_mode & 0o222 == 0, f"{name} external receipt is writable")
    value = _parse_canonical(receipt)
    _require(
        isinstance(value, dict) and value.get("format") == RECEIPT_FORMAT,
        f"{name} receipt format is not v1",
    )
    return observation


def _verify_refusal(root: Path, name: str, *, root_may_exist: bool) -> None:
    receipt = root / f"receipt-{name}.json"
    _require(not receipt.exists(), f"{name} refusal published a receipt")
    if not root_may_exist:
        _require(
            not (root / f"environment-{name}").exists(),
            f"{name} refusal left an environment root",
        )
    _require(not list(root.glob(".environment-receipt-*")), f"{name} left a receipt temporary")


def _verify_race(root: Path, first_status: int, second_status: int) -> None:
    _require(
        sorted((first_status, second_status)) == [0, 13],
        f"receipt race statuses were {first_status}, {second_status}; expected 0 and 13",
    )
    receipt = (root / "receipt-race.json").read_bytes()
    _require(receipt == _receipt(root, "cold"), "receipt race did not publish complete exact bytes")
    _require(
        (root / "receipt-race.json").stat().st_mode & 0o222 == 0,
        "receipt race winner is writable",
    )
    _require(not list(root.glob(".environment-receipt-*")), "receipt race left a temporary")


def _receipt(root: Path, name: str) -> bytes:
    path = root / f"receipt-{name}.json"
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProofFailure(f"{name} receipt is absent: {exc}") from exc


def _observation(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ProofFailure(f"observation is not an object: {path}")
    return value


def _target() -> dict[str, str]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "accelerator_abi": "none",
        "accelerator_backend": "none",
        "libc": f"{libc_name}{libc_version}" if libc_name else "none",
        "os_arch": f"{platform.system().lower()}/{_architecture()}",
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def _architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    return machine


def _paths(value: object) -> dict[str, Path]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(path, Path) for key, path in value.items()
    ):
        raise ProofFailure("prepared wheel_files is not a digest-to-Path map")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
