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
from typing import Any, cast

from proof_data import ProofFailure, Wheel
from proof_data import canonical as _canonical
from proof_data import digest as _digest
from proof_data import installed_base_ownership as _installed_base_ownership
from proof_data import inventory as _inventory
from proof_data import json_file as _json
from proof_data import pack_installed as _pack_installed
from proof_data import parse_canonical as _parse_canonical
from proof_data import read_json as _read_json
from proof_data import ref as _ref
from proof_data import require as _require
from proof_data import wheel as _wheel
from proof_data import write as _write

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

RECEIPT_FORMAT = "cozy.runtime.EnvironmentReceipt/1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("root", type=Path)
    prepare.add_argument("runtime", type=Path)
    prepare.add_argument("project", type=Path)
    prepare.add_argument("resolved", type=Path)
    prepare.add_argument("torch", type=Path)
    prepare.add_argument("numpy", type=Path)
    prepare.add_argument("numpy_shadow", type=Path)
    prepare.add_argument("shared_root_a", type=Path)
    prepare.add_argument("shared_root_b", type=Path)
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

    runtime_path = args.runtime.resolve()
    project_path = args.project.resolve()
    resolved_path = args.resolved.resolve()
    torch_path = args.torch.resolve()
    runtime = _inventory(runtime_path)
    project = _inventory(project_path)
    resolved = _inventory(resolved_path)
    torch = _inventory(torch_path)
    numpy = _inventory(args.numpy.resolve())
    numpy_shadow = _inventory(args.numpy_shadow.resolve())
    shared_root_a = _inventory(args.shared_root_a.resolve())
    shared_root_b = _inventory(args.shared_root_b.resolve())
    base_wheel_root = variants / "base-wheels"
    base_wheel_root.mkdir()
    dependency_paths = [
        _pack_installed(base_wheel_root, name)
        for name in ("grpcio", "msgspec", "protobuf", "typing-extensions")
    ]
    base_wheels = tuple(
        sorted(
            (runtime, *(_inventory(path) for path in dependency_paths)),
            key=lambda wheel: wheel.distribution,
        )
    )
    _require(len(base_wheels) == 5, "proof base does not carry five control wheels")

    normal_paths = {
        project.digest: project_path,
        resolved.digest: resolved_path,
    }
    normal = _release(
        inputs / "normal",
        _target(),
        project,
        (resolved,),
        base_wheels,
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
        base_wheels,
        normal_paths,
    )
    _request(root, "renamed", renamed, root / "environment-renamed", args.python)

    oversized_fact = _release(
        inputs / "oversized-fact",
        _target(),
        project,
        (replace(resolved, length=(2 << 30) + 1),),
        base_wheels,
        normal_paths,
    )
    _request(
        root,
        "oversized-fact",
        oversized_fact,
        root / "environment-oversized-fact",
        args.python,
    )

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
        base_wheels,
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
        base_wheels,
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
    build_request = dict(normal)
    build_request["source_path"] = str(root / "untrusted-source")
    _request(
        root,
        "build-request",
        build_request,
        root / "environment-build-request",
        args.python,
    )

    cuda_target = {**_target(), "accelerator_backend": "cuda", "accelerator_abi": "cu129"}
    cuda = _release(
        inputs / "cuda",
        cuda_target,
        project,
        (resolved,),
        base_wheels,
        normal_paths,
    )
    _request(root, "cuda", cuda, root / "environment-cuda", args.python)

    base_distribution = _release(
        inputs / "base-distribution",
        _target(),
        project,
        (numpy,),
        base_wheels,
        {project.digest: project_path, numpy.digest: args.numpy.resolve()},
    )
    _request(
        root,
        "base-distribution",
        base_distribution,
        root / "environment-base-distribution",
        args.python,
    )

    base_import_root = _release(
        inputs / "base-import-root",
        _target(),
        project,
        (numpy_shadow,),
        base_wheels,
        {project.digest: project_path, numpy_shadow.digest: args.numpy_shadow.resolve()},
    )
    _request(
        root,
        "base-import-root",
        base_import_root,
        root / "environment-base-import-root",
        args.python,
    )

    lying_import_root = _release(
        inputs / "lying-import-root",
        _target(),
        project,
        (numpy_shadow,),
        base_wheels,
        {project.digest: project_path, numpy_shadow.digest: args.numpy_shadow.resolve()},
        import_root_overrides={numpy_shadow.distribution: ("declared_new_root",)},
    )
    _request(
        root,
        "lying-import-root",
        lying_import_root,
        root / "environment-lying-import-root",
        args.python,
    )

    duplicate_import_root = _release(
        inputs / "duplicate-import-root",
        _target(),
        project,
        (shared_root_a, shared_root_b),
        base_wheels,
        {
            project.digest: project_path,
            shared_root_a.digest: args.shared_root_a.resolve(),
            shared_root_b.digest: args.shared_root_b.resolve(),
        },
    )
    _request(
        root,
        "duplicate-import-root",
        duplicate_import_root,
        root / "environment-duplicate-import-root",
        args.python,
    )


def _release(
    root: Path,
    target: Mapping[str, str],
    project: Wheel,
    resolved_wheels: Sequence[Wheel],
    wheelhouse_wheels: Sequence[Wheel],
    wheel_paths: Mapping[str, Path],
    *,
    import_root_overrides: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    root.mkdir()
    base_distributions, base_import_roots = _installed_base_ownership()
    cpu = target["accelerator_backend"] == "none"
    profile = {
        "accelerator_build": "cpu" if cpu else target["accelerator_abi"],
        "os_cpu": "linux-x86",
        "python_abi": target["python_abi"],
        "torch_release": "2.13.0",
    }
    wheelhouse = _canonical(
        {
            "base_distributions": base_distributions,
            "base_import_roots": base_import_roots,
            "compatibility_profile": profile,
            "format": "WheelhouseManifest/3",
            "minimum_cuda_version": "none" if cpu else "12.6",
            "minimum_driver_version": "none" if cpu else "560.28.03",
            "platform_target": dict(target),
            "wheels": [_wheel(wheel, import_roots=True) for wheel in wheelhouse_wheels],
        }
    )
    selected_rows = []
    for wheel in resolved_wheels:
        row = _wheel(wheel, import_roots=True)
        if import_root_overrides and wheel.distribution in import_root_overrides:
            row["import_roots"] = list(import_root_overrides[wheel.distribution])
        selected_rows.append(_nested_wheel(row))
    project_fact = _nested_wheel(_wheel(project, import_roots=True))
    package_release = _canonical(
        {
            "dependency_wheels": [],
            "format": "cozy.package.release/1",
            "project_wheel": project_fact,
            "requires_python": ">=3.11",
            "requirements": [],
            "source_manifest": _ref(b'{"entries":[]}'),
        }
    )
    environment = pb.Environment(
        wheelhouse_manifest=pb.Ref(digest=documents.digest_of(wheelhouse), length=len(wheelhouse)),
        wheels=[_proto_wheel(row) for row in selected_rows],
    )
    return {
        "package_release": _write(root / "package-release.json", package_release),
        "environment": _write(
            root / "environment.json",
            _canonical({**documents.body(environment), "wheels": selected_rows}),
        ),
        "wheelhouse_manifest": _write(root / "wheelhouse.json", wheelhouse),
        "wheel_files": dict(wheel_paths),
    }


def _nested_wheel(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ref": {"digest": row["digest"], "length": row["length"]},
        "distribution": row["distribution"],
        "filename": row["filename"],
        "import_roots": row["import_roots"],
        "tags": row["tags"],
        "version": row["version"],
    }


def _proto_wheel(row: Mapping[str, Any]) -> pb.WheelFact:
    ref = cast(Mapping[str, Any], row["ref"])
    return pb.WheelFact(
        ref=pb.Ref(digest=documents.raw(str(ref["digest"])), length=int(ref["length"])),
        distribution=str(row["distribution"]),
        filename=str(row["filename"]),
        import_roots=[str(value) for value in cast(Sequence[Any], row["import_roots"])],
        tags=[str(value) for value in cast(Sequence[Any], row["tags"])],
        version=str(row["version"]),
    )


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
    value = _parse_canonical(receipt)
    _require(isinstance(value, dict), f"{name} receipt is not an object")
    receipt_value = cast(dict[str, Any], value)
    rows = receipt_value.get("overlay_wheels")
    _require(isinstance(rows, list), f"{name} receipt wheel rows are absent")
    wheel_rows = cast(list[dict[str, Any]], rows)
    content = _canonical(
        {
            "base_family_digest": receipt_value.get("base_family_digest"),
            "wheel_digests": sorted(str(row.get("digest")) for row in wheel_rows),
        }
    )
    content_digest = _digest(content)
    _require(
        generation.name == content_digest.removeprefix("sha256:"),
        f"{name} installed content is not keyed by base family plus wheel digests",
    )
    _require(generation.is_dir(), f"{name} installed content is absent")
    site = generation / "site-packages"
    for metadata in ("package.toml", "pyproject.toml", "uv.lock"):
        _require((site / metadata).is_file(), f"{name} omitted project metadata {metadata}")
    _require(
        (generation / "package-overlay-content.json").read_bytes() == content,
        f"{name} installed-content identity disagrees with its receipt",
    )
    _require(
        not list((root / f"environment-{name}").glob(".staging-*")), f"{name} staging survived"
    )
    external = root / f"receipt-{name}.json"
    _require(external.read_bytes() == receipt, f"{name} external receipt changed while read")
    _require(external.stat().st_mode & 0o222 == 0, f"{name} external receipt is writable")
    _require(
        receipt_value.get("format") == RECEIPT_FORMAT,
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
