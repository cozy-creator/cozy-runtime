"""Prepare and inspect exact documents for the public model-contract proof.

This module owns document and wheel mechanics that shell cannot safely express. The shell
driver owns every process transition and invokes the installed public Runtime commands.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from proof_data import ProofFailure
from proof_data import canonical as _canonical
from proof_data import digest as _digest
from proof_data import installed_base_ownership as _installed_base_ownership
from proof_data import inventory as _inventory
from proof_data import json_file as _json
from proof_data import pack_installed as _pack_installed
from proof_data import parse_canonical as _parse_canonical
from proof_data import parse_json as _parse_json
from proof_data import read_json as _read_json
from proof_data import ref as _ref
from proof_data import require as _require
from proof_data import wheel as _wheel
from proof_data import write as _write

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ENVIRONMENT_REQUEST_FORMAT = "cozy.runtime.EnvironmentProofRequest/1"
MODEL_REQUEST_FORMAT = "cozy.runtime.ModelConstructionProofRequest/1"
STATE_FORMAT = "cozy.runtime.proof.ModelContractState/1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("root", type=Path)
    prepare.add_argument("package_descriptor", type=Path)
    prepare.add_argument("runtime_wheel", type=Path)
    prepare.add_argument("package_wheel", type=Path)
    prepare.add_argument("torch_wheel", type=Path)
    prepare.add_argument("python", type=Path)

    model = sub.add_parser("model-request")
    model.add_argument("root", type=Path)
    model.add_argument("name", choices=("cold", "rebuild"))

    generation = sub.add_parser("generation")
    generation.add_argument("observation", type=Path)

    red = sub.add_parser("red-request")
    red.add_argument("root", type=Path)

    verify = sub.add_parser("verify")
    verify.add_argument("root", type=Path)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "model-request":
        _write_model_request(args.root, args.name)
    elif args.command == "generation":
        print(_observation(args.observation)["generation"])
    elif args.command == "red-request":
        request = _read_json(args.root / "model-cold.json")
        if not isinstance(request, dict):
            raise ProofFailure("cold model request is not an object")
        request["bindings"] = []
        _json(args.root / "model-red.json", request)
    else:
        _verify(args.root)
    return 0


def _prepare(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    wheels = root / "wheels"
    inputs = root / "inputs"
    wheels.mkdir()
    inputs.mkdir()

    runtime_path = args.runtime_wheel.resolve()
    package_path = args.package_wheel.resolve()
    dependency_paths = [
        _pack_installed(wheels, name)
        for name in ("grpcio", "msgspec", "protobuf", "typing-extensions")
    ]
    project = _inventory(package_path)
    base_pairs = sorted(
        [(_inventory(path), path) for path in (runtime_path, *dependency_paths)],
        key=lambda pair: pair[0].distribution,
    )
    _require(len(base_pairs) == 5, "proof base does not carry five control wheels")
    target = _target()
    profile = {
        "accelerator_build": "cpu",
        "os_cpu": "linux-x86",
        "python_abi": target["python_abi"],
        "torch_release": "2.13.0",
    }
    base_distributions, base_import_roots = _installed_base_ownership()
    wheelhouse = _canonical(
        {
            "base_distributions": base_distributions,
            "base_import_roots": base_import_roots,
            "compatibility_profile": profile,
            "format": "WheelhouseManifest/3",
            "minimum_cuda_version": "none",
            "minimum_driver_version": "none",
            "platform_target": target,
            "wheels": [_wheel(entry, import_roots=True) for entry, _ in base_pairs],
        }
    )
    resolved = _canonical(
        {
            "compatibility_profile": profile,
            "format": "ResolvedWheelSet/3",
            "overlay_bytes": 0,
            "platform_target": target,
            "resolution_lock_digest": _digest(b"proof-resolution-lock"),
            "resolution_lock_length": len(b"proof-resolution-lock"),
            "wheelhouse_manifest_digest": _digest(wheelhouse),
            "wheels": [],
        }
    )
    config = _canonical({"width": 4})
    descriptor_path = args.package_descriptor.resolve()
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor_doc = _parse_json(descriptor_bytes)
    if not isinstance(descriptor_doc, dict) or set(descriptor_doc) != {
        "application",
        "entrypoints",
        "format",
        "jobs",
    }:
        raise ProofFailure("fixture descriptor is not one closed PackageDescriptor/1")
    _require(
        descriptor_doc.get("format") == "cozy.package.descriptor/1",
        "fixture descriptor has the wrong format",
    )
    descriptor_bytes = _canonical(descriptor_doc)
    descriptor_path = _write(inputs / "package-descriptor.json", descriptor_bytes)
    names = [row.get("name") for row in descriptor_doc["entrypoints"] if isinstance(row, dict)]
    _require(names == ["metadata", "render"], f"unexpected published functions: {names!r}")
    _require(
        all("hidden" not in row for row in descriptor_doc["entrypoints"] if isinstance(row, dict)),
        "the descriptor still carries the retired hidden field",
    )
    bundle = _canonical(
        {
            "package_descriptor": _ref(descriptor_bytes),
            "package_release_id": "proof-release",
            "format": "tensorhub.package_bundle/2",
            "project_wheel": _wheel(project, import_roots=True),
            "resolved_wheel_set": _ref(resolved),
            "source_archive": _ref(b"proof-source"),
            "source_lock": _ref(b"proof-lock"),
            "wheelhouse_manifest": _ref(wheelhouse),
        }
    )
    spec = documents.canonical_bytes(
        pb.PackageEnvironmentSpec(
            compatibility_profile=pb.CompatibilityProfile(**profile),
            platform_target=pb.PlatformTarget(**target),
            package_bundle_digest=documents.digest_of(bundle),
            project_wheel_digest=documents.raw(project.digest),
            wheelhouse_manifest_digest=documents.digest_of(wheelhouse),
        )
    )
    paths = {
        "package_environment_spec": _write(inputs / "environment.json", spec),
        "package_bundle": _write(inputs / "bundle.json", bundle),
        "resolved_wheel_set": _write(inputs / "resolved.json", resolved),
        "wheelhouse_manifest": _write(inputs / "wheelhouse.json", wheelhouse),
    }
    config_path = _write(inputs / "model-config.json", config)
    wheel_files = {project.digest: str(package_path)}
    base_request: dict[str, Any] = {
        **{name: str(path) for name, path in paths.items()},
        "format": ENVIRONMENT_REQUEST_FORMAT,
        "python": str(args.python.resolve()),
        "wheel_files": wheel_files,
    }
    for name in ("cold", "rebuild"):
        _json(
            root / f"environment-{name}.json",
            {**base_request, "environment_root": str(root / f"environment-{name}")},
        )
    _json(
        root / "proof-state.json",
        {
            "config": str(config_path),
            "config_digest": _digest(config),
            "config_length": len(config),
            "package_descriptor": str(descriptor_path),
            "package_descriptor_digest": _digest(descriptor_bytes),
            "descriptor_length": len(descriptor_bytes),
            "environment_spec_digest": _digest(spec),
            "format": STATE_FORMAT,
            "project_wheel_digest": project.digest,
            "wheelhouse_manifest": str(paths["wheelhouse_manifest"]),
        },
    )


def _write_model_request(root: Path, name: str) -> None:
    state = _state(root)
    receipt = (root / f"receipt-{name}.json").read_bytes()
    environment = _observation(root / f"environment-{name}.stdout")
    _json(
        root / f"model-{name}.json",
        {
            "bindings": [
                {
                    "config": {
                        "digest": state["config_digest"],
                        "length": state["config_length"],
                        "path": state["config"],
                    },
                    "hardware_variants": ["sm90"],
                    "path": "render.models.model",
                    "snapshot": _digest(b"proof-snapshot"),
                }
            ],
            "package_descriptor": {
                "digest": state["package_descriptor_digest"],
                "length": state["descriptor_length"],
                "path": state["package_descriptor"],
            },
            "package_release_id": "proof-release",
            "environment_generation": environment["generation"],
            "environment_spec_digest": state["environment_spec_digest"],
            "format": MODEL_REQUEST_FORMAT,
            "installed_environment_receipt": {
                "digest": _digest(receipt),
                "length": len(receipt),
            },
            "project_wheel_digest": state["project_wheel_digest"],
            "wheelhouse_manifest": state["wheelhouse_manifest"],
        },
    )


def _verify(root: Path) -> None:
    state = _state(root)
    receipts = [(root / f"receipt-{name}.json").read_bytes() for name in ("cold", "rebuild")]
    _require(receipts[0] == receipts[1], "cold and rebuild receipts differ")
    environment_observations = [
        _observation(root / f"environment-{name}.stdout") for name in ("cold", "rebuild")
    ]
    _require(
        all(observation["reused"] is False for observation in environment_observations),
        "an independent materialization reported reuse",
    )
    _require(
        all(
            observation["digest"] == _digest(receipt) and observation["length"] == len(receipt)
            for observation, receipt in zip(environment_observations, receipts, strict=True)
        ),
        "an environment observation does not identify its stored receipt bytes",
    )
    outputs = [_tree(root / f"model-output-{name}") for name in ("cold", "rebuild")]
    _require(outputs[0] == outputs[1], "cold and rebuild model-contract output trees differ")
    observations = [_observation(root / f"model-{name}.stdout") for name in ("cold", "rebuild")]
    _require(
        observations[0]["digest"] == observations[1]["digest"],
        "cold and rebuild model-contract observation digests differ",
    )
    index = _parse_canonical(outputs[0]["model-construction-proof.json"])
    if not isinstance(index, dict):
        raise ProofFailure("model-construction proof index is not an object")
    _require(
        index["format"] == "cozy.runtime.ModelConstructionProof/1",
        "model-construction proof index has the wrong format",
    )
    names = [row["name"] for row in index["functions"]]
    _require(names == ["metadata", "render"], f"unexpected visible functions: {names!r}")
    _require(index["functions"][0]["bindings"] == [], "metadata unexpectedly has a binding")
    contracts = [raw for path, raw in outputs[0].items() if path.startswith("contracts/")]
    _require(len(contracts) == 1, f"expected one MCC, found {len(contracts)}")
    parsed = _parse_canonical(contracts[0])
    if not isinstance(parsed, dict):
        raise ProofFailure("MCC is not an object")
    candidate = parsed.get("candidate")
    logical_rows = parsed.get("logical_tensors")
    if not isinstance(candidate, dict) or not isinstance(logical_rows, list):
        raise ProofFailure("MCC candidate or logical tensor census is malformed")
    _require(
        candidate.get("config_digest") == state["config_digest"],
        "MCC does not retain the exact config digest",
    )
    logical_tensors = [row.get("key") for row in logical_rows if isinstance(row, dict)]
    _require(
        logical_tensors == ["core.weight"],
        f"unexpected logical tensors: {logical_tensors!r}",
    )
    print(
        "model contract proof: independent offline materializations produced byte-identical "
        "receipt/index/MCC bytes; hidden function omitted; missing binding exited 6 before "
        f"output; mcc={_digest(contracts[0])}"
    )


def _state(root: Path) -> dict[str, Any]:
    value = _read_json(root / "proof-state.json")
    if not isinstance(value, dict) or value.get("format") != STATE_FORMAT:
        raise ProofFailure("proof state is not the expected closed local format")
    return value


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
        "libc": f"{libc_name}{libc_version}",
        "os_arch": "linux/amd64",
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


if __name__ == "__main__":
    raise SystemExit(main())
