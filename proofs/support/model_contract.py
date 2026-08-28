"""Prepare and inspect exact documents for the public model-contract proof.

This module owns document and wheel mechanics that shell cannot safely express. The shell
driver owns every process transition and invokes the installed public Runtime commands.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import platform
import stat
import sys
import zipfile
from collections.abc import Mapping, Sequence
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, cast

from proof_data import ProofFailure
from proof_data import canonical as _canonical
from proof_data import digest as _digest
from proof_data import inventory as _inventory
from proof_data import json_file as _json
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
    prepare.add_argument("descriptor", type=Path)
    prepare.add_argument("runtime_wheel", type=Path)
    prepare.add_argument("endpoint_wheel", type=Path)
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
    endpoint_path = args.endpoint_wheel.resolve()
    torch_path = args.torch_wheel.resolve()
    dependency_paths = [
        _pack_installed(wheels, name)
        for name in ("grpcio", "msgspec", "protobuf", "typing-extensions")
    ]
    project = _inventory(endpoint_path)
    base_pairs = sorted(
        [(_inventory(path), path) for path in (runtime_path, torch_path, *dependency_paths)],
        key=lambda pair: pair[0].distribution,
    )
    target = _target()
    wheelhouse = _canonical(
        {
            "format": "WheelhouseManifest/1",
            "platform_target": target,
            "wheels": [_wheel(entry) for entry, _ in base_pairs],
        }
    )
    resolved = _canonical(
        {
            "format": "ResolvedWheelSet/1",
            "lock_digest": _digest(b"proof-lock"),
            "platform_target": target,
            "wheelhouse_manifest_digest": _digest(wheelhouse),
            "wheels": [],
        }
    )
    config = _canonical({"width": 4})
    descriptor_path = args.descriptor.resolve()
    descriptor_bytes = descriptor_path.read_bytes()
    descriptor_doc = _parse_json(descriptor_bytes)
    if not isinstance(descriptor_doc, dict) or set(descriptor_doc) != {
        "application",
        "entrypoints",
        "format",
        "jobs",
    }:
        raise ProofFailure("fixture descriptor is not one closed EndpointDescriptor/1")
    _require(
        descriptor_doc.get("format") == "cozy.endpoint.descriptor/1",
        "fixture descriptor has the wrong format",
    )
    descriptor_bytes = _canonical(descriptor_doc)
    descriptor_path = _write(inputs / "endpoint-descriptor.json", descriptor_bytes)
    hidden = [
        row["name"]
        for row in descriptor_doc["entrypoints"]
        if isinstance(row, dict) and row.get("hidden") is True
    ]
    _require(hidden == ["hidden_render"], f"unexpected hidden functions: {hidden!r}")
    bundle = _canonical(
        {
            "descriptor": _ref(descriptor_bytes),
            "endpoint_release_id": "proof-release",
            "evaluated_config": _ref(config),
            "format": "tensorhub.endpoint_bundle/2",
            "project_wheel": _wheel(project),
            "resolved_wheel_set": _ref(resolved),
            "source_archive": _ref(b"proof-source"),
            "source_lock": _ref(b"proof-lock"),
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
    paths = {
        "endpoint_environment_spec": _write(inputs / "environment.json", spec),
        "endpoint_bundle": _write(inputs / "bundle.json", bundle),
        "resolved_wheel_set": _write(inputs / "resolved.json", resolved),
        "wheelhouse_manifest": _write(inputs / "wheelhouse.json", wheelhouse),
    }
    config_path = _write(inputs / "model-config.json", config)
    wheel_files = {project.digest: str(endpoint_path)}
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
            "descriptor": str(descriptor_path),
            "descriptor_digest": _digest(descriptor_bytes),
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
            "endpoint_descriptor": {
                "digest": state["descriptor_digest"],
                "length": state["descriptor_length"],
                "path": state["descriptor"],
            },
            "endpoint_release_id": "proof-release",
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


def _pack_installed(root: Path, distribution: str) -> Path:
    installed = importlib.metadata.distribution(distribution)
    files: dict[str, bytes] = {}
    record = ""
    for member in installed.files or ():
        name = PurePosixPath(str(member)).as_posix()
        if ".." in PurePosixPath(name).parts or name.endswith(
            ("/INSTALLER", "/RECORD", "/REQUESTED", "/direct_url.json", "/uv_cache.json")
        ):
            continue
        path = cast(Path, installed.locate_file(member))
        if path.is_file():
            files[name] = path.read_bytes()
        if name.endswith(".dist-info/METADATA"):
            record = str(PurePosixPath(name).parent / "RECORD")
    _require(
        bool(record and files), f"installed distribution has no wheel inventory: {distribution}"
    )
    rows = [[name, _record_hash(raw), str(len(raw))] for name, raw in sorted(files.items())]
    rows.append([record, "", ""])
    rendered = io.StringIO(newline="")
    csv.writer(rendered, lineterminator="\n").writerows(rows)
    files[record] = rendered.getvalue().encode()
    wheel_meta = next(raw for name, raw in files.items() if name.endswith(".dist-info/WHEEL"))
    tags = BytesParser().parsebytes(wheel_meta).get_all("Tag") or []
    _require(bool(tags), f"installed distribution has no wheel tag: {distribution}")
    normalized = str(installed.metadata["Name"]).replace("-", "_")
    path = root / f"{normalized}-{installed.version}-{tags[0]}.whl"
    _zip(path, files)
    return path


def _target() -> dict[str, str]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "accelerator_abi": "none",
        "accelerator_backend": "none",
        "libc": f"{libc_name}{libc_version}",
        "os_arch": "linux/amd64",
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def _record_hash(raw: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
    return "sha256=" + encoded


def _zip(path: Path, files: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            wheel.writestr(info, raw)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


if __name__ == "__main__":
    raise SystemExit(main())
