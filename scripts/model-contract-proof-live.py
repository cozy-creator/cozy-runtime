#!/usr/bin/env python
"""Cold/rebuild/red proof for the installed-environment model-contract seat."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, cast

from cozy_runtime.internal import canonical, contract, descriptor
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.environment import WheelInventoryEntry
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT_REQUEST_FORMAT = "cozy.runtime.EnvironmentProofRequest/1"
MODEL_REQUEST_FORMAT = "cozy.runtime.ModelConstructionProofRequest/1"

ENDPOINT_SOURCE = b"""from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Config, Loader, Model, uses_components


class Request(msgspec.Struct):
    value: int


class Result(msgspec.Struct):
    value: int


class Pipeline:
    def __init__(self, core: object) -> None:
        self.components = {"core": core}


def build(config: Config) -> Pipeline:
    import torch

    if config.as_int("width") != 4:
        raise ValueError("wrong exact config")
    return Pipeline(torch.nn.Module())


class ToyModel(Model[Pipeline]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build)

    @uses_components("core")
    def touch(self) -> None:
        return None


class HiddenToyModel(ToyModel):
    pass


app = App()


@app.entrypoint
def metadata(payload: Request) -> Result:
    return Result(payload.value)


@app.entrypoint
def render(payload: Request, model: ToyModel) -> Result:
    model.touch()
    return Result(payload.value)


@app.entrypoint(hidden=True)
def hidden_render(payload: Request, model: HiddenToyModel) -> Result:
    model.touch()
    return Result(payload.value)
"""

TORCH_SOURCE = b"""__version__ = "0.0.0-proof"


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class device(_Context):
    def __init__(self, name):
        self.type = str(name)

    def __str__(self):
        return self.type


class _Future:
    value = False

    @classmethod
    def get_swap_module_params_on_conversion(cls):
        return cls.value

    @classmethod
    def set_swap_module_params_on_conversion(cls, value):
        cls.value = value


__future__ = _Future()


class _Cuda:
    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def get_device_capability():
        return (0, 0)

    @staticmethod
    def get_device_name():
        return "none"

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0


cuda = _Cuda()


class _Storage:
    _cdata = 1


class Tensor:
    shape = (4,)
    dtype = "bfloat16"
    device = device("meta")
    nbytes = 8

    def untyped_storage(self):
        return _Storage()

    def storage_offset(self):
        return 0


class Parameter(Tensor):
    pass


class Module:
    training = False
    _non_persistent_buffers_set = set()

    def __init__(self):
        self._parameters = {}
        self._buffers = {}
        self.register_parameter("weight", Parameter())

    def register_parameter(self, name, value):
        self._parameters[name] = value
        setattr(self, name, value)

    def register_buffer(self, name, value, persistent=True):
        self._buffers[name] = value
        setattr(self, name, value)

    def state_dict(self):
        return {"weight": self.weight}

    def named_parameters(self, remove_duplicate=False):
        return (("weight", self.weight),)

    def named_buffers(self, remove_duplicate=False):
        return ()

    def named_modules(self):
        return (("", self),)

    @staticmethod
    def compile(*args, **kwargs):
        return None


class _NN:
    Module = Module
    Parameter = Parameter


nn = _NN()


class TorchFunctionMode(_Context):
    pass


class _Overrides:
    TorchFunctionMode = TorchFunctionMode


overrides = _Overrides()


def compile(*args, **kwargs):
    return None
"""


def main() -> None:
    helpers = runpy.run_path(str(ROOT / "scripts" / "environment-materializer-live.py"))
    build_wheel = cast(Callable[..., Path], helpers["build_wheel"])
    writable_cleanup = cast(Callable[[Path], None], helpers["writable_cleanup"])
    root = Path(tempfile.mkdtemp(prefix="model-contract-proof-live-"))
    try:
        wheels = root / "wheels"
        wheels.mkdir()
        runtime_path = _build_runtime(wheels)
        endpoint_path = build_wheel(
            wheels, "proof-endpoint", "1.0.0", "proof_endpoint", ENDPOINT_SOURCE
        )
        torch_path = build_wheel(wheels, "torch", "0.0.0", "torch", TORCH_SOURCE)
        dependency_paths = [
            _pack_installed(wheels, name)
            for name in ("grpcio", "msgspec", "protobuf", "typing-extensions")
        ]

        project = _inventory(endpoint_path, "project")
        resolved_pairs = sorted(
            [
                (_inventory(path, "bundle"), path)
                for path in [runtime_path, torch_path, *dependency_paths]
            ],
            key=lambda pair: pair[0].distribution,
        )
        resolved_entries = [entry for entry, _ in resolved_pairs]
        target = _target()
        wheelhouse = canonical.write(
            {"format": "WheelhouseManifest/1", "platform_target": target, "wheels": []}
        )
        resolved = canonical.write(
            {
                "format": "ResolvedWheelSet/1",
                "lock_digest": _digest(b"proof-lock"),
                "platform_target": target,
                "wheelhouse_manifest_digest": _digest(wheelhouse),
                "wheels": [_wheel(entry) for entry in resolved_entries],
            }
        )
        config = canonical.write({"width": 4})
        descriptor_path = _descriptor(root)
        descriptor_bytes = descriptor_path.read_bytes()
        descriptor_doc = json.loads(descriptor_bytes)
        assert isinstance(descriptor_doc, dict)
        assert [
            row["name"]
            for row in descriptor_doc["entrypoints"]
            if isinstance(row, dict) and row.get("hidden") is True
        ] == ["hidden_render"]
        bundle = canonical.write(
            {
                "descriptor": _ref(descriptor_bytes),
                "endpoint_release_id": "proof-release",
                "evaluated_config": _ref(config),
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
        inputs = root / "inputs"
        inputs.mkdir()
        paths = {
            "endpoint_environment_spec": _write(inputs / "environment.json", spec),
            "endpoint_bundle": _write(inputs / "bundle.json", bundle),
            "resolved_wheel_set": _write(inputs / "resolved.json", resolved),
            "wheelhouse_manifest": _write(inputs / "wheelhouse.json", wheelhouse),
        }
        config_path = _write(inputs / "model-config.json", config)
        uv = Path(shutil.which("uv") or "").resolve()
        proof = Path(sys.executable).parent / "cozy-environment-proof"
        assert uv.is_file() and proof.is_file()
        wheel_files = {project.digest: str(endpoint_path.resolve())}
        wheel_files.update({entry.digest: str(path.resolve()) for entry, path in resolved_pairs})
        base_request: dict[str, Any] = {
            **{name: str(path) for name, path in paths.items()},
            "format": ENVIRONMENT_REQUEST_FORMAT,
            "python": str(Path(sys.executable).resolve()),
            "uv": str(uv),
            "wheel_files": wheel_files,
        }

        receipts: list[bytes] = []
        generations: list[Path] = []
        outputs: list[dict[str, bytes]] = []
        observations: list[dict[str, Any]] = []
        for name in ("cold", "rebuild"):
            environment_request = {
                **base_request,
                "environment_root": str(root / f"environment-{name}"),
            }
            environment_request_path = _json(root / f"environment-{name}.json", environment_request)
            receipt_path = root / f"receipt-{name}.json"
            result = _run(proof, environment_request_path, receipt_path)
            assert result.returncode == 0, result.stderr
            observation = json.loads(result.stdout)
            generation = Path(observation["generation"])
            generations.append(generation)
            receipt = receipt_path.read_bytes()
            model_request = _model_request(
                environment_spec=spec,
                receipt=receipt,
                project=project,
                descriptor_path=descriptor_path,
                descriptor_bytes=descriptor_bytes,
                config_path=config_path,
                config=config,
            )
            model_request_path = _json(root / f"model-{name}.json", model_request)
            output = root / f"model-output-{name}"
            command = generation / "venv" / "bin" / "cozy-model-contract-proof"
            assert command.is_file(), sorted(path.name for path in command.parent.iterdir())
            result = _run(command, model_request_path, output)
            assert result.returncode == 0, result.stderr
            observations.append(json.loads(result.stdout))
            receipts.append(receipt)
            outputs.append(_tree(output))

        assert receipts[0] == receipts[1]
        assert outputs[0] == outputs[1]
        assert observations[0]["digest"] == observations[1]["digest"]
        index = canonical.parse_canonical(outputs[0]["model-construction-proof.json"])
        assert (
            isinstance(index, dict) and index["format"] == "cozy.runtime.ModelConstructionProof/1"
        )
        assert [row["name"] for row in index["functions"]] == ["metadata", "render"]
        assert index["functions"][0]["bindings"] == []
        contracts = [raw for path, raw in outputs[0].items() if path.startswith("contracts/")]
        assert len(contracts) == 1
        parsed = contract.read_contract(contracts[0])
        assert parsed.candidate.config_digest == _digest(config)
        assert [row.key for row in parsed.logical_tensors] == ["core.weight"]

        red_request = _model_request(
            environment_spec=spec,
            receipt=receipts[0],
            project=project,
            descriptor_path=descriptor_path,
            descriptor_bytes=descriptor_bytes,
            config_path=config_path,
            config=config,
        )
        red_request["bindings"] = []
        red_path = _json(root / "model-red.json", red_request)
        red_output = root / "model-output-red"
        refused = _run(
            generations[0] / "venv" / "bin" / "cozy-model-contract-proof",
            red_path,
            red_output,
        )
        assert refused.returncode == 6 and "model_contract_binding_set_mismatch" in refused.stderr
        assert not red_output.exists()

        print(
            "model contract proof live: independent offline materializations produced "
            "byte-identical receipt/index/MCC bytes; hidden function omitted; "
            "missing binding refused before output; "
            f"mcc={_digest(contracts[0])}"
        )
    finally:
        writable_cleanup(root)


def _build_runtime(root: Path) -> Path:
    result = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(root), str(ROOT)],
        cwd=ROOT,
        env={**os.environ, "UV_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    paths = list(root.glob("cozy_runtime-*.whl"))
    assert len(paths) == 1
    return paths[0]


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
    assert record and files
    rows = [[name, _record_hash(raw), str(len(raw))] for name, raw in sorted(files.items())]
    rows.append([record, "", ""])
    rendered = io.StringIO(newline="")
    csv.writer(rendered, lineterminator="\n").writerows(rows)
    files[record] = rendered.getvalue().encode()
    wheel_meta = next(raw for name, raw in files.items() if name.endswith(".dist-info/WHEEL"))
    tags = BytesParser().parsebytes(wheel_meta).get_all("Tag") or []
    assert tags
    normalized = str(installed.metadata["Name"]).replace("-", "_")
    path = root / f"{normalized}-{installed.version}-{tags[0]}.whl"
    _zip(path, files)
    return path


def _inventory(path: Path, owner: str) -> WheelInventoryEntry:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_name = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        wheel = BytesParser().parsebytes(archive.read(wheel_name))
    raw = path.read_bytes()
    return WheelInventoryEntry(
        distribution=str(metadata["Name"]),
        version=str(metadata["Version"]),
        filename=path.name,
        digest=_digest(raw),
        length=len(raw),
        tags=tuple(sorted(wheel.get_all("Tag") or [])),
        owner=owner,
    )


def _descriptor(root: Path) -> Path:
    project = root / "source"
    package = project / "proof_endpoint"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(ENDPOINT_SOURCE)
    (project / "endpoint.toml").write_text('[application]\nobject = "proof_endpoint:app"\n')
    found = discover(project)
    path = root / "endpoint.descriptor.json"
    path.write_text(canonical.pretty(descriptor.document(descriptor.build(found))))
    return path.resolve()


def _target() -> dict[str, str]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "accelerator_abi": "none",
        "accelerator_backend": "none",
        "libc": f"{libc_name}{libc_version}",
        "os_arch": "linux/amd64",
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def _model_request(
    *,
    environment_spec: bytes,
    receipt: bytes,
    project: WheelInventoryEntry,
    descriptor_path: Path,
    descriptor_bytes: bytes,
    config_path: Path,
    config: bytes,
) -> dict[str, Any]:
    return {
        "bindings": [
            {
                "config": {
                    "digest": _digest(config),
                    "length": len(config),
                    "path": str(config_path),
                },
                "hardware_variants": ["sm90"],
                "path": "render.models.model",
                "snapshot": _digest(b"proof-snapshot"),
            }
        ],
        "endpoint_descriptor": {
            "digest": _digest(descriptor_bytes),
            "length": len(descriptor_bytes),
            "path": str(descriptor_path),
        },
        "endpoint_release_id": "proof-release",
        "environment_spec_digest": _digest(environment_spec),
        "format": MODEL_REQUEST_FORMAT,
        "installed_environment_receipt": {"digest": _digest(receipt), "length": len(receipt)},
        "project_wheel_digest": project.digest,
    }


def _run(command: Path, request: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(command), str(request), str(output)],
        cwd="/",
        env={
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "PATH": os.environ["PATH"],
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _wheel(entry: WheelInventoryEntry) -> dict[str, Any]:
    return {
        "digest": entry.digest,
        "distribution": entry.distribution,
        "filename": entry.filename,
        "length": entry.length,
        "tags": list(entry.tags),
        "version": entry.version,
    }


def _ref(raw: bytes) -> dict[str, Any]:
    return {"digest": _digest(raw), "length": len(raw)}


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


def _json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))
    return path.resolve()


def _write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path.resolve()


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    main()
