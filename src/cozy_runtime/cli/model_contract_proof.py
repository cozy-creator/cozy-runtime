"""Author-side writer of publish-time derive evidence (cr-067).

Runs on the author's machine inside the project venv (`uv run --locked`), at publish.
Per declared (model source x hardware variant) pair it delegates package import and
derive-mode execution to one bounded child whose audit hooks allow no network, weight
reads, writes, or subprocesses, and publishes one complete digest-addressed envelope --
`result.json` index plus `contracts/` and `code-topologies/` sibling documents -- by
atomic rename. The envelope binds the release VERSION, never a release digest: the
digest only exists once tensorhub finalizes the upload this envelope rides in.
The hub validates these documents statically and never executes package code (th-106).
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.author import Artifact, Config, ConformanceError, ModelBinding
from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import canonical, contract, sandbox
from cozy_runtime.internal import descriptor as package_descriptor
from cozy_runtime.internal.derive import DeriveError, derive
from cozy_runtime.internal.descriptor import StalePackageDescriptor
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.encoding import TORCH_DTYPES, launch_providers
from cozy_runtime.internal.exits import Exit

INDEX_NAME = "result.json"
CONTRACTS_DIR = "contracts"
TOPOLOGIES_DIR = "code-topologies"
RESULT_PREFIX = "COZY_MODEL_CONTRACT_PROOF="
MEMORY_CAP_BYTES = 24 << 30
CPU_CAP_SECONDS = 600
WALL_CAP_SECONDS = 900
_VARIANT = re.compile(r"^(?:cpu|sm[0-9]{2,})$")

INDEX_FORMAT = "cozy.package.derive-evidence/1"

REQUEST_FIELDS = {
    "bindings",
    "package_descriptor",
    "package",
    "project",
}
REF_FIELDS = {"digest", "length"}
LOCAL_REF_FIELDS = {"digest", "length", "path"}
BINDING_FIELDS = {"config", "hardware_variants", "path", "snapshot"}


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
class BindingRequest:
    path: str
    snapshot: str
    config: LocalDocumentRef
    hardware_variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProofRequest:
    project: Path
    package: Mapping[str, Any]
    package_descriptor: LocalDocumentRef
    bindings: tuple[BindingRequest, ...]

    @property
    def release_identity(self) -> str:
        """The version-bound identity every derived Candidate carries."""
        return f"{self.package['package']}@{self.package['release']}"


def main(argv: Sequence[str] | None = None) -> int:
    """Derive exact MCCs from one already materialized release environment."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--derive-child"]:
        return _child_entry(args[1:])

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
                "model_contract_output_invalid",
                "the evidence output must be outside the project tree",
                "choose a separate fresh output directory",
                Exit.validation,
            )
        # The author's own venv is the derivation environment: the parent runs inside
        # `uv run --locked`, so sys.executable resolves the exact locked interpreter.
        response = _run_child(Path(sys.executable), request_path)
        index, contracts, topologies = _decode_child(response, request)
        index_bytes = canonical.write(index)
        _publish(output, index_bytes, contracts, topologies)
        print(
            json.dumps(
                {
                    "contracts": len(contracts),
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
    except ProofRefusal as err:
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
    except OSError as err:
        emit_error(
            CliError(
                "model_contract_proof_io",
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
                f"unexpected model-contract proof fault: {type(err).__name__}: {err}",
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
    bindings_value = value["bindings"]
    if not isinstance(bindings_value, list):
        raise _request_error("bindings is not an array")
    bindings = tuple(_binding(row, index) for index, row in enumerate(bindings_value))
    paths = [row.path for row in bindings]
    if len(paths) != len(set(paths)):
        raise _request_error("bindings contains a duplicate descriptor model path")
    return ProofRequest(
        project=_existing_directory(value["project"], "project"),
        package=_package(value["package"]),
        package_descriptor=_local_ref(value["package_descriptor"], "package_descriptor"),
        bindings=bindings,
    )


def _binding(value: object, index: int) -> BindingRequest:
    name = f"bindings[{index}]"
    if not isinstance(value, dict) or set(value) != BINDING_FIELDS:
        raise _request_error(f"{name} does not have the closed binding field set")
    path = value["path"]
    if not isinstance(path, str) or not path or len(path) > 512 or path.strip() != path:
        raise _request_error(f"{name}.path is not one bounded non-empty string")
    variants = value["hardware_variants"]
    if (
        not isinstance(variants, list)
        or not variants
        or any(not isinstance(item, str) or _VARIANT.fullmatch(item) is None for item in variants)
        or variants != sorted(set(variants))
    ):
        raise _request_error(
            f"{name}.hardware_variants must be a sorted unique non-empty cpu|sm<major><minor> array"
        )
    return BindingRequest(
        path=path,
        snapshot=_digest(value["snapshot"], f"{name}.snapshot"),
        config=_local_ref(value["config"], f"{name}.config"),
        hardware_variants=tuple(variants),
    )


def _package(value: object) -> Mapping[str, Any]:
    # No release_digest: evidence is derived BEFORE finalize mints the manifest digest,
    # so the Candidate binds `<org>/<name>@<version>` (cr-067's circularity fix).
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
        "model_contract_proof_request_invalid",
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
            "model_contract_proof_output_conflict",
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


def _run_child(python: Path, request: Path) -> dict[str, Any]:
    child_env = sandbox.child_environment({})
    child_env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    result = subprocess.run(
        [
            str(python),
            "-I",
            # -I implies -E, which DISCARDS PYTHONDONTWRITEBYTECODE from the child
            # environment; -B is the flag form the isolation cannot strip. Without it,
            # importing any module whose bytecode cache is stale trips the sandbox's
            # no_filesystem_writes fence on the .pyc write (found live, cr-067 e2e).
            "-B",
            "-m",
            "cozy_runtime.cli.model_contract_proof",
            "--derive-child",
            str(request),
        ],
        cwd="/",
        env=child_env,
        capture_output=True,
        text=True,
        timeout=WALL_CAP_SECONDS,
        check=False,
    )
    encoded = next(
        (
            line.removeprefix(RESULT_PREFIX)
            for line in reversed(result.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        "",
    )
    if not encoded:
        detail = result.stderr.strip()[-1200:] or (
            f"child exited {result.returncode} without a result"
        )
        raise ProofRefusal("model_contract_child_failed", detail)
    try:
        raw = base64.b64decode(encoded, validate=True)
        response = canonical.parse_canonical(raw)
    except (ValueError, canonical.CanonicalError) as err:
        raise ProofRefusal("model_contract_child_invalid", str(err)) from err
    if not isinstance(response, dict):
        raise ProofRefusal("model_contract_child_invalid", "child result is not an object")
    if "error" in response:
        error_code = response.get("error")
        error_detail = response.get("detail")
        raise ProofRefusal(
            str(error_code or "model_contract_child_failed"), str(error_detail or "refused")
        )
    if result.returncode != 0:
        raise ProofRefusal("model_contract_child_failed", f"child exited {result.returncode}")
    return response


def _child_entry(args: Sequence[str]) -> int:
    if len(args) != 1:
        _emit_child({"error": "model_contract_child_usage", "detail": "one request path required"})
        return int(Exit.usage)
    try:
        request_path = _existing_file(args[0], "request")
        request = _read_request(request_path)
        descriptor_doc = _read_descriptor(request.package_descriptor)
        application = descriptor_doc.get("application")
        if not isinstance(application, str) or not application:
            raise ProofRefusal("model_contract_package_descriptor_invalid", "application is absent")
        _warm_torch()
        _warm_tensorfs()
        sandbox.install(allow_write_prefixes=())
        sandbox.bound(address_bytes=MEMORY_CAP_BYTES, cpu_seconds=CPU_CAP_SECONDS)
        found = discover(request.project)
        derived_descriptor = package_descriptor.build(found)
        package_descriptor.compare(
            descriptor_doc,
            derived_descriptor,
            ("the publishing descriptor", "the project source"),
        )
        if found.application != application:
            raise ProofRefusal(
                "model_contract_package_descriptor_invalid",
                "descriptor application does not name the discovered application",
            )
        response = _derive_all(request, found, derived_descriptor)
    except CliError as err:
        _emit_child({"error": err.name, "detail": err.message})
        return int(err.code)
    except ProofRefusal as err:
        _emit_child({"error": err.code, "detail": err.detail})
        return int(Exit.structural)
    except (ConformanceError, StalePackageDescriptor, contract.ContractError) as err:
        _emit_child(
            {
                "error": str(getattr(err, "code", "model_contract_conformance")),
                "detail": str(err),
            }
        )
        return int(Exit.structural)
    except (DeriveError, sandbox.FenceViolation) as err:
        name = getattr(err, "construct", None) or getattr(err, "fence", None)
        _emit_child({"error": str(name or "model_contract_derivation_refused"), "detail": str(err)})
        return int(Exit.structural)
    except Exception as err:
        _emit_child(
            {
                "error": "model_contract_derivation_internal",
                "detail": f"{type(err).__name__}: {err}",
            }
        )
        return int(Exit.internal)
    _emit_child(response)
    return int(Exit.ok)


def _warm_torch() -> None:
    """Warm torch's own lazy imports before rlimits, never package code or a GPU."""

    __import__("torch")
    for name in ("torch._dynamo", "torch._refs"):
        try:
            __import__(name)
        except ModuleNotFoundError as err:
            if err.name != name:
                raise


def _warm_tensorfs() -> None:
    """Load the pinned native facade before the no-write audit hook is installed."""

    import tensorfs

    if not hasattr(tensorfs, "CodeTensor") or not hasattr(tensorfs, "code_topology"):
        raise ProofRefusal(
            "model_contract_tensorfs_unavailable",
            "the base image TensorFS lacks the reviewed CodeTopology author seam",
        )


def _derive_all(
    request: ProofRequest, found: Any, descriptor_doc: Mapping[str, Any]
) -> dict[str, Any]:
    surfaces = {surface.name: surface for surface in found.surfaces if not surface.hidden}
    requested = {row.path: row for row in request.bindings}
    owners = {
        binding.path: surface.name
        for surface in surfaces.values()
        for binding in surface.model_bindings
    }
    unknown = sorted(set(requested) - set(owners))
    selected_names = {owners[path] for path in requested if path in owners}
    incomplete: dict[str, dict[str, list[str]]] = {}
    for name in selected_names:
        complete = {binding.path for binding in surfaces[name].model_bindings}
        supplied = set(requested) & complete
        if supplied != complete:
            incomplete[name] = {
                "missing": sorted(complete - supplied),
                "extra": sorted(supplied - complete),
            }
    if not requested or unknown or incomplete:
        raise ProofRefusal(
            "model_contract_binding_set_mismatch",
            f"unknown={unknown}, incomplete={incomplete}, selected={sorted(selected_names)}",
        )

    documents: dict[str, bytes] = {}
    topologies: dict[str, bytes] = {}
    resources: dict[str, dict[str, Any]] = {}
    functions: list[dict[str, Any]] = []
    for collection in ("entrypoints", "jobs"):
        entries = descriptor_doc[collection]
        if not isinstance(entries, list):
            raise ProofRefusal(
                "model_contract_package_descriptor_invalid", f"{collection} is not an array"
            )
        for described in entries:
            if not isinstance(described, dict) or not isinstance(described.get("name"), str):
                raise ProofRefusal(
                    "model_contract_package_descriptor_invalid", f"{collection} row is invalid"
                )
            name = described["name"]
            surface = surfaces.get(name)
            if surface is None:
                raise ProofRefusal(
                    "model_contract_package_descriptor_invalid",
                    f"{collection} surface {name!r} is absent",
                )
            if name not in selected_names:
                continue
            binding_rows: list[dict[str, Any]] = []
            for binding in surface.model_bindings:
                source = requested[binding.path]
                config_bytes = _read_exact(source.config, f"config for {binding.path}")
                try:
                    config_doc = canonical.parse_canonical(config_bytes)
                except canonical.CanonicalError as err:
                    raise ProofRefusal(
                        "model_contract_config_invalid", f"{binding.path}: {err}"
                    ) from err
                if not isinstance(config_doc, dict):
                    raise ProofRefusal(
                        "model_contract_config_invalid", f"{binding.path}: config is not an object"
                    )
                config = Config._verified(config_doc, "config", source.config.digest)
                contracts: list[dict[str, Any]] = []
                for variant in source.hardware_variants:
                    raw = _derive_twice(
                        binding,
                        source,
                        config,
                        release=request.release_identity,
                        application=found.application,
                        variant=variant,
                    )
                    digest = _sha(raw)
                    previous = documents.setdefault(digest, raw)
                    if previous != raw:
                        raise ProofRefusal("model_contract_digest_collision", digest)
                    if digest not in resources:
                        resources[digest] = _resource_projection(raw)
                    topology = _code_topology(raw)
                    topology_digest = _sha(topology)
                    previous_topology = topologies.setdefault(topology_digest, topology)
                    if previous_topology != topology:
                        raise ProofRefusal("model_contract_digest_collision", topology_digest)
                    contracts.append(
                        {
                            "code_topology": {
                                "digest": topology_digest,
                                "length": len(topology),
                            },
                            "hardware_variant": variant,
                            "model_construction_contract": {
                                "digest": digest,
                                "length": len(raw),
                            },
                        }
                    )
                binding_rows.append(
                    {
                        "class": binding.class_key,
                        "config": source.config.render(),
                        "param": binding.param,
                        "path": binding.path,
                        "snapshot": source.snapshot,
                        "variants": contracts,
                    }
                )
            functions.append(
                {
                    "bindings": binding_rows,
                    "kind": surface.kind,
                    "name": surface.name,
                }
            )
    return {
        "application": found.application,
        "contracts": [
            {
                "bytes": base64.b64encode(raw).decode(),
                "digest": digest,
                "length": len(raw),
            }
            for digest, raw in sorted(documents.items())
        ],
        "code_topologies": [
            {
                "bytes": base64.b64encode(raw).decode(),
                "digest": digest,
                "length": len(raw),
            }
            for digest, raw in sorted(topologies.items())
        ],
        "functions": functions,
        "resources": resources,
    }


def _derive_twice(
    binding: ModelBinding,
    source: BindingRequest,
    config: Config,
    *,
    release: str,
    application: str,
    variant: str,
) -> bytes:
    def run() -> bytes:
        result = derive(
            binding.model_class(),
            Artifact(source.snapshot, {}, config, custody="canonical"),
            release=release,
            application=application,
            hardware_variant=variant,
            declared_variants=source.hardware_variants,
            component_use=binding.components,
        )
        return canonical.write(result.contract.render())

    first = run()
    second = run()
    if first != second:
        raise ProofRefusal(
            "model_contract_nondeterministic",
            f"{binding.path} for {variant} produced different canonical bytes on rebuild",
        )
    parsed = contract.read_contract(first)
    if parsed.candidate.config_digest != source.config.digest:
        raise ProofRefusal(
            "model_contract_config_digest_mismatch",
            f"{binding.path} for {variant} did not retain the exact config document identity",
        )
    return first


def _resource_projection(raw: bytes) -> dict[str, Any]:
    """Per-contract resident-byte and encoding-pin facts (was the hub's third container)."""

    import tensorfs

    parsed = contract.read_contract(raw)
    widths = contract.tensorfs_destination_widths(tensorfs)
    components: dict[str, int] = {}
    tensors: list[contract.LogicalTensor | contract.DerivedTensor] = [
        *parsed.logical_tensors,
        *parsed.derived,
    ]
    for tensor in tensors:
        count = widths[tensor.dtype or contract.DEFAULT_DTYPE]
        for dimension in tensor.shape:
            count *= dimension
        components[tensor.component] = components.get(tensor.component, 0) + count
    return {"components": components, "encoding_pins": sorted(launch_providers())}


def _code_topology(raw: bytes) -> bytes:
    """Project one Runtime-owned MCC through TensorFS's existing document writer."""

    parsed = contract.read_contract(raw)
    storage_dtype = {torch: stored for stored, torch in TORCH_DTYPES.items()}
    try:
        import tensorfs

        rows = [
            tensorfs.CodeTensor(
                component=tensor.component,
                key=component_local_key(tensor.component, tensor.key),
                dtype=storage_dtype[tensor.dtype or contract.DEFAULT_DTYPE],
                shape=tensor.shape,
                accepts="any",
            )
            for tensor in sorted(
                parsed.logical_tensors, key=lambda tensor: (tensor.component, tensor.key)
            )
        ]
        return bytes(tensorfs.code_topology(_sha(raw), rows))
    except ProofRefusal:
        raise
    except Exception as err:
        raise ProofRefusal(
            "model_contract_code_topology_refused",
            f"TensorFS CodeTopology projection refused: {type(err).__name__}: {err}",
        ) from err


def component_local_key(component: str, key: str) -> str:
    """Project Runtime's full destination key into one TensorFS component."""

    prefix = f"{component}."
    if not key.startswith(prefix) or len(key) == len(prefix):
        raise ProofRefusal(
            "model_contract_component_key_invalid",
            f"logical tensor {key!r} is not under component {component!r}",
        )
    return key[len(prefix) :]


def _read_exact(ref: LocalDocumentRef, name: str) -> bytes:
    raw = _document_bytes(ref.path, name)
    if len(raw) != ref.length or _sha(raw) != ref.digest:
        raise ProofRefusal(
            "model_contract_input_identity_mismatch",
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
            "model_contract_input_identity_mismatch",
            f"package_descriptor at {ref.path} does not mean {ref.digest}/{ref.length}",
        )
    return document


def _emit_child(value: Mapping[str, Any]) -> None:
    raw = canonical.write(dict(value))
    print(RESULT_PREFIX + base64.b64encode(raw).decode())


def _decode_child(
    response: Mapping[str, Any],
    request: ProofRequest,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    if set(response) != {"application", "code_topologies", "contracts", "functions", "resources"}:
        raise ProofRefusal("model_contract_child_invalid", "child result has an open field set")
    if not isinstance(response["application"], str):
        raise ProofRefusal("model_contract_child_invalid", "child result application")
    rows = response["contracts"]
    if not isinstance(rows, list):
        raise ProofRefusal("model_contract_child_invalid", "contracts is not an array")
    documents: dict[str, bytes] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"bytes", "digest", "length"}:
            raise ProofRefusal("model_contract_child_invalid", "contract row field set")
        digest = row["digest"]
        length = row["length"]
        encoded = row["bytes"]
        if (
            not isinstance(digest, str)
            or canonical.DIGEST_RE.fullmatch(digest) is None
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            or not isinstance(encoded, str)
        ):
            raise ProofRefusal("model_contract_child_invalid", "contract row values")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as err:
            raise ProofRefusal("model_contract_child_invalid", "contract bytes base64") from err
        if len(raw) != length or _sha(raw) != digest:
            raise ProofRefusal("model_contract_child_invalid", "contract stored-byte identity")
        parsed = contract.read_contract(raw)
        if parsed.candidate.release != request.release_identity:
            raise ProofRefusal("model_contract_child_invalid", "contract release identity")
        documents[digest] = raw
    topology_rows = response["code_topologies"]
    if not isinstance(topology_rows, list):
        raise ProofRefusal("model_contract_child_invalid", "code_topologies is not an array")
    topologies: dict[str, bytes] = {}
    for row in topology_rows:
        if not isinstance(row, dict) or set(row) != {"bytes", "digest", "length"}:
            raise ProofRefusal("model_contract_child_invalid", "code topology row field set")
        digest = row["digest"]
        length = row["length"]
        encoded = row["bytes"]
        if (
            not isinstance(digest, str)
            or canonical.DIGEST_RE.fullmatch(digest) is None
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            or not isinstance(encoded, str)
        ):
            raise ProofRefusal("model_contract_child_invalid", "code topology row values")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as err:
            raise ProofRefusal(
                "model_contract_child_invalid", "code topology bytes base64"
            ) from err
        if len(raw) != length or _sha(raw) != digest:
            raise ProofRefusal("model_contract_child_invalid", "code topology stored-byte identity")
        topologies[digest] = raw
    functions = response["functions"]
    if not isinstance(functions, list):
        raise ProofRefusal("model_contract_child_invalid", "functions is not an array")
    resources = response["resources"]
    if not isinstance(resources, dict) or set(resources) != set(documents):
        raise ProofRefusal(
            "model_contract_child_invalid", "resources are not exactly one row per contract"
        )
    for projected in resources.values():
        if (
            not isinstance(projected, dict)
            or set(projected) != {"components", "encoding_pins"}
            or not isinstance(projected["components"], dict)
            or not projected["components"]
            or any(
                not isinstance(name, str)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
                for name, count in projected["components"].items()
            )
            or not isinstance(projected["encoding_pins"], list)
            or not projected["encoding_pins"]
            or any(not isinstance(pin, str) or not pin for pin in projected["encoding_pins"])
            or projected["encoding_pins"] != sorted(set(projected["encoding_pins"]))
        ):
            raise ProofRefusal("model_contract_child_invalid", "resource projection shape")
    evidence = {
        "format": INDEX_FORMAT,
        "application": response["application"],
        "package_descriptor": DocumentRef(
            request.package_descriptor.digest, request.package_descriptor.length
        ).render(),
        "package": dict(request.package),
        "functions": functions,
        "resources": resources,
    }
    return evidence, documents, topologies


def _publish(
    output: Path,
    index: bytes,
    documents: Mapping[str, bytes],
    topologies: Mapping[str, bytes],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not stat.S_ISDIR(output.parent.lstat().st_mode):
        raise ProofRefusal("model_contract_output_invalid", "output parent is not a directory")
    temporary = output.parent / f".model-contract-proof-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o700)
        contracts_dir = temporary / CONTRACTS_DIR
        contracts_dir.mkdir(mode=0o700)
        topologies_dir = temporary / TOPOLOGIES_DIR
        topologies_dir.mkdir(mode=0o700)
        for digest, raw in sorted(documents.items()):
            _write_durable(contracts_dir / f"{digest.removeprefix('sha256:')}.json", raw)
        for digest, raw in sorted(topologies.items()):
            _write_durable(topologies_dir / f"{digest.removeprefix('sha256:')}.json", raw)
        _write_durable(temporary / INDEX_NAME, index)
        contracts_dir.chmod(0o500)
        topologies_dir.chmod(0o500)
        temporary.chmod(0o500)
        _fsync_dir(contracts_dir)
        _fsync_dir(topologies_dir)
        _fsync_dir(temporary)
        try:
            os.rename(temporary, output)
        except OSError as err:
            if err.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise CliError(
                    "model_contract_proof_output_conflict",
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
