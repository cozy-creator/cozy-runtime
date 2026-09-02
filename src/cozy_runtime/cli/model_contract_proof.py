"""Author-side writer of publish-time slot facts (cr-077, was cr-067's pair evidence).

Runs on the author's machine inside the project venv (`uv run --locked`), at publish.
Per slot CLASS it delegates package import and derive-mode execution to one bounded
child whose audit hooks allow no network, weight reads, writes, or subprocesses, and
publishes exactly TWO facts per class -- the shape-only code topology (a TensorFS
CodeTopology document, digest-addressed) and the acceptable encodings set -- as one
digest-addressed envelope: `result.json` index plus `code-topologies/` sibling
documents, by atomic rename. The construction SEED (the checkpoint snapshot/config the
factory runs against, resolved by the publisher from the slot's default binding) is an
input only: its identity never enters the published facts. The envelope binds the
release VERSION, never a release digest: the digest only exists once tensorhub
finalizes the upload this envelope rides in. The hub validates these documents
statically and never executes package code (th-106).
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
TOPOLOGIES_DIR = "code-topologies"
RESULT_PREFIX = "COZY_MODEL_CONTRACT_PROOF="
MEMORY_CAP_BYTES = 24 << 30
CPU_CAP_SECONDS = 600
WALL_CAP_SECONDS = 900
_VARIANT = re.compile(r"^(?:cpu|sm[0-9]{2,})$")

INDEX_FORMAT = "cozy.package.slot-facts/1"

REQUEST_FIELDS = {
    "package_descriptor",
    "package",
    "project",
    "slots",
}
REF_FIELDS = {"digest", "length"}
LOCAL_REF_FIELDS = {"digest", "length", "path"}
SLOT_FIELDS = {"class", "config", "hardware_variants", "snapshot"}


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
    """One slot class's construction seed: which exact checkpoint snapshot/config the
    class's factory runs against on the meta device. Seed identity is an input, never a
    published fact."""

    class_key: str
    snapshot: str
    config: LocalDocumentRef
    hardware_variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProofRequest:
    project: Path
    package: Mapping[str, Any]
    package_descriptor: LocalDocumentRef
    slots: tuple[SlotRequest, ...]

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
        index, topologies = _decode_child(response, request)
        index_bytes = canonical.write(index)
        _publish(output, index_bytes, topologies)
        print(
            json.dumps(
                {
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
    if not isinstance(value, dict) or set(value) != SLOT_FIELDS:
        raise _request_error(f"{name} does not have the closed slot field set")
    class_key = value["class"]
    if (
        not isinstance(class_key, str)
        or not class_key
        or len(class_key) > 512
        or class_key.strip() != class_key
    ):
        raise _request_error(f"{name}.class is not one bounded non-empty string")
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
    return SlotRequest(
        class_key=class_key,
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
        response = _derive_facts(request, found, derived_descriptor)
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


def _derive_facts(
    request: ProofRequest, found: Any, descriptor_doc: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the per-class slot facts: ONE shape-only code topology and ONE acceptable
    encodings set per requested slot class (cr-077). The MCC is derived internally --
    it is the machinery the topology projects from -- and is never published."""

    del descriptor_doc  # byte-compared against the source by the caller; nothing pair-wise
    bindings_by_class: dict[str, ModelBinding] = {}
    for surface in found.surfaces:
        if surface.hidden:
            continue
        for binding in surface.model_bindings:
            bindings_by_class.setdefault(binding.class_key, binding)
    unknown = sorted({row.class_key for row in request.slots} - set(bindings_by_class))
    if unknown:
        raise ProofRefusal(
            "slot_facts_class_unknown",
            f"requested classes {unknown} are not model slot classes of this package "
            f"(have {sorted(bindings_by_class)})",
        )

    topologies: dict[str, bytes] = {}
    slots: list[dict[str, Any]] = []
    encodings = sorted(launch_providers())
    for source in request.slots:
        binding = bindings_by_class[source.class_key]
        config_bytes = _read_exact(source.config, f"config for {source.class_key}")
        try:
            config_doc = canonical.parse_canonical(config_bytes)
        except canonical.CanonicalError as err:
            raise ProofRefusal(
                "model_contract_config_invalid", f"{source.class_key}: {err}"
            ) from err
        if not isinstance(config_doc, dict):
            raise ProofRefusal(
                "model_contract_config_invalid",
                f"{source.class_key}: config is not an object",
            )
        config = Config._verified(config_doc, "config", source.config.digest)
        class_topologies: dict[str, bytes] = {}
        for variant in source.hardware_variants:
            raw = _derive_twice(
                binding,
                source,
                config,
                release=request.release_identity,
                application=found.application,
                variant=variant,
            )
            topology = _code_topology(raw)
            class_topologies[_sha(topology)] = topology
        if len(class_topologies) != 1:
            raise ProofRefusal(
                "slot_facts_variant_divergent",
                f"{source.class_key}: hardware variants produced {len(class_topologies)} "
                "distinct shape topologies; one slot class carries exactly one topology "
                "fact -- a hardware-conditional topology is unrepresentable as slot facts",
            )
        ((topology_digest, topology),) = class_topologies.items()
        previous = topologies.setdefault(topology_digest, topology)
        if previous != topology:
            raise ProofRefusal("model_contract_digest_collision", topology_digest)
        slots.append(
            {
                "class": source.class_key,
                "code_topology": {
                    "digest": topology_digest,
                    "length": len(topology),
                },
                "encodings": encodings,
            }
        )
    return {
        "application": found.application,
        "code_topologies": [
            {
                "bytes": base64.b64encode(raw).decode(),
                "digest": digest,
                "length": len(raw),
            }
            for digest, raw in sorted(topologies.items())
        ],
        "slots": slots,
    }


def _derive_twice(
    binding: ModelBinding,
    source: SlotRequest,
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
            f"{binding.class_key} for {variant} produced different canonical bytes on rebuild",
        )
    parsed = contract.read_contract(first)
    if parsed.candidate.config_digest != source.config.digest:
        raise ProofRefusal(
            "model_contract_config_digest_mismatch",
            f"{binding.class_key} for {variant} did not retain the exact config document identity",
        )
    return first


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
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if set(response) != {"application", "code_topologies", "slots"}:
        raise ProofRefusal("model_contract_child_invalid", "child result has an open field set")
    if not isinstance(response["application"], str):
        raise ProofRefusal("model_contract_child_invalid", "child result application")
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
    slots = response["slots"]
    if not isinstance(slots, list) or not slots:
        raise ProofRefusal("model_contract_child_invalid", "slots is not a non-empty array")
    referenced: set[str] = set()
    classes: list[str] = []
    for row in slots:
        if not isinstance(row, dict) or set(row) != {"class", "code_topology", "encodings"}:
            raise ProofRefusal("model_contract_child_invalid", "slot row field set")
        class_key = row["class"]
        if not isinstance(class_key, str) or not class_key:
            raise ProofRefusal("model_contract_child_invalid", "slot row class")
        classes.append(class_key)
        topology_ref = row["code_topology"]
        if (
            not isinstance(topology_ref, dict)
            or set(topology_ref) != REF_FIELDS
            or not isinstance(topology_ref["digest"], str)
            or topology_ref["digest"] not in topologies
            or topology_ref["length"] != len(topologies[topology_ref["digest"]])
        ):
            raise ProofRefusal("model_contract_child_invalid", "slot row code topology ref")
        referenced.add(topology_ref["digest"])
        encodings = row["encodings"]
        if (
            not isinstance(encodings, list)
            or not encodings
            or any(
                not isinstance(pin, str) or canonical.DIGEST_RE.fullmatch(pin) is None
                for pin in encodings
            )
            or encodings != sorted(set(encodings))
        ):
            raise ProofRefusal("model_contract_child_invalid", "slot row encodings")
    if classes != sorted(set(classes)):
        raise ProofRefusal("model_contract_child_invalid", "slot rows are not sorted unique")
    if set(classes) != {row.class_key for row in request.slots}:
        raise ProofRefusal(
            "model_contract_child_invalid", "slot rows do not answer the requested classes"
        )
    if referenced != set(topologies):
        raise ProofRefusal("model_contract_child_invalid", "unreferenced code topology document")
    evidence = {
        "format": INDEX_FORMAT,
        "application": response["application"],
        "package_descriptor": DocumentRef(
            request.package_descriptor.digest, request.package_descriptor.length
        ).render(),
        "package": dict(request.package),
        "slots": slots,
    }
    return evidence, topologies


def _publish(
    output: Path,
    index: bytes,
    topologies: Mapping[str, bytes],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not stat.S_ISDIR(output.parent.lstat().st_mode):
        raise ProofRefusal("model_contract_output_invalid", "output parent is not a directory")
    temporary = output.parent / f".model-contract-proof-{uuid.uuid4().hex}"
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
