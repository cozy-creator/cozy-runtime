"""Canonical release-document adapter for the offline environment materializer.

Tensorhub owns the document schemas and delivery grants.  This module owns the one Runtime
join from their already-fetched exact bytes to :class:`environment.EnvironmentInput`.  Local
paths are deliberately function arguments: they never enter a canonical release document.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import ssl
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, environment
from cozy_runtime.internal.config import AdapterConfig

ENDPOINT_BUNDLE_FORMAT = "tensorhub.endpoint_bundle/1"
ENVIRONMENT_SPEC_FORMAT = "cozy.worker.v1.EndpointEnvironmentSpec/1"
RESOLVED_WHEEL_SET_FORMAT = "ResolvedWheelSet/1"
WHEELHOUSE_MANIFEST_FORMAT = "WheelhouseManifest/1"
PROVISION_SPEC_FORMAT = "tensorhub.rental_provision_spec/1"
PROVISION_BUNDLE_FORMAT = "tensorhub.rental_provision_bundle/1"
CONTROL_RUNTIME_WHEEL = Path("/opt/cozy/control-runtime/cozy-runtime.whl")
PROVISION_ROOT = Path("/var/lib/cozy/provision")
ENVIRONMENT_ROOT = Path("/var/lib/cozy/environments")
PLANS_ROOT = Path("/run/cozy/binding-plans")
REQUIRED_DOCUMENT_ROLES = (
    "artifact_object_set",
    "endpoint_build_record",
    "endpoint_bundle",
    "endpoint_environment_spec",
    "expected_installed_environment_receipt",
    "initial_placement_set",
    "resolved_wheel_set",
    "wheelhouse_manifest",
)
MAX_ARTIFACT_BYTES = 8 << 30
CHUNK_BYTES = 1 << 20


class LaunchRefusal(Exception):
    """A fail-closed release/grant join verdict with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class PreparedLaunch:
    spec: dict[str, Any]
    bundle: dict[str, Any]
    documents: Mapping[str, bytes]
    artifacts: Mapping[str, Path]
    installed: environment.InstalledEnvironment
    placement_set_digest: str


def prepare(
    config: AdapterConfig,
    *,
    uv: Path,
    provision_root: Path = PROVISION_ROOT,
    environment_root: Path = ENVIRONMENT_ROOT,
    plans_root: Path = PLANS_ROOT,
    control_runtime_wheel: Path = CONTROL_RUNTIME_WHEEL,
) -> PreparedLaunch:
    """Acquire one exact provision closure, materialize it offline, and stage its plans."""

    if time.time() >= config.readiness_deadline_unix:
        raise LaunchRefusal("readiness_deadline_elapsed", str(config.readiness_deadline_unix))
    spec_bytes = _local_exact(
        config.provision_spec_path, config.provision_spec_digest, canonical.DOC_MAX_BYTES
    )
    bundle_bytes = _local_exact(
        config.provision_bundle_path, config.provision_bundle_digest, canonical.DOC_MAX_BYTES
    )
    spec, bundle = _provision_pair(spec_bytes, bundle_bytes, config)
    control_ref = _ref(spec["execution_substrate"]["control_runtime"], "control_runtime")
    _verify_local(control_runtime_wheel, control_ref["digest"], control_ref["length"])
    verify_control_install(control_runtime_wheel)

    root = provision_root / config.provision_bundle_digest.removeprefix("sha256:")
    document_root = root / "documents"
    artifact_root = root / "artifacts"
    _directory(document_root)
    _directory(artifact_root)
    document_paths: dict[str, Path] = {}
    for row in bundle["documents"]:
        document_paths[row["role"]] = fetch_exact(
            row, document_root, deadline=config.readiness_deadline_unix
        )
    artifact_paths = {
        row["digest"]: fetch_exact(row, artifact_root, deadline=config.readiness_deadline_unix)
        for row in bundle["artifact_grants"]
    }
    document_bytes = {role: path.read_bytes() for role, path in document_paths.items()}
    release = _join_release(spec, bundle, document_bytes, artifact_paths)
    source = environment_input(
        environment_spec_bytes=document_bytes["endpoint_environment_spec"],
        endpoint_bundle_bytes=document_bytes["endpoint_bundle"],
        resolved_wheel_set_bytes=document_bytes["resolved_wheel_set"],
        wheelhouse_manifest_bytes=document_bytes["wheelhouse_manifest"],
        wheel_paths=release["wheel_paths"],
    )
    installed = materialize_exact(
        source,
        environment_root,
        uv=uv,
        expected_receipt_bytes=document_bytes["expected_installed_environment_receipt"],
    )
    _stage_plans(release["plan_subjects"], artifact_paths, plans_root)
    return PreparedLaunch(
        spec=spec,
        bundle=bundle,
        documents=document_bytes,
        artifacts=artifact_paths,
        installed=installed,
        placement_set_digest=_digest(document_bytes["initial_placement_set"]),
    )


def verify_control_install(wheel: Path) -> None:
    """Require this imported control Runtime to be the exact package in ``wheel``.

    A Python console script's executable inode is CPython, not its distribution. The
    substrate therefore binds the wheel bytes, and this comparison closes the other half:
    every shipped control-Runtime package file imported by this interpreter must be the
    byte-identical member of that wheel. Editable/source/PYTHONPATH execution cannot pass.
    """

    spec = importlib.util.find_spec("cozy_runtime")
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(locations) != 1:
        raise LaunchRefusal("control_runtime_install_invalid", "package root is ambiguous")
    package = Path(locations[0])
    try:
        package = package.resolve(strict=True)
        distribution = importlib.metadata.distribution("cozy-runtime")
        distribution_package = Path(str(distribution.locate_file("cozy_runtime"))).resolve(
            strict=True
        )
        direct_url = distribution.read_text("direct_url.json")
        if package != distribution_package or (
            direct_url is not None and json.loads(direct_url).get("dir_info", {}).get("editable")
        ):
            raise LaunchRefusal(
                "control_runtime_install_invalid", "adapter package is not the installed wheel"
            )
        with zipfile.ZipFile(wheel) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.startswith("cozy_runtime/") and not name.endswith("/")
            )
            if not members:
                raise LaunchRefusal(
                    "control_runtime_install_invalid", "wheel has no cozy_runtime package"
                )
            for name in members:
                relative = Path(name).relative_to("cozy_runtime")
                installed = package / relative
                info = installed.lstat()
                if not stat.S_ISREG(info.st_mode) or installed.read_bytes() != archive.read(name):
                    raise LaunchRefusal(
                        "control_runtime_install_mismatch", f"package member {relative} differs"
                    )
    except LaunchRefusal:
        raise
    except (OSError, KeyError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise LaunchRefusal(
            "control_runtime_install_invalid", f"package comparison failed: {type(exc).__name__}"
        ) from exc

    module = Path(__file__).resolve(strict=True)
    if package not in module.parents:
        raise LaunchRefusal(
            "control_runtime_install_invalid", "adapter is not executing from the installed package"
        )


def fetch_exact(grant: Mapping[str, Any], root: Path, *, deadline: int) -> Path:
    """Fetch one bounded HTTPS grant without proxy, redirect, content transform, or alias."""

    digest = grant.get("digest")
    length = grant.get("length")
    url = grant.get("url")
    _require_digest(digest, "grant.digest")
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        or length > MAX_ARTIFACT_BYTES
    ):
        raise LaunchRefusal("grant_length_invalid", str(length))
    if not isinstance(url, str):
        raise LaunchRefusal("grant_url_invalid", "URL is not a string")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or len(url) > 4096
    ):
        raise LaunchRefusal("grant_url_invalid", "grant URL is not one bounded HTTPS authority")
    _directory(root)
    target = root / str(digest).removeprefix("sha256:")
    if target.exists():
        _verify_local(target, str(digest), length)
        return target
    temporary = root / f".fetch-{uuid.uuid4().hex}"
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
        _NoRedirect(),
    )
    request = urllib.request.Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "cozy-materialize-launch/1"},
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    measured = hashlib.sha256()
    total = 0
    try:
        timeout = max(float(deadline) - time.time(), 0.001)
        try:
            response = opener.open(request, timeout=timeout)
        except (OSError, urllib.error.URLError) as exc:
            raise LaunchRefusal(
                "grant_fetch_failed", f"HTTPS grant fetch failed: {type(exc).__name__}"
            ) from exc
        with response, os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            encoding = response.headers.get("Content-Encoding", "identity")
            if response.status != 200 or encoding not in {"", "identity"}:
                raise LaunchRefusal("grant_response_invalid", f"status={response.status}")
            stated = response.headers.get("Content-Length")
            if stated is not None and stated != str(length):
                raise LaunchRefusal("grant_length_mismatch", f"header {stated}, expected {length}")
            while True:
                if time.time() >= deadline:
                    raise LaunchRefusal("readiness_deadline_elapsed", str(deadline))
                chunk = response.read(min(CHUNK_BYTES, length + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > length:
                    raise LaunchRefusal("grant_length_mismatch", f"more than {length} bytes")
                measured.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        got = "sha256:" + measured.hexdigest()
        if total != length or got != digest:
            raise LaunchRefusal(
                "grant_identity_mismatch",
                f"measured digest={got} length={total}, expected digest={digest} length={length}",
            )
        os.replace(temporary, target)
        os.chmod(target, 0o400)
        _fsync(root)
        return target
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _provision_pair(
    spec_bytes: bytes, bundle_bytes: bytes, config: AdapterConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _document(spec_bytes, "provision_spec")
    bundle = _document(bundle_bytes, "provision_bundle")
    _closed(
        spec,
        {
            "acquisition_attempt_id",
            "acquisition_attempt_ordinal",
            "artifact_object_set",
            "capacity",
            "execution_substrate",
            "format",
            "initial_placement_set",
            "network",
            "renter_token_sha256",
            "rental_id",
        },
        "provision_spec",
    )
    if spec["format"] != PROVISION_SPEC_FORMAT:
        raise LaunchRefusal("provision_spec_format", str(spec["format"]))
    _closed(
        spec["capacity"],
        {
            "accelerator_count",
            "accelerator_model",
            "cache_mount_path",
            "cache_volume_id",
            "datacenter",
            "minimum_local_disk_bytes",
            "provider",
            "provider_offer_key",
            "provider_sku",
            "provider_volume_id",
        },
        "provision_capacity",
    )
    _closed(
        spec["execution_substrate"],
        {"control_runtime", "oci_image", "wheelhouse_manifest_digest"},
        "provision_substrate",
    )
    _closed(
        spec["network"],
        {
            "media_port",
            "media_protocol",
            "worker_port",
            "worker_protocol",
            "worker_wire_schema_digest",
        },
        "provision_network",
    )
    _ref(spec["artifact_object_set"], "artifact_object_set")
    _ref(spec["initial_placement_set"], "initial_placement_set")
    _ref(spec["execution_substrate"]["control_runtime"], "control_runtime")
    _require_digest(
        spec["execution_substrate"]["wheelhouse_manifest_digest"],
        "execution_substrate.wheelhouse_manifest_digest",
    )
    network = spec["network"]
    if (
        network["worker_protocol"] != "cozy.worker.v1"
        or network["media_protocol"] != "https"
        or network["worker_port"] != config.worker_internal_port
        or network["media_port"] != config.media_internal_port
    ):
        raise LaunchRefusal("provision_network_mismatch", "bootstrap and spec disagree")
    _require_digest(network["worker_wire_schema_digest"], "worker_wire_schema_digest")
    from cozy_runtime.protocol import SCHEMA_DIGEST

    if network["worker_wire_schema_digest"] != SCHEMA_DIGEST:
        raise LaunchRefusal(
            "worker_wire_schema_mismatch", "provisioned schema is not this control Runtime"
        )
    if spec["renter_token_sha256"] != list(config.renter_token_sha256):
        raise LaunchRefusal("renter_token_hash_mismatch", "bootstrap and spec disagree")
    if (
        not isinstance(spec["acquisition_attempt_ordinal"], int)
        or isinstance(spec["acquisition_attempt_ordinal"], bool)
        or spec["acquisition_attempt_ordinal"] <= 0
    ):
        raise LaunchRefusal("provision_attempt_invalid", "ordinal is not positive")
    for name in ("rental_id", "acquisition_attempt_id"):
        value = spec[name]
        if not isinstance(value, str) or not value or value.strip() != value or len(value) > 256:
            raise LaunchRefusal("provision_attempt_invalid", name)
    capacity = spec["capacity"]
    if (
        not isinstance(capacity["accelerator_count"], int)
        or isinstance(capacity["accelerator_count"], bool)
        or capacity["accelerator_count"] <= 0
        or not isinstance(capacity["minimum_local_disk_bytes"], int)
        or isinstance(capacity["minimum_local_disk_bytes"], bool)
        or capacity["minimum_local_disk_bytes"] <= 0
    ):
        raise LaunchRefusal("provision_capacity_invalid", "numeric capacity is not positive")
    for name in (
        "accelerator_model",
        "cache_mount_path",
        "cache_volume_id",
        "datacenter",
        "provider",
        "provider_offer_key",
        "provider_sku",
        "provider_volume_id",
    ):
        value = capacity[name]
        if not isinstance(value, str) or not value or value.strip() != value or len(value) > 256:
            raise LaunchRefusal("provision_capacity_invalid", name)
    image = spec["execution_substrate"]["oci_image"]
    if not isinstance(image, str) or image.count("@") != 1 or not image.split("@", 1)[0]:
        raise LaunchRefusal("provision_substrate_invalid", "OCI image is not digest pinned")
    _require_digest(image.split("@", 1)[1], "execution_substrate.oci_image")

    _closed(
        bundle,
        {"artifact_grants", "control_runtime", "documents", "format", "provision_spec"},
        "provision_bundle",
    )
    if bundle["format"] != PROVISION_BUNDLE_FORMAT:
        raise LaunchRefusal("provision_bundle_format", str(bundle["format"]))
    if _ref(bundle["provision_spec"], "provision_spec") != {
        "digest": config.provision_spec_digest,
        "length": len(spec_bytes),
    }:
        raise LaunchRefusal("provision_spec_ref_mismatch", "bundle and bootstrap disagree")
    if (
        _ref(bundle["control_runtime"], "control_runtime")
        != spec["execution_substrate"]["control_runtime"]
    ):
        raise LaunchRefusal("control_runtime_ref_mismatch", "bundle and spec disagree")
    documents = bundle["documents"]
    if not isinstance(documents, list) or len(documents) != len(REQUIRED_DOCUMENT_ROLES):
        raise LaunchRefusal("provision_document_roles_invalid", "wrong role count")
    roles: list[str] = []
    for row in documents:
        _closed(row, {"digest", "length", "role", "url"}, "document_grant")
        document_ref = _ref(row, "document_grant", allow_extra=True)
        if document_ref["length"] > canonical.DOC_MAX_BYTES:
            raise LaunchRefusal("document_grant_invalid", "document exceeds canonical cap")
        if not isinstance(row["role"], str):
            raise LaunchRefusal("provision_document_roles_invalid", "role is not a string")
        roles.append(row["role"])
    if tuple(roles) != REQUIRED_DOCUMENT_ROLES:
        raise LaunchRefusal("provision_document_roles_invalid", str(roles))
    grants = bundle["artifact_grants"]
    if not isinstance(grants, list):
        raise LaunchRefusal("artifact_grants_invalid", "artifact_grants is not an array")
    prior = ""
    for row in grants:
        _closed(row, {"digest", "length", "url"}, "artifact_grant")
        ref = _ref(row, "artifact_grant", allow_extra=True)
        if ref["digest"] <= prior:
            raise LaunchRefusal("artifact_grants_invalid", "digests are not sorted unique")
        prior = ref["digest"]
    return spec, bundle


def _join_release(
    spec: dict[str, Any],
    bundle: dict[str, Any],
    docs: Mapping[str, bytes],
    artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    grants = {row["role"]: row for row in bundle["documents"]}
    if set(docs) != set(REQUIRED_DOCUMENT_ROLES):
        raise LaunchRefusal("provision_document_roles_invalid", "fetched role set changed")
    for role, raw in docs.items():
        _ref_matches(grants[role], raw, role, allow_extra=True)
    build = _document(docs["endpoint_build_record"], "endpoint_build_record")
    _closed(
        build,
        {
            "control_runtime",
            "endpoint_bundle",
            "endpoint_environment_spec",
            "expected_installed_environment_receipt",
            "format",
            "resolved_wheel_set",
            "wheelhouse_manifest",
        },
        "endpoint_build_record",
    )
    if build["format"] != "tensorhub.endpoint_build_record/1":
        raise LaunchRefusal("endpoint_build_record_format", str(build["format"]))
    joins = {
        "endpoint_bundle": build["endpoint_bundle"],
        "endpoint_environment_spec": build["endpoint_environment_spec"],
        "expected_installed_environment_receipt": build["expected_installed_environment_receipt"],
        "resolved_wheel_set": build["resolved_wheel_set"],
        "wheelhouse_manifest": build["wheelhouse_manifest"],
    }
    for role, value in joins.items():
        granted = {"digest": grants[role]["digest"], "length": grants[role]["length"]}
        if _ref(value, role) != granted:
            raise LaunchRefusal("endpoint_build_record_mismatch", role)
    if (
        _ref(build["control_runtime"], "control_runtime")
        != spec["execution_substrate"]["control_runtime"]
    ):
        raise LaunchRefusal("control_runtime_ref_mismatch", "build record and spec disagree")
    if (
        _ref(build["wheelhouse_manifest"], "wheelhouse_manifest")["digest"]
        != spec["execution_substrate"]["wheelhouse_manifest_digest"]
    ):
        raise LaunchRefusal(
            "wheelhouse_manifest_ref_mismatch", "build record and substrate disagree"
        )
    if _ref(spec["artifact_object_set"], "artifact_object_set") != {
        "digest": grants["artifact_object_set"]["digest"],
        "length": grants["artifact_object_set"]["length"],
    }:
        raise LaunchRefusal("artifact_object_set_ref_mismatch", "spec and bundle disagree")
    if _ref(spec["initial_placement_set"], "initial_placement_set") != {
        "digest": grants["initial_placement_set"]["digest"],
        "length": grants["initial_placement_set"]["length"],
    }:
        raise LaunchRefusal("initial_placement_set_ref_mismatch", "spec and bundle disagree")

    endpoint = _document(docs["endpoint_bundle"], "endpoint_bundle")
    resolved = _document(docs["resolved_wheel_set"], "resolved_wheel_set")
    wheelhouse = _document(docs["wheelhouse_manifest"], "wheelhouse_manifest")
    wheel_rows = [endpoint.get("project_wheel")]
    resolved_wheels = resolved.get("wheels", [])
    wheel_rows.extend(resolved_wheels if isinstance(resolved_wheels, list) else [])
    wheel_rows.extend(
        wheelhouse.get("wheels", []) if isinstance(wheelhouse.get("wheels"), list) else []
    )
    wheel_paths: dict[str, Path] = {}
    for row in wheel_rows:
        if not isinstance(row, dict):
            raise LaunchRefusal("wheel_inventory_invalid", "wheel row is not an object")
        ref = _ref(row, "wheel", allow_extra=True)
        path = artifacts.get(ref["digest"])
        if path is None or path.stat().st_size != ref["length"]:
            raise LaunchRefusal("wheel_grant_missing", ref["digest"])
        wheel_paths[ref["digest"]] = path
    for role in ("descriptor", "evaluated_config"):
        ref = _ref(endpoint.get(role), role)
        path = artifacts.get(ref["digest"])
        if path is None or path.stat().st_size != ref["length"]:
            raise LaunchRefusal(f"{role}_grant_missing", ref["digest"])

    placement = _document(docs["initial_placement_set"], "initial_placement_set")
    _closed(placement, {"format", "placements"}, "initial_placement_set")
    if placement["format"] != "cozy.worker.v1.PlacementSet/1":
        raise LaunchRefusal("initial_placement_set_format", str(placement["format"]))
    placements = placement["placements"]
    if not isinstance(placements, list) or len(placements) != 1:
        raise LaunchRefusal("initial_placement_set_invalid", "launch requires one placement")
    row = placements[0]
    _closed(row, {"placement_id", "spec"}, "initial_placement")
    placement_spec = row["spec"]
    _closed(
        placement_spec,
        {
            "binding_plans",
            "descriptor_digest",
            "endpoint_release_id",
            "environment_spec_digest",
            "installed_environment_receipt_digest",
        },
        "initial_placement_spec",
    )
    expected = {
        "descriptor_digest": _ref(endpoint["descriptor"], "descriptor")["digest"],
        "endpoint_release_id": endpoint["endpoint_release_id"],
        "environment_spec_digest": grants["endpoint_environment_spec"]["digest"],
        "installed_environment_receipt_digest": grants["expected_installed_environment_receipt"][
            "digest"
        ],
    }
    for name, value in expected.items():
        if placement_spec[name] != value:
            raise LaunchRefusal("initial_placement_set_mismatch", name)
    subjects = placement_spec["binding_plans"]
    if not isinstance(subjects, list):
        raise LaunchRefusal("binding_plan_subjects_invalid", "subjects are not an array")
    plan_subjects: dict[str, tuple[str, int]] = {}
    prior = ""
    for subject in subjects:
        _closed(subject, {"digest", "kind", "length", "subject_id"}, "binding_plan_subject")
        ref = _ref(subject, "binding_plan_subject", allow_extra=True)
        subject_id = subject["subject_id"]
        if (
            subject["kind"] != "plan"
            or not isinstance(subject_id, str)
            or not subject_id
            or subject_id <= prior
        ):
            raise LaunchRefusal("binding_plan_subjects_invalid", str(subject_id))
        path = artifacts.get(ref["digest"])
        if path is None or path.stat().st_size != ref["length"]:
            raise LaunchRefusal("binding_plan_grant_missing", ref["digest"])
        prior = subject_id
        plan_subjects[subject_id] = (ref["digest"], ref["length"])
    return {"wheel_paths": wheel_paths, "plan_subjects": plan_subjects}


def _stage_plans(
    subjects: Mapping[str, tuple[str, int]], artifacts: Mapping[str, Path], root: Path
) -> None:
    _directory(root)
    for digest, length in subjects.values():
        raw = _local_exact(artifacts[digest], digest, length)
        target = root / f"{digest.removeprefix('sha256:')}.json"
        if target.exists():
            if _local_exact(target, digest, length) != raw:
                raise LaunchRefusal("binding_plan_stage_conflict", str(target))
            continue
        temporary = root / f".plan-{uuid.uuid4().hex}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o400)
            _fsync(root)
        finally:
            temporary.unlink(missing_ok=True)


def _ref(value: object, name: str, *, allow_extra: bool = False) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not {"digest", "length"}.issubset(value)
        or (not allow_extra and set(value) != {"digest", "length"})
    ):
        raise LaunchRefusal(f"{name}_ref_invalid", "reference is not {digest,length}")
    digest = value.get("digest")
    length = value.get("length")
    _require_digest(digest, f"{name}.digest")
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        or length > MAX_ARTIFACT_BYTES
    ):
        raise LaunchRefusal(f"{name}_ref_invalid", "length is not positive and bounded")
    return {"digest": digest, "length": length}


def _local_exact(path: Path, digest: str, length: int) -> bytes:
    _verify_local(path, digest, length)
    return path.read_bytes()


def _verify_local(path: Path, digest: str, length: int) -> None:
    _require_digest(digest, "local.digest")
    try:
        info = path.lstat()
    except OSError as exc:
        raise LaunchRefusal("local_artifact_unreadable", f"{path}: {exc}") from exc
    if (
        not path.is_absolute()
        or not stat.S_ISREG(info.st_mode)
        or info.st_size != length
        or length > MAX_ARTIFACT_BYTES
    ):
        raise LaunchRefusal("local_artifact_invalid", str(path))
    measured = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            measured.update(chunk)
    got = "sha256:" + measured.hexdigest()
    if got != digest:
        raise LaunchRefusal("local_artifact_digest_mismatch", f"measured {got}, expected {digest}")


def _directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not path.is_absolute() or not stat.S_ISDIR(info.st_mode):
        raise LaunchRefusal("provision_store_invalid", str(path))
    path.chmod(0o700)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def environment_input(
    *,
    environment_spec_bytes: bytes,
    endpoint_bundle_bytes: bytes,
    resolved_wheel_set_bytes: bytes,
    wheelhouse_manifest_bytes: bytes,
    wheel_paths: Mapping[str, Path],
) -> environment.EnvironmentInput:
    """Join exact th-046 documents and fetched wheel paths without resolving anything."""

    spec = _document(environment_spec_bytes, "environment_spec")
    bundle = _document(endpoint_bundle_bytes, "endpoint_bundle")
    resolved = _document(resolved_wheel_set_bytes, "resolved_wheel_set")
    wheelhouse = _document(wheelhouse_manifest_bytes, "wheelhouse_manifest")
    _closed(
        spec,
        {
            "endpoint_bundle_digest",
            "format",
            "platform_target",
            "project_wheel_digest",
            "wheelhouse_manifest_digest",
        },
        "environment_spec",
    )
    if spec["format"] != ENVIRONMENT_SPEC_FORMAT:
        raise LaunchRefusal("environment_spec_format", str(spec["format"]))
    _closed(
        bundle,
        {
            "descriptor",
            "endpoint_release_id",
            "evaluated_config",
            "format",
            "project_wheel",
            "resolved_wheel_set",
            "wheelhouse_manifest",
        },
        "endpoint_bundle",
    )
    if bundle["format"] != ENDPOINT_BUNDLE_FORMAT:
        raise LaunchRefusal("endpoint_bundle_format", str(bundle["format"]))
    _closed(
        resolved,
        {"format", "lock_digest", "platform_target", "wheelhouse_manifest_digest", "wheels"},
        "resolved_wheel_set",
    )
    if resolved["format"] != RESOLVED_WHEEL_SET_FORMAT:
        raise LaunchRefusal("resolved_wheel_set_format", str(resolved["format"]))
    _require_digest(resolved["lock_digest"], "resolved_wheel_set.lock_digest")
    _closed(wheelhouse, {"format", "platform_target", "wheels"}, "wheelhouse_manifest")
    if wheelhouse["format"] != WHEELHOUSE_MANIFEST_FORMAT:
        raise LaunchRefusal("wheelhouse_manifest_format", str(wheelhouse["format"]))

    _require_digest(spec["endpoint_bundle_digest"], "environment_spec.endpoint_bundle_digest")
    _require_digest(spec["project_wheel_digest"], "environment_spec.project_wheel_digest")
    _require_digest(
        spec["wheelhouse_manifest_digest"], "environment_spec.wheelhouse_manifest_digest"
    )
    _matches(endpoint_bundle_bytes, str(spec["endpoint_bundle_digest"]), "endpoint_bundle")
    _matches(
        wheelhouse_manifest_bytes,
        str(spec["wheelhouse_manifest_digest"]),
        "wheelhouse_manifest",
    )
    _ref_matches(bundle["resolved_wheel_set"], resolved_wheel_set_bytes, "resolved_wheel_set")
    _ref_matches(bundle["wheelhouse_manifest"], wheelhouse_manifest_bytes, "wheelhouse_manifest")
    if bundle["wheelhouse_manifest"] != {
        "digest": spec["wheelhouse_manifest_digest"],
        "length": len(wheelhouse_manifest_bytes),
    }:
        raise LaunchRefusal(
            "wheelhouse_manifest_ref_mismatch", "bundle and environment spec disagree"
        )
    if resolved["wheelhouse_manifest_digest"] != spec["wheelhouse_manifest_digest"]:
        raise LaunchRefusal(
            "wheelhouse_manifest_ref_mismatch", "resolved set and environment spec disagree"
        )
    if not isinstance(spec["platform_target"], dict) or not (
        spec["platform_target"] == resolved["platform_target"] == wheelhouse["platform_target"]
    ):
        raise LaunchRefusal("platform_target_mismatch", "release documents disagree")

    project = _wheel(bundle["project_wheel"], "project")
    if project.digest != spec["project_wheel_digest"]:
        raise LaunchRefusal("project_wheel_mismatch", project.digest)
    wheels = [project]
    wheels.extend(_wheels(resolved["wheels"], "bundle"))
    wheels.extend(_wheels(wheelhouse["wheels"], "wheelhouse"))
    digests = [wheel.digest for wheel in wheels]
    distributions = [wheel.distribution for wheel in wheels]
    if len(digests) != len(set(digests)):
        raise LaunchRefusal("wheel_digest_duplicate", "one digest has multiple owners")
    if len(distributions) != len(set(distributions)):
        raise LaunchRefusal("wheel_distribution_duplicate", "one distribution has multiple owners")
    if set(wheel_paths) != set(digests):
        missing = sorted(set(digests) - set(wheel_paths))
        extra = sorted(set(wheel_paths) - set(digests))
        raise LaunchRefusal("wheel_grant_set_mismatch", f"missing={missing[:1]} extra={extra[:1]}")
    grants = tuple(
        environment.BlobGrant(wheel.digest, wheel.length, wheel_paths[wheel.digest])
        for wheel in wheels
    )
    return environment.EnvironmentInput(
        environment_spec_bytes=environment_spec_bytes,
        environment_spec_digest=_digest(environment_spec_bytes),
        endpoint_bundle_bytes=endpoint_bundle_bytes,
        wheelhouse_manifest_bytes=wheelhouse_manifest_bytes,
        wheels=tuple(wheels),
        grants=grants,
    )


def materialize_exact(
    source: environment.EnvironmentInput,
    root: Path,
    *,
    uv: Path,
    expected_receipt_bytes: bytes,
) -> environment.InstalledEnvironment:
    """Materialize and require byte identity with th-046's proof-seat receipt."""

    try:
        expected = _document(expected_receipt_bytes, "expected_environment_receipt")
    except LaunchRefusal:
        raise
    if expected.get("format") != environment.RECEIPT_FORMAT:
        raise LaunchRefusal("expected_environment_receipt_format", str(expected.get("format")))
    expected_digest = _digest(expected_receipt_bytes)
    try:
        installed = environment.materialize(
            source,
            root,
            uv=uv,
            expected_receipt_digest=expected_digest,
        )
    except environment.EnvironmentRefusal as exc:
        raise LaunchRefusal(exc.code, exc.detail) from exc
    if installed.receipt_bytes != expected_receipt_bytes:
        raise LaunchRefusal(
            "expected_environment_receipt_bytes_mismatch",
            f"measured {installed.receipt_digest}, exact expected bytes differ",
        )
    return installed


def _document(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        raise LaunchRefusal(f"{name}_invalid", str(exc)) from exc
    if not isinstance(value, dict):
        raise LaunchRefusal(f"{name}_invalid", "document is not an object")
    return value


def _closed(value: object, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        fields = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise LaunchRefusal(f"{name}_invalid", f"fields={fields} expected={sorted(keys)}")


def _ref_matches(value: object, raw: bytes, name: str, *, allow_extra: bool = False) -> None:
    if (
        not isinstance(value, dict)
        or not {"digest", "length"}.issubset(value)
        or (not allow_extra and set(value) != {"digest", "length"})
    ):
        raise LaunchRefusal(f"{name}_ref_invalid", "reference is not {digest,length}")
    digest = value.get("digest")
    length = value.get("length")
    _require_digest(digest, f"{name}.digest")
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise LaunchRefusal(f"{name}_ref_invalid", "length is not positive")
    if length != len(raw):
        raise LaunchRefusal(f"{name}_length_mismatch", f"measured {len(raw)}, expected {length}")
    _matches(raw, str(digest), name)


def _matches(raw: bytes, digest: str, name: str) -> None:
    if _digest(raw) != digest:
        raise LaunchRefusal(
            f"{name}_digest_mismatch", f"measured {_digest(raw)}, expected {digest}"
        )


def _wheels(value: object, owner: str) -> list[environment.WheelInventoryEntry]:
    if not isinstance(value, list):
        raise LaunchRefusal(f"{owner}_wheel_inventory_invalid", "wheels is not a list")
    wheels = [_wheel(row, owner) for row in value]
    names = [wheel.distribution for wheel in wheels]
    if names != sorted(names) or len(names) != len(set(names)):
        raise LaunchRefusal(
            f"{owner}_wheel_inventory_invalid", "distributions are not sorted unique"
        )
    return wheels


def _wheel(value: object, owner: str) -> environment.WheelInventoryEntry:
    keys = {"digest", "distribution", "filename", "length", "tags", "version"}
    if not isinstance(value, dict) or set(value) != keys:
        raise LaunchRefusal(f"{owner}_wheel_invalid", "wheel row has the wrong fields")
    digest = value["digest"]
    _require_digest(digest, f"{owner}_wheel.digest")
    strings = (value["distribution"], value["filename"], value["version"])
    if not all(isinstance(item, str) and item for item in strings):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "wheel strings are empty or invalid")
    length = value["length"]
    tags = value["tags"]
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise LaunchRefusal(f"{owner}_wheel_invalid", "length is not positive")
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(tag, str) and tag for tag in tags)
        or tags != sorted(set(tags))
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "tags are not non-empty sorted unique")
    return environment.WheelInventoryEntry(
        distribution=value["distribution"],
        version=value["version"],
        filename=value["filename"],
        digest=str(digest),
        length=length,
        tags=tuple(tags),
        owner=owner,
    )


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or canonical.DIGEST_RE.fullmatch(value) is None:
        raise LaunchRefusal(f"{name}_invalid", "not sha256:<64 lowercase hex>")


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
