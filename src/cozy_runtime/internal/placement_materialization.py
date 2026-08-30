"""Exact artifact intake and package-overlay document join.

Tensorhub owns the document schemas and delivery grants.  This module owns the one Runtime
join from their already-fetched exact bytes to :class:`overlay.Input`. Local paths are
deliberately function arguments: they never enter a canonical release document.

cr-048 RETIRED THE BOOT CLOSURE. ``tensorhub.rental_provision_spec/1`` and
``tensorhub.rental_provision_bundle/1`` named one package's whole closure at boot — eight
document roles, one artifact grant per wheel and weight, the whole materialization root
keyed by the bundle digest — which is the pre-#459 one-pod-one-package shape that rev-2's
``DesiredPlacementSet`` + direct Tensorhub artifact redemption replaced. The pod now boots to
ready-but-EMPTY: GPU inventory measured, listeners proven, receipt
published, zero placements. What survives here is the DISCIPLINE, not the vehicle:
``fetch_exact`` (one bounded exact-digest HTTPS grant) and ``materialize_objects`` (exact CAS
admission) are the primitives the dynamic grant lane fetches through, and
``package_overlay_input`` is the small source/custom-wheel join a placement converges to.
Only the boot-only bundling retired.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
import urllib.error
import urllib.parse
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, host_paths, overlay, pull
from cozy_runtime.internal.config import WorkerHostConfig
from cozy_runtime.internal.refusal import LaunchRefusal

PACKAGE_RELEASE_FORMAT = "cozy.package.release/1"
ENVIRONMENT_FORMAT = "cozy.worker.v1.Environment/1"
MAX_OVERLAY_BYTES = 4 << 30
BOOT_ROOT = host_paths.BOOT_DATA
ENVIRONMENT_ROOT = host_paths.ENVIRONMENT_ROOT
LOCAL_ARTIFACT_STORE_ROOT = host_paths.ARTIFACT_STORE
ARTIFACT_CACHE_ROOT = host_paths.ARTIFACT_CACHE
MAX_ARTIFACT_BYTES = 8 << 30
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class PreparedBoot:
    """What a pod knows about itself once it is ready and still empty."""

    artifact_store: Path
    artifact_cache: Path


def prepare_boot(
    config: WorkerHostConfig,
    *,
    boot_root: Path = BOOT_ROOT,
    artifact_store_root: Path = LOCAL_ARTIFACT_STORE_ROOT,
    artifact_cache_root: Path = ARTIFACT_CACHE_ROOT,
) -> PreparedBoot:
    """Bring the pod to READY-BUT-EMPTY: check the deadline and lay its two CAS roots.

    The deadline check converts a boot that is already too late into a NAMED refusal.
    Tensorhub stops polling at the same instant either way, so all it decides is whether
    the operator reads a reason or reads silence.

    THE POD MEASURES NOTHING ABOUT ITS OWN IMAGE. cr-048 measured the control-Runtime wheel
    baked into the base worker image and reported the digest so "the hub learns a value".
    But the hub pinned that image BY DIGEST — tensorhub registers a base worker image as
    `name@sha256:…` and refuses a tag — so the wheel inside it is a fact the hub CHOSE,
    not one the pod can teach it. Hashing image-resident bytes to echo them back proves
    something the OCI digest already proved before this process existed.

    PlacementSet/3 names model Manifests directly. There is no empty object-set document to
    publish or reconcile at boot.
    """

    if time.time() >= config.lease_expiry_unix:
        raise LaunchRefusal("worker_lease_elapsed", str(config.lease_expiry_unix))
    # Package executors read verified TensorFS objects but never publish them. Root owns
    # the store; execute/read bits admit the fixed executor uid without making any object
    # or directory writable to it.
    _directory(artifact_store_root, mode=0o755)
    _directory(artifact_cache_root)
    _directory(boot_root)
    return PreparedBoot(
        artifact_store=artifact_store_root,
        artifact_cache=artifact_cache_root,
    )


def fetch_exact(
    grant: Mapping[str, Any], root: Path, *, deadline: int, allow_local: bool = False
) -> Path:
    """Fetch one bounded exact grant without proxy, redirect, transform, or alias."""

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
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    measured = hashlib.sha256()
    total = 0
    try:
        timeout = max(float(deadline) - time.time(), 0.001)
        try:
            response = pull.open_guarded(
                url,
                pull.Deadline.after(timeout),
                headers={
                    "Accept-Encoding": "identity",
                    "User-Agent": "cozy-runtime-worker/1",
                },
                allow_local=allow_local,
                no_proxy=True,
            )
        except (OSError, urllib.error.URLError, pull.PullRefusal) as exc:
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


def _publish_exact(target: Path, payload: bytes, root: Path) -> None:
    """Write one exact document once; an exact replay converges, different bytes refuse."""

    if target.exists():
        if target.read_bytes() != payload:
            raise LaunchRefusal("boot_document_conflict", str(target))
        return
    temporary = root / f".boot-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise LaunchRefusal("boot_document_conflict", str(target)) from None
            return
        _fsync(root)
    finally:
        temporary.unlink(missing_ok=True)


def _directory(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not path.is_absolute() or not stat.S_ISDIR(info.st_mode):
        raise LaunchRefusal("provision_store_invalid", str(path))
    path.chmod(mode)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def baked_base_manifest(
    manifest_path: Path = overlay.DEFAULT_BASE_MANIFEST,
) -> overlay.BaseManifest:
    """Read the trusted image-owned base profile before any hardware probe."""

    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise LaunchRefusal("control_runtime_manifest_unreadable", str(exc)) from exc
    manifest_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        return overlay.parse_base_manifest(raw, manifest_digest)
    except overlay.OverlayRefusal as exc:
        raise LaunchRefusal(exc.code, exc.detail) from exc


def control_runtime_digest(
    manifest_path: Path = overlay.DEFAULT_BASE_MANIFEST,
    *,
    manifest: overlay.BaseManifest | None = None,
) -> str:
    """Read the exact image-owned control Runtime wheel identity.

    This is not the OCI manifest digest and it is not accepted from provider environment.
    The registered base worker image bakes the complete wheelhouse manifest; Runtime
    validates those exact canonical bytes and reports its one ``cozy-runtime`` wheel fact
    so the hub can compare it with the selected image record.
    """

    if manifest is None:
        manifest = baked_base_manifest(manifest_path)
    matches = [wheel.digest for wheel in manifest.wheels if wheel.distribution == "cozy-runtime"]
    if len(matches) != 1:
        raise LaunchRefusal(
            "control_runtime_identity_invalid",
            f"base manifest names {len(matches)} cozy-runtime wheels",
        )
    return matches[0]


def package_overlay_input(
    *,
    package_release_bytes: bytes,
    environment: Mapping[str, Any],
    environment_digest: str,
    wheelhouse_manifest_bytes: bytes,
    wheel_paths: Mapping[str, Path],
    verified_wheels: Mapping[str, overlay.VerifiedFile] | None = None,
) -> overlay.Input:
    """Join one PackageRelease with the selected inline Environment.

    PackageRelease is environment-independent. The Placement selects a WheelhouseManifest and
    exact non-base overlay wheels directly, so there is no bundle, resolution lock, resolved
    wheel set, or independent environment-specification document to reconcile.
    """

    release = _document(package_release_bytes, "package_release")
    _closed(
        release,
        {
            "dependency_wheels",
            "format",
            "package",
            "package_descriptor",
            "project_wheel",
            "release",
            "requires_python",
            "requirements",
            "source_files",
        },
        "package_release",
    )
    if release["format"] != PACKAGE_RELEASE_FORMAT:
        raise LaunchRefusal("package_release_format", str(release["format"]))
    package = release["package"]
    version = release["release"]
    if (
        not isinstance(package, str)
        or package.count("/") != 1
        or not all(package.split("/"))
        or not isinstance(version, str)
        or not version
    ):
        raise LaunchRefusal("package_release_invalid", "package or release identity is invalid")
    _ref(release["package_descriptor"], "package_descriptor")
    source_files = release["source_files"]
    if not isinstance(source_files, list):
        raise LaunchRefusal("package_release_invalid", "source_files is not an array")
    prior_path = ""
    for source in source_files:
        _closed(source, {"digest", "length", "path"}, "source_file")
        source_path = source["path"]
        if (
            not isinstance(source["digest"], str)
            or not source["digest"].startswith("sha256:")
            or not isinstance(source["length"], int)
            or source["length"] < 0
            or not isinstance(source_path, str)
            or not source_path
            or source_path <= prior_path
        ):
            raise LaunchRefusal("package_release_invalid", "source_files is invalid or unsorted")
        prior_path = source_path
    requires_python = release["requires_python"]
    requirements = release["requirements"]
    if not isinstance(requires_python, str) or not requires_python:
        raise LaunchRefusal("package_release_invalid", "requires_python is empty")
    if (
        not isinstance(requirements, list)
        or any(not isinstance(item, str) or not item for item in requirements)
        or requirements != sorted(set(requirements))
    ):
        raise LaunchRefusal("package_release_invalid", "requirements are not sorted unique")
    published_dependencies = _wheels(
        release["dependency_wheels"], "package_release", import_roots=True, nested_ref=True
    )

    _closed(environment, {"wheelhouse_manifest", "wheels"}, "environment")
    wheelhouse_ref = _ref(environment["wheelhouse_manifest"], "wheelhouse_manifest")
    measured_environment = (
        "sha256:"
        + hashlib.sha256(canonical.write({**environment, "format": ENVIRONMENT_FORMAT})).hexdigest()
    )
    if measured_environment != environment_digest:
        raise LaunchRefusal(
            "environment_digest_mismatch",
            f"measured {measured_environment}, expected {environment_digest}",
        )
    _ref_matches(wheelhouse_ref, wheelhouse_manifest_bytes, "wheelhouse_manifest")
    try:
        overlay.parse_base_manifest(wheelhouse_manifest_bytes, str(wheelhouse_ref["digest"]))
    except overlay.OverlayRefusal as exc:
        raise LaunchRefusal(exc.code, exc.detail) from exc

    project = _wheel(release["project_wheel"], "project", import_roots=True, nested_ref=True)
    if project.tags != ("py3-none-any",) or not project.filename.endswith("-py3-none-any.whl"):
        raise LaunchRefusal("project_wheel_not_pure", "project wheel must be exactly py3-none-any")
    wheels = [project]
    custom_wheels = _wheels(
        environment.get("wheels", []), "environment", import_roots=True, nested_ref=True
    )
    published_by_distribution = {wheel.distribution: wheel for wheel in published_dependencies}
    for selected in custom_wheels:
        published = published_by_distribution.get(selected.distribution)
        if published is not None and selected != published:
            raise LaunchRefusal(
                "package_dependency_selection_mismatch",
                f"selected dependency {selected.distribution} contradicts its published wheel",
            )
    measured_custom_bytes = sum(wheel.length for wheel in custom_wheels)
    if project.length > MAX_OVERLAY_BYTES - measured_custom_bytes:
        raise LaunchRefusal(
            "overlay_byte_budget_exceeded",
            f"project_plus_custom={project.length + measured_custom_bytes} "
            f"budget={MAX_OVERLAY_BYTES}",
        )
    wheels.extend(custom_wheels)
    digests = [wheel.digest for wheel in wheels]
    distributions = [wheel.distribution for wheel in wheels]
    if len(digests) != len(set(digests)):
        raise LaunchRefusal("wheel_digest_duplicate", "one digest has multiple owners")
    if len(distributions) != len(set(distributions)):
        raise LaunchRefusal("wheel_distribution_duplicate", "one distribution has multiple owners")
    granted_digests = {wheel.digest for wheel in wheels}
    if set(wheel_paths) != granted_digests:
        missing = sorted(granted_digests - set(wheel_paths))
        extra = sorted(set(wheel_paths) - granted_digests)
        raise LaunchRefusal("wheel_grant_set_mismatch", f"missing={missing[:1]} extra={extra[:1]}")
    return overlay.Input(
        environment_digest=environment_digest,
        base_family_digest=_digest(wheelhouse_manifest_bytes),
        project_wheel_digest=project.digest,
        wheels=tuple(wheels),
        grants=tuple(
            overlay.BlobGrant(
                wheel.digest,
                wheel.length,
                wheel_paths[wheel.digest],
                (verified_wheels or {}).get(wheel.digest),
            )
            for wheel in wheels
        ),
    )


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


def _wheels(
    value: object, owner: str, *, import_roots: bool = False, nested_ref: bool = False
) -> list[overlay.Wheel]:
    if not isinstance(value, list):
        raise LaunchRefusal(f"{owner}_wheel_inventory_invalid", "wheels is not a list")
    wheels = [_wheel(row, owner, import_roots=import_roots, nested_ref=nested_ref) for row in value]
    names = [wheel.distribution for wheel in wheels]
    if names != sorted(names) or len(names) != len(set(names)):
        raise LaunchRefusal(
            f"{owner}_wheel_inventory_invalid", "distributions are not sorted unique"
        )
    return wheels


def _wheel(
    value: object, owner: str, *, import_roots: bool = False, nested_ref: bool = False
) -> overlay.Wheel:
    keys = (
        {"ref", "distribution", "filename", "tags", "version"}
        if nested_ref
        else {
            "digest",
            "distribution",
            "filename",
            "length",
            "tags",
            "version",
        }
    )
    if import_roots:
        keys.add("import_roots")
    if not isinstance(value, dict) or set(value) != keys:
        raise LaunchRefusal(f"{owner}_wheel_invalid", "wheel row has the wrong fields")
    reference = _ref(value["ref"], f"{owner}_wheel") if nested_ref else value
    digest = reference["digest"]
    _require_digest(digest, f"{owner}_wheel.digest")
    distribution = value["distribution"]
    filename = value["filename"]
    version = value["version"]
    strings = (distribution, filename, version)
    if (
        not all(isinstance(item, str) and item for item in strings)
        or not overlay.normalized_distribution(str(distribution))
        or not overlay.valid_wheel_filename(str(filename))
        or not overlay.valid_wheel_version(str(version))
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "wheel strings are empty or invalid")
    length = reference["length"]
    tags = value["tags"]
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        or length > overlay.MAX_WHEEL_BYTES
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "length is not positive and bounded")
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(tag, str) and overlay.valid_wheel_tag(tag) for tag in tags)
        or tags != sorted(set(tags))
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "tags are not non-empty sorted unique")
    roots = value.get("import_roots", [])
    if (
        not isinstance(roots, list)
        or not all(isinstance(root, str) and overlay.valid_import_root(root) for root in roots)
        or roots != sorted(set(roots))
    ):
        raise LaunchRefusal(
            f"{owner}_wheel_invalid", "import_roots are not sorted unique identifiers"
        )
    return overlay.Wheel(
        distribution=value["distribution"],
        version=value["version"],
        filename=value["filename"],
        digest=str(digest),
        length=length,
        tags=tuple(tags),
        owner={
            "project": "project",
            "package_release": "custom",
            "environment": "custom",
        }[owner],
        import_roots=tuple(roots),
    )


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or canonical.DIGEST_RE.fullmatch(value) is None:
        raise LaunchRefusal(f"{name}_invalid", "not sha256:<64 lowercase hex>")


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
