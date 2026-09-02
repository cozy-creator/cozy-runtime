"""Exact artifact intake and package-environment document join.

Tensorhub owns the document schemas and delivery grants.  This module owns the one Runtime
join from their already-fetched exact bytes to :class:`package_environment.Input`. Local paths are
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
``package_environment_input`` is the small source/dependency-wheel join a placement converges to.
Only the boot-only bundling retired.
"""

from __future__ import annotations

import hashlib
import os
import stat
import urllib.error
import urllib.parse
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, egress, host_paths, package_environment, pull
from cozy_runtime.internal.config import WorkerHostConfig
from cozy_runtime.internal.refusal import LaunchRefusal

ENVIRONMENT_FORMAT = "cozy.worker.v1.Environment/1"
MAX_PACKAGE_ENVIRONMENT_BYTES = 4 << 30
BOOT_ROOT = host_paths.BOOT_DATA
ENVIRONMENT_ROOT = host_paths.ENVIRONMENT_ROOT
LOCAL_WEIGHTS_STORE_ROOT = host_paths.WEIGHTS_STORE
WEIGHTS_CACHE_ROOT = host_paths.WEIGHTS_CACHE
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
    artifact_store_root: Path = LOCAL_WEIGHTS_STORE_ROOT,
    artifact_cache_root: Path = WEIGHTS_CACHE_ROOT,
) -> PreparedBoot:
    """Bring the pod to READY-BUT-EMPTY: lay its two CAS roots.

    No lease check (xs-007 row 18). This used to refuse `worker_lease_elapsed` against
    `COZY_WORKER_LEASE_EXPIRY_UNIX`, the rental cap tensorhub has since deleted; the
    supervisor still sends the name with a far-future sentinel and the runtime decides
    nothing on it. A boot ends when the supervisor cancels it or the leg exits.

    THE POD MEASURES NOTHING ABOUT ITS OWN IMAGE. cr-048 measured the control-Runtime wheel
    baked into the base worker image and reported the digest so "the hub learns a value".
    But the hub pinned that image BY DIGEST — tensorhub registers a base worker image as
    `name@sha256:…` and refuses a tag — so the wheel inside it is a fact the hub CHOSE,
    not one the pod can teach it. Hashing image-resident bytes to echo them back proves
    something the OCI digest already proved before this process existed.

    PlacementSet/1 names model Manifests directly. There is no empty object-set document to
    publish or reconcile at boot.
    """

    del config  # the handoff carries nothing this step decides on
    # Package executors read verified TensorFS objects but never publish them. Root owns
    # the store; execute/read bits admit the fixed executor uid without making any object
    # or directory writable to it. The artifact cache is the supervisor's landing area
    # (it takes ownership of it after this boot) and also carries the worker's staged
    # selection documents, which the executor reads by path: the execute bit alone lets
    # the fixed executor uid traverse to `selections/`; `sets/` stays the supervisor's 0700.
    _directory(artifact_store_root, mode=0o755)
    _grant_executor_holds(artifact_store_root)
    _directory(artifact_cache_root, mode=0o711)
    _directory(boot_root)
    return PreparedBoot(
        artifact_store=artifact_store_root,
        artifact_cache=artifact_cache_root,
    )


def fetch_exact(grant: Mapping[str, Any], root: Path, *, allow_local: bool = False) -> Path:
    """Fetch one bounded exact grant without proxy, redirect, transform, or alias.

    No deadline: the grant's expiry is the origin's to enforce, and a socket that moves
    nothing for `egress.SILENCE_SECONDS` is the one thing that ends the read early. The
    bytes take as long as they take.
    """

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
        try:
            response = pull.open_guarded(
                url,
                pull.Deadline(silence=egress.SILENCE_SECONDS),
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


def _grant_executor_holds(store_root: Path) -> None:
    """Lay the store and grant the executor uid exactly the two writes a reader makes.

    A TensorFS reader registers its hold before the first byte moves — a marker it creates
    under `tmp/leases` and a shared lock on `tmp/writers/recovery.lock` — so a sweep can
    never reap an object under a live lease. Those are the only writes package code ever
    makes to the store: the marker directory takes create-only (sticky, unlistable) and the
    lock file is openable read-write by everyone; every object, manifest and the catalog
    stay root's, read-only to the executor.
    """

    try:
        import tensorfs
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise LaunchRefusal("provision_store_invalid", "tensorfs reader is absent") from exc
    try:
        pull.open_store(tensorfs, store_root)
    except pull.PullRefusal as exc:
        raise LaunchRefusal("provision_store_invalid", str(exc)) from exc
    leases = store_root / "tmp" / "leases"
    writers = store_root / "tmp" / "writers"
    for path in (leases, writers):
        path.mkdir(parents=True, exist_ok=True)
    (store_root / "tmp").chmod(0o755)
    leases.chmod(0o1733)
    writers.chmod(0o755)
    lock = writers / "recovery.lock"
    os.close(os.open(lock, os.O_RDWR | os.O_CREAT, 0o666))
    lock.chmod(0o666)


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


def package_environment_input(
    *,
    package: Mapping[str, Any],
    environment: Mapping[str, Any],
    environment_digest: str,
    wheel_paths: Mapping[str, Path],
    verified_wheels: Mapping[str, package_environment.VerifiedFile] | None = None,
) -> package_environment.Input:
    """Join direct executable package facts with the selected inline Environment."""

    fields = set(package)
    published = {"package", "project_wheel", "release", "release_digest"}
    development = {
        "package",
        "local_revision_digest",
        "project_wheel",
        "release",
        "source_digest",
    }
    if fields not in (published, development):
        raise LaunchRefusal(
            "package_invalid", f"fields={sorted(fields)} expected published or local development"
        )
    identity = package["package"]
    version = package["release"]
    if (
        not isinstance(identity, str)
        or identity.count("/") != 1
        or not all(identity.split("/"))
        or not isinstance(version, str)
        or not version
    ):
        raise LaunchRefusal("package_invalid", "package or release identity is invalid")
    revision_field = "release_digest" if fields == published else "local_revision_digest"
    _require_digest(package[revision_field], f"package.{revision_field}")

    _closed(environment, {"wheels"}, "environment")
    measured_environment = (
        "sha256:"
        + hashlib.sha256(canonical.write({**environment, "format": ENVIRONMENT_FORMAT})).hexdigest()
    )
    if measured_environment != environment_digest:
        raise LaunchRefusal(
            "environment_digest_mismatch",
            f"measured {measured_environment}, expected {environment_digest}",
        )
    project = _wheel(package["project_wheel"], "project", import_roots=True, nested_ref=True)
    if project.distribution != identity.split("/", 1)[1] or project.version != version:
        raise LaunchRefusal("project_wheel_mismatch", "project wheel differs from package release")
    if project.tags != ("py3-none-any",) or not project.filename.endswith("-py3-none-any.whl"):
        raise LaunchRefusal("project_wheel_not_pure", "project wheel must be exactly py3-none-any")
    wheels = [project]
    custom_wheels = _wheels(
        environment.get("wheels", []), "environment", import_roots=True, nested_ref=True
    )
    measured_custom_bytes = sum(wheel.length for wheel in custom_wheels)
    if project.length > MAX_PACKAGE_ENVIRONMENT_BYTES - measured_custom_bytes:
        raise LaunchRefusal(
            "package_environment_byte_budget_exceeded",
            f"project_plus_custom={project.length + measured_custom_bytes} "
            f"budget={MAX_PACKAGE_ENVIRONMENT_BYTES}",
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
    return package_environment.Input(
        environment_digest=environment_digest,
        project_wheel_digest=project.digest,
        wheels=tuple(wheels),
        grants=tuple(
            package_environment.BlobGrant(
                wheel.digest,
                wheel.length,
                wheel_paths[wheel.digest],
                (verified_wheels or {}).get(wheel.digest),
            )
            for wheel in wheels
        ),
    )


def _closed(value: object, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        fields = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise LaunchRefusal(f"{name}_invalid", f"fields={fields} expected={sorted(keys)}")


def _matches(raw: bytes, digest: str, name: str) -> None:
    if _digest(raw) != digest:
        raise LaunchRefusal(
            f"{name}_digest_mismatch", f"measured {_digest(raw)}, expected {digest}"
        )


def _wheels(
    value: object, owner: str, *, import_roots: bool = False, nested_ref: bool = False
) -> list[package_environment.Wheel]:
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
) -> package_environment.Wheel:
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
        or not package_environment.normalized_distribution(str(distribution))
        or not package_environment.valid_wheel_filename(str(filename))
        or not package_environment.valid_wheel_version(str(version))
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "wheel strings are empty or invalid")
    length = reference["length"]
    tags = value["tags"]
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        or length > package_environment.MAX_WHEEL_BYTES
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "length is not positive and bounded")
    if (
        not isinstance(tags, list)
        or not tags
        or not all(
            isinstance(tag, str) and package_environment.valid_wheel_tag(tag) for tag in tags
        )
        or tags != sorted(set(tags))
    ):
        raise LaunchRefusal(f"{owner}_wheel_invalid", "tags are not non-empty sorted unique")
    roots = value.get("import_roots", [])
    if (
        not isinstance(roots, list)
        or not all(
            isinstance(root, str) and package_environment.valid_import_root(root) for root in roots
        )
        or roots != sorted(set(roots))
    ):
        raise LaunchRefusal(
            f"{owner}_wheel_invalid", "import_roots are not sorted unique identifiers"
        )
    return package_environment.Wheel(
        distribution=value["distribution"],
        version=value["version"],
        filename=value["filename"],
        digest=str(digest),
        length=length,
        tags=tuple(tags),
        owner={
            "project": "project",
            "environment": "custom",
        }[owner],
        import_roots=tuple(roots),
    )


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or canonical.DIGEST_RE.fullmatch(value) is None:
        raise LaunchRefusal(f"{name}_invalid", "not sha256:<64 lowercase hex>")


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
