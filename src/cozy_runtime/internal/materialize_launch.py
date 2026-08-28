"""Canonical release-document adapter for the offline environment materializer.

Tensorhub owns the document schemas and delivery grants.  This module owns the one Runtime
join from their already-fetched exact bytes to :class:`environment.EnvironmentInput`.  Local
paths are deliberately function arguments: they never enter a canonical release document.

cr-048 RETIRED THE BOOT CLOSURE. ``tensorhub.rental_provision_spec/1`` and
``tensorhub.rental_provision_bundle/1`` named one endpoint's whole closure at boot — eight
document roles, one artifact grant per wheel and weight, the whole materialization root
keyed by the bundle digest — which is the pre-#459 one-pod-one-endpoint shape that rev-2's
``DesiredPlacementSet`` + ``ArtifactGrantUpdate`` replaced. The pod now boots to
ready-but-EMPTY: control runtime verified, GPU inventory measured, listeners proven, receipt
published, zero placements. What survives here is the DISCIPLINE, not the vehicle:
``fetch_exact`` (one bounded exact-digest HTTPS grant) and ``materialize_objects`` (exact CAS
admission) are the primitives the dynamic grant lane fetches through, and
``materialize_exact``/``environment_input`` are the receipt-keyed join a placement converges
to. Only the boot-only bundling retired.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import time
import urllib.error
import urllib.parse
import uuid
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, environment, fill, provisioned, pull
from cozy_runtime.internal.config import AdapterConfig

ENDPOINT_BUNDLE_FORMAT = "tensorhub.endpoint_bundle/1"
ENVIRONMENT_SPEC_FORMAT = "cozy.worker.v1.EndpointEnvironmentSpec/1"
RESOLVED_WHEEL_SET_FORMAT = "ResolvedWheelSet/1"
WHEELHOUSE_MANIFEST_FORMAT = "WheelhouseManifest/1"
CONTROL_RUNTIME_WHEEL = Path("/opt/cozy/control-runtime/cozy-runtime.whl")
# execution-substrates creates and grants the runtime exactly this container-local data root.
# Every durable-for-this-pod child stays beneath it; /run/cozy remains restart-local control state.
LOCAL_DATA_ROOT = Path("/var/lib/cozy")
BOOT_ROOT = LOCAL_DATA_ROOT / "boot"
ENVIRONMENT_ROOT = LOCAL_DATA_ROOT / "environments"
LOCAL_ARTIFACT_STORE_ROOT = LOCAL_DATA_ROOT / "store"
PLANS_ROOT = Path("/run/cozy/binding-plans")
BINDING_DOCUMENTS_ROOT = Path("/run/cozy/binding-documents")
#: The canonical EMPTY ArtifactObjectSet. A pod that boots with no placement holds no
#: objects, and the worker's launch grammar requires the document to exist and be exact
#: rather than letting "absent" mean "anything".
EMPTY_ARTIFACT_OBJECT_SET = b'{"kind":"tensorhub.resolved_object_set/1","roots":[]}'
MAX_ARTIFACT_BYTES = 8 << 30
CHUNK_BYTES = 1 << 20


class LaunchRefusal(Exception):
    """A fail-closed release/grant join verdict with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class PreparedBoot:
    """What a pod knows about itself once it is ready and still empty."""

    control_runtime_wheel_digest: str
    control_runtime_wheel_length: int
    artifact_store: Path
    artifact_object_set: Path


@dataclass(frozen=True, slots=True)
class MaterializedObjects:
    store: Path
    sources: dict[str, dict[str, int]]
    network_cache_degradations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheMount:
    """One attached provider network-cache volume, for :func:`materialize_objects`.

    The retired provision spec was this value's only carrier. Nothing supplies it at boot
    any more — an empty pod materializes nothing — so it arrives with the dynamic grant lane
    that will drive object materialization, or not at all.
    """

    mount_path: Path
    cache_volume_id: str
    provider_volume_id: str


def prepare_boot(
    config: AdapterConfig,
    *,
    boot_root: Path = BOOT_ROOT,
    artifact_store_root: Path = LOCAL_ARTIFACT_STORE_ROOT,
    control_runtime_wheel: Path = CONTROL_RUNTIME_WHEEL,
) -> PreparedBoot:
    """Bring the pod to READY-BUT-EMPTY and MEASURE what the hub will compare.

    No document is read, and nothing is fetched: this whole function is local. It binds the
    imported control Runtime to the wheel the substrate baked, measures that wheel's exact
    identity for the receipt, and lays down the empty artifact store plus the exact empty
    ArtifactObjectSet the worker's launch grammar requires.

    THE CONTROL-RUNTIME DIGEST IS REPORTED, NOT DICTATED. It used to be
    `_verify_local(wheel, spec["execution_substrate"]["control_runtime"]…)` — the pod
    comparing the image against a number the hub had just told it, which teaches the hub
    only pass/fail about a fact it already asserted. `verify_control_install` below is the
    half that could never be an env var (it binds the IMPORTED package to the wheel bytes)
    and it is unchanged; the other half becomes `measured_control_runtime_wheel_digest` in
    the receipt, which the hub joins against its own ExecutionSubstrate row before it will
    trust the TLS peer at all. Same fail-closed property, one fewer env name, and the hub
    learns the measured value instead of a boolean.
    """

    if time.time() >= config.readiness_deadline_unix:
        raise LaunchRefusal("readiness_deadline_elapsed", str(config.readiness_deadline_unix))
    verify_control_install(control_runtime_wheel)
    wheel_digest, wheel_length = _measure_local(control_runtime_wheel)
    _directory(artifact_store_root)
    _directory(boot_root)
    object_set = boot_root / "artifact-object-set.json"
    _publish_exact(object_set, EMPTY_ARTIFACT_OBJECT_SET, boot_root)
    return PreparedBoot(
        control_runtime_wheel_digest=wheel_digest,
        control_runtime_wheel_length=wheel_length,
        artifact_store=artifact_store_root,
        artifact_object_set=object_set,
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
                    "User-Agent": "cozy-materialize-launch/1",
                },
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


def materialize_objects(
    grants: Mapping[str, str],
    object_set: provisioned.ArtifactObjectSet,
    *,
    cache: CacheMount | None,
    deadline: int | None,
    store_root: Path,
) -> MaterializedObjects:
    """Admit each exact model closure into the required worker-local TensorFS CAS.

    `grants` is a per-object-id location map — the shape rev-2's `ArtifactGrantUpdate`
    carries. It used to be spelled as `bundle["artifact_grants"]`; the closure vehicle
    retired, the exact-digest admission it drove did not.
    """

    refs = object_set.refs
    required = set(refs)
    missing = sorted(required - set(grants))
    if missing:
        raise LaunchRefusal("object_grant_missing", missing[0])
    cache_root: Path | None = None
    if cache is not None:
        cache_root = cache.mount_path / "tensorfs"
        _require_separate_store_roots(store_root, cache.mount_path, cache_root)
    degradations: set[str] = set()
    source_by_object: dict[str, str] = {}
    try:
        if not object_set.roots:
            _directory(store_root)
            return MaterializedObjects(
                store=store_root,
                sources=_source_usage(source_by_object, refs),
                network_cache_degradations=(),
            )
        tensorfs = fill.tensorfs_module()
        store = pull.open_store(tensorfs, store_root)
        cache_store = None
        if cache is not None:
            assert cache_root is not None
            cache_store = _open_cache_store(tensorfs, cache.mount_path, cache_root, degradations)
        for group in object_set.roots:
            closure = (group.root, *group.objects)
            rows = sorted(
                (
                    pull.Read(
                        object_id=ref.digest,
                        length=ref.length,
                        url=grants[ref.digest],
                    )
                    for ref in closure
                ),
                key=lambda row: row.object_id,
            )
            timeout = None if deadline is None else float(deadline) - time.time()
            if timeout is not None and timeout <= 0:
                raise LaunchRefusal("readiness_deadline_elapsed", str(deadline))
            outcome = pull.pull_granted_snapshot(
                tensorfs,
                store,
                group.root.digest,
                group.root.length,
                rows,
                timeout=timeout,
                session="provision-" + group.root.digest.removeprefix("sha256:")[:12],
                cache_store=cache_store,
            )
            for object_id, source in outcome.sources.items():
                source_by_object.setdefault(object_id, source)
            degradations.update(outcome.cache_degradations)
        if set(source_by_object) != required:
            raise LaunchRefusal(
                "object_materialization_refused", "source evidence does not cover the object set"
            )
    except LaunchRefusal:
        raise
    except pull.PullRefusal as exc:
        raise LaunchRefusal("object_materialization_refused", f"{exc.code}: {exc.message}") from exc
    except Exception as exc:
        raise LaunchRefusal(
            "object_materialization_refused", f"TensorFS unavailable: {type(exc).__name__}"
        ) from exc
    return MaterializedObjects(
        store=store_root,
        sources=_source_usage(source_by_object, refs),
        network_cache_degradations=tuple(sorted(degradations)),
    )


def _open_cache_store(
    tensorfs: Any, cache_mount: Path, cache_root: Path, degradations: set[str]
) -> Any | None:
    try:
        if not stat.S_ISDIR(cache_mount.stat().st_mode):
            raise NotADirectoryError(cache_mount)
        return pull.open_store(tensorfs, cache_root)
    except (OSError, pull.PullRefusal):
        degradations.add("unavailable")
        return None


def _source_usage(
    source_by_object: Mapping[str, str], refs: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    usage = {source: {"bytes": 0, "objects": 0} for source in ("cloud", "local", "network_cache")}
    for object_id, source in source_by_object.items():
        row = usage.get(source)
        if row is None or object_id not in refs:
            raise LaunchRefusal("object_materialization_refused", "invalid source evidence")
        row["bytes"] += refs[object_id]
        row["objects"] += 1
    return usage


def _require_separate_store_roots(local_store: Path, cache_mount: Path, cache_root: Path) -> None:
    """Refuse lexical, symlink, and bind-mount aliases before either store can be touched."""

    def overlaps(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    lexical_local = Path(os.path.abspath(local_store))
    lexical_mount = Path(os.path.abspath(cache_mount))
    lexical_cache = Path(os.path.abspath(cache_root))
    if overlaps(lexical_local, lexical_mount) or overlaps(lexical_local, lexical_cache):
        raise LaunchRefusal("artifact_store_alias", "local CAS and network cache paths overlap")
    try:
        resolved_local = local_store.resolve(strict=False)
        resolved_mount = cache_mount.resolve(strict=False)
        resolved_cache = cache_root.resolve(strict=False)
    except OSError:
        resolved_local, resolved_mount, resolved_cache = (
            lexical_local,
            lexical_mount,
            lexical_cache,
        )
    if overlaps(resolved_local, resolved_mount) or overlaps(resolved_local, resolved_cache):
        raise LaunchRefusal("artifact_store_alias", "local CAS and network cache paths alias")

    local_lineage = (local_store, *local_store.parents[:-1])
    cache_lineage = (cache_mount, *cache_mount.parents[:-1])
    try:
        if any(
            candidate.exists() and os.path.samefile(candidate, cache_mount)
            for candidate in local_lineage
        ) or any(
            candidate.exists() and os.path.samefile(candidate, local_store)
            for candidate in cache_lineage
        ):
            raise LaunchRefusal("artifact_store_alias", "local CAS and network cache paths alias")
        if (
            cache_root.exists()
            and local_store.exists()
            and os.path.samefile(local_store, cache_root)
        ):
            raise LaunchRefusal("artifact_store_alias", "local CAS and network cache paths alias")
    except OSError:
        # An unavailable cache is allowed; lexical and resolvable aliases have already refused.
        return


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


def _measure_local(path: Path) -> tuple[str, int]:
    """Measure one local regular file's exact identity. Reported, never asserted."""

    try:
        info = path.lstat()
    except OSError as exc:
        raise LaunchRefusal("local_artifact_unreadable", f"{path}: {exc}") from exc
    if (
        not path.is_absolute()
        or not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size <= MAX_ARTIFACT_BYTES
    ):
        raise LaunchRefusal("local_artifact_invalid", str(path))
    measured = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            measured.update(chunk)
    return "sha256:" + measured.hexdigest(), info.st_size


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
