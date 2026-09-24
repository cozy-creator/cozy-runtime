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
published, zero placements. What survives here is
``package_environment_input`` — the small source/dependency-wheel join a placement converges
to — and ``prepare_boot``. The grant-fetch primitive (``fetch_exact``) left with cr-090:
this runtime downloads nothing, and the bytes a placement resolves are already local — the
TensorFS Store filled by TensorFS's one transport, or supervisor-verified files.
"""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical, fill, host_paths, package_environment
from cozy_runtime.internal.config import WorkerHostConfig
from cozy_runtime.internal.refusal import LaunchRefusal

ENVIRONMENT_FORMAT = "cozy.worker.v1.Environment/1"
MAX_PACKAGE_ENVIRONMENT_BYTES = 4 << 30
BOOT_ROOT = host_paths.BOOT_DATA
INSTALL_ROOT = host_paths.INSTALL_ROOT
LOCAL_TENSORFS_ROOT = host_paths.TENSORFS_ROOT
MAX_ARTIFACT_BYTES = 8 << 30


def prepare_boot(
    config: WorkerHostConfig,
    *,
    boot_root: Path = BOOT_ROOT,
    tensorfs_root: Path = LOCAL_TENSORFS_ROOT,
) -> Path:
    """Bring the pod to READY-BUT-EMPTY by opening its local TensorFS Store.

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
    # pod-supervisor already ensured this local Store and prepared its reader permissions
    # through TensorFS. Runtime validates it without creating, chmodding, or reading layout.
    try:
        fill.open_store(fill.tensorfs_module(), tensorfs_root)
    except fill.FillRefusal as exc:
        raise LaunchRefusal("provision_store_invalid", str(exc)) from exc
    _directory(boot_root)
    return tensorfs_root


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


def _directory(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not path.is_absolute() or not stat.S_ISDIR(info.st_mode):
        raise LaunchRefusal("provision_store_invalid", str(path))
    if stat.S_IMODE(info.st_mode) != mode:
        path.chmod(mode)


def package_environment_input(
    *,
    package_name: str,
    release: str,
    project_wheel: Mapping[str, Any],
    environment: Mapping[str, Any],
    environment_digest: str,
    wheel_paths: Mapping[str, Path],
    verified_wheels: Mapping[str, package_environment.VerifiedFile] | None = None,
) -> package_environment.Input:
    """Join one local revision's exact wheel facts with its inline Environment.

    Wire 30 leaves this join to LOCAL packages only: a published install materializes from
    its locked-requirements export and never re-derives wheel facts. The project wheel fact
    rides DevelopmentPackage; the supplied dependency wheels ride Environment.local_wheels.
    """

    if (
        not isinstance(package_name, str)
        or package_name.count("/") != 1
        or not all(package_name.split("/"))
        or not isinstance(release, str)
        or not release
    ):
        raise LaunchRefusal("package_invalid", "package or release identity is invalid")
    _closed(environment, {"local_wheels"}, "environment")
    measured_environment = (
        "sha256:"
        + hashlib.sha256(canonical.write({**environment, "format": ENVIRONMENT_FORMAT})).hexdigest()
    )
    if measured_environment != environment_digest:
        raise LaunchRefusal(
            "environment_digest_mismatch",
            f"measured {measured_environment}, expected {environment_digest}",
        )
    project = _wheel(project_wheel, "project", import_roots=True, nested_ref=True)
    if project.distribution != package_name.split("/", 1)[1] or project.version != release:
        raise LaunchRefusal("project_wheel_mismatch", "project wheel differs from package release")
    wheels = [project]
    custom_wheels = _wheels(
        environment.get("local_wheels", []), "environment", import_roots=True, nested_ref=True
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
