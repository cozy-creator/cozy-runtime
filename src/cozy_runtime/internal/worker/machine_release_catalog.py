"""Published releases as this machine reads them at its public catalog: the newest release,
a release's interface and install facts, and the published callees its lock names. A
controller supplies none of these; each is immutable per release and read once."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote, urlsplit

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from cozy_runtime.internal import canonical, package_interface
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

from .machine_model_resolve import Catalog, hub_key, own
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .session import Worker

_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _path(package: str) -> str:
    if not _PACKAGE.fullmatch(package) or package.startswith("local/"):
        raise WorkspaceRefusal(f"{package!r} names no published package")
    org, _, name = package.partition("/")
    return f"/v1/packages/{quote(org)}/{quote(name)}"


def newest(catalog: Catalog, package: str) -> str:
    """The newest non-yanked release by PEP 440 order; a prerelease only when there is no
    final release."""
    card = catalog.get(_path(package))
    rows = card.get("releases") if isinstance(card, dict) else None
    if not isinstance(rows, list):
        raise WorkspaceRefusal(f"{package} is not a published package at the catalog")
    ranked: list[tuple[bool, Version, str]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("yanked") or row.get("yanked_at"):
            continue
        try:
            version = Version(str(row.get("release", "")))
        except InvalidVersion:
            continue
        ranked.append((not version.is_prerelease, version, str(row["release"])))
    if not ranked:
        raise WorkspaceRefusal(f"{package} has no unyanked release")
    return max(ranked)[2]


def facts(catalog: Catalog, package: str, release: str) -> pb.DeferredInstallation:
    """One release's install facts: its interface, Python and locked requirements."""
    path = f"{_path(package)}/releases/{quote(release)}"
    detail = catalog.get(path)
    if not isinstance(detail, dict):
        raise WorkspaceRefusal(f"{package}@{release} is not a published release at the catalog")
    summary = detail.get("release")
    if not isinstance(summary, dict) or summary.get("release") != release:
        raise WorkspaceRefusal(f"the catalog answered another release for {package}@{release}")
    interface = canonical.write(detail.get("package_interface"))
    body = package_interface.read_bytes(interface, f"{package}@{release}")
    locked = catalog.text(
        path + "/locked-requirements", weights_limits.MAX_LOCKED_REQUIREMENTS_BYTES
    )
    if not locked:
        raise WorkspaceRefusal(f"the catalog holds no locked requirements for {package}@{release}")
    return pb.DeferredInstallation(
        key=f"{package}@{release}",
        package=package,
        release=release,
        preparation=pb.PreparePackageSetRequest(
            download_delegation=canonical.write(
                {
                    "format": "cozy.worker.v1.DownloadDelegation/1",
                    "models": [],
                    "packages": [{"package": package, "release": release}],
                }
            ),
            application=str(body["application"]),
            python_requires=str(detail.get("requires_python") or ""),
            python_version=str(detail.get("python_version") or ""),
            locked_requirements=locked,
            package_interface=interface,
        ),
    )


def callees(package: str, locked: bytes) -> list[str]:
    """The published releases a lock names: its own org's index wheels other than the
    package itself, as `org/name@version`."""
    org, _, name = package.partition("/")
    own = re.sub(r"[-_.]+", "-", name.lower())
    prefix = f"/v1/index/{org}/files/"
    out: set[str] = set()
    for line in locked.decode("utf-8", errors="replace").splitlines():
        _, at, rest = line.partition(" @ ")
        location = urlsplit(rest.split(" ", 1)[0]) if at else None
        if location is None or not location.path.startswith(prefix):
            continue
        try:
            distribution, version, _, _ = parse_wheel_filename(
                unquote(PurePosixPath(location.path).name)
            )
        except InvalidWheelFilename:
            continue
        if distribution != own:
            out.add(f"{org}/{distribution}@{version}")
    return sorted(out)


def complete(worker: Worker, request: pb.PreparePackageSetRequest) -> pb.PreparePackageSetRequest:
    """A preparation naming a release without its facts, completed from the Hub it names
    (empty: this machine's default); one that carries them is prepared as sent. Its Hub is
    spelled as `hub_key` keys it, so what it installs is held as that Hub's."""
    hub = hub_key(worker, request.hub)
    if request.locked_requirements and hub == request.hub:
        return request
    completed = pb.PreparePackageSetRequest()
    completed.CopyFrom(request)
    completed.hub = hub
    if request.locked_requirements:
        return completed
    try:
        packages = documents.read(request.download_delegation, pb.DownloadDelegation)["packages"]
    except (documents.DocumentError, KeyError) as exc:
        raise WorkspaceRefusal("preparation names no package release") from exc
    if not isinstance(packages, list) or len(packages) != 1 or not isinstance(packages[0], dict):
        raise WorkspaceRefusal("preparation names no single package release")
    selected = facts(
        own(worker, hub=hub), str(packages[0].get("package")), str(packages[0].get("release"))
    )
    for name in (
        "application",
        "python_requires",
        "python_version",
        "locked_requirements",
        "package_interface",
    ):
        setattr(completed, name, getattr(selected.preparation, name))
    return completed
