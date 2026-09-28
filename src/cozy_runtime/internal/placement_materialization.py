"""Prepare an empty worker and its TensorFS Store; package installs are uv-owned."""

from __future__ import annotations

import stat
from pathlib import Path

from cozy_runtime.internal import fill
from cozy_runtime.internal.config import WorkerHostConfig
from cozy_runtime.internal.refusal import LaunchRefusal


def prepare_boot(
    config: WorkerHostConfig,
    *,
    boot_root: Path | None = None,
    tensorfs_root: Path | None = None,
) -> Path:
    """Bring the pod to READY-BUT-EMPTY by opening its local TensorFS Store.

    A boot ends when the supervisor cancels it or the leg exits.

    THE POD MEASURES NOTHING ABOUT ITS OWN IMAGE. cr-048 measured the control-Runtime wheel
    baked into the base worker image and reported the digest so "the hub learns a value".
    But the hub pinned that image BY DIGEST — tensorhub registers a base worker image as
    `name@sha256:…` and refuses a tag — so the wheel inside it is a fact the hub CHOSE,
    not one the pod can teach it. Hashing image-resident bytes to echo them back proves
    something the OCI digest already proved before this process existed.

    PlacementSet/1 names model Manifests directly. There is no empty object-set document to
    publish or reconcile at boot.
    """

    boot_root = boot_root or config.layout.boot_data
    tensorfs_root = tensorfs_root or config.layout.tensorfs_root
    # The Host already ensured this local Store and prepared its reader permissions
    # through TensorFS. Runtime validates it without creating, chmodding, or reading layout.
    try:
        fill.open_store(tensorfs_root)
    except fill.FillRefusal as exc:
        raise LaunchRefusal("provision_store_invalid", str(exc)) from exc
    _directory(boot_root)
    return tensorfs_root


def _directory(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not path.is_absolute() or not stat.S_ISDIR(info.st_mode):
        raise LaunchRefusal("provision_store_invalid", str(path))
    if stat.S_IMODE(info.st_mode) != mode:
        path.chmod(mode)
