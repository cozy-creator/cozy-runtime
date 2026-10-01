"""The machine filesystem layout, rooted where the Host says.

A pod roots it at `/`; a laptop's Host roots it at a directory it owns. The layout under the
root is identical, so the Host and Runtime share one relative table and nothing else. The one
TensorFS Store may live elsewhere (a laptop's `~/.tensorfs`); the Host names it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

POD_ROOT: Final = Path("/")


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path = POD_ROOT
    #: The box's TensorFS Store when the Host names one; else under the root.
    tensorfs: Path | None = None

    def _at(self, relative: str) -> Path:
        return self.root / relative

    @property
    def prepare_cache(self) -> Path:
        """The worker's content-addressed cache of small control documents (a prepared
        package's PackageInterface and the selections `acquire` stages beneath it). The
        worker creates it itself; no image mkdir decides who may write here."""
        return self._at("var/lib/cozy/prepare-cache")

    @property
    def tensorfs_root(self) -> Path:
        return self.tensorfs or self._at("var/lib/tensorfs")

    @property
    def boot_data(self) -> Path:
        return self._at("var/lib/cozy/boot")

    @property
    def install_root(self) -> Path:
        return self._at("var/lib/cozy/installs")

    @property
    def runtime_settings(self) -> Path:
        """The Runtime's settings file (`memo:`), written from the owner's config."""
        return self._at("etc/cozy/runtime.yaml")

    @property
    def machine_wheels(self) -> Path:
        """The Runtime and TensorFS wheels `cozy machine install` installed, kept verbatim."""
        return self._at("opt/cozy/wheels")

    @classmethod
    def of_install_root(cls, install_root: Path) -> Layout | None:
        """The machine layout an install root belongs to, or None for a root outside one."""
        root = install_root.parents[3] if len(install_root.parents) > 3 else None
        return cls(root) if root is not None and cls(root).install_root == install_root else None

    @property
    def dependency_cache(self) -> Path:
        return self._at("var/lib/cozy/dependencies")

    @property
    def kernel_cache(self) -> Path:
        """Compiled kernel objects keyed by every compile input (`internal/kernel_cache.py`):
        persistent machine state, unlike the per-boot JIT scopes under `run/cozy`."""
        return self._at("var/lib/cozy/kernels")

    @property
    def media_root(self) -> Path:
        return self._at("var/lib/cozy/media")

    @property
    def cozy_home(self) -> Path:
        return self._at("run/cozy")

    @property
    def bootstrap(self) -> Path:
        """The Host-to-worker handoff directory: the Host creates and owns it."""
        return self._at("run/cozy/bootstrap")

    @property
    def boot_id(self) -> Path:
        return self.bootstrap / "pod-boot-id"

    @property
    def readiness_payload(self) -> Path:
        return self.bootstrap / "readiness-payload"

    @property
    def readiness_envelope(self) -> Path:
        return self.bootstrap / "readiness-envelope.json"

    @property
    def worker_activity(self) -> Path:
        """What this worker is doing, rewritten every reporter tick for the Host's idle
        decision (`worker/activity.py`). It lives in the bootstrap directory because the
        Host can read there; `run/cozy/worker` is 0700 to the worker."""
        return self.bootstrap / "worker-activity"

    @property
    def tls_certificate(self) -> Path:
        return self.bootstrap / "tls.crt"

    @property
    def tls_private_key(self) -> Path:
        return self.bootstrap / "tls.key"

    @property
    def machine_publication_authority(self) -> Path:
        return self.bootstrap / "machine-publication-authority.json"

    @property
    def machine_hubs(self) -> Path:
        """Every other Hub this machine is registered at (wire 67); the Host writes it."""
        return self.bootstrap / "machine-hubs.json"

    @property
    def tensorhub_ca(self) -> Path:
        """The grant's private Hub CA, which the Host writes when its Hub uses one."""
        return self.bootstrap / "tensorhub-ca.crt"

    @property
    def worker_state(self) -> Path:
        return self._at("run/cozy/worker")

    @property
    def worker_control_address(self) -> Path:
        return self.worker_state / "control.addr"

    @property
    def triage_root(self) -> Path:
        """Under the media subtree while the media plane still serves `/v1/triage`."""
        return self.media_root / "triage"


def _absolute(raw: str, name: str) -> Path:
    if not os.path.isabs(raw) or os.path.normpath(raw) != raw:
        raise ValueError(f"{name} must be an absolute normalized path")
    return Path(raw)


def layout(raw: str | None, tensorfs: str | None = None) -> Layout:
    """The Host's `COZY_MACHINE_ROOT` and `COZY_TENSORFS_ROOT`: each absolute and normalized.
    An absent machine root is `/`; an absent Store is `var/lib/tensorfs` under it."""
    return Layout(
        _absolute(raw, "machine root") if raw else POD_ROOT,
        _absolute(tensorfs, "TensorFS root") if tensorfs else None,
    )


def input_media_root() -> Path:
    """Disposable downloaded media and prepared derivatives, shared with the granting host."""
    return Path(tempfile.gettempdir()) / "cozy"
