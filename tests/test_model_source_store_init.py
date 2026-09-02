"""Source preparation initializes a truly empty store root (pull.py's rule).

Live incident (run 82, 2026-09-02): worker images bake /var/lib/cozy/store
as an EMPTY directory, and bare Store.open refused every fresh pod's first
source preparation with STORE_ERA. Three arms over the real tensorfs module:
empty → init; real store → open; foreign bytes → tensorfs's own typed
refusal, never an init beside them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorfs", reason="model-execution extra absent")

from cozy_runtime.internal import fill
from cozy_runtime.internal.worker.model_source_prepare import _open_or_init


def test_an_empty_root_initializes(tmp_path: Path) -> None:
    tensorfs = fill.tensorfs_module()
    root = tmp_path / "store"
    _open_or_init(tensorfs, root)
    for name in ("repos", "manifests", "blobs"):
        assert (root / name).is_dir()


def test_a_real_store_opens(tmp_path: Path) -> None:
    tensorfs = fill.tensorfs_module()
    root = tmp_path / "store"
    _open_or_init(tensorfs, root)
    _open_or_init(tensorfs, root)  # idempotent: opens, never re-inits


def test_foreign_bytes_still_refuse(tmp_path: Path) -> None:
    tensorfs = fill.tensorfs_module()
    root = tmp_path / "store"
    root.mkdir()
    (root / "stray.bin").write_bytes(b"not a store")
    with pytest.raises(Exception) as caught:
        _open_or_init(tensorfs, root)
    assert "STORE_ERA" in str(caught.value)
