"""Child calls whose wire lies about what the attempt holds, as a package could send them past the Runtime's own checks
(`cozy_runtime.bench.child_wire`; the release gate's `child-saved-file` cell, cr-251/cr-256): a saved file's ref with a
digest of other bytes, a tree manifest claimed as saved, and a spool symlink to a host file. TensorD alone stands between
such a call and the bytes, and must refuse each (`child.asset_ungranted`)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from tensorfs.executor import ChildCall, SavedFile

from cozy_runtime.author import Outputs
from cozy_runtime.media import File
from cozy_runtime.author._calls import _current

TREE = "application/vnd.cozy.tree-manifest"
REFUSED = "child.asset_ungranted"
#: An export every environment holds: a call let through would run it, and fail on its empty request.
TARGET = ("cozy_runtime.bench", "kernel_ready")


def _sha(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def check(out: Outputs) -> tuple[dict[str, str], File]:
    """Each forged call's refusal code ("" had it been let through), and the file it saved, which the call returns (an
    unreturned saved file fails it); raises unless every code is `REFUSED`."""
    broker = _current.get()
    if broker is None:
        raise RuntimeError("this attempt holds no call broker")
    data = os.urandom(4096)
    blob = out.save_bytes(data, media_type="application/octet-stream")
    local = blob._local
    if local is None:
        raise RuntimeError("a saved file has no spool path")
    spool = Path(local).parent
    tree = b"a tree's manifest names other objects"
    (spool / "file-link").symlink_to("/etc/hostname")
    (spool / "file-tree").write_bytes(tree)
    hostname = _sha(Path("/etc/hostname").read_bytes())

    def code(saved: SavedFile) -> str:
        answer = broker.exchange(
            ChildCall(call_index=900, module=TARGET[0], export=TARGET[1], payload="{}", saved=(saved,))
        )
        return "" if answer.ok else answer.code

    found = {
        "lying_digest": code(SavedFile(asset_ref=blob.ref, digest=_sha(b"not these bytes"), media_type=blob.media_type)),
        "claimed_tree": code(SavedFile(asset_ref="attempt:x/file/tree", digest=_sha(tree), media_type=TREE)),
        "symlink": code(SavedFile(asset_ref="attempt:x/file/link", digest=hostname, media_type="text/plain")),
    }
    wrong = {case: got for case, got in found.items() if got != REFUSED}
    if wrong:
        raise RuntimeError(f"forged child calls not refused {REFUSED}: {wrong}")
    return found, blob
