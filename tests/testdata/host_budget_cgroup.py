"""Real cgroup2/UID regression; run in a private, disposable 256MiB container.

Requires root and a writable private cgroup2 mount with this process as its only
member. It never runs a GPU operation or accepts a shared host's root cgroup.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cozy_runtime.internal import proctree


def main() -> None:
    root = Path("/sys/fs/cgroup")
    assert os.geteuid() == 0
    assert Path("/proc/self/cgroup").read_text().strip() == "0::/"
    assert (root / "memory.max").read_text().strip() == "268435456"
    assert not os.statvfs(root).f_flag & os.ST_RDONLY
    assert set((root / "cgroup.procs").read_text().split()) == {str(os.getpid())}
    parent = root / "host-budget-supervisor"
    parent.mkdir(mode=0o711)
    parent.chmod(0o711)
    (parent / "cgroup.procs").write_text("0\n")
    (root / "cgroup.subtree_control").write_text("+memory\n")
    (parent / "memory.max").write_text(str(64 << 20))
    previous_umask = os.umask(0o077)
    try:
        scope = proctree.create_executor_cgroup("a" * 16)
    finally:
        os.umask(previous_umask)
    path = root / scope.relative_path.lstrip("/")
    assert path.stat().st_mode & 0o777 == 0o711
    child = """
import json,os,sys
from pathlib import Path
from cozy_runtime.internal import proctree
p=Path(sys.argv[1]); s=proctree.CgroupScope(relative_path=sys.argv[2],inode=p.stat().st_ino)
proctree.join_executor_cgroup(s)
os.setgroups([]);os.setgid(65533);os.setuid(65533)
denied={}
for name in ('cgroup.procs','cgroup.freeze','cgroup.kill'):
 try:
  fd=os.open(p/name,os.O_WRONLY);os.close(fd);denied[name]=False
 except PermissionError:denied[name]=True
try: list(p.iterdir());listing_denied=False
except PermissionError:listing_denied=True
print(json.dumps({'uid':os.getuid(),'membership':Path('/proc/self/cgroup').read_text(),
 'available':proctree.available_host_bytes(),'ancestor_limit':int((p.parent/'memory.max').read_text()),
 'control_writes_denied':denied,'listing_denied':listing_denied}))
"""

    def observe() -> dict[str, Any]:
        return json.loads(  # type: ignore[no-any-return]
            subprocess.check_output(
                [sys.executable, "-c", child, str(path), scope.relative_path], text=True
            )
        )

    fixed = observe()
    assert 0 < fixed["available"] <= fixed["ancestor_limit"] == 64 << 20
    assert all(fixed["control_writes_denied"].values()) and fixed["listing_denied"]
    path.chmod(0o700)
    denied = observe()
    assert denied["available"] == -1
    path.chmod(0o711)
    recovered = observe()
    assert 0 < recovered["available"] <= 64 << 20
    path.rmdir()
    print(json.dumps({"fixed": fixed, "unreadable_refuses": denied, "recovered": recovered}))


if __name__ == "__main__":
    main()
