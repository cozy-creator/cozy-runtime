# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.30"]
# ///
"""The machine's host memory as the kernel counts it, read-only: MemTotal and MemAvailable, and memory.events,
memory.peak, memory.max and memory.current of this process's cgroup and every ancestor this container can see (cgroup
v2). The memory matrix runs it around every cell for the OOM killer's count (S8's diag/hostmem.py)."""
from pathlib import Path


def read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        return f"<{error.strerror}>"


def main() -> dict[str, str]:
    out = {"meminfo": "\n".join(line for line in read(Path("/proc/meminfo")).splitlines()
                                if line.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal")),
           "self_cgroup": read(Path("/proc/self/cgroup"))}
    rel = out["self_cgroup"].split("::", 1)[-1].strip() if "::" in out["self_cgroup"] else "/"
    node = Path("/sys/fs/cgroup") / rel.lstrip("/")
    while True:
        for name in ("memory.events", "memory.peak", "memory.max", "memory.current"):
            out[f"{node}/{name}"] = read(node / name)
        if node == Path("/sys/fs/cgroup") or node == node.parent:
            break
        node = node.parent
    return out
