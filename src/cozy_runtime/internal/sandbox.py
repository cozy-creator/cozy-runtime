"""The derivation sandbox: capability fences that hold because they are IN the process.

v1 had no sandbox at all (quarry: `python-gen-worker` derives with a live network, a live
filesystem and a live CUDA context; its fixtures merely CONVENTIONALLY carry no weights,
and nothing checks). Its one structural protection — a spawn pool — exists for parallelism.
That is the gap this module closes.

Harness isolation is LOAD-BEARING (§1.1), so the fences are audit hooks, not politeness:
`sys.addaudithook` cannot be removed once installed, which is why derivation runs in a
child process — the fences live exactly as long as the derivation and cannot leak into the
caller. `scripts/derive-corpus.py` proves each one live by attempting the capability.

Not fenced here and deliberately so: CUDA. `CUDA_VISIBLE_DEVICES=""` in the child's
environment is the fence, and it is stronger than a hook — torch itself then answers
`No CUDA GPUs are available` to `.to("cuda")`, and `torch.cuda.is_initialized()` staying
False is the assertion the harness makes afterwards.
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Sequence

#: Extensions whose bytes ARE the weights. The derivation reads configs and code; a
#: checkpoint byte reaching a factory means the derive/serve fence has already failed.
WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".msgpack",
    ".h5",
    ".pkl",
    ".npz",
    ".cozy",
)

#: Audit events that are network egress under any spelling. ONE definition, shared by every
#: process in this runtime that claims to have no outbound network (cr-042).
NETWORK_EVENTS: tuple[str, ...] = (
    "socket.connect",
    "socket.getaddrinfo",
    "socket.sendto",
    "urllib.Request",
    "http.client.connect",
    "ftplib.connect",
)

#: Audit events that spawn something the fences would not cover.
SPAWN_EVENTS: tuple[str, ...] = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.fork",
    "pty.spawn",
)

FENCES: tuple[str, ...] = (
    "no_network",
    "no_weight_bytes",
    "no_filesystem_writes",
    "no_subprocess",
    "no_cuda",
    "bounded_memory",
    "bounded_cpu",
)


class FenceViolation(Exception):
    """A derivation attempted a capability the harness does not grant.

    Carries the fence name so a corpus case can assert WHICH fence caught it — a test that
    only asserts "something refused" passes for the wrong reason (v1's `retryable` lesson).
    """

    def __init__(self, fence: str, detail: str) -> None:
        super().__init__(f"{fence}: {detail}")
        self.fence = fence
        self.detail = detail


def unix_seam(event: str, args: tuple[object, ...]) -> bool:
    """True when a `NETWORK_EVENTS` hit is an AF_UNIX connect: a local seam, not egress."""
    if event != "socket.connect" or not args:
        return False
    return getattr(args[0], "family", None) == socket.AF_UNIX


def refuse_network(why: str, *, allow_unix_seam: bool = False) -> None:
    """Refuse every `NETWORK_EVENTS` spelling in THIS process. IRREVERSIBLE.

    WHICH ADVERSARY (law 15). This is a bound on Python-level dialling: `socket`, `urllib`,
    `http.client`, `requests`, and everything built on them. It stops a BUGGY or careless
    package — a dependency that phones home, a `from_pretrained` that falls through to the
    hub — which is the same-tenant-code rung, the rung that exists today.

    It is NOT a bound on hostile code. `ctypes` or a C extension reaches `connect(2)` below
    CPython, and a subprocess inherits no audit hook at all. The third-party-code rung needs
    a kernel bound (`CLONE_NEWNET`, seccomp) that a process cannot install for itself, and
    the runtime does not have one. `scripts/lifecycle-live.py egress` observes BOTH halves
    on the real serving executor, so this paragraph cannot quietly become a bigger claim.
    """

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event in NETWORK_EVENTS and not (allow_unix_seam and unix_seam(event, args)):
            raise FenceViolation("no_network", f"{event}{args[1:2] or args[:1]} — {why}")

    sys.addaudithook(hook)


def install(*, allow_write_prefixes: Sequence[str] = ()) -> None:
    """Install the audit fences. IRREVERSIBLE — call it in the child, never in a caller."""
    writable = tuple(allow_write_prefixes)
    refuse_network(
        "a derivation reads configs and code, never the network. Kernel binaries are pinned "
        "image inputs; a boot-time kernel fetch is exactly what this refuses (cr-004)"
    )

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event in SPAWN_EVENTS:
            raise FenceViolation(
                "no_subprocess",
                f"{event} — a child would inherit none of these fences, so spawning one is "
                "the fence's own bypass",
            )
        if event == "open":
            path, mode = str(args[0]), str(args[1] or "r")
            if any(c in mode for c in "wxa+"):
                if not any(path.startswith(p) for p in writable):
                    raise FenceViolation(
                        "no_filesystem_writes",
                        f"opening {path!r} for {mode!r} — a derivation's only output is "
                        "contract bytes, validated as data",
                    )
            elif path.endswith(WEIGHT_SUFFIXES):
                raise FenceViolation(
                    "no_weight_bytes",
                    f"reading {path!r} — the derivation constructs under meta parameters and "
                    "never sees a weight byte; a factory that needs them is not a factory",
                )

    sys.addaudithook(hook)


def bound(*, address_bytes: int, cpu_seconds: int) -> None:
    """Bound the child. A construction that tries to allocate real weights dies here."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (address_bytes, address_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def child_environment(base: dict[str, str]) -> dict[str, str]:
    """The environment a derivation child gets: no accelerator, no hub credentials.

    Class C of tracker-v2/spawn-allowlists.md (#616.d): the class-B result with
    credential prefixes stripped, then the four offline impositions. Keep equal to
    that row.
    """
    env = {
        k: v
        for k, v in base.items()
        if not k.startswith(("HF_", "HUGGINGFACE_", "TENSORHUB_", "CIVITAI_", "COMFY_"))
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env
