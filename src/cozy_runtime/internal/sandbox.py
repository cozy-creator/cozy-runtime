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

#: Audit events that are network egress under any spelling.
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


def install(*, allow_write_prefixes: Sequence[str] = ()) -> None:
    """Install the audit fences. IRREVERSIBLE — call it in the child, never in a caller."""
    writable = tuple(allow_write_prefixes)

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event in NETWORK_EVENTS:
            raise FenceViolation(
                "no_network",
                f"{event}{args[1:2] or args[:1]} — a derivation reads configs and code, never "
                "the network. Kernel binaries are pinned image inputs; a boot-time kernel "
                "fetch is exactly what this refuses (cr-004)",
            )
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
    """The environment a derivation child gets: no accelerator, no hub credentials."""
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
