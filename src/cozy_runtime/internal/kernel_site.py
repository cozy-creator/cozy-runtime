"""The worker image's Runtime-owned attention kernel site.

The image bakes kernels built against ITS torch/CUDA under `/opt/cozy/kernels`, described by
the `manifest.json` its kernel gate writes (tensorhub `worker-images/kernel_gate.py`):

    {"site": "/opt/cozy/kernels/site", "torch": "2.14.0+cu130", "cuda": "13.0",
     "python_abi": "cp312", "kernels": {...}}

`site` is the importable directory; the rest is the build its native code needs; other
fields are the image's own record. The executor appends the site AFTER the package
environment's own packages, so a package's pinned version always wins, and only when this
environment's torch release, CUDA build and Python ABI are the manifest's: a mismatch leaves
every site kernel typed-absent with the reason, never half-imported against the wrong libtorch.
"""

from __future__ import annotations

import json
import re
import site
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path("/opt/cozy/kernels")
MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class KernelSite:
    """What this process placed on its import path, or why it placed nothing."""

    root: str
    path: str = ""
    reason: str = ""
    torch: str = ""
    cuda: str = ""

    def document(self) -> dict[str, str]:
        return asdict(self)

    def describe(self) -> str:
        return f"kernel site {self.path}" if self.path else f"no kernel site: {self.reason}"


_ACTIVE: KernelSite | None = None


def torch_build() -> tuple[str, str]:
    """This environment's torch release and CUDA build, read from its files, never imported:
    the executor imports torch only after its environment seal is re-imposed."""
    try:
        installed = metadata.distribution("torch")
        text = Path(str(installed.locate_file("torch/version.py"))).read_text()
    except (metadata.PackageNotFoundError, OSError):
        return "", ""
    cuda = re.search(r"^cuda\b[^=\n]*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    return installed.version.split("+", 1)[0], cuda.group(1) if cuda else ""


def _place(root: Path) -> KernelSite:
    release, cuda = torch_build()
    refused = KernelSite(str(root), torch=release, cuda=cuda)
    try:
        manifest = json.loads((root / MANIFEST).read_text())
    except FileNotFoundError:
        return _why(refused, f"this image bakes no {root / MANIFEST}")
    except (OSError, ValueError) as exc:
        return _why(refused, f"{root / MANIFEST} is unreadable: {exc}")
    if not isinstance(manifest, dict):
        return _why(refused, f"{root / MANIFEST} is not an object")
    built = (str(manifest.get("torch", "")).split("+", 1)[0], str(manifest.get("cuda", "")))
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if not all(built):
        return _why(refused, f"{root / MANIFEST} names no torch/cuda build")
    abi = str(manifest.get("python_abi", ""))
    if built != (release, cuda) or abi != python:
        return _why(
            refused,
            f"built for torch {built[0]} CUDA {built[1]} {abi or 'unknown ABI'}; this "
            f"environment has torch {release or 'absent'} CUDA {cuda or 'none'} {python}",
        )
    path = root / str(manifest.get("site", "site"))
    if not path.is_dir():
        return _why(refused, f"{path} is not a directory")
    # addsitedir, not a bare append: the CuTe DSL wheels expose `cutlass` through a .pth.
    site.addsitedir(str(path))
    return KernelSite(str(root), str(path), "", release, cuda)


def _why(refused: KernelSite, reason: str) -> KernelSite:
    return KernelSite(refused.root, "", reason, refused.torch, refused.cuda)


def activate(root: Path = ROOT) -> KernelSite:
    """Place the site once per process, at executor entry, before any kernel is imported."""
    global _ACTIVE
    if _ACTIVE is None or _ACTIVE.root != str(root):
        _ACTIVE = _place(root)
    return _ACTIVE


def current() -> KernelSite:
    return _ACTIVE or KernelSite(str(ROOT), reason="not activated in this process")


def arches(name: str) -> list[str]:
    """The architectures the image's bake gate proved the site's build of kernel `name`
    carries (`kernel_gate.py` refuses an extension missing a declared one); empty when no site
    is placed or its manifest does not declare the kernel."""
    if not current().path:
        return []
    try:
        declared = json.loads((Path(current().root) / MANIFEST).read_text())["kernels"][name]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    found = declared.get("arches") if isinstance(declared, dict) else None
    return [str(arch) for arch in found] if isinstance(found, list) else []


def snapshot(name: str) -> Path:
    """A `kernels` Hub snapshot the image bakes beside the site, e.g. `flash-attn3`."""
    return Path(current().root) / name
