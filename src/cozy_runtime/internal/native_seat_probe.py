"""Measure the Python/base half of a native qualification seat before package import."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import sys

from cozy_runtime.internal import canonical


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def facts() -> dict[str, object]:
    distributions = sorted(
        (
            {
                "distribution": _name(item.metadata["Name"]),
                "version": item.version,
            }
            for item in importlib.metadata.distributions()
            if item.metadata["Name"]
        ),
        key=lambda item: (item["distribution"], item["version"]),
    )
    torch_version = ""
    torch_cuda = ""
    cuda_available = False
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda = str(torch.version.cuda or "")
        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, AttributeError):
        pass
    libc_name, libc_version = platform.libc_ver()
    return {
        "cuda_available": cuda_available,
        "distributions": distributions,
        "implementation": platform.python_implementation().lower(),
        "libc_name": libc_name.lower(),
        "libc_version": libc_version,
        "machine": platform.machine().lower(),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "system": platform.system().lower(),
        "torch_cuda": torch_cuda,
        "torch_version": torch_version,
    }


def main() -> int:
    sys.stdout.buffer.write(canonical.write(facts()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
