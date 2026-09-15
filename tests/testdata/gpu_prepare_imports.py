"""Actual Runtime/Diffusers/TorchAO import ordering on an allocated CUDA worker."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    import torch

    from cozy_runtime.internal.executor import _CONSTRUCTION_MODULES, Executor

    mode, directory = sys.argv[1:]
    assert torch.cuda.is_available() and not torch.cuda.is_initialized()
    if mode == "preload":
        # This is the actual preconstruct import list, before any package import
        # or admitted device boundary. No replacement kernel/allocator is used.
        for module in _CONSTRUCTION_MODULES:
            importlib.import_module(module)
        assert not torch.cuda.is_initialized(), "CPU preloading opened CUDA"
        print(json.dumps({"initialized": False, "preloaded": list(_CONSTRUCTION_MODULES)}))
        return
    assert mode == "prepare"
    executor = Executor(None, Path(directory))  # type: ignore[arg-type]
    # Exercise the real prepare/import boundary, then stop at the ordinary
    # missing-package-identity check before loading any model or weight bytes.
    try:
        reply = executor._prepare_first(
            {
                "devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "authorized_device_limit_bytes": 1 << 30,
                "binding": {"model_class": "Model", "model_parameter_name": "model"},
                "budgets": {},
            }
        )
    except RuntimeError as exc:
        assert str(exc) == "executor requires the canonical installed release identity"
    else:
        raise AssertionError(f"prepare did not reach the post-import identity check: {reply}")
    assert executor._device_ready and torch.cuda.is_initialized()
    assert "diffusers.quantizers.torchao.torchao_quantizer" in sys.modules
    assert "torchao.prototype.mx_formats.kernels" in sys.modules
    print(json.dumps({"initialized_by_runtime": True, "optional_quantizer_imported": True}))


if __name__ == "__main__":
    main()
