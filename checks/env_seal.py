"""CPU contract proof for the executor's pre-CUDA environment seal (cr-059)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from cozy_runtime.internal import accel, child_env, executor, spawn
from cozy_runtime.internal.worker import child

ROOT = Path(__file__).resolve().parents[1]
ALLOCATOR = "PYTORCH_CUDA_ALLOC_CONF"


class _Cuda:
    def __init__(self, initialized: bool) -> None:
        self.value = initialized

    def is_initialized(self) -> bool:
        return self.value


class _Torch:
    def __init__(self, initialized: bool) -> None:
        self.cuda = _Cuda(initialized)


def main() -> None:
    imposed = {
        "CUDA_VISIBLE_DEVICES": "0",
        ALLOCATOR: "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
    }
    sealed = spawn.seal((), imposed)
    assert sealed[ALLOCATOR] == "expandable_segments:True"
    assert spawn.seal_digest(sealed) == spawn.seal_digest(dict(sealed))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        supervision = child.ExecutorSupervision(
            root=root / "run",
            python="/usr/bin/python3",
            base_env=(),
            cozy_home=root / "home",
        )
        try:
            jit_env = supervision._prepare_jit_cache()
            assert set(jit_env) == {name for name, _directory in child_env.JIT_CACHE_ENV}
            assert all(Path(value).is_absolute() for value in jit_env.values())
        finally:
            supervision.close()

    original = os.environ.get(ALLOCATOR)
    captured = dict(executor._SEALED)
    try:
        os.environ[ALLOCATOR] = sealed[ALLOCATOR]
        executor._SEALED.clear()
        executor._capture_seal()
        assert executor._env_document() == {"env_intact": True, "env_changed": []}

        os.environ[ALLOCATOR] = "max_split_size_mb:64"
        assert executor._env_document() == {
            "env_intact": False,
            "env_changed": [ALLOCATOR],
        }
    finally:
        executor._SEALED.clear()
        executor._SEALED.update(captured)
        if original is None:
            os.environ.pop(ALLOCATOR, None)
        else:
            os.environ[ALLOCATOR] = original

    assert accel.initialized(_Torch(False), "cuda") is False
    assert accel.initialized(_Torch(True), "cuda") is True

    source = (ROOT / "src/cozy_runtime/internal/executor.py").read_text()
    torch_import = source.index("import torch  # the ONE torch import")
    preinit_refusal = source.index("if accel.initialized(torch, self.device_kind)", torch_import)
    runtime_init = source.index("accel.initialize(torch, self.device_kind)", preinit_refusal)
    assert torch_import < preinit_refusal < runtime_init

    print("env seal: allocator imposed before Torch; preinitialized CUDA and late mutation refuse")


if __name__ == "__main__":
    main()
