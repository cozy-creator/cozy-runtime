"""The Runtime's numerics policy, the one owner of Torch's global numerics (CONTRACT §9), and
its cheap salt, independent of SDK and package inventories.

Callee implementation declarations own Python helpers, data, and installed native
build identities. This salt carries only the execution device and imposed numeric
policy; it must never inspect distributions, RECORD files, or native binaries.
"""

from __future__ import annotations

import functools
import hashlib
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType

from cozy_runtime import canonical_json
from cozy_runtime.author import _model
from cozy_runtime.internal.hostfacts import HostFacts


class NumericalEnvironmentRefusal(Exception):
    pass


_NUMERICAL_SETTINGS = frozenset(
    {
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_SCHEDULE",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "MKL_CBWR",
        "OPENBLAS_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "PYTHONHASHSEED",
    }
)


def fingerprint(
    device: HostFacts,
    *,
    threads: int,
    inherited: Mapping[str, str],
) -> bytes:
    """Bind device architecture and numerical policy; callee declarations bind libraries.

    The host driver is not a numerical fact: kernels come from the environment's own
    libraries, so a provider's driver update must not orphan interrupted derived work.
    """
    if device.backend == "cuda" and device.gpu_sm <= 0:
        raise NumericalEnvironmentRefusal("numerical device identity is unavailable")
    facts = {
        "device": {
            "backend": device.backend or "cpu",
            "architecture": device.gpu_sm,
        },
        "settings": {
            "threads": threads,
            "environment": {
                key: value for key, value in inherited.items() if key in _NUMERICAL_SETTINGS
            },
        },
    }
    return hashlib.sha256(
        b"cozy.runtime.numerical-environment/4\0" + canonical_json.encode(facts)
    ).digest()


#: torch `_native`'s DSLs that compile at an op's first call.
JIT_DSLS = ("triton", "cutedsl", "helion")


def apply(torch: ModuleType, kind: str) -> None:
    """The policy, set once per executor before CUDA init and its first op: torch's own
    defaults, pinned so no package moves them (float32 matmuls in full precision, no TF32;
    cuDNN convolutions with TF32, as torch has them); `deterministic=True` scopes' cuDNN mode;
    and on CUDA no JIT DSL op override, since a request never compiles (the native-op
    selection is kept). TF32 goes through the `allow_tf32` setters: in torch 2.15 setting
    `fp32_precision` makes the `cudnn.allow_tf32` getter raise, and stage memos and recorded
    loops read it."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    _model.DETERMINISTIC[0] = functools.partial(_DETERMINISM.held, torch)
    native = getattr(torch, "_native", None)
    if kind == "cuda" and native is not None:
        native.registry.deregister_op_overrides(disable_dsl_names=list(JIT_DSLS))


class _Determinism:
    """cuDNN deterministic, without benchmarking, while any `deterministic=True` scope of this
    process is open; the settings before the first come back after the last. Counted rather
    than saved per scope, because scopes on two threads (a remote scope beside a local one)
    may close in either order: a deterministic scope never runs with the flag cleared."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open = 0
        self._before = (False, False)

    @contextmanager
    def held(self, torch: ModuleType) -> Iterator[None]:
        cudnn = torch.backends.cudnn
        with self._lock:
            if not self._open:
                self._before = (bool(cudnn.deterministic), bool(cudnn.benchmark))
                cudnn.deterministic, cudnn.benchmark = True, False
            self._open += 1
        try:
            yield
        finally:
            with self._lock:
                self._open -= 1
                if not self._open:
                    cudnn.deterministic, cudnn.benchmark = self._before


_DETERMINISM = _Determinism()


def deterministic_open() -> bool:
    """Whether a `deterministic=True` scope of this process holds cuDNN's mode now."""
    return _DETERMINISM._open > 0
