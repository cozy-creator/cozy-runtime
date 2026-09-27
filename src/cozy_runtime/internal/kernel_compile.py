"""Sol-Attn's SM90 forward: compiled once per key, with no GPU, and loaded everywhere else.

Upstream (NVlabs/Sana `sol_attn` 0.5.0) keeps compiled kernels in a per-process dict, and
`cute.compile` forces CuTe-DSL's own file cache off, so every executor process used to pay
the compile on its first sparse call. Its SM90 kernel specializes on the token count
(`make_kernel(tokens, kv_splits)`), never on heads, batch or device: every operand layout is
dynamic. So the object is exactly a function of `sol_key`'s inputs, and tracing it from host
placeholders that report CUDA device 0 through DLPack yields the same bytes as tracing it
beside live H100 tensors (measured, h3a-087). That makes the compile a CPU job that can run
beside a request's dense steps, in a niced child with no GPU visible, and publish into the
machine kernel store for every rank and every later executor.

`python -m cozy_runtime.internal.kernel_compile` is that child. The executor starts it with
the key's build lock already taken, handed down as `--lock-fd`, so a rank that needs the
kernel before it is done waits on the lock (on a live compile, never a clock) and then loads.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import json
import os
import sys
import tempfile
import time
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from cozy_runtime.internal import kernel_cache

KERNEL = "sol-attn.sm90"
OPTIONS = "--enable-tvm-ffi"
FUNCTION = "func"
_DLPACK_CUDA = 2
_SCALE = 128**-0.5


def target(capability: tuple[int, int]) -> str:
    """CuTe-DSL's own arch spelling (`env_manager.detect_gpu_arch`) unless the env names one."""
    named = os.environ.get("CUTE_DSL_ARCH", "")
    if named:
        return named
    major, minor = capability
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


@functools.cache
def sol_library() -> dict[str, Any]:
    """Every library input to the object: sources, DSL and toolkit, FFI ABI, Python tag."""
    cutlass = import_module("cutlass")
    tvm_ffi = import_module("tvm_ffi")
    toolkit = import_module("cutlass.base_dsl.version_info").CUDA_VERSION
    root = Path(str(import_module("sol_attn").__file__)).resolve().parent
    sources = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        data = path.read_bytes()
        sources.update(path.relative_to(root).as_posix().encode())
        sources.update(len(data).to_bytes(8, "little"))
        sources.update(data)
    return {
        "sol_attn": metadata.version("sol-attn"),
        "sol_attn_sources": sources.hexdigest(),
        "cutlass_dsl": str(cutlass.__version__),
        "cuda_toolkit": f"{toolkit.major}.{toolkit.minor}",
        "tvm_ffi": str(tvm_ffi.__version__),
        "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "cute_dsl_env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("CUTE_DSL_") and name != "CUTE_DSL_ARCH"
        },
    }


def sol_key(tokens: int, kv_splits: int, arch: str) -> kernel_cache.Key:
    if type(tokens) is not int or tokens < 1 or kv_splits not in (1, 2, 4):
        raise ValueError("Sol kernel key needs a positive token count and 1, 2 or 4 splits")
    return kernel_cache.Key.of(
        KERNEL,
        {
            "library": sol_library(),
            "target": arch,
            "options": OPTIONS,
            "tokens": tokens,
            "kv_splits": kv_splits,
        },
    )


class _Placeholder:
    """A host tensor that reports CUDA device 0 through DLPack. Tracing reads only dtype, rank,
    strides and memory space; no pointer is dereferenced, so no GPU is needed."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor
        self.ndim = tensor.ndim

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_DLPACK_CUDA, 0)

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        capsule = self.tensor.__dlpack__(*args, **kwargs)
        name_of = ctypes.pythonapi.PyCapsule_GetName
        name_of.restype, name_of.argtypes = ctypes.c_char_p, [ctypes.py_object]
        pointer_of = ctypes.pythonapi.PyCapsule_GetPointer
        pointer_of.restype = ctypes.c_void_p
        pointer_of.argtypes = [ctypes.py_object, ctypes.c_char_p]
        name = name_of(capsule)
        # DLManagedTensorVersioned: version(8) manager_ctx(8) deleter(8) flags(8), then the
        # DLTensor, whose device follows its data pointer. Unversioned: DLTensor first.
        offset = 40 if name == b"dltensor_versioned" else 8
        ctypes.c_int32.from_address(pointer_of(capsule, name) + offset).value = _DLPACK_CUDA
        return capsule


def _operands(kv_splits: int) -> list[tuple[tuple[int, ...], str]]:
    """Upstream `_sol_attn_cute`'s SM90 operand ranks and dtypes, at a size of two per mode."""
    rows: list[tuple[tuple[int, ...], str]] = [((2, 2, 2, 128), "bfloat16")] * 6
    rows += [((2, 2, 2), "float32")] * 2
    if kv_splits > 1:
        rows += [((2, 2, 2, 128), "bfloat16"), ((2, 2, 2), "float32")]
    return rows


def compile_sol(tokens: int, kv_splits: int) -> bytes:
    """Upstream `_compile_sm90`, traced from placeholders; the object for `sol_key`."""
    torch = import_module("torch")
    cute = import_module("cutlass.cute")
    cuda = import_module("cuda.bindings.driver")
    to_cute_tensor = import_module("sol_attn.common").to_cute_tensor
    make_kernel = import_module("sol_attn.sm90").make_kernel
    arguments = [
        to_cute_tensor(_Placeholder(torch.empty(shape, dtype=getattr(torch, dtype))))
        for shape, dtype in _operands(kv_splits)
    ]
    compiled = cute.compile(
        make_kernel(tokens, kv_splits),
        *arguments,
        _SCALE,
        0,
        stream=cuda.CUstream(0),
        options=OPTIONS,
    )
    with tempfile.TemporaryDirectory(prefix="cozy-kernel-") as scratch:
        path = Path(scratch) / "kernel.o"
        compiled.export_to_c(object_file_path=str(path), function_name=FUNCTION)
        return path.read_bytes()


class Loaded:
    """Upstream's call convention over a loaded object: `stream` is positional in TVM-FFI."""

    __slots__ = ("function",)

    def __init__(self, function: Any) -> None:
        self.function = function

    def __call__(self, *arguments: Any, stream: Any = None) -> Any:
        return self.function(*arguments, stream)


def load(data: bytes) -> Loaded:
    """Link verified bytes from an anonymous file: what was hashed is exactly what runs."""
    cute = import_module("cutlass.cute")
    for library in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
        if Path(library).exists():
            # As flash-attn-4's own disk cache does: a loaded object resolves the DSL
            # runtime's symbols globally.
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    descriptor = os.memfd_create("cozy-kernel", os.MFD_CLOEXEC)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        module = cute.runtime.load_module(f"/proc/self/fd/{descriptor}", enable_tvm_ffi=True)
        return Loaded(getattr(module, FUNCTION))
    finally:
        os.close(descriptor)


def producer(started: float) -> dict[str, Any]:
    return {
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "compile_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cozy_runtime.internal.kernel_compile")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--kv-splits", type=int, default=1)
    parser.add_argument("--target", required=True)
    parser.add_argument("--lock-fd", type=int, default=-1)
    options = parser.parse_args(argv)
    os.nice(19)
    if os.environ.get("CUTE_DSL_ARCH") != options.target:
        # The DSL reads its target once at import; the parent imposes it with the key.
        raise SystemExit("CUTE_DSL_ARCH must name the key's target before the DSL imports")
    store = kernel_cache.from_environment(os.environ)
    if store is None:
        raise SystemExit(f"{kernel_cache.OWN_ENV} names no kernel store")
    started = time.perf_counter()
    key = sol_key(options.tokens, options.kv_splits, options.target)
    if store.get(key) is None:
        store.put(key, compile_sol(options.tokens, options.kv_splits), producer(started))
    print(json.dumps({"key": key.digest, "ms": round((time.perf_counter() - started) * 1000)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
