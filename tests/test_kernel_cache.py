"""The machine kernel store across real processes; Sol's compile inputs where the site exists.

Every leg runs real processes on a real filesystem with real `flock`. The Sol legs need the
image kernel site (cutlass-dsl, tvm-ffi, sol-attn); the load leg needs an SM90 device.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cozy_runtime.internal import attention_sol, jit_cache, kernel_cache, kernel_compile

KEY_DOC = {"library": {"sol_attn": "0.5.0"}, "target": "sm_90a", "tokens": 73439}


def _run(
    code: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=True,
    )


def test_key_is_the_exact_document() -> None:
    a = kernel_cache.Key.of("sol-attn.sm90", {"tokens": 1, "target": "sm_90a"})
    b = kernel_cache.Key.of("sol-attn.sm90", {"target": "sm_90a", "tokens": 1})
    assert a == b and a.digest == b.digest
    assert kernel_cache.Key.of("sol-attn.sm90", {"tokens": 2, "target": "sm_90a"}) != a
    with pytest.raises(kernel_cache.KernelCacheRefusal):
        kernel_cache.Key.of("sol-attn.sm90", {"scale": 0.5})
    with pytest.raises(kernel_cache.KernelCacheRefusal):
        kernel_cache.Key.of("../escape", {})


def test_entry_outlives_its_writer_and_is_verified(tmp_path: Path) -> None:
    own = tmp_path / "u1"
    _run(
        f"""
        from pathlib import Path
        from cozy_runtime.internal import kernel_cache
        key = kernel_cache.Key.of("sol-attn.sm90", {KEY_DOC!r})
        store = kernel_cache.Store(Path({str(own)!r}))
        with store.building(key):
            store.put(key, b"object-bytes", {{"pid": 1}})
        """
    )
    key = kernel_cache.Key.of("sol-attn.sm90", KEY_DOC)
    store = kernel_cache.Store(own)
    assert store.get(key) == b"object-bytes"
    entry = json.loads((own / key.kernel / key.digest / "entry.json").read_text())
    assert entry["key"] == key.document() and entry["size"] == len(b"object-bytes")

    # A torn object is a miss, and the entry is removed so it cannot shadow the rebuild.
    (own / key.kernel / key.digest / "object").write_bytes(b"object-bytez")
    assert store.get(key) is None
    assert not (own / key.kernel / key.digest).exists()

    # An entry filed under this digest for another key document is a miss too.
    other = kernel_cache.Key.of("sol-attn.sm90", {**KEY_DOC, "tokens": 1})
    with store.building(other):
        store.put(other, b"other", {})
    (own / key.kernel / other.digest).rename(own / key.kernel / key.digest)
    assert store.get(key) is None


def test_trusted_namespace_is_read_and_never_repaired(tmp_path: Path) -> None:
    worker, own = tmp_path / "u0", tmp_path / "u64001"
    key = kernel_cache.Key.of("sol-attn.sm90", KEY_DOC)
    kernel_cache.Store(worker).put(key, b"built-by-worker", {})
    reader = kernel_cache.Store(own, worker)
    assert reader.get(key) == b"built-by-worker"
    assert not (own / key.kernel / key.digest).exists()
    (worker / key.kernel / key.digest / "object").write_bytes(b"tampered-bytes!")
    assert reader.get(key) is None
    assert (worker / key.kernel / key.digest / "object").exists()


def test_concurrent_builders_compile_once(tmp_path: Path) -> None:
    """Four processes want one key at once: one compiles, three wait on its lock and load."""
    own, log = tmp_path / "u1", tmp_path / "compiles.log"
    code = f"""
        import os, time
        from pathlib import Path
        from cozy_runtime.internal import kernel_cache
        key = kernel_cache.Key.of("sol-attn.sm90", {KEY_DOC!r})
        store = kernel_cache.Store(Path({str(own)!r}))
        data = store.get(key)
        if data is None:
            with store.building(key):
                data = store.get(key)
                if data is None:
                    with open({str(log)!r}, "a") as f:
                        f.write(f"{{os.getpid()}}\\n")
                    time.sleep(0.5)
                    data = b"x" * 4096
                    store.put(key, data, {{"pid": os.getpid()}})
        print(len(data))
    """
    processes = [
        subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], stdout=subprocess.PIPE)
        for _ in range(4)
    ]
    outputs = [p.communicate()[0].decode().strip() for p in processes]
    assert all(p.returncode == 0 for p in processes)
    assert outputs == ["4096"] * 4
    assert len(log.read_text().split()) == 1
    assert not list((own / "sol-attn.sm90").glob(".tmp-*"))


def test_a_claimed_lock_lives_exactly_as_long_as_the_child_holding_it(tmp_path: Path) -> None:
    own = tmp_path / "u1"
    key = kernel_cache.Key.of("sol-attn.sm90", KEY_DOC)
    store = kernel_cache.Store(own)
    lock = store.claim(key)
    assert lock is not None
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from pathlib import Path
                from cozy_runtime.internal import kernel_cache
                time.sleep(0.6)
                store = kernel_cache.Store(Path({str(own)!r}))
                key = kernel_cache.Key.of("sol-attn.sm90", {KEY_DOC!r})
                store.put(key, b"from-the-builder", {{}})
                """
            ),
        ],
        pass_fds=(lock,),
    )
    os.close(lock)
    assert store.claim(key) is None  # the child holds it now
    started = time.monotonic()
    with store.building(key):
        waited = time.monotonic() - started
        assert store.get(key) == b"from-the-builder"
    assert child.wait() == 0
    assert waited > 0.3


def test_machine_namespaces_under_one_uid(tmp_path: Path) -> None:
    root = tmp_path / "var/lib/cozy/kernels"
    environment = jit_cache.machine(root)
    own = kernel_cache.namespace(root, os.geteuid())
    assert environment == {
        kernel_cache.OWN_ENV: str(own),
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(own / "flash-attn4"),
    }
    assert root.stat().st_mode & 0o777 == 0o711 and own.stat().st_mode & 0o777 == 0o755
    store = kernel_cache.from_sealed(environment)
    assert store is not None and store.own == own and store.trusted is None
    assert jit_cache.machine(root) == environment


def _site() -> None:
    for module in ("sol_attn", "cutlass", "tvm_ffi", "torch"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"requires the image kernel site ({module})")


_COMPILE = """
    import hashlib, json, os, sys
    from cozy_runtime.internal import kernel_compile
    tokens = int(sys.argv[1])
    data = kernel_compile.compile_sol(tokens, 1)
    key = kernel_compile.sol_key(tokens, 1, os.environ["CUTE_DSL_ARCH"])
    print(json.dumps({"sha": hashlib.sha256(data).hexdigest(), "key": key.digest}))
"""

_LIVE = """
    import hashlib, json, os, sys, tempfile
    import torch, cutlass.cute as cute, cuda.bindings.driver as cuda
    from sol_attn.common import to_cute_tensor
    from sol_attn.sm90 import make_kernel
    tokens, heads = int(sys.argv[1]), 12
    blocks = (tokens + 63) // 64
    e = lambda *s, dt=torch.bfloat16: torch.empty(*s, device="cuda", dtype=dt)
    live = [e(1, tokens, heads, 128)] * 4 + [e(1, blocks, heads, 128)] * 2
    live += [e(1, blocks, heads, dt=torch.float32), e(1, tokens, heads, dt=torch.float32)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(make_kernel(tokens, 1), *[to_cute_tensor(t) for t in live],
                            128 ** -0.5, 0, stream=stream, options="--enable-tvm-ffi")
    with tempfile.TemporaryDirectory() as d:
        compiled.export_to_c(object_file_path=d + "/k.o", function_name="func")
        data = open(d + "/k.o", "rb").read()
    print(json.dumps({"sha": hashlib.sha256(data).hexdigest()}))
"""


def _object(code: str, tokens: int, *, gpu: bool) -> dict[str, str]:
    env = {"CUTE_DSL_ARCH": "sm_90a"}
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(tokens)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


def test_sol_sm90_object_is_a_function_of_its_key_and_needs_no_gpu() -> None:
    """GPU-less placeholders trace upstream's SM90 kernel to the bytes live tensors give."""
    _site()
    torch = pytest.importorskip("torch")
    first = _object(_COMPILE, 4096, gpu=False)
    assert _object(_COMPILE, 4096, gpu=False) == first
    other = _object(_COMPILE, 4032, gpu=False)
    assert other["sha"] != first["sha"] and other["key"] != first["key"]
    if torch.cuda.is_available():
        assert _object(_LIVE, 4096, gpu=True)["sha"] == first["sha"]


def test_second_sm90_process_loads_the_kernel_without_compiling(tmp_path: Path) -> None:
    """Process A resolves (compiles and publishes); process B resolves by loading only, and
    both produce bitwise-identical attention."""
    _site()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires an SM90 device to execute the loaded object")
    code = """
        import json, sys, torch, cutlass.cute as cute
        from pathlib import Path
        from cozy_runtime.internal import attention_sol, kernel_cache
        kernel_cache.configure(kernel_cache.Store(Path(sys.argv[2])))
        compiles = []
        original = cute.compile
        cute.compile = lambda *a, **k: compiles.append(1) or original(*a, **k)
        g = torch.Generator(device="cuda").manual_seed(7)
        q, k, v = (torch.randn(1, 4097, 4, 128, generator=g, device="cuda",
                   dtype=torch.bfloat16) for _ in range(3))
        rows = []
        with attention_sol.observing(kernels=rows):
            out = attention_sol._native(q, k, v, None, 0)
        torch.save(out.cpu(), sys.argv[1])
        print(json.dumps({"compiles": len(compiles), "sources": [r["source"] for r in rows]}))
    """
    own = str(tmp_path / "u1")
    first = json.loads(_run(code, str(tmp_path / "out0.pt"), own).stdout.splitlines()[-1])
    second = json.loads(_run(code, str(tmp_path / "out1.pt"), own).stdout.splitlines()[-1])
    assert first["sources"] == ["compiled"]
    assert second == {"compiles": 0, "sources": ["store"]}
    assert torch.equal(torch.load(tmp_path / "out0.pt"), torch.load(tmp_path / "out1.pt"))


def test_fa4_second_process_loads_from_its_persistent_cache(tmp_path: Path) -> None:
    """flash-attn-4's own disk cache, at the directory the machine store gives it."""
    torch = pytest.importorskip("torch")
    if importlib.util.find_spec("flash_attn") is None or not torch.cuda.is_available():
        pytest.skip("requires flash-attn-4 and a CUDA device")
    env = {
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(tmp_path / "flash-attn4"),
    }
    code = """
        import json, sys, torch, cutlass.cute as cute
        compiles = []
        original = cute.compile
        cute.compile = lambda *a, **k: compiles.append(1) or original(*a, **k)
        from flash_attn.cute import flash_attn_func
        g = torch.Generator(device="cuda").manual_seed(3)
        q, k, v = (torch.randn(1, 1024, 2, 128, generator=g, device="cuda",
                   dtype=torch.bfloat16) for _ in range(3))
        out = flash_attn_func(q, k, v)
        out = out[0] if isinstance(out, tuple) else out
        torch.save(out.cpu(), sys.argv[1])
        print(json.dumps({"compiles": len(compiles)}))
    """
    first = json.loads(_run(code, str(tmp_path / "fa0.pt"), env=env).stdout.splitlines()[-1])
    second = json.loads(_run(code, str(tmp_path / "fa1.pt"), env=env).stdout.splitlines()[-1])
    assert first["compiles"] >= 1 and second["compiles"] == 0
    assert torch.equal(torch.load(tmp_path / "fa0.pt"), torch.load(tmp_path / "fa1.pt"))


def test_background_builder_publishes_under_the_inherited_lock(tmp_path: Path) -> None:
    """The exact child the executor starts at a dense step: GPU-less, niced, holding the
    key's lock it was handed; a rank that needs the kernel waits on that lock, then loads
    the bytes an in-process compile would have produced."""
    _site()
    own = tmp_path / "u1"
    store = kernel_cache.Store(own)
    lock = store.claim(kernel_compile.sol_key(4096, 1, "sm_90a"))
    assert lock is not None
    argv, environment = attention_sol.builder_command(store, 4096, 1, "sm_90a", lock)
    child = subprocess.Popen(
        argv,
        env=environment,
        pass_fds=(lock,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.close(lock)
    key = kernel_compile.sol_key(4096, 1, "sm_90a")
    with store.building(key):
        data = store.get(key)
    _, err = child.communicate()
    assert child.returncode == 0, err.decode()[-2000:]
    assert data is not None
    entry = json.loads((own / key.kernel / key.digest / "entry.json").read_text())
    assert entry["producer"]["compile_ms"] > 0
    assert hashlib.sha256(data).hexdigest() == _object(_COMPILE, 4096, gpu=False)["sha"]
