"""The machine kernel store across real processes; Sol's compile inputs where CuTe exists.

Every leg runs real processes on a real filesystem with real `flock`. The Sol legs need the
kernel-python artifact on the path (cutlass-dsl, tvm-ffi, sol-attn); the load leg needs a card
one of Sol's CuTe routes serves.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import attention_sol, jit_cache, kernel_cache

KEY_DOC = {"library": {"sol_attn_sources": "0" * 64}, "target": "sm_90a", "variant": {}}


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
        "TRITON_CACHE_DIR": str(own / "triton"),
    }
    assert root.stat().st_mode & 0o777 == 0o711 and own.stat().st_mode & 0o777 == 0o755
    store = kernel_cache.from_sealed(environment)
    assert store is not None and store.own == own and store.trusted is None
    assert jit_cache.machine(root) == environment
    installed = jit_cache.machine(root, installation="release-45299ff1")
    assert installed == {
        **environment,
        "PYTORCH_KERNEL_CACHE_PATH": str(own / "torch-kernels.release-45299ff1"),
    }


def _site() -> None:
    for module in ("sol_attn", "cutlass", "tvm_ffi", "torch"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"requires the kernel-python artifact on the path ({module})")


_COMPILE = """
    import hashlib, json, os, sys
    from cozy_runtime.internal import attention_sol
    tokens, capability = int(sys.argv[1]), (9, 0)
    shape = attention_sol.variant(tokens, capability)
    data = attention_sol.compile_variant("sm90", shape["blocks"], shape["full"])
    key = attention_sol.job(tokens, capability).key
    print(json.dumps({"sha": hashlib.sha256(data).hexdigest(), "key": key.digest}))
"""


def _object(code: str, tokens: int) -> dict[str, str]:
    env = {"CUTE_DSL_ARCH": "sm_90a", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(tokens)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


def test_sol_sm90_object_is_a_function_of_its_block_layout_and_needs_no_gpu() -> None:
    """GPU-less placeholders trace upstream's SM90 kernel; two lengths in one 64-token block
    that agree on fullness are one object and one key, and another block layout is not."""
    _site()
    first = _object(_COMPILE, 4095)
    assert _object(_COMPILE, 4033) == first  # the same 64 blocks, the last one partial
    full = _object(_COMPILE, 4096)
    assert full["sha"] != first["sha"] and full["key"] != first["key"]
    assert _object(_COMPILE, 4032)["key"] != full["key"]


def test_one_sol_object_per_card_off_sm90() -> None:
    _site()
    keys = {attention_sol.job(tokens, (12, 0)).key for tokens in (1, 4095, 109_104)}
    assert len(keys) == 1
    assert json.loads(next(iter(keys)).inputs)["variant"] == {}


_SERVE = """
    import json, sys, time, torch, cutlass.cute as cute
    from pathlib import Path
    from cozy_runtime.internal import attention_sol, kernel_cache, kernel_compile
    kernel_cache.configure(store := kernel_cache.Store(Path(sys.argv[2])))
    compiles = []
    original = cute.compile
    cute.compile = lambda *a, **k: compiles.append(1) or original(*a, **k)
    g = torch.Generator(device="cuda").manual_seed(7)
    q, k, v = (torch.randn(1, 4097, 4, 128, generator=g, device="cuda",
               dtype=torch.bfloat16) for _ in range(3))
    rows = []
    with attention_sol.observing(kernels=rows):
        while (loaded := attention_sol._ready(4097, q.device, 5)) is None:
            job = attention_sol._job_for(4097, torch.cuda.get_device_capability())
            while kernel_compile.status(store, job).state in ("compiling", "pending"):
                time.sleep(0.2)
        out = attention_sol._native(q, k, v, None, 0, loaded)
    torch.save(out.cpu(), sys.argv[1])
    print(json.dumps({"compiles": len(compiles), "sources": [r["source"] for r in rows]}))
"""


def test_a_second_process_loads_the_sol_object_the_first_compiled(tmp_path: Path) -> None:
    """Process A's first sparse step finds no object, submits its compile and runs dense; the
    builder publishes; A then loads it. Process B loads it without compiling, and both produce
    bitwise-identical attention."""
    _site()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    if tuple(torch.cuda.get_device_capability()) not in attention_sol.ROUTES:
        pytest.skip("requires a card one of Sol's CuTe routes serves")
    own = str(tmp_path / "u1")
    first = json.loads(_run(_SERVE, str(tmp_path / "out0.pt"), own).stdout.splitlines()[-1])
    second = json.loads(_run(_SERVE, str(tmp_path / "out1.pt"), own).stdout.splitlines()[-1])
    assert first["sources"][0] == "cold_compile" and first["sources"][-1] == "store"
    assert first["compiles"] == 0  # the builder process compiled, never this one
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


def test_a_triton_kernel_compiles_once_per_machine(tmp_path: Path) -> None:
    """Triton's cache is in the machine store's namespace, so a later process (another
    executor, a later boot) loads what an earlier one compiled and compiles nothing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env = jit_cache.machine(tmp_path / "var/lib/cozy/kernels")
    cache = Path(env["TRITON_CACHE_DIR"])
    code = """
        import json, sys, torch, triton, triton.language as tl

        @triton.jit
        def double(x, y, n, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(y + offsets, 2 * tl.load(x + offsets, mask=offsets < n), mask=offsets < n)

        x = torch.arange(1000, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        double[(8,)](x, y, 1000, BLOCK=128)
        print(json.dumps({"sum": float(y.sum())}))
    """
    script = tmp_path / "double.py"  # Triton reads a kernel's source from its file
    script.write_text(textwrap.dedent(code))

    def run() -> Any:
        done = subprocess.run(
            [sys.executable, str(script)], env={**os.environ, **env}, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr[-2000:]
        return json.loads(done.stdout.splitlines()[-1])

    first = run()
    entries = sorted(path.name for path in cache.iterdir())
    second = run()
    assert first == second == {"sum": 999000.0}
    assert entries and sorted(path.name for path in cache.iterdir()) == entries


def test_a_torch_nvrtc_kernel_compiles_once_per_installation(tmp_path: Path) -> None:
    """PyTorch's NVRTC cache (jiterator ops) lives in the machine store under the executor's
    installation, so a later executor of that environment reads what an earlier one built."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env = jit_cache.machine(tmp_path / "var/lib/cozy/kernels", installation="release-x")
    cache = Path(env["PYTORCH_KERNEL_CACHE_PATH"])
    code = (
        "import json, torch\n"
        "x = torch.linspace(0.5, 4.0, 64, device='cuda')\n"
        "print(json.dumps({'sum': round(float(torch.special.bessel_j0(x).sum()), 4)}))\n"
    )

    def run() -> Any:
        done = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        assert "disables kernel caching" not in done.stderr
        return json.loads(done.stdout.splitlines()[-1])

    first = run()
    entries = sorted(path.name for path in cache.iterdir())
    assert entries, "the first process compiled nothing into the cache"
    assert run() == first
    assert sorted(path.name for path in cache.iterdir()) == entries
