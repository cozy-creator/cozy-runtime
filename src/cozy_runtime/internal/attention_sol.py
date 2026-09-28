"""Upstream BF16 Sol-Attn with explicit document and conditioning semantics.

No sparse kernel is implemented here. Sparse block selection is approximate;
protected query rows are recomputed with the selected dense reference. The
model supplies its live layout, step policy and dense module paths explicitly.

The kernel compiles at most once per key per machine (h3a-087). Upstream's SM90 kernel is
specialized on the live token count, which only the request knows, so the model's intake
statement (`expect`), or else the first dense step that sees a count, starts a niced, GPU-less
builder for it and conditioning and the dense steps cover the compile; the first sparse call
then loads the object from the machine kernel store, or waits on the builder's lock, never
switching kernels. Upstream's preprocess autotuning is pinned to
one configuration, because its candidates differ bitwise and a timing race must not choose
the output.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from inspect import Parameter, Signature, signature
from types import MethodType
from typing import Any

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
from cozy_runtime.author._observations import Scalar
from cozy_runtime.internal import config, kernel_cache, kernel_compile
from cozy_runtime.internal.attention_ulysses import Local

# Tensors, devices, modules and the dense reference are `Any`: torch is not in the check venv.


@dataclass(frozen=True, slots=True)
class Site:
    component: str
    path: str


_SITE: ContextVar[Site | None] = ContextVar("cozy_sol_site", default=None)
_DENSE: Callable[..., Any] | None = None
_DENSE_IDENTITY = ""
_COUNTS: ContextVar[dict[str, int] | None] = ContextVar("cozy_sol_counts", default=None)
_RANK_COUNTS: ContextVar[dict[int, dict[str, int]] | None] = ContextVar(
    "cozy_sol_rank_counts", default=None
)
_KERNEL_ROWS: ContextVar[list[dict[str, Scalar]] | None] = ContextVar(
    "cozy_sol_kernel_rows", default=None
)


@contextmanager
def observing(
    *,
    ranks: dict[int, dict[str, int]] | None = None,
    kernels: list[dict[str, Scalar]] | None = None,
) -> Iterator[dict[str, int]]:
    """Count successful calls in this request, restoring any outer observation. `kernels`
    receives one row per kernel this request started building or resolved."""
    counts = dict.fromkeys(("sparse", "dense_step", "dense_path", "dense_prefix"), 0)
    token = _COUNTS.set(counts)
    rank_token = _RANK_COUNTS.set(ranks)
    kernel_token = _KERNEL_ROWS.set(kernels)
    try:
        yield counts
    finally:
        _COUNTS.reset(token)
        _RANK_COUNTS.reset(rank_token)
        _KERNEL_ROWS.reset(kernel_token)


def _kernel_row(row: dict[str, Scalar]) -> None:
    rows = _KERNEL_ROWS.get()
    if rows is not None:
        rows.append(row)


def _count(kind: str) -> None:
    counts = _COUNTS.get()
    if counts is not None:
        counts[kind] += 1


def record_rank(rank: int, counts: dict[str, int]) -> None:
    """Retain counts acknowledged by a follower on the existing mirrored-call reply."""
    ranks = _RANK_COUNTS.get()
    if ranks is None:
        return
    if (
        type(rank) is not int
        or rank < 1
        or set(counts) != {"sparse", "dense_step", "dense_path", "dense_prefix"}
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError("invalid follower Sol attention counts")
    total = ranks.setdefault(rank, dict.fromkeys(counts, 0))
    for name, value in counts.items():
        total[name] += value


def bind_dense(call: Callable[..., Any], identity: str) -> None:
    """Install the already-resolved dense reference once before construction warmup."""
    if not callable(call) or not identity:
        raise ValueError("Sol requires a resolved dense attention reference and its identity")
    global _DENSE, _DENSE_IDENTITY
    if _DENSE is not None and identity != _DENSE_IDENTITY:
        raise RuntimeError("Sol dense reference is already bound to a different implementation")
    _DENSE, _DENSE_IDENTITY = call, identity


def dense_identity() -> str:
    return _DENSE_IDENTITY


@dataclass(frozen=True, slots=True)
class _InstalledSite:
    site: Site
    forward: Callable[..., Any]
    original: Callable[..., Any]
    bind_original: bool
    had_instance_forward: bool
    prior_instance_forward: Any


def install_site(module: Any, component: str, path: str) -> None:
    """Scope the actual forward, after pre-hooks and before post-hooks.

    Paired hooks cannot identify an invocation whose pre-hook never ran: an
    always-call post-hook can consume an outer recursive call's token. A local
    try/finally around the real forward has exactly one token per entered call.
    """
    site = Site(component, path)
    existing = getattr(module, "_cozy_sol_site", None)
    if existing is not None:
        if existing.site != site:
            raise ValueError("attention module was already assigned another Sol site")
        if module.forward is not existing.forward:
            raise ValueError("attention forward was replaced after installing its Sol site")
        return
    original = module.forward
    bound = isinstance(original, MethodType) and original.__self__ is module
    call = original.__func__ if bound else original
    had_forward = "forward" in vars(module)
    prior = vars(module).get("forward")
    original_signature = signature(original)
    self_name = "_cozy_module"
    while self_name in original_signature.parameters:
        self_name += "_"

    @wraps(call, updated=())
    def scoped_forward(self: Any, /, *args: Any, **kwargs: Any) -> Any:
        # Read the copied module's record rather than closing over a bound method
        # on the original module. deepcopy must execute the copied weights.
        installed = self._cozy_sol_site
        token = _SITE.set(installed.site)
        try:
            if installed.bind_original:
                return installed.original(self, *args, **kwargs)
            return installed.original(*args, **kwargs)
        finally:
            _SITE.reset(token)

    if not bound:
        # A callable object's metadata must not retain its original mutable state.
        # Its explicit signature below still preserves the public call contract.
        del scoped_forward.__wrapped__
    scoped_forward.__dict__["__signature__"] = Signature(
        [Parameter(self_name, Parameter.POSITIONAL_ONLY), *original_signature.parameters.values()],
        return_annotation=original_signature.return_annotation,
    )
    wrapper = MethodType(scoped_forward, module)
    module.forward = wrapper
    try:
        module._cozy_sol_site = _InstalledSite(site, wrapper, call, bound, had_forward, prior)
    except BaseException:
        if had_forward:
            module.forward = prior
        else:
            del module.forward
        raise


def remove_site(module: Any, component: str, path: str) -> None:
    """Restore the previous forward only while our exact wrapper still owns it."""
    existing = getattr(module, "_cozy_sol_site", None)
    if existing is None:
        return
    if existing.site != Site(component, path):
        raise ValueError("attention module is assigned another Sol site")
    if module.forward is not existing.forward:
        raise ValueError("attention forward was replaced after installing its Sol site")
    if existing.had_instance_forward:
        module.forward = existing.prior_instance_forward
    else:
        del module.forward
    del module._cozy_sol_site


def _dense(query: Any, key: Any, value: Any, scale: float | None) -> Any:
    if _DENSE is None:
        raise RuntimeError("Sol dense reference was not resolved before execution")
    output = _DENSE(query=query, key=key, value=value, scale=scale)
    _validate_result(output, query)
    return output


def _validate_result(output: Any, query: Any) -> None:
    import torch

    if (
        not isinstance(output, torch.Tensor)
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("attention backend returned incompatible output")


#: Loaded SM90 kernels by (tokens, kv_splits). One object serves every head count and device:
#: neither is a compile input (h3a-087).
_KERNELS: dict[tuple[int, int], kernel_compile.Loaded] = {}
_KEYS: dict[tuple[int, int], str] = {}
#: Counts this process already considered for a background build, and the builders it started.
_CONSIDERED: set[tuple[int, int]] = set()
_BUILDERS: dict[tuple[int, int], _Builder] = {}
_PINNED: list[bool] = []
#: The builder is `kernel_compile.main` under this process's own import path and bytecode
#: cache (`-I` drops the PYTHON* variables that carried the latter).
_BUILDER_BOOT = (
    "import json, sys; boot = json.loads(sys.argv.pop(1)); sys.path[:] = boot['path']; "
    "sys.pycache_prefix = boot['pycache_prefix']; "
    "from cozy_runtime.internal.kernel_compile import main; raise SystemExit(main(sys.argv[1:]))"
)


@dataclass(slots=True)
class _Builder:
    key: str
    started: float
    step: int
    done: threading.Event
    exit_status: int | None = None
    ms: float = 0.0
    detail: str = ""

    def document(self) -> dict[str, Scalar]:
        """The observation row's builder fields; `builder_exit` is None while it runs."""
        return {
            "builder_step": self.step,
            "builder_exit": self.exit_status,
            "builder_ms": self.ms,
            **({"detail": self.detail} if self.detail else {}),
        }


def _sm90(device: Any) -> bool:
    return device.type == "cuda" and bool(
        import_module("torch").cuda.get_device_capability(device) == (9, 0)
    )


def _key(tokens: int, kv_splits: int, device: Any) -> kernel_cache.Key:
    capability = tuple(import_module("torch").cuda.get_device_capability(device))
    return kernel_compile.sol_key(tokens, kv_splits, kernel_compile.target(capability))


def _prefetch(tokens: int, kv_splits: int, device: Any, step: int) -> None:
    """At the first dense step that shows this count: load it, or start its builder."""
    wanted = (tokens, kv_splits)
    if wanted in _CONSIDERED or wanted in _KERNELS or not _sm90(device):
        return
    _CONSIDERED.add(wanted)
    store = kernel_cache.configured()
    if store is None:
        return
    started = time.perf_counter()
    try:
        key = _key(tokens, kv_splits, device)
        data = store.get(key)
        if data is not None:
            _KERNELS[wanted], _KEYS[wanted] = kernel_compile.load(data), key.digest[:16]
            _kernel_row(_row(wanted, key.digest[:16], "store", started, step=step))
            return
        lock = store.claim(key)
        if lock is None:
            # Another rank or executor on this machine is building it.
            _kernel_row(_row(wanted, key.digest[:16], "peer_building", started, step=step))
            return
        try:
            _BUILDERS[wanted] = _spawn(store, key, tokens, kv_splits, device, lock, step)
        finally:
            os.close(lock)
        _kernel_row(_row(wanted, key.digest[:16], "background_started", started, step=step))
    except Exception as exc:
        # Nothing needs the kernel yet. The sparse call resolves it again and fails THERE,
        # precisely; this row keeps the first failure from being invisible.
        _kernel_row(
            {
                "source": "prefetch_failed",
                "tokens": tokens,
                "step": step,
                "detail": f"{type(exc).__name__}: {exc}"[:300],
            }
        )


def expect(tokens: int, device: Any) -> None:
    """A model stated its live length at request intake: start that count's kernel now, a
    whole text conditioning ahead of the first dense step. Its row carries step -1."""
    _prefetch(tokens, 1, device, -1)


def builder_command(
    store: kernel_cache.Store, tokens: int, kv_splits: int, arch: str, lock: int
) -> tuple[list[str], dict[str, str]]:
    """The builder's argv and environment: this interpreter and import path, this store, no
    GPU, the key's target imposed before the DSL imports, and the inherited build lock."""
    argv = [
        sys.executable,
        "-I",
        "-c",
        _BUILDER_BOOT,
        json.dumps({"path": sys.path, "pycache_prefix": sys.pycache_prefix}),
        "--tokens",
        str(tokens),
        "--kv-splits",
        str(kv_splits),
        "--target",
        arch,
        "--lock-fd",
        str(lock),
        "--store",
        str(store.own),
        *(["--trusted", str(store.trusted)] if store.trusted is not None else []),
    ]
    return argv, {
        **config.inherited_environment(),
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": arch,
    }


def _spawn(
    store: kernel_cache.Store,
    key: kernel_cache.Key,
    tokens: int,
    kv_splits: int,
    device: Any,
    lock: int,
    step: int,
) -> _Builder:
    capability = tuple(import_module("torch").cuda.get_device_capability(device))
    argv, environment = builder_command(
        store, tokens, kv_splits, kernel_compile.target(capability), lock
    )
    process = subprocess.Popen(
        argv,
        env=environment,
        pass_fds=(lock,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    builder = _Builder(key.digest[:16], time.perf_counter(), step, threading.Event())

    def reap() -> None:
        _, err = process.communicate()
        builder.exit_status = process.returncode
        builder.ms = round((time.perf_counter() - builder.started) * 1000, 1)
        if process.returncode:
            builder.detail = err.decode(errors="replace")[-600:]
        builder.done.set()

    threading.Thread(target=reap, name=f"sol-builder-{tokens}", daemon=True).start()
    return builder


def _row(
    wanted: tuple[int, int], key: str, source: str, started: float, **extra: Scalar
) -> dict[str, Scalar]:
    builder = _BUILDERS.get(wanted)
    # `source` becomes the observation's value; the key digest names the split count too.
    return {
        "source": source,
        "tokens": wanted[0],
        "key": key,
        "ms": round((time.perf_counter() - started) * 1000, 1),
        **(builder.document() if builder is not None else {}),
        **extra,
    }


def _resolve(tokens: int, kv_splits: int, device: Any) -> kernel_compile.Loaded | None:
    """The kernel for this count: loaded, waited for, or compiled here and published.

    Every path yields the same bytes, so none of them changes the output. With no store
    (no rooted worker) upstream compiles in-process exactly as it always has."""
    wanted = (tokens, kv_splits)
    loaded = _KERNELS.get(wanted)
    if loaded is not None:
        rows = _KERNEL_ROWS.get()
        if rows is not None and not any(row.get("tokens") == tokens for row in rows):
            # Resident from an earlier request: say so once, rather than print nothing.
            rows.append(_row(wanted, _KEYS[wanted], "memory", time.perf_counter()))
        return loaded
    store = kernel_cache.configured()
    if store is None:
        return None
    started = time.perf_counter()
    key = _key(tokens, kv_splits, device)
    data, source = store.get(key), "store"
    if data is None:
        with store.building(key):
            data, source = store.get(key), "waited"
            if data is None:
                data, source = kernel_compile.compile_sol(tokens, kv_splits), "compiled"
                store.put(key, data, kernel_compile.producer(started))
    loaded = _KERNELS[wanted] = kernel_compile.load(data)
    _KEYS[wanted] = key.digest[:16]
    _kernel_row(_row(wanted, _KEYS[wanted], source, started))
    return loaded


def _pin_preprocess() -> None:
    """Upstream autotunes its K/V block reductions over 4 and 8 warps by timing, and the two
    reduce in different orders: measured bitwise-different summaries, which can flip a
    routing decision. Keep upstream's first candidate only, so no timing race picks it."""
    if _PINNED:
        return
    preprocess = import_module("sol_attn.preprocess")
    for tuned in (preprocess._reduce_kc_kernel, preprocess._reduce_vc_kernel):
        chosen = [c for c in tuned.configs if c.num_warps == 4 and c.num_stages == 1]
        if len(chosen) != 1:
            raise RuntimeError("Sol preprocess no longer offers its 4-warp, 1-stage candidate")
        tuned.configs = chosen
    _PINNED.append(True)


def _native(query: Any, key: Any, value: Any, scale: float | None, sink_tokens: int) -> Any:
    """Call the upstream kernel; a full-document sink selects every KV block.

    This does not perform the wrapper's protected-query recomputation. The same
    public upstream call therefore provides an all-keys arithmetic control.
    """
    if query.device.type == "cuda":
        _pin_preprocess()
    if _sm90(query.device):
        batch, tokens, heads, _ = query.shape
        loaded = _resolve(tokens, 1, query.device)
        if loaded is not None:
            import_module("sol_attn.interface")._compiled[
                (query.device.index, (9, 0), batch, tokens, heads, 1)
            ] = loaded
    return import_module("sol_attn").sol_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        scale=scale,
        tau=1.0,
        thresh_type="exact",
        kv_splits=1,
        sink_start=0,
        sink_tokens=sink_tokens,
    )


def sol_attention(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """One BF16 NHD document; preserve its prefix and zero explicit trailing padding.

    Policy v1 uses upstream tau=1, exact threshold statistics and one KV split.
    `exact` names threshold computation, not dense-equivalent attention. Initial
    dense steps and paths are explicit model policy in the active layout. Under
    Ulysses this runs after the exchange, on the full document of a head subset;
    Sol's block routing is computed per head, so the split matches one GPU.
    """
    import torch

    if torch.compiler.is_compiling():
        raise ValueError("Sol's model layout and site scope require eager execution")
    layout, site = _ACTIVE_LAYOUT.get(), _SITE.get()
    if layout is None or site is None:
        raise ValueError(
            "Sol requires model-owned attention layout and a registered attention site"
        )
    if attn_mask is not None or is_causal or dropout_p != 0 or enable_gqa:
        raise ValueError("Sol supports unmasked noncausal inference without dropout or GQA")
    tensors = (query, key, value)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in tensors):
        raise ValueError("Sol requires batch/sequence/head/dimension tensors")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Sol requires equal Q/K/V geometry")
    if query.shape[0] != 1 or query.shape[-1] != 128 or not query.shape[2]:
        raise ValueError("Sol requires one document with head dimension 128")
    if any(
        t.dtype != torch.bfloat16 or t.device != query.device or t.requires_grad for t in tensors
    ):
        raise ValueError("Sol requires BF16 inference inputs on one device")
    if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device) != (9, 0):
        raise ValueError("This Sol adapter supports the H100 SM90 route only")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    return _execute(query, key, value, scale, layout, site)


def backend(device: Any) -> str:
    """The route upstream selects on this device: `cute_sm90`, or its silent Triton fallback."""
    return str(import_module("sol_attn").get_sol_attn_backend(device))


SOL = Local(sol_attention, lambda query, *_: backend(query.device))


def _execute(
    query: Any, key: Any, value: Any, scale: float | None, layout: AttentionLayout, site: Site
) -> Any:
    """Apply the model's explicit policy after hardware/operand validation."""
    import torch

    dense_path = any(site.path == p or site.path.startswith(p + ".") for p in layout.dense_paths)
    if dense_path and query.shape[1] < layout.live_tokens:
        # A refiner has its own shorter document; its explicit dense policy wins.
        output = _dense(query, key, value, scale)
        _count("dense_path")
        return output
    if layout.live_tokens > query.shape[1]:
        raise ValueError("attention layout exceeds Q/K/V sequence length")
    q, k, v = (t[:, : layout.live_tokens] for t in (query, key, value))
    if dense_path or layout.step < layout.dense_until_step:
        if not dense_path:
            # The dense steps run on the GPU while this count's kernel compiles on the CPU.
            _prefetch(layout.live_tokens, 1, query.device, layout.step)
        live_output = _dense(q, k, v, scale)
        _count("dense_path" if dense_path else "dense_step")
    else:
        live_output = _native(q, k, v, scale, layout.protected_prefix)
        _validate_result(live_output, q)
        if any(
            live_output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (query, key, value)
        ):
            raise ValueError("Sol returned output sharing input storage")
        _count("sparse")
        if layout.protected_prefix:
            live_output[:, : layout.protected_prefix] = _dense(
                q[:, : layout.protected_prefix], k, v, scale
            )
            _count("dense_prefix")
    _validate_result(live_output, q)
    if layout.live_tokens == query.shape[1]:
        return live_output
    output = torch.zeros_like(query)
    output[:, : layout.live_tokens].copy_(live_output)
    return output
