"""Every kernel the Runtime ships, on this card, through its real launch path, with a numeric check against a slow
reference (`cozy_runtime.bench`'s `kernels_inventory` and `kernel_check`; the release gate's `kernels` suite). Paul,
11-04: "prior to a release, all of our kernels should be exercised at least once". One kernel per call, so a fault ends
only its executor and names it.

- `inventory`: what this card claims, read from the code (`declared`, `launched_jits`): `attention.KERNELS` on this row,
  every Runtime `TritonKernels` set, each `_kernels` module, each `@triton.jit` launch. A name with no exerciser fails its
  own call, so a new kernel can't go unnoticed; an entry whose launcher reads its signature out of order fails it too.
- `exercise`: one name, against its reference. Attention takes a long-key tile, several heads, every head dimension its
  row admits, and must beat every answer a kernel can give without computing (V, V's mean, zero); a sparse kernel must
  really take its sparse path (`attention_sol.observing`). `big:<name>` is one call past 2^31 elements. It returns the
  compiled entries and jit functions it launched, which the gate's `check.py` holds against the inventory.
A package's own kernels are its own to check: minimax-h3 holds its glue against float64 before it serves
(`glue.checked`)."""

from __future__ import annotations

import ast
import contextlib
import gc
import json
import math
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from functools import partial
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, cast

import torch
import triton
from diffusers.models.attention_dispatch import dispatch_attention_fn
from triton.runtime.jit import JITFunction

import cozy_runtime
from cozy_runtime import _kernels, lm
from cozy_runtime.accel import kernels as author_kernels
from cozy_runtime.accel import moe
from cozy_runtime.accel.attention import AttentionLayout, attention_scope
from cozy_runtime.accel.fusion import _kernels as fused
from cozy_runtime.accel.moe import _kernels as moe_kernels
from cozy_runtime.internal import (
    arch,
    attention,
    attention_fp8,
    attention_sage2_fixed_mean,
    attention_sage3_fixed,
    attention_sol,
    int8_linear,
    kernel_cache,
    kernel_ready,
    lora_bake,
    nvfp4_linear,
    philox,
    prebuilt,
    sage3,
    svdq_linear,
    triton_kernels,
    verify,
)
from cozy_runtime.internal.contract import unsealed
from cozy_runtime.internal.encoding.selection import launch_providers, measure_device
from cozy_runtime.internal.parallel import turn
from cozy_runtime.internal.probe import UnmeasuredEvidence, conform
from cozy_runtime.lm import _kernels as lm_kernels

if TYPE_CHECKING:
    from cozy_runtime.internal.encoding.selection import DeviceFacts

#: What each `_kernels` module exports, and the names that launch it: a new one fails `inventory`.
NATIVE = {
    "_gemm": "native:int8 native:fp8 native:w4a8 native:svdq",
    "_norm": "native:norm",
    "_fa3": "attention:flash-attn3",
    "_sage3": "attention:sageattention3-fp4",
    "_attn4": "attention:kitchen-int4qk",
}


def _free(index: int) -> int:
    """The driver's free bytes on `index`, read on purpose: a big call needs them, and this tests the memory manager."""
    found = cast("Callable[[int], tuple[int, int]]", unsealed(torch, "mem_get_info"))
    return found(index)[0]


class Outcome(TypedDict, total=False):
    """An exerciser's verdict, and what it measured beside it."""

    passed: bool
    detail: str
    outliers: str
    shapes: str


class Declared(TypedDict):
    """What this card claims (`declared`), and what `inventory` adds to it."""

    names: list[str]
    entries: dict[str, list[str]]
    row: str
    sm: int
    jits: NotRequired[dict[str, list[str]]]
    gpus: NotRequired[int]
    mismatches: NotRequired[list[str]]


class _Kernel(Protocol):
    """A Triton jit function, as the launch recorders read it."""

    __name__: str
    arg_names: list[str]


class _Probed(Protocol):
    """A Runtime module with a Triton set and its probe against the exact law."""

    KERNELS: author_kernels.TritonKernels

    def probe(self, torch_: ModuleType, device: torch.device, launch: author_kernels.Launch, /) -> verify.Verdict: ...


class _ProbedOn(Protocol):
    """A Runtime module with a Triton set and its probe, which takes no torch."""

    KERNELS: author_kernels.TritonKernels

    def probe(self, device: torch.device, launch: author_kernels.Launch, /) -> verify.Verdict: ...


@contextlib.contextmanager
def _recording() -> Iterator[tuple[set[str], set[str]]]:
    """Every compiled entry and jit function launched inside: the two launch paths wrapped, then restored."""
    entries: set[str] = set()
    jits: set[str] = set()
    entry, run = triton_kernels._Launcher.__call__, JITFunction.run

    def launched(self: triton_kernels._Launcher, kernel: author_kernels.JitKernel, grid: tuple[int, ...], *args: object) -> None:
        params = kernel.arg_names
        values = {name: args[params.index(name)] for name in self.names[kernel.__name__]}
        entries.add(author_kernels.entry_key(kernel.__name__, values))
        jits.add(kernel.__name__)
        return entry(self, kernel, grid, *args)

    def jit(self: _Kernel, *args: object, **kwargs: object) -> object:
        jits.add(self.__name__)
        for value in (*args, *kwargs.values()):
            if isinstance(value, JITFunction):
                jits.add(value.__name__)
        return run(self, *args, **kwargs)

    triton_kernels._Launcher.__call__ = launched  # type: ignore[method-assign]  # every launch counted
    JITFunction.run = jit
    try:
        yield entries, jits
    finally:
        triton_kernels._Launcher.__call__ = entry  # type: ignore[method-assign]  # restored
        JITFunction.run = run


def _sources() -> dict[str, ast.Module]:
    root = Path(cozy_runtime.__file__).parent
    return {
        ".".join(("cozy_runtime", *f.relative_to(root).with_suffix("").parts)): ast.parse(f.read_text())
        for f in root.rglob("*.py")
        if "triton" in f.read_text()
    }


def _jit(decorator: ast.expr) -> bool:
    """`@triton.jit`, or `@jit` (`accel.kernels.jit`, a set's typed kernels), called or not."""
    return ast.unparse(getattr(decorator, "func", decorator)).rpartition(".")[2] == "jit"


def launched_jits(sources: dict[str, ast.Module], names: list[str]) -> dict[str, list[str]]:
    """Each module's jit functions this row launches: one in a Triton set only where the row declares
    that set, one in none on every row. A helper, which any jit function calls by name or attribute, is
    launched by its caller."""
    defs = {
        module: [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and any(_jit(d) for d in node.decorator_list)
        ]
        for module, tree in sources.items()
    }
    helpers = {
        call.func.id if isinstance(call.func, ast.Name) else call.func.attr
        for nodes in defs.values()
        for node in nodes
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, (ast.Name, ast.Attribute))
    }
    owners: dict[tuple[str, str], set[str]] = {}
    for kernels in author_kernels._SETS.values():
        for entry in kernels.entries:
            owners.setdefault((entry.kernel.__module__, entry.kernel.__name__), set()).add(kernels.name)
    claimed = {name.removeprefix("triton:") for name in names if name.startswith("triton:")}
    found = {
        module: sorted(
            node.name
            for node in nodes
            if node.name not in helpers
            and (not owners.get((module, node.name)) or owners[(module, node.name)] & claimed)
        )
        for module, nodes in defs.items()
    }
    return {module: jits for module, jits in found.items() if jits}


def declared(row: arch.Row, sm: int) -> Declared:
    """What this card claims, read from the code."""
    sources = _sources()
    for name, tree in sources.items():
        if "TritonKernels(" in ast.unparse(tree) and not name.startswith("cozy_runtime.author"):
            import_module(name.removesuffix(".__init__"))
    on_row = [c for c in attention.KERNELS if not attention._off_row(c, row)]
    names = [f"attention:{c.name}" for c in on_row]
    names += [f"big:{c.name}" for c in on_row if c.artifact and c.admits("bfloat16", 128)]
    named = {c.kernels for c in attention.KERNELS if c.kernels}
    used = {import_module(c.kernels).KERNELS.name for c in on_row if c.kernels}
    every = {import_module(n).KERNELS.name for n in named}
    route = attention_sol.route(divmod(sm, 10))
    # The Runtime's own sets: a package's are its own to check.
    sets = {name: k for name, k in author_kernels._SETS.items() if triton_kernels.home(k) == "cozy_runtime"}
    for name in sets:
        if name in every and name not in used:
            continue  # only its own kernel launches it, and this row runs none
        if name == svdq_linear.KERNELS.name and not row.fp4:
            continue
        if name == "cozy-sol-fa4-merge" and route != attention_sol.FA4:
            continue
        names.append(f"triton:{name}")
    # A format's kernels exist where its row has it; FP8 GEMMs only where the wheel carries them.
    names += ["native:int8", "native:norm", "decode:conform", "kitchen:int8", "big:gemm-int8"]
    if row.fp8:
        names += ["native:fp8", "big:gemm-fp8"]
    names.append("native:w4a8")
    if row.fp4:
        names.append("kitchen:nvfp4")
    if row.family == arch.MMA_SYNC and row.fp4:
        names.append("native:svdq")
    for module in _kernels.__all__:
        if module != "rms_rope_split_half" and module not in NATIVE:
            names.append(f"native-module:{module}")
    entries = {
        s.name: sorted(e.key for e in s.entries)
        for s in sets.values()
        if f"triton:{s.name}" in names
    }
    # A process holds one build of a module, and a `-full` build serves the kernels of the narrower one:
    # those first, or a kernel the narrower build opened refuses them (`attention_kernel_needs_executor`).
    full = {f"attention:{c.name}" for c in on_row if c.artifact.endswith("-full")}
    ordered = sorted(dict.fromkeys(names), key=lambda name: (name not in full, name))
    return {"names": ordered, "entries": entries, "row": row.name, "sm": sm}


def _launcher_order() -> list[str]:
    """Each entry as Triton's launcher reads it: a signature's order is the call's (run 28135)."""
    return [
        f"{kernels.name} {entry.key}: launches by {list(triton_kernels._source(entry).signature)}"
        for kernels in author_kernels._SETS.values()
        for entry in kernels.entries
        if list(triton_kernels._source(entry).signature) != list(entry.kernel.arg_names)
    ]


# ---- attention


def _exact(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    q, k, v = (t.transpose(1, 2).double() for t in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)


def _beats_trivial(torch_: ModuleType, got: torch.Tensor, want: torch.Tensor, v: torch.Tensor) -> tuple[float, float]:
    """(its error, the best of the answers a kernel can return without computing: V, V's mean, 0)."""
    error = verify.error(torch_, got, want)
    trivial = [verify.error(torch_, v.expand_as(want), want)] if v.shape[1] == want.shape[1] else []
    trivial += [verify.error(torch_, v.mean(1, keepdim=True).expand_as(want), want), 1.0]
    return error, min(trivial)


def _attention(name: str, index: int) -> Outcome:
    device = measure_device(torch, index)
    candidate = attention.known()[name]
    ready = attention.pinned(name, device)[0]
    if candidate.sparse:
        return _sparse(ready, device)
    budget = prebuilt.ATTENTION_BUDGET[candidate.exact]
    target = torch.device("cuda", index)
    generator = torch.Generator(device=target).manual_seed(20261104)
    errors: dict[str, float]
    skipped: list[str]
    errors, skipped = {}, []
    dims = sorted(candidate.head_dims | {d for d, _ in candidate.shapes}) or (64, 128, 256)
    for dim in dims:
        for dtype_name in sorted(candidate.dtypes) or ("bfloat16", "float16"):
            if not candidate.admits(dtype_name, dim):
                continue
            # A tile's ragged tail, past Kitchen's 1024 long-key rule, several heads.
            for batch, tokens, heads in ((1, 257, 2), (1, 1536, 4), (2, 300, 8)):
                if batch > 1 and not candidate.batches:
                    continue
                q, k, v = (
                    torch.randn(batch, tokens, heads, dim, generator=generator, device=target,
                                dtype=torch.float32)
                    for _ in range(3)
                )  # fmt: skip
                want = _exact(q, k, v)
                dtype = getattr(torch, dtype_name)
                # cast: diffusers types it one tensor but returns the registered backend's own value, which may be (out, lse)
                out = cast("torch.Tensor | tuple[torch.Tensor, ...]", dispatch_attention_fn(q.to(dtype), k.to(dtype), v.to(dtype), backend=attention._member(ready.candidate)))
                got = out[0] if isinstance(out, tuple) else out
                case = f"d{dim} {dtype_name} {batch}x{tokens}x{heads}"
                error, trivial = _beats_trivial(torch, got, want, v.double())
                errors[case] = error if error < 0.9 * min(trivial, 1.0) else math.inf
    if not errors:
        return {"passed": False, "detail": "no case it admits ran"}
    worst = max(errors, key=errors.__getitem__)
    return {
        "passed": errors[worst] <= budget,
        "detail": f"rel L2 {errors[worst]:.2e} at {worst} (budget {budget:g}, {len(errors)} cases)",
        "outliers": _outliers(ready, candidate, target, generator),
    }


def _outliers(ready: "attention.Ready", candidate: "attention.Candidate", target: torch.device, generator: torch.Generator) -> str:
    """The same call on activations with real outliers (a few rows far past the rest, as H3's key
    gain has): FA3 FP8 passed the random probe and garbled H3 (runs 1240-1252). Measured, no bar yet."""
    dim = 128 if candidate.admits("bfloat16", 128) else 64
    if not candidate.admits("bfloat16", dim):
        return "bf16 not served"
    q, k, v = (
        torch.randn(1, 1024, 4, dim, generator=generator, device=target, dtype=torch.float32)
        for _ in range(3)
    )
    for x in (q, k, v):
        x[:, ::97] *= 64.0
    want = _exact(q, k, v)
    half = [t.to(torch.bfloat16) for t in (q, k, v)]
    # cast: diffusers types it one tensor but returns the registered backend's own value, which may be (out, lse)
    out = cast("torch.Tensor | tuple[torch.Tensor, ...]", dispatch_attention_fn(*half, backend=attention._member(ready.candidate)))  # type: ignore[arg-type]  # a splat of the 3-list (mypy can't count it)
    got = out[0] if isinstance(out, tuple) else out
    return f"{verify.error(torch, got, want):.3f}"


#: The largest attention a public function reaches on one GPU: H3 Regenerate at 1440p 15 s, or a
#: 2160p window (minimax-h3 `regenerate._MOST_ROWS`, 385,200 rows, with its text and audio): past 2^31
#: elements, where an int32 element offset wraps.
_BIG = (1, 392_000, 56, 128)


def _not_fit(name: str, index: int, work: Callable[[str, int, tuple[int, ...], bool], Outcome]) -> Outcome:
    """H3's largest, else the least past 2^31 elements with Q, K and V one tensor, on a card that
    can't hold that. A card that can't hold either hasn't exercised it: said, and no failure (its real
    runs cut attention to fit, so no kernel call is this big there)."""
    why = ""
    for shape, shared in ((_BIG, False), ((1, 300_000, 56, 128), True)):
        before = _released(index)
        try:
            return work(name, index, shape, shared)
        except torch.OutOfMemoryError as exc:
            why = str(exc)[:80]
            # The attempt's tensors are its frames' locals: dropped here, whatever else keeps the exception.
            traceback.clear_frames(exc.__traceback__)
        after = _released(index)
        if after < before - (1 << 30):
            # Memory an OOM never gives back starves the next attempt and cell (xs-174): the finding, not a "not run".
            return {"passed": False, "detail": f"{(before - after) >> 30} GiB stayed held after {shape}'s OOM, its tensors "
                                               f"dropped and the cache emptied ({why})"}  # fmt: skip
    return {"passed": True, "detail": f"not run: does not fit this card ({why})"}


def _released(index: int) -> int:
    """The driver's free bytes once everything dropped is collected and the cache emptied."""
    gc.collect()
    torch.cuda.synchronize(index)
    torch.cuda.empty_cache()
    return _free(index)


def _big(name: str, index: int, shape: tuple[int, ...], shared: bool) -> Outcome:
    """One call past the int32 element limit: its heads hold one head's values, so a wild offset
    shows as heads that differ, and sampled rows of the last head against exact attention in
    float64, which costs no N^2."""
    device = measure_device(torch, index)
    ready = attention.pinned(name, device)[0]
    candidate, target = ready.candidate, torch.device("cuda", index)
    free = _released(index)
    if free < 24 << 30:
        return {"passed": True, "detail": f"not run: {free >> 30} GiB free cannot hold 2^31 elements of Q, K, V and out"}
    tokens, heads, dim = shape[1:]
    g = torch.Generator(device=target).manual_seed(9)
    if candidate.sparse:
        block, blocks = 64, -(-tokens // 64)
        centers = torch.randn(blocks, 1, dim, generator=g, device=target, dtype=torch.float32) * 2.0
        values = torch.randn(blocks, 1, dim, generator=g, device=target, dtype=torch.float32)
        home = torch.arange(tokens, device=target) // block
        goes = torch.randint(0, blocks, (blocks,), generator=g, device=target)[home]
        one = [
            x + torch.randn(tokens, 1, dim, generator=g, device=target, dtype=torch.float32) * s
            for x, s in ((centers[goes], 0.1), (centers[home], 0.1), (values[home], 0.05))
        ]
        del centers, values, home, goes
    else:
        one = [torch.randn(tokens, 1, dim, generator=g, device=target, dtype=torch.float32) for _ in range(3)]
    if shared:
        one = [one[0]] * 3
        q = k = v = one[0].to(torch.bfloat16)[None].expand(1, tokens, heads, dim).contiguous()
    else:
        q, k, v = (x.to(torch.bfloat16)[None].expand(1, tokens, heads, dim).contiguous() for x in one)
    if candidate.sparse:
        attention.bind_dense(device)  # its undeclared dense rows, as a selection with a sparse site binds them
        if candidate.import_name == "sol_attn":
            _sol_built(q)
        layout = AttentionLayout(live_tokens=tokens, protected_prefix=1259, step=1, dense_until_step=1)
        site = attention_sol._SITE.set(attention_sol.Site("kernels-suite", "attn"))
        try:
            with attention_scope(layout), attention_sol.observing() as counts:
                # cast: diffusers types it one tensor but returns the registered backend's own value, which may be (out, lse)
                out = cast("torch.Tensor | tuple[torch.Tensor, ...]", dispatch_attention_fn(q, k, v, backend=attention._member(ready.candidate)))
        finally:
            attention_sol._SITE.reset(site)
        took = dict(counts)
        budget = 1 / 3
    else:
        out = cast("torch.Tensor | tuple[torch.Tensor, ...]", dispatch_attention_fn(q, k, v, backend=attention._member(ready.candidate)))  # cast: as above
        took = {}
        budget = prebuilt.ATTENTION_BUDGET[candidate.exact]
    got = out[0] if isinstance(out, tuple) else out
    spread = verify.error(torch, got[:, :, -1], got[:, :, 0])
    rows = torch.tensor([0, 1, tokens // 2, tokens - 1], device=target)
    scores = (one[0][rows, 0].double() @ one[1][:, 0].double().T) / dim**0.5
    want = torch.softmax(scores, -1) @ one[2][:, 0].double()
    error = verify.error(torch, got[0, rows, -1], want)
    ok = spread <= budget and error <= budget and (not candidate.sparse or took.get("sparse") == 1)
    return {"passed": ok, "detail": f"{tokens}x{heads}x{dim}: last head vs first {spread:.2e}, sampled rows vs exact {error:.2e} (budget {budget:.3g}) {took or ''}"}


def _sol_built(q: torch.Tensor) -> None:
    """Sol's object for this length, on a card whose route has one, and upstream's preprocess, on
    every route (FA4's too): until they land a sparse step serves its dense reference."""
    card, store, index = attention_sol._capability(q.device), kernel_cache.configured(), q.device.index or 0
    if store is None:
        return
    if card is not None:
        found = attention_sol._job_for(q.shape[1], card)
        kernel_ready.wait(f"{attention_sol.KERNEL} {found.key.digest[:16]}",
                          partial(attention_sol._open, store, found, card))  # fmt: skip
    kernel_ready.wait(f"{attention_sol.KERNEL} preprocess cuda:{index}",
                      partial(attention_sol._warm, index))  # fmt: skip


def _sparse(ready: "attention.Ready", device: "DeviceFacts") -> Outcome:
    """A sparse kernel as a sparse step calls it: inside its model's layout and a Sol site. Its
    routed blocks are exact and the rest approximate, so the bar is sanity, and that the sparse
    path really ran."""
    target = torch.device("cuda", device.index)
    attention.bind_dense(device)  # its undeclared dense rows, as a selection with a sparse site binds them
    tokens, heads, prefix = 8192, 8, 256
    half, want, trivial = _routable(tokens, heads, target)
    if ready.candidate.import_name == "sol_attn":
        _sol_built(half[0])
    layout = AttentionLayout(live_tokens=tokens, protected_prefix=prefix, step=1, dense_until_step=1)
    site = attention_sol._SITE.set(attention_sol.Site("kernels-suite", "attn"))
    try:
        with attention_scope(layout), attention_sol.observing() as counts:
            # cast: diffusers types it one tensor but returns the registered backend's own value, which may be (out, lse)
            out = cast("torch.Tensor | tuple[torch.Tensor, ...]", dispatch_attention_fn(*half, backend=attention._member(ready.candidate)))  # type: ignore[arg-type]  # a splat of the 3-list (mypy can't count it)
    finally:
        attention_sol._SITE.reset(site)
    got = out[0] if isinstance(out, tuple) else out
    error = verify.error(torch, got, want)
    # Kitchen rounds Q, K and V to int8 and every kernel routes only some blocks: measured 0.0x where
    # garbage is O(1), so a third of the nothing-computed answer is far outside any of them.
    ok = counts["sparse"] == 1 and error < trivial / 3
    return {"passed": ok, "detail": f"rel L2 {error:.3f} (nothing-computed answers {trivial:.3f}), calls {dict(counts)}"}


def _routable(tokens: int, heads: int, target: torch.device) -> tuple[list[torch.Tensor], torch.Tensor, float]:
    """Q, K and V a sparse kernel can route: each query row belongs to one block of keys, so exact
    attention lands there, and a kernel that routes it wrong or misaddresses a tile misses it by O(1)."""
    g = torch.Generator(device=target).manual_seed(20261105)
    block, blocks = 64, -(-tokens // 64)
    centers = torch.randn(blocks, heads, 128, generator=g, device=target, dtype=torch.float32) * 2.0
    values = torch.randn(blocks, heads, 128, generator=g, device=target, dtype=torch.float32)
    home = torch.arange(tokens, device=target) // block
    goes = torch.randint(0, blocks, (blocks,), generator=g, device=target)[home]  # a video's are local too
    noise = [torch.randn(tokens, heads, 128, generator=g, device=target, dtype=torch.float32) * s for s in (0.1, 0.1, 0.05)]
    q = (centers[goes] + noise[0])[None]
    k = (centers[home] + noise[1])[None]
    v = (values[home] + noise[2])[None]
    want = _exact(q, k, v)
    trivial = min(verify.error(torch, v.double(), want), verify.error(torch, v.double().mean(1, keepdim=True).expand_as(want), want), 1.0)
    return [t.to(torch.bfloat16) for t in (q, k, v)], want, trivial


# ---- Triton sets


def _launcher(kernels: author_kernels.TritonKernels, index: int) -> author_kernels.Launch:
    return cast(author_kernels.Launch, triton_kernels.ready(kernels, True, device=f"cuda:{index}"))  # waited: never None


def _e4m3(index: int) -> Outcome:
    target = torch.device("cuda", index)
    attention_fp8.serve(index, _launcher(attention_fp8.KERNELS, index))
    errors: dict[str, tuple[float, float]]
    scales: dict[str, float]
    generator, errors, scales = torch.Generator(device=target).manual_seed(1), {}, {}
    for dtype in (torch.bfloat16, torch.float16):
        for dim in attention_fp8.DIMS:
            x = torch.randn(2, 300, 4, dim, generator=generator, device=target, dtype=torch.float32).to(dtype)
            out, descale = attention_fp8.quantise(x)
            law, law_scale = attention_fp8.quantise_eager(x)
            case = f"{dtype} d{dim}"
            errors[case] = (
                verify.error(torch, out.double() * descale[:, None, :, None].double(), x.double()),
                verify.error(torch, law.double() * law_scale[:, None, :, None].double(), x.double()),
            )
            scales[case] = verify.error(torch, descale, law_scale)
    rounding, scale = verify.against_law(errors), verify.within(scales, 1e-5)
    return {"passed": rounding.ok and scale.ok, "detail": f"{rounding.detail}; scale {scale.detail}"}


def _verdict(verdict: verify.Verdict) -> Outcome:
    return {"passed": verdict.ok, "detail": verdict.detail}


def _means(launch: author_kernels.Launch, index: int, errors: dict[str, float]) -> None:
    """Every `finish_mean` specialization: a tile count at each power of two, a ragged last tile."""
    target = torch.device("cuda", index)
    generator = torch.Generator(device=target).manual_seed(2)
    for shift in range(attention_sage2_fixed_mean.REDUCE_MAX.bit_length()):
        tiles = 1 << shift
        tokens = tiles * attention_sage2_fixed_mean.TOKEN_TILE - 7
        heads = 3 if shift < 8 else 1
        x = torch.randn(1, tokens, heads, 128, generator=generator, device=target, dtype=torch.float32).to(torch.bfloat16)
        got = attention_sage2_fixed_mean.fixed_mean(launch, x, tokens)
        want = x[0].double().sum(0) / tokens
        errors[f"mean of {tokens}"] = verify.error(torch, got, want)
        del x


def _merged(index: int, errors: dict[str, float]) -> None:
    """`merge_keys`: a query's rows over two key sets merged by their log-sum-exps (Sage2's log2 one over keys less their
    mean), against FP64, at a ragged row tile and at one head."""
    target = torch.device("cuda", index)
    generator = torch.Generator(device=target).manual_seed(4)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=target, dtype=torch.float32)

    scale = 128**-0.5
    for count, heads in ((300, 3), (64, 1)):
        q, rows, keyed = (randn(1, count, heads, 128).to(torch.bfloat16) for _ in range(3))
        km = (randn(1, 1, heads, 128) * 0.5).to(torch.bfloat16)
        lse, raw = randn(1, count, heads) * 3.0, randn(1, heads, count) * 3.0
        keyed_lse = raw.double().transpose(1, 2) / 1.44269504 + scale * (q.double() * km.double()).sum(-1)
        top = torch.maximum(lse.double(), keyed_lse)
        a, b = (lse.double() - top).exp()[..., None], (keyed_lse - top).exp()[..., None]
        want = (rows.double() * a + keyed.double() * b) / (a + b)
        got = rows.clone()
        attention_sage2_fixed_mean.merge(got, lse, keyed, raw, q, km, scale)
        errors[f"merge of {count} rows x {heads} heads"] = verify.error(torch, got, want)


def _kmean(index: int) -> Outcome:
    errors: dict[str, float]
    launch, errors = _launcher(attention_sage2_fixed_mean.KERNELS, index), {}
    attention_sage2_fixed_mean.serve(index, launch)
    _means(launch, index, errors)
    _merged(index, errors)
    return _verdict(verify.within(errors, prebuilt.BUDGET))


def _sage3_fixed(index: int) -> Outcome:
    target = torch.device("cuda", index)
    errors: dict[str, float]
    launch, errors = _launcher(attention_sage3_fixed.KERNELS, index), {}
    attention_sage3_fixed.serve(index, launch)
    _means(launch, index, errors)
    generator = torch.Generator(device=target).manual_seed(3)
    for tokens in (127, 128, 300):
        q, k, v = (
            torch.randn(1, tokens, 4, 128, generator=generator, device=target, dtype=torch.float32).to(torch.bfloat16)
            for _ in range(3)
        )
        padded = triton.cdiv(tokens, 128) * 128
        for fixed in (False, True):
            got = attention_sage3_fixed.fixed_preprocess(q, k, v, fixed_correction=fixed)
            pad = torch.zeros(1, padded - tokens, 4, 128, device=target, dtype=torch.float32)
            qm = torch.cat([q.float(), pad], 1).double().sum(1, keepdim=True) / padded
            km = k.double().sum(1, keepdim=True) / tokens
            wq = (torch.cat([q.double(), pad.double()], 1) - qm).transpose(1, 2)
            wk = (torch.cat([k.double() - km, pad.double()], 1)).transpose(1, 2)
            wv = torch.cat([v.double(), pad.double()], 1).transpose(1, 2)
            case = f"{tokens} fixed={fixed}"
            errors[f"q {case}"] = verify.error(torch, got["q"], wq)
            errors[f"k {case}"] = verify.error(torch, got["k"], wk)
            errors[f"v {case}"] = verify.error(torch, got["v"], wv)
    return _verdict(verify.within(errors, prebuilt.BUDGET))


def _group_mean(index: int) -> Outcome:
    target = torch.device("cuda", index)
    sage3.serve(index, _launcher(sage3.KERNELS, index))
    errors: dict[str, float]
    generator, errors = torch.Generator(device=target).manual_seed(4), {}
    for dtype in (torch.bfloat16, torch.float16):
        for dim in (64, 128):
            q = torch.randn(2, 3, 384, dim, generator=generator, device=target, dtype=torch.float32).to(dtype)
            out, qm = sage3.triton_group_mean(q)
            groups = q.double().view(2, 3, 3, 128, dim)
            want = groups.sum(3) / 128
            errors[f"qm {dtype} d{dim}"] = verify.error(torch, qm, want)
            errors[f"q {dtype} d{dim}"] = verify.error(
                torch, out, (groups - want[:, :, :, None]).view(q.shape)
            )
    # Upstream's mean of 128 bf16 values rounds as it adds, so its error is above a kernel's own.
    return _verdict(verify.within(errors, 0.1))


def _probed(module: _Probed, index: int) -> Outcome:
    launch = _launcher(module.KERNELS, index)
    return _verdict(module.probe(torch, torch.device("cuda", index), launch))


def _probed_on(module: _ProbedOn, index: int) -> Outcome:
    launch = _launcher(module.KERNELS, index)
    with torch.cuda.device(index):
        return _verdict(module.probe(torch.device("cuda", index), launch))


def _bf16(*shape: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16)


def _lm(index: int) -> Outcome:
    """Decode's conv and recurrent kernels against their arithmetic, in float32, in Python."""
    target = torch.device("cuda", index)
    errors: dict[str, float]
    launch, errors = _launcher(lm.KERNELS, index), {}
    g = torch.Generator(device=target).manual_seed(5)
    conv, block = lm_kernels.CONV, lm_kernels.CONV_BLOCK
    batch, channels = 2, block + 13
    x, state, weight = _bf16(batch, channels, 1, generator=g, device=target), _bf16(batch, channels, conv, generator=g, device=target), _bf16(channels, conv, generator=g, device=target)
    windows = torch.cat([state.float(), x.float()], -1)[..., 1:]
    want = (windows * weight.float()[None]).sum(-1).to(torch.bfloat16).float()
    want = want * torch.sigmoid(want)
    held = state.clone()
    lm_slot = lm._conv_slot(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("took the eager path")), launch)
    got = lm_slot(x, held, weight, None, "silu")
    errors["conv step"] = verify.error(torch, got[..., 0], want)
    errors["conv state"] = verify.error(torch, held, windows.to(torch.bfloat16))
    tokens = 3
    xs = _bf16(batch, channels, conv + tokens, generator=g, device=target)
    out = xs.new_empty(batch, conv + tokens, channels).transpose(1, 2)
    launch(lm_kernels.conv_steps_kernel, (batch, -(-channels // block), 1), xs, weight, out[..., conv:], channels, tokens, xs.stride(0), xs.stride(1), out.stride(0), out.stride(2), conv, block)
    stepped = torch.stack([(xs.float()[..., t + 1 : t + 1 + conv] * weight.float()[None]).sum(-1) for t in range(tokens)], -1).to(torch.bfloat16).float()
    errors["conv steps"] = verify.error(torch, out[..., conv:], stepped * torch.sigmoid(stepped))
    dk, dv, bv = lm_kernels.KEY_DIM, lm_kernels.VALUE_DIM, lm_kernels.VALUE_BLOCK
    heads = 3
    q, k = (_bf16(batch, tokens, heads, dk, generator=g, device=target) for _ in range(2))
    v = _bf16(batch, tokens, heads, dv, generator=g, device=target)
    gate = torch.randn(batch, tokens, heads, generator=g, device=target, dtype=torch.float32) * -0.1
    beta = torch.rand(batch, tokens, heads, generator=g, device=target, dtype=torch.float32).to(torch.bfloat16)
    start = torch.randn(batch, heads, dk, dv, generator=g, device=target, dtype=torch.float32)
    states = start.new_empty((tokens, *start.shape))
    out = v.new_empty((batch, tokens, heads, dv))
    scale = float(dk) ** 0.5
    launch(lm_kernels.recurrent_steps_kernel, (batch * heads, dv // bv, 1), q, k, v, gate, beta, start, states, out, heads, tokens, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], scale, dk, dv, bv)
    wants: list[torch.Tensor]
    state, wants = start.clone(), []
    for t in range(tokens):
        qq = torch.nn.functional.normalize(q[:, t].float(), dim=-1, eps=1e-3) / scale
        kk = torch.nn.functional.normalize(k[:, t].float(), dim=-1, eps=1e-3)
        state = state * torch.exp(gate[:, t])[..., None, None]
        delta = (v[:, t].float() - (state * kk[..., None]).sum(-2)) * beta[:, t].float()[..., None]
        state = state + kk[..., None] * delta[..., None, :]
        wants.append((state * qq[..., None]).sum(-2))
    errors["recurrent out"] = verify.error(torch, out, torch.stack(wants, 1))
    errors["recurrent state"] = verify.error(torch, states[-1], state)
    return _verdict(verify.within(errors, prebuilt.BUDGET))


def _moe(index: int) -> Outcome:
    """Routed experts against each expert's own product."""
    target = torch.device("cuda", index)
    errors: dict[str, float]
    launch, errors = _launcher(moe.KERNELS, index), {}
    g = torch.Generator(device=target).manual_seed(6)
    n, k, pairs, experts = moe_kernels.BLOCK_N + 5, moe_kernels.BLOCK_K + 9, 7, 3
    route = torch.randint(0, experts, (pairs,), generator=g, device=target, dtype=torch.int64)
    grid, blocks = (pairs, -(-n // moe_kernels.BLOCK_N), 1), (moe_kernels.BLOCK_N, moe_kernels.BLOCK_K)
    weights = [_bf16(n, k, generator=g, device=target) for _ in range(experts)]
    table = torch.tensor([w.data_ptr() for w in weights], device=target, dtype=torch.int64)
    x = _bf16(pairs, k, generator=g, device=target)
    out = torch.empty((pairs, n), dtype=torch.bfloat16, device=target)
    launch(moe_kernels.experts_bf16_kernel, grid, x, table, route, out, n, k, 1, *blocks)
    want = torch.stack([x[p].double() @ weights[route[p]].double().T for p in range(pairs)])
    errors["bf16"] = verify.error(torch, out, want)
    data = [torch.randint(-127, 128, (n, k), generator=g, device=target, dtype=torch.int8) for _ in range(experts)]
    scales = [torch.rand(n, generator=g, device=target, dtype=torch.float32).add_(0.5) * 1e-3 for _ in range(experts)]
    dtable = torch.tensor([d.data_ptr() for d in data], device=target, dtype=torch.int64)
    stable = torch.tensor([s.data_ptr() for s in scales], device=target, dtype=torch.int64)
    rows = torch.randint(-127, 128, (pairs, k), generator=g, device=target, dtype=torch.int8)
    row_scale = torch.rand(pairs, generator=g, device=target, dtype=torch.float32).add_(0.5) * 1e-2
    launch(moe_kernels.experts_int8_kernel, grid, rows, row_scale, dtable, stable, route, out, n, k, 1, *blocks)
    want = torch.stack([(rows[p].double() @ data[route[p]].double().T) * row_scale[p].double() * scales[route[p]].double() for p in range(pairs)])
    errors["int8"] = verify.error(torch, out, want)
    _moe_grouped(launch, target, g, errors)
    return _verdict(verify.within(errors, prebuilt.BUDGET))


def _moe_grouped(launch: author_kernels.Launch, target: torch.device, g: torch.Generator, errors: dict[str, float]) -> None:
    """The grouped GEMM over the pairs `Experts._tiles` sorts by expert, both ways (GATHER: token rows in, sorted rows
    out; else sorted rows in, each pair's row out), bf16 and int8, against each pair's own product. More pairs than
    `GROUPED`, so a layer takes this path; widths ragged against the tiles."""
    experts, top, tokens = 3, 2, moe.GROUPED + 29
    n, k, pairs = 2 * moe_kernels.GEMM_N + 16, 2 * moe_kernels.GEMM_K["int8"] + 16, (moe.GROUPED + 29) * 2
    index = torch.randint(0, experts, (tokens, top), generator=g, device=target, dtype=torch.int64)
    layer = moe.Experts([moe.RoutedExpert(16, 8, gate_first=True) for _ in range(experts)], layer=0)
    order, spans, tiles = layer._tiles(index)
    route = index.reshape(-1)[order]  # each sorted row's expert
    grid = (tiles * -(-n // moe_kernels.GEMM_N), 1, 1)
    weights = torch.stack([_bf16(n, k, generator=g, device=target) for _ in range(experts)])
    data = torch.randint(-127, 128, (experts, n, k), generator=g, device=target, dtype=torch.int8)
    scales = torch.rand(experts, n, generator=g, device=target, dtype=torch.float32).add_(0.5) * 1e-3
    table, dtable, stable = (torch.tensor([t.data_ptr() for t in ts], device=target, dtype=torch.int64) for ts in (weights, data, scales))
    out = torch.empty((pairs, n), dtype=torch.bfloat16, device=target)
    for gather in (1, 0):
        source = order // top if gather else torch.arange(pairs, device=target)
        written = torch.arange(pairs, device=target) if gather else order
        div, rows = (top, tokens) if gather else (1, pairs)

        def placed(sorted_rows: torch.Tensor) -> torch.Tensor:
            found = torch.empty_like(sorted_rows)
            found[written] = sorted_rows
            return found

        x = _bf16(rows, k, generator=g, device=target)
        rest = (order, spans, out, n, k, div, tiles, moe_kernels.GEMM_M, moe_kernels.GEMM_N)
        launch(moe_kernels.grouped_bf16_kernel, grid, x, table, *rest, moe_kernels.GEMM_K["bf16"], moe_kernels.GROUP, gather)
        want = torch.einsum("rk,rnk->rn", x[source].double(), weights[route].double())
        errors[f"grouped bf16 gather={gather}"] = verify.error(torch, out, placed(want))
        q = torch.randint(-127, 128, (rows, k), generator=g, device=target, dtype=torch.int8)
        q_scale = torch.rand(rows, generator=g, device=target, dtype=torch.float32).add_(0.5) * 1e-2
        launch(moe_kernels.grouped_int8_kernel, grid, q, q_scale, dtable, stable, *rest, moe_kernels.GEMM_K["int8"],
               moe_kernels.GROUP, gather)  # fmt: skip
        want = torch.einsum("rk,rnk->rn", q[source].double(), data[route].double())
        want = want * q_scale[source, None].double() * scales[route].double()
        errors[f"grouped int8 gather={gather}"] = verify.error(torch, out, placed(want))


# Fused rows are one Runtime kernel per constexpr set. A package declares its own set; these two take each
# axis both ways (what a package's fused plan can specialize), and a set a package adds is its package's to check.
_FUSED = fused.entries(
    256, 64, 32, norm=fused.RMS, weight=True, offset=0, weights="*bf16", once=False, modulate=2, tanh=True,
    f16=False, interleaved=False, round=fused.BF16, product=False, qk_weight=True, parallel=False, composed=False,
)  # fmt: skip
_FUSED16 = fused.entries(
    256, 64, 32, norm=fused.LAYER, weight=True, offset=1, weights="*fp32", once=True, modulate=1, tanh=False,
    f16=True, interleaved=True, round=fused.FP16, product=True, qk_weight=False, parallel=True, composed=False,
)  # fmt: skip
_CHANNEL = fused.channel_entries("bf16", True, True)
FUSED = author_kernels.TritonKernels(
    "cozy-bench-fused", (*_FUSED.all(), *_FUSED16.all(), *_CHANNEL), origin=__name__
)


def _fused_row(launch: author_kernels.Launch, entry: author_kernels.TritonEntry, target: torch.device, g: torch.Generator,
               errors: dict[str, float]) -> None:
    c = dict(entry.constexprs)
    n, width = 5, 256
    if c["F16"]:
        row, residual = torch.float16, torch.float32
    else:
        row, residual = torch.bfloat16, torch.bfloat16
    x_type = row if not c["F16"] or c["X16"] else torch.float32
    x = torch.randn(n, width, generator=g, device=target, dtype=torch.float32).to(x_type)
    branch, other = (torch.randn(n, width, generator=g, device=target, dtype=torch.float32).to(row) for _ in range(2))
    tables = [torch.randn(3, width, generator=g, device=target, dtype=torch.float32).to(residual) * 0.5 for _ in range(3)]
    weight = torch.randn(width, generator=g, device=target, dtype=torch.float32).to(torch.float32 if c["F16"] else torch.bfloat16) * 0.1
    index = torch.tensor([0, 2, 1, 1, 0], device=target, dtype=torch.int64)
    gate, scale, shift = tables
    args: dict[str, torch.Tensor] = {}
    if c["BRANCHES"]:
        args = {"branch": branch, "gate": gate}
        if c["BRANCHES"] == 2:
            args["other"] = other
    hidden, out = fused.norm_modulate(
        launch, entry, x, weight, scale if c["MODULATE"] else None, shift if c["MODULATE"] == 2 else None, index, 1e-5, **args
    )
    h = x.double()
    if c["BRANCHES"]:
        b = branch.double() + (other.double() if c["BRANCHES"] == 2 else 0)
        g_row = gate[index].double()
        h = h + (torch.tanh(g_row) if c["TANH"] else g_row) * b
    key = entry.key
    errors[f"hidden {key}"] = verify.error(torch, hidden, h)
    if c["NORM"]:
        y = h - h.mean(-1, keepdim=True) if c["NORM"] == fused.LAYER else h
        want = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)
        if c["WEIGHT"]:
            want = want * (weight.double() + c["OFFSET"])
        if c["MODULATE"]:
            want = want * (1 + scale[index].double())
        if c["MODULATE"] == 2:
            want = want + shift[index].double()
        assert out is not None  # a NORM entry writes out
        errors[f"out {key}"] = verify.error(torch, out, want)


def _fused_rope(launch: author_kernels.Launch, entry: author_kernels.TritonEntry, target: torch.device, g: torch.Generator,
                errors: dict[str, float]) -> None:
    c = dict(entry.constexprs)
    seq, heads, dim, rotary = 7, c["HEADS"] + 3, c["HEAD_DIM"], c["ROTARY"]
    dtype = torch.float32 if c["X32"] else (torch.float16 if c["ROUND"] == fused.FP16 else torch.bfloat16)
    x = torch.randn(1, seq, heads, dim, generator=g, device=target, dtype=torch.float32).to(dtype)
    weight = torch.randn(dim, generator=g, device=target, dtype=torch.float32).to(torch.bfloat16) * 0.1
    angle = torch.rand(seq, rotary if not c["INTERLEAVED"] else rotary // 2, generator=g, device=target, dtype=torch.float32) * 6.0
    value = (torch.cos(angle), torch.sin(angle))
    tables = fused.rope_tables(value, interleaved=bool(c["INTERLEAVED"]), rotary=rotary)
    out_type = torch.float16 if c["ROUND"] == fused.FP16 else torch.bfloat16
    got = fused.head_norm_rope(launch, entry, x, weight if c["WEIGHT"] else None, 1e-5, tables, out_type)
    y = x.double()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)
    if c["WEIGHT"]:
        y = y * (weight.double() + c["OFFSET"])
    want = y.clone()
    cos, sin = (v.double() for v in value)
    for j in range(rotary):
        if c["INTERLEAVED"]:
            low, partner, col = j % 2 == 0, j ^ 1, j // 2
        else:
            low, partner, col = j < rotary // 2, j + rotary // 2 if j < rotary // 2 else j - rotary // 2, j
        paired = -y[..., partner] if low else y[..., partner]
        want[..., j] = y[..., j] * cos[:, col][None, :, None] + paired * sin[:, col][None, :, None]
    errors[f"rope {entry.key}"] = verify.error(torch, got, want)


def _fused_channel(launch: author_kernels.Launch, entry: author_kernels.TritonEntry, target: torch.device, g: torch.Generator,
                   errors: dict[str, float]) -> None:
    c = dict(entry.constexprs)
    batch, channels, plane = 2, 4, fused.CHANNEL_BLOCK + 11
    x = _bf16(batch, channels, plane, generator=g, device=target)
    weight, bias = (_bf16(channels, 1, generator=g, device=target) for _ in range(2))
    out = fused.channel_norm(launch, entry, x, weight, bias, 2.0, 1e-5)
    v = x.double() / x.double().norm(dim=1, keepdim=True).clamp(min=1e-5)
    want = v * 2.0 * weight.double()[None] + bias.double()[None]
    if c["SILU"]:
        want = torch.nn.functional.silu(want)
    errors[f"channel {entry.key}"] = verify.error(torch, out, want)


def _fused(index: int) -> Outcome:
    target = torch.device("cuda", index)
    errors: dict[str, float]
    launch, errors = _launcher(FUSED, index), {}
    g = torch.Generator(device=target).manual_seed(8)
    for entry in FUSED.entries:
        name = entry.kernel.__name__
        print(f"KERNEL-ENTRY {entry.key}", file=sys.stderr, flush=True)  # a fault's log names its entry
        if name == "norm_modulate_kernel":
            _fused_row(launch, entry, target, g, errors)
        elif name == "head_norm_rope_kernel":
            _fused_rope(launch, entry, target, g, errors)
        elif name == "channel_norm_kernel":
            _fused_channel(launch, entry, target, g, errors)
        else:
            # `n_cols` is one of the entry's aligned integers: a width 16 divides, as `validate` requires.
            gate = _bf16(9, 304, generator=g, device=target) if not entry.constexprs["F16"] else torch.randn(9, 304, generator=g, device=target, dtype=torch.float32).to(torch.float16)
            up = torch.randn(9, 304, generator=g, device=target, dtype=torch.float32).to(gate.dtype)
            got = fused.swiglu(launch, entry, gate, up)
            errors[f"swiglu {entry.key}"] = verify.error(torch, got, torch.nn.functional.silu(gate.double()) * up.double())
        torch.cuda.synchronize(index)
    return _verdict(verify.within(errors, prebuilt.BUDGET))


# ---- native


def _library(index: int) -> "prebuilt.Prebuilt":
    device = measure_device(torch, index)
    ops, why = prebuilt.library(torch, index, device.sm)
    if ops is None:
        raise RuntimeError(f"the 8-bit library does not serve here: {why}")
    return ops


def _format(index: int, f: "prebuilt._Format") -> Outcome:
    return _verdict(prebuilt.probe(torch, _library(index), f))


def _w4a8(index: int) -> Outcome:
    return _verdict(prebuilt.probe_w4a8(torch, _library(index)))


def _norm(index: int) -> Outcome:
    """RMS norm and split-half RoPE against the arithmetic it fuses: rotation matrices over the
    first `rot_dim` channels, each half of them multiplied by its 2x2."""
    target = torch.device("cuda", index)
    g = torch.Generator(device=target).manual_seed(7)
    batch, tokens, heads, dim, rotary = 1, 64, 4, 128, 96
    q, k = (_bf16(batch, tokens, heads, dim, generator=g, device=target) for _ in range(2))
    q_scale, k_scale = (torch.rand(dim, generator=g, device=target, dtype=torch.float32).to(torch.bfloat16) for _ in range(2))
    angle = torch.rand(tokens, rotary // 2, generator=g, device=target, dtype=torch.float32) * 6.0
    cos, sin = torch.cos(angle), torch.sin(angle)
    freqs = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)[None, :, None]
    q_out, k_out = author_kernels.rms_rope_split_half(q, k, freqs, q_scale, k_scale, 1e-5, rotary)

    def law(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x = x.double()
        rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * scale.double()
        one, two, rest = rms[..., : rotary // 2], rms[..., rotary // 2 : rotary], rms[..., rotary:]
        c, s = cos[None, :, None].double(), sin[None, :, None].double()
        return torch.cat([one * c - two * s, one * s + two * c, rest], -1)

    errors = {"q": verify.error(torch, q_out, law(q, q_scale)), "k": verify.error(torch, k_out, law(k, k_scale))}
    verdict = verify.within(errors, prebuilt.BUDGET)
    return {**_verdict(verdict), "shapes": str(freqs.shape)}


def _big_gemm(name: str, index: int) -> Outcome:
    """The 8-bit GEMM at H3's largest: past 2^31 elements of its activation, sampled rows against
    the product in float64 (M x K wraps an int32 in upstream's CUTLASS, if anywhere)."""
    target = torch.device("cuda", index)
    ops = _library(index)
    fp8 = name == "fp8"
    if fp8 and not ops.cutlass_fp8:
        return {"passed": True, "detail": "not run: no wheel GEMM for fp8 on this card"}
    free = _released(index)
    if free < 12 << 30:
        return {"passed": True, "detail": f"not run: {free >> 30} GiB free cannot hold it"}
    g = torch.Generator(device=target).manual_seed(10)
    m, n, k = 392_000, 7168, 7168

    def data(*shape: int) -> torch.Tensor:
        raw = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device=target)
        # Signed, as real activations are: Hopper's fast-accumulation fp8 loses 3% at K 7168 when every
        # operand has one sign (run 29196) and 0.2% otherwise (run 29211). 0xF7 masks off NaN.
        return (raw & 0xF7).view(torch.float8_e4m3fn) if fp8 else raw.view(torch.int8)

    q, w = data(m, k), data(n, k)
    xs = torch.rand(m, generator=g, device=target, dtype=torch.float32).add_(0.5)
    scale = torch.rand(n, generator=g, device=target, dtype=torch.float32).add_(0.5)
    gemm = ops.fp8_gemm if fp8 else ops.int8_gemm
    out = gemm(q, xs, w, scale, None, torch.bfloat16, -1)
    rows = torch.tensor([0, 1, m // 2, m - 1], device=target)
    want = (q[rows].double() @ w.double().T) * xs[rows, None].double() * scale[None].double()
    error = verify.error(torch, out[rows], want)
    return {"passed": error <= prebuilt.BUDGET, "detail": f"{m}x{k}x{n}: sampled rows vs exact {error:.2e} (budget {prebuilt.BUDGET:g})"}


def _svdq_native(index: int) -> Outcome:
    return _verdict(svdq_linear.probe_native(torch, torch.device("cuda", index), _kernels._gemm))


def _kitchen(index: int, probe: "Callable[[ModuleType, int, ModuleType], verify.Verdict]") -> Outcome:
    found = int8_linear.kitchen_module(torch, index, True)
    if found is None:
        return {"passed": False, "detail": "comfy-kitchen has no build here"}
    return _verdict(probe(torch, index, found[0]))


def _decode(index: int) -> Outcome:
    device = measure_device(torch, index)
    report = conform(torch, launch_providers(), device, ("bfloat16", "float16", "float32"), release=cozy_runtime.__version__)
    failed = [
        f"{one.implementation}@{one.output_dtype}"
        for one in report.observations
        if not one.evidence.passed and not isinstance(one.evidence, UnmeasuredEvidence)
    ]
    return {"passed": not failed, "detail": f"{len(report.observations)} cells; failed {failed}"}


EXERCISES: dict[str, Callable[[int], Outcome]] = {
    "big:gemm-int8": lambda i: _big_gemm("int8", i),
    "big:gemm-fp8": lambda i: _big_gemm("fp8", i),
    "native:int8": lambda i: _format(i, prebuilt.INT8),
    "native:fp8": lambda i: _format(i, prebuilt.FP8),
    "native:w4a8": _w4a8,
    "native:norm": _norm,
    "native:svdq": _svdq_native,
    "kitchen:int8": lambda i: _kitchen(i, int8_linear._probe_kitchen),
    "kitchen:nvfp4": lambda i: _kitchen(i, nvfp4_linear._probe_kitchen),
    "decode:conform": _decode,
    "triton:cozy-attention-e4m3": _e4m3,
    "triton:cozy-sage2-fixed-kmean": _kmean,
    "triton:cozy-sage3-fixed-means": _sage3_fixed,
    "triton:cozy-sage3-group-mean": _group_mean,
    "triton:cozy-nvfp4": partial(_probed, nvfp4_linear),
    "triton:cozy-svdq-nvfp4": partial(_probed, svdq_linear),
    "triton:cozy-lora-bake": partial(_probed, lora_bake),
    "triton:cozy-counter-noise": partial(_probed_on, philox),
    "triton:cozy-ulysses-turn": partial(_probed_on, turn),
    "triton:cozy-lm-decode": _lm,
    "triton:cozy-moe-experts": _moe,
    "triton:cozy-bench-fused": _fused,
    # Opened beside its FA4 forwards, inside Sol's own route; its FA4 Sol attention cell launches it.
    "triton:cozy-sol-fa4-merge": lambda i: _attention("sol-attn", i),
}


def _exercise(name: str, index: int) -> Outcome:
    kind, _, kernel = name.partition(":")
    if kind == "attention":
        return _attention(kernel, index)
    if kind == "big" and name not in EXERCISES:
        return _not_fit(kernel, index, _big)
    if name not in EXERCISES:
        return {"passed": False, "detail": f"{name} has no exerciser: add one to cozy_runtime/internal/bench/kernels.py"}
    return EXERCISES[name](index)


def _off_request(work: Callable[[], Outcome]) -> Outcome:
    """On a thread of its own: a request never compiles or probes, so a kernel opened here is not
    the gate's request-path blocking, and each opens as `kernel_ready`'s background openers do."""
    found: Outcome = {}

    def body() -> None:
        try:
            with torch.no_grad(), kernel_ready.EXECUTION[0]():
                found.update(work())
        except Exception as exc:  # noqa: BLE001 - a kernel that raises is that cell's finding
            found.update({"passed": False, "detail": f"{type(exc).__name__}: {exc}"[:1500]})

    thread = threading.Thread(target=body, name="kernels-bench")
    thread.start()
    thread.join()
    return found


def inventory() -> Declared:
    """What this card claims, its jit functions, and each entry whose launcher reads its signature out of order. Raises
    on one of those: every launch of it sends its arguments one off (run 28135's Xid 31)."""
    device = measure_device(torch, 0)
    found = declared(arch.row(device.sm), device.sm)
    found["jits"] = launched_jits(_sources(), found["names"])
    found["gpus"] = torch.cuda.device_count()
    found["mismatches"] = _launcher_order()
    if found["mismatches"]:
        raise RuntimeError(f"kernels that launch wrong: {found['mismatches']}"[:3000])
    return found


def exercise(name: str, index: int, seed: int) -> dict[str, object]:
    """`name` on GPU `index` (1 is what a group's follower is), in this call's own RNG state. Raises when it fails."""
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"cuda:{index} of {torch.cuda.device_count()}")
    device = measure_device(torch, index)
    execution = list(kernel_ready.EXECUTION)
    if index:  # this process's execution stream is on GPU 0
        kernel_ready.EXECUTION[:] = [partial(torch.cuda.device, index)]
    try:
        with _recording() as (entries, jits), torch.random.fork_rng(devices=[index]):
            torch.manual_seed(seed)
            found = _off_request(partial(_exercise, name, index))
            _released(index)  # what this call made goes back, so the next call has the card
    finally:
        kernel_ready.EXECUTION[:] = execution
    # The verdict, the card, and what it launched (the gate's check.py reads `entries` and `jits`).
    checked: dict[str, object] = {
        "kernel": name,
        "card": device.name,
        "sm": device.sm,
        "driver": device.driver,
        "index": index,
        **found,
        "entries": sorted(entries),
        "jits": sorted(jits),
    }
    if not found.get("passed"):
        raise RuntimeError(f"kernel failed: {json.dumps(checked, sort_keys=True)}"[:3000])
    return checked
