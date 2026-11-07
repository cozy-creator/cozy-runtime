"""The package contract at describe (ENFORCEMENT §1): every `.py` under a package's own import
roots, read as `ast` like the rest of the static reader, executing nothing of it.

A construct the contract gives the Runtime alone is refused in one line, `file:line: <rule>:
<construct>: use <API>`; a rule turns on in the release that lands the API it names. Names
resolve through each file's imports, its own package's re-exports and single-assignment aliases;
an attribute of an object (`self.vae.decode`) is not judged here: the executor's seal catches
those calls."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cozy_runtime.internal.static_interface as static_interface

if TYPE_CHECKING:
    from cozy_runtime.internal.static_interface import Module, Reader

CONSTRUCT_CHARS = 72


@dataclass(frozen=True, slots=True)
class Rule:
    code: str
    use: str


RULES = {
    rule.code: rule
    for rule in (
        Rule("oom_catch", "author.split(fn, x, Cut(...)): varena picks a level that fits"),
        Rule("room_query", "author.split or author.retained: a package never asks for room"),
        Rule("allocator", "nothing: varena owns the allocator and reclaims"),
        Rule("env_write", "Config or Settings: TensorD seals the environment"),
        Rule(
            "global_flag",
            "the Runtime's numerics policy, hold_dtype, @uses_components(deterministic=True)",
        ),
        Rule("own_stream", "denoise_loop, uncaptured, graph_keys: the Runtime owns streams"),
        Rule("compile", "author.kernels.TritonKernels, prebuilt"),
        Rule("weights_io", "loader.construct or loader.load, loader.assets.read or .open"),
        Rule("own_decode", "lm.generate (allowed=lm.Constraint for a constrained decode)"),
        Rule("global_rng", "rng.noise, rng.generator(seed), view.generator"),
        Rule("process", "concurrently, spread or an invocable child call"),
        Rule("thread", "author.ThreadPoolExecutor"),
        Rule("internal_import", "the public names (CONTRACT.md); a package split is Final"),
        Rule("host_sync_in_step", "a step label, or compute it before the loop"),
        Rule(
            "attention",
            "cozy_runtime.attention.Attention(heads, head_dim, dtype, causal=) or a Diffusers "
            "attention module",
        ),
    )
}


def _words(names: str) -> frozenset[str]:
    return frozenset(names.split())


def _names(rule: str, names: str) -> dict[str, str]:
    """`names` (whitespace-separated) -> `rule`; `cuda:x` is `torch.cuda.x` and
    `torch.cuda.memory.x`, `os:x` is `os.x`."""
    out: dict[str, str] = {}
    for name in _words(names):
        prefix, _, rest = name.rpartition(":")
        spelled = {"cuda": ("torch.cuda.", "torch.cuda.memory."), "os": ("os.",), "": ("",)}
        out |= dict.fromkeys((f"{p}{rest}" for p in spelled[prefix]), rule)
    return out


_EXEC = " ".join(f"os:{k}{s}" for k in ("exec", "spawn") for s in _words("l le lp lpe v ve vp vpe"))
#: Dotted names, and every name under them, by rule.
_NAMED: dict[str, str] = {
    **_names("oom_catch", "torch.OutOfMemoryError cuda:OutOfMemoryError"),
    **_names(
        "room_query",
        """cuda:mem_get_info cuda:memory_allocated cuda:memory_reserved cuda:memory_cached
        cuda:memory_stats cuda:memory_stats_as_nested_dict cuda:memory_summary cuda:memory_snapshot
        cuda:max_memory_allocated cuda:max_memory_reserved cuda:max_memory_cached
        cuda:list_gpu_processes torch.mps.current_allocated_memory torch.mps.driver_allocated_memory
        torch.mps.recommended_max_memory pynvml nvidia_ml_py nvsmi py3nvml""",
    ),
    **_names(
        "allocator",
        """cuda:empty_cache cuda:reset_peak_memory_stats cuda:reset_max_memory_allocated
        cuda:reset_max_memory_cached cuda:reset_accumulated_memory_stats
        cuda:set_per_process_memory_fraction cuda:caching_allocator_alloc
        cuda:caching_allocator_delete cuda:change_current_allocator cuda:CUDAPluggableAllocator
        cuda:MemPool cuda:use_mem_pool torch.cuda.memory._set_allocator_settings
        torch.mps.empty_cache torch.mps.set_per_process_memory_fraction""",
    ),
    **_names("env_write", "os:putenv os:unsetenv"),
    **_names(
        "global_flag",
        """torch.set_float32_matmul_precision torch.set_default_dtype torch.set_default_device
        torch.set_default_tensor_type torch.use_deterministic_algorithms
        torch.set_deterministic_debug_mode torch.set_flush_denormal
        torch.backends.cuda.enable_flash_sdp torch.backends.cuda.enable_mem_efficient_sdp
        torch.backends.cuda.enable_math_sdp torch.backends.cuda.enable_cudnn_sdp
        torch.backends.cuda.sdp_kernel torch.backends.cuda.preferred_linalg_library
        torch.backends.cuda.preferred_blas_library torch.backends.cudnn.flags
        torch.backends.mkldnn.flags torch.nn.attention.sdpa_kernel""",
    ),
    **_names(
        "own_stream",
        """torch.cuda.Stream torch.cuda.Event torch.cuda.ExternalStream torch.cuda.StreamContext
        torch.cuda.stream torch.cuda.streams torch.cuda.set_stream torch.cuda.synchronize
        torch.cuda.CUDAGraph torch.cuda.graph torch.cuda.graphs torch.cuda.graph_pool_handle
        torch.cuda.make_graphed_callables torch.cuda.is_current_stream_capturing
        torch.cuda.set_device torch.cuda.device torch.cuda.device_of torch.Stream torch.Event
        torch.accelerator.set_stream torch.accelerator.synchronize
        torch.accelerator.set_device_index torch.accelerator.set_device_idx torch.mps.synchronize
        torch.mps.Event cuda.bindings cuda.cuda cuda.cudart cuda.core pycuda""",
    ),
    **_names("compile", "torch.compile torch._dynamo torch._inductor triton.autotune"),
    **_names(
        "weights_io",
        """huggingface_hub safetensors.safe_open safetensors.torch.load_file
        safetensors.numpy.load_file torch.load torch.hub numpy.load""",
    ),
    **_names(
        "global_rng",
        """torch.manual_seed torch.seed torch.default_generator torch.set_rng_state torch.random
        torch.cuda.manual_seed torch.cuda.manual_seed_all torch.cuda.seed torch.cuda.seed_all
        torch.cuda.set_rng_state torch.cuda.set_rng_state_all torch.cuda.default_generators
        torch.cuda.random numpy.random.seed""",
    ),
    **_names(
        "process",
        f"""subprocess multiprocessing torch.multiprocessing pty concurrent.futures.process
        concurrent.futures.ProcessPoolExecutor asyncio.create_subprocess_exec
        asyncio.create_subprocess_shell os:fork os:forkpty os:system os:popen os:posix_spawn
        os:posix_spawnp {_EXEC}""",
    ),
    **_names(
        "thread",
        """threading.Thread threading.Timer concurrent.futures.ThreadPoolExecutor
        concurrent.futures.thread _thread.start_new_thread _thread.start_new
        multiprocessing.pool.ThreadPool multiprocessing.dummy""",
    ),
    **_names("internal_import", "tensorfs varena"),
    **_names(
        "attention",
        """torch.nn.functional.scaled_dot_product_attention
        torch._C._nn.scaled_dot_product_attention torch.nn.attention.flex_attention flash_attn
        flash_attn_interface sageattention xformers""",
    ),
}
#: `cozy_runtime`'s modules a package may import (CONTRACT.md).
_PUBLIC = _words(
    "author lm vlm rng tiles stepcache lora attention moe fusion rows schedule weights_store derive"
)
#: Draws that are not the global stream: a generator built from a seed.
_SEEDED = frozenset({"Random", "SystemRandom"})
_NUMPY_SEEDED = _words(
    """default_rng Generator RandomState SeedSequence BitGenerator PCG64 PCG64DXSM Philox MT19937
    SFC64"""
)
_CACHES = _words(
    """DynamicCache StaticCache HybridCache HybridChunkedCache SlidingWindowCache OffloadedCache
    OffloadedStaticCache QuantizedCache EncoderDecoderCache"""
)
_ENVIRON_WRITES = frozenset({"setdefault", "update", "pop", "popitem", "clear", "__setitem__"})
_WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
_HOST_READS = frozenset({"item", "tolist", "cpu", "numpy"})
_LOOP = frozenset({"cozy_runtime.author.denoise_loop", "cozy_runtime.author._loop.denoise_loop"})


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    rule: Rule
    construct: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule.code}: {self.construct}: use {self.rule.use}"


def check(reader: Reader, modules: Sequence[Module]) -> None:
    """Refuse the package (`StaticRefusal` `package_contract`) on any construct the contract
    gives the Runtime: the first on the first line, every other after it."""
    files = [_File(reader, module) for module in _files(reader, modules)]
    every = {file.module.name: file.aliases for file in files}
    found = [finding for file in files for finding in file.findings(every)]
    if found:
        more = f" (and {len(found) - 1} more)" if len(found) > 1 else ""
        lines = [f"{found[0]}{more}", *(f"  {f}" for f in found[1:])]
        raise static_interface.StaticRefusal("\n".join(lines), code="package_contract")


# ------------------------------------------------------------------------------ the files


def _files(reader: Reader, modules: Sequence[Module]) -> list[Module]:
    """Every module under the import roots of the package's own reachable modules."""
    tops = sorted({m.name.partition(".")[0] for m in modules if not m.trusted} - {"cozy_runtime"})
    loaded = {m.path.resolve(): m for m in reader.modules.values()}
    seen: dict[Path, Module] = {}
    for top in tops:
        for root in reader.roots:
            single = root / f"{top}.py"
            paths = sorted((root / top).rglob("*.py")) if (root / top).is_dir() else []
            for path in paths or ([single] if single.is_file() else []):
                resolved = path.resolve()
                if resolved in seen or "__pycache__" in path.parts:
                    continue
                parts = path.relative_to(root).with_suffix("").parts
                package = parts[-1] == "__init__"
                name = ".".join(parts[:-1] if package else parts)
                module = loaded.get(resolved) or _parsed(name, path, package)
                if module is not None:
                    seen[resolved] = module
    return list(seen.values())


def _parsed(name: str, path: Path, package: bool) -> Module | None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        return static_interface.Module(name, path, source, tree, package, False)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None  # a file Python cannot import holds nothing that runs


def _indexed(tree: ast.Module) -> tuple[list[ast.AST], dict[int, ast.AST]]:
    """Every node, breadth first, and each one's parent: the one walk of the file."""
    nodes: list[ast.AST] = [tree]
    parents: dict[int, ast.AST] = {}
    for node in nodes:  # grows as it goes
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
            nodes.append(child)
    return nodes, parents


def _type_only(nodes: Sequence[ast.AST]) -> set[int]:
    """Nodes under `if TYPE_CHECKING:`: never executed."""
    skipped: set[int] = set()
    for node in nodes:
        if isinstance(node, ast.If):
            test = node.test
            name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", None)
            if name == "TYPE_CHECKING":
                for stmt in node.body:
                    skipped.update(id(n) for n in ast.walk(stmt))
    return skipped


def _aliases(reader: Reader, module: Module, nodes: Sequence[ast.AST]) -> dict[str, str]:
    """Local name -> dotted name: every import in the file, then a name assigned once to a
    dotted name (`cudnn = torch.backends.cudnn`)."""
    out: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.partition(".")[0]
                out[alias.asname or top] = alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom):
            base = reader._import_base(module, node)
            for alias in node.names:
                if alias.name != "*":
                    out[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
    assigned: dict[str, list[ast.expr]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, []).append(node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and isinstance(
            node.target, ast.Name
        ):
            assigned.setdefault(node.target.id, []).append(node.value or node.target)
        elif isinstance(
            node,
            (
                ast.For,
                ast.comprehension,
                ast.With,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
            ),
        ):
            for name in _bound(node):
                assigned.setdefault(name, []).append(node)  # type: ignore[arg-type]
    for name, values in assigned.items():
        if name in out or len(values) != 1 or not isinstance(values[0], (ast.Name, ast.Attribute)):
            continue
        dotted = _dotted(values[0], out)
        if dotted is not None:
            out[name] = dotted
    return out


def _bound(node: ast.AST) -> Iterator[str]:
    """Names a loop, `with`, comprehension or function binds (its arguments)."""
    targets: list[ast.AST] = []
    if isinstance(node, (ast.For, ast.comprehension)):
        targets = [node.target]
    elif isinstance(node, ast.With):
        targets = [item.optional_vars for item in node.items if item.optional_vars is not None]
    else:
        args = node.args  # type: ignore[attr-defined]
        yield from (a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs))
        yield from (a.arg for a in (args.vararg, args.kwarg) if a is not None)
    for target in targets:
        yield from (n.id for n in ast.walk(target) if isinstance(n, ast.Name))


def _dotted(node: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return None if base is None else f"{base}.{node.attr}"
    return None


# ------------------------------------------------------------------------------- the rules


class _File:
    def __init__(self, reader: Reader, module: Module):
        self.reader, self.module = reader, module
        self.nodes, self.parents = _indexed(module.tree)
        self.aliases = _aliases(reader, module, self.nodes)
        self.path = reader.spell(module.path)
        self.skipped = _type_only(self.nodes)
        self.found: dict[tuple[int, str], Finding] = {}

    def findings(self, every: dict[str, dict[str, str]]) -> list[Finding]:
        """`every`: each of the package's modules' aliases, for its re-exports."""
        self.every = every
        self.own = {name.partition(".")[0] for name in every}
        for node in self.nodes:
            if id(node) in self.skipped:
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._named(node, alias.name, f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                base = self.reader._import_base(self.module, node)
                for alias in node.names:
                    if base is not None:
                        self._named(
                            node, f"{base}.{alias.name}", f"from {base} import {alias.name}"
                        )
            elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
                if not isinstance(self.parents.get(id(node)), ast.Attribute):
                    self._load(node)
                if isinstance(node, ast.Attribute) and node.attr == "total_memory":
                    self._add(node, "room_query", ".total_memory")
            elif isinstance(node, ast.Call):
                self._call(node)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                self._store(node)
            elif isinstance(node, ast.ClassDef):
                self._class(node)
            elif isinstance(node, ast.ExceptHandler):
                self._handler(node)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "nvidia-smi" in node.value
                and not isinstance(self.parents.get(id(node)), ast.Expr)
            ):
                self._add(node, "room_query", '"nvidia-smi"')
        return sorted(self.found.values(), key=lambda f: f.line)

    # ------------------------------------------------------------------ resolution

    def _resolve(self, node: ast.expr) -> str | None:
        dotted = _dotted(node, self.aliases)
        for _ in range(8):  # through this package's own re-exports
            if dotted is None:
                return None
            parts = dotted.split(".")
            for cut in range(len(parts) - 1, 0, -1):
                owner = self.every.get(".".join(parts[:cut]))
                if owner is not None and parts[cut] in owner:
                    dotted = ".".join([owner[parts[cut]], *parts[cut + 1 :]])
                    break
            else:
                return dotted
        return dotted

    def _rule(self, dotted: str) -> str | None:
        parts = dotted.split(".")
        if parts[0] in self.own:
            return None
        if parts[0] == "cozy_runtime":
            private = any(p.startswith("_") and not p.startswith("__") for p in parts[1:])
            named = len(parts) > 1 and not parts[1].startswith("__")
            return "internal_import" if private or (named and parts[1] not in _PUBLIC) else None
        if parts[0] == "random" and len(parts) > 1 and parts[1] not in _SEEDED:
            return "global_rng"
        if parts[:2] == ["numpy", "random"] and len(parts) > 2 and parts[2] not in _NUMPY_SEEDED:
            return "global_rng"
        if parts[0] in ("transformers", "diffusers") and parts[-1] in (
            "generate",
            "generate_captions",
        ):
            return "own_decode"
        for cut in range(len(parts), 0, -1):
            if (rule := _NAMED.get(".".join(parts[:cut]))) is not None:
                return rule
        return None

    # ------------------------------------------------------------------- visitors

    def _named(self, node: ast.AST, dotted: str, said: str) -> None:
        if (rule := self._rule(dotted)) is not None:
            self._add(node, rule, said)

    def _load(self, node: ast.expr) -> None:
        dotted = self._resolve(node)
        if dotted is not None:
            self._named(node, dotted, dotted)

    def _call(self, node: ast.Call) -> None:
        func = node.func
        dotted = self._resolve(func)
        if isinstance(func, ast.Attribute):
            if func.attr in ("from_pretrained", "from_single_file"):
                self._add(node, "weights_io", f".{func.attr}(...)")
            if func.attr in ("synchronize", "wait_stream", "wait_event", "record_event") and (
                isinstance(func.value, ast.Call)
                and self._resolve(func.value.func)
                in ("torch.cuda.current_stream", "torch.cuda.default_stream")
            ):
                self._add(node, "own_stream", f"current_stream().{func.attr}()")
            if func.attr in _ENVIRON_WRITES and self._resolve(func.value) == "os.environ":
                self._add(node, "env_write", f"os.environ.{func.attr}(...)")
        if dotted is not None:
            if dotted.startswith("transformers.") and dotted.rsplit(".", 1)[-1] in _CACHES:
                self._add(node, "own_decode", f"{dotted.rsplit('.', 1)[-1]}(...)")
            if dotted == "torch.Generator" and self._device_generator(node):
                self._add(node, "global_rng", "torch.Generator(device=<not cpu>)")
            if dotted in _LOOP:
                self._loop(node)
        name = func.id if isinstance(func, ast.Name) and func.id not in self.aliases else None
        opens = name == "open" or dotted in ("io.open", "builtins.open")
        if opens and node.args and (suffix := _literal_suffix(node.args[0])) is not None:
            self._add(node, "weights_io", f"open(...{suffix})")
        if name == "setattr" and len(node.args) > 1:
            label = node.args[1]
            if isinstance(label, ast.Constant) and label.value == "PURE_WINDOWS":
                self._add(node, "internal_import", "PURE_WINDOWS")
        for keyword in node.keywords:
            value = keyword.value
            if (
                keyword.arg == "use_cache"
                and isinstance(value, ast.Constant)
                and value.value is True
            ):
                self._add(node, "own_decode", "use_cache=True")

    def _store(self, node: ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete) -> None:
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        value = getattr(node, "value", None)
        for target in (t for each in targets for t in _unpacked(each)):
            if isinstance(target, ast.Subscript) and self._resolve(target.value) == "os.environ":
                self._add(target, "env_write", "os.environ[...]")
            elif isinstance(target, ast.Attribute):
                if target.attr == "PURE_WINDOWS":
                    self._add(target, "internal_import", "PURE_WINDOWS")
                dotted = self._resolve(target)
                if dotted is None:
                    continue
                if dotted == "os.environ":
                    self._add(target, "env_write", "os.environ")
                elif dotted == "torch.backends.cudnn.benchmark" and (
                    isinstance(value, ast.Constant) and value.value is True
                ):
                    self._add(target, "compile", "torch.backends.cudnn.benchmark = True")
                elif dotted.startswith("torch.backends."):
                    self._add(target, "global_flag", f"{dotted} = ...")
            elif (
                isinstance(target, ast.Name)
                and target.id == "PURE_WINDOWS"
                and isinstance(self.parents.get(id(node)), ast.ClassDef)
            ):
                self._add(target, "internal_import", "PURE_WINDOWS")

    def _class(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            dotted = self._resolve(base) or ""
            cache = dotted.rsplit(".", 1)[-1]
            if dotted.startswith("transformers.") and cache in _CACHES:
                self._add(node, "own_decode", f"class {node.name}({cache})")

    def _handler(self, node: ast.ExceptHandler) -> None:
        for inner in (n for stmt in node.body for n in ast.walk(stmt)):
            if (
                isinstance(inner, ast.Constant)
                and isinstance(inner.value, str)
                and not isinstance(self.parents.get(id(inner)), ast.Expr)
                and "out of memory" in inner.value.lower().replace("outofmemory", "out of memory")
            ):
                self._add(node, "oom_catch", f'except ...: "{inner.value.strip()[:40]}"')
                return

    def _device_generator(self, node: ast.Call) -> bool:
        device = node.args[0] if node.args else None
        for keyword in node.keywords:
            if keyword.arg == "device":
                device = keyword.value
        if device is None:
            return False
        if isinstance(device, ast.Call) and self._resolve(device.func) == "torch.device":
            device = device.args[0] if device.args else device
        return not (isinstance(device, ast.Constant) and device.value == "cpu")

    # ------------------------------------------------------------- the step loop

    def _loop(self, node: ast.Call) -> None:
        """`denoise_loop(step, ..., before=, after=)`: the step and its hooks read no device
        value on the host."""
        given = [node.args[0]] if node.args else []
        given += [k.value for k in node.keywords if k.arg in ("step", "before", "after")]
        for argument in given:
            body = self._function(node, argument)
            if body is None:
                continue
            for inner in ast.walk(body):
                if not isinstance(inner, ast.Call):
                    continue
                func = inner.func
                if isinstance(func, ast.Attribute) and func.attr in _HOST_READS and not inner.args:
                    self._add(inner, "host_sync_in_step", f".{func.attr}()")
                elif (
                    isinstance(func, ast.Name)
                    and func.id in ("bool", "int", "float")
                    and func.id not in self.aliases
                    and len(inner.args) == 1
                    and isinstance(inner.args[0], ast.Call)
                    and isinstance(inner.args[0].func, ast.Attribute)
                ):
                    self._add(
                        inner,
                        "host_sync_in_step",
                        f"{func.id}(<tensor>.{inner.args[0].func.attr}())",
                    )

    def _function(self, at: ast.AST, argument: ast.expr) -> ast.AST | None:
        if isinstance(argument, ast.Lambda):
            return argument
        if isinstance(argument, ast.Call) and argument.args:
            if self._resolve(argument.func) == "functools.partial":
                return self._function(at, argument.args[0])
            return None
        method = isinstance(argument, ast.Attribute) and isinstance(argument.value, ast.Name)
        if not (method or isinstance(argument, ast.Name)):
            return None
        scope: ast.AST | None = at
        while (scope := self.parents.get(id(scope))) is not None:
            if method and isinstance(scope, ast.ClassDef):
                assert isinstance(argument, ast.Attribute)
                return _defined(scope.body, argument.attr)
            if not method and isinstance(
                scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
            ):
                assert isinstance(argument, ast.Name)
                if (found := _defined(list(ast.walk(scope))[1:], argument.id)) is not None:
                    return found
        return None

    def _add(self, node: ast.AST, code: str, construct: str) -> None:
        line = getattr(node, "lineno", 0)
        if len(construct) > CONSTRUCT_CHARS:
            construct = construct[:CONSTRUCT_CHARS] + "…"
        self.found.setdefault((line, code), Finding(self.path, line, RULES[code], construct))


def _unpacked(target: ast.expr) -> Iterator[ast.expr]:
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _unpacked(element)
    else:
        yield target


def _defined(nodes: Sequence[ast.AST], name: str) -> ast.AST | None:
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _literal_suffix(node: ast.expr) -> str | None:
    """The weight suffix a path literal ends in: `"x.pt"`, `f"{d}/x.pt"`, `d + ".pt"`."""
    tail = node
    if isinstance(node, ast.JoinedStr) and node.values:
        tail = node.values[-1]
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        tail = node.right
    if isinstance(tail, ast.Constant) and isinstance(tail.value, str):
        return next((s for s in _WEIGHT_FILES if tail.value.endswith(s)), None)
    return None
