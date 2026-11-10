"""The package contract at describe (ENFORCEMENT §1): every `.py` under a package's own import
roots, read as `ast` like the rest of the static reader, executing nothing of it.

A construct that harms the process a package shares is refused in one line, `file:line: <rule>:
<construct>: <the harm>`. Only harm is judged, never the shape of the code: loops, attention
calls, host reads and waits, threads and library decode are the author's. Names resolve through each
file's imports, its own package's re-exports and single-assignment aliases; an attribute of an
object (`self.vae.decode`) is not judged here: the executor's seal catches those calls."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cozy_runtime.internal.static_interface as static_interface
from cozy_runtime.internal.contract import WHY

if TYPE_CHECKING:
    from cozy_runtime.internal.static_interface import Module, Reader

CONSTRUCT_CHARS = 72


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
        torch.cuda.stream torch.cuda.streams torch.cuda.set_stream
        torch.cuda.CUDAGraph torch.cuda.graph torch.cuda.graphs torch.cuda.graph_pool_handle
        torch.cuda.make_graphed_callables torch.cuda.set_device torch.cuda.device
        torch.cuda.device_of torch.Stream torch.Event
        torch.accelerator.set_stream
        torch.accelerator.set_device_index torch.accelerator.set_device_idx
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
    **_names("internal_import", "tensorfs varena"),
}
#: Reads of capture state: they change nothing.
_READS = frozenset({"torch.cuda.CUDAGraph.get_currently_capturing_graph"})
#: `cozy_runtime`'s own machinery: every other module of it is the author's to import.
_INTERNAL = _words("internal graphs cli")
#: Draws that are not the global stream: a generator built from a seed.
_SEEDED = frozenset({"Random", "SystemRandom"})
_NUMPY_SEEDED = _words(
    """default_rng Generator RandomState SeedSequence BitGenerator PCG64 PCG64DXSM Philox MT19937
    SFC64"""
)
_ENVIRON_WRITES = frozenset({"setdefault", "update", "pop", "popitem", "clear", "__setitem__"})
_WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    code: str
    construct: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code}: {self.construct}: {WHY[self.code]}"


def check(reader: Reader, modules: Sequence[Module]) -> None:
    """Refuse the package (`StaticRefusal` `package_contract`) on any construct that harms the
    process it shares: the first on the first line, every other after it."""
    files = [_File(reader, module) for module in _files(reader, modules)]
    every = {file.module.name: file.aliases for file in files}
    found = [finding for file in files for finding in file.findings(every)]
    if found:
        more = f" (and {len(found) - 1} more)" if len(found) > 1 else ""
        lines = [f"{found[0]}{more}", *(f"  {f}" for f in found[1:])]
        raise static_interface.StaticRefusal("\n".join(lines), code="package_contract")


# ------------------------------------------------------------------------------ the files


def _files(reader: Reader, modules: Sequence[Module]) -> list[Module]:
    """Every module under the package's own import roots its reachable modules import: its own
    tree, never a dependency's (`Reader.own`, in its project roots)."""
    reached = {m.name.partition(".")[0] for m in modules if not m.trusted}
    tops = sorted(reached & reader.own - {"cozy_runtime"})
    loaded = {m.path.resolve(): m for m in reader.modules.values()}
    seen: dict[Path, Module] = {}
    for top in tops:
        for root in reader.project_roots:
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
    assigned: dict[str, list[ast.AST]] = {}
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
                assigned.setdefault(name, []).append(node)
    for name, values in assigned.items():
        if name in out or len(values) != 1 or not isinstance(values[0], (ast.Name, ast.Attribute)):
            continue
        dotted = _dotted(values[0], out)
        if dotted is not None:
            out[name] = dotted
    return out


def _bound(
    node: ast.For
    | ast.comprehension
    | ast.With
    | ast.FunctionDef
    | ast.AsyncFunctionDef
    | ast.Lambda,
) -> Iterator[str]:
    """Names a loop, `with`, comprehension or function binds (its arguments)."""
    targets: list[ast.AST] = []
    if isinstance(node, (ast.For, ast.comprehension)):
        targets = [node.target]
    elif isinstance(node, ast.With):
        targets = [item.optional_vars for item in node.items if item.optional_vars is not None]
    else:
        args = node.args
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
            internal = private or (len(parts) > 1 and parts[1] in _INTERNAL)
            return "internal_import" if internal else None
        if parts[0] == "random" and len(parts) > 1 and parts[1] not in _SEEDED:
            return "global_rng"
        if parts[:2] == ["numpy", "random"] and len(parts) > 2 and parts[2] not in _NUMPY_SEEDED:
            return "global_rng"
        if dotted in _READS:
            return None
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
            if func.attr in ("wait_stream", "wait_event", "record_event") and (
                isinstance(func.value, ast.Call)
                and self._resolve(func.value.func)
                in ("torch.cuda.current_stream", "torch.cuda.default_stream")
            ):
                self._add(node, "own_stream", f"current_stream().{func.attr}()")
            if func.attr in _ENVIRON_WRITES and self._resolve(func.value) == "os.environ":
                self._add(node, "env_write", f"os.environ.{func.attr}(...)")
        if dotted == "torch.Generator" and self._device_generator(node):
            self._add(node, "device_rng", "torch.Generator(device=<not cpu>)")
        name = func.id if isinstance(func, ast.Name) and func.id not in self.aliases else None
        opens = name == "open" or dotted in ("io.open", "builtins.open")
        if opens and node.args and (suffix := _literal_suffix(node.args[0])) is not None:
            self._add(node, "weights_io", f"open(...{suffix})")
        if name == "setattr" and len(node.args) > 1:
            label = node.args[1]
            if isinstance(label, ast.Constant) and label.value == "PURE_WINDOWS":
                self._add(node, "internal_import", "PURE_WINDOWS")

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

    def _add(self, node: ast.AST, code: str, construct: str) -> None:
        line = getattr(node, "lineno", 0)
        if len(construct) > CONSTRUCT_CHARS:
            construct = construct[:CONSTRUCT_CHARS] + "…"
        self.found.setdefault((line, code), Finding(self.path, line, code, construct))


def _unpacked(target: ast.expr) -> Iterator[ast.expr]:
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _unpacked(element)
    else:
        yield target


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
