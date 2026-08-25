#!/usr/bin/env python3
"""Architecture fences (boundaries.md, cozy-runtime.md §7). Not a test: a dependency scan.

Twelve fences, each provable red by planting the violation (`scripts/redarm.py`):
  forbidden-dependency · canonical-numbers · torch-free-import · package-imports
  env-authority · single-config · no-interactive · excluded-keywords
  no-identifiers-in-endpoint-code · one-attempt-kernel · timing-decides-nothing
  endpoint-owns-no-residency
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cozy_runtime"

# boundaries.md: forbidden Cozy dependencies for cozy-runtime core + the author subpackage.
DENY_TOKENS = (
    "varena",
    "tensorhub",
    "cozy_creator",
    "cozy-creator",
    "openrails",
    "runpod",
    "boto3",
    "google.cloud",
    "shared_config",
)

# Per-package import fences. Values are forbidden module prefixes.
PACKAGE_FENCES: dict[str, tuple[str, ...]] = {
    "author": ("cozy_runtime.internal", "cozy_runtime.cli", "worker_protocol", "tensorfs"),
    "internal": ("cozy_runtime.cli",),
    "cli": (),
}
# The author surface is import-fenced to stdlib + schema/typing libraries only.
AUTHOR_ALLOWED_THIRD_PARTY = ("msgspec",)

ENV_AUTHORITY = SRC / "internal" / "config.py"
CONFIG_SYMBOLS = ("RuntimeConfig", "read_config")
CONFIG_NAMES = ("CONFIG", "SETTINGS")


def py_files() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def module_of(path: pathlib.Path) -> str:
    return ".".join(("cozy_runtime", *path.relative_to(SRC).with_suffix("").parts))


def imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def denied(name: str) -> str | None:
    """The deny token this module/requirement name matches, if any. Names, never prose."""
    norm = name.strip().lower().replace("-", "_")
    for token in DENY_TOKENS:
        t = token.replace("-", "_")
        if norm == t or norm.startswith(t + "."):
            return token
    return None


def declared_requirements() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data.get("project", {})
    reqs: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        reqs += list(group)
    for group in data.get("dependency-groups", {}).values():
        reqs += [g for g in group if isinstance(g, str)]
    reqs += list(data.get("tool", {}).get("uv", {}).get("sources", {}))
    return [re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0] for r in reqs]


def fence_forbidden_dependency() -> list[str]:
    bad: list[str] = []
    for req in declared_requirements():
        if token := denied(req):
            bad.append(f"pyproject.toml: forbidden Cozy dependency {token!r} declared as {req!r}")
    for path in py_files():
        for name in imports(ast.parse(path.read_text())):
            if token := denied(name):
                bad.append(f"{path.relative_to(ROOT)}: forbidden import {name!r} ({token})")
    return bad


def fence_package_imports() -> list[str]:
    bad: list[str] = []
    stdlib = sys.stdlib_module_names
    for path in py_files():
        parts = path.relative_to(SRC).parts
        pkg = parts[0] if len(parts) > 1 else ""
        forbidden = PACKAGE_FENCES.get(pkg, ())
        for name in imports(ast.parse(path.read_text())):
            rel = path.relative_to(ROOT)
            if any(name == f or name.startswith(f + ".") for f in forbidden):
                bad.append(f"{rel}: {module_of(path)} may not import {name}")
            root = name.split(".")[0]
            if (
                pkg == "author"
                and root not in stdlib
                and root != "cozy_runtime"
                and root not in AUTHOR_ALLOWED_THIRD_PARTY
            ):
                bad.append(f"{rel}: author surface may not import third-party {name}")
    return bad


def fence_env_authority() -> list[str]:
    bad: list[str] = []
    for path in py_files():
        if path == ENV_AUTHORITY:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if re.search(r"\bos\.(environ|getenv)\b|\bgetenv\(|\benviron\[", line):
                rel = path.relative_to(ROOT)
                bad.append(f"{rel}:{i}: environment read outside the config authority")
    return bad


def fence_single_config() -> list[str]:
    bad: list[str] = []
    for path in py_files():
        if path == ENV_AUTHORITY:
            continue
        tree = ast.parse(path.read_text())
        rel = path.relative_to(ROOT)
        for node in tree.body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in CONFIG_SYMBOLS:
                bad.append(f"{rel}:{node.lineno}: second config authority {node.name!r}")
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.lstrip("_").upper() in CONFIG_NAMES:
                        bad.append(f"{rel}:{node.lineno}: module-level config singleton {t.id!r}")
    return bad


def fence_no_interactive() -> list[str]:
    bad: list[str] = []
    for path in py_files():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if re.search(r"(?<![\w.])input\(|getpass", line):
                rel = path.relative_to(ROOT)
                bad.append(f"{rel}:{i}: interactive prompt — anything that would ask is a flag")
    return bad


def fence_excluded_keywords() -> list[str]:
    """No author-callable may take a device, dtype, path, offload, pin, eviction, compile or
    source-converter keyword: every one names a decision the runtime owns (§1.1/§7).

    Dunder and private callables are the RUNTIME's own plumbing (the runtime hands the
    Loader its device); the fence covers exactly what author code can call.
    """
    from cozy_runtime.author._model import EXCLUDED_KEYWORDS

    # `path=` stays out of the SCAN: `out.save_file(path)` is a legitimate author surface
    # (a file the handler produced), and the model-method clock still refuses it in-band.
    # `author/fakes.py` supplies the RUNTIME's side of the contract, so it is exempt too.
    fenced = {k: v for k, v in EXCLUDED_KEYWORDS.items() if k != "path"}
    bad: list[str] = []
    for path in py_files():
        if path.relative_to(SRC).parts[0] != "author" or path.name == "fakes.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if arg.arg in fenced:
                    bad.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: {node.name}({arg.arg}=) — "
                        f"{fenced[arg.arg]}"
                    )
    return bad


#: An endpoint module is one that builds an `App()` — the file `describe` imports.
_MEDIA_PREFIXES = ("image/", "video/", "audio/", "text/", "application/", "model/")
_IDENTIFIER = re.compile(r"^(hf|civitai|comfy|tensorhub):|^[a-z0-9][\w.-]*/[\w.-]+(@[\w.-]+)?$")
#: A versioned FORMAT id (`adaln-tables/1`, `ggml-q4-k/v1`) is a schema name, not an artifact.
_FORMAT_ID = re.compile(r"^[\w.-]+/v?\d+$")


def endpoint_modules() -> list[pathlib.Path]:
    modules = []
    for path in sorted((ROOT / "examples").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "App":
                modules.append(path)
                break
    return modules


def endpoint_and_model_code() -> list[pathlib.Path]:
    """Every module an ENDPOINT AUTHOR writes: the entrypoint modules and their models.

    `corpus/` counts — those are real endpoints against real weights, and the claim cr-008b
    makes is about what an endpoint contains, not about what a weightless example contains.
    The vendored venv is not author code and is excluded by path.
    """
    paths = list(endpoint_modules())
    for base in ("corpus/pipeline", "corpus/endpoint"):
        paths += sorted((ROOT / base).glob("*.py"))
    for name in ("sdxl_real.py", "sdxl_pipeline.py", "sdxl.py", "h3.py"):
        candidate = ROOT / "corpus" / name
        if candidate.is_file():
            paths.append(candidate)
    return paths


#: Residency vocabulary. An endpoint that can name one of these can move bytes, and §1.1's
#: whole claim is that it cannot — the declaration is `@uses_components` and there is no
#: second surface. `to`/`cuda` are deliberately ABSENT: casting a request tensor's dtype or
#: placing a request input is ordinary module code, and the excluded-keyword fence already
#: covers the model METHOD surface where placement would be a contract violation.
RESIDENCY_NAMES = (
    "to_empty",
    "empty_cache",
    "pin_memory",
    "pinned_memory",
    "cpu_offload",
    "enable_model_cpu_offload",
    "enable_sequential_cpu_offload",
    "reset_peak_memory_stats",
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "set_per_process_memory_fraction",
    "evict",
    "materialize",
    "register_backend",
)


def fence_endpoint_owns_no_residency() -> list[str]:
    """cr-008b's done-when, as a scan: an endpoint serves with ZERO memory-management calls.

    Not because its author is disciplined — because the vocabulary is not reachable from
    endpoint code. The runtime admits and materializes a declared component set before
    method entry, event-fences it, and makes it evictable after; an endpoint that could
    also do those things would be a second policy, and two policies is the v1 defect.
    """
    bad: list[str] = []
    for path in endpoint_and_model_code():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            reached, line = "", 0
            if isinstance(node, ast.Attribute) and node.attr in RESIDENCY_NAMES:
                reached, line = node.attr, node.lineno
            elif isinstance(node, ast.Name) and node.id in RESIDENCY_NAMES:
                reached, line = node.id, node.lineno
            if reached:
                bad.append(
                    f"{path.relative_to(ROOT)}:{line}: endpoint code names {reached!r} — "
                    "residency is the runtime's, declared with @uses_components and "
                    "nothing else; an endpoint that can move bytes is a second policy"
                )
    return bad


def fence_no_identifiers() -> list[str]:
    """HARD RULE (§1.1): author code never names an artifact identifier of any kind — no
    repo, release, checkpoint or adapter ref, in bindings or in string literals.

    Code states capability; SELECTION is a binding (endpoint.toml, `--model`, the deploy
    binding). Identity is exclusively deployment-side, so branching on one is unspellable.
    """
    bad: list[str] = []
    for path in endpoint_modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            text = node.value.strip()
            if text.startswith(_MEDIA_PREFIXES) or "://" in text or text.startswith("/"):
                continue
            if _FORMAT_ID.match(text):
                continue
            if _IDENTIFIER.match(text):
                bad.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: artifact identifier {text!r} in "
                    "endpoint code — selection is a binding, never a literal"
                )
    return bad


def fence_canonical_numbers() -> list[str]:
    """Run the REAL writer over the frozen ES6 number vectors (`vectors/es6-numbers.txt`).

    Frozen data, exercised by running the real writer over it — never a golden-test
    harness. Each row is `<IEEE754 little-endian hex> <expected ES6 string>`, so the vector
    is language-neutral and cannot drift with anyone's float parser. The oracle was V8
    (`JSON.stringify`); the corpus banks the boundary neighbourhoods plus a deterministic
    sample of a 1,000,000-value differential run.

    It exists because RFC 8785's number rule is the one place this codec had a real defect:
    an `is_integer()` fast path printed a double's EXACT expansion instead of its shortest
    round-tripping digits, so `1.8645733457839102e20` serialized as
    `186457334578391023616` where every ES6 reader writes `186457334578391020000`.
    """
    import struct

    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal import canonical

    bad: list[str] = []
    path = ROOT / "vectors" / "es6-numbers.txt"
    rows = path.read_text().strip().splitlines()
    for line in rows:
        raw, expected = line.split(" ", 1)
        value = struct.unpack("<d", bytes.fromhex(raw))[0]
        got = canonical.write(value).decode()
        if got != expected:
            bad.append(f"{raw}: wrote {got!r}, ES6 says {expected!r}")
    if not rows:
        bad.append(f"{path} is empty — a vector file that vanishes must not read green")
    return bad[:10]


def fence_torch_free_import() -> list[str]:
    """`import cozy_runtime` must never pull torch (§1.0), derivation harness included.

    torch is an optional extra: the derive harness imports it INSIDE `torch_module()`, so a
    cloned endpoint stays inspectable and the base package installs without a 3 GB wheel. A
    module-scope `import torch` anywhere under `src/` breaks that, silently, for everyone.
    """
    bad = []
    for path in py_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if not any(n.split(".")[0] == "torch" for n in names):
                continue
            if node.col_offset == 0:
                bad.append(
                    f"{module_of(path)}:{node.lineno} imports torch at module scope — the "
                    "harness imports it inside a function so the base package stays torch-free"
                )
    return bad


#: §8's ONE ATTEMPT KERNEL, as a structural rule rather than a promise.
#:
#: `run`, `job` and `serve` are three ENTRANCES to one execution state machine. The v1
#: failure mode this exists to prevent is exact and was expensive: the developer door and
#: the production door ran different lifecycle code, so everything proven about one proved
#: nothing about the other. Two facts hold the shape:
#:
#:   * the author invocation kernel is CALLED from exactly one module — the device executor;
#:   * the attempt engine is CONSTRUCTED in exactly one module — the supervisor session.
#:
#: Anything that wants to run an attempt therefore has to go through the supervisor, over
#: the seam, into the executor. A direct-call runner beside it cannot be written without
#: turning this fence red.
KERNEL_CALLS = ("attempt", "prepare", "invoke", "run_prepared")
KERNEL_CALLER = "internal/executor.py"
ENGINE_CLASS = "AttemptEngine"
ENGINE_BUILDER = "internal/worker/session.py"


def fence_one_attempt_kernel() -> list[str]:
    bad: list[str] = []
    for path in py_files():
        rel = path.relative_to(SRC)
        # The author package IS the kernel and its declared testing seam (§1.6).
        if rel.parts[0] == "author":
            continue
        tree = ast.parse(path.read_text())
        allowed_caller = rel.as_posix() == KERNEL_CALLER
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "cozy_runtime.author"
            ):
                names = {a.name for a in node.names}
                hit = names & set(KERNEL_CALLS)
                if hit and not allowed_caller:
                    bad.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: imports the invocation "
                        f"kernel {sorted(hit)} — only {KERNEL_CALLER} may, so that `run`, "
                        "`job` and `serve` cannot diverge into two lifecycles"
                    )
            if isinstance(node, ast.Call) and not allowed_caller:
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else ""
                )
                if name in KERNEL_CALLS:
                    bad.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: calls the invocation "
                        f"kernel {name}() — the ONE kernel runs in the device executor"
                    )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == ENGINE_CLASS
                and rel.as_posix() != ENGINE_BUILDER
            ):
                bad.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: constructs {ENGINE_CLASS} — "
                    f"only {ENGINE_BUILDER} may; a second engine is a second lifecycle"
                )
    return bad


#: The OBSERVATION plane (cr-011). `tel.stage()` records a NUMBER; it must be structurally
#: incapable of changing what is resident, where it sits, or what a method may touch.
OBSERVATION_MODULES = (
    "author/_observations.py",
    "internal/worker/observe.py",
    "internal/worker/triage.py",
)
#: Modules that DECIDE residency, placement or the component-use contract.
DECIDING_MODULES = (
    "cozy_runtime.internal.fill",
    "cozy_runtime.internal.contract",
    "cozy_runtime.internal.derive",
    "cozy_runtime.internal.worker.plan",
    "cozy_runtime.internal.worker.ledger",
    "cozy_runtime.author._model",
    "torch",
)
#: Names through which residency could be reached even without an import.
DECIDING_NAMES = (
    "_cozy_scope",
    "_cozy_active",
    "_cozy_check_access",
    "materialize",
    "evict",
    "reserve",
    "rollback",
    "commit_fill",
    "empty_cache",
)


def fence_timing_decides_nothing() -> list[str]:
    """`stage_ms` per-stage timing is a COMMITTED telemetry fact and NOTHING else.

    cozy-runtime.md §1.3 keeps it structurally distinct from a component-use scope. The
    proof is not a comment: the observation plane cannot import a deciding module and cannot
    name a residency operation, so a timing bracket has no reachable way to alter component
    availability, placement, or the ComponentUseContract.
    """
    bad: list[str] = []
    for path in py_files():
        rel = path.relative_to(SRC).as_posix()
        if rel not in OBSERVATION_MODULES:
            continue
        tree = ast.parse(path.read_text())
        for name in imports(tree):
            for deciding in DECIDING_MODULES:
                if name == deciding or name.startswith(deciding + "."):
                    bad.append(
                        f"{path.relative_to(ROOT)}: the observation plane imports {name} — "
                        "timing and triage RECORD, they never decide residency or placement"
                    )
        for node in ast.walk(tree):
            reached = ""
            if isinstance(node, ast.Attribute) and node.attr in DECIDING_NAMES:
                reached, line = node.attr, node.lineno
            elif isinstance(node, ast.Name) and node.id in DECIDING_NAMES:
                reached, line = node.id, node.lineno
            if reached:
                bad.append(
                    f"{path.relative_to(ROOT)}:{line}: the observation plane names "
                    f"{reached!r}, a residency operation — an observer that can move bytes "
                    "is not an observer"
                )
    return bad


FENCES = {
    "forbidden-dependency": fence_forbidden_dependency,
    "one-attempt-kernel": fence_one_attempt_kernel,
    "canonical-numbers": fence_canonical_numbers,
    "torch-free-import": fence_torch_free_import,
    "package-imports": fence_package_imports,
    "env-authority": fence_env_authority,
    "single-config": fence_single_config,
    "no-interactive": fence_no_interactive,
    "excluded-keywords": fence_excluded_keywords,
    "no-identifiers-in-endpoint-code": fence_no_identifiers,
    "timing-decides-nothing": fence_timing_decides_nothing,
    "endpoint-owns-no-residency": fence_endpoint_owns_no_residency,
}


def main() -> int:
    red = 0
    for name, check in FENCES.items():
        violations = check()
        if violations:
            red += 1
            print(f"FENCE RED — {name}:", file=sys.stderr)
            for v in violations:
                print("  " + v, file=sys.stderr)
        else:
            print(f"fence green — {name}")
    print(f"{len(FENCES) - red}/{len(FENCES)} fences green over {len(py_files())} modules")
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
