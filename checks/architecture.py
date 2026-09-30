#!/usr/bin/env python3
"""Architecture fences (boundaries.md, cozy-runtime.md §7). Not a test: a dependency scan.

Every fence here is a straight AST import/dependency scan over a boundary — plus one
exception, `canonical-numbers`, which replays the REAL RFC 8785 number writer over a frozen
ES6 vector corpus and is kept because it caught an actual cross-language defect (an fp8
rowwise scale off by one ULP against tensorfs, 9b3e81c). Nothing here asserts that a source
file contains a particular string.

`--only NAME` runs one; `python checks/architecture.py` runs them all and names each red.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import pathlib
import re
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cozy_runtime"


def _literal_assignments(path: pathlib.Path, names: frozenset[str]) -> dict[str, list[object]]:
    found: dict[str, list[object]] = {name: [] for name in names}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in found:
                found[target.id].append(ast.literal_eval(node.value))
    return found


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
    # Public derivation surface (cr-071): job packages import it directly. It builds only
    # on the author surface and scoped TensorFS API, never worker storage or CLI authority.
    "derive": ("cozy_runtime.internal", "cozy_runtime.cli", "worker_protocol", "tensorfs"),
    # The activation probe (cr-075): a public mechanism surface shaped like `derive` — job
    # code and the runtime verb both build on it; it never reaches the worker planes.
    "probe": ("cozy_runtime.internal", "cozy_runtime.cli", "worker_protocol", "tensorfs"),
    "models": ("cozy_runtime.internal", "cozy_runtime.cli", "worker_protocol", "tensorfs"),
    "internal": ("cozy_runtime.cli",),
    "cli": (),
}
# cr-174: TensorFS owns public derivation declarations and scoped handles. These
# execution bindings may import that API; direct Store and native internals stay fenced.
SCOPED_TENSORFS_FILES = frozenset(
    {
        "author/_context.py",
        "author/_model_reader.py",
        "author/_invoke.py",
        "derive/facade.py",
        "derive/operations.py",
        "derive/quantization.py",
    }
)
GENERATED_PROTOCOL_PREFIX = "cozy.worker.v1"
# Pillow is a base dependency and the actual public Image value type; imports stay at module scope.
AUTHOR_ALLOWED_THIRD_PARTY = ("msgspec", "PIL")
# The `media` extra's encoder (cr-017): allowed in the author package ONLY function-scoped,
# the same shape as torch in the harness — importing the author surface never pulls it, and
# an absent wheel is the typed `encoder_unavailable` refusal, not an ImportError at import.
AUTHOR_ALLOWED_LAZY: dict[str, frozenset[str]] = {
    "av": frozenset({"_codec.py", "_decode.py"}),
}

ENV_AUTHORITY = SRC / "internal" / "config.py"
CONFIG_SYMBOLS = ("RuntimeConfig", "read_config")
CONFIG_NAMES = ("CONFIG", "SETTINGS")


BASE_DISTRIBUTIONS = ROOT / "base-distributions.json"
BASE_OBSERVATION = SRC / "internal" / "base_observation.py"


def base_distributions_bytes() -> bytes:
    """Render the worker-image vocabulary from its ONE source, `_PROTECTED_IMPORT_ROOTS`."""

    found = _literal_assignments(
        BASE_OBSERVATION, frozenset({"_PROTECTED_IMPORT_ROOTS", "_PROTECTED_PREFIXES"})
    )
    roots = found["_PROTECTED_IMPORT_ROOTS"]
    prefixes = found["_PROTECTED_PREFIXES"]
    if len(roots) != 1 or len(prefixes) != 1:
        raise SystemExit("base_observation.py must assign each protected constant exactly once")
    names = roots[0]
    families = prefixes[0]
    if not isinstance(names, dict) or not isinstance(families, tuple):
        raise SystemExit("the protected constants are not a name->roots mapping and a prefix tuple")
    # No `format` key and no kind name: this is a GENERATED CONSTANT, not a document.
    # A `cozy.<name>/N` spelling here would mint a document kind with no boundary,
    # storage, or digest party behind it.
    document = {
        "distributions": sorted(str(name) for name in names),
        "prefixes": sorted(str(prefix) for prefix in families),
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def fence_base_distributions() -> list[str]:
    """`base-distributions.json` is generated, not maintained.

    It is the only thing cozy-creator vendors from this repo: the client's upload-side prune
    list is derived from these bytes, so a name added here reaches the client and a name the
    client prunes is one the worker actually owns. Regenerate with
    `python checks/architecture.py --write-base-distributions`.
    """

    expected = base_distributions_bytes()
    if not BASE_DISTRIBUTIONS.is_file():
        return [f"{BASE_DISTRIBUTIONS.name} is missing; regenerate it"]
    if BASE_DISTRIBUTIONS.read_bytes() != expected:
        return [
            f"{BASE_DISTRIBUTIONS.name} is not the rendering of "
            "base_observation._PROTECTED_IMPORT_ROOTS; regenerate it"
        ]
    return []


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


def fence_dependency_version_ranges() -> list[str]:
    """Published requirements permit patch upgrades; exact cohorts live in uv.lock."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    groups = {"dependencies": project.get("dependencies", [])}
    groups.update(project.get("optional-dependencies", {}))
    bad: list[str] = []
    for group, requirements in groups.items():
        for declaration in requirements:
            requirement = Requirement(declaration)
            for bound in requirement.specifier:
                if bound.operator == "===":
                    bad.append(
                        f"pyproject.toml: {group} requirement {declaration!r} pins "
                        "an exact version; pin resolved versions in uv.lock"
                    )
                    continue
                version = bound.version.removesuffix(".*")
                release = Version(version).release
                pinned = (
                    (
                        bound.operator == "=="
                        and (not bound.version.endswith(".*") or len(release) > 2)
                    )
                    or (bound.operator == "~=" and len(release) > 3)
                    or (bound.operator == "<=" and len(release) > 2)
                    or (bound.operator == "<" and any(release[2:]))
                )
                if pinned:
                    bad.append(
                        f"pyproject.toml: {group} requirement {declaration!r} restricts "
                        "patch versions; use a minimum or major/minor range, and pin "
                        "resolved versions in uv.lock"
                    )
    return bad


def fence_package_imports() -> list[str]:
    bad: list[str] = []
    stdlib = sys.stdlib_module_names
    for path in py_files():
        parts = path.relative_to(SRC).parts
        pkg = parts[0] if len(parts) > 1 else ""
        forbidden = PACKAGE_FENCES.get(pkg, ())
        tree = ast.parse(path.read_text())
        module_scope = {
            a.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.col_offset == 0
            for a in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.col_offset == 0
        }
        for name in imports(tree):
            rel = path.relative_to(ROOT)
            scoped_tensorfs = (
                name == "tensorfs.derived"
                and path.relative_to(SRC).as_posix() in SCOPED_TENSORFS_FILES
            )
            if pkg != "protocol" and (
                name == GENERATED_PROTOCOL_PREFIX
                or name.startswith(GENERATED_PROTOCOL_PREFIX + ".")
            ):
                bad.append(
                    f"{rel}: Runtime modules must import generated bindings through "
                    "cozy_runtime.protocol"
                )
            if not scoped_tensorfs and any(
                name == f or name.startswith(f + ".") for f in forbidden
            ):
                bad.append(f"{rel}: {module_of(path)} may not import {name}")
            root = name.split(".")[0]
            if (
                pkg == "author"
                and root not in stdlib
                and root != "cozy_runtime"
                and not scoped_tensorfs
                and root not in AUTHOR_ALLOWED_THIRD_PARTY
                and not (
                    root in AUTHOR_ALLOWED_LAZY
                    and path.name in AUTHOR_ALLOWED_LAZY[root]
                    and root not in module_scope
                )
            ):
                bad.append(f"{rel}: author surface may not import third-party {name}")
    return bad


def fence_author_import_boundary() -> list[str]:
    """Package-shaped code uses the public author roots, never their implementation files."""

    bad: list[str] = []
    paths = (
        *(ROOT / "examples").rglob("*.py"),
        *(ROOT / "corpus").rglob("*.py"),
    )
    for path in sorted(paths):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                private = [a.name for a in node.names if a.name.startswith("cozy_runtime.author._")]
            elif isinstance(node, ast.ImportFrom):
                private = []
                if (node.module or "").startswith("cozy_runtime.author._"):
                    private.append(node.module or "")
                elif node.module == "cozy_runtime.author":
                    private.extend(a.name for a in node.names if a.name.startswith("_"))
            else:
                continue
            for name in private:
                bad.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: package-shaped code imports "
                    f"private author module/name {name!r}"
                )
    return bad


def fence_env_authority() -> list[str]:
    # The one env read the fence protects builds the class-B/C/D child envs of
    # tracker-v2/spawn-allowlists.md (#616.d).
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


def fence_workspace_database_only() -> list[str]:
    """Only the shared native workspace journal may own SQLite, never product history."""

    bad: list[str] = []
    database_modules = ("sqlite3", "sqlalchemy", "libsql", "turso")
    for path in py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for name in imports(tree):
            if name == "sqlite3" and path == SRC / "internal" / "worker" / "workspace.py":
                continue
            if any(name == module or name.startswith(module + ".") for module in database_modules):
                bad.append(f"{path.relative_to(ROOT)} imports database module {name}")
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
    """The API the runtime PUBLISHES to authors declares no device, dtype, offload, pin,
    eviction, compile or source-converter keyword: every one names a decision it owns
    (§1.1/§7).

    Scope is `src/cozy_runtime/author/**` — the runtime's own definitions, which is a fact
    about the surface we ship and nothing observes at runtime. A PACKAGE's model methods are
    a different surface and need no fence: `_model._check_signature` refuses one at
    decoration and `_describe._model_clock` again at describe, both `excluded_keyword`, and
    `tests/test_author_surface.py` drives that refusal.

    Dunder and private callables are the RUNTIME's own plumbing (the runtime hands the
    Loader its device). `path=` stays out of the SCAN because `out.save_file(path)` is a
    legitimate published surface; the in-band clock still refuses it on a model method.
    """
    from cozy_runtime.author._model import EXCLUDED_KEYWORDS

    # `author/fakes.py` supplies the RUNTIME's side of the contract, so it is exempt.
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


def package_modules() -> list[pathlib.Path]:
    """Every module in the tree that builds an `App()` — the file `describe` imports."""
    modules = []
    for path in sorted((ROOT / "examples").rglob("*.py")) + sorted((ROOT / "corpus").rglob("*.py")):
        if ".venv" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "App":
                modules.append(path)
                break
    return modules


def package_and_model_code() -> list[pathlib.Path]:
    """Every module an PACKAGE AUTHOR writes: the entrypoint modules and their models.

    `corpus/` counts — those are real packages against real weights, and the claim cr-008b
    makes is about what a package contains, not about what a weightless example contains.
    The vendored venv is not author code and is excluded by path.
    """
    paths = list(package_modules())
    for base in ("corpus/pipeline", "corpus/package"):
        paths += sorted((ROOT / base).glob("*.py"))
    for name in ("sdxl_real.py", "sdxl_pipeline.py", "sdxl.py", "h3.py"):
        candidate = ROOT / "corpus" / name
        if candidate.is_file():
            paths.append(candidate)
    return paths


#: Residency vocabulary. A package that can name one of these can move bytes, and §1.1's
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


def fence_package_owns_no_residency() -> list[str]:
    """cr-008b's done-when, as a scan: a package serves with ZERO memory-management calls.

    Not because its author is disciplined — because the vocabulary is not reachable from
    package code. The runtime admits and materializes a declared component set before
    method entry, event-fences it, and makes it evictable after; a package that could
    also do those things would be a second policy, and two policies is the v1 defect.
    """
    bad: list[str] = []
    for path in package_and_model_code():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            reached, line = "", 0
            if isinstance(node, ast.Attribute) and node.attr in RESIDENCY_NAMES:
                reached, line = node.attr, node.lineno
            elif isinstance(node, ast.Name) and node.id in RESIDENCY_NAMES:
                reached, line = node.id, node.lineno
            if reached:
                bad.append(
                    f"{path.relative_to(ROOT)}:{line}: package code names {reached!r} — "
                    "residency is the runtime's, declared with @uses_components and "
                    "nothing else; a package that can move bytes is a second policy"
                )
    return bad


def fence_package_owns_no_media_decode() -> list[str]:
    """cr-012's one-decoder law: package code consumes `MediaDecoder` values.

    Opening a container, reaching the runtime's private spool path, or calling a library
    path helper would create another fetch/decode authority with different bounds and clock
    semantics. Those spellings are absent from package/model code, not merely discouraged.
    """
    bad: list[str] = []
    for path in package_and_model_code():
        tree = ast.parse(path.read_text())
        rel = path.relative_to(ROOT)
        for name in imports(tree):
            if name == "av" or name.startswith("av.") or name == "PIL" or name.startswith("PIL."):
                bad.append(
                    f"{rel}: package code imports {name!r} — hydrated media decodes only "
                    "through the Runtime-owned MediaDecoder"
                )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr in {"_local", "_path"}:
                bad.append(
                    f"{rel}:{node.lineno}: package code reaches private media path "
                    f"{node.attr} — verified "
                    "input paths are Runtime-private"
                )
            elif node.attr == "demux":
                bad.append(
                    f"{rel}:{node.lineno}: package code demuxes packets — only immutable "
                    "Runtime media events cross the author boundary"
                )
            elif node.attr == "from_file":
                bad.append(
                    f"{rel}:{node.lineno}: package code calls from_file — a library path/URL "
                    "decoder is a second media authority"
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
    from cozy_runtime.author import canonical_json

    bad: list[str] = []
    safe = (1 << 53) - 1
    admitted = refused = 0
    path = ROOT / "vectors" / "es6-numbers.txt"
    rows = path.read_text().strip().splitlines()
    for line in rows:
        raw, expected = line.split(" ", 1)
        value = struct.unpack("<d", bytes.fromhex(raw))[0]
        try:
            got = canonical_json.encode(value).decode()
        except ValueError:
            if abs(value) <= safe:
                bad.append(f"{raw}: in-range {value!r} refused")
            else:
                refused += 1
            continue
        admitted += 1
        if abs(value) > safe:
            bad.append(f"{raw}: out-of-profile {value!r} was admitted as {got!r}")
        elif got != expected:
            bad.append(f"{raw}: wrote {got!r}, ES6 says {expected!r}")
    if not rows:
        bad.append(f"{path} is empty — a vector file that vanishes must not read green")
    if admitted == 0 or refused == 0:
        bad.append(f"numeric profile did not exercise both outcomes: {admitted=} {refused=}")

    in_range = (
        b'{"v":9007199254740991}',
        b'{"v":9007199254740991.0}',
        b'{"v":9.007199254740991e15}',
        b'{"v":-9007199254740991}',
        b'{"v":-9007199254740991.0}',
        b'{"v":-9.007199254740991e15}',
    )
    identities = set()
    for source in in_range:
        try:
            canonical_json.decode(source)
            normalized = canonical_json.normalize(source)
            if canonical_json.normalize(normalized) != normalized:
                bad.append(f"{source!r}: normalization is not idempotent")
            identities.add(canonical_json.digest_bytes(source))
        except ValueError as exc:
            bad.append(f"{source!r}: safe boundary spelling refused: {exc}")
    if len(identities) != 2:
        bad.append(f"safe +/- boundary twins produced {len(identities)} identities, want 2")

    out_of_range = (
        b'{"v":9007199254740992}',
        b'{"v":9007199254740992.0}',
        b'{"v":9.007199254740992e15}',
        b'{"v":-9007199254740992}',
        b'{"v":-9007199254740992.0}',
        b'{"v":-9.007199254740992e15}',
        b'{"v":1000000000000000000000}',
        b'{"v":1e21}',
    )
    for source in out_of_range:
        for operation in (
            canonical_json.decode,
            canonical_json.normalize,
            canonical_json.digest_bytes,
        ):
            try:
                operation(source)
            except ValueError:
                continue
            bad.append(f"{source!r}: {operation.__name__} admitted an out-of-profile value")
    return bad[:10]


def fence_torch_free_import() -> list[str]:
    """`import cozy_runtime` must never pull torch (§1.0), derivation harness included.

    torch is an optional extra: the derive harness imports it INSIDE `torch_module()`, so a
    cloned package stays inspectable and the base package installs without a 3 GB wheel. A
    Builtin implementations under models/ may import optional ML libraries; their
    namespace modules remain lazy and the import-boundary tests verify base imports.
    """
    bad = []
    for path in py_files():
        if path.is_relative_to(SRC / "models") and path.name != "__init__.py":
            continue
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


STATIC_READERS = tuple(
    SRC / "internal" / name for name in ("static_interface.py", "static_invocable.py")
)
#: Modules and builtins through which source becomes execution. The static reader may name none.
EXECUTION_MODULES = frozenset({"importlib", "sys", "runpy", "subprocess", "builtins"})
EXECUTION_CALLS = frozenset({"exec", "eval", "compile", "__import__"})
#: The runtime modules the static reader may import: the author vocabulary, the shared
#: document builders and the pure readers. `discovery` (the importer) is not among them.
STATIC_READER_IMPORTS = frozenset(
    {
        "cozy_runtime",
        "cozy_runtime.author",
        "cozy_runtime._interpreter",  # packaged policy reader; no authored code is imported
        "cozy_runtime.internal.package_interface",
        "cozy_runtime.internal.bindings",
        "cozy_runtime.internal.canonical",
        "cozy_runtime.internal.static_interface",
        "cozy_runtime.internal.static_invocable",
        "cozy_runtime.internal.invocable_interface",
    }
)


def fence_static_describe_executes_nothing() -> list[str]:
    """`internal/static_interface.py` reads package source and runs none of it (#713).

    Package code is untrusted, and the static reader is what `describe` runs by default —
    on a publisher's host, in packages CI, wherever a source tree is. A module that can
    import, exec or compile is a module that can be made to run the tree it was handed, so
    the reader's whole import list and call surface is held here: no execution module, no
    execution builtin, no `sys.path`, and no runtime module that imports packages.
    """
    bad = []
    for reader in STATIC_READERS:
        tree = ast.parse(reader.read_text())
        where = reader.relative_to(ROOT)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    if name.split(".")[0] in EXECUTION_MODULES:
                        bad.append(f"{where}:{node.lineno} imports {name} — a door to execution")
                    elif name.split(".")[0] == "cozy_runtime" and not (
                        name in STATIC_READER_IMPORTS or name.startswith("cozy_runtime.author.")
                    ):
                        bad.append(
                            f"{where}:{node.lineno} imports {name}, which is not a pure reader"
                        )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in EXECUTION_CALLS:
                    bad.append(f"{where}:{node.lineno} calls {node.func.id}()")
    return bad


KERNEL_CALLS = ("attempt", "prepare", "invoke", "run_prepared")
KERNEL_CALLER = "internal/executor.py"
ENGINE_CLASS = "AttemptEngine"
ENGINE_BUILDER = "internal/worker/session.py"


def _dotted(node: ast.expr) -> str:
    """`a.b.c` as a string, or "" for anything that is not a plain dotted name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def fence_one_attempt_kernel() -> list[str]:
    """ONE invocation kernel, reachable from ONE module, so `run`, `job` and `serve` cannot
    drift into two lifecycles.

    A dependency-direction rule: nothing observes it at runtime, because the second
    lifecycle a second caller would build works perfectly well. What it scans is REACH, not
    a verb — an import of a kernel name, or an attribute on a binding that resolves to the
    author package. `prepare`, `invoke` and `attempt` are ordinary words that ordinary
    objects answer to, and matching them on any receiver is the name-for-capability
    conflation PR #181 deleted the surface blocklists for.
    """
    bad: list[str] = []
    for path in py_files():
        rel = path.relative_to(SRC)
        # The author package IS the kernel and its declared testing seam (§1.6).
        if rel.parts[0] == "author":
            continue
        tree = ast.parse(path.read_text())
        allowed_caller = rel.as_posix() == KERNEL_CALLER
        where = path.relative_to(ROOT)
        # Names bound in THIS module to the author package itself — `import
        # cozy_runtime.author [as x]`, `from cozy_runtime import author [as x]` — through
        # which a kernel call is reachable without ever importing its name.
        aliases = {"cozy_runtime.author"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "cozy_runtime.author" or alias.name.startswith(
                        "cozy_runtime.author."
                    ):
                        aliases.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module == "cozy_runtime":
                aliases.update(a.asname or a.name for a in node.names if a.name == "author")
        if allowed_caller:
            aliases.clear()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("cozy_runtime.author")
                and not allowed_caller
            ):
                hit = sorted({a.name for a in node.names} & set(KERNEL_CALLS))
                if hit:
                    bad.append(
                        f"{where}:{node.lineno}: imports the invocation kernel {hit} — only "
                        f"{KERNEL_CALLER} may, so that `run`, `job` and `serve` cannot "
                        "diverge into two lifecycles"
                    )
            if (
                isinstance(node, ast.Attribute)
                and node.attr in KERNEL_CALLS
                and _dotted(node.value) in aliases
            ):
                bad.append(
                    f"{where}:{node.lineno}: reaches the invocation kernel "
                    f"{_dotted(node.value)}.{node.attr} — the ONE kernel runs in "
                    f"{KERNEL_CALLER}"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == ENGINE_CLASS
                and rel.as_posix() != ENGINE_BUILDER
            ):
                bad.append(
                    f"{where}:{node.lineno}: constructs {ENGINE_CLASS} — only "
                    f"{ENGINE_BUILDER} may; a second engine is a second lifecycle"
                )
    return bad


#: The OBSERVATION plane (cr-011). `tel.stage()` records a NUMBER; it must be structurally
#: incapable of changing what is resident, where it sits, or what a method may touch.
OBSERVATION_MODULES = (
    "author/_observations.py",
    "internal/worker/activity.py",
    "internal/worker/observe.py",
    "internal/worker/triage.py",
)
#: Modules that DECIDE residency, placement or the component-use contract.
DECIDING_MODULES = (
    "cozy_runtime.internal.fill",
    "cozy_runtime.internal.derive",
    "cozy_runtime.internal.worker.plan",
    "cozy_runtime.internal.worker.ledger",
    "cozy_runtime.author._model",
    "torch",
)
#: Names through which residency could be reached even without an import — an object PASSED
#: in closes no import. Only this runtime's own residency vocabulary belongs here: `reserve`
#: and `rollback` were removed because every container and every transaction answers to
#: them, and `commit_fill` because it names nothing that exists.
DECIDING_NAMES = (
    "_cozy_scope",
    "_cozy_active",
    "materialize",
    "evict",
    "empty_cache",
)


def fence_timing_decides_nothing() -> list[str]:
    """`stage_ms` per-stage timing is a COMMITTED telemetry fact and NOTHING else.

    cozy-runtime.md §1.3 keeps it structurally distinct from a component-use scope. The
    proof is not a comment: the observation plane cannot import a deciding module and cannot
    name a residency operation, so a timing bracket has no reachable way to alter component
    availability, placement, or the ComponentUseContract.

    A TYPE cannot hold this. `stage_ms` is a float the observation plane itself must
    min/max/sum (`_observations.Track.add`) and round for the wire, so an opaque
    non-comparable number would break the observer before it stopped anything — and it
    would still leave `internal/worker/observe.py` free to import `internal.fill`. The
    invariant is about a module's import graph, which is not a property of a value.

    `author/_observations.py` is listed for completeness only: `package-imports` already
    denies the whole author package `cozy_runtime.internal`. The two `internal/worker`
    modules are what nothing else covers.
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


SESSION = pathlib.Path("src/cozy_runtime/internal/worker/session.py")


def fence_supervised_worker_lanes() -> list[str]:
    """A WORKER LANE NEVER DIES SILENTLY (cr-061).

    Every long-lived thread in the worker session is the sole producer of some terminal: the
    device lane produces attempt outcomes, convergence produces a placement's verdict, the
    sender is the only thing that puts a frame on the stream. Nothing on the wire describes a
    dead Python thread, so a lane that dies leaves whoever waits on that terminal waiting on
    a producer that no longer exists — and the owner's only remaining escape is a timeout.

    `SessionState._lane` is the one answer: it turns any lane death into
    `WORKER_PHASE_FAILED`, which every owner already reads, and closes a non-persistent
    host's stream so a one-shot run EXITS. This fence says no thread may be started around
    it — `target=` in the session is a `self._lane(...)` call or it is a hole.
    """
    bad: list[str] = []
    path = ROOT / SESSION
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        named = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if named != "Thread":
            continue
        target = next((kw.value for kw in node.keywords if kw.arg == "target"), None)
        supervised = (
            isinstance(target, ast.Call)
            and isinstance(target.func, ast.Attribute)
            and target.func.attr == "_lane"
        )
        if not supervised:
            bad.append(
                f"{SESSION}:{node.lineno}: a worker lane is started outside `self._lane(...)` "
                "— a thread that dies here stops producing a terminal and says nothing"
            )
    return bad


#: cr-067 hard guard, kept under model-code-fit D9: the worker's preparation plane may not
#: IMPORT the derive machinery or TensorFS's author facade. `Slot.components` is what the
#: private bounded child constructed in the package's own environment through the
#: `internal/derive_child.py` pipe seam; the worker process never loads the meta construction.
WORKER_DERIVE_FORBIDDEN = ("cozy_runtime.internal.derive", "cozy_runtime.derive", "tensorfs")


def worker_derive_free_violations(worker_dir: pathlib.Path) -> list[str]:
    problems: list[str] = []
    for path in sorted(worker_dir.glob("*.py")):
        for name in imports(ast.parse(path.read_text())):
            for banned in WORKER_DERIVE_FORBIDDEN:
                if name == banned or name.startswith(banned + "."):
                    problems.append(
                        f"{path.relative_to(worker_dir.parent.parent.parent)}: imports {name} — "
                        "worker prep modules may not reach model derivation (cr-067)"
                    )
    return problems


def fence_worker_derive_free() -> list[str]:
    return worker_derive_free_violations(SRC / "internal" / "worker")


#: cr-080 (residency-aware-routing.md §5): residency crosses to an orchestrator as TWO SETS —
#: `DeviceLane.resident_placement_ids`, `ObservedWorkerState.held_manifests` — and nothing that
#: would let it price VRAM. The emitter names none of the ledger's pricing vocabulary, and the
#: three residency-bearing messages are constructed with exactly their vendored fields.
RESIDENCY_EMITTER = pathlib.Path("src/cozy_runtime/internal/worker/lane_wire.py")
RESIDENCY_MESSAGES = ("DeviceLane", "ObservedWorkerState", "WorkerSnapshotBody")
DEVICE_LANE_FIELDS = {
    "lane_id": 1,
    "device_ordinals": 2,
    "available_attempt_slots": 3,
    "placement_ids": 4,
    "resident_placement_ids": 5,
}
PRICING_NAMES = frozenset(
    {
        "resident_bytes",
        "authorized",
        "authorize",
        "authorized_device_limit_bytes",
        "outstanding",
        "outstanding_bytes",
        "allocatable",
        "headroom",
        "free_bytes",
        "total_bytes",
        "ledger",
        "chooser",
        "victims",
        "last_used",
    }
)


def _residency_words(tree: ast.AST) -> list[tuple[int, str]]:
    """Every name, attribute, keyword or string literal in `tree` that is a pricing word."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in PRICING_NAMES:
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in PRICING_NAMES:
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.keyword) and node.arg in PRICING_NAMES:
            found.append((node.value.lineno, node.arg))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in PRICING_NAMES
        ):
            found.append((node.lineno, node.value))
    return found


def fence_residency_sets_only() -> list[str]:
    """Two sets, no bytes: the residency emitter and every construction of a residency-
    bearing message carry membership only (cr-080's grep-level fence, as an AST scan).

    Three arms. (1) `lane_wire.py` names no pricing word anywhere — not as a name, an
    attribute, a keyword or a document key, so a planted `"resident_bytes"` row key is
    refused as surely as a planted constructor argument. (2) Every `pb.DeviceLane(...)`,
    `pb.ObservedWorkerState(...)` and `pb.WorkerSnapshotBody(...)` call under `src/` passes
    keywords from the vendored package interface only, none of them a pricing word. (3) The
    positional proof: the vendored `DeviceLane` carries exactly the five proto-026 fields
    at their numbers, so a field added to the wire message is refused here before any
    emitter could fill it.
    """
    bad: list[str] = []
    emitter = ROOT / RESIDENCY_EMITTER
    for line, word in _residency_words(ast.parse(emitter.read_text())):
        bad.append(
            f"{RESIDENCY_EMITTER}:{line}: the residency emitter names {word!r} — the wire "
            "carries two SETS (resident placements, held manifests) and no pricing fact"
        )
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from cozy.worker.v1 import worker_pb2 as pb
    finally:
        sys.path.pop(0)
    fields = {
        name: {f.name: f.number for f in getattr(pb, name).DESCRIPTOR.fields}
        for name in RESIDENCY_MESSAGES
    }
    if fields["DeviceLane"] != DEVICE_LANE_FIELDS:
        bad.append(
            f"vendored DeviceLane carries {fields['DeviceLane']}; proto-026 fixes it at "
            f"{DEVICE_LANE_FIELDS} — a residency field beyond the two sets is refused"
        )
    for name, numbered in fields.items():
        for word in sorted(set(numbered) & PRICING_NAMES):
            bad.append(f"vendored {name}.{word} prices residency on the wire")
    for path in py_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in RESIDENCY_MESSAGES or _dotted(node.func.value) != "pb":
                continue
            for keyword in node.keywords:
                if keyword.arg is None:
                    bad.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: pb.{node.func.attr}(**...) "
                        "hides its fields from the residency fence"
                    )
                elif keyword.arg not in fields[node.func.attr] or keyword.arg in PRICING_NAMES:
                    bad.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: pb.{node.func.attr}("
                        f"{keyword.arg}=...) is not a vendored field of the message"
                    )
    return bad


#: The one module that may open an HTTP connection, and only for the worker's
#: control-plane-minted output grant (`egress.put_from`) — until th-132 moves the upload
#: half into tensorfs too, when this allowlist shrinks to nothing.
HTTP_CLIENT_ALLOWED = frozenset({"internal/egress.py"})
HTTP_CLIENT_MODULES = frozenset({"urllib.request", "http.client"})


def fence_one_address_predicate() -> list[str]:
    """No second downloader beside the one egress door (cr-012, retargeted by cr-090).

    The failure mode this fence exists against is not "the SSRF check was wrong", it is
    "there were two downloaders and only one had the check". The download transport and its
    address predicate live in tensorfs (`tensorfs-core/src/transport/policy.rs`); what this
    runtime still holds is the upload half in `internal/egress.py`, whose `blocked()` is the
    ONE address predicate here. A urllib.request/http.client import anywhere else is a new
    byte-mover chain that the predicate does not cover — the exemption is never granted; a
    new mover belongs in tensorfs.
    """
    bad = []
    for path in py_files():
        where = path.relative_to(SRC).as_posix()
        if where in HTTP_CLIENT_ALLOWED:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            hit = sorted(
                n
                for n in names
                if n in HTTP_CLIENT_MODULES
                or any(n.startswith(m + ".") for m in HTTP_CLIENT_MODULES)
            )
            if hit:
                bad.append(
                    f"{module_of(path)}:{node.lineno} imports {', '.join(hit)} — a second "
                    "network byte mover beside internal/egress.py; downloads belong to "
                    "tensorfs's transport, and the exemption is never granted"
                )
    return bad


#: `Any` switches mypy strict off for everything it touches (the watchdog died on a
#: `dict[str, Any]` record's missing key, 2026-09-28). Records are decoded once, at their
#: boundary, into typed structs; this count under `internal/` may only fall. Lower it in
#: the change that removes some.
ANY_CEILING = 883


def any_references(tree: ast.AST) -> int:
    """`Any` / `typing.Any` uses, including inside string annotations and `cast` targets."""

    def count(node: ast.AST) -> int:
        return sum(
            (isinstance(n, ast.Name) and n.id == "Any")
            or (isinstance(n, ast.Attribute) and n.attr == "Any")
            for n in ast.walk(node)
        )

    spelled: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            spelled.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
            spelled.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            spelled.append(node.annotation)
        elif isinstance(node, ast.Call) and _dotted(node.func) in ("cast", "typing.cast"):
            spelled.extend(node.args[:1])
    total = count(tree)
    for expr in spelled:
        for n in ast.walk(expr):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                with contextlib.suppress(SyntaxError):  # a Literal's value, not a type
                    total += count(ast.parse(n.value, mode="eval"))
    return total


def fence_any_ratchet() -> list[str]:
    total = sum(any_references(ast.parse(p.read_text())) for p in (SRC / "internal").rglob("*.py"))
    if total > ANY_CEILING:
        return [
            f"internal/ spells `Any` {total} times, over the ratchet's {ANY_CEILING}: decode "
            "the record once into a typed struct (msgspec.Struct or the protobuf message)"
        ]
    if total < ANY_CEILING:
        print(f"any-ratchet: {total} < {ANY_CEILING} — lower ANY_CEILING to {total}")
    return []


FENCES = {
    "one-address-predicate": fence_one_address_predicate,
    "forbidden-dependency": fence_forbidden_dependency,
    "dependency-version-ranges": fence_dependency_version_ranges,
    "one-attempt-kernel": fence_one_attempt_kernel,
    "canonical-numbers": fence_canonical_numbers,
    "torch-free-import": fence_torch_free_import,
    "static-describe-executes-nothing": fence_static_describe_executes_nothing,
    "package-imports": fence_package_imports,
    "author-import-boundary": fence_author_import_boundary,
    "env-authority": fence_env_authority,
    "single-config": fence_single_config,
    "workspace-custody-database-only": fence_workspace_database_only,
    "no-interactive": fence_no_interactive,
    "excluded-keywords": fence_excluded_keywords,
    "timing-decides-nothing": fence_timing_decides_nothing,
    "package-owns-no-residency": fence_package_owns_no_residency,
    "package-owns-no-media-decode": fence_package_owns_no_media_decode,
    "supervised-worker-lanes": fence_supervised_worker_lanes,
    "base-distributions": fence_base_distributions,
    "worker-derive-free": fence_worker_derive_free,
    "residency-sets-only": fence_residency_sets_only,
    "any-ratchet": fence_any_ratchet,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the runtime's architecture fences")
    parser.add_argument("--only", action="append", choices=FENCES, default=[])
    parser.add_argument("--exclude", action="append", choices=FENCES, default=[])
    parser.add_argument(
        "--write-base-distributions",
        action="store_true",
        help="regenerate base-distributions.json from base_observation.py",
    )
    args = parser.parse_args(argv)
    if args.write_base_distributions:
        BASE_DISTRIBUTIONS.write_bytes(base_distributions_bytes())
        print(f"wrote {BASE_DISTRIBUTIONS}")
        return 0
    if args.only and args.exclude:
        parser.error("--only and --exclude are mutually exclusive")
    selected = {
        name: check
        for name, check in FENCES.items()
        if (not args.only or name in args.only) and name not in args.exclude
    }
    red = 0
    for name, check in selected.items():
        violations = check()
        if violations:
            red += 1
            print(f"FENCE RED — {name}:", file=sys.stderr)
            for v in violations:
                print("  " + v, file=sys.stderr)
        else:
            print(f"fence green — {name}")
    print(f"{len(selected) - red}/{len(selected)} fences green over {len(py_files())} modules")
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
