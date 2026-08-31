#!/usr/bin/env python3
"""Architecture fences (boundaries.md, cozy-runtime.md §7). Not a test: a dependency scan.

Architecture fences, each provable red by planting the violation (`scripts/redarm.py`):
  forbidden-dependency · canonical-numbers · torch-free-import
  package-imports · author-import-boundary · prelaunch-version · python-abi
  env-authority · single-config · no-interactive · excluded-keywords
  document-kinds · qualification-selection · implementation-ids · report-format-tags
  no-identifiers-in-package-code · one-attempt-kernel · timing-decides-nothing
  package-owns-no-residency · package-owns-no-media-decode · reviewed-specs-exist
  construction-fill-order
  one-address-predicate
  dynamic-pods-skip-boot-warm
  worker-control-memory-state
  delivery-plan-observation
  pod-media-grant-root
  measured-backend-version
  base-ownership-vectors
  typed-model-binding
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
import tempfile
import tomllib
import types
from collections.abc import Iterator

from packaging.version import InvalidVersion, Version

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


def fence_prelaunch_version() -> list[str]:
    """The v1 compatibility promise begins only when proto-014 cuts it deliberately."""

    bad: list[str] = []
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    distribution = project.get("version")
    source = _literal_assignments(
        SRC / "__init__.py", frozenset({"__version__", "PYTHON_CONTRACT_VERSION"})
    )
    for name, values in source.items():
        if len(values) != 1 or not isinstance(values[0], str):
            bad.append(f"src/cozy_runtime/__init__.py: {name} must have one literal string value")
    if bad or not isinstance(distribution, str):
        if not isinstance(distribution, str):
            bad.append("pyproject.toml: project.version must be a string")
        return bad
    runtime = source["__version__"][0]
    contract = source["PYTHON_CONTRACT_VERSION"][0]
    assert isinstance(runtime, str) and isinstance(contract, str)
    if runtime != distribution:
        bad.append(
            "src/cozy_runtime/__init__.py: __version__ does not match pyproject.toml "
            f"({runtime!r} != {distribution!r})"
        )
    for where, value in (
        ("pyproject.toml: project.version", distribution),
        ("src/cozy_runtime/__init__.py: PYTHON_CONTRACT_VERSION", contract),
    ):
        try:
            major = Version(value).major
        except InvalidVersion:
            bad.append(f"{where} is not a valid version: {value!r}")
        else:
            if major != 0:
                bad.append(f"{where} must remain 0.x until proto-014 cuts the v1 contract")
    return bad


def fence_python_abi() -> list[str]:
    """The distribution, tooling and bundled packages target standard CPython 3.12."""

    bad: list[str] = []
    requirement = ">=3.12,<3.13"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if project.get("project", {}).get("requires-python") != requirement:
        bad.append(f"pyproject.toml: requires-python must be {requirement!r}")
    if project.get("tool", {}).get("mypy", {}).get("python_version") != "3.12":
        bad.append("pyproject.toml: mypy must check the Python 3.12 language surface")

    pin = ROOT / ".python-version"
    if not pin.is_file() or pin.read_text() != "3.12\n":
        bad.append(".python-version: must pin uv to CPython 3.12")

    bundled = sorted((ROOT / "examples").rglob("pyproject.toml"))
    bundled += sorted((ROOT / "proofs" / "fixtures").rglob("pyproject.toml"))
    for path in bundled:
        value = tomllib.loads(path.read_text()).get("project", {}).get("requires-python")
        if value != requirement:
            bad.append(f"{path.relative_to(ROOT)}: requires-python must be {requirement!r}")

    for path in (
        ROOT / "uv.lock",
        ROOT / "examples" / "marco-polo" / "uv.lock",
        ROOT / "proofs" / "fixtures" / "environment-project" / "uv.lock",
    ):
        if tomllib.loads(path.read_text()).get("requires-python") != "==3.12.*":
            bad.append(f"{path.relative_to(ROOT)}: lock does not cover exactly Python 3.12")

    scaffold = (SRC / "cli" / "new.py").read_text()
    for expected in (
        'requires-python = ">=3.12,<3.13"',
        'dependencies = ["cozy-runtime>={__version__},<0.1"]',
        'python_version = "3.12"',
    ):
        if expected not in scaffold:
            bad.append(f"src/cozy_runtime/cli/new.py: generated package omits {expected!r}")

    runtime = (SRC / "_interpreter.py").read_text()
    for expected in (
        'sys.implementation.name != "cpython"',
        "sys.version_info[:2] != (3, 12)",
        'sysconfig.get_config_var("Py_GIL_DISABLED") == 1',
    ):
        if expected not in runtime:
            bad.append(f"src/cozy_runtime/_interpreter.py: runtime ABI guard omits {expected!r}")
    return bad


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
GENERATED_PROTOCOL_PREFIX = "cozy.worker.v1"
# The author surface is import-fenced to stdlib + schema/typing libraries only.
AUTHOR_ALLOWED_THIRD_PARTY = ("msgspec",)
# The `media` extra's encoder (cr-017): allowed in the author package ONLY function-scoped,
# the same shape as torch in the harness — importing the author surface never pulls it, and
# an absent wheel is the typed `encoder_unavailable` refusal, not an ImportError at import.
AUTHOR_ALLOWED_LAZY: dict[str, frozenset[str]] = {
    "av": frozenset({"_codec.py", "_decode.py"}),
}

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
            if pkg != "protocol" and (
                name == GENERATED_PROTOCOL_PREFIX
                or name.startswith(GENERATED_PROTOCOL_PREFIX + ".")
            ):
                bad.append(
                    f"{rel}: Runtime modules must import generated bindings through "
                    "cozy_runtime.protocol"
                )
            if any(name == f or name.startswith(f + ".") for f in forbidden):
                bad.append(f"{rel}: {module_of(path)} may not import {name}")
            root = name.split(".")[0]
            if (
                pkg == "author"
                and root not in stdlib
                and root != "cozy_runtime"
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
        *(ROOT / "proofs" / "fixtures").rglob("*.py"),
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


def fence_no_runtime_database() -> list[str]:
    """Runtime may use process memory and worker-scoped files, but owns no database."""

    bad: list[str] = []
    database_modules = ("sqlite3", "sqlalchemy", "libsql", "turso")
    for path in py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for name in imports(tree):
            if any(name == module or name.startswith(module + ".") for module in database_modules):
                bad.append(f"{path.relative_to(ROOT)} imports database module {name}")
    for retired in (SRC / "internal" / "cache.py", SRC / "internal" / "artifacts.py"):
        if retired.exists():
            bad.append(f"{retired.relative_to(ROOT)} restores retired Runtime-owned state")
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


#: A package module is one that builds an `App()` — the file `describe` imports.
_MEDIA_PREFIXES = ("image/", "video/", "audio/", "text/", "application/", "model/")
_IDENTIFIER = re.compile(r"^(hf|civitai|comfy|tensorhub):|^[a-z0-9][\w.-]*/[\w.-]+(@[\w.-]+)?$")
#: A versioned FORMAT id (`adaln-tables/1`, `ggml-q4-k/v1`) is a schema name, not an artifact.
_FORMAT_ID = re.compile(r"^[\w.-]+/v?\d+$")


def package_modules() -> list[pathlib.Path]:
    modules = []
    for path in sorted((ROOT / "proofs" / "fixtures" / "tiny-models").glob("*.py")):
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


def fence_no_identifiers() -> list[str]:
    """HARD RULE (§1.1): author code never names an artifact identifier of any kind — no
    model, release, checkpoint or adapter ref, in bindings or in string literals.

    Code states capability; SELECTION is a binding (package.toml, `--model`, the deploy
    binding). Identity is exclusively deployment-side, so branching on one is unspellable.
    """
    bad: list[str] = []
    for path in package_modules():
        tree = ast.parse(path.read_text())
        contract_values = {
            id(value)
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and (
                (isinstance(call.func, ast.Name) and call.func.id == "ProductionContract")
                or (isinstance(call.func, ast.Attribute) and call.func.attr == "ProductionContract")
            )
            for keyword in call.keywords
            if keyword.arg in {"structure", "encodings"}
            for value in ast.walk(keyword.value)
            if isinstance(value, ast.Constant)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in contract_values:
                # A production contract is a requirement Tensorhub independently derives
                # from output bytes, not a model/release/checkpoint selection.
                continue
            text = node.value.strip()
            if text.startswith(_MEDIA_PREFIXES) or "://" in text or text.startswith("/"):
                continue
            if _FORMAT_ID.match(text):
                continue
            if _IDENTIFIER.match(text):
                bad.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: artifact identifier {text!r} in "
                    "package code — selection is a binding, never a literal"
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


def fence_base_ownership_vectors() -> list[str]:
    """Run the strict complete-base parser over frozen v1/refusal vectors."""

    import hashlib

    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal import canonical, overlay

    bad: list[str] = []
    path = ROOT / "vectors" / "base-ownership.jsonl"
    names: set[str] = set()
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    for line_number, line in enumerate(rows, 1):
        vector = json.loads(line)
        if not isinstance(vector, dict):
            bad.append(f"{path}:{line_number}: vector is not an object")
            continue
        name = vector.get("name")
        document = vector.get("document")
        expected = vector.get("expect")
        if (
            not isinstance(name, str)
            or name in names
            or not isinstance(document, dict)
            or not isinstance(expected, str)
        ):
            bad.append(f"{path}:{line_number}: vector fields are incomplete or duplicated")
            continue
        names.add(name)
        raw = canonical.write(document)
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        try:
            overlay.parse_base_manifest(raw, digest)
            got = "ok"
        except overlay.OverlayRefusal as exc:
            got = exc.code
        if got != expected:
            bad.append(f"{path}:{line_number}: got {got}, expected {expected}")
    wanted = {"minimal-complete-v1", "unknown-format-refused", "unknown-provider-refused"}
    if names != wanted:
        bad.append(f"{path}: vector names are {sorted(names)}, want {sorted(wanted)}")
    root = ROOT / "vectors" / "tensorhub-wheel-contract"
    mirror_specs = (
        (
            "base-worker-image-wheelhouse-cpu.json",
            24_653,
            "sha256:0618d4ad492648320b8c79ed087ef3bfd252a918b80326789f104c4ef1614eed",
            (26, 323, 25),
            ("none", "none"),
        ),
        (
            "base-worker-image-wheelhouse-cu130.json",
            25_305,
            "sha256:73738b08458a6c83cae1b97a9675f775d27b616e08a7d034b85a481e63b1b2da",
            (86, 363, 9),
            ("cuda", "cu130"),
        ),
    )
    manifests: dict[str, overlay.BaseManifest] = {}
    manifest_digests: dict[str, str] = {}
    for filename, length, expected_digest, expected_census, accelerator in mirror_specs:
        cross = root / filename
        raw = cross.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if len(raw) != length or digest != expected_digest:
            bad.append(
                f"{cross}: exact Tensorhub stored-byte identity changed: {len(raw)} B / {digest}"
            )
        try:
            manifest = overlay.parse_base_manifest(raw, digest)
        except overlay.OverlayRefusal as exc:
            bad.append(f"{cross}: Runtime refused Tensorhub bytes: {exc.code}: {exc.detail}")
            continue
        manifests[filename] = manifest
        manifest_digests[filename] = digest
        census = (
            len(manifest.base_distributions),
            len(manifest.base_import_roots),
            len(manifest.wheels),
        )
        if census != expected_census:
            bad.append(f"{cross}: ownership census is {census}, want {expected_census}")
        target = dict(manifest.platform_target)
        expected_target = {
            "os_arch": "linux/amd64",
            "libc": "glibc2.36" if accelerator[0] == "none" else "glibc2.39",
            "python_abi": "cp312",
            "accelerator_backend": accelerator[0],
            "accelerator_abi": accelerator[1],
        }
        if target != expected_target:
            bad.append(f"{cross}: platform target is {target}, want {expected_target}")
        versions = manifest.distribution_versions
        expected_wheels = {
            "cozy-runtime": (
                "0.0.11",
                "cozy_runtime-0.0.11-py3-none-any.whl",
                "sha256:9ad5642698d4879d4982152c5f88d0599a2b2c2f418f20c38bbb7d7fc5760334",
                705_603,
            ),
            "tensorfs": (
                "0.0.3",
                "tensorfs-0.0.3-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                "sha256:55db3ee75eda51e77ef9c821d5a7b62294036b67c893bccb1bab910b783f9269",
                2_092_918,
            ),
        }
        wheels = {wheel.distribution: wheel for wheel in manifest.wheels}
        for distribution, expected_wheel in expected_wheels.items():
            version = expected_wheel[0]
            if versions.get(distribution) != version:
                bad.append(
                    f"{cross}: base owns {distribution}=={versions.get(distribution)}, "
                    f"want {version}"
                )
            wheel = wheels.get(distribution)
            fact = (
                None
                if wheel is None
                else (wheel.version, wheel.filename, wheel.digest, wheel.length)
            )
            if fact != expected_wheel:
                bad.append(f"{cross}: {distribution} wheel is {fact}, want {expected_wheel}")

    cu130_manifest = manifests.get("base-worker-image-wheelhouse-cu130.json")
    if cu130_manifest is not None:
        base_family_digest = manifest_digests["base-worker-image-wheelhouse-cu130.json"]

        def counted_wheel(index: int, owner: str) -> overlay.Wheel:
            label = "project" if owner == "project" else f"custom-{index:03d}"
            distribution = f"cr027-{label}"
            return overlay.Wheel(
                distribution=distribution,
                version="1.0.0",
                filename=f"{distribution}-1.0.0-py3-none-any.whl",
                digest="sha256:" + hashlib.sha256(label.encode()).hexdigest(),
                length=1,
                tags=("py3-none-any",),
                owner=owner,
                import_roots=(distribution.replace("-", "_"),),
            )

        project = counted_wheel(0, "project")

        def counted_input(custom_count: int) -> overlay.Input:
            wheels = (
                project,
                *(counted_wheel(index, "custom") for index in range(custom_count)),
            )
            return overlay.Input(
                environment_digest="sha256:" + "1" * 64,
                base_family_digest=base_family_digest,
                project_wheel_digest=project.digest,
                wheels=wheels,
                grants=(),
            )

        admitted = overlay._verify_inventory(counted_input(127), cu130_manifest)
        if len(admitted) != 128:
            bad.append(f"cr-027 wheel-count boundary admitted {len(admitted)} rows, want 128")
        try:
            overlay._verify_inventory(counted_input(128), cu130_manifest)
        except overlay.OverlayRefusal as exc:
            if exc.code != "overlay_wheel_set_invalid" or exc.detail != "129":
                bad.append(
                    "cr-027 128-custom refusal was "
                    f"{exc.code}:{exc.detail}, want overlay_wheel_set_invalid:129"
                )
        else:
            bad.append("cr-027 admitted 128 custom wheels plus the project wheel")
    return bad


#: §8's ONE ATTEMPT KERNEL, as a structural rule rather than a promise.
#:
#: `run`, `job` and `serve` are three ENTRANCES to one execution state machine. The v1
#: failure mode this exists to prevent is exact and was expensive: the developer door and
#: the production door ran different lifecycle code, so everything proven about one proved
#: nothing about the other. Two facts hold the shape:
#:
#:   * the author invocation kernel is CALLED from exactly one module — the device executor;
#:   * the attempt engine is CONSTRUCTED in exactly one module — the worker session.
#:
#: Anything that wants to run an attempt therefore has to go through the worker, over
#: the seam, into the executor. A direct-call runner beside it cannot be written without
#: turning this fence red.
KERNEL_CALLS = ("attempt", "prepare", "invoke", "run_prepared")
KERNEL_CALLER = "internal/executor.py"
ENGINE_CLASS = "AttemptEngine"
ENGINE_BUILDER = "internal/worker/session.py"


def fence_tensorhub_package_release() -> list[str]:
    """Consume Tensorhub's exact nested-wheel PackageRelease fixture."""

    import hashlib

    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal import canonical
    from cozy_runtime.internal.placement_materialization import package_overlay_input
    from cozy_runtime.internal.refusal import LaunchRefusal

    root = ROOT / "vectors" / "tensorhub-wheel-contract"
    path = root / "package-release-v1.json"
    stored = path.read_bytes()
    bad: list[str] = []
    got = hashlib.sha256(stored).hexdigest()
    want = "5f868fe549a22151e3a925d5638a13a315ae3508b16cbbad07ca0382d552313e"
    if got != want:
        return [f"{path.relative_to(ROOT)}: stored fixture hashes to {got}, want {want}"]
    if not stored.endswith(b"\n") or stored.endswith(b"\n\n"):
        return [f"{path.relative_to(ROOT)}: producer fixture must carry one trailing LF"]
    raw = stored[:-1]
    document_digest = hashlib.sha256(raw).hexdigest()
    if document_digest != "f894cff72dce53a2f85943bc10d1a1226f3fcacc983faddad6afbd6d61c425cf":
        return [f"{path.relative_to(ROOT)}: canonical document hashes to {document_digest}"]
    try:
        release = canonical.parse_canonical(raw)
    except canonical.CanonicalError as exc:
        return [f"{path.relative_to(ROOT)}: producer bytes are not canonical: {exc}"]
    if not isinstance(release, dict):
        return [f"{path.relative_to(ROOT)}: PackageRelease is not an object"]
    project = release.get("project_wheel")
    dependencies = release.get("dependency_wheels")
    if (
        not isinstance(project, dict)
        or not isinstance(dependencies, list)
        or len(dependencies) != 1
    ):
        return [f"{path.relative_to(ROOT)}: nested wheel fixture is incomplete"]
    dependency = dependencies[0]
    if not isinstance(dependency, dict):
        return [f"{path.relative_to(ROOT)}: dependency wheel is not an object"]
    fields = {"distribution", "filename", "import_roots", "ref", "tags", "version"}
    for name, wheel in (("project_wheel", project), ("dependency_wheels[0]", dependency)):
        if set(wheel) != fields:
            bad.append(f"{path.relative_to(ROOT)}: {name} is not one nested selected wheel")
    if bad:
        return bad

    manifest_path = root / "base-worker-image-wheelhouse-cu130.json"
    manifest = manifest_path.read_bytes()
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    environment = {
        "wheelhouse_manifest": {"digest": manifest_digest, "length": len(manifest)},
        "wheels": dependencies,
    }
    environment_digest = (
        "sha256:"
        + hashlib.sha256(
            canonical.write({**environment, "format": "cozy.worker.v1.Environment/1"})
        ).hexdigest()
    )
    project_ref = project.get("ref")
    dependency_ref = dependency.get("ref")
    if not isinstance(project_ref, dict) or not isinstance(dependency_ref, dict):
        return [f"{path.relative_to(ROOT)}: nested wheel ref is absent"]
    wheel_paths = {
        str(project_ref.get("digest")): root / "wheels" / "proof_package-1.0.0-py3-none-any.whl",
        str(dependency_ref.get("digest")): root / "wheels" / "proof_custom-1.0.0-py3-none-any.whl",
    }
    package = {
        "package": release.get("package"),
        "project_wheel": project,
        "release": release.get("release"),
        "release_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
    try:
        joined = package_overlay_input(
            package=package,
            environment=environment,
            environment_digest=environment_digest,
            wheelhouse_manifest_bytes=manifest,
            wheel_paths=wheel_paths,
        )
    except LaunchRefusal as exc:
        return [f"{path.relative_to(ROOT)}: Runtime refused Tensorhub fixture: {exc}"]
    facts = [(wheel.distribution, wheel.owner) for wheel in joined.wheels]
    if facts != [("proof-package", "project"), ("proof-custom", "custom")]:
        bad.append(f"{path.relative_to(ROOT)}: Runtime joined wheel facts as {facts}")

    flat = dict(project)
    flat.update(project_ref)
    flat.pop("ref", None)
    try:
        package_overlay_input(
            package={**package, "project_wheel": flat},
            environment=environment,
            environment_digest=environment_digest,
            wheelhouse_manifest_bytes=manifest,
            wheel_paths=wheel_paths,
        )
    except LaunchRefusal as exc:
        if exc.code != "project_wheel_invalid":
            bad.append(f"flat project wheel refused as {exc.code}, want project_wheel_invalid")
    else:
        bad.append("Runtime admitted the retired flat project WheelFact")
    return bad


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


def fence_reviewed_specs_exist() -> list[str]:
    """Every spec digest this build declares reviewed EXISTS in the installed registry.

    A provider's reviewed digest set is its claim about which physical byte contracts it was
    read against. TensorFS owns those documents, so the two repositories can drift — and
    they did: tensorfs `940973f` moved both `fp8-scaled-scalar/1` seed specs and this side
    kept the old pair, which made every fp8-scaled-scalar artifact unservable. The refusal
    was correct and it fired at PREPARE, on a rented H200, after 92.42 GiB had been pulled.
    It costs one call to find out at $0 instead.

    An absent `tensorfs` is RED, not skipped: this fence's whole subject is agreement with a
    registry, and a fence that goes green because it could not look is the fake green the
    fences exist to prevent.
    """
    try:
        import tensorfs
    except ImportError as exc:
        return [
            f"the tensorfs reader is not installed, so no declared spec digest can be "
            f"checked against the registry that owns it ({exc}). Install the wheel this "
            "release pins — an unchecked digest set is not a green one"
        ]
    from cozy_runtime.internal.encoding import formats

    carried = {digest: alias for alias, digest in tensorfs.seed_digests()}
    bad: list[str] = []
    for name in sorted(n for n in dir(formats) if n.startswith("SPEC_")):
        declared = getattr(formats, name)
        if not isinstance(declared, str) or not declared.startswith("sha256:"):
            continue
        if declared not in carried:
            bad.append(
                f"formats.{name} = {declared[:23]}… is not in the installed registry, which "
                f"carries {len(carried)} seed spec(s). A registry that moved a spec has "
                "changed what those bytes mean; inferring the replacement by alias is the "
                "substitution the launch refusal forbids, so the constant is REVIEWED and "
                "updated by hand or the wheel is the wrong one"
            )
    return bad


def fence_descriptor_minimality() -> list[str]:
    """Two fresh derivations produce one closed, compact descriptor/1 document."""
    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal import descriptor
    from cozy_runtime.internal.discovery import discover

    try:
        first = descriptor.build(discover(ROOT / "proofs" / "fixtures" / "tiny-models"))
        second = descriptor.build(discover(ROOT / "proofs" / "fixtures" / "tiny-models"))
        first_bytes = descriptor.canonical_bytes(first)
        second_bytes = descriptor.canonical_bytes(second)
        document = descriptor.read_bytes(first_bytes, "fresh example derivation")
    except Exception as exc:
        return [f"the fresh example descriptor is not readable: {exc}"]
    bad: list[str] = []
    if first_bytes != second_bytes:
        bad.append("two fresh example derivations produced different canonical bytes")
    expected = {"format", "application", "entrypoints", "jobs", "model_productions"}
    if set(document) != expected:
        bad.append(f"descriptor fields are {sorted(document)}, want {sorted(expected)}")
    if document.get("format") != "cozy.package.descriptor/1":
        bad.append("descriptor has the wrong format")
    forbidden = {
        "attribute",
        "capabilities",
        "config_schema",
        "context_facts",
        "default",
        "default_sources",
        "demand",
        "discriminator",
        "emits_media",
        "error_model",
        "frozen",
        "gpu",
        "hidden",
        "kind",
        "max_audio_channels",
        "max_audio_samples",
        "max_decoded_bytes",
        "max_pixels_per_frame",
        "max_video_frames",
        "placement",
        "param",
        "preflight",
        "protocol",
        "request_features",
        "requires",
        "schema",
        "secret_schema",
        "secrets",
        "services",
        "settings",
        "shape_axes",
        "struct",
        "surface_digest",
        "values",
    }

    def visit(value: object, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                scheduling_resource = (
                    (".model_productions[" in where or ".jobs[" in where)
                    and ".resources" in where
                    and key in {"placement", "requires"}
                )
                if key in forbidden and not scheduling_resource:
                    bad.append(f"{where}.{key} is an unused descriptor fact")
                if key == "wire" and item == "required":
                    bad.append(f"{where}.wire repeats the required-field default")
                visit(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{where}[{index}]")

    visit(document, "package_descriptor")
    return bad


#: proto-007 (#616.a): the registry of document kinds this repo AUTHORS, each with its one
#: declaring module. A format name exists only for a document that crosses a repo/process
#: boundary AND is stored or digested. Provider implementation IDs are separately fenced below:
#: they enter implementation/plan digests but are not documents.
DOCUMENT_KINDS = {
    "cozy.package.descriptor/1": "internal/descriptor.py",
    "cozy.runtime.EnvironmentReceipt/1": "internal/overlay.py",
    "cozy.runtime.ModelConstructionContract/1": "internal/contract.py",
    "cozy.runtime.Qualification/1": "qualification.py",
}
#: Kinds another repo authors and this one only READS (the wire, tensorhub, tensorfs).
#: Naming them is decoding, not authoring; the owner's fence polices the name.
FOREIGN_KIND_PREFIXES = ("cozy.worker.v1.", "tensorhub.", "tensorfs.", "cozytensors")
#: Tensorhub authors the environment-independent package release. PlacementSet is covered by the
#: worker-protocol prefix above; no binding-plan or model-binding documents survive.
FOREIGN_DOCUMENT_KINDS = {
    "cozy.package.release/1",
}
#: Modules that name a registered kind without declaring it (a second reader is fine; a
#: second DECLARATION is the drift this fence exists for).
KIND_READERS = {
    "cozy.runtime.ModelConstructionContract/1": {"internal/executor.py"},
}
_VERSIONED_LITERAL_PATTERN = (
    r'"((?:cozy|cozytensors|tensorhub|tensorfs)\.'
    r'[A-Za-z0-9_.-]+/\d+)"'
)
VERSIONED_LITERAL = re.compile(_VERSIONED_LITERAL_PATTERN)

#: Provider `name()` values are executable implementation identity, not document formats. The
#: exact strings are in every implementation_digest and therefore in every plan/generation digest;
#: moving them between fence registries must never rename or renumber them.
IMPLEMENTATION_IDS = {
    "cozy.fp8-rowwise.dequant/1": "internal/encoding/formats.py",
    "cozy.fp8-scaled-scalar.dequant/1": "internal/encoding/formats.py",
    "cozy.mxfp8.dequant/1": "internal/encoding/formats.py",
    "cozy.plain.verbatim/1": "internal/encoding/formats.py",
    "cozy.fp8-rowwise.native-leaf/1": "internal/encoding/leaves.py",
    "cozy.mxfp8.native-leaf/1": "internal/encoding/leaves.py",
}
IMPLEMENTATION_ID_READERS = {
    "cozy.mxfp8.dequant/1": {
        "internal/encoding/leaves.py",
        "internal/encoding/selection.py",
    },
}


def versioned_literals() -> Iterator[tuple[pathlib.Path, str, int, str]]:
    for path in py_files():
        rel = path.relative_to(SRC).as_posix()
        if rel.endswith("_pb2.py"):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for literal in VERSIONED_LITERAL.findall(line):
                yield path, rel, lineno, literal


def fence_document_kinds() -> list[str]:
    """proto-007: every document-kind literal this repo authors is in DOCUMENT_KINDS."""
    bad: list[str] = []
    expected = {
        "cozy.package.descriptor/1",
        "cozy.runtime.EnvironmentReceipt/1",
        "cozy.runtime.ModelConstructionContract/1",
        "cozy.runtime.Qualification/1",
    }
    if set(DOCUMENT_KINDS) != expected:
        bad.append(
            "Runtime document survivors differ from the proto-007 hardcut: "
            f"got={sorted(DOCUMENT_KINDS)} expected={sorted(expected)}"
        )
    for path, rel, lineno, kind in versioned_literals():
        if (
            kind.startswith(FOREIGN_KIND_PREFIXES)
            or kind in FOREIGN_DOCUMENT_KINDS
            or kind in IMPLEMENTATION_IDS
        ):
            continue
        owner = DOCUMENT_KINDS.get(kind)
        if owner is None:
            bad.append(
                f"{path.relative_to(ROOT)}:{lineno}: names {kind!r}, a document kind "
                "outside DOCUMENT_KINDS — a new kind needs a decision row (#616.a) and "
                "a boundary + storage/digest party (proto-007)"
            )
        elif rel != owner and rel not in KIND_READERS.get(kind, set()):
            bad.append(
                f"{path.relative_to(ROOT)}:{lineno}: names {kind!r}, declared in "
                f"{owner} — a second declaration of one kind"
            )

    from cozy.worker.v1 import worker_pb2
    from cozy_runtime.protocol import documents

    for descriptor in worker_pb2.DESCRIPTOR.message_types_by_name.values():
        want = f"{descriptor.full_name}/1"
        if documents.doc_format(descriptor.full_name) != want:
            bad.append(
                "src/cozy_runtime/protocol/documents.py: worker document "
                f"{descriptor.full_name!r} is not hardcut to {want!r}"
            )
    return bad


def fence_qualification_selection() -> list[str]:
    """Qualification binds exactly one selected PlacementSet fact tuple."""

    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime import qualification
    from cozy_runtime.internal import canonical

    ref = {"digest": "sha256:" + "1" * 64, "length": 1}
    package = {
        "package": "proof/proof-package",
        "project_wheel": {
            "distribution": "proof-package",
            "filename": "proof_package-1.0.0-py3-none-any.whl",
            "import_roots": ["proof_package"],
            "ref": ref,
            "tags": ["py3-none-any"],
            "version": "1.0.0",
        },
        "release": "1.0.0",
        "release_digest": "sha256:" + "5" * 64,
    }
    try:
        selected = qualification.selection_digest(
            bindings_digest="sha256:" + "2" * 64,
            environment_digest="sha256:" + "3" * 64,
            environment_receipt=ref,
            package_descriptor=ref,
            package=package,
        )
        raw = canonical.write(qualification.document(selected))
        if qualification.read(raw) != selected:
            return ["Qualification selection does not round-trip"]
        changed = qualification.selection_digest(
            bindings_digest="sha256:" + "4" * 64,
            environment_digest="sha256:" + "3" * 64,
            environment_receipt=ref,
            package_descriptor=ref,
            package=package,
        )
        if changed == selected:
            return ["Qualification selection ignores bindings_digest"]
    except qualification.QualificationError as exc:
        return [f"Qualification selection refused its own closed tuple: {exc}"]
    return []


def fence_development_placement() -> list[str]:
    """Development is explicit and never aliases source identity to an Environment."""

    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal import canonical, descriptor, local
    from cozy_runtime.internal.discovery import discover
    from cozy_runtime.internal.worker import acquire, session
    from cozy_runtime.protocol import documents
    from cozy_runtime.protocol import worker_pb2 as pb

    source_digest = "sha256:" + "7" * 64
    built = local.development_placement_documents(
        descriptor.build(discover(ROOT / "examples" / "marco-polo")),
        package="paul/marco-polo",
        release="1.0.3",
        source_digest=source_digest,
    )
    placement_set = pb.PlacementSet(placements=[built.placement])
    positive = documents.canonical_bytes(placement_set)
    desired = pb.DesiredPlacementSet(
        placement_set_digest=documents.digest_of(positive),
        placement_set_canonical_bytes=positive,
    )
    try:
        parsed = session.read_placement_set(desired)[0]
    except documents.DocumentError as exc:
        return [f"development Placement refused: {exc}"]
    forbidden = {
        "environment",
        "environment_digest",
        "environment_receipt_digest",
        "qualification",
    } & set(parsed)
    if forbidden:
        return [f"development Placement carries published field {sorted(forbidden)[0]}"]

    mutant = documents.document(placement_set)
    placement = mutant["placements"][0]
    assert isinstance(placement, dict)
    placement["environment_digest"] = source_digest
    mutant_bytes = canonical.write(mutant)
    try:
        session.read_placement_set(
            pb.DesiredPlacementSet(
                placement_set_digest=documents.digest_of(mutant_bytes),
                placement_set_canonical_bytes=mutant_bytes,
            )
        )
    except documents.DocumentError as exc:
        if exc.code != "development_fields_invalid":
            return [f"development environment alias refused as {exc.code}"]
    else:
        return ["development Placement admitted source_digest as environment_digest"]

    descriptor_digest = documents.spell(built.placement.package_descriptor.digest)
    with tempfile.TemporaryDirectory() as directory:
        cache = pathlib.Path(directory)
        staged = local.development_placement_documents(
            descriptor.build(discover(ROOT / "examples" / "marco-polo")),
            package="paul/marco-polo",
            release="1.0.3",
            source_digest=source_digest,
            artifact_cache=cache,
        )
        document = documents.body(staged.placement)
        acquired = acquire.preinstalled_development(
            types.SimpleNamespace(
                models=(),
                document=document,
                entrypoints=tuple(document["entrypoints"]),
                package_descriptor=(descriptor_digest, len(staged.descriptor)),
                development={
                    "package": "paul/marco-polo",
                    "release": "1.0.3",
                    "source_digest": source_digest,
                },
            ),
            cache,
            staged.placement,
            staged.bindings,
        )
    if any(
        binding.environment_digest or not binding.development
        for binding in acquired.bindings.values()
    ):
        return ["development binding invented an Environment identity"]
    for call, code in (
        (
            lambda: local._development_stamps("proof/tiny@1", {"family": "tiny"}),
            "development_model_stamps_unavailable",
        ),
        (
            lambda: local._development_assets("proof/tiny@1", {"tokenizer": {}}),
            "development_model_assets_unsupported",
        ),
    ):
        try:
            call()
        except RuntimeError as exc:
            if not str(exc).startswith(code + ":"):
                return [f"modeled development limitation refused as {exc}"]
        else:
            return [f"modeled development admitted unsupported fact instead of {code}"]
    return []


def fence_implementation_ids() -> list[str]:
    """Every executable provider ID has one owner and is never a document kind."""
    bad: list[str] = []
    overlap = sorted(DOCUMENT_KINDS.keys() & IMPLEMENTATION_IDS.keys())
    if overlap:
        bad.append("versioned identifier registry overlap: " + repr(overlap))

    owned: dict[str, int] = {implementation: 0 for implementation in IMPLEMENTATION_IDS}
    for path, rel, lineno, implementation in versioned_literals():
        owner = IMPLEMENTATION_IDS.get(implementation)
        if owner is None:
            continue
        readers = IMPLEMENTATION_ID_READERS.get(implementation, set())
        if rel == owner:
            owned[implementation] += 1
        elif rel not in readers:
            bad.append(
                f"{path.relative_to(ROOT)}:{lineno}: names implementation ID "
                f"{implementation!r}, owned by {owner} — add a reader explicitly or "
                "remove the second declaration"
            )

    for implementation, owner in IMPLEMENTATION_IDS.items():
        if owned[implementation] == 0:
            bad.append(f"{implementation!r} has no literal in its owner {owner}")
    return bad


RETIRED_REPORT_FORMAT = re.compile(
    r"cozy\.(?:jobs\.StructuralCensus|runtime\.(?:leaf_(?:fill|forward)|"
    r"mxfp8_(?:delta|lane_ranking|native|rowwise_derivation|w8a8)|"
    r"proof\.ModelContractState))/\d+"
)


def fence_report_format_tags() -> list[str]:
    """Proof-local state and reports are path/command typed, never global document formats."""
    bad: list[str] = []
    roots = (ROOT / "corpus", ROOT / "proofs", ROOT / "scripts", SRC)
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if retired := RETIRED_REPORT_FORMAT.search(line):
                    bad.append(
                        f"{path.relative_to(ROOT)}:{lineno}: retired proof/report format "
                        f"{retired.group(0)!r}; keep the plain typed data and let its "
                        "command/path establish the local shape"
                    )
    return bad


#: cr-042: the address-class attributes only `internal/egress.py` may name. Two downloaders
#: each grew their own chain and the two differed; this is the fence that stops a third.
ADDRESS_CLASSES = re.compile(
    r"\.is_(?:private|loopback|link_local|multicast|reserved|unspecified)\b"
)
ADDRESS_PREDICATE_OWNER = "internal/egress.py"


def fence_one_address_predicate() -> list[str]:
    """cr-042: ONE private/unroutable-address predicate, in the module that owns egress."""
    bad: list[str] = []
    for path in py_files():
        rel = path.relative_to(SRC).as_posix()
        if rel == ADDRESS_PREDICATE_OWNER:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if ADDRESS_PREDICATE_OWNER in line or ADDRESS_CLASSES.search(line) is None:
                continue
            bad.append(
                f"{path.relative_to(ROOT)}:{lineno}: classifies an address itself; "
                f"`egress.blocked()` is the ONE predicate every downloader calls — a second "
                "chain is how two downloaders came to disagree (cr-042)"
            )
    owner = (SRC / ADDRESS_PREDICATE_OWNER).read_text()
    if "def blocked(address: str) -> str:" not in owner:
        bad.append(f"src/cozy_runtime/{ADDRESS_PREDICATE_OWNER}: `blocked()` is gone")
    return bad


def fence_construction_fill_order() -> list[str]:
    """cr-041: stored-header presentation is never a Runtime traversal authority."""
    bad: list[str] = []
    source = (SRC / "internal" / "fill.py").read_text()
    if re.search(r"self\.order\s*=", source):
        bad.append(
            "src/cozy_runtime/internal/fill.py: retains a second `self.order` authority; "
            "the enqueue transaction is the construction traversal"
        )
    if "rows = [self.plan[key] for key in self.enqueued]" not in source:
        bad.append(
            "src/cozy_runtime/internal/fill.py: the TensorFS traversal no longer comes "
            "directly from construction-order enqueues"
        )
    if "order = _require_construction_order(plan.order, component)" not in source:
        bad.append(
            "src/cozy_runtime/internal/fill.py: a non-construction TensorFS plan no longer "
            "refuses before copies"
        )
    if 'self.stats.setdefault("traversals", [])' not in source:
        bad.append(
            "src/cozy_runtime/internal/fill.py: exact traversal evidence is no longer banked"
        )
    derivation = (SRC / "internal" / "derive.py").read_text()
    if "tuple(sorted(rich.sets" in derivation:
        bad.append(
            "src/cozy_runtime/internal/derive.py: alphabetically sorts DestinationSets "
            "instead of preserving the Loader census"
        )
    if "destination_sets=_ordered_destination_sets(shallow, rich)" not in derivation:
        bad.append(
            "src/cozy_runtime/internal/derive.py: MCC order no longer comes from the "
            "Loader's state_dict census"
        )
    proof = (ROOT / "scripts" / "mxfp8-live.py").read_text()
    for spelling in ("json.loads(book.header_bytes)", "json.loads(raw)"):
        if spelling in proof:
            bad.append(f"scripts/mxfp8-live.py: {spelling} bypasses TensorFS's CozyTensors codec")
    for name in ("leaf-live.py", "mxfp8-live.py", "quant-live.py", "leaf-pod.py", "mxfp8-pod.py"):
        producer = (ROOT / "scripts" / name).read_text()
        if '"--order"' not in producer:
            bad.append(
                f"scripts/{name}: an artifact-producing ingest does not supply an exact "
                "construction-order document"
            )
    return bad


def fence_dynamic_pods_skip_boot_warm() -> list[str]:
    """The fixed product worker never runs package inference before dispatchability."""

    path = SRC / "cli" / "runtime_worker.py"
    tree = ast.parse(path.read_text())
    command = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_worker_options"
        ),
        None,
    )
    if command is None:
        return [f"{path.relative_to(ROOT)}: fixed product-worker options are gone"]
    no_warm = [
        keyword.value
        for node in ast.walk(command)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "no_warm"
    ]
    if not any(isinstance(value, ast.Constant) and value.value is True for value in no_warm):
        return [
            f"{path.relative_to(ROOT)}: fixed product worker omits `no_warm=True`; "
            "package inference can block activation before DISPATCHABLE"
        ]
    return []


def fence_worker_control_memory_state() -> list[str]:
    """Every WorkerControl server is process-local; its caller owns restart recovery."""

    bad: list[str] = []
    session = (SRC / "internal" / "worker" / "session.py").read_text()
    if "self.records = WorkerRecords()" not in session:
        bad.append(
            "src/cozy_runtime/internal/worker/session.py: memory-only worker state no longer "
            "uses one WorkerRecords value"
        )
    for spelling in ("persistent_state", "restore_ownership(", ".recover()"):
        if spelling in session:
            bad.append(
                "src/cozy_runtime/internal/worker/session.py: WorkerControl retained "
                f"restart state through {spelling!r}"
            )
    runtime_worker = (SRC / "cli" / "runtime_worker.py").read_text()
    if "persistent_state" in runtime_worker:
        bad.append(
            "src/cozy_runtime/cli/runtime_worker.py: persistence is a launch option; all "
            "WorkerControl children must be process-local"
        )
    records = (SRC / "internal" / "worker" / "records.py").read_text()
    for spelling in ("Path", "open(", "os.", "fsync", "_load", "torn_tail"):
        if spelling in records:
            bad.append(
                "src/cozy_runtime/internal/worker/records.py: same-process records retained "
                f"disk machinery through {spelling!r}"
            )
    forbidden = ("journal.log", "cozy_runtime.internal.journal", "Journal(", "--instance-id")
    paths = [
        *SRC.rglob("*.py"),
        *(ROOT / "scripts").glob("*.py"),
        *(ROOT / "proofs").rglob("*.py"),
    ]
    for path in paths:
        if path.name.startswith("worker_pb2"):
            continue
        source = path.read_text()
        for spelling in forbidden:
            if spelling in source:
                bad.append(
                    f"{path.relative_to(ROOT)}: retired Runtime recovery surface {spelling!r}"
                )
    return bad


def fence_delivery_plan_observation() -> list[str]:
    """Delivery narration consumes the current plural kernel census, never retired keys."""

    path = SRC / "internal" / "worker" / "session.py"
    source = path.read_text()
    bad: list[str] = []
    stale = re.compile(r"delivery\[['\"](?:cost_class|kernel)['\"]\]")
    for lineno, line in enumerate(source.splitlines(), 1):
        if stale.search(line):
            bad.append(
                f"{path.relative_to(ROOT)}:{lineno}: delivery observation reads a retired "
                "singular/cost key; Plan.wire_document carries `kernels` and no cost_class"
            )
    if 'self.note("delivery", _delivery_observation(delivery))' not in source:
        bad.append(
            f"{path.relative_to(ROOT)}: prepare no longer sends delivery narration through "
            "the total current-schema renderer"
        )
    if re.search(r"census\(['\"]kernels['\"]\)", source) is None:
        bad.append(f"{path.relative_to(ROOT)}: current `kernels` census is not observed")
    return bad


def fence_pod_media_grant_root() -> list[str]:
    """The host-path vector is the one pod-supervisor/Runtime media-root authority."""

    path = SRC / "cli" / "runtime_worker.py"
    vector_path = SRC / "vectors" / "runtime-worker-host-paths.json"
    vector = json.loads(vector_path.read_text())
    if vector.get("media_root") != "/var/lib/cozy/media":
        return [
            f"{vector_path.relative_to(ROOT)}: media_root must equal pod-supervisor's exact "
            "`/var/lib/cozy/media` root"
        ]
    if "grant_roots=(str(host_paths.MEDIA_ROOT),)" not in path.read_text():
        return [
            f"{path.relative_to(ROOT)}: fixed worker no longer derives grant_roots from "
            "host_paths.MEDIA_ROOT"
        ]
    return []


def fence_measured_backend_version() -> list[str]:
    """Installed torch metadata is the torch-free backend authority, never driver maximum."""

    path = SRC / "internal" / "worker" / "session.py"
    source = path.read_text()
    tree = ast.parse(source)
    worker = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker"),
        None,
    )
    resources = next(
        (
            node
            for node in (worker.body if worker is not None else [])
            if isinstance(node, ast.FunctionDef) and node.name == "resources"
        ),
        None,
    )
    body = ast.get_source_segment(source, resources) if resources is not None else None
    bad: list[str] = []
    if body is None:
        return [f"{path.relative_to(ROOT)}: Worker.resources is gone"]
    if "backend_version=facts.cuda_version" not in body.replace(" ", ""):
        bad.append(
            f"{path.relative_to(ROOT)}: WorkerResources.backend_version does not come "
            "directly from HostFacts.cuda_version"
        )

    facts_path = SRC / "internal" / "hostfacts.py"
    facts_source = facts_path.read_text()
    facts_tree = ast.parse(facts_source)
    for node in ast.walk(facts_tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        if any(name == "torch" or name.startswith("torch.") for name in names):
            bad.append(
                f"{facts_path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}: imports torch"
            )
    helper = next(
        (
            node
            for node in facts_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_installed_cuda_version"
        ),
        None,
    )
    helper_body = ast.get_source_segment(facts_source, helper) if helper is not None else None
    if helper_body is None:
        bad.append(f"{facts_path.relative_to(ROOT)}: installed CUDA metadata reader is gone")
        return bad
    if "subprocess" in helper_body or "nvidia-smi" in helper_body:
        bad.append(
            f"{facts_path.relative_to(ROOT)}: installed backend version reaches the driver; "
            "nvidia-smi reports a maximum, not the image toolkit"
        )
    for spelling in ("sysconfig.get_paths()", '"torch" / "version.py"', "ast.parse"):
        if spelling not in helper_body:
            bad.append(
                f"{facts_path.relative_to(ROOT)}: installed backend reader no longer uses "
                f"its required {spelling} boundary"
            )
    if "def _cuda_version(" in facts_source:
        bad.append(f"{facts_path.relative_to(ROOT)}: retired driver-maximum CUDA reader returned")
    return bad


def fence_typed_model_binding() -> list[str]:
    """Package defaults name a `model`; the retired generic binding key cannot return."""

    bad: list[str] = []
    paths = [
        ROOT / "proofs" / "fixtures" / "tiny-models" / "package.toml",
        ROOT / "corpus" / "package" / "package.toml",
        ROOT / "corpus" / "pipeline" / "package.toml",
    ]
    for path in paths:
        document = tomllib.loads(path.read_text())
        bindings = document.get("bindings", {})
        if not isinstance(bindings, dict):
            bad.append(f"{path.relative_to(ROOT)}: [bindings] is not a table")
            continue
        for name, value in bindings.items():
            if not isinstance(value, dict) or not isinstance(value.get("model"), str):
                bad.append(f"{path.relative_to(ROOT)}: binding {name!r} does not name its model")
            if "repo" in value:
                bad.append(
                    f"{path.relative_to(ROOT)}: binding {name!r} uses retired generic repo key"
                )
    return bad


FENCES = {
    "prelaunch-version": fence_prelaunch_version,
    "python-abi": fence_python_abi,
    "forbidden-dependency": fence_forbidden_dependency,
    "reviewed-specs-exist": fence_reviewed_specs_exist,
    "descriptor-minimality": fence_descriptor_minimality,
    "one-attempt-kernel": fence_one_attempt_kernel,
    "canonical-numbers": fence_canonical_numbers,
    "base-ownership-vectors": fence_base_ownership_vectors,
    "tensorhub-package-release": fence_tensorhub_package_release,
    "torch-free-import": fence_torch_free_import,
    "package-imports": fence_package_imports,
    "author-import-boundary": fence_author_import_boundary,
    "env-authority": fence_env_authority,
    "single-config": fence_single_config,
    "no-runtime-database": fence_no_runtime_database,
    "no-interactive": fence_no_interactive,
    "excluded-keywords": fence_excluded_keywords,
    "no-identifiers-in-package-code": fence_no_identifiers,
    "timing-decides-nothing": fence_timing_decides_nothing,
    "package-owns-no-residency": fence_package_owns_no_residency,
    "package-owns-no-media-decode": fence_package_owns_no_media_decode,
    "document-kinds": fence_document_kinds,
    "qualification-selection": fence_qualification_selection,
    "development-placement": fence_development_placement,
    "implementation-ids": fence_implementation_ids,
    "report-format-tags": fence_report_format_tags,
    "construction-fill-order": fence_construction_fill_order,
    "one-address-predicate": fence_one_address_predicate,
    "dynamic-pods-skip-boot-warm": fence_dynamic_pods_skip_boot_warm,
    "worker-control-memory-state": fence_worker_control_memory_state,
    "delivery-plan-observation": fence_delivery_plan_observation,
    "pod-media-grant-root": fence_pod_media_grant_root,
    "measured-backend-version": fence_measured_backend_version,
    "typed-model-binding": fence_typed_model_binding,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the runtime's architecture fences")
    parser.add_argument("--only", action="append", choices=FENCES, default=[])
    parser.add_argument("--exclude", action="append", choices=FENCES, default=[])
    args = parser.parse_args(argv)
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
