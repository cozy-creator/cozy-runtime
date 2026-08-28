#!/usr/bin/env python3
"""Architecture fences (boundaries.md, cozy-runtime.md §7). Not a test: a dependency scan.

Architecture fences, each provable red by planting the violation (`scripts/redarm.py`):
  forbidden-dependency · canonical-numbers · artifact-object-sets · torch-free-import
  package-imports
  env-authority · single-config · no-interactive · excluded-keywords
  no-identifiers-in-endpoint-code · one-attempt-kernel · timing-decides-nothing
  endpoint-owns-no-residency · endpoint-owns-no-media-decode · reviewed-specs-exist
  construction-fill-order
  one-address-predicate
  dynamic-pods-skip-boot-warm
  delivery-plan-observation
  pod-media-grant-root
  measured-backend-version
"""

from __future__ import annotations

import argparse
import ast
import json
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


def fence_endpoint_owns_no_media_decode() -> list[str]:
    """cr-012's one-decoder law: endpoint code consumes `MediaDecoder` values.

    Opening a container, reaching the runtime's private spool path, or calling a library
    path helper would create another fetch/decode authority with different bounds and clock
    semantics. Those spellings are absent from endpoint/model code, not merely discouraged.
    """
    bad: list[str] = []
    for path in endpoint_and_model_code():
        tree = ast.parse(path.read_text())
        rel = path.relative_to(ROOT)
        for name in imports(tree):
            if name == "av" or name.startswith("av.") or name == "PIL" or name.startswith("PIL."):
                bad.append(
                    f"{rel}: endpoint code imports {name!r} — hydrated media decodes only "
                    "through the Runtime-owned MediaDecoder"
                )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr in {"_local", "_path"}:
                bad.append(
                    f"{rel}:{node.lineno}: endpoint code reaches private media path "
                    f"{node.attr} — verified "
                    "input paths are Runtime-private"
                )
            elif node.attr == "demux":
                bad.append(
                    f"{rel}:{node.lineno}: endpoint code demuxes packets — only immutable "
                    "Runtime media events cross the author boundary"
                )
            elif node.attr == "from_file":
                bad.append(
                    f"{rel}:{node.lineno}: endpoint code calls from_file — a library path/URL "
                    "decoder is a second media authority"
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


def fence_artifact_object_sets() -> list[str]:
    """Reproduce the model-only ArtifactObjectSet bytes and digests from frozen vectors."""

    import hashlib
    import json

    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal import canonical, local, provisioned
    from cozy_runtime.internal.worker.plan import PlanRefusal

    bad: list[str] = []
    path = ROOT / "vectors" / "artifact-object-sets.jsonl"
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    names: set[str] = set()
    rendered: dict[str, bytes] = {}
    for line_number, line in enumerate(rows, 1):
        vector = json.loads(line)
        if not isinstance(vector, dict):
            bad.append(f"{path}:{line_number}: vector is not an object")
            continue
        name = vector.get("name")
        if not isinstance(name, str) or name in names:
            bad.append(f"{path}:{line_number}: vector name is absent or duplicated")
            continue
        names.add(name)
        document = vector.get("document")
        expected = vector.get("canonical")
        expected_digest = vector.get("digest")
        if (
            not isinstance(document, dict)
            or not isinstance(expected, str)
            or not isinstance(expected_digest, str)
        ):
            bad.append(f"{path}:{line_number}: vector fields are incomplete")
            continue
        raw = canonical.write(document)
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if raw.decode() != expected or digest != expected_digest:
            bad.append(
                f"{path}:{line_number}: wrote {raw.decode()!r} / {digest}, "
                f"expected {expected!r} / {expected_digest}"
            )
            continue
        try:
            provisioned.read_artifact_object_set(raw)
        except PlanRefusal as exc:
            bad.append(f"{path}:{line_number}: strict reader refused {exc.code}: {exc.detail}")
        rendered[name] = raw
    if not rows:
        bad.append(f"{path} is empty — a vector file that vanishes must not read green")
    elif names != {"empty", "one-model-root"}:
        bad.append(f"{path}: vector names are {sorted(names)}, want empty and one-model-root")
    if "empty" in rendered:
        local_bytes = local.weightless_release_documents(
            {
                "application": "vector:weightless",
                "entrypoints": [
                    {"hidden": False, "kind": "entrypoint", "models": [], "name": "health"}
                ],
            }
        ).artifact_object_set
        if local_bytes != rendered["empty"]:
            bad.append(f"{path}: the local weightless writer diverges from the empty vector")
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
#:   * the attempt engine is CONSTRUCTED in exactly one module — the worker session.
#:
#: Anything that wants to run an attempt therefore has to go through the worker, over
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
    """The committed example carries only the descriptor/1 facts with real readers."""
    path = ROOT / "examples" / "endpoint.descriptor.json"
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.relative_to(ROOT)} is not a readable descriptor: {exc}"]
    bad: list[str] = []
    expected = {"format", "application", "entrypoints", "jobs"}
    if set(document) != expected:
        bad.append(
            f"{path.relative_to(ROOT)} fields are {sorted(document)}, want {sorted(expected)}"
        )
    if document.get("format") != "cozy.endpoint.descriptor/1":
        bad.append(f"{path.relative_to(ROOT)} has the wrong format")
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
        "kind",
        "max_audio_channels",
        "max_audio_samples",
        "max_decoded_bytes",
        "max_pixels_per_frame",
        "max_video_frames",
        "placement",
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
                if key in forbidden:
                    bad.append(f"{where}.{key} is an unused descriptor fact")
                visit(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{where}[{index}]")

    visit(document, "descriptor")
    return bad


#: proto-007 (#616.a): the registry of document kinds this repo AUTHORS, each with its one
#: declaring module. A format name exists only for a document that crosses a repo/process
#: boundary AND is stored or digested; a provider `name()` is in the preimage of every
#: implementation_digest and therefore every plan digest, so it is registered too. A literal
#: outside this table is a new kind without its decision row.
DOCUMENT_KINDS = {
    "cozy.endpoint.descriptor/1": "internal/descriptor.py",
    "cozy.endpoint.EntrypointBindingPlan/1": "internal/provisioned.py",
    "cozy.endpoint.ResolvedModelBinding/1": "internal/provisioned.py",
    "cozy.fp8-rowwise.dequant/1": "internal/encoding/formats.py",
    "cozy.fp8-scaled-scalar.dequant/1": "internal/encoding/formats.py",
    "cozy.mxfp8.dequant/1": "internal/encoding/formats.py",
    "cozy.plain.verbatim/1": "internal/encoding/formats.py",
    "cozy.fp8-rowwise.native-leaf/1": "internal/encoding/leaves.py",
    "cozy.mxfp8.native-leaf/1": "internal/encoding/leaves.py",
    "cozy.pod-readiness/1": "internal/readiness.py",
    "cozy.runtime.AttemptPlan/1": "internal/worker/plan.py",
    "cozy.runtime.ConstructedGeneration/2": "internal/executor.py",
    "cozy.runtime.EndpointOverlayReceipt/1": "internal/overlay.py",
    "cozy.runtime.EnvironmentProofRequest/1": "cli/environment_proof.py",
    "cozy.runtime.JobDescriptor/1": "internal/descriptor.py",
    "cozy.runtime.Ledger/0": "internal/worker/ledger.py",
    "cozy.runtime.LocalArtifact/1": "internal/artifacts.py",
    "cozy.runtime.ModelConstructionContract/1": "internal/contract.py",
    "cozy.runtime.ModelConstructionProof/1": "cli/model_contract_proof.py",
    "cozy.runtime.ModelConstructionProofRequest/1": "cli/model_contract_proof.py",
    "cozy.runtime.plan-fact/1": "internal/planfacts.py",
    "cozy.runtime.probe/1": "internal/probe.py",
    "cozy.runtime.PublicationReceipt/1": "internal/worker/grants.py",
    "cozy.runtime.ResolvedExecutionPlan/1": "internal/resolution.py",
    "cozy.runtime.TriageWriteReceipt/1": "internal/worker/triage.py",
    "cozy.runtime.WitnessBoot/1": "internal/witness.py",
    "cozy.runtime.WitnessDeath/1": "internal/witness.py",
    "cozy.runtime.WorkerTriageBundle/1": "internal/worker/triage.py",
}
#: Kinds another repo authors and this one only READS (the wire, tensorhub, tensorfs).
#: Naming them is decoding, not authoring; the owner's fence polices the name.
FOREIGN_KIND_PREFIXES = ("cozy.worker.v1.", "tensorhub.", "tensorfs.", "cozytensors")
#: Modules that name a registered kind without declaring it (a second reader is fine; a
#: second DECLARATION is the drift this fence exists for).
KIND_READERS = {
    "cozy.endpoint.EntrypointBindingPlan/1": {"internal/local.py"},
    "cozy.runtime.Ledger/0": {"internal/local.py"},
    "cozy.runtime.ModelConstructionContract/1": {"internal/executor.py"},
}
KIND_LITERAL = re.compile(r'"((?:cozy|cozytensors|tensorhub|tensorfs)\.[A-Za-z0-9_.-]+/\d+)"')


def fence_document_kinds() -> list[str]:
    """proto-007: every document-kind literal this repo authors is in DOCUMENT_KINDS."""
    bad: list[str] = []
    for path in py_files():
        rel = path.relative_to(SRC).as_posix()
        if rel.endswith("_pb2.py"):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for kind in KIND_LITERAL.findall(line):
                if kind.startswith(FOREIGN_KIND_PREFIXES):
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
    """The fixed remote adapter must never run endpoint inference before dispatchability."""

    path = SRC / "cli" / "materialize_launch.py"
    tree = ast.parse(path.read_text())
    command = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_worker_command"
        ),
        None,
    )
    if command is None:
        return [f"{path.relative_to(ROOT)}: fixed dynamic-pod worker command is gone"]
    returned = [node.value for node in ast.walk(command) if isinstance(node, ast.Return)]
    if not any(
        isinstance(value, ast.List)
        and any(isinstance(item, ast.Constant) and item.value == "--no-warm" for item in value.elts)
        for value in returned
    ):
        return [
            f"{path.relative_to(ROOT)}: fixed dynamic-pod child argv omits `--no-warm`; "
            "endpoint inference can block activation before DISPATCHABLE"
        ]
    return []


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
    """The fixed adapter declares pod-supervisor/podmedia's one exact shared filesystem root."""

    path = SRC / "cli" / "materialize_launch.py"
    tree = ast.parse(path.read_text())
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MEDIA_ROOT" for target in node.targets
        )
    ]
    if len(assignments) != 1:
        return [
            f"{path.relative_to(ROOT)}: MEDIA_ROOT has {len(assignments)} authorities; "
            "pod-supervisor and Runtime share one fixed root"
        ]
    value = assignments[0].value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "Path"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Constant)
        and value.args[0].value == "/var/lib/cozy/media"
    ):
        return [
            f"{path.relative_to(ROOT)}: MEDIA_ROOT must equal pod-supervisor's exact "
            "`/var/lib/cozy/media` root"
        ]
    source = (
        ast.get_source_segment(
            path.read_text(),
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_worker_command"
            ),
        )
        or ""
    )
    if "grant_roots = str(MEDIA_ROOT)" not in source:
        return [
            f"{path.relative_to(ROOT)}: fixed worker argv no longer derives --grant-root "
            "from MEDIA_ROOT"
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


FENCES = {
    "forbidden-dependency": fence_forbidden_dependency,
    "reviewed-specs-exist": fence_reviewed_specs_exist,
    "descriptor-minimality": fence_descriptor_minimality,
    "one-attempt-kernel": fence_one_attempt_kernel,
    "canonical-numbers": fence_canonical_numbers,
    "artifact-object-sets": fence_artifact_object_sets,
    "torch-free-import": fence_torch_free_import,
    "package-imports": fence_package_imports,
    "env-authority": fence_env_authority,
    "single-config": fence_single_config,
    "no-interactive": fence_no_interactive,
    "excluded-keywords": fence_excluded_keywords,
    "no-identifiers-in-endpoint-code": fence_no_identifiers,
    "timing-decides-nothing": fence_timing_decides_nothing,
    "endpoint-owns-no-residency": fence_endpoint_owns_no_residency,
    "endpoint-owns-no-media-decode": fence_endpoint_owns_no_media_decode,
    "document-kinds": fence_document_kinds,
    "construction-fill-order": fence_construction_fill_order,
    "one-address-predicate": fence_one_address_predicate,
    "dynamic-pods-skip-boot-warm": fence_dynamic_pods_skip_boot_warm,
    "delivery-plan-observation": fence_delivery_plan_observation,
    "pod-media-grant-root": fence_pod_media_grant_root,
    "measured-backend-version": fence_measured_backend_version,
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
