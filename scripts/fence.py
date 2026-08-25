#!/usr/bin/env python3
"""Architecture fences (boundaries.md, cozy-runtime.md §7). Not a test: a dependency scan.

Five fences, each provable red by planting the violation (`scripts/redarm.py`):
  forbidden-dependency · package-imports · env-authority · single-config · no-interactive
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


FENCES = {
    "forbidden-dependency": fence_forbidden_dependency,
    "package-imports": fence_package_imports,
    "env-authority": fence_env_authority,
    "single-config": fence_single_config,
    "no-interactive": fence_no_interactive,
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
