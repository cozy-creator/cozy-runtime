#!/usr/bin/env python3
"""The installed Runtime against the installed TensorFS, in a fresh venv (`ci.yaml`).

1. Every `cozy_runtime` module imports. Only an optional dependency this venv lacks (torch,
   av, ...) may stop one; a missing or mismatched `tensorfs` name never may.
2. Every TensorFS name the Runtime spells (`from tensorfs... import x`, `tensorfs.x`, an
   imported TensorFS module's `m.x`) resolves in the installed TensorFS: by import where
   that imports here, else in the module's own source or stub.

Run as `python -I checks/import_smoke.py` so both packages come from site-packages. Exits 1
naming each failure.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path

OURS = ("cozy_runtime", "tensorfs")


def _installed(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"{name} is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def import_all() -> list[str]:
    failures: list[str] = []
    absent: set[str] = set()
    package = importlib.import_module("cozy_runtime")
    for info in pkgutil.walk_packages(package.__path__, "cozy_runtime."):
        if info.name.rsplit(".", 1)[-1] == "__main__":
            continue
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").partition(".")[0]
            if missing in OURS or not missing:
                failures.append(f"{info.name}: {exc}")
            else:
                absent.add(missing)
        except Exception as exc:  # a mismatched name, a raising import: the pair is broken
            failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
    print(f"imported cozy_runtime; optional dependencies not installed: {sorted(absent)}")
    return failures


def _module_path(module: str, root: Path) -> Path:
    return root.joinpath(*module.split(".")[1:])


def _is_module(module: str, root: Path) -> bool:
    base = _module_path(module, root)
    return base.is_dir() or any(base.with_suffix(s).exists() for s in (".py", ".pyi"))


def _names(module: str, root: Path, seen: set[str]) -> set[str] | None:
    """Top-level names `module` defines in its source or stub (stars followed); None: absent."""
    if module in seen:
        return set()
    seen.add(module)
    base = _module_path(module, root)
    files = [p for p in (base / "__init__.py", base / "__init__.pyi") if p.exists()]
    files += [p for p in (base.with_suffix(".py"), base.with_suffix(".pyi")) if p.exists()]
    if not files and not base.is_dir():
        return None
    names = {p.stem for p in base.iterdir()} if base.is_dir() else set()
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
            elif isinstance(node, ast.TypeAlias):
                names.add(node.name.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        names.add((alias.asname or alias.name).partition(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        names |= _names(_star(module, path, node), root, seen) or set()
    return names


def _star(module: str, path: Path, node: ast.ImportFrom) -> str:
    """The module a `from ... import *` in `module`'s file names."""
    if node.level == 0:
        return node.module or ""
    parts = (module if path.stem == "__init__" else module.rpartition(".")[0]).split(".")
    return ".".join([*parts[: len(parts) - node.level + 1], *filter(None, [node.module])])


def resolves(module: str, name: str, root: Path) -> bool:
    """By import where `module` imports here; else (an optional dependency it needs is absent)
    in its own source or stub."""
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        missing = (exc.name or "").partition(".")[0] if isinstance(exc, ModuleNotFoundError) else ""
        if not missing or missing in OURS:
            return False
    else:
        if hasattr(imported, name):
            return True
        return importlib.util.find_spec(f"{module}.{name}") is not None
    return name in (_names(module, root, set()) or set())


def references(source: Path, root: Path) -> list[tuple[str, str, str]]:
    """(where, module, name) for each TensorFS name `source` spells."""
    tree = ast.parse(source.read_text())
    stored = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    aliases: dict[str, str] = {}
    found: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.partition(".")[0] != "tensorfs":
                continue
            for alias in node.names:
                found.append((f"{source}:{node.lineno}", node.module, alias.name))
                if _is_module(f"{node.module}.{alias.name}", root):
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.partition(".")[0] == "tensorfs":
                    aliases[alias.asname or alias.name.partition(".")[0]] = (
                        alias.name if alias.asname else "tensorfs"
                    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain, value = [node.attr], node.value
        while isinstance(value, ast.Attribute):
            chain.insert(0, value.attr)
            value = value.value
        if isinstance(value, ast.Name) and value.id in aliases and value.id not in stored:
            module = aliases[value.id]
            for part in chain:
                found.append((f"{source}:{node.lineno}", module, part))
                module = f"{module}.{part}"
                if not _is_module(module, root):
                    break
    return found


def tensorfs_names() -> list[str]:
    runtime, root = _installed("cozy_runtime"), _installed("tensorfs")
    failures: list[str] = []
    checked: set[tuple[str, str]] = set()
    for source in sorted(runtime.rglob("*.py")):
        for where, module, name in references(source, root):
            if (module, name) in checked:
                continue
            checked.add((module, name))
            if not resolves(module, name, root):
                failures.append(f"{where}: {module}.{name} is not in the installed TensorFS")
    print(f"{len(checked)} TensorFS names the Runtime spells, checked")
    return failures


def main() -> int:
    failures = import_all() + tensorfs_names()
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
