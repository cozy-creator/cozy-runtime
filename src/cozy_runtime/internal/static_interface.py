"""The package interface from SOURCE: `ast` over the package's files, executing no package code.

INVARIANT (decision #713 — package code is UNTRUSTED): this module never imports, execs,
compiles or evaluates a package module. No `importlib`, no `sys.path`, no `exec`/`eval`/
`compile`, no annotation evaluation, no module top level. It reads text, parses it with
`ast`, follows the package's own module-scope import graph (plus cozy-runtime's own source
for `cozy_runtime.derive.*` and `cozy_runtime.models.*`), folds constants with its own
evaluator over literal and arithmetic nodes and values of the package's committed JSON assets
(`author.data_values`), and translates every interface-bearing declaration into real types built
from the CLOSED author vocabulary — `msgspec.defstruct`, `enum.Enum`, `typing.Literal`/
`Annotated`, cozy-runtime's own asset, marker and service classes. Rendering is then
`schema.render`, the SAME renderer the import path uses, so the two derivations cannot drift
by construction. `checks/architecture.py`'s `static-describe-executes-nothing` fence holds
this module's import list to that.

Anything outside the closed vocabulary — a name bound to no definition in the tree, a
computed value (call, comprehension, unknown attribute) in an interface position, a dynamic
base, a star import — is a typed refusal naming file:line and the construct. Never a guess.
"""

from __future__ import annotations

import ast
import configparser
import csv
import email.parser
import enum
import json
import re
import sysconfig
import tomllib
import types
import typing
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Literal

import msgspec

import cozy_runtime
import cozy_runtime.internal.invocable_interface as invocable_interface
import cozy_runtime.internal.package_interface as package_interface
import cozy_runtime.internal.static_invocable as static_invocable
from cozy_runtime import author
from cozy_runtime.author._markers import MODEL_DEFAULT, PREFLIGHT
from cozy_runtime.author._model import Model
from cozy_runtime.author._model_defaults import DefaultLadder, check_model_arguments, model_defaults
from cozy_runtime.author._script_metadata import byte_parameter, declarations, main_definition
from cozy_runtime.author._services import SERVICES
from cozy_runtime.author._signature import AssetsBinding, _classify, assets_binding, assets_payload
from cozy_runtime.author._walker import is_struct
from cozy_runtime.internal.bindings import read_table
from cozy_runtime.internal.canonical import Json

CONSTRUCT_CHARS = 96


class StaticRefusal(author.ConformanceError):
    """A construct the static reader cannot admit, named by file:line and source text."""

    default_code = "static_unresolvable"


Code = Literal["static_unresolvable", "static_computed", "static_unsupported", "static_module"]


# ------------------------------------------------------------------------- the modules


@dataclass(frozen=True, slots=True)
class Module:
    name: str
    path: Path
    source: str
    tree: ast.Module
    is_package: bool
    trusted: bool
    """cozy-runtime's own derive/model-library source: followed for types and constants,
    never scanned for registrations."""
    bindings: dict[str, ast.stmt] = field(default_factory=dict)

    @property
    def package(self) -> str:
        return self.name if self.is_package else self.name.rpartition(".")[0]


@dataclass(frozen=True, slots=True)
class Definition:
    module: Module
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class Assigned:
    module: Module
    stmt: ast.stmt
    value: ast.expr

    @property
    def identity(self) -> tuple[Path, int]:
        return self.module.path, self.stmt.lineno


@dataclass(frozen=True, slots=True)
class Installed:
    """The source of the OTHER cozy packages installed beside this runtime.

    A package may build its surface out of a dependency's declarations — a `Model` subclass,
    a shared request struct — and that dependency's source ships in its wheel. So the reader
    reads it, confined to distributions declaring a `cozy.application` entry point: the same
    fact `discovery.py` calls an app. torch declares none, so torch is never in the tree, and
    site-packages is never a root. Their modules are followed for TYPES only, never scanned
    for registrations: another package's callables are its own.
    """

    roots: tuple[Path, ...]
    """Editable installs: the source tree `direct_url.json` names."""
    files: frozenset[Path]
    """Wheel installs: exactly the `.py` files that distribution's RECORD lists."""


@dataclass(frozen=True, slots=True)
class Vocab:
    """One closed-vocabulary name: `author.ImageAsset`, `msgspec.Struct`, `typing.Literal`."""

    key: str


@dataclass(frozen=True, slots=True)
class ModuleRef:
    module: Module


@dataclass(frozen=True, slots=True)
class VocabModule:
    prefix: str


@dataclass(frozen=True, slots=True)
class External:
    """An import from outside the tree and the vocabulary. Refuses only when USED."""

    spelling: str


Resolved = Definition | Assigned | Vocab | ModuleRef | VocabModule | External

_VOCAB_MODULES = {
    "msgspec": "msgspec",
    "typing": "typing",
    "collections.abc": "abc",
    "enum": "enum",
    "cozy_runtime.author": "author",
}
_TRUSTED_ROOTS = ("cozy_runtime.derive", "cozy_runtime.models")


def _trusted_module(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _TRUSTED_ROOTS)


#: The entry-point group `discovery.py` reads. A distribution that declares one IS a cozy
#: package; every other distribution beside it — torch included — is not, and is never read.
_APPLICATION_GROUP = "[cozy.application]"
_BUILTIN_TYPES: dict[str, object] = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "object": object,
}
_ABC_TYPES: dict[str, object] = {"Sequence": Sequence, "Mapping": Mapping, "Set": Set}
_ENUM_BASES: dict[str, type[enum.Enum]] = {
    "Enum": enum.Enum,
    "IntEnum": enum.IntEnum,
    "StrEnum": enum.StrEnum,
}
#: Author-surface classes that translate to themselves in a TYPE position.
_AUTHOR_TYPES = (
    "Asset",
    "ImageAsset",
    "AudioAsset",
    "VideoAsset",
    "FileAsset",
    "Tree",
    "ModelArtifact",
    "Image",
    "DecodedVideo",
    "DecodedAudio",
    "Mixed",
    "Context",
    "Assets",
    "Settings",
    "Secrets",
    *(cls.__name__ for cls in SERVICES.values()),
)
#: Author-surface callables a folded constant may reach: each validates its own arguments.
#: `data_values` reads committed JSON through the same function as the imported app.
_AUTHOR_VALUES = (
    "AssetBound",
    "AssetLimits",
    "ImagePreparation",
    "WeightsOutput",
    "Bound",
    "Shape",
    "data_values",
)

#: Struct class keywords `msgspec.defstruct` takes verbatim. `rename` and `array_like` change
#: the wire and are refused; everything else here is either wire-neutral or passed through.
_STRUCT_KEYWORDS = frozenset(
    {
        "tag",
        "tag_field",
        "kw_only",
        "frozen",
        "forbid_unknown_fields",
        "omit_defaults",
        "eq",
        "order",
        "gc",
        "weakref",
        "dict",
        "cache_hash",
        "repr_omit_defaults",
    }
)
#: Constructors over a folded sequence; scalar min/max reduction is handled separately.
_SEQUENCE_BUILTINS: dict[str, Callable[[Sequence[object]], object]] = {
    "builtins.tuple": tuple,
    "builtins.list": list,
}
#: A fold carries no local names unless a comprehension binds its own target.
_NO_SCOPE: Mapping[str, object] = types.MappingProxyType({})
_MODEL_MEMBERS = frozenset(name for name in dir(Model) if not name.startswith("_"))
_ENTRYPOINT_KEYWORDS = frozenset(
    {
        "name",
        "preflight",
        "demand",
        "hidden",
        "memoize",
        "memo_version",
        "memo_dependencies",
        "defaults",
        "internal",
    }
)
_JOB_KEYWORDS = frozenset(
    {"name", "publishes", "emits_media", "weights", "demand", "defaults", "internal", "accelerator"}
)


# -------------------------------------------------------------------------- the reader


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    encoded_leaves: str
    fusion: str
    sequence_parallel: tuple[int, ...]
    components: dict[str, tuple[str, ...]]


class Reader:
    """One package tree, read lazily. Every method resolves by NAME, never by execution."""

    def __init__(self, project: Path, *, dependency_roots: Sequence[Path] = ()) -> None:
        self.project = project
        self.project_roots = (project / "src", project)
        self.roots = (*self.project_roots, *dependency_roots)
        self.runtime_root = Path(cozy_runtime.__file__).resolve().parent.parent
        self.installed = _installed_packages(_site_packages(self.runtime_root))
        self.modules: dict[str, Module] = {}
        self.structs: dict[tuple[Path, str], type[msgspec.Struct] | None] = {}
        self.enums: dict[tuple[Path, str], type[enum.Enum]] = {}
        self.models: dict[tuple[Path, str], ModelSpec] = {}
        self.type_aliases: set[tuple[Path, int]] = set()

    # ---------------------------------------------------------------- modules

    def module(self, name: str, *, where: tuple[Module, ast.AST] | None = None) -> Module:
        if (loaded := self.modules.get(name)) is not None:
            return loaded
        trusted = _trusted_module(name)
        roots = (self.runtime_root,) if trusted else self.roots
        parts = name.split(".")
        for root in roots:
            for path, is_package in (
                (root.joinpath(*parts).with_suffix(".py"), False),
                (root.joinpath(*parts, "__init__.py"), True),
            ):
                if path.is_file():
                    return self._load(name, path, is_package, trusted)
        if not trusted and (sibling := self._installed(parts)) is not None:
            # `trusted` is the same rule cozy_runtime.derive gets: followed for types and
            # constants, never scanned for registrations.
            return self._load(name, sibling[0], sibling[1], True)
        detail = f"module {name!r} is not a file under {' or '.join(str(r) for r in roots)}"
        if where is not None:
            self.refuse(where[0], where[1], detail, "static_module")
        raise StaticRefusal(detail, code="static_module")

    def _load(self, name: str, path: Path, is_package: bool, trusted: bool) -> Module:
        source = path.read_text()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise StaticRefusal(
                f"{self.spell(path)}:{exc.lineno}: {exc.msg}", code="static_module"
            ) from exc
        module = Module(name, path, source, tree, is_package, trusted)
        self.modules[name] = module
        for stmt in _module_scope(tree.body):
            for bound in _bound_names(stmt):
                module.bindings[bound] = stmt
        return module

    def _installed(self, parts: Sequence[str]) -> tuple[Path, bool] | None:
        """One module of another INSTALLED cozy package, or None. Never site-packages at large."""
        for root in (*self.installed.roots, self.runtime_root):
            for path, is_package in (
                (root.joinpath(*parts).with_suffix(".py"), False),
                (root.joinpath(*parts, "__init__.py"), True),
            ):
                if path.is_file() and (
                    root in self.installed.roots or path in self.installed.files
                ):
                    return path, is_package
        return None

    def application_module(self, name: str) -> Module:
        """The configured module, with the package `__init__`s Python would run first."""
        parts = name.split(".")
        for depth in range(1, len(parts)):
            prefix = ".".join(parts[:depth])
            for root in self.roots:
                if root.joinpath(*parts[:depth], "__init__.py").is_file():
                    self.module(prefix)
                    break
        return self.module(name)

    def reachable(self, start: Module) -> list[Module]:
        """Package modules the application module imports at module scope, transitively."""
        seen: dict[str, Module] = {}
        pending = [start]
        while pending:
            module = pending.pop()
            if module.name in seen or module.trusted:
                continue
            seen[module.name] = module
            for stmt in _module_scope(module.tree.body):
                for target in self._imported_modules(module, stmt):
                    if target is not None and target.name not in seen:
                        pending.append(target)
        return list(seen.values())

    def _imported_modules(self, module: Module, stmt: ast.stmt) -> list[Module | None]:
        if isinstance(stmt, ast.Import):
            return [self._module_or_none(alias.name) for alias in stmt.names]
        if isinstance(stmt, ast.ImportFrom):
            base = self._import_base(module, stmt)
            if base is None:
                return []
            found = [self._module_or_none(base)]
            found += [self._module_or_none(f"{base}.{alias.name}") for alias in stmt.names]
            return found
        return []

    def _module_or_none(self, name: str) -> Module | None:
        top = name.split(".")[0]
        # Dependency source is followed when a declared type needs it. Importing
        # torch at module scope must not scan its entire tree for app registrations.
        if name in _VOCAB_MODULES or not any(
            (root / top).is_dir() or (root / f"{top}.py").is_file() for root in self.project_roots
        ):
            return None
        try:
            return self.module(name)
        except StaticRefusal:
            return None

    def _tree_tops(self, name: str) -> set[str]:
        top = name.split(".")[0]
        for root in self.roots:
            if (root / top).is_dir() or (root / f"{top}.py").is_file():
                return {top}
        return {top} if self._installed((top,)) is not None else set()

    def _import_base(self, module: Module, stmt: ast.ImportFrom) -> str | None:
        if stmt.level == 0:
            return stmt.module
        package = module.package
        for _ in range(stmt.level - 1):
            package = package.rpartition(".")[0]
        if stmt.module:
            return f"{package}.{stmt.module}" if package else stmt.module
        return package or None

    # ------------------------------------------------------------- resolution

    def resolve(self, module: Module, node: ast.expr) -> Resolved:
        if isinstance(node, ast.Name):
            return self.lookup(module, node.id, node)
        if isinstance(node, ast.Attribute):
            base = self.resolve(module, node.value)
            if isinstance(base, ModuleRef):
                return self.lookup(base.module, node.attr, node, origin=module)
            if isinstance(base, VocabModule):
                return Vocab(f"{base.prefix}.{node.attr}")
            if isinstance(base, External):
                return External(f"{base.spelling}.{node.attr}")
            self.refuse(module, node, "an attribute of something that is not a module")
        self.refuse(module, node, "not a name")

    def lookup(
        self, module: Module, name: str, node: ast.AST, *, origin: Module | None = None
    ) -> Resolved:
        stmt = module.bindings.get(name)
        if stmt is None:
            if name in _BUILTIN_TYPES or name in ("min", "max"):
                return Vocab(f"builtins.{name}")
            if module.is_package:
                try:
                    return ModuleRef(self.module(f"{module.name}.{name}"))
                except StaticRefusal:
                    pass
            self.refuse(
                origin or module,
                node,
                f"{name!r} is not bound at module scope of {self.spell(module.path)}"
                + self._local_import_hint(module, name),
            )
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound = alias.asname or alias.name.partition(".")[0]
                if bound == name:
                    target = alias.name if alias.asname else alias.name.partition(".")[0]
                    return self._module_binding(module, target, stmt)
        if isinstance(stmt, ast.ImportFrom):
            base = self._import_base(module, stmt)
            for alias in stmt.names:
                if alias.name == "*":
                    self.refuse(module, stmt, "a star import", "static_unsupported")
                if (alias.asname or alias.name) == name:
                    return self._from_binding(module, base, alias.name, stmt)
        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return Definition(module, stmt)
        if isinstance(stmt, ast.Assign):
            assert stmt.value is not None
            return Assigned(module, stmt, stmt.value)
        if isinstance(stmt, ast.AnnAssign):
            assert stmt.value is not None
            return Assigned(module, stmt, stmt.value)
        self.refuse(module, stmt, "a binding form the static reader does not follow")

    def _module_binding(self, module: Module, target: str, stmt: ast.stmt) -> Resolved:
        if target in _VOCAB_MODULES:
            return VocabModule(_VOCAB_MODULES[target])
        if _trusted_module(target) or self._tree_tops(target):
            return ModuleRef(self.module(target, where=(module, stmt)))
        return External(target)

    def _from_binding(
        self, module: Module, base: str | None, name: str, stmt: ast.ImportFrom
    ) -> Resolved:
        if base is None:
            self.refuse(module, stmt, "a relative import above the package", "static_module")
        if base in _VOCAB_MODULES:
            return Vocab(f"{_VOCAB_MODULES[base]}.{name}")
        if base.startswith("cozy_runtime.") and not _trusted_module(base):
            return External(f"{base}.{name}")
        if not (_trusted_module(base) or self._tree_tops(base)):
            return External(f"{base}.{name}")
        source = self.module(base, where=(module, stmt))
        if source.bindings.get(name) is stmt:
            # `from . import operations` in the package __init__ binds the child
            # module, not this same import statement recursively.
            return ModuleRef(self.module(f"{base}.{name}", where=(module, stmt)))
        if name in source.bindings:
            return self.lookup(source, name, stmt, origin=module)
        return ModuleRef(self.module(f"{base}.{name}", where=(module, stmt)))

    def _local_import_hint(self, module: Module, name: str) -> str:
        for node in ast.walk(module.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)) and any(
                    (alias.asname or alias.name.partition(".")[0]) == name for alias in inner.names
                ):
                    return (
                        f" (line {inner.lineno} imports it inside `def {node.name}` — every "
                        "import goes at the top of the file)"
                    )
        return ""

    # ------------------------------------------------------------- constants

    def fold(
        self, module: Module, node: ast.expr, scope: Mapping[str, object] = _NO_SCOPE
    ) -> object:
        """The value of a literal/arithmetic expression, through module-level names.

        `scope` carries a comprehension's own target and nothing else is ever local.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            items = [self.fold(module, item, scope) for item in node.elts]
            return tuple(items) if isinstance(node, ast.Tuple) else items
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                self.refuse(module, node, "a dict splat", "static_computed")
            return {
                self.fold(module, key, scope): self.fold(module, value, scope)
                for key, value in zip(node.keys, node.values, strict=True)
                if key is not None
            }
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            return self._comprehension(module, node, scope)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
            operand = self.fold(module, node.operand, scope)
            if isinstance(operand, bool) or not isinstance(operand, (int, float)):
                self.refuse(module, node, "unary arithmetic on a non-number", "static_computed")
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            return ~int(operand)
        if isinstance(node, ast.BinOp):
            return self._arithmetic(module, node, scope)
        if isinstance(node, ast.Name) and node.id in scope:
            return scope[node.id]
        if isinstance(node, ast.Name) and node.id == "__file__":
            # The one dunder a data asset is addressed from: this module's own file.
            return str(module.path)
        if isinstance(node, ast.Attribute):
            base = self.resolve(module, node.value)
            if (
                isinstance(base, Definition)
                and isinstance(base.node, ast.ClassDef)
                and self.class_kind(base.module, base.node) == "enum"
            ):
                built = typing.cast(type[enum.Enum], self.class_type(base.module, base.node, node))
                if node.attr in built.__members__:
                    return built.__members__[node.attr]
                self.refuse(module, node, "unknown enumeration member")
        if isinstance(node, (ast.Name, ast.Attribute)):
            found = self.resolve(module, node)
            if isinstance(found, Assigned):
                return self.fold(found.module, found.value)
            if isinstance(found, Vocab) and found.key == "author.DEFAULT_DECODE_LIMITS":
                return author.DEFAULT_DECODE_LIMITS
            if isinstance(found, Vocab) and found.key == "author.MAX_WEIGHTS_NEW_BYTES":
                return author.MAX_WEIGHTS_NEW_BYTES
            self.refuse(module, node, "not a module-level constant")
        if isinstance(node, ast.Call):
            callee = self.resolve(module, node.func)
            if isinstance(callee, Vocab) and callee.key in ("builtins.min", "builtins.max"):
                if len(node.args) != 1 or node.keywords:
                    self.refuse(
                        module, node, "min/max needs one literal sequence", "static_computed"
                    )
                values = self.fold(module, node.args[0], scope)
                if (
                    not isinstance(values, (tuple, list))
                    or not values
                    or any(type(value) not in (int, float, bool, str) for value in values)
                ):
                    self.refuse(
                        module, node, "min/max needs a nonempty literal sequence", "static_computed"
                    )
                try:
                    literals = typing.cast(Sequence[typing.Any], values)
                    return min(literals) if callee.key == "builtins.min" else max(literals)
                except TypeError:
                    self.refuse(
                        module, node, "min/max literal values are not comparable", "static_computed"
                    )
            if isinstance(callee, Vocab) and callee.key in _SEQUENCE_BUILTINS:
                if len(node.args) != 1 or node.keywords:
                    self.refuse(module, node, "a sequence built from several arguments")
                sequence = self.fold(module, node.args[0], scope)
                if not isinstance(sequence, (tuple, list, dict)):
                    self.refuse(module, node, "a sequence built from a non-sequence")
                return _SEQUENCE_BUILTINS[callee.key](list(sequence))
            if isinstance(callee, Vocab) and callee.key in (
                *(f"author.{name}" for name in _AUTHOR_VALUES),
                "msgspec.Meta",
            ):
                if callee.key == "author.Shape":
                    # A demand-axis marker: its tables are code, and it is not interface content.
                    return author.Shape(pixels={})
                constructor = _vocab_object(callee.key)
                assert callable(constructor)
                args = [self.fold(module, arg, scope) for arg in node.args]
                kwargs = {
                    keyword.arg: self.fold(module, keyword.value, scope)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                if len(kwargs) != len(node.keywords):
                    self.refuse(module, node, "a keyword splat", "static_computed")
                try:
                    return constructor(*args, **kwargs)
                except (author.AuthorError, OSError, TypeError, ValueError) as exc:
                    self.refuse(module, node, str(exc), "static_unsupported")
            self.refuse(
                module, node, "a call: its value exists only at run time", "static_computed"
            )
        self.refuse(
            module, node, "an expression the constant folder does not evaluate", "static_computed"
        )

    def _comprehension(
        self, module: Module, node: ast.GeneratorExp | ast.ListComp, scope: Mapping[str, object]
    ) -> object:
        """One comprehension over a folded sequence: `tuple(F(x) for x in CONSTANT)`.

        Deterministic by construction: the element is folded exactly as any other constant
        is, with the clause's own target bound, and nothing of the package runs.
        """
        if len(node.generators) != 1:
            self.refuse(module, node, "a comprehension with several clauses", "static_computed")
        clause = node.generators[0]
        if clause.ifs or clause.is_async or not isinstance(clause.target, ast.Name):
            self.refuse(
                module, node, "a filtered, async or unpacking comprehension", "static_computed"
            )
        items = self.fold(module, clause.iter, scope)
        if not isinstance(items, (tuple, list, dict)):
            self.refuse(
                module, clause.iter, "a comprehension over a non-sequence", "static_computed"
            )
        folded = [self.fold(module, node.elt, {**scope, clause.target.id: item}) for item in items]
        return folded if isinstance(node, ast.ListComp) else tuple(folded)

    def _arithmetic(self, module: Module, node: ast.BinOp, scope: Mapping[str, object]) -> object:
        left = self.fold(module, node.left, scope)
        right = self.fold(module, node.right, scope)
        ints = (
            isinstance(left, int)
            and isinstance(right, int)
            and not isinstance(left, bool)
            and not isinstance(right, bool)
        )
        numbers = ints or all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in (left, right)
        )
        if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            return left + right
        if not numbers:
            self.refuse(module, node, "arithmetic on non-numbers", "static_computed")
        assert isinstance(left, (int, float)) and isinstance(right, (int, float))
        ops: dict[type[ast.operator], Callable[[int | float, int | float], object]] = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a**b,
        }
        int_ops: dict[type[ast.operator], Callable[[int, int], int]] = {
            ast.LShift: lambda a, b: a << b,
            ast.RShift: lambda a, b: a >> b,
            ast.BitOr: lambda a, b: a | b,
            ast.BitAnd: lambda a, b: a & b,
            ast.BitXor: lambda a, b: a ^ b,
        }
        if (op := ops.get(type(node.op))) is not None:
            return op(left, right)
        if ints and (int_op := int_ops.get(type(node.op))) is not None:
            return int_op(int(left), int(right))
        self.refuse(
            module, node, "an operator the constant folder does not evaluate", "static_computed"
        )

    # ----------------------------------------------------------------- types

    def translate(self, module: Module, node: ast.expr) -> object:
        """A type expression as the real typing object the shared renderer reads."""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return type(None)
            if isinstance(node.value, str):
                return self.translate(module, ast.parse(node.value, mode="eval").body)
            self.refuse(module, node, "a literal where a type belongs")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return typing.Union[  # noqa: UP007 — built from values, not spelled
                self.translate(module, node.left), self.translate(module, node.right)
            ]
        if isinstance(node, ast.Subscript):
            return self._subscript(module, node)
        if isinstance(node, (ast.Name, ast.Attribute)):
            found = self.resolve(module, node)
            if isinstance(found, Vocab):
                return self._vocab_type(module, node, found)
            if isinstance(found, Assigned):
                if found.identity in self.type_aliases:
                    self.refuse(module, node, "a cyclic type alias", "static_unsupported")
                self.type_aliases.add(found.identity)
                try:
                    return self.translate(found.module, found.value)
                finally:
                    self.type_aliases.remove(found.identity)
            if isinstance(found, Definition) and isinstance(found.node, ast.ClassDef):
                return self.class_type(found.module, found.node, node)
            if isinstance(found, External):
                self.refuse(module, node, f"{found.spelling} is outside the closed vocabulary")
            self.refuse(module, node, "not a type")
        self.refuse(module, node, "not a type expression")

    def _vocab_type(self, module: Module, node: ast.expr, found: Vocab) -> object:
        prefix, _, name = found.key.partition(".")
        if prefix == "builtins" and name in _BUILTIN_TYPES:
            return _BUILTIN_TYPES[name]
        if prefix == "abc" and name in _ABC_TYPES:
            return _ABC_TYPES[name]
        if prefix == "typing" and name in ("Any", "ClassVar"):
            self.refuse(module, node, f"typing.{name} is not an interface type")
        if prefix == "author" and name in _AUTHOR_TYPES:
            return getattr(author, name)
        self.refuse(module, node, f"{found.key} is not an interface type")

    def _subscript(self, module: Module, node: ast.Subscript) -> object:
        found = self.resolve(module, node.value)
        args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if isinstance(found, Vocab):
            if found.key == "typing.Annotated":
                inner = self.translate(module, args[0])
                markers = []
                for marker in args[1:]:
                    value = self.fold(module, marker)
                    if not isinstance(value, author.Shape):
                        markers.append(value)
                return Annotated[(inner, *markers)] if markers else inner
            if found.key == "typing.Literal":
                values: list[object] = []
                for arg in args:
                    value = self.fold(module, arg)
                    values.extend(value) if isinstance(value, tuple) else values.append(value)
                return Literal[tuple(values)]
            if found.key == "typing.Optional":
                return typing.Optional[self.translate(module, args[0])]  # noqa: UP045
            if found.key == "typing.Union":
                return typing.Union[tuple(self.translate(module, arg) for arg in args)]  # noqa: UP007
            if found.key == "author.Preflight":
                return Annotated[self.translate(module, args[0]), PREFLIGHT]
            if found.key == "author.ModelDefault":
                return Annotated[self.translate(module, args[0]), MODEL_DEFAULT]
            generic = self._vocab_type(module, node.value, found)
            params = tuple(
                Ellipsis
                if isinstance(arg, ast.Constant) and arg.value is Ellipsis
                else self.translate(module, arg)
                for arg in args
            )
            return generic[params if len(params) != 1 else params[0]]  # type: ignore[index]
        self.refuse(module, node, "a subscript of something that is not a generic type")

    # --------------------------------------------------------------- classes

    def class_type(self, module: Module, node: ast.ClassDef, where: ast.AST) -> object:
        """A struct or enum, built once per class from its declaration."""
        key = (module.path, node.name)
        if key in self.enums:
            return self.enums[key]
        if key in self.structs:
            if self.structs[key] is None:
                self.refuse(module, where, f"{node.name} refers to itself", "static_unsupported")
            return self.structs[key]
        kind = self.class_kind(module, node)
        if kind == "enum":
            return self._enum(module, node)
        if kind == "struct":
            return self._struct(module, node)
        self.refuse(module, where, f"{node.name} is neither a msgspec.Struct nor an Enum")

    def class_kind(self, module: Module, node: ast.ClassDef) -> str:
        for base in node.bases:
            target = base.value if isinstance(base, ast.Subscript) else base
            found = self.resolve(module, target)
            if isinstance(found, Vocab):
                if found.key == "msgspec.Struct":
                    return "struct"
                if found.key == "author.Model":
                    return "model"
                if found.key.startswith("enum."):
                    return "enum"
                if found.key == "builtins.str":
                    continue
                self.refuse(module, base, f"{found.key} is not a base the static reader admits")
            if isinstance(found, Definition) and isinstance(found.node, ast.ClassDef):
                return self.class_kind(found.module, found.node)
            self.refuse(module, base, "a dynamic base", "static_unsupported")
        return "plain"

    def _enum(self, module: Module, node: ast.ClassDef) -> type[enum.Enum]:
        base: type[enum.Enum] = enum.Enum
        for candidate in node.bases:
            found = self.resolve(module, candidate)
            if isinstance(found, Vocab) and found.key.startswith("enum."):
                base = _ENUM_BASES.get(found.key.partition(".")[2], enum.Enum)
        members: list[tuple[str, object]] = []
        for stmt in node.body:
            target = _single_target(stmt)
            if target is None:
                continue
            assert isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None
            members.append((target, self.fold(module, stmt.value)))
        functional = typing.cast(Callable[[str, list[tuple[str, object]]], type[enum.Enum]], base)
        built = functional(node.name, members)
        self.enums[(module.path, node.name)] = built
        return built

    def _struct(self, module: Module, node: ast.ClassDef) -> type[msgspec.Struct]:
        key = (module.path, node.name)
        self.structs[key] = None
        bases: list[type[msgspec.Struct]] = []
        for base in node.bases:
            found = self.resolve(module, base)
            if isinstance(found, Vocab) and found.key == "msgspec.Struct":
                continue
            if isinstance(found, Definition) and isinstance(found.node, ast.ClassDef):
                parent = self.class_type(found.module, found.node, base)
                assert is_struct(parent)
                bases.append(typing.cast(type[msgspec.Struct], parent))
                continue
            self.refuse(module, base, "a dynamic base", "static_unsupported")
        keywords: dict[str, object] = {}
        for keyword in node.keywords:
            if keyword.arg not in _STRUCT_KEYWORDS:
                self.refuse(
                    module,
                    keyword,
                    f"class keyword {keyword.arg}= changes the wire",
                    "static_unsupported",
                )
            keywords[keyword.arg] = self.fold(module, keyword.value)
        fields: list[tuple[str, object] | tuple[str, object, object]] = []
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            if _is_classvar(self, module, stmt.annotation):
                continue
            annotation = self.translate(module, stmt.annotation)
            fields.append(self._field(module, stmt, annotation))
        try:
            built = msgspec.defstruct(
                node.name,
                typing.cast(list[tuple[str, type, typing.Any]], fields),
                bases=tuple(bases) or None,
                module=module.name,
                **keywords,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            self.refuse(module, node, f"msgspec refuses the struct: {exc}", "static_unsupported")
        self.structs[key] = built
        return built

    def _field(
        self, module: Module, stmt: ast.AnnAssign, annotation: object
    ) -> tuple[str, object] | tuple[str, object, object]:
        assert isinstance(stmt.target, ast.Name)
        name = stmt.target.id
        value = stmt.value
        if value is None:
            return name, annotation
        if isinstance(value, ast.Call):
            callee = self.resolve(module, value.func)
            if isinstance(callee, Vocab) and callee.key == "msgspec.field":
                options = {k.arg: k.value for k in value.keywords if k.arg is not None}
                wire = str(self.fold(module, options["name"])) if "name" in options else None
                if "default" in options:
                    default = self.fold(module, options["default"])
                    return name, annotation, msgspec.field(default=default, name=wire)
                if "default_factory" in options:
                    factory = self.resolve(module, options["default_factory"])
                    if isinstance(factory, Vocab) and factory.key in (
                        "builtins.list",
                        "builtins.dict",
                    ):
                        return (
                            name,
                            annotation,
                            msgspec.field(
                                default_factory=list if factory.key == "builtins.list" else dict,
                                name=wire,
                            ),
                        )
                    return name, annotation, msgspec.field(default_factory=_unfolded, name=wire)
                return name, annotation, msgspec.field(name=wire)
        # Portable callable metadata includes scalar defaults. An unresolved value
        # must refuse here, rather than change identity after worker-side import.
        default = self.fold(module, value)
        return name, annotation, default

    def model(self, module: Module, node: ast.ClassDef) -> ModelSpec:
        key = (module.path, node.name)
        if key in self.models:
            return self.models[key]
        encoded_leaves, fusion = "refuse", "refuse"
        degrees: tuple[int, ...] = ()
        components: dict[str, tuple[str, ...]] = {}
        for base in node.bases:
            target = base.value if isinstance(base, ast.Subscript) else base
            found = self.resolve(module, target)
            if isinstance(found, Definition) and isinstance(found.node, ast.ClassDef):
                parent = self.model(found.module, found.node)
                encoded_leaves, fusion = parent.encoded_leaves, parent.fusion
                degrees = parent.sequence_parallel
                components.update(parent.components)
        for keyword in node.keywords:
            value = self.fold(module, keyword.value)
            if keyword.arg == "encoded_leaves" and value in author._model.ENCODED_LEAVES:
                encoded_leaves = str(value)
            elif keyword.arg == "fusion" and value in author._model.FUSION:
                fusion = str(value)
            else:
                self.refuse(module, keyword, "a Model class keyword outside the closed set")
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            found = self.resolve(module, call.func if call else decorator)
            if isinstance(found, Vocab) and found.key == "author.placeable" and call:
                continue  # a runtime placement capability; the interface does not carry it
            if not (isinstance(found, Vocab) and found.key == "author.sequence_parallel" and call):
                self.refuse(module, decorator, "a Model class decorator outside the closed set")
            for keyword in call.keywords:
                if keyword.arg == "degrees":
                    folded = self.fold(module, keyword.value)
                    degrees = tuple(int(d) for d in typing.cast(Sequence[int], folded))
        for stmt in node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if stmt.name.startswith("_") or stmt.name in _MODEL_MEMBERS:
                continue
            for decorator in stmt.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                found = self.resolve(module, decorator.func)
                if isinstance(found, Vocab) and found.key == "author.uses_components":
                    names = tuple(str(self.fold(module, arg)) for arg in decorator.args)
                    components[stmt.name] = names
        spec = ModelSpec(node.name, encoded_leaves, fusion, degrees, components)
        self.models[key] = spec
        return spec

    def model_annotation(self, module: Module, node: ast.expr) -> ModelSpec | None:
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return None
        found = self.resolve(module, node)
        if (
            isinstance(found, Definition)
            and isinstance(found.node, ast.ClassDef)
            and self.class_kind(found.module, found.node) == "model"
        ):
            return self.model(found.module, found.node)
        return None

    # ---------------------------------------------------------------- errors

    def spell(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project))
        except ValueError:
            return (
                str(path.relative_to(self.runtime_root))
                if path.is_relative_to(self.runtime_root)
                else str(path)
            )

    def refuse(
        self, module: Module, node: ast.AST, why: str, code: Code = "static_unresolvable"
    ) -> typing.NoReturn:
        segment = ast.get_source_segment(module.source, node) or type(node).__name__
        construct = " ".join(segment.split())
        if len(construct) > CONSTRUCT_CHARS:
            construct = construct[:CONSTRUCT_CHARS] + "…"
        line = getattr(node, "lineno", 0)
        raise StaticRefusal(
            f"{self.spell(module.path)}:{line}: `{construct}` — {why}; the static interface "
            "reads a closed vocabulary and never guesses (cr-114)",
            code=code,
        )


def _unfolded() -> None:
    """An unevaluated custom factory; its per-call value is not interface content."""
    return None


def _vocab_object(key: str) -> object:
    prefix, _, name = key.partition(".")
    if prefix == "author":
        return getattr(author, name)
    if key == "msgspec.Meta":
        return msgspec.Meta
    raise AssertionError(key)


def _is_classvar(reader: Reader, module: Module, annotation: ast.expr) -> bool:
    target = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    if not isinstance(target, (ast.Name, ast.Attribute)):
        return False
    try:
        found = reader.resolve(module, target)
    except StaticRefusal:
        return False
    return isinstance(found, Vocab) and found.key == "typing.ClassVar"


def _site_packages(runtime_root: Path) -> tuple[Path, ...]:
    """Where THIS interpreter's distributions live. `sysconfig` reads build configuration;
    it imports nothing and runs nothing. `runtime_root` is included because an editable
    cozy-runtime resolves to its own `src/`, which is not where the packages are."""
    found = [Path(sysconfig.get_paths()[key]) for key in ("purelib", "platlib")]
    return tuple(dict.fromkeys([*found, runtime_root]))


def _installed_packages(sites: Sequence[Path]) -> Installed:
    """Every OTHER cozy package installed beside this runtime, and nothing else.

    A distribution qualifies only by declaring a `cozy.application` entry point. An editable
    install contributes the source tree its `direct_url.json` names; a wheel install
    contributes exactly the `.py` files its RECORD lists. Reading a `dist-info` is reading
    text — no importlib, no `sys.path`, no execution (#713).
    """
    roots: list[Path] = []
    files: set[Path] = set()
    for site in sites:
        _read_site(site, roots, files)
    return Installed(tuple(roots), frozenset(files))


def _read_site(site: Path, roots: list[Path], files: set[Path]) -> None:
    for info in sorted(site.glob("*.dist-info")):
        entry_points = info / "entry_points.txt"
        try:
            if not entry_points.is_file() or _APPLICATION_GROUP not in entry_points.read_text():
                continue
            direct = info / "direct_url.json"
            document = json.loads(direct.read_text()) if direct.is_file() else {}
            url = document.get("url", "") if document.get("dir_info", {}).get("editable") else ""
            if url.startswith("file://"):
                roots.append(Path(url.removeprefix("file://")))
                continue
            record = info / "RECORD"
            if not record.is_file():
                continue
            for line in record.read_text().splitlines():
                relative = line.partition(",")[0]
                if relative.endswith(".py") and not relative.startswith(("/", "..")):
                    files.add(site / relative)
        except (OSError, ValueError):
            continue


def _module_scope(body: Sequence[ast.stmt]) -> list[ast.stmt]:
    """Top-level statements, descending `if`/`try` blocks (`TYPE_CHECKING`, optional imports)."""
    out: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.If):
            out += _module_scope(stmt.body) + _module_scope(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            out += _module_scope(stmt.body) + _module_scope(stmt.orelse)
            for handler in stmt.handlers:
                out += _module_scope(handler.body)
        else:
            out.append(stmt)
    return out


def _bound_names(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, ast.Import):
        return [alias.asname or alias.name.partition(".")[0] for alias in stmt.names]
    if isinstance(stmt, ast.ImportFrom):
        return [alias.asname or alias.name for alias in stmt.names if alias.name != "*"]
    if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return [stmt.name]
    if (name := _single_target(stmt)) is not None:
        return [name]
    return []


def _single_target(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        return stmt.target.id if isinstance(stmt.target, ast.Name) else None
    return None


# ------------------------------------------------------------------- the interface


def read_application(project: Path) -> str:
    """`[application] object` from package.toml, or the zero/ambiguous refusal."""
    path = project / "package.toml"
    if not path.is_file():
        raise StaticRefusal(
            f"{path} does not exist: a package repository has package.toml naming its "
            'one app export ([application] object = "my_package:app")',
            code="no_package_toml",
        )
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise StaticRefusal(f"{path} is not valid TOML: {exc}", code="package_toml") from exc
    application = data.get("application", {})
    obj = application.get("object") if isinstance(application, dict) else None
    if not isinstance(obj, str) or not obj:
        raise StaticRefusal(
            f"{path} names no [application] object — discovery reads exactly ONE configured "
            "export and nothing else, so zero named apps is ambiguous by construction",
            code="zero_application",
        )
    if obj.count(":") != 1 or not all(part.strip() for part in obj.split(":")):
        raise StaticRefusal(
            f'[application] object = "{obj}" is not <module>:<attribute>', code="application_path"
        )
    return obj


@dataclass(frozen=True, slots=True)
class Registration:
    module: Module
    fn: ast.FunctionDef | ast.AsyncFunctionDef
    kind: package_interface.Kind
    name: str
    publishes: bool
    weights: tuple[author.WeightsOutput, ...]
    invocable: bool = False
    memoize: bool = False
    model_defaults: Mapping[str, DefaultLadder] = field(default_factory=dict)
    internal: bool = False
    accelerator: bool | None = None


def build(project: Path, *, environment_python: Path | None = None) -> dict[str, Json]:
    """The package interface from source alone. Executes nothing of the package."""
    project = project.resolve()
    application = read_application(project)
    reader = Reader(
        project, dependency_roots=environment_roots(environment_python, project=project)
    )
    return _build(reader, application, read_table(project / "package.toml"))


def build_builtin(name: str) -> dict[str, Json]:
    """Read only the explicitly selected Runtime-owned App, never caller source."""
    if name != "operations":
        raise StaticRefusal("unknown Runtime builtin surface", code="static_builtin")
    reader = Reader(Path(cozy_runtime.__file__).resolve().parent.parent)
    application = "cozy_runtime.derive.operations:app"
    # Ordinary caller discovery must keep skipping Runtime registrations. Only
    # this reader instance's selected built-in module becomes an App root.
    selected = reader.module("cozy_runtime.derive.operations")
    reader.modules[selected.name] = replace(selected, trusted=False)
    return _build(reader, application, {})


def build_installed(distribution: str, *, environment_python: Path) -> dict[str, Json]:
    """Describe one explicitly selected wheel using only its metadata and source bytes."""

    def normalize(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", distribution):
        raise StaticRefusal("installed distribution name is invalid", code="static_distribution")
    prefix = environment_python.absolute().parent.parent
    roots = environment_roots(environment_python, project=prefix)
    matches: list[Path] = []
    for root in roots:
        if not root.is_relative_to(prefix.resolve()):
            continue
        directories = list(root.glob("*.dist-info"))
        if len(directories) > 512:
            raise StaticRefusal(
                "installed distribution count exceeds 512", code="static_distribution"
            )
        for directory in directories:
            name, _, _version = directory.name.removesuffix(".dist-info").rpartition("-")
            if normalize(name) == normalize(distribution):
                matches.append(directory)
    if len(matches) != 1:
        raise StaticRefusal(
            "the selected environment must contain exactly one matching distribution",
            code="static_distribution",
        )
    metadata = matches[0]

    def read(name: str) -> str:
        path = metadata / name
        if metadata.is_symlink() or path.is_symlink() or not path.is_file():
            raise StaticRefusal(
                "installed wheel metadata is absent or linked", code="static_distribution"
            )
        if path.stat().st_size > 8 << 20:
            raise StaticRefusal(
                "installed wheel metadata exceeds 8 MiB", code="static_distribution"
            )
        return path.read_text()

    headers = email.parser.Parser().parsestr(read("METADATA"), headersonly=True)
    if normalize(headers.get("Name", "")) != normalize(distribution):
        raise StaticRefusal("installed wheel metadata identity differs", code="static_distribution")
    entry_points = configparser.ConfigParser(interpolation=None)
    try:
        entry_points.read_string(read("entry_points.txt"))
        entries = list(entry_points.items("cozy.application"))
    except configparser.Error as exc:
        raise StaticRefusal(
            "installed application metadata is invalid", code="static_distribution"
        ) from exc
    if len(entries) != 1:
        raise StaticRefusal("installed wheel must name exactly one App", code="static_distribution")
    application = entries[0][1].strip()
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", application):
        raise StaticRefusal("installed application export is invalid", code="static_distribution")
    reader = Reader(metadata.parent, dependency_roots=roots)
    entry = reader.application_module(application.partition(":")[0])
    owned = {
        metadata.parent.joinpath(row[0]).resolve()
        for row in csv.reader(read("RECORD").splitlines())
        if len(row) == 3 and row[0].endswith(".py")
    }
    if (
        not entry.path.resolve().is_relative_to(metadata.parent.resolve())
        or entry.path.resolve() not in owned
    ):
        raise StaticRefusal(
            "installed App source is not owned by its wheel", code="static_distribution"
        )
    return _build(reader, application, {})


def _build(
    reader: Reader, application: str, bindings: Mapping[str, Mapping[str, object]]
) -> dict[str, Json]:
    module_name, _, attribute = application.partition(":")
    entry = reader.application_module(module_name)
    app = reader.lookup(entry, attribute, entry.tree)
    if isinstance(app, Assigned) and _is_script_call(reader, app):
        return package_interface.assemble(
            application, [("job", _script_doc(reader, app))], bindings
        )
    if not (isinstance(app, Assigned) and _is_app_call(reader, app)):
        raise StaticRefusal(
            f"{application}: {attribute!r} is not `{attribute} = App()` or "
            f'`{attribute} = script_app("<module>")` at module scope of '
            f"{reader.spell(entry.path)}",
            code="not_an_app",
        )
    docs: dict[str, tuple[package_interface.Kind, dict[str, Json]]] = {}
    for registration in _registrations(reader, reader.reachable(entry), app):
        if registration.name in docs:
            reader.refuse(
                registration.module,
                registration.fn,
                f"duplicate registration {registration.name!r}",
                "static_unsupported",
            )
        docs[registration.name] = registration.kind, _callable(reader, registration)
    return package_interface.assemble(application, list(docs.values()), bindings)


def _is_app_call(reader: Reader, app: Assigned) -> bool:
    value = app.value
    if not isinstance(value, ast.Call) or value.args or value.keywords:
        return False
    found = reader.resolve(app.module, value.func)
    return isinstance(found, Vocab) and found.key == "author.App"


def _registrations(reader: Reader, modules: Sequence[Module], app: Assigned) -> list[Registration]:
    found: list[Registration] = []
    for module in modules:
        for stmt in _module_scope(module.tree.body):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                target = call.func
                keywords = call.keywords
                if isinstance(target, ast.Call):
                    method = _app_method(reader, module, target.func, app)
                    if method is not None:
                        if target.args or call.keywords:
                            reader.refuse(
                                module,
                                call,
                                "computed App registration arguments",
                                "static_unsupported",
                            )
                        keywords = target.keywords
                else:
                    method = _app_method(reader, module, target, app)
                if method is not None:
                    if len(call.args) != 1 or not isinstance(
                        call.args[0], (ast.Name, ast.Attribute)
                    ):
                        reader.refuse(
                            module,
                            call,
                            "App registration must name one function defined in the tree",
                            "static_unsupported",
                        )
                    function = reader.resolve(module, call.args[0])
                    if not (
                        isinstance(function, Definition)
                        and isinstance(function.node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ):
                        reader.refuse(
                            module,
                            call.args[0],
                            "App registration target is not a function defined in the tree",
                        )
                    registration = _registration(
                        reader, function.module, function.node, method, keywords, declaration=module
                    )
                    if registration is not None:
                        found.append(registration)
                continue
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in stmt.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                method = _app_method(reader, module, target, app)
                if method is None:
                    continue
                keywords = decorator.keywords if isinstance(decorator, ast.Call) else []
                if isinstance(decorator, ast.Call) and decorator.args:
                    reader.refuse(
                        module, decorator, "positional decorator arguments", "static_unsupported"
                    )
                registration = _registration(
                    reader, module, stmt, method, keywords, decorator=decorator
                )
                if registration is not None:
                    found.append(registration)
    return found


def _is_invocable(reader: Reader, module: Module, target: ast.expr) -> bool:
    if not isinstance(target, (ast.Name, ast.Attribute)):
        return False
    try:
        found = reader.resolve(module, target)
    except StaticRefusal:
        return False
    return isinstance(found, Vocab) and found.key == "author.invocable"


def _app_method(reader: Reader, module: Module, target: ast.expr, app: Assigned) -> str | None:
    if not isinstance(target, ast.Attribute) or target.attr not in ("entrypoint", "job"):
        return None
    try:
        receiver = reader.resolve(module, target.value)
    except StaticRefusal:
        return None
    if isinstance(receiver, Assigned) and receiver.identity == app.identity:
        return target.attr
    return None


def _registration(
    reader: Reader,
    module: Module,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    method: str,
    keywords: Sequence[ast.keyword],
    *,
    declaration: Module | None = None,
    decorator: ast.expr | None = None,
) -> Registration | None:
    function_module = module
    invocable, memoize = False, False
    default_models: dict[str, DefaultLadder] = {}
    for index, export in enumerate(fn.decorator_list):
        target = export.func if isinstance(export, ast.Call) else export
        if not _is_invocable(reader, function_module, target):
            continue
        if invocable or (decorator is not None and index < fn.decorator_list.index(decorator)):
            reader.refuse(
                function_module,
                export,
                "@invocable must occur once, inside the App decorator",
                "static_unsupported",
            )
        invocable = True
        if isinstance(export, ast.Call):
            if export.args or any(
                keyword.arg not in {"memoize", "defaults", "memo_version", "memo_dependencies"}
                for keyword in export.keywords
            ):
                reader.refuse(
                    function_module,
                    export,
                    "@invocable accepts memoize, memo_version, memo_dependencies and defaults",
                    "static_unsupported",
                )
            for keyword in export.keywords:
                if keyword.arg in {"memo_version", "memo_dependencies"}:
                    # Operation identity is discovered in the installed environment.
                    # Static client inspection cannot author or compare that key.
                    continue
                value = reader.fold(function_module, keyword.value)
                if keyword.arg == "defaults":
                    default_models = model_defaults(value)
                    continue
                if type(value) is not bool:
                    reader.refuse(
                        function_module, keyword.value, "invocable memoize must be a boolean"
                    )
                memoize = value
    if invocable:
        for export in fn.decorator_list:
            target = export.func if isinstance(export, ast.Call) else export
            if export is not decorator and not _is_invocable(reader, function_module, target):
                reader.refuse(
                    function_module,
                    export,
                    "an invocable decorator outside the closed vocabulary",
                    "static_unsupported",
                )
    module = declaration or module
    allowed = _ENTRYPOINT_KEYWORDS if method == "entrypoint" else _JOB_KEYWORDS
    name = fn.name
    publishes = False
    internal = False
    accelerator: bool | None = None
    hidden = False
    weights: tuple[author.WeightsOutput, ...] = ()
    for keyword in keywords:
        if keyword.arg not in allowed:
            reader.refuse(
                module,
                keyword,
                f"@app.{method}({keyword.arg}=) is outside the closed keyword set",
                "static_unsupported",
            )
        if keyword.arg == "preflight":
            hook = reader.resolve(module, keyword.value)
            if not (
                isinstance(hook, Definition)
                and isinstance(hook.node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                reader.refuse(
                    module, keyword.value, "preflight= must name a function defined in the tree"
                )
            continue
        if keyword.arg == "demand":
            reader.fold(module, keyword.value)
            continue
        if keyword.arg in {"memo_version", "memo_dependencies"}:
            # Like @invocable options, installed-environment memo identities
            # are not part of the source-stable client contract.
            continue
        value = reader.fold(module, keyword.value)
        if keyword.arg == "defaults":
            declared_defaults = model_defaults(value)
            if default_models and declared_defaults:
                reader.refuse(
                    module, keyword, "declare defaults once, on invocable or App registration"
                )
            default_models = declared_defaults or default_models
        elif keyword.arg == "name":
            name = str(value)
        elif keyword.arg == "hidden" and value is True:
            hidden = True
        elif keyword.arg == "internal":
            if type(value) is not bool:
                reader.refuse(module, keyword.value, "internal must be a boolean")
            internal = value
        elif keyword.arg == "accelerator":
            if type(value) is not bool:
                reader.refuse(module, keyword.value, "accelerator must be a boolean")
            accelerator = value
        elif keyword.arg == "publishes":
            publishes = bool(value)
        elif keyword.arg == "memoize":
            if type(value) is not bool:
                reader.refuse(module, keyword.value, "entrypoint memoize must be a boolean")
            memoize = value
        elif keyword.arg == "weights":
            assert isinstance(value, tuple)
            weights = tuple(value)
    if hidden and internal:
        reader.refuse(module, fn, "an internal callable cannot also be hidden")
    if hidden:
        return None
    return Registration(
        function_module,
        fn,
        typing.cast(package_interface.Kind, method),
        name,
        publishes,
        weights,
        invocable,
        memoize,
        default_models,
        internal,
        accelerator,
    )


def _parameter_annotation(
    reader: Reader,
    module: Module,
    node: ast.expr,
) -> tuple[Module, ast.expr, Resolved | None]:
    """Follow declared aliases before classifying the parameter's role.

    The full original annotation is still translated for Assets, preserving every
    collection/element bound. Resolution never imports or evaluates the alias.
    """
    seen: set[tuple[Path, int]] = set()
    while True:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node = ast.parse(node.value, mode="eval").body
        target = node.value if isinstance(node, ast.Subscript) else node
        if isinstance(target, ast.Subscript):
            target = target.value
        found = (
            reader.resolve(module, target)
            if isinstance(target, (ast.Name, ast.Attribute))
            else None
        )
        if not isinstance(found, Assigned):
            return module, node, found
        if found.identity in seen:
            reader.refuse(module, node, "a cyclic parameter type alias", "static_unsupported")
        seen.add(found.identity)
        module, node = found.module, found.value


def _callable(reader: Reader, registration: Registration) -> dict[str, Json]:
    if registration.invocable:
        doc = static_invocable.build(reader, registration)
        if registration.internal:
            doc["internal"] = True
        return doc
    module, fn = registration.module, registration.fn
    if fn.args.vararg or fn.args.kwarg:
        reader.refuse(module, fn, "*args/**kwargs on a handler", "static_unsupported")
    payloads: list[object] = []
    models: list[dict[str, Json]] = []
    contexts: list[str] = []
    capabilities: list[str] = []
    assets: AssetsBinding | None = None
    for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        if arg.annotation is None:
            reader.refuse(module, arg, "an unannotated handler parameter")
        annotation_module, node, found = _parameter_annotation(reader, module, arg.annotation)
        if isinstance(found, Vocab):
            vocab_name = found.key.partition(".")[2]
            if found.key in ("author.Context", "author.Preflight") or vocab_name in (
                "Settings",
                "Secrets",
            ):
                if isinstance(node, ast.Subscript):
                    reader.translate(annotation_module, node.slice)
                parameter = _classify(
                    arg.arg, reader.translate(module, arg.annotation), f"{module.name}:{fn.name}"
                )
                if parameter.role == "context":
                    contexts.append(arg.arg)
                if parameter.capability:
                    capabilities.append(parameter.capability)
                continue
            if vocab_name in {cls.__name__ for cls in SERVICES.values()}:
                parameter = _classify(
                    arg.arg, reader.translate(module, arg.annotation), f"{module.name}:{fn.name}"
                )
                if parameter.capability:
                    capabilities.append(parameter.capability)
                continue
            if vocab_name in ("Assets", "Annotated"):
                if assets is not None:
                    reader.refuse(module, arg, "a second Assets parameter", "static_unsupported")
                assets = assets_binding(arg.arg, reader.translate(module, arg.annotation))
                if assets.decoded:
                    capabilities.append("media_decode")
                continue

        if isinstance(found, Definition) and isinstance(found.node, ast.ClassDef):
            kind = reader.class_kind(found.module, found.node)
            if kind == "model":
                spec = reader.model(found.module, found.node)
                models.append(
                    package_interface.model_slot(
                        f"{registration.name}.models.{arg.arg}",
                        spec.name,
                        spec.encoded_leaves,
                        spec.fusion,
                        spec.components,
                        spec.sequence_parallel,
                        defaults=registration.model_defaults.get(arg.arg, ()),
                    )
                )
                continue
            if kind == "struct":
                payloads.append(reader.translate(module, arg.annotation))
                continue
        reader.refuse(module, arg.annotation, f"parameter {arg.arg}: not an author surface type")
    check_model_arguments(
        registration.model_defaults,
        [str(slot["path"]).rsplit(".", 1)[-1] for slot in models],
    )
    if len(payloads) != 1:
        reader.refuse(
            module, fn, f"exactly one typed request struct is required, found {len(payloads)}"
        )
    if fn.returns is None:
        reader.refuse(module, fn, "no return annotation")
    result = reader.translate(module, fn.returns)
    if not is_struct(result):
        reader.refuse(
            module, fn.returns, "the return annotation must be a msgspec.Struct result schema"
        )
    payload = payloads[0]
    if assets is not None:
        payload = assets_payload(
            typing.cast(type[msgspec.Struct], payload), assets, fn.name, module.name
        )
    doc = package_interface.callable_doc(
        name=registration.name,
        kind=registration.kind,
        payload_type=payload,
        result_type=result,
        models=models,
        assets=assets,
        publishes=registration.publishes,
        weights_outputs=registration.weights,
        internal=registration.internal,
        accelerator=registration.accelerator,
    )
    if registration.kind == "entrypoint":
        doc["invocable"] = invocable_interface.serving_metadata(
            payload,
            result,
            module=module.name,
            export=fn.name,
            context=contexts[0] if contexts else "",
            capabilities=sorted(capabilities),
            memoize=registration.memoize,
        )
    return doc


def environment_roots(python: Path | None, *, project: Path) -> tuple[Path, ...]:
    """Read the caller's explicit venv layout; never launch its Python or execute .pth."""
    if python is None:
        return ()
    prefix = python.absolute().parent.parent  # Do not resolve the Python symlink to the base.
    if not python.is_file() or not (prefix / "pyvenv.cfg").is_file():
        raise StaticRefusal(
            "--environment-python must name a venv Python", code="static_environment"
        )
    roots: list[Path] = []
    for directory in sorted((prefix / "lib").glob("python*/site-packages")):
        roots.append(directory.resolve())
        for path in sorted(directory.glob("*.pth")):
            if path.stat().st_size > 1 << 20:
                raise StaticRefusal("venv .pth exceeds source bound", code="static_environment")
            for line in path.read_text().splitlines():
                if not line or line.startswith(("#", "import ", "import\t")):
                    continue  # Executable finder/setup lines are never interpreted.
                target = (directory / line.rstrip()).resolve()
                if target.is_dir():
                    if not (
                        target.is_relative_to(prefix.parent.resolve())
                        or target.is_relative_to(project.resolve())
                    ):
                        raise StaticRefusal(
                            "venv editable source escapes the retained install/project root",
                            code="static_environment",
                        )
                    roots.append(target)
    if not roots or len(roots) > 128:
        raise StaticRefusal(
            "venv source roots are absent or exceed the bound", code="static_environment"
        )
    return tuple(dict.fromkeys(roots))


def _is_script_call(reader: Reader, app: Assigned) -> bool:
    if not isinstance(app.value, ast.Call):
        return False
    found = reader.resolve(app.module, app.value.func)
    return isinstance(found, Vocab) and found.key == "author.script_app"


def _script_annotation(reader: Reader, module: Module, node: ast.expr) -> ast.expr:
    """Use the same imported-type spelling as the execution adapter, without importing."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node = ast.parse(node.value, mode="eval").body
    imported = {
        alias.asname or (alias.name.split(".")[0] if isinstance(stmt, ast.Import) else alias.name)
        for stmt in module.tree.body
        if isinstance(stmt, (ast.Import, ast.ImportFrom))
        for alias in stmt.names
    }
    for part in ast.walk(node):
        if isinstance(part, ast.Name) and part.id not in {*_BUILTIN_TYPES, *imported}:
            reader.refuse(module, part, "script types must be explicitly imported or builtin")
        if isinstance(part, ast.Call):
            metadata = reader.resolve(module, part.func)
            if not isinstance(metadata, Vocab) or metadata.key != "author.AssetBound":
                reader.refuse(module, part, "a call in a script annotation", "static_computed")
    return node


def _script_doc(reader: Reader, app: Assigned) -> dict[str, Json]:
    call = app.value
    assert isinstance(call, ast.Call)
    if (
        len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
        or not call.args[0].value.isidentifier()
    ):
        reader.refuse(app.module, call, "script_app requires one literal module name")
    module = reader.module(call.args[0].value, where=(app.module, call))
    if len(module.source.encode()) > 32 << 20:
        reader.refuse(module, module.tree, "script exceeds its source bound")
    try:
        fn = main_definition(module.tree)
    except author.ConformanceError as exc:
        reader.refuse(module, module.tree.body[0] if module.tree.body else call, str(exc))
    try:
        defaults, weights, accelerator = declarations(module.source.encode())
    except (ValueError, TypeError) as exc:
        reader.refuse(module, fn, f"invalid script metadata: {exc}")
    models: list[dict[str, Json]] = []
    payload_fields: list[tuple[str, type]] = []
    model_names: set[str] = set()
    services: set[str] = set()
    native_context = False
    for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        if arg.arg == "ctx":
            if arg.annotation is not None:
                found_context = reader.resolve(
                    module, _script_annotation(reader, module, arg.annotation)
                )
                native_context = (
                    isinstance(found_context, Vocab) and found_context.key == "author.Context"
                )
            continue
        if arg.annotation is None:
            reader.refuse(module, arg, "script injected parameters require a type")
        node = _script_annotation(reader, module, arg.annotation)
        target = node.value if isinstance(node, ast.Subscript) else node
        found = reader.resolve(module, target)
        spec: ModelSpec | None = None
        if isinstance(found, Vocab):
            if found.key == "author.Model":
                if isinstance(node, ast.Subscript):
                    reader.translate(module, node.slice)
                spec = ModelSpec("_ScriptSource", "refuse", "refuse", (), {})
            else:
                service = next(
                    (key for key, cls in SERVICES.items() if found.key == "author." + cls.__name__),
                    None,
                )
                if service is not None:
                    if service in services:
                        reader.refuse(module, arg, "script requests a service twice")
                    if service in ("settings", "secrets"):
                        if not isinstance(node, ast.Subscript):
                            reader.refuse(module, arg, "script typed service needs its schema")
                        reader.translate(module, node.slice)
                    services.add(service)
                    continue
        if (
            isinstance(found, Definition)
            and isinstance(found.node, ast.ClassDef)
            and reader.class_kind(found.module, found.node) == "model"
        ):
            spec = reader.model(found.module, found.node)
        if spec is None:
            translated = reader.translate(module, node)
            if byte_parameter(translated):
                payload_fields.append((arg.arg, typing.cast(type, translated)))
                continue
            reader.refuse(
                module,
                arg,
                "script parameters must be declared models or services, or typed assets",
            )
        model_names.add(arg.arg)
        models.append(
            package_interface.model_slot(
                f"main.models.{arg.arg}",
                spec.name,
                spec.encoded_leaves,
                spec.fusion,
                spec.components,
                spec.sequence_parallel,
            )
        )
    if set(defaults) - model_names:
        reader.refuse(module, fn, "script model default names an undeclared model")
    if "weights_read" in services and not models:
        reader.refuse(module, fn, "script WeightsReader requires a model input")
    if weights and not native_context:
        reader.refuse(module, fn, "script model outputs require ctx: Context")
    if weights and "save" in services:
        reader.refuse(module, fn, "script weights cannot mix with media outputs")
    returned = (
        reader.translate(module, _script_annotation(reader, module, fn.returns))
        if fn.returns
        else type(None)
    )
    result = (
        msgspec.defstruct("_ScriptResult", [("value", typing.cast(type, returned))])
        if returned is not type(None)
        else msgspec.defstruct("_ScriptCompleted", [])
    )
    return package_interface.callable_doc(
        name="main",
        kind="job",
        payload_type=msgspec.defstruct("_ScriptRequest", payload_fields),
        result_type=result,
        models=models,
        assets=None,
        publishes=False,
        weights_outputs=weights,
        accelerator=accelerator,
    )
