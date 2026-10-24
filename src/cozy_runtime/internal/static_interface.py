"""The package interface from SOURCE: `ast` over the package's files, executing no package code.

INVARIANT (decision #713 — package code is UNTRUSTED): this module never imports, execs,
compiles or evaluates a package module. No `importlib`, no `sys.path`, no `exec`/`eval`/
`compile`, no annotation evaluation, no module top level. It reads text, parses it with
`ast`, follows the package's own module-scope import graph (plus cozy-runtime's own source
for `cozy_runtime.derive.*`), folds constants with a shared
evaluator over literal and arithmetic nodes and values of the package's committed JSON assets
(`author.data_values`), and translates every interface-bearing declaration into real types built
from the CLOSED author vocabulary — `msgspec.defstruct`, `enum.Enum`, `typing.Literal`/
`Annotated`, cozy-runtime's own asset, marker and service classes. Rendering is then
`schema.render`, the SAME renderer the import path uses, so the two derivations cannot drift
by construction.

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
import hashlib
import json
import operator
import os
import re
import sysconfig
import tomllib
import types
import typing
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import msgspec

import cozy_runtime
import cozy_runtime.internal.locked_sources as locked
import cozy_runtime.internal.memo_identity as memo_identity
import cozy_runtime.internal.package_interface as package_interface
import cozy_runtime.internal.static_invocable as static_invocable
from cozy_runtime import _interpreter, author
from cozy_runtime.author._markers import MODEL_DEFAULT, PREFLIGHT
from cozy_runtime.author._memo import memo_policy, one_memo
from cozy_runtime.author._model import (
    ENCODED_LEAVES,
    MIN_BITS,
    PHASES,
    Model,
    attention_bits,
    closed,
    component_method,
    component_set,
    coverage_patterns,
)
from cozy_runtime.author._model_defaults import model_components, model_defaults, model_on_demand
from cozy_runtime.author._recipes import CALIBRATED_ENTRYPOINT, canonical, fingerprint, outputs
from cozy_runtime.author._recipes import ENTRYPOINT as RECIPES_ENTRYPOINT
from cozy_runtime.author._services import SERVICES
from cozy_runtime.author._signature import (
    AssetsBinding,
    Declaration,
    ModelSpec,
    Slot,
    _classify,
    assets_binding,
    assets_payload,
)
from cozy_runtime.author._walker import is_struct
from cozy_runtime.internal.canonical import Json

CONSTRUCT_CHARS = 96


class StaticRefusal(author.ConformanceError):
    """A construct the static reader cannot admit, named by file:line and source text."""

    default_code = "static_unresolvable"


# ------------------------------------------------------------------------- the modules


@dataclass(frozen=True, slots=True)
class Module:
    name: str
    path: Path
    source: str
    tree: ast.Module
    is_package: bool
    trusted: bool
    """cozy-runtime's own derive source: followed for types and constants, never scanned
    for registrations."""
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
#: Runtime modules a package builds its surface from: followed for types, models and constants.
_TRUSTED_ROOTS = ("cozy_runtime.derive", "cozy_runtime.vlm")


_ARITHMETIC: dict[type[ast.operator], Callable[[Any, Any], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_INTEGER: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}


class Unfolded(ValueError):
    """The node the constant folder stops at, and why."""

    def __init__(self, node: ast.AST, why: str) -> None:
        super().__init__(why)
        self.node = node


def fold_constant(node: ast.expr, other: Callable[[ast.expr], Any]) -> Any:
    """A literal/arithmetic expression's value, executing nothing: `other` values names and
    calls."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        items: list[Any] = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                spread = fold_constant(item.value, other)
                if not isinstance(spread, (tuple, list)):
                    raise Unfolded(item, "a splat of a non-sequence")
                items.extend(spread)
            else:
                items.append(fold_constant(item, other))
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                if value.conversion != -1 or value.format_spec is not None:
                    raise Unfolded(value, "a formatted f-string field")
                value = value.value
            part = fold_constant(value, other)
            if not isinstance(part, str):
                raise Unfolded(value, "an f-string field that is not a string")
            parts.append(part)
        return "".join(parts)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise Unfolded(node, "a dict splat")
        return {
            fold_constant(key, other): fold_constant(value, other)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
        operand = fold_constant(node.operand, other)
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise Unfolded(node, "unary arithmetic on a non-number")
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand if isinstance(node.op, ast.UAdd) else ~int(operand)
    if isinstance(node, ast.BinOp):
        left, right = fold_constant(node.left, other), fold_constant(node.right, other)
        if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            return left + right
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (left, right)):
            raise Unfolded(node, "arithmetic on non-numbers")
        if (arithmetic := _ARITHMETIC.get(type(node.op))) is not None:
            return arithmetic(left, right)
        if isinstance(left, int) and isinstance(right, int) and type(node.op) in _INTEGER:
            return _INTEGER[type(node.op)](left, right)
        raise Unfolded(node, "an operator the constant folder does not evaluate")
    return other(node)


def _author_structs() -> tuple[str, ...]:
    return tuple(f"author.{name}" for name in _AUTHOR_STRUCTS)


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
    "Context",
    "Assets",
    "Settings",
    "Secrets",
    "QuantizationPlan",
    "SelectionPlan",
    "EncodingRule",
    "ImageRequest",
    "ImageResult",
    "ReferenceImages",
    "Box",
    "Region",
    "AspectRatio",
    "Seed",
    *(cls.__name__ for cls in SERVICES.values()),
)
#: Author-surface structs a package's struct may extend (cr-219): the image interface.
_AUTHOR_STRUCTS = ("ImageRequest", "ImageResult")
#: Author-surface callables a folded constant may reach: each validates its own arguments.
#: `data_values` reads committed JSON through the same function as the imported app.
_AUTHOR_VALUES = (
    "AssetBound",
    "AssetLimits",
    "ImagePreparation",
    "VideoPreparation",
    "AudioPreparation",
    "WeightsOutput",
    "Bound",
    "Shape",
    "Memoize",
    "data_values",
    "QuantizationPlan",
    "SelectionPlan",
    "EncodingRule",
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
        "components",
        "on_demand",
        "internal",
        "publishes",
        "emits_media",
        "weights",
        "accelerator",
    }
)
_INVOCABLE_KEYWORDS = frozenset(
    {"memoize", "defaults", "components", "memo_version", "memo_dependencies"}
)


# -------------------------------------------------------------------------- the reader


#: Bare `Weights` (or `Weights[Model]`): raw checkpoint data of no declared layout.
_ANY_LAYOUT = ModelSpec(Model.__name__)


class Reader:
    """One package tree, read lazily. Every method resolves by NAME, never by execution."""

    def __init__(
        self,
        project: Path,
        *,
        dependency_roots: Sequence[Path] = (),
        wheels: Mapping[str, locked.LockedWheel] | None = None,
        own: frozenset[str] | None = None,
    ) -> None:
        self.project = project
        self.project_roots = (project / "src", project)
        #: top-level import names that are this package's own; None: whatever its roots hold
        self.own = own
        self.roots = (*self.project_roots, *dependency_roots)
        #: published callees, read from the wheels the project's uv.lock pins when needed
        self.wheels = locked.Wheels(wheels or {})
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
        if not trusted and self._fetched(parts[0]):
            return self.module(name, where=where)
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

    def _fetched(self, top: str) -> bool:
        """A published callee package, unpacked from the exact wheel the project's uv.lock
        pins for `top` and followed as a dependency root. False: nothing pinned for it."""
        try:
            root = self.wheels.root(top)
        except locked.LockedSourceRefusal as exc:
            raise StaticRefusal(str(exc), code="static_module") from exc
        if root is not None:
            self.roots = (*self.roots, root)
        return root is not None

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
        if name in _VOCAB_MODULES or not self._own(top):
            return None
        try:
            return self.module(name)
        except StaticRefusal:
            return None

    def _own(self, top: str) -> bool:
        if self.own is not None:
            return top in self.own
        return any(
            (root / top).is_dir() or (root / f"{top}.py").is_file() for root in self.project_roots
        )

    def _tree_tops(self, name: str) -> set[str]:
        top = name.split(".")[0]
        for root in self.roots:
            if (root / top).is_dir() or (root / f"{top}.py").is_file():
                return {top}
        return {top} if self._installed((top,)) is not None or self._fetched(top) else set()

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
        try:
            return fold_constant(node, lambda inner: self._named(module, inner, scope))
        except Unfolded as exc:
            self.refuse(module, exc.node, str(exc), "static_computed")

    def _named(self, module: Module, node: ast.expr, scope: Mapping[str, object]) -> object:
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            return self._comprehension(module, node, scope)
        if isinstance(node, ast.Name) and node.id in scope:
            return scope[node.id]
        if isinstance(node, ast.Name) and node.id == "__file__":
            # The one dunder a data asset is addressed from: this module's own file.
            return str(module.path)
        if isinstance(node, ast.Attribute):
            base = self._attribute_base(module, node.value)
            if (
                isinstance(base, Definition)
                and isinstance(base.node, ast.ClassDef)
                and self.class_kind(base.module, base.node) == "enum"
            ):
                built = typing.cast(type[enum.Enum], self.class_type(base.module, base.node, node))
                if node.attr in built.__members__:
                    return built.__members__[node.attr]
                self.refuse(module, node, "unknown enumeration member")
            if base is None or isinstance(base, Assigned):
                # A field of a declared value: `INT8.include`, `INT8.plan.components`.
                value = self.fold(module, node.value, scope)
                if isinstance(value, (msgspec.Struct, StaticRecipe)) and node.attr in (
                    getattr(value, "__struct_fields__", None) or StaticRecipe.__slots__
                ):
                    return getattr(value, node.attr)
                self.refuse(module, node, "not a field of a declared value", "static_computed")
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
            if isinstance(callee, Vocab) and callee.key == "author.Recipe":
                return self.recipe(module, node, scope)
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

    def _attribute_base(self, module: Module, node: ast.expr) -> Resolved | None:
        """What an attribute's base names, or None for a value that is no name."""
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return None
        try:
            return self.resolve(module, node)
        except StaticRefusal:
            if isinstance(node, ast.Attribute):
                return None
            raise

    def _comprehension(
        self, module: Module, node: ast.GeneratorExp | ast.ListComp, scope: Mapping[str, object]
    ) -> object:
        """A comprehension over folded sequences: `tuple(F(x, y) for x in A for y in B)`.

        Deterministic by construction: the element is folded exactly as any other constant
        is, with the clauses' own targets bound, and nothing of the package runs.
        """

        def expand(index: int, bound: Mapping[str, object]) -> list[object]:
            if index == len(node.generators):
                return [self.fold(module, node.elt, bound)]
            clause = node.generators[index]
            if clause.ifs or clause.is_async or not isinstance(clause.target, ast.Name):
                self.refuse(
                    module, node, "a filtered, async or unpacking comprehension", "static_computed"
                )
            items = self.fold(module, clause.iter, bound)
            if not isinstance(items, (tuple, list, dict)):
                self.refuse(
                    module, clause.iter, "a comprehension over a non-sequence", "static_computed"
                )
            name = clause.target.id
            return [v for item in items for v in expand(index + 1, {**bound, name: item})]

        folded = expand(0, scope)
        return folded if isinstance(node, ast.ListComp) else tuple(folded)

    def recipe(self, module: Module, node: ast.Call, scope: Mapping[str, object]) -> StaticRecipe:
        """`Recipe(lane=, model=, plan=)` read from source: its model a Model class (`Model`:
        any layout), its plan a folded plan or a `DerivationHook` naming a module function."""
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if node.args or not {"lane", "model", "plan"} <= set(keywords) <= {
            "lane", "model", "plan", "calibrate",
        }:  # fmt: skip
            self.refuse(
                module, node, "Recipe(lane=, model=, plan=[, calibrate=]) by keyword",
                "static_unsupported",
            )  # fmt: skip
        lane = self.fold(module, keywords["lane"], scope)
        try:
            author.WeightsOutput(typing.cast(str, lane), max_new_bytes=0)
        except author.ConformanceError as exc:
            self.refuse(module, keywords["lane"], f"recipe lane: {exc}", "static_unsupported")
        found = self.resolve(module, keywords["model"])
        spec = (
            _ANY_LAYOUT
            if isinstance(found, Vocab) and found.key == "author.Model"
            else self._model_class(found)
        )
        if spec is None:
            self.refuse(module, keywords["model"], "a recipe's model is a Model class")
        value = keywords["plan"]
        hook = self.resolve(module, value.func) if isinstance(value, ast.Call) else None
        if isinstance(hook, Vocab) and hook.key == "author.DerivationHook":
            function = (
                self.resolve(module, value.args[0])
                if isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords
                else None
            )
            if not (
                isinstance(function, Definition) and not isinstance(function.node, ast.ClassDef)
            ):
                self.refuse(module, value, "DerivationHook(<a module-level function>)")
            identity = f"{function.module.name}:{function.node.name}"
            if "calibrate" in keywords:
                self.refuse(module, node, "a recipe's calibrate is beside a QuantizationPlan")
            return StaticRecipe(typing.cast(str, lane), spec, identity)
        plan = self.fold(module, value, scope)
        if not isinstance(plan, (author.QuantizationPlan, author.SelectionPlan)):
            self.refuse(module, value, "a recipe's plan is a QuantizationPlan or SelectionPlan")
        calibrate = ""
        if "calibrate" in keywords:
            function = self.resolve(module, keywords["calibrate"])
            if not (
                isinstance(plan, author.QuantizationPlan)
                and isinstance(function, Definition)
                and not isinstance(function.node, ast.ClassDef)
            ):
                self.refuse(
                    module, keywords["calibrate"],
                    "calibrate=<a module-level function>, beside a QuantizationPlan",
                )  # fmt: skip
            calibrate = f"{function.module.name}:{function.node.name}"
        return StaticRecipe(typing.cast(str, lane), spec, plan, calibrate)

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
                if found.key == "msgspec.Struct" or found.key in _author_structs():
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
            if isinstance(found, Vocab) and found.key in _author_structs():
                bases.append(getattr(author, found.key.partition(".")[2]))
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
        spec = _ANY_LAYOUT
        components: dict[str, tuple[str, ...]] = {}
        for base in node.bases:
            target = base.value if isinstance(base, ast.Subscript) else base
            found = self.resolve(module, target)
            if isinstance(found, Definition) and isinstance(found.node, ast.ClassDef):
                spec = self.model(found.module, found.node)
                components.update(spec.components)
        encoded_leaves, min_bits, phase = spec.encoded_leaves, spec.min_bits, spec.phase
        degrees, placeable = spec.sequence_parallel, spec.placeable
        for keyword in node.keywords:
            value = self.fold(module, keyword.value)
            if keyword.arg == "encoded_leaves" and closed(ENCODED_LEAVES, value):
                encoded_leaves = typing.cast(str, value)
            elif keyword.arg == "min_bits" and closed(MIN_BITS, value):
                min_bits = typing.cast(int, value)
            elif keyword.arg == "phase" and closed(PHASES, value):
                phase = typing.cast(str, value)
            elif keyword.arg == "attention_bits":
                try:  # Runtime reads it at selection; the interface does not carry it
                    attention_bits(node.name, value)
                except author.ConformanceError as exc:
                    self.refuse(module, keyword, str(exc))
            elif keyword.arg in ("unloaded", "derived"):
                try:  # Runtime reads them at load; the interface does not carry them
                    coverage_patterns(node.name, keyword.arg, value)
                except author.ConformanceError as exc:
                    self.refuse(module, keyword, str(exc))
            else:
                self.refuse(module, keyword, "a Model class keyword outside the closed set")
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            found = self.resolve(module, call.func if call else decorator)
            if call and isinstance(found, Vocab) and found.key == "author.placeable":
                self.declared(module, call, author.placeable)
                placeable = tuple(dict.fromkeys(str(self.fold(module, a)) for a in call.args))
            elif call and isinstance(found, Vocab) and found.key == "author.sequence_parallel":
                self.declared(module, call, author.sequence_parallel)
                folded = self.fold(module, call.keywords[0].value)
                degrees = tuple(typing.cast(Sequence[int], folded))
            else:
                self.refuse(module, decorator, "a Model class decorator outside the closed set")
        loads = spec.loads
        arguments = spec.arguments
        constructors: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        for stmt in node.body:
            loads = loads or "load" in _bound_names(stmt)
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if stmt.name in ("__new__", "__init__"):
                constructors[stmt.name] = stmt
            for decorator in stmt.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                found = self.resolve(module, decorator.func)
                if not (isinstance(found, Vocab) and found.key == "author.uses_components"):
                    continue
                names = self._components(module, node, stmt, decorator)
                if not (stmt.name.startswith("_") or stmt.name in _MODEL_MEMBERS):
                    components[stmt.name] = names
        if constructors:
            arguments = _required(constructors.get("__new__") or constructors["__init__"])
        spec = ModelSpec(
            node.name, encoded_leaves, min_bits, phase, degrees, components, placeable, loads,
            arguments,
        )  # fmt: skip
        self.models[key] = spec
        return spec

    def _components(
        self,
        module: Module,
        cls: ast.ClassDef,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        decorator: ast.Call,
    ) -> tuple[str, ...]:
        """One `@uses_components` declaration, by the rules its decorator applies at import."""
        keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
        helpers = keywords.get("memo_dependencies")
        memo = "memo_version" in keywords or not (
            helpers is None or (isinstance(helpers, (ast.Tuple, ast.List)) and not helpers.elts)
        )
        signature = method.args
        parameters = (
            *signature.posonlyargs, *signature.args, signature.vararg, *signature.kwonlyargs,
            signature.kwarg,
        )  # fmt: skip
        try:
            names = component_set(
                [self.fold(module, arg) for arg in decorator.args],
                memoize="memoize" in keywords and bool(self.fold(module, keywords["memoize"])),
                memo=memo,
            )
            component_method(f"{cls.name}.{method.name}", [p.arg for p in parameters if p])
        except author.ConformanceError as exc:
            self.refuse(module, decorator, exc.message, exc.code)
        return names

    def declared(self, module: Module, call: ast.Call, factory: Callable[..., object]) -> None:
        """Refuse a declaration its author factory refuses: the import-time rule, applied to
        the folded arguments."""
        try:
            factory(
                *(self.fold(module, arg) for arg in call.args),
                **{str(k.arg): self.fold(module, k.value) for k in call.keywords},
            )
        except author.ConformanceError as exc:
            self.refuse(module, call, exc.message, exc.code)
        except TypeError as exc:
            self.refuse(module, call, str(exc), "static_unsupported")

    def model_annotation(self, module: Module, node: ast.expr) -> tuple[ModelSpec, bool] | None:
        """A model slot's class and whether it is `Weights[M]` (raw), or None."""
        target = node.value if isinstance(node, ast.Subscript) else node
        if not isinstance(target, (ast.Name, ast.Attribute)):
            return None
        found = self.resolve(module, target)
        if isinstance(found, Vocab) and found.key == "author.Weights":
            if not isinstance(node, ast.Subscript):
                return _ANY_LAYOUT, True
            layout = (
                self.resolve(module, node.slice)
                if isinstance(node.slice, (ast.Name, ast.Attribute))
                else None
            )
            if isinstance(layout, Vocab) and layout.key in ("author.Model", "typing.Any"):
                return _ANY_LAYOUT, True
            spec = self._model_class(layout)
            if spec is None:
                self.refuse(module, node, "Weights[M] needs a Model class M")
            return spec, True
        if isinstance(node, ast.Subscript):
            return None
        spec = self._model_class(found)
        return None if spec is None else (spec, False)

    def _model_class(self, found: Resolved | None) -> ModelSpec | None:
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
        self, module: Module, node: ast.AST, why: str, code: str = "static_unresolvable"
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


def _required(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """The arguments a call to a constructor must pass, beyond the class or instance."""
    positional = [*fn.args.posonlyargs, *fn.args.args][1:]
    required = positional[: len(positional) - len(fn.args.defaults)]
    keywords = zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
    return tuple(arg.arg for arg in (*required, *(a for a, d in keywords if d is None)))


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
    """pyproject's one `cozy.application` entry point (the one the wheel carries), or the
    zero/ambiguous refusal."""
    path = project / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise StaticRefusal(
            f"{path} does not exist: a package's pyproject.toml names its one App export "
            '([project.entry-points."cozy.application"] default = "my_package:app")',
            code="no_pyproject",
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise StaticRefusal(f"{path} is not valid TOML: {exc}", code="pyproject") from exc
    table = data.get("project")
    points = table.get("entry-points") if isinstance(table, dict) else None
    group = points.get("cozy.application") if isinstance(points, dict) else None
    objects = list(group.values()) if isinstance(group, dict) else []
    if len(objects) != 1 or not isinstance(objects[0], str) or not objects[0]:
        raise StaticRefusal(
            f'{path} names {len(objects)} [project.entry-points."cozy.application"] entries; '
            "discovery reads exactly one",
            code="application_entrypoint_count",
        )
    obj = objects[0]
    if obj.count(":") != 1 or not all(part.strip() for part in obj.split(":")):
        raise StaticRefusal(
            f'cozy.application entry point "{obj}" is not <module>:<attribute>',
            code="application_path",
        )
    return obj


@dataclass(frozen=True, slots=True)
class Registration:
    module: Module
    fn: ast.FunctionDef | ast.AsyncFunctionDef
    name: str
    publishes: bool
    weights: tuple[author.WeightsOutput, ...]
    invocable: bool = False
    memoize: author.Memoize | None = None
    model_defaults: Mapping[str, str] = field(default_factory=dict)
    internal: bool = False
    accelerator: bool = True
    #: `memo_version=` / `memo_dependencies=` as written, and the module they are read in.
    memo_declaration: tuple[tuple[str, Module, ast.expr], ...] = ()
    model_components: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    model_on_demand: tuple[str, ...] = ()
    #: A model argument's class where the registration names it, not its annotation.
    model_classes: Mapping[str, ModelSpec] = field(default_factory=dict)
    emits_media: bool = False

    def declaration(self, **found: Any) -> Declaration:
        """Its record, with what its signature declares (`found`)."""
        return Declaration(
            name=self.name,
            module=self.module.name,
            export=self.fn.name,
            publishes=self.publishes,
            emits_media=self.emits_media,
            weights_outputs=self.weights,
            memoize=self.memoize,
            internal=self.internal,
            accelerator=self.accelerator,
            model_defaults=self.model_defaults,
            model_components=self.model_components,
            model_on_demand=self.model_on_demand,
            **found,
        )


@dataclass(frozen=True, slots=True)
class StaticRecipe:
    """A `Recipe` read from source; a `DerivationHook` plan, and a `calibrate`, is its function's
    `module:name`."""

    lane: str
    model: ModelSpec
    plan: author.QuantizationPlan | author.SelectionPlan | str
    calibrate: str = ""

    def canonical(self) -> dict[str, object]:
        if isinstance(self.plan, str):
            return {"kind": "derivation", "hook": self.plan}
        return canonical(self.plan)


def build(project: Path, *, environment_python: Path | None = None) -> dict[str, Json]:
    """The package interface from source alone. Executes nothing of the package."""
    project = project.resolve()
    application = read_application(project)
    # A callee package is read as source too: a local one from the directory the project's
    # uv.lock names, a published one from the exact wheel it pins (cr-114).
    pinned = locked.read(project)
    roots = (*environment_roots(environment_python, project=project), *pinned.roots)
    reader = Reader(project, dependency_roots=roots, wheels=pinned.wheels)
    return _build(reader, application)


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
    return _build(reader, application)


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
        # uv's symlink link mode (a seeded cache) links each installed file into its cache.
        path = metadata / name
        if not path.is_file():
            raise StaticRefusal("installed wheel metadata is absent", code="static_distribution")
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

    # Ownership is by installed path, never a link target: a linked install's targets are
    # in uv's cache, outside site-packages.
    def installed(path: Path) -> Path:
        return Path(os.path.normpath(path.absolute()))

    site = installed(metadata.parent)
    owned = {
        installed(site / row[0])
        for row in csv.reader(read("RECORD").splitlines())
        if len(row) == 3 and row[0].endswith(".py")
    }
    # The wheel's own modules are the package: site-packages at large is its dependencies,
    # followed only where a declared type needs them, never scanned for registrations.
    own = frozenset(
        path.relative_to(site).parts[0].removesuffix(".py")
        for path in owned
        if path.is_relative_to(site)
    )
    reader = Reader(metadata.parent, dependency_roots=roots, own=own)
    entry = reader.application_module(application.partition(":")[0])
    source = installed(entry.path)
    if not source.is_relative_to(site) or source not in owned:
        raise StaticRefusal(
            "installed App source is not owned by its wheel", code="static_distribution"
        )
    return _build(reader, application, installed=True)


def _build(reader: Reader, application: str, *, installed: bool = False) -> dict[str, Json]:
    module_name, _, attribute = application.partition(":")
    entry = reader.application_module(module_name)
    app = reader.lookup(entry, attribute, entry.tree)
    if not (isinstance(app, Assigned) and _is_app_call(reader, app)):
        raise StaticRefusal(
            f"{application}: {attribute!r} is not `{attribute} = App()` at module scope of "
            f"{reader.spell(entry.path)}",
            code="not_an_app",
        )
    declared: dict[str, Declaration] = {}
    modules = reader.reachable(entry)
    recipes, calibrated = _recipes(reader, modules)
    registrations = _registrations(reader, modules, app)
    if recipes:
        registrations.append(_recipes_registration(reader, recipes, calibrated))
    for registration in registrations:
        if registration.name in declared:
            reader.refuse(
                registration.module,
                registration.fn,
                f"duplicate registration {registration.name!r}",
                "static_unsupported",
            )
        row = _callable(reader, registration)
        if installed and (registration.memoize or registration.weights):
            identity = _operation_identity(reader, registration)
            if identity is not None:
                row = replace(row, identity={"operation_identity": identity})
        declared[registration.name] = row
    return package_interface.assemble(
        application, list(declared.values()), installed=installed, recipes=recipes
    )


def _recipes(
    reader: Reader, modules: Sequence[Module]
) -> tuple[list[dict[str, Json]], list[ModelSpec]]:
    """Every `Recipe` bound at module scope of `modules`, as the interface's `recipes` rows, and
    the models of those that calibrate."""
    rows: list[dict[str, Json]] = []
    calibrated: list[ModelSpec] = []
    for module in modules:
        for stmt in _module_scope(module.tree.body):
            name = _single_target(stmt)
            value = getattr(stmt, "value", None)
            if name is None or not isinstance(value, ast.Call):
                continue
            if not isinstance(value.func, (ast.Name, ast.Attribute)):
                continue
            try:
                callee = reader.resolve(module, value.func)
            except StaticRefusal:
                continue
            if not (isinstance(callee, Vocab) and callee.key == "author.Recipe"):
                continue
            recipe = reader.recipe(module, value, _NO_SCOPE)
            row: dict[str, Json] = {
                "name": name.lower(),
                "lane": recipe.lane,
                "module": module.name,
                "export": name,
                "model": {
                    "class": recipe.model.name,
                    "component_use": {
                        stage: sorted(names)
                        for stage, names in sorted(recipe.model.components.items())
                    },
                },
                "plan": recipe.canonical(),
            }
            if recipe.calibrate:
                row["calibrate"] = recipe.calibrate
                if recipe.model not in calibrated:
                    calibrated.append(recipe.model)
            elif not isinstance(recipe.plan, str):
                row["fingerprint"] = fingerprint(recipe.canonical())
            if any(other["name"] == row["name"] for other in rows):
                reader.refuse(module, stmt, f"a second recipe named {row['name']!r}")
            if len(calibrated) > 1:
                reader.refuse(module, stmt, "calibrated recipes of two model classes")
            rows.append(row)
    return rows, calibrated


def _recipes_registration(
    reader: Reader, recipes: Sequence[dict[str, Json]], calibrated: Sequence[ModelSpec]
) -> Registration:
    """The Runtime's `quantize` (`cozy_runtime.derive.recipes`), read from its source, writing
    the lanes of `recipes`: a package with recipes registers it as its own. One with a calibrated
    recipe registers `quantize_calibrated` under that name: on a GPU, its model built on demand."""
    runtime = reader.module("cozy_runtime.derive.recipes")
    entry = CALIBRATED_ENTRYPOINT if calibrated else RECIPES_ENTRYPOINT
    function = runtime.bindings[entry]
    assert isinstance(function, ast.AsyncFunctionDef)
    found = _registration(reader, runtime, function, "entrypoint", ())
    assert found is not None
    lanes = outputs(str(row["lane"]) for row in recipes)
    if not calibrated:
        return replace(found, weights=lanes, accelerator=False)
    return replace(
        found, name=RECIPES_ENTRYPOINT, weights=lanes, model_on_demand=("model",),
        model_classes={"model": calibrated[0]},
    )  # fmt: skip


def _operation_identity(reader: Reader, registration: Registration) -> str | None:
    """A memoized callable's operation identity read from installed source alone: the same key
    `memo_implementation.describe` computes by import. None when any part of its declaration
    cannot be read statically; its machine then keys the memo by its package's files."""
    version: object = ""
    dependencies: list[tuple[Module, ast.expr]] = []
    try:
        for argument, module, node in registration.memo_declaration:
            if argument == "memo_version":
                version = reader.fold(module, node)
            elif isinstance(node, (ast.Tuple, ast.List)):
                dependencies.extend((module, element) for element in node.elts)
            else:
                return None
        if not isinstance(version, str):
            return None
        fn = registration.fn
        definitions = [
            (f"{registration.module.name}.{fn.name}", memo_identity.normalized_definition(fn))
        ]
        for module, node in dependencies:
            definition = _memo_dependency(reader, module, node)
            if definition is None:
                return None
            definitions.append(definition)
    except (StaticRefusal, OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None
    return memo_identity.definitions_identity(version, definitions)


def _memo_dependency(reader: Reader, module: Module, node: ast.expr) -> tuple[str, str] | None:
    """One declared memo dependency as `describe` names and hashes it, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        path = _installed_source(reader, node.value.split("."))
        if path is None or path.stat().st_size > memo_identity.MAX_SOURCE_BYTES:
            return None
        return node.value, memo_identity.normalized(path.read_bytes())
    if isinstance(node, ast.Call):
        found = reader.resolve(module, node.func)
        arguments = [reader.fold(module, a) for a in node.args] + [
            reader.fold(module, k.value) for k in node.keywords
        ]
        if not isinstance(found, Vocab) or not all(isinstance(a, str) for a in arguments):
            return None
        if found.key == "author.MemoDistribution" and len(arguments) == 1:
            name = typing.cast(str, arguments[0])
            # Each environment root once: the reader names its site-packages more than one way.
            built = memo_identity.installed_distribution_identity(
                name, reader.roots, _interpreter.CACHE_TAG
            )
            return None if built is None else ("distribution:" + name, built)
        if found.key == "author.MemoResource" and len(arguments) == 2:
            package, name = typing.cast(list[str], arguments)
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                return None
            for root in reader.roots:
                resource = root.joinpath(*package.split("."), *relative.parts)
                if resource.is_file():
                    digest = hashlib.sha256(resource.read_bytes()).hexdigest()
                    return f"resource:{package}/{name}", digest
        return None
    found = reader.resolve(module, node)
    if isinstance(found, Definition):
        return f"{found.module.name}.{found.node.name}", memo_identity.normalized_definition(
            found.node
        )
    return None


def _installed_source(reader: Reader, parts: Sequence[str]) -> Path | None:
    """A module's source file in the described environment, read as a file, never imported."""
    for root in (*reader.roots, reader.runtime_root):
        for path in (
            root.joinpath(*parts).with_suffix(".py"),
            root.joinpath(*parts, "__init__.py"),
        ):
            if path.is_file():
                return path
    return None


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
    if not isinstance(target, ast.Attribute) or target.attr != "entrypoint":
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
    invocable = False
    exported: author.Memoize | None = None
    memoize: author.Memoize | None = None
    default_models: dict[str, str] = {}
    used_components: dict[str, tuple[str, ...]] = {}
    memo_declaration: list[tuple[str, Module, ast.expr]] = []
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
                keyword.arg not in _INVOCABLE_KEYWORDS for keyword in export.keywords
            ):
                reader.refuse(
                    function_module,
                    export,
                    "@invocable accepts memoize, memo_version, memo_dependencies, defaults "
                    "and components",
                    "static_unsupported",
                )
            for keyword in export.keywords:
                if keyword.arg in {"memo_version", "memo_dependencies"}:
                    # Operation identity is an installed environment's fact (build_installed).
                    memo_declaration.append((keyword.arg, function_module, keyword.value))
                    continue
                value = reader.fold(function_module, keyword.value)
                if keyword.arg == "defaults":
                    default_models = model_defaults(value)
                    continue
                if keyword.arg == "components":
                    used_components = model_components(value)
                    continue
                exported = _memoize(reader, function_module, keyword.value, value)
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
    name = fn.name
    publishes = emits_media = False
    internal = False
    accelerator = True
    hidden = False
    weights: tuple[author.WeightsOutput, ...] = ()
    on_demand: tuple[str, ...] = ()
    for keyword in keywords:
        if keyword.arg not in _ENTRYPOINT_KEYWORDS:
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
            memo_declaration.append((keyword.arg, module, keyword.value))
            continue
        value = reader.fold(module, keyword.value)
        if keyword.arg == "defaults":
            declared_defaults = model_defaults(value)
            if default_models and declared_defaults:
                reader.refuse(
                    module, keyword, "declare defaults once, on invocable or App registration"
                )
            default_models = declared_defaults or default_models
        elif keyword.arg == "components":
            declared_components = model_components(value)
            if used_components and declared_components:
                reader.refuse(
                    module, keyword, "declare components once, on invocable or App registration"
                )
            used_components = declared_components or used_components
        elif keyword.arg == "on_demand":
            on_demand = model_on_demand(value)
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
                reader.refuse(module, keyword.value, "accelerator is True or False")
            accelerator = value
        elif keyword.arg == "publishes":
            publishes = bool(value)
        elif keyword.arg == "emits_media":
            emits_media = bool(value)
        elif keyword.arg == "memoize":
            memoize = _memoize(reader, module, keyword.value, value)
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
        name,
        publishes,
        weights,
        invocable,
        one_memo(exported, memoize),
        default_models,
        internal,
        accelerator,
        tuple(memo_declaration),
        used_components,
        on_demand,
        emits_media=emits_media,
    )


def _memoize(
    reader: Reader, module: Module, node: ast.expr, value: object
) -> author.Memoize | None:
    if isinstance(value, (bool, author.Memoize)):
        return memo_policy(value)
    reader.refuse(module, node, "memoize must be a boolean or Memoize")


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


def _callable(reader: Reader, registration: Registration) -> Declaration:
    if registration.invocable:
        return static_invocable.build(reader, registration)
    module, fn = registration.module, registration.fn
    if fn.args.vararg or fn.args.kwarg:
        reader.refuse(module, fn, "*args/**kwargs on a handler", "static_unsupported")
    if fn.args.defaults or any(fn.args.kw_defaults):
        reader.refuse(
            module, fn, "a handler parameter is injected, so it has no default", "injected_default"
        )
    payloads: list[object] = []
    slots: list[Slot] = []
    contexts: list[str] = []
    services: list[str] = []
    facts = 0
    assets: AssetsBinding | None = None
    for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        if arg.annotation is None:
            reader.refuse(module, arg, "an unannotated handler parameter")
        annotation_module, node, found = _parameter_annotation(reader, module, arg.annotation)
        slot = reader.model_annotation(annotation_module, node)
        if slot is not None:
            slots.append(Slot(arg.arg, *slot))
            continue
        if isinstance(found, Vocab):
            vocab_name = found.key.partition(".")[2]
            if found.key in ("author.Context", "author.Preflight") or vocab_name in (
                "Settings",
                "Secrets",
                *(cls.__name__ for cls in SERVICES.values()),
            ):
                if isinstance(node, ast.Subscript):
                    reader.translate(annotation_module, node.slice)
                parameter = _classify(
                    arg.arg, reader.translate(module, arg.annotation), f"{module.name}:{fn.name}"
                )
                if parameter.role == "context":
                    contexts.append(arg.arg)
                facts += parameter.role == "preflight"
                if parameter.capability:
                    services.append(parameter.capability)
                continue
            if vocab_name in ("Assets", "Annotated"):
                if assets is not None:
                    reader.refuse(module, arg, "a second Assets parameter", "static_unsupported")
                assets = assets_binding(arg.arg, reader.translate(module, arg.annotation))
                continue
            if found.key in _author_structs():
                payloads.append(reader.translate(module, arg.annotation))
                continue

        if (
            isinstance(found, Definition)
            and isinstance(found.node, ast.ClassDef)
            and reader.class_kind(found.module, found.node) == "struct"
        ):
            payloads.append(reader.translate(module, arg.annotation))
            continue
        reader.refuse(module, arg.annotation, f"parameter {arg.arg}: not an author surface type")
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
    return registration.declaration(
        payload_type=payload,
        result_type=result,
        contexts=tuple(contexts),
        services=tuple(services),
        facts=facts,
        slots=tuple(slots),
        assets=assets,
    )


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
    # POSIX venvs keep lib/pythonX.Y/site-packages; Windows venvs keep Lib/site-packages.
    sites = [*prefix.glob("lib/python*/site-packages"), *prefix.glob("Lib/site-packages")]
    for directory in sorted(dict.fromkeys(site.resolve() for site in sites)):
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
