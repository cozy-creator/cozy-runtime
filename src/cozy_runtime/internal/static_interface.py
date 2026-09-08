"""The package interface from SOURCE: `ast` over the package's files, executing no package code.

INVARIANT (decision #713 — package code is UNTRUSTED): this module never imports, execs,
compiles or evaluates a package module. No `importlib`, no `sys.path`, no `exec`/`eval`/
`compile`, no annotation evaluation, no module top level. It reads text, parses it with
`ast`, follows the package's own module-scope import graph (plus cozy-runtime's own source
for `cozy_runtime.derive.*`), folds constants with its own evaluator over literal and
arithmetic nodes, and translates every interface-bearing declaration into real types built
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
import enum
import tomllib
import typing
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import msgspec

import cozy_runtime
import cozy_runtime.internal.package_interface as package_interface
from cozy_runtime import author
from cozy_runtime.author._markers import MODEL_DEFAULT, PREFLIGHT
from cozy_runtime.author._model import Model
from cozy_runtime.author._services import SERVICES
from cozy_runtime.author._signature import AssetsBinding, assets_binding, assets_payload
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
    """cozy-runtime's own source (`cozy_runtime.derive.*`): followed for types and constants,
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
_TRUSTED_ROOT = "cozy_runtime.derive"
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
#: Author-surface constructors a folded constant may call: each validates its own arguments.
_AUTHOR_VALUES = ("AssetBound", "AssetLimits", "WeightsOutput", "Bound", "Shape")
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
_MODEL_MEMBERS = frozenset(name for name in dir(Model) if not name.startswith("_"))
_ENTRYPOINT_KEYWORDS = frozenset({"name", "preflight", "demand", "hidden"})
_JOB_KEYWORDS = frozenset({"name", "publishes", "emits_media", "weights", "demand"})


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

    def __init__(self, project: Path) -> None:
        self.project = project
        self.roots = (project / "src", project)
        self.runtime_root = Path(cozy_runtime.__file__).resolve().parent.parent
        self.modules: dict[str, Module] = {}
        self.structs: dict[tuple[Path, str], type[msgspec.Struct] | None] = {}
        self.enums: dict[tuple[Path, str], type[enum.Enum]] = {}
        self.models: dict[tuple[Path, str], ModelSpec] = {}

    # ---------------------------------------------------------------- modules

    def module(self, name: str, *, where: tuple[Module, ast.AST] | None = None) -> Module:
        if (loaded := self.modules.get(name)) is not None:
            return loaded
        trusted = name == _TRUSTED_ROOT or name.startswith(_TRUSTED_ROOT + ".")
        roots = (self.runtime_root,) if trusted else self.roots
        parts = name.split(".")
        for root in roots:
            for path, is_package in (
                (root.joinpath(*parts).with_suffix(".py"), False),
                (root.joinpath(*parts, "__init__.py"), True),
            ):
                if path.is_file():
                    return self._load(name, path, is_package, trusted)
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
        if name in _VOCAB_MODULES or name.split(".")[0] not in self._tree_tops(name):
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
        return set()

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
            if name in _BUILTIN_TYPES:
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
        if target.startswith(_TRUSTED_ROOT) or self._tree_tops(target):
            return ModuleRef(self.module(target, where=(module, stmt)))
        return External(target)

    def _from_binding(
        self, module: Module, base: str | None, name: str, stmt: ast.ImportFrom
    ) -> Resolved:
        if base is None:
            self.refuse(module, stmt, "a relative import above the package", "static_module")
        if base in _VOCAB_MODULES:
            return Vocab(f"{_VOCAB_MODULES[base]}.{name}")
        if base.startswith("cozy_runtime.") and not base.startswith(_TRUSTED_ROOT):
            return External(f"{base}.{name}")
        if not (base.startswith(_TRUSTED_ROOT) or self._tree_tops(base)):
            return External(f"{base}.{name}")
        source = self.module(base, where=(module, stmt))
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

    def fold(self, module: Module, node: ast.expr) -> object:
        """The value of a literal/arithmetic expression, through module-level names."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            items = [self.fold(module, item) for item in node.elts]
            return tuple(items) if isinstance(node, ast.Tuple) else items
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                self.refuse(module, node, "a dict splat", "static_computed")
            return {
                self.fold(module, key): self.fold(module, value)
                for key, value in zip(node.keys, node.values, strict=True)
                if key is not None
            }
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
            operand = self.fold(module, node.operand)
            if isinstance(operand, bool) or not isinstance(operand, (int, float)):
                self.refuse(module, node, "unary arithmetic on a non-number", "static_computed")
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            return ~int(operand)
        if isinstance(node, ast.BinOp):
            return self._arithmetic(module, node)
        if isinstance(node, (ast.Name, ast.Attribute)):
            found = self.resolve(module, node)
            if isinstance(found, Assigned):
                return self.fold(found.module, found.value)
            if isinstance(found, Vocab) and found.key == "author.DEFAULT_DECODE_LIMITS":
                return author.DEFAULT_DECODE_LIMITS
            self.refuse(module, node, "not a module-level constant")
        if isinstance(node, ast.Call):
            callee = self.resolve(module, node.func)
            if isinstance(callee, Vocab) and callee.key in (
                *(f"author.{name}" for name in _AUTHOR_VALUES),
                "msgspec.Meta",
            ):
                if callee.key == "author.Shape":
                    # A demand-axis marker: its tables are code, and it is not interface content.
                    return author.Shape(pixels={})
                constructor = _vocab_object(callee.key)
                assert callable(constructor)
                args = [self.fold(module, arg) for arg in node.args]
                kwargs = {
                    keyword.arg: self.fold(module, keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                if len(kwargs) != len(node.keywords):
                    self.refuse(module, node, "a keyword splat", "static_computed")
                return constructor(*args, **kwargs)
            self.refuse(
                module, node, "a call: its value exists only at run time", "static_computed"
            )
        self.refuse(
            module, node, "an expression the constant folder does not evaluate", "static_computed"
        )

    def _arithmetic(self, module: Module, node: ast.BinOp) -> object:
        left, right = self.fold(module, node.left), self.fold(module, node.right)
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
                return self.translate(found.module, found.value)
            if isinstance(found, Definition) and isinstance(found.node, ast.ClassDef):
                return self.class_type(found.module, found.node, node)
            if isinstance(found, External):
                self.refuse(module, node, f"{found.spelling} is outside the closed vocabulary")
            self.refuse(module, node, "not a type")
        self.refuse(module, node, "not a type expression")

    def _vocab_type(self, module: Module, node: ast.expr, found: Vocab) -> object:
        prefix, _, name = found.key.partition(".")
        if prefix == "builtins":
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
                if "default" in options or "default_factory" in options:
                    return name, annotation, msgspec.field(default_factory=_unfolded, name=wire)
                return name, annotation, msgspec.field(name=wire)
        # The default's VALUE is never interface content — only that one exists.
        return name, annotation, msgspec.field(default_factory=_unfolded)

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

    # ---------------------------------------------------------------- errors

    def spell(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project))
        except ValueError:
            return str(path.relative_to(self.runtime_root))

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
    """A default the reader did not evaluate: a default's VALUE is not interface content."""
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


def build(project: Path) -> dict[str, Json]:
    """The package interface from source alone. Executes nothing of the package."""
    project = project.resolve()
    application = read_application(project)
    reader = Reader(project)
    module_name, _, attribute = application.partition(":")
    entry = reader.application_module(module_name)
    app = reader.lookup(entry, attribute, entry.tree)
    if not (isinstance(app, Assigned) and _is_app_call(reader, app)):
        raise StaticRefusal(
            f"{application}: {attribute!r} is not `{attribute} = App()` at module scope of "
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
    return package_interface.assemble(
        application, list(docs.values()), read_table(project / "package.toml")
    )


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
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and _app_method(reader, module, stmt.value.func, app) is not None
            ):
                reader.refuse(
                    module,
                    stmt,
                    "registration by call is not a declaration; use the decorator",
                    "static_unsupported",
                )
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in stmt.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if _is_invocable(reader, module, target):
                    reader.refuse(
                        module,
                        decorator,
                        "@invocable exports are described by the conformance oracle only",
                        "static_unsupported",
                    )
                method = _app_method(reader, module, target, app)
                if method is None:
                    continue
                keywords = decorator.keywords if isinstance(decorator, ast.Call) else []
                if isinstance(decorator, ast.Call) and decorator.args:
                    reader.refuse(
                        module, decorator, "positional decorator arguments", "static_unsupported"
                    )
                registration = _registration(reader, module, stmt, method, keywords)
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
) -> Registration | None:
    allowed = _ENTRYPOINT_KEYWORDS if method == "entrypoint" else _JOB_KEYWORDS
    name = fn.name
    publishes = False
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
        value = reader.fold(module, keyword.value)
        if keyword.arg == "name":
            name = str(value)
        elif keyword.arg == "hidden" and value is True:
            return None
        elif keyword.arg == "publishes":
            publishes = bool(value)
        elif keyword.arg == "weights":
            assert isinstance(value, tuple)
            weights = tuple(value)
    return Registration(
        module, fn, typing.cast(package_interface.Kind, method), name, publishes, weights
    )


def _callable(reader: Reader, registration: Registration) -> dict[str, Json]:
    module, fn = registration.module, registration.fn
    if fn.args.vararg or fn.args.kwarg:
        reader.refuse(module, fn, "*args/**kwargs on a handler", "static_unsupported")
    payloads: list[object] = []
    models: list[dict[str, Json]] = []
    assets: AssetsBinding | None = None
    for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        if arg.annotation is None:
            reader.refuse(module, arg, "an unannotated handler parameter")
        node = arg.annotation
        target = node.value if isinstance(node, ast.Subscript) else node
        if isinstance(target, ast.Subscript):  # Annotated[Assets[...], AssetLimits(...)]
            target = target.value
        found = (
            reader.resolve(module, target)
            if isinstance(target, (ast.Name, ast.Attribute))
            else None
        )
        if isinstance(found, Vocab):
            vocab_name = found.key.partition(".")[2]
            if found.key in ("author.Context", "author.Preflight") or vocab_name in (
                "Settings",
                "Secrets",
            ):
                if isinstance(node, ast.Subscript):
                    reader.translate(module, node.slice)
                continue
            if vocab_name in {cls.__name__ for cls in SERVICES.values()}:
                continue
            if vocab_name in ("Assets", "Annotated"):
                if assets is not None:
                    reader.refuse(module, arg, "a second Assets parameter", "static_unsupported")
                assets = assets_binding(arg.arg, reader.translate(module, node))
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
                    )
                )
                continue
            if kind == "struct":
                payloads.append(reader.translate(module, node))
                continue
        reader.refuse(module, node, f"parameter {arg.arg}: not an author surface type")
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
    return package_interface.callable_doc(
        name=registration.name,
        kind=registration.kind,
        payload_type=payload,
        result_type=result,
        models=models,
        assets=assets,
        publishes=registration.publishes,
        weights_outputs=registration.weights,
    )
