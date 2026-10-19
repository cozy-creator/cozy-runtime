"""Checkpoints through the tensorfs API: header rows in construction order, files, the census fit.

A managed executor is granted a checkpoint's header and files by TensorD and parses the header
with `tensorfs.Checkpoint.parse`; a standalone caller reads a `tensorfs.Store` itself. Either
way the Runtime reads headers only through the wheel. Rows carry each tensor's scheme, params
and parts (`Encoded`); refusals are spoken in `FillRefusal`'s vocabulary.
"""

from __future__ import annotations

import contextlib
import functools
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import msgspec
import tensorfs

from cozy_runtime.author._loader import TENSORFS_DTYPES, TensorSpec, canonical_dtype
from cozy_runtime.internal.encoding import PLAIN, TORCH_DTYPES, Encoded, RolePart, params_of
from cozy_runtime.internal.encoding.formats import part_rule
from cozy_runtime.internal.resolution import PlanRefusal, ResolvedModelPlan

#: The header's path in a manifest.
HEADER = "model.cozytensors"
#: TensorD's `ModelSource` name for a checkpoint's manifest (its file list); `""` is its header.
LISTING = "."


class _Listing(msgspec.Struct):
    """A `tensorfs.manifest/1` document, as far as its files' paths."""

    files: dict[str, object]


#: Element bytes per Runtime dtype name.
WIDTHS = {
    "f64": 8,
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "f8_e4m3fn": 1,
    "f8_e5m2": 1,
    "f8_e8m0": 1,
    "i64": 8,
    "i32": 4,
    "i16": 2,
    "i8": 1,
    "u8": 1,
    "bool": 1,
}


class FillRefusal(Exception):
    """A typed fill refusal, raised before any byte reaches a destination."""

    def __init__(self, message: str, *, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(fields)


#: Every code a fill can refuse with, and only those.
REFUSALS = frozenset(
    {
        "missing_object",  # a chunk or file the store does not hold
        "digest_mismatch",  # bytes that do not match their digest
        "length_mismatch",  # lengths disagree
        "dtype_mismatch",  # a dtype this runtime has no binding for, or not the destination's
        "destination_absent",  # a component or key the checkpoint does not carry
        "incomplete_fill",  # a destination the census never reached
        "lease_live",  # a staging slot reused before its completion event
        "derived_unmaterialized",  # a non-persistent buffer the fill plane cannot supply
        "unknown_encoding",  # a scheme no provider here decodes
        "encoding_unqualified",  # no capability record for this device
        "role_mismatch",  # parts no provider accounts for
        "checkpoint_unreadable",  # a header or document that does not parse
        "store_unusable",  # a store TensorFS refuses to open
        "carrier_geometry",  # parts that break their scheme's dtype or shape rule
        "geometry_unsupported",  # every implementation refuses this shape or params
        "leaf_unconsented",  # the package did not consent to a replaced linear op
        "leaf_schema",  # the module tree cannot take the leaf the plan substitutes
        "plan_divergence",  # the resolved plan names what this checkpoint or build does not
        "plan_unresolved",  # no executable plan resolves for these bytes here
    }
)


def _refuse(code: str, message: str, fields: Sequence[str] = ()) -> FillRefusal:
    if code not in REFUSALS:
        raise AssertionError(f"{code} is not a declared fill refusal")
    return FillRefusal(message, code=code, fields=fields)


#: TensorFS error families in this vocabulary; any other code passes through unchanged.
STORE_REFUSALS: dict[str, str] = {
    "integrity": "digest_mismatch",
    "not_found": "missing_object",
    "invalid": "checkpoint_unreadable",
    "format": "checkpoint_unreadable",
    "decode": "carrier_geometry",
    "unknown_scheme": "unknown_encoding",
    "io": "store_unusable",
}


@contextlib.contextmanager
def store_refusals(what: str) -> Iterator[None]:
    """Speak a TensorFS error in this vocabulary, or let it through untranslated."""
    try:
        yield
    except tensorfs.Error as exc:
        code = STORE_REFUSALS.get(exc.code)
        if code is None:
            raise
        raise _refuse(code, f"{what}: TensorFS refused {exc.code} - {exc.detail}") from exc


@dataclass(frozen=True, slots=True)
class PlanRow:
    """One header row: what the checkpoint holds for one logical tensor."""

    key: str
    """The contract's key, component-qualified (`unet.conv_in.weight`)."""
    name: str
    """The header's key (`conv_in.weight`)."""
    component: str
    dtype: str
    """The logical dtype: the destination's, and what a decode writes."""
    shape: tuple[int, ...]
    nbytes: int
    """Logical bytes; an encoded tensor stores `stored_nbytes`."""
    encoded: Encoded

    def role_what(self, role: str) -> str:
        """The part's name in a read plan, `<component>/<name>#<role>`."""
        return f"{self.component}/{self.name}#{role}"

    @property
    def stored_nbytes(self) -> int:
        return self.encoded.stored_nbytes


def _dtype(key: str, name: str) -> str:
    dtype = TENSORFS_DTYPES.get(name)
    if dtype is None or dtype not in TORCH_DTYPES:
        raise _refuse("dtype_mismatch", f"{key}: dtype {name!r} has no torch binding here", [key])
    return dtype


def _part(key: str, role: str, part: tensorfs.Part) -> RolePart:
    shape = tuple(part.shape)
    if part.dtype == "F4":  # packed two per byte: a decoder reads u8 [..., C/2]
        if len(shape) < 2 or shape[-1] % 2:
            raise _refuse("carrier_geometry", f"{key}: F4 part {role} is {list(shape)}", [key])
        return RolePart(role, "u8", (*shape[:-1], shape[-1] // 2), part.length)
    return RolePart(role, _dtype(key, part.dtype), shape, part.length)


def row(component: str, name: str, info: tensorfs.Info, bound: str = "") -> PlanRow:
    """One header tensor as a plan row, bound to the module name `bound` (default: its own);
    a part that breaks its scheme refuses."""
    key = f"{component}.{bound or name}"
    dtype = _dtype(key, info.dtype)
    shape = tuple(info.shape)
    nbytes = WIDTHS[dtype]
    for extent in shape:
        nbytes *= extent
    if info.encoding is None:
        encoded = Encoded(PLAIN, (), (_part(key, "value", info.parts[""]),))
    else:
        parts = tuple(_part(key, role, part) for role, part in info.parts.items())
        encoded = Encoded(info.encoding.scheme, params_of(info.encoding.params), parts)
        why = part_rule(encoded, shape, dtype)
        if why:
            raise _refuse("carrier_geometry", f"{key}: {why}", [key])
    return PlanRow(key, name, component, dtype, shape, nbytes, encoded)


class Checkpoint:
    """One checkpoint's header rows (in its construction order) and files."""

    def __init__(
        self,
        manifest_id: str,
        ck: tensorfs.Checkpoint,
        files: Sequence[str],
        read_file: Callable[[str], bytes],
    ) -> None:
        self.manifest_id = manifest_id
        self.ck = ck
        self.files = tuple(files)
        self._read = read_file

    @classmethod
    def granted(cls, manifest_id: str, source: Callable[[str], bytes]) -> Checkpoint:
        """A checkpoint TensorD granted this executor (`ModelSource`): its manifest (`LISTING`)
        names the files, each read through the grant once (the header as `""`)."""
        files = msgspec.json.decode(source(LISTING), type=_Listing).files
        read = functools.cache(lambda path: source("" if path == HEADER else path))
        with store_refusals(f"parsing {manifest_id[:23]}"):
            ck = tensorfs.Checkpoint.parse(read(HEADER))
        return cls(manifest_id, ck, tuple(files), read)

    @classmethod
    def from_store(cls, store: tensorfs.Store, ref: str) -> Checkpoint:
        """A checkpoint of a store this process opened itself (standalone use, tests)."""
        with store_refusals(f"opening {ref}"):
            ck = store.checkpoint(ref)
        manifest = ck.manifest
        assert manifest is not None
        return cls(f"sha256:{manifest.sha256}", ck, ck.files, ck.file)

    def has(self, component: str) -> bool:
        return component in self.ck.tensors

    def rows(self, component: str) -> list[PlanRow]:
        """The component's rows in the header's (construction) order."""
        tensors = self.ck.tensors.get(component)
        if tensors is None:
            raise _refuse(
                "destination_absent",
                f"this checkpoint has no component {component!r} (it has "
                f"{', '.join(self.ck.tensors) or 'none'})",
            )
        bind = self.binding(component)
        return [row(component, name, info, bind(name)) for name, info in tensors.items()]

    def binding(self, component: str) -> Callable[[str], str]:
        """Stored name -> the name the component's module tree uses: a Transformers model's
        own checkpoint conversion mapping, found by its config's class (`CLIPTextModel`
        drops the `text_model.` its checkpoint stores); a name binds as stored otherwise."""
        mapping = sys.modules.get("transformers.conversion_mapping")
        loading = sys.modules.get("transformers.core_model_loading")
        path = f"{component}/config.json"
        if mapping is None or loading is None or path not in self.files:
            return str
        config = json.loads(self.file(path))
        transforms = None
        for name in (*config.get("architectures", ())[:1], config.get("model_type")):
            transforms = mapping.get_checkpoint_conversion_mapping(name) if name else None
            if transforms:
                break
        renamings = [t for t in transforms or () if isinstance(t, loading.WeightRenaming)]
        if not renamings:
            return str
        return lambda name: str(loading.rename_source_key(name, renamings, [])[0])

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self.ck.metadata)

    def file(self, path: str) -> bytes:
        with store_refusals(f"reading {path} of {self.manifest_id[:23]}"):
            return self._read(path)

    def model_assets(self) -> dict[str, bytes]:
        """Every file of the manifest but the header: configs, tokenizers, adapter graphs."""
        return {path: self.file(path) for path in self.files if path != HEADER}

    def table(self, component: str) -> dict[str, list[object]]:
        """A component's stored tensors as plain data: what a memo of its weights is keyed on
        (chunks by digest, so it moves with the bytes)."""
        chunks = self.ck.chunks
        out: dict[str, list[object]] = {}
        for key, info in self.ck.tensors.get(component, {}).items():
            encoding = info.encoding
            out[key] = [
                info.dtype,
                list(info.shape),
                encoding.scheme if encoding else PLAIN,
                sorted(encoding.params.items()) if encoding else [],
                sorted(
                    [
                        role,
                        part.dtype,
                        list(part.shape),
                        chunks[part.chunk][0],
                        part.offset,
                        part.length,
                    ]
                    for role, part in info.parts.items()
                ),
            ]
        return out

    def configs(self) -> dict[str, bytes]:
        """Construction configs by component: `<component>/config.json` (or a scheduler's
        `scheduler_config.json`), and a root `config.json` as `model`."""
        found: dict[str, bytes] = {}
        for path in self.files:
            folder, _, name = path.rpartition("/")
            if name in ("config.json", "scheduler_config.json") and "/" not in folder:
                found[folder or "model"] = self.file(path)
        return found


@dataclass(frozen=True, slots=True)
class TensorRequirement:
    """One tensor the construction built."""

    component: str
    key: str
    shape: Sequence[int]
    logical_dtype: str | None = None


def fit(
    requirements: Sequence[TensorRequirement],
    checkpoint: Checkpoint,
    *,
    encoded_leaves: bool,
    custody: str = "",
    device: str = "",
) -> dict[str, object]:
    """The construction census against the header: every built tensor is stored with its shape
    (and its dtype where the census pins one); stored tensors nothing builds are skipped.
    `ok` False carries the first disagreement's code, component, key and sentence. The header's
    order is the construction's traversal (#624): `order` names where the census builds in
    another order (weights then stream out of step), "" when it agrees. Whether a device
    serves an encoding is plan resolution's (`Selector`); custody and device echo."""
    routes = {"verbatim": 0, "decoded_float": 0, "encoded_gemm": 0}
    ignored: list[str] = []
    disorder: list[str] = []
    ignored_bytes = 0

    def refused(
        code: str, detail: str, component: str = "", key: str = "", encoding: str = ""
    ) -> dict[str, object]:
        return {"ok": False, "code": code, "component": component, "key": key,
                "detail": detail, "encoding": encoding, "custody": custody,
                "device": device}  # fmt: skip

    built: dict[str, dict[str, TensorRequirement]] = {}
    for requirement in requirements:
        built.setdefault(requirement.component, {})[requirement.key] = requirement
    for component, wanted in built.items():
        if not checkpoint.has(component):
            return refused(
                "component_missing",
                f"the code builds component {component!r} and the checkpoint stores "
                f"{', '.join(checkpoint.ck.tensors) or 'none'}",
                component,
            )
        bind = checkpoint.binding(component)
        stored = {bind(name): info for name, info in checkpoint.ck.tensors[component].items()}
        for key, requirement in wanted.items():
            info = stored.get(key)
            if info is None:
                return refused(
                    "tensor_missing",
                    f"the code builds {component}/{key} and the checkpoint does not store it",
                    component,
                    key,
                )
            if tuple(info.shape) != tuple(requirement.shape):
                return refused(
                    "shape_mismatch",
                    f"{component}/{key}: the code builds {list(requirement.shape)}, the checkpoint "
                    f"stores {list(info.shape)}",
                    component,
                    key,
                )
            stored_dtype = TENSORFS_DTYPES.get(info.dtype, info.dtype)
            if requirement.logical_dtype and stored_dtype != requirement.logical_dtype:
                return refused(
                    "dtype_mismatch",
                    f"{component}/{key}: the code requires {requirement.logical_dtype}, the "
                    f"checkpoint stores {stored_dtype}",
                    component,
                    key,
                )
            if info.encoding is None:
                routes["verbatim"] += 1
            elif not encoded_leaves:
                return refused(
                    "encoding_unsupported",
                    f"{component}/{key} is stored {info.encoding.scheme} and the package does not "
                    "consent to encoded leaves",
                    component,
                    key,
                    info.encoding.scheme,
                )
            else:
                routes["encoded_gemm"] += 1
        for key, info in stored.items():
            if key not in wanted:
                ignored.append(f"{component}/{key}")
                ignored_bytes += sum(part.length for part in info.parts.values())
        header = [key for key in stored if key in wanted]
        census = list(wanted)
        if header != census:
            at = next(i for i, (a, b) in enumerate(zip(header, census, strict=True)) if a != b)
            disorder.append(
                f"{component}: the code builds {census[at]} where the header's construction "
                f"order has {header[at]}"
            )
    warning = ""
    if ignored:
        more = f", and {len(ignored) - 5} more" if len(ignored) > 5 else ""
        warning = (
            f"the code builds {len(ignored)} fewer tensors than the checkpoint stores "
            f"({ignored_bytes} B, not read): {', '.join(ignored[:5])}{more}"
        )
    return {
        "ok": True,
        "code": "ok",
        "component": "",
        "key": "",
        "detail": "",
        "encoding": "",
        "custody": custody,
        "device": device,
        "ignored": ignored,
        "ignored_bytes": ignored_bytes,
        "routes": routes,
        "warning": warning,
        "order": "; ".join(disorder),
    }


@dataclass(frozen=True, slots=True)
class Servability:
    """This runtime's own answer to "can I serve these bytes", from plan resolution."""

    serves: bool
    by_scheme: dict[str, int]
    """Every scheme the header stores, and how many tensors."""
    routes: dict[str, int]
    refusal: str
    code: str

    def line(self) -> str:
        cited = ", ".join(f"{scheme} x{n}" for scheme, n in sorted(self.by_scheme.items()))
        if self.serves:
            served = ", ".join(f"{route} x{n}" for route, n in sorted(self.routes.items()))
            return f"serves {cited} via {served}"
        return f"CANNOT serve {cited}: [{self.code}] {self.refusal}"

    def require(self) -> None:
        if not self.serves:
            raise _refuse(self.code, self.refusal)


def servability(rows: Sequence[PlanRow], answer: ResolvedModelPlan | PlanRefusal) -> Servability:
    """The serve answer derived from plan resolution, never from a second selector."""
    by_scheme: dict[str, int] = {}
    for one in rows:
        by_scheme[one.encoded.scheme] = by_scheme.get(one.encoded.scheme, 0) + 1
    if isinstance(answer, ResolvedModelPlan):
        return Servability(True, by_scheme, answer.route_census(), "", "")
    summary = {"no_rows", "unconsented", "unscored", "eligible", "chosen", "chosen_uncalibrated"}
    for step in answer.steps:
        if step.verdict in summary:
            continue
        code = "encoding_unqualified" if step.verdict == "unqualified" else step.verdict
        if code in REFUSALS:
            return Servability(False, by_scheme, {}, step.why, code)
    return Servability(False, by_scheme, {}, answer.detail, "plan_unresolved")


def tensor_schema_of(rows: Sequence[PlanRow]) -> dict[str, TensorSpec]:
    """The tensor schema the loader matches the code against."""
    return {
        one.key: TensorSpec(one.shape, dtype_name(one.dtype), one.encoded.scheme, one.nbytes)
        for one in rows
    }


def dtype_name(dtype: object) -> str:
    """The one dtype spelling: torch's (`float16`), from either side's name."""
    name = str(getattr(dtype, "name", dtype)).removeprefix("torch.")
    return TORCH_DTYPES.get(name, name)


def tensorfs_requirement_dtype(dtype: object | None) -> str | None:
    """A census dtype the fit pins; bf16 and f16 stay convertible."""
    if dtype is None:
        return None
    name = canonical_dtype(dtype)
    return None if name in ("bf16", "f16") else name


def normalized(checkpoint: Checkpoint) -> str:
    """Why a construction may not read this checkpoint (stored as-is, its keys not
    normalized), else ""."""
    if checkpoint.metadata.get("normalized") is not False:
        return ""
    converters = checkpoint.metadata.get("converters")
    return (
        "this checkpoint was stored as-is from a layout no converter recognized "
        f"({json.dumps(converters)}), so its keys are not normalized: re-import it once a "
        "TensorFS converter recognizes its layout"
    )
