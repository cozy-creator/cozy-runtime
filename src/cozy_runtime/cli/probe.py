"""`cozy-runtime probe` — the standalone tier-2 verb over an installed serving package.

The composed gate pipeline (cr-073 tier 3: serving generates, eval judges) needs the same
teacher-forced per-layer map a quant job takes in-process through the author API. This verb
is that second caller over the one core: it discovers the package like `describe`, binds
the probed entrypoint's ONE constructing slot to each of two LOCAL LANE TREES, constructs
both lanes through the package's own `Model.load` -> `Loader.construct` (the same factory
that runs under derive and serve — constructing-slot evidence covers every probe), drives
ONE real generation of the entrypoint through the real invocation kernel to record the
reference trajectory, and emits the raw ``cozy.probe.report/1`` document. Stats only —
the verdict is the gate's (cozy-eval's ladder), never the runtime's.

A lane tree is a directory of ``*.safetensors`` whose keys are full destination keys
(component-prefixed, e.g. ``unet.down_blocks.…``) in a plain float carrier, plus an
optional ``config.json`` (the artifact's immutable construction config). That is the
derive world's own carrier shape; encoded lanes (fp8 payload+scale leaves) ride the
encoded-delivery plane, not this verb. The drive's runtime side is deliberately inert —
the per-service fakes over the production kernel — because the generation is a measurement
pass whose outputs are discarded; nothing here serves, publishes or records an attempt.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    Artifact,
    AuthorError,
    Config,
    ConformanceError,
    Device,
    Model,
    ModelBinding,
    ModelRegistry,
    probe_activations,
)
from cozy_runtime.cli import payload as payload_grammar
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit
from cozy_runtime.probe import REPORT_FORMAT, ProbeRefusal

VALUE_FLAGS = ("--component", "--reference", "--candidate", "--zoom", "--seed", "--in", "--out")

USAGE = (
    "cozy-runtime [--json] [--dir <path>] probe <function> --component <name> "
    "--reference [label=]<dir> --candidate [label=]<dir> [--zoom <tap>] "
    "[--seed <n>] [--in <file>] [--out <report.json>] [k=v …]"
)

#: safetensors carrier names -> the runtime's dtype spellings (census/_dtype_of).
_DTYPES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def run(args: list[str], opts: Options) -> Result:
    terms, flags = _split(args)
    if not terms:
        raise CliError("usage", "probe needs a function name", USAGE, Exit.usage)
    function, payload_terms = terms[0], terms[1:]
    component = _one(flags, "--component")
    if not component:
        raise CliError("usage", "probe needs --component <name>", USAGE, Exit.usage)
    reference_label, reference_dir = _lane_spec(_one(flags, "--reference"), "--reference")
    candidate_label, candidate_dir = _lane_spec(_one(flags, "--candidate"), "--candidate")
    zoom = _one(flags, "--zoom") or None

    project = Path(opts.dir).expanduser().resolve()
    try:
        found = discover(project)
        body = package_interface.build(found)
    except ConformanceError as exc:
        raise CliError(
            "validation",
            str(exc),
            "a package's surface must be valid before it can be probed (§1.0)",
            Exit.validation,
        ) from exc
    surface = next(
        (
            s
            for s in found.surfaces
            if s.name == function and s.kind == "entrypoint" and not s.hidden
        ),
        None,
    )
    if surface is None:
        known = sorted(s.name for s in found.surfaces if s.kind == "entrypoint" and not s.hidden)
        raise CliError(
            "not_found",
            f"no registered entrypoint named {function!r}",
            f"this release registers: {', '.join(known) or '0 entrypoints'}",
            Exit.not_found,
            next=("cozy-runtime describe — the full list",),
        )
    slots = surface.model_bindings
    if len(slots) != 1:
        raise CliError(
            "structural",
            f"{function} declares {len(slots)} constructing model slots; the probe drives "
            "exactly one",
            "probe an entrypoint whose signature declares one Model parameter — both lanes "
            "bind that slot",
            Exit.structural,
        )
    slot = slots[0]

    doc = _entrypoint_doc(body, function)
    document = payload_grammar.parse(
        payload_terms, doc["request"], infile=_one(flags, "--in") or ""
    )
    seed = _one(flags, "--seed")
    if seed and "seed" in payload_grammar.fields(doc["request"]):
        document["seed"] = int(seed)

    release = package_interface.package_interface_digest(body)
    try:
        reference_model = _construct(slot, reference_label, reference_dir, release)
        candidate_model = _construct(slot, candidate_label, candidate_dir, release)
        report = probe_activations(
            reference_model,
            candidate_model,
            _drive(found.app.get(function), document, reference_model, slot.param),
            component=component,
            reference_label=reference_label,
            candidate_label=candidate_label,
            zoom=zoom,
        )
    except ProbeRefusal as exc:
        raise CliError(
            exc.code,
            str(exc),
            "the probe measures or refuses typed — it never emits a wrong number",
            Exit.structural,
        ) from exc
    except AuthorError as exc:
        raise CliError(
            getattr(exc, "code", "validation") or "validation",
            str(exc),
            "repair the lane trees / payload and probe again",
            Exit.validation,
        ) from exc

    rendered = canonical_json.encode(report)
    rows = [
        ("format", REPORT_FORMAT),
        ("reference", reference_label),
        ("candidate", candidate_label),
        ("component", component),
        ("zoom", zoom or "none"),
        ("steps", str(report["steps"])),
        ("taps", str(len(report["taps"]))),
        ("rows", str(len(report["rows"]))),
    ]
    out = _one(flags, "--out")
    if out:
        out_path = Path(out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(rendered)
        rows.append(("out", str(out_path)))
    return Result(
        rows=tuple(rows),
        document=report,
        canonical_json=rendered,
        next=("the verdict is gate policy: fold this document with cozy-eval's ladder",),
    )


def _drive(registration: Any, document: dict[str, Any], model: Model[Any], param: str) -> Any:
    """One ordinary generation of the entrypoint through the REAL invocation kernel, with
    the reference lane bound. The runtime side is inert (per-service fakes over that same
    kernel) and the spool is discarded: this run exists to be recorded, not served."""

    def drive() -> None:
        from cozy_runtime.author.fakes import invoke_with_fakes

        with tempfile.TemporaryDirectory(prefix="cozy-probe-drive.") as spool:
            invoke_with_fakes(registration, document, models={param: model}, spool=Path(spool))

    return drive


def _construct(slot: ModelBinding, label: str, tree: Path, release: str) -> Model[Any]:
    """One lane, constructed through the package's own `Model.load` -> `Loader.construct`
    with the lane tree as its artifact — tensor-schema matched before a byte moves."""
    index, tensor_schema = _lane_index(label, tree)
    config_path = tree / "config.json"
    config = Config(json.loads(config_path.read_text())) if config_path.is_file() else Config({})
    artifact = Artifact(
        snapshot=f"file://{tree}",
        tensor_schema=tensor_schema,
        config=config,
        custody="local",
    )
    registry = ModelRegistry(release=release, backend=_LaneBackend(label, index), device=Device())
    return registry.acquire(slot.path, slot.model_class, artifact)


def _lane_index(
    label: str, tree: Path
) -> tuple[dict[str, tuple[Path, dict[str, Any], int]], dict[str, Any]]:
    from cozy_runtime.author._loader import TensorSpec
    from cozy_runtime.derive import safetensors_io

    files = sorted(tree.glob("*.safetensors"))
    if not files:
        raise CliError(
            "not_found",
            f"lane {label!r}: {tree} holds no *.safetensors",
            "a lane tree is a directory of safetensors whose keys are full destination "
            "keys, plus an optional config.json",
            Exit.not_found,
        )
    index: dict[str, tuple[Path, dict[str, Any], int]] = {}
    tensor_schema: dict[str, Any] = {}
    for path in files:
        header, base = safetensors_io.read_header(path)
        for key, row in header.items():
            if key in index:
                raise CliError(
                    "validation",
                    f"lane {label!r}: {key!r} appears in more than one safetensors file",
                    "a destination key has exactly one supplier per lane",
                    Exit.validation,
                )
            index[key] = (path, header, base)
            shape = tuple(int(extent) for extent in row["shape"])
            start, end = (int(x) for x in row["data_offsets"])
            tensor_schema[key] = TensorSpec(
                shape,
                _DTYPES.get(str(row["dtype"]), str(row["dtype"]).lower()),
                nbytes=end - start,
            )
    return index, tensor_schema


class _LaneBackend:
    """The Backend seam over one local lane tree: real stored bytes into the exact
    destinations the census agreed on. The package cannot observe which backend served it."""

    def __init__(self, label: str, index: dict[str, tuple[Path, dict[str, Any], int]]) -> None:
        self._label = label
        self._index = index

    def fit(self, walked: Any, *, encoded_leaves: str) -> None:
        """A lane tree is safetensors, not a cozytensors header: there is no checkpoint
        document for `tensorfs.fit` to judge, so nothing is judged. A destination the tree
        does not supply, or supplies at another shape, refuses at `fill` by name."""
        return None

    def materialize(self, obj: object, walked: Any, *, encoded_leaves: str = "refuse") -> Any:
        return walked.live

    def fill(self, key: str, spec: Any, destination: Any) -> None:
        import torch

        from cozy_runtime.derive import safetensors_io

        if key not in self._index:
            raise CliError(
                "validation",
                f"lane {self._label!r} supplies no {key!r}; the code constructs it",
                "a lane tree supplies every destination the entrypoint's model constructs",
                Exit.validation,
            )
        path, header, base = self._index[key]
        row = header[key]
        if tuple(int(x) for x in row["shape"]) != tuple(destination.shape):
            raise CliError(
                "validation",
                f"lane {self._label!r}: {key} is stored {list(row['shape'])}, the code "
                f"constructs {list(destination.shape)}",
                "a lane tree supplies every destination at the shape the code constructs",
                Exit.validation,
            )
        raw = safetensors_io.read_raw(path, header, base, key)
        values = safetensors_io.to_f32(raw, str(row["dtype"]), [int(x) for x in row["shape"]])
        with torch.no_grad():
            destination.copy_(torch.from_numpy(values.copy()))

    def commit(self) -> None:
        return None

    def poison(self, keys: Sequence[str]) -> None:
        """The verb process is the transaction: a failed lane ends the verb, typed."""
        return None


def _entrypoint_doc(body: dict[str, Any], function: str) -> dict[str, Any]:
    entrypoints = body["entrypoints"]
    assert isinstance(entrypoints, list)
    for doc in entrypoints:
        assert isinstance(doc, dict)
        if doc["name"] == function:
            return doc
    raise CliError("not_found", f"{function!r} left the package interface", USAGE, Exit.internal)


def _lane_spec(value: str, flag: str) -> tuple[str, Path]:
    if not value:
        raise CliError("usage", f"probe needs {flag} [label=]<dir>", USAGE, Exit.usage)
    label, eq, spelled = value.partition("=")
    if not eq:
        label, spelled = "", value
    tree = Path(spelled).expanduser().resolve()
    if not tree.is_dir():
        raise CliError(
            "not_found",
            f"{flag} {spelled}: no such directory",
            f"{flag} names a local lane tree (safetensors + optional config.json)",
            Exit.not_found,
        )
    return label or tree.name, tree


def _one(flags: dict[str, list[str]], name: str) -> str:
    values = flags.get(name) or [""]
    if len(values) > 1:
        raise CliError("usage", f"{name} was passed {len(values)} times", USAGE, Exit.usage)
    return values[0]


def _split(args: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
    terms: list[str] = []
    flags: dict[str, list[str]] = {}
    rest = list(args)
    while rest:
        item = rest.pop(0)
        if item.split("=")[0] in VALUE_FLAGS:
            name, _, inline = item.partition("=")
            if not inline:
                if not rest:
                    raise CliError("usage", f"{name} needs a value", USAGE, Exit.usage)
                inline = rest.pop(0)
            flags.setdefault(name, []).append(inline)
        elif item.startswith("--"):
            raise CliError("usage", f"unknown flag {item!r}", USAGE, Exit.usage)
        else:
            terms.append(item)
    return terms, flags
