"""`cozy-runtime derive` — the CodeTopology of (slot, construction document).

    cozy-runtime [--json] derive --python <venv python> --slot <path> --config <document>
                                 [--application <module:attr> | --dir <project>] [--out <dir>]

`derive(package_release, slot, config_document) = census(factory(config_document))` on the
meta device, in the package's own interpreter, in one bounded child (model-code-fit §3).
The answer is the canonical TensorFS CodeTopology — rows with `accepts` from the class's
`encoded_leaves`, plus the components the factory constructed, in order (D9). It is a pure
function of its two inputs: no weights, no device, no candidate, and byte-identical on every
run — the child derives twice and refuses a construction that differs from itself.

`--python` is the interpreter whose environment holds the package: a project venv for an
author, the install venv for the worker. `--dir` names a project tree (`package.toml`);
`--application` names an installed export instead. `--out` writes the exact canonical bytes
to `<dir>/<sha256-hex>.json`; `--json` prints them.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cozy_runtime.author import Artifact, Config, ConformanceError, ModelBinding
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import canonical, derive_child, sandbox
from cozy_runtime.internal.derive import DeriveError, derive
from cozy_runtime.internal.derive_child import (
    DerivedTopology,
    DeriveRefusal,
    DeriveRequest,
    SlotRequest,
)
from cozy_runtime.internal.discovery import discover, discover_installed
from cozy_runtime.internal.exits import Exit

USAGE = (
    "cozy-runtime [--json] derive --python <venv python> --slot <path> --config <document> "
    "[--application <module:attr> | --dir <project>] [--out <dir>]"
)
VALUE_FLAGS = ("--python", "--slot", "--config", "--application", "--out")


def run(args: Sequence[str], opts: Options) -> Result:
    flags = _flags(args)
    # NOT resolved: a venv's `bin/python` is a symlink to the base interpreter, and the
    # venv is exactly what the symlink's own location selects.
    python = Path(flags["--python"]).expanduser().absolute()
    if not python.is_file():
        raise CliError(
            "not_found", f"--python {python}: no such interpreter", USAGE, Exit.not_found
        )
    slot = flags["--slot"]
    config_path = Path(flags["--config"]).expanduser().resolve()
    application = flags.get("--application", "")
    project = None if application else Path(opts.dir).expanduser().resolve()
    scratch = Path(tempfile.mkdtemp(prefix="cozy-derive-"))
    try:
        # The document is read as JSON and re-emitted canonical: the derive is keyed on the
        # construction document's IDENTITY, which is its canonical bytes.
        document = canonical.parse(derive_child.read_exact(config_path, "config"))
        if not isinstance(document, dict):
            raise DeriveRefusal("derive_input_invalid", f"{config_path}: config is not an object")
        staged = scratch / "config.json"
        staged.write_bytes(canonical.write(document))
        request = DeriveRequest(
            (SlotRequest(slot, derive_child.config_ref(staged)),),
            project=project,
            application=application,
        )
        (held,) = derive_child.derive_in(python, request, scratch)
    except canonical.CanonicalError as exc:
        raise CliError("validation", f"{config_path}: {exc}", USAGE, Exit.validation) from exc
    except DeriveRefusal as exc:
        raise CliError(
            exc.code,
            exc.detail,
            "repair the slot path, the config document or the package's factory and derive again",
            Exit.structural,
        ) from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    constructed = derive_child.constructed_components(held.document)
    rows: list[tuple[str, str]] = [
        ("slot", held.slot),
        ("config_digest", held.config_digest),
        ("code_topology", held.digest),
        ("length", str(len(held.document))),
        ("constructed", ", ".join(constructed)),
    ]
    lines: tuple[str, ...] = ()
    if flags.get("--out"):
        out_dir = Path(flags["--out"]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{held.digest.removeprefix('sha256:')}.json"
        target.write_bytes(held.document)
        lines = (f"wrote {len(held.document)} canonical bytes to {target}",)
    return Result(
        rows=tuple(rows),
        canonical_json=held.document,
        lines=lines,
        next=(f"tfs fit {held.digest.removeprefix('sha256:')}.json <store> <header> --json",),
    )


def _flags(args: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    rest = list(args)
    while rest:
        item = rest.pop(0)
        name, eq, inline = item.partition("=")
        if name not in VALUE_FLAGS:
            raise CliError("usage", f"derive received invalid argument {item!r}", USAGE, Exit.usage)
        if not eq:
            if not rest:
                raise CliError("usage", f"{name} needs a value", USAGE, Exit.usage)
            inline = rest.pop(0)
        if name in out:
            raise CliError("usage", f"{name} was passed twice", USAGE, Exit.usage)
        out[name] = inline
    for required in ("--python", "--slot", "--config"):
        if not out.get(required):
            raise CliError("usage", f"derive needs {required}", USAGE, Exit.usage)
    return out


# ------------------------------------------------------------------------- the child


def child_main(argv: Sequence[str]) -> int:
    """`python -I -B -m cozy_runtime.cli.derive --derive-child <request>`: derive every
    requested slot under the audit fences and print the one result line."""
    if len(argv) != 1:
        derive_child.emit({"error": "derive_child_usage", "detail": "one request path required"})
        return int(Exit.usage)
    try:
        request = derive_child.read_request(derive_child.read_exact(Path(argv[0]), "request"))
        _warm_torch()
        sandbox.install(allow_write_prefixes=())
        sandbox.bound(
            address_bytes=derive_child.MEMORY_CAP_BYTES,
            cpu_seconds=derive_child.CPU_CAP_SECONDS,
        )
        found = (
            discover(request.project)
            if request.project is not None
            else discover_installed(request.application)
        )
        bindings: dict[str, ModelBinding] = {
            binding.path: binding
            for surface in found.surfaces
            if not surface.hidden
            for binding in surface.model_bindings
        }
        topologies: list[DerivedTopology] = []
        for row in request.slots:
            binding = bindings.get(row.slot)
            if binding is None:
                raise DeriveRefusal(
                    "derive_slot_unknown",
                    f"{row.slot!r} is no model slot of this package (have "
                    f"{', '.join(sorted(bindings)) or 'none'})",
                )
            raw = derive_child.read_exact(row.config.path, f"config for {row.slot}")
            measured = "sha256:" + hashlib.sha256(raw).hexdigest()
            if len(raw) != row.config.length or measured != row.config.digest:
                raise DeriveRefusal(
                    "derive_input_identity_mismatch",
                    f"{row.config.path} does not match {row.config.digest}/{row.config.length}",
                )
            try:
                document = canonical.parse_canonical(raw)
            except canonical.CanonicalError as err:
                raise DeriveRefusal("derive_input_invalid", f"{row.slot}: {err}") from err
            if not isinstance(document, dict):
                raise DeriveRefusal("derive_input_invalid", f"{row.slot}: config is not an object")
            config = Config._verified(document, "config", row.config.digest)
            topologies.append(
                DerivedTopology(row.slot, row.config.digest, derive_twice(binding, config))
            )
    except DeriveRefusal as err:
        derive_child.emit({"error": err.code, "detail": err.detail})
        return int(Exit.structural)
    except ConformanceError as err:
        derive_child.emit(
            {"error": str(getattr(err, "code", "derive_conformance")), "detail": str(err)}
        )
        return int(Exit.structural)
    except (DeriveError, sandbox.FenceViolation) as err:
        name = getattr(err, "construct", None) or getattr(err, "fence", None)
        derive_child.emit({"error": str(name or "derive_refused"), "detail": str(err)})
        return int(Exit.structural)
    except Exception as err:
        derive_child.emit({"error": "derive_internal", "detail": f"{type(err).__name__}: {err}"})
        return int(Exit.internal)
    derive_child.emit(derive_child.render(topologies))
    return int(Exit.ok)


def derive_twice(binding: ModelBinding, config: Config) -> bytes:
    """The derive is deterministic or it is not a fact: two runs, byte-identical."""

    def once() -> bytes:
        result = derive(
            binding.model_class(),
            Artifact("derive://" + config.digest(), {}, config, custody="canonical"),
            component_use=binding.components,
        )
        if result.config_digest != config.digest():
            raise DeriveRefusal(
                "derive_config_digest_mismatch",
                f"{binding.path} did not retain the exact config document identity",
            )
        return result.tensorfs_bytes(encoded_leaves=binding.encoded_leaves)

    first = once()
    if once() != first:
        raise DeriveRefusal(
            "derive_nondeterministic",
            f"{binding.path} produced different canonical bytes on rebuild",
        )
    return first


def _warm_torch() -> None:
    """Warm torch's own lazy imports before rlimits, never package code or a GPU."""
    __import__("torch")
    for name in ("torch._dynamo", "torch._refs"):
        try:
            __import__(name)
        except ModuleNotFoundError as err:
            if err.name != name:
                raise


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--derive-child"]:
        return child_main(args[1:])
    print(USAGE, file=sys.stderr)
    return int(Exit.usage)


if __name__ == "__main__":
    raise SystemExit(main())
