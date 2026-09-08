"""The `cozy-runtime` entrypoint: the ONE config authority, flag parsing, verb dispatch.

No interactive prompt exists anywhere; anything that would ask is a flag.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from cozy_runtime.cli import describe as describe_verb
from cozy_runtime.cli import host as host_verbs
from cozy_runtime.cli import hostfacts as hostfacts_verb
from cozy_runtime.cli import image_prepare as image_prepare_verb
from cozy_runtime.cli import interface_wheel as interface_wheel_verb
from cozy_runtime.cli import new as new_verb
from cozy_runtime.cli import package_prepare as package_prepare_verbs
from cozy_runtime.cli import run as run_verb
from cozy_runtime.cli import version as version_verb
from cozy_runtime.cli.io import CliError, Options, Result, emit, emit_error
from cozy_runtime.internal.config import ConfigError, RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit

BUILT = (
    "run",
    "job",
    "serve",
    "describe",
    "bindings",
    "doctor",
    "capability",
    "hostfacts",
    "new",
    "version",
)
PRIVATE = (
    "development-placement",
    "prepare-package",
    "interface-wheel",
    "image-prepare",
    "image-preparation-profile",
)
HELP = """cozy-runtime — everything a single package venv can do alone

usage: cozy-runtime [--json] [--full] [--fields a,b] [--dir <path>] <verb> [args]

verbs:
  run <fn> [<primary>] [k=v …]   one-shot inference: payload in, outputs to disk, exit
  job <fn> [k=v …]               run one @app.job to completion, same attempt kernel
  serve --socket <path> <mode>   speak the worker protocol for a RecordOwner
  describe [<fn>]                the PackageInterface, from source alone (no package code runs)
  bindings                       the RESOLVED binding records this project selects
  doctor                         this host's device/driver/credential facts
  capability [--out <path>]      the OBSERVED (encoding, device) qualification of this card
  hostfacts [<fn>] [k=v …]       this host's capacity facts (exact with a payload)
  new <name>                     scaffold a minimal valid package skeleton
  version                        tag, commit, and the two independent semvers

run/job flags:
  --model [path=]<ref>  override a model binding (repeatable)
  --tensorfs-root <dir> explicit local Store root for a model-bound source run
  --input <ref>=<dir>   grant one materialized input tree (job only, repeatable)
  --org <name>          publishing org; a job's writes land in <org>/_job-<request-id>
  --out <dir>           output directory (default ./outputs)
  --seed <n>            deterministic RNG
  --capture <json>      activation capture: {"components":["unet"],"steps":[0,1]}
  --in <file>           whole payload as JSON, key=value merged on top
  key=@/absolute/file   stage a local file only at a declared Asset field
  kernel.attention=<k>  DEV: pin one attention kernel for this run instead of the fastest
                        one this device and image support; refuses if it cannot serve
  --offline             CAS-only; a miss is exit 8, never a fetch
  --timeout <dur>       attempt deadline (<n>ms|s|m|h)
  --strict-keys         a local bind's unknown extra keys refuse instead of confessing
  --stream              typed deltas as NDJSON

serve storage:
  --tensorfs-root <dir>    serve selected model Manifests from one explicit local Store

global flags:
  --json           the full typed result as one JSON document
  --full           widen output, lift truncation
  --fields a,b,c   pick columns
  --dir <path>     package project root (default: cwd)
  --package-interface <path>
                   exact PackageInterface input for describe/bindings (no package import)
  --environment-python <venv/bin/python>
                   describe/bindings source roots of an explicit venv; never executes it
  --conformance    describe only: also IMPORT the app in this interpreter — which must be
                   the package's own locked venv — and refuse unless it matches the source
  -h, --help       this text
"""

_VALUE_FLAGS = ("--fields", "--dir", "--package-interface", "--environment-python")
#: Verbs that own their own flag parsing: everything after the verb belongs to them, so a
#: payload term like `--in` is not eaten by the global pass.
_OWNS_ARGS = (
    "run",
    "job",
    "hostfacts",
    "capability",
    "serve",
    "development-placement",
    "prepare-package",
)


def parse(argv: Sequence[str]) -> tuple[str | None, list[str], Options, bool]:
    """Split argv into (verb, args, options, help). Unknown flags refuse with exit 2."""
    as_json = full = want_help = conformance = False
    fields: tuple[str, ...] = ()
    directory = "."
    package_interface = ""
    environment_python = ""
    verb: str | None = None
    args: list[str] = []
    unknown: list[str] = []
    rest = list(argv)
    while rest:
        item = rest.pop(0)
        if verb in _OWNS_ARGS:
            # The verb owns everything after itself. `run` has flags of its own and a
            # payload grammar that spells `key=value`; a global pass that tried to
            # interpret those would have to know the package's schema.
            args.append(item)
        elif item in ("-h", "--help"):
            want_help = True
        elif item == "--json":
            as_json = True
        elif item == "--full":
            full = True
        elif item == "--conformance":
            conformance = True
        elif item.startswith("--") and item.split("=")[0] in _VALUE_FLAGS:
            flag, _, inline = item.partition("=")
            if not inline:
                if not rest:
                    raise CliError(
                        "usage", f"{flag} needs a value", f"pass {flag} <value>", Exit.usage
                    )
                inline = rest.pop(0)
            if flag == "--fields":
                fields = tuple(f.strip() for f in inline.split(",") if f.strip())
            elif flag == "--dir":
                directory = inline
            elif flag == "--environment-python":
                environment_python = inline
            else:
                package_interface = inline
        elif item.startswith("-") and item != "-":
            unknown.append(item)
        elif verb is None:
            verb = item
        else:
            args.append(item)
    if unknown:
        raise CliError(
            "usage",
            f"unknown flag {unknown[0]!r}",
            "run `cozy-runtime --help` for the flags this build accepts",
            Exit.usage,
            next=("cozy-runtime --help",),
        )
    return (
        verb,
        args,
        Options(
            as_json, full, fields, directory, package_interface, conformance, environment_python
        ),
        want_help,
    )


def dispatch(verb: str, args: list[str], opts: Options, config: RuntimeConfig) -> Result:
    if opts.environment_python and verb not in ("describe", "bindings"):
        raise CliError(
            "usage",
            "--environment-python belongs only to describe or bindings",
            "cozy-runtime describe --environment-python <venv/bin/python>",
            Exit.usage,
        )
    if opts.package_interface and verb not in ("describe", "bindings"):
        raise CliError(
            "usage",
            "--package-interface belongs only to describe or bindings",
            "use --package-interface <exact-package-interface> with "
            "cozy-runtime describe or bindings",
            Exit.usage,
        )
    if opts.conformance and verb != "describe":
        raise CliError(
            "usage",
            "--conformance belongs only to describe",
            "cozy-runtime --conformance describe, from the package's locked venv",
            Exit.usage,
        )
    if verb == "describe":
        if len(args) > 1:
            raise CliError(
                "usage",
                f"describe takes at most one function name, got {len(args)}",
                "cozy-runtime [--json] [--dir <path>] [--package-interface <path>] "
                "[--conformance] describe [<function>]",
                Exit.usage,
            )
        return describe_verb.run(args[0] if args else None, opts)
    if verb == "version":
        return version_verb.run(config, opts.full)
    if verb == "new":
        if len(args) != 1:
            raise CliError(
                "usage",
                f"new takes exactly one name, got {len(args)}",
                "cozy-runtime new <name>",
                Exit.usage,
            )
        return new_verb.run(args[0], opts.dir)
    if verb in ("run", "job"):
        return run_verb.run(verb, args, opts, config)
    if verb == "bindings":
        if args:
            raise CliError(
                "usage",
                f"bindings takes no arguments, got {len(args)}",
                "cozy-runtime bindings [--json] [--dir <path>] — it resolves the project's "
                "declared selection and constructs nothing, so it takes no payload",
                Exit.usage,
            )
        return host_verbs.bindings(opts)
    if verb == "development-placement":
        return host_verbs.development_placement(config, args, opts)
    if verb == "prepare-package":
        return package_prepare_verbs.prepare_local_package(config, args, opts)
    if verb == "interface-wheel":
        return interface_wheel_verb.run(args)
    if verb == "image-prepare":
        return image_prepare_verb.run(args)
    if verb == "image-preparation-profile":
        return image_prepare_verb.profile(args)
    if verb == "hostfacts":
        return hostfacts_verb.run(args, opts, config)
    if verb == "capability":
        return host_verbs.capability(config, args, opts)
    if verb == "doctor":
        if args:
            raise CliError(
                "usage",
                f"doctor takes no arguments, got {len(args)}",
                "cozy-runtime doctor [--json] — it takes no payload; `cozy-runtime hostfacts` "
                "is the capacity verb",
                Exit.usage,
            )
        return host_verbs.doctor(config, opts)
    raise CliError(
        "usage",
        f"unknown verb {verb!r}",
        "run `cozy-runtime --help` for the verbs this build accepts",
        Exit.usage,
        next=("cozy-runtime --help",),
    )


def main(argv: Sequence[str] | None = None) -> int:
    opts = Options()
    try:
        verb, args, opts, want_help = parse(sys.argv[1:] if argv is None else argv)
        if want_help or verb is None:
            print(HELP, end="")
            return int(Exit.ok)
        config = read_config()
        if verb == "serve":
            # `serve` never returns a Result: it IS the process, and it runs until the
            # RecordOwner closes the session.
            return host_verbs.serve(config, args)
        emit(dispatch(verb, args, opts, config), opts)
        return int(Exit.ok)
    except CliError as err:
        emit_error(err, opts)
        return int(err.code)
    except ConfigError as err:
        emit_error(CliError(err.name, err.message, err.remedy, err.code), opts)
        return int(err.code)
    except Exception as err:
        emit_error(
            CliError(
                "internal",
                f"unexpected fault: {type(err).__name__}: {err}",
                "this is a bug in cozy-runtime — report it with the command you ran",
                Exit.internal,
            ),
            opts,
        )
        return int(Exit.internal)


if __name__ == "__main__":
    raise SystemExit(main())
