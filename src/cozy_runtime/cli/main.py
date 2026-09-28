"""The `cozy-runtime` entrypoint: the ONE config authority, flag parsing, verb dispatch.

No interactive prompt exists anywhere; anything that would ask is a flag.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from cozy_runtime.cli.io import CliError, Options, Result, emit, emit_error
from cozy_runtime.internal.config import ConfigError, RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit

BUILT = (
    "describe",
    "new",
    "version",
)
PRIVATE = (
    "model-ingestion-plan",
    "image-prepare",
    "image-preparation-profile",
    "python-interpreters",
    "python-ensure",
)
HELP = """cozy-runtime — everything a single package venv can do alone

usage: cozy-runtime [--json] [--full] [--fields a,b] [--dir <path>] <verb> [args]

verbs:
  describe [<fn>]                the PackageInterface, from source alone (no package code runs)
  new <name>                     scaffold a minimal valid package skeleton
  version                        tag, commit, and the two independent semvers

Packages run on a machine through `cozy run`; this CLI only reads and scaffolds them.

global flags:
  --json           the full typed result as one JSON document
  --full           widen output, lift truncation
  --fields a,b,c   pick columns
  --dir <path>     package project root (default: cwd)
  --package-interface <path>
                   exact PackageInterface input for describe (no package import)
  --environment-python <venv/bin/python>
                   describe source roots of an explicit venv; never executes it
  --distribution <name>
                   describe one installed wheel in --environment-python, without imports
  --builtin operations
                   describe Runtime's fixed managed operation surface, without imports
  --conformance    describe only: also IMPORT the app in this interpreter — which must be
                   the package's own locked venv — and refuse unless it matches the source
  -h, --help       this text
"""

_VALUE_FLAGS = (
    "--fields",
    "--dir",
    "--package-interface",
    "--environment-python",
    "--distribution",
    "--builtin",
)


def parse(argv: Sequence[str]) -> tuple[str | None, list[str], Options, bool]:
    """Split argv into (verb, args, options, help). Unknown flags refuse with exit 2."""
    as_json = full = want_help = conformance = False
    fields: tuple[str, ...] = ()
    directory = "."
    package_interface = ""
    environment_python = ""
    distribution = ""
    builtin = ""
    verb: str | None = None
    args: list[str] = []
    unknown: list[str] = []
    rest = list(argv)
    while rest:
        item = rest.pop(0)
        if item in ("-h", "--help"):
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
            elif flag == "--distribution":
                distribution = inline
            elif flag == "--builtin":
                builtin = inline
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
            as_json,
            full,
            fields,
            directory,
            package_interface,
            conformance,
            environment_python,
            distribution,
            builtin,
        ),
        want_help,
    )


def dispatch(verb: str, args: list[str], opts: Options, config: RuntimeConfig) -> Result:
    if verb == "model-ingestion-plan":
        from cozy_runtime.cli import model_ingestion

        return model_ingestion.run(args, opts)
    if opts.builtin and (
        verb != "describe"
        or opts.distribution
        or opts.environment_python
        or opts.package_interface
        or opts.conformance
    ):
        raise CliError(
            "usage",
            "--builtin belongs only to static describe",
            "cozy-runtime describe --builtin operations",
            Exit.usage,
        )
    if opts.distribution and (
        verb != "describe"
        or not opts.environment_python
        or opts.package_interface
        or opts.conformance
    ):
        raise CliError(
            "usage",
            "--distribution requires static describe with --environment-python only",
            "cozy-runtime describe --distribution <name> --environment-python <venv/bin/python>",
            Exit.usage,
        )
    if opts.environment_python and verb != "describe":
        raise CliError(
            "usage",
            "--environment-python belongs only to describe",
            "cozy-runtime describe --environment-python <venv/bin/python>",
            Exit.usage,
        )
    if opts.package_interface and verb != "describe":
        raise CliError(
            "usage",
            "--package-interface belongs only to describe",
            "use --package-interface <exact-package-interface> with cozy-runtime describe",
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
        from cozy_runtime.cli import describe as describe_verb

        if len(args) > 1:
            raise CliError(
                "usage",
                f"describe takes at most one function name, got {len(args)}",
                "cozy-runtime [--json] [--dir <path>] [--package-interface <path>] "
                "[--conformance] describe [<function>]",
                Exit.usage,
            )
        return describe_verb.run(args[0] if args else None, opts)
    if verb == "python-interpreters":
        from cozy_runtime.internal import python_interpreters

        if args:
            raise CliError(
                "usage",
                "python-interpreters takes no arguments",
                "cozy-runtime --json python-interpreters",
            )
        try:
            return Result(document=python_interpreters.document(root=config.managed_python_root))
        except python_interpreters.InterpreterRefusal as exc:
            raise CliError(exc.code, exc.detail, "install a supported Python interpreter") from exc
    if verb == "python-ensure":
        from cozy_runtime.internal import python_interpreters, storage_admission

        if not 1 <= len(args) <= 2:
            raise CliError(
                "usage",
                "python-ensure takes a requirement and optional version",
                "cozy-runtime --json python-ensure '<requires-python>' [<version>]",
            )
        try:
            selected = python_interpreters.ensure(
                args[0],
                args[1] if len(args) == 2 else "",
                root=config.managed_python_root,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            return Result(document=selected.document())
        except (python_interpreters.InterpreterRefusal, storage_admission.StorageRefusal) as exc:
            raise CliError(
                exc.code, exc.detail, "check Python requirements and uv availability"
            ) from exc
    # Interpreter plumbing stays portable without importing executor-only modules.
    from cozy_runtime.cli import image_prepare as image_prepare_verb
    from cozy_runtime.cli import new as new_verb
    from cozy_runtime.cli import version as version_verb

    if verb == "version":
        return version_verb.run(config, opts.full, opts.fields)
    if verb == "new":
        if len(args) != 1:
            raise CliError(
                "usage",
                f"new takes exactly one name, got {len(args)}",
                "cozy-runtime new <name>",
                Exit.usage,
            )
        return new_verb.run(args[0], opts.dir)
    if verb == "image-prepare":
        return image_prepare_verb.run(args)
    if verb == "image-preparation-profile":
        return image_prepare_verb.profile(args)
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
