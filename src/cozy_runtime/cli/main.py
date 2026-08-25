"""The `cozy-runtime` entrypoint: the ONE config authority, flag parsing, verb dispatch.

No interactive prompt exists anywhere — anything that would ask is a flag, and a
destructive act without `--yes` is a typed refusal (exit 7), never a question.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from cozy_runtime.cli import describe as describe_verb
from cozy_runtime.cli import new as new_verb
from cozy_runtime.cli import version as version_verb
from cozy_runtime.cli.io import CliError, Options, Result, emit, emit_error
from cozy_runtime.internal.config import ConfigError, RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit

BUILT = ("describe", "new", "version")
# Spec verbs whose implementation lands with cr-016, over the real attempt kernel.
PENDING = ("run", "serve", "job", "pull", "ingest", "list", "rm", "doctor", "fit")

HELP = f"""cozy-runtime — everything a single endpoint venv can do alone

usage: cozy-runtime [--json] [--full] [--fields a,b] [--dir <path>] <verb> [args]

verbs:
  describe [<fn>]  the EndpointDescriptor, without loading weights
  new <name>       scaffold a minimal valid endpoint skeleton
  version          tag, commit, and the two independent semvers

describe flags:
  --write-descriptor  commit endpoint.descriptor.json (bare describe is read-only)
  --check             refuse a stale committed descriptor (exit 13)

not in this build (cr-016): {", ".join(PENDING)}

global flags:
  --json           the full typed result as one JSON document
  --full           widen output, lift truncation
  --fields a,b,c   pick columns
  --dir <path>     endpoint project root (default: cwd)
  -h, --help       this text
"""

_VALUE_FLAGS = ("--fields", "--dir")
_VERB_FLAGS = describe_verb.FLAGS


def parse(argv: Sequence[str]) -> tuple[str | None, list[str], Options, bool]:
    """Split argv into (verb, args, options, help). Unknown flags refuse with exit 2."""
    as_json = full = want_help = False
    fields: tuple[str, ...] = ()
    directory = "."
    verb: str | None = None
    args: list[str] = []
    verb_flags: list[str] = []
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
            else:
                directory = inline
        elif item in _VERB_FLAGS:
            verb_flags.append(item)
        elif item.startswith("-") and item != "-":
            unknown.append(item)
        elif verb is None:
            verb = item
        else:
            args.append(item)
    # A pending verb's own refusal outranks its unknown flags: the verb is the real reason.
    if unknown and verb not in PENDING:
        raise CliError(
            "usage",
            f"unknown flag {unknown[0]!r}",
            "run `cozy-runtime --help` for the flags this build accepts",
            Exit.usage,
            next=("cozy-runtime --help",),
        )
    return verb, args + verb_flags, Options(as_json, full, fields, directory), want_help


def dispatch(verb: str, args: list[str], opts: Options, config: RuntimeConfig) -> Result:
    if verb == "describe":
        flags = [a for a in args if a in describe_verb.FLAGS]
        names = [a for a in args if a not in describe_verb.FLAGS]
        if len(names) > 1:
            raise CliError(
                "usage",
                f"describe takes at most one function name, got {len(names)}",
                "cozy-runtime describe [<function>] [--json|--write-descriptor|--check]",
                Exit.usage,
            )
        return describe_verb.run(
            names[0] if names else None,
            opts,
            write="--write-descriptor" in flags,
            check="--check" in flags,
        )
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
    if verb in PENDING:
        raise CliError(
            "usage",
            f"{verb!r} is a spec verb this build does not implement yet",
            "cr-016 implements it over the real attempt kernel; this build has: "
            + ", ".join(BUILT),
            Exit.usage,
        )
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
