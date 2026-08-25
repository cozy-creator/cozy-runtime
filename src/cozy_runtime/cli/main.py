"""The `cozy-runtime` entrypoint: the ONE config authority, flag parsing, verb dispatch.

No interactive prompt exists anywhere — anything that would ask is a flag, and a
destructive act without `--yes` is a typed refusal (exit 7), never a question.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from cozy_runtime.cli import describe as describe_verb
from cozy_runtime.cli import fit as fit_verb
from cozy_runtime.cli import host as host_verbs
from cozy_runtime.cli import new as new_verb
from cozy_runtime.cli import run as run_verb
from cozy_runtime.cli import version as version_verb
from cozy_runtime.cli.io import CliError, Options, Result, emit, emit_error
from cozy_runtime.internal.config import ConfigError, RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit

BUILT = ("run", "job", "serve", "describe", "doctor", "fit", "list", "rm", "new", "version")
#: `pull`/`ingest` PARSE and refuse typed: both are thin invocations of TensorFS's one
#: fetch+ingest implementation and that border is not a dependency of this build yet
#: (tfs-002/tfs-003). They are verbs, not unknown words — the refusal names the owner.
UNAVAILABLE = ("pull", "ingest")

HELP = """cozy-runtime — everything a single endpoint venv can do alone

usage: cozy-runtime [--json] [--full] [--fields a,b] [--dir <path>] <verb> [args]

verbs:
  run <fn> [<primary>] [k=v …]   one-shot inference: payload in, outputs to disk, exit
  job <fn> [k=v …]               run one @app.job to completion, same attempt kernel
  serve --socket <path>          speak the worker protocol for a coordinator
  describe [<fn>]                the EndpointDescriptor, without loading weights
  doctor                         this host's device/driver/CAS/credential facts
  fit [<fn>] [k=v …]             this host's fit verdicts (exact with a payload)
  list [<ref-prefix>]            CAS-resident checkpoints, one artifact one size
  rm <ref> … --yes               remove artifacts from the local CAS
  new <name>                     scaffold a minimal valid endpoint skeleton
  version                        tag, commit, and the two independent semvers

run/job flags:
  --model [path=]<ref>  override a model binding (repeatable)
  --out <dir>           output directory (default ./outputs)
  --seed <n>            deterministic RNG
  --in <file>           whole payload as JSON, key=value merged on top
  --offline             CAS-only; a miss is exit 8, never a fetch
  --timeout <dur>       attempt deadline (<n>ms|s|m|h)
  --strict-keys         a local bind's unknown extra keys refuse instead of confessing
  --stream              typed deltas as NDJSON

describe flags:
  --write-descriptor  commit endpoint.descriptor.json (bare describe is read-only)
  --check             refuse a stale committed descriptor (exit 13)

global flags:
  --json           the full typed result as one JSON document
  --full           widen output, lift truncation
  --fields a,b,c   pick columns
  --dir <path>     endpoint project root (default: cwd)
  -h, --help       this text
"""

_VALUE_FLAGS = ("--fields", "--dir")
_VERB_FLAGS = describe_verb.FLAGS
#: Verbs that own their own flag parsing: everything after the verb belongs to them, so a
#: payload term like `--in` is not eaten by the global pass.
_OWNS_ARGS = ("run", "job", "fit", "serve", "rm", "pull", "ingest")


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
        if verb in _OWNS_ARGS:
            # The verb owns everything after itself. `run` has flags of its own and a
            # payload grammar that spells `key=value`; a global pass that tried to
            # interpret those would have to know the endpoint's schema.
            args.append(item)
        elif item in ("-h", "--help"):
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
    if unknown:
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
    if verb in ("run", "job"):
        return run_verb.run(verb, args, opts, config)
    if verb == "fit":
        return fit_verb.run(args, opts, config)
    if verb == "doctor":
        if args:
            raise CliError(
                "usage",
                f"doctor takes no arguments, got {len(args)}",
                "cozy-runtime doctor [--json] — it derives no fit verdict, so it takes no "
                "payload; `cozy-runtime fit` is the verdict verb",
                Exit.usage,
            )
        return host_verbs.doctor(config, opts)
    if verb == "list":
        if len(args) > 1:
            raise CliError(
                "usage",
                f"list takes at most one ref prefix, got {len(args)}",
                "cozy-runtime list [<ref-prefix>]",
                Exit.usage,
            )
        return host_verbs.list_artifacts(config, args[0] if args else "", opts)
    if verb == "rm":
        refs = [a for a in args if not a.startswith("-")]
        flags = [a for a in args if a.startswith("-")]
        if unknown := [f for f in flags if f != "--yes"]:
            raise CliError(
                "usage", f"unknown flag {unknown[0]!r}", "rm takes --yes only", Exit.usage
            )
        return host_verbs.remove(config, refs, "--yes" in flags)
    if verb in UNAVAILABLE:
        return host_verbs.unavailable(verb)
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
            # coordinator closes the session.
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
