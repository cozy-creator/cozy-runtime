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
from cozy_runtime.cli import pull as pull_verb
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
    "fit",
    "list",
    "rm",
    "new",
    "pull",
    "version",
)
#: `ingest` PARSES and refuses typed: it is a thin invocation of TensorFS's one INGEST
#: implementation — the carrier decode, classification and semantic converters — and that
#: half of the border is not a dependency of this build yet (tfs-003). It is a verb, not an
#: unknown word, and the refusal names the owner. `pull` is BUILT: the fetch half landed
#: with tfs-003 row 220, and it invokes TensorFS's `fetch_plan`/`fetch_admit` rather than
#: carrying any transfer-to-store logic of its own.
UNAVAILABLE = ("ingest",)

HELP = """cozy-runtime — everything a single package venv can do alone

usage: cozy-runtime [--json] [--full] [--fields a,b] [--dir <path>] <verb> [args]

verbs:
  run <fn> [<primary>] [k=v …]   one-shot inference: payload in, outputs to disk, exit
  job <fn> [k=v …]               run one @app.job to completion, same attempt kernel
  serve --socket <path> <mode>   speak the worker protocol for a RecordOwner
  describe [<fn>]                the PackageDescriptor, without loading weights
  bindings                       the RESOLVED binding records this project selects
  doctor                         this host's device/driver/CAS/credential facts
  fit [<fn>] [k=v …]             this host's fit verdicts (exact with a payload)
  pull <ref> --hub <url>         fetch a hub checkpoint into the local CAS
  list [<ref-prefix>]            CAS-resident checkpoints, one artifact one size
  rm <ref> … --yes               remove artifacts from the local CAS
  new <name>                     scaffold a minimal valid package skeleton
  version                        tag, commit, and the two independent semvers

run/job flags:
  --model [path=]<ref>  override a model binding (repeatable)
  --input <ref>=<dir>   grant one materialized input tree (job only, repeatable)
  --org <name>          publishing org; a job's writes land in <org>/_job-<request-id>
  --out <dir>           output directory (default ./outputs)
  --seed <n>            deterministic RNG
  --in <file>           whole payload as JSON, key=value merged on top
  key=@/absolute/file   stage a local file only at a declared Asset field
  --offline             CAS-only; a miss is exit 8, never a fetch
  --timeout <dur>       attempt deadline (<n>ms|s|m|h)
  --strict-keys         a local bind's unknown extra keys refuse instead of confessing
  --stream              typed deltas as NDJSON

pull flags:
  --hub <url>           the tensorhub to resolve and read from (required)
  --lane <k=v,...>      narrow a release with several lanes
  --store <dir>         local CAS root (default $COZY_HOME/store)
  --as <ref>            index the pulled checkpoint under this ref
  --timeout <dur>       a deadline for the whole pull; absent means none
  --dry-run             the plan — what would move — without moving it

serve modes (exactly one):
  --weightless-package <dir>                 stage a local model-free package
  --artifact-store <dir>                       serve selected model Manifests

global flags:
  --json           the full typed result as one JSON document
  --full           widen output, lift truncation
  --fields a,b,c   pick columns
  --dir <path>     package project root (default: cwd)
  --descriptor <path>
                   exact PackageDescriptor input for describe (no package import)
  -h, --help       this text
"""

_VALUE_FLAGS = ("--fields", "--dir", "--descriptor")
#: Verbs that own their own flag parsing: everything after the verb belongs to them, so a
#: payload term like `--in` is not eaten by the global pass.
_OWNS_ARGS = ("run", "job", "fit", "serve", "rm", "pull", "ingest")


def parse(argv: Sequence[str]) -> tuple[str | None, list[str], Options, bool]:
    """Split argv into (verb, args, options, help). Unknown flags refuse with exit 2."""
    as_json = full = want_help = False
    fields: tuple[str, ...] = ()
    directory = "."
    descriptor = ""
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
            else:
                descriptor = inline
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
    return verb, args, Options(as_json, full, fields, directory, descriptor), want_help


def dispatch(verb: str, args: list[str], opts: Options, config: RuntimeConfig) -> Result:
    if opts.descriptor and verb != "describe":
        raise CliError(
            "usage",
            "--descriptor belongs only to describe",
            "use --descriptor <exact-package-descriptor> with cozy-runtime describe",
            Exit.usage,
        )
    if verb == "describe":
        if len(args) > 1:
            raise CliError(
                "usage",
                f"describe takes at most one function name, got {len(args)}",
                "cozy-runtime [--json] [--dir <path>] [--descriptor <path>] describe [<function>]",
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
        return host_verbs.bindings(config, opts)
    if verb == "pull":
        return pull_verb.run(args, opts, config)
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
