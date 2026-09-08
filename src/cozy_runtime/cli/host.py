"""Runtime host facts, development placement, and serving.

`doctor` reports THIS host's facts and derives no fit verdict (cozy-runtime-cli.md): device,
driver and CUDA as tri-state facts and per-source credential presence.
cozy-creator renders these unmodified; nothing here is a policy read.

"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import canonical, hostfacts
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.exits import Exit

GIB = 1 << 30


def doctor(config: RuntimeConfig, opts: Options) -> Result:
    facts = hostfacts.measure()
    present = config.credentials.present()
    rows: tuple[tuple[str, str], ...] = (
        ("device", f"{facts.gpu_name or facts.state('gpu_name')}"),
        (
            "vram",
            f"{facts.vram_total_bytes / GIB:.2f} GiB"
            if facts.vram_total_bytes
            else facts.state("vram_total_bytes"),
        ),
        ("compute_capability", f"sm{facts.gpu_sm}" if facts.gpu_sm else facts.state("gpu_sm")),
        ("driver", facts.driver_version or facts.state("driver_version")),
        ("cuda", facts.cuda_version or "unreadable"),
        ("host_ram", f"{facts.host_ram_total_bytes / GIB:.1f} GiB"),
        ("vcpus", str(facts.vcpu_count)),
        ("platform", facts.platform),
        ("credentials", ", ".join(present) if present else "0 sources configured"),
        ("unreadable", ", ".join(facts.unreadable) if facts.unreadable else "none"),
    )
    document = {
        "device": {
            "name": facts.gpu_name,
            "state": facts.state("gpu_name"),
            "sm": facts.gpu_sm,
            "vram_total_bytes": facts.vram_total_bytes,
            "driver_version": facts.driver_version,
            "cuda_version": facts.cuda_version,
        },
        "host": {
            "platform": facts.platform,
            "ram_total_bytes": facts.host_ram_total_bytes,
            "vcpu_count": facts.vcpu_count,
        },
        "credentials": list(present),
        "unreadable": list(facts.unreadable),
    }
    return Result(
        rows=rows,
        document=document if opts.json else None,
        next=("cozy-runtime hostfacts <function> — this host's capacity facts for a function",),
    )


#: `serve` flag -> the worker flag it translates to. The set is CLOSED and it is the
#: whole launch surface a RecordOwner needs: who this worker is, what release it serves,
#: which devices it may touch, and how long a cancel gets before it stops being cooperative.
#: cl-010 measured the cost of the smaller set — the product could not use `serve` at all
#: and launched `internal.worker.session:main` directly, which is a second process entry
#: into the same loop and exactly what this verb exists to prevent.
_SERVE_FLAGS: dict[str, str] = {
    "--worker-id": "--worker-id",
    "--release-id": "--release-id",
    "--git-commit": "--git-commit",
    "--devices": "--devices",
    # The DECLARED grant roots (#506b): every directory tree a `file://` grant may address
    # on this machine. A launcher knows its own root and nothing can discover it for one,
    # which is exactly what puts it in this closed set.
    "--grant-root": "--grant-root",
    "--install-root": "--install-root",
    "--environment-python": "--environment-python",
    "--tensorfs-root": "--tensorfs-root",
    "--artifact-cache": "--artifact-cache",
    "--accelerator-backend": "--accelerator-backend",
    "--executor-uid": "--executor-uid",
    "--executor-gid": "--executor-gid",
    "--threads": "--threads",
    "--grace": "--grace",
    # The REMOTE leg (cl-014 item 1): a non-loopback listener requires both, and the
    # worker refuses plaintext off-loopback. The bearer owner token arrives via
    # COZY_BOOTSTRAP_CREDENTIAL and is checked at Claim (#445).
    "--tls-cert": "--tls-cert",
    "--tls-key": "--tls-key",
}

_DEVELOPMENT_FLAGS = (
    "--development-project",
    "--development-package",
    "--development-release",
    "--development-source-digest",
)


def serve(config: RuntimeConfig, args: list[str]) -> int:
    """The process entry into cr-007's worker — a TRANSLATION, never a second loop.

    `--socket`/`--out` are the launcher-issued path grants; `_SERVE_FLAGS` are
    the launch facts a launcher must state because nothing can discover them for it —
    identity, release, device envelope, budgets and the grace clock. Everything else —
    directives, grants, config — arrives over the protocol, and always will.

    `--socket` is the local binding this worker SERVES on (#436: the owner dials the
    worker): a Unix socket path, or `host:port` loopback on win32 (`host:0` binds an
    ephemeral port and publishes the real address to `<root>/control.addr`).
    """
    from cozy_runtime.internal.worker.session import serve_worker

    values: dict[str, str] = {}
    rest = list(args)
    while rest:
        item = rest.pop(0)
        name, eq, inline = item.partition("=")
        if name not in ("--socket", "--out", *_DEVELOPMENT_FLAGS) and name not in _SERVE_FLAGS:
            raise CliError(
                "usage",
                f"unknown flag {item!r}",
                "serve takes --socket <path> [--out <dir>] "
                + " ".join(f"[{f} <value>]" for f in sorted(_SERVE_FLAGS))
                + " [internal development flags]; directives, grants and "
                "config arrive over the protocol",
                Exit.usage,
            )
        if not eq:
            if not rest:
                raise CliError("usage", f"{name} needs a value", f"pass {name} <value>", Exit.usage)
            inline = rest.pop(0)
        values[name] = inline
    socket = values.get("--socket", "")
    if not socket:
        raise CliError(
            "usage",
            "serve requires --socket",
            "cozy-runtime serve --socket <path> — the local binding this worker serves on",
            Exit.usage,
        )
    root = Path(values.get("--out") or (config.cozy_home / "worker"))
    argv = ["--listen", socket if ":" in socket else f"unix:{socket}", "--root", str(root)]
    development = {flag: values.get(flag, "") for flag in _DEVELOPMENT_FLAGS}
    if any(development.values()) and not all(development.values()):
        missing = [flag for flag, value in development.items() if not value]
        raise CliError(
            "usage",
            "development launch is missing " + ", ".join(missing),
            "Creator supplies project, package, release, and source digest together",
            Exit.usage,
        )
    artifact_flags = ("--tensorfs-root", "--artifact-cache")
    supplied_artifacts = [flag for flag in artifact_flags if flag in values]
    forbidden_development = [
        flag for flag in ("--install-root", "--environment-python") if flag in values
    ]
    if all(development.values()) and forbidden_development:
        raise CliError(
            "usage",
            f"--development-project cannot be combined with {', '.join(forbidden_development)}",
            "a source checkout has no published Environment; Creator supplies the local "
            "--tensorfs-root and the install's --artifact-cache",
            Exit.usage,
        )
    if all(development.values()) and len(supplied_artifacts) != len(artifact_flags):
        missing = [flag for flag in artifact_flags if flag not in values]
        raise CliError(
            "usage",
            "development launch is missing " + " and ".join(missing),
            "the launcher names where the checkout's derived documents live (its install's "
            "artifact cache) and the local TensorFS store; nothing is written under COZY_HOME",
            Exit.usage,
        )
    if all(development.values()):
        from cozy_runtime.author import ConformanceError
        from cozy_runtime.internal import local, package_interface
        from cozy_runtime.internal.discovery import discover

        project = Path(development["--development-project"]).expanduser().resolve()
        try:
            found = discover(project)
            staged = local.stage_development_placement(
                config.cozy_home,
                package_interface.build(found),
                package=development["--development-package"],
                release=development["--development-release"],
                source_digest=development["--development-source-digest"],
                artifact_cache=Path(values["--artifact-cache"]).expanduser().resolve(),
                tensorfs_root=Path(values["--tensorfs-root"]).expanduser().resolve(),
                found=found,
            )
        except ConformanceError as exc:
            raise CliError(
                "validation",
                str(exc),
                "repair the package surface before launching its local development worker",
                Exit.validation,
            ) from exc
        except RuntimeError as exc:
            code, separator, detail = str(exc).partition(": ")
            raise CliError(
                code if code.startswith("development_") else "structural",
                detail if separator else str(exc),
                "repair the local source selection or download its exact model release",
                Exit.structural,
            ) from exc
        argv += [
            "--tensorfs-root",
            str(staged.tensorfs_root),
            "--artifact-cache",
            str(staged.artifact_cache),
            "--development-project",
            str(project),
            "--development-package",
            development["--development-package"],
            "--development-release",
            development["--development-release"],
            "--development-source-digest",
            development["--development-source-digest"],
        ]
        os.chdir(project)
    elif len(supplied_artifacts) != len(artifact_flags):
        missing = [flag for flag in artifact_flags if flag not in values]
        raise CliError(
            "usage",
            "ordinary serve requires " + " and ".join(missing),
            "pass the exact provisioned --tensorfs-root, or "
            "use Creator's explicit development launch for a local source checkout",
            Exit.usage,
        )
    for flag, worker_flag in _SERVE_FLAGS.items():
        if all(development.values()) and flag in {
            "--tensorfs-root",
            "--artifact-cache",
            "--install-root",
            "--environment-python",
        }:
            continue
        if flag in values:
            argv += [worker_flag, values[flag]]
    # THE CONFIG THIS VERB ALREADY READ. `session.main` reads it itself because it is a
    # process entrypoint too; calling that from here was a second read and refused.
    return serve_worker(config, argv)


def development_placement(config: RuntimeConfig, args: list[str], opts: Options) -> Result:
    """Internal Creator plumbing for one honest source-checkout PlacementSet."""

    from cozy_runtime.author import ConformanceError
    from cozy_runtime.internal import local, static_interface
    from cozy_runtime.internal.discovery import discover
    from cozy_runtime.internal.worker.session import SEMVER
    from cozy_runtime.protocol import documents
    from cozy_runtime.protocol import worker_pb2 as pb

    values: dict[str, str] = {}
    rest = list(args)
    required = {"--package", "--release", "--source-digest", "--tensorfs-root"}
    allowed = set(required) | {"--environment-python"}
    while rest:
        flag = rest.pop(0)
        if flag not in allowed or not rest:
            raise CliError(
                "usage",
                f"development-placement received invalid argument {flag!r}",
                "pass --package <org/name> --release <semver> --source-digest sha256:<hex> "
                "--tensorfs-root <TensorFS Store root>",
                Exit.usage,
            )
        values[flag] = rest.pop(0)
    missing = sorted(required - set(values))
    if missing:
        raise CliError(
            "usage",
            "development-placement is missing " + ", ".join(missing),
            "Creator supplies all three exact development facts",
            Exit.usage,
        )
    package = values["--package"]
    release = values["--release"]
    source_digest = values["--source-digest"]
    if (
        package.count("/") != 1
        or not all(package.split("/"))
        or SEMVER.fullmatch(release) is None
        or len(source_digest) != 71
        or not source_digest.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in source_digest[7:])
    ):
        raise CliError(
            "validation",
            "development package, release, or source digest is invalid",
            "use org/name, SemVer, and sha256:<64 lowercase hex>",
            Exit.validation,
        )
    project = Path(opts.dir).expanduser().resolve()
    try:
        interface = static_interface.build(
            project,
            environment_python=Path(values["--environment-python"])
            if "--environment-python" in values
            else None,
        )
        entries = interface["entrypoints"]
        assert isinstance(entries, list)
        # Job-only and weightless placement use only the closed interface. Serving
        # models still follow the existing construction path until their preparation
        # is moved behind the admitted worker boundary.
        modeled = any(isinstance(entry, dict) and entry.get("models") for entry in entries)
        found = discover(project) if modeled else None
        development = local.development_placement_documents(
            interface,
            package=package,
            release=release,
            source_digest=source_digest,
            found=found,
            cozy_home=config.cozy_home,
            tensorfs_root=Path(values["--tensorfs-root"]).expanduser().resolve(),
        )
    except ConformanceError as exc:
        raise CliError(
            "validation", str(exc), "repair the package surface", Exit.validation
        ) from exc
    except RuntimeError as exc:
        code, separator, detail = str(exc).partition(": ")
        raise CliError(
            code if code.startswith("development_") else "structural",
            detail if separator else str(exc),
            "repair the local source selection or download its exact model release",
            Exit.structural,
        ) from exc
    placement_bytes = documents.canonical_bytes(pb.PlacementSet(placements=[development.placement]))
    placement_digest = "sha256:" + hashlib.sha256(placement_bytes).hexdigest()
    interface_digest = "sha256:" + hashlib.sha256(development.package_interface).hexdigest()
    document = {
        "package": package,
        "release": release,
        "source_digest": source_digest,
        "placement_set": {
            "canonical_bytes_base64": base64.b64encode(placement_bytes).decode(),
            "digest": placement_digest,
            "length": len(placement_bytes),
        },
        "package_interface": {
            "canonical_bytes_base64": base64.b64encode(development.package_interface).decode(),
            "digest": interface_digest,
            "length": len(development.package_interface),
        },
    }
    return Result(
        rows=(("package", package), ("release", release), ("source_digest", source_digest)),
        document=document,
    )


def bindings(opts: Options) -> Result:
    """`cozy-runtime bindings` — the RESOLVED binding records, from the one resolver.

    cl-010's finding: no runtime verb emitted the resolved record, so cozy-creator grew a
    second narrow reader of `package.toml`'s selection grammar
    (`internal/launch/bindings.go`) to get at the same three keys. Two readers of one closed
    grammar is exactly the duplication the CLI reference forbids, and the fix belongs here
    because this side owns the resolution. With this verb that file deletes — the deletion
    is cl-003's to make.

    It resolves and prints; it constructs nothing, touches no device and loads no weights.
    """
    from cozy_runtime.cli import run as run_verb

    records, weightless_plans = run_verb.resolved_bindings(opts)
    rows = tuple(
        (
            str(record["model_binding_path"]),
            f"{record['ref']}"
            + (f" lane={record['lane']}" if record["lane"] else "")
            + f" ({record['source']})",
        )
        for record in records
    )
    return Result(
        rows=rows,
        document=(
            {
                "bindings": records,
                **({"weightless_plans": weightless_plans} if weightless_plans else {}),
            }
            if opts.json
            else None
        ),
        lines=()
        if records
        else ("note: this project declares no model binding — nothing to resolve",),
    )


#: The float output dtypes a device probe qualifies against. Every producer-mined case the
#: vendored vector documents carry is at one of these, so a narrower set would report a
#: route unqualified for a dtype nothing measured rather than for one nothing serves.
_PROBE_DTYPES = ("bfloat16", "float16", "float32")


def capability(config: RuntimeConfig, args: list[str], opts: Options) -> Result:
    """`cozy-runtime capability` — the OBSERVED (encoding, device) qualification of THIS card.

    The runtime is the only component holding the device, so it is the only one that may say
    what a card is qualified to execute. TensorFS owns the admission RULE and consumes this
    document through `tfs capability --records`; Tensorhub joins on the answer. Nothing here
    crosses the worker wire — this is an artifact, produced once per (card, runtime build).

    `--out` writes the EXACT canonical bytes TensorFS parses. Stdout is for people: the
    `--json` renderer appends a newline, and a canonical document with a trailing byte is
    not the canonical encoding of its own content.
    """
    from cozy_runtime import __version__
    from cozy_runtime.internal.encoding import launch_providers, measure_device
    from cozy_runtime.internal.probe import observed_capability_records, qualify

    out = ""
    rest = list(args)
    while rest:
        flag = rest.pop(0)
        if flag != "--out" or not rest:
            raise CliError(
                "usage",
                f"capability received invalid argument {flag!r}",
                "cozy-runtime [--json] capability [--out <path>] — it probes this host's "
                "device and takes no other argument",
                Exit.usage,
            )
        out = rest.pop(0)

    try:
        import torch
    except ImportError as exc:
        raise CliError(
            "structural",
            "capability needs torch to run the qualification suite",
            "install the accelerator extra: this verb MEASURES a device and cannot infer one",
            Exit.structural,
        ) from exc

    device = measure_device(torch)
    try:
        _, result = qualify(torch, launch_providers(), device, _PROBE_DTYPES, release=__version__)
    except Exception as exc:
        raise CliError(
            "structural",
            f"the qualification suite did not complete: {type(exc).__name__}: {exc}",
            "an unqualified device reports NOTHING rather than a partial table; repair the "
            "backend or run this on the card you mean to qualify",
            Exit.structural,
        ) from exc
    document = observed_capability_records(result)
    raw = canonical.write(document)
    if out:
        Path(out).expanduser().write_bytes(raw)

    records = document["records"]
    rows: tuple[tuple[str, str], ...] = (
        ("device", device.predicate),
        ("card", f"{device.name} #{device.index}"),
        ("runtime", __version__),
        ("suite_ms", str(result.get("suite_ms", 0))),
        ("observations", str(len(result.get("observations", ())))),
        ("qualified", str(len(records))),
        *tuple((r["encoding"][:19], r["implementation"]) for r in records),
    )
    return Result(
        rows=rows,
        canonical_json=raw,
        lines=(f"wrote {len(raw)} canonical bytes to {out}",) if out else (),
        next=(
            "tfs capability <encoding> "
            f"{device.predicate} --records <path> --json  # TensorFS owns the rule",
        ),
    )
