"""`doctor`, `list`, `rm`, `pull`, `ingest` — the host and artifact verbs.

`doctor` reports THIS host's facts and derives no fit verdict (cozy-runtime-cli.md): device,
driver and CUDA as tri-state facts, CAS reachability, and per-source credential presence.
cozy-creator renders these unmodified; nothing here is a policy read.

`list`/`rm` speak checkpoint-and-lane vocabulary over the local artifact index — one
artifact, one size, object mechanics stay in diagnostics. `rm` without `--yes` is a typed
exit-7 refusal, never a prompt, and an absent ref is an idempotent 0 that says so.

`pull`/`ingest` are the two verbs this build cannot honestly implement: both are thin
invocations of TensorFS's ONE fetch+ingest implementation, and that border (the safetensors
carrier decode and the semantic converters, tfs-002/tfs-003) is not in this runtime's
dependency yet. They refuse typed and NAME the owner rather than growing a second converter
here, which is exactly the duplication the CLI reference forbids.
"""

from __future__ import annotations

import os
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import artifacts, hostfacts
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.exits import Exit

GIB = 1 << 30


def doctor(config: RuntimeConfig, opts: Options) -> Result:
    facts = hostfacts.measure()
    store = config.cozy_home
    installed = artifacts.rows(config.cozy_home)
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
        ("cas", f"{store} ({'present' if store.is_dir() else 'absent'})"),
        ("artifacts", f"{len(installed)} installed"),
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
        "cas": {"root": str(store), "present": store.is_dir(), "artifacts": len(installed)},
        "credentials": list(present),
        "unreadable": list(facts.unreadable),
    }
    return Result(
        rows=rows,
        document=document if opts.json else None,
        next=("cozy-runtime fit <function> — this host's verdicts for a function",),
    )


def list_artifacts(config: RuntimeConfig, prefix: str, opts: Options) -> Result:
    installed = artifacts.rows(config.cozy_home, prefix)
    rows: list[tuple[str, str]] = []
    total = 0
    for row in installed:
        total += row.bytes
        rows.append(
            (
                row.ref,
                f"{row.lane}  {row.bytes / GIB:.3f} GiB  {len(row.snapshots)} component"
                f"{'s' if len(row.snapshots) != 1 else ''}  {row.custody}",
            )
        )
    # Empty states are definitive and aggregates ride the listing.
    rows.append(
        (
            "total",
            f"{len(installed)} artifact{'s' if len(installed) != 1 else ''}, {total / GIB:.3f} GiB",
        )
    )
    return Result(
        rows=tuple(rows),
        document=[row.view() for row in installed] if opts.json else None,
        next=("cozy-runtime rm <ref> --yes — remove one from the local CAS",),
    )


def remove(config: RuntimeConfig, refs: list[str], yes: bool) -> Result:
    if not refs:
        raise CliError(
            "usage", "rm needs at least one ref", "cozy-runtime rm <ref> … --yes", Exit.usage
        )
    if not yes:
        raise CliError(
            "confirm",
            f"rm would remove {len(refs)} artifact(s) from the local CAS: {', '.join(refs)}",
            "rerun with --yes; a destructive act is a typed refusal, never a prompt",
            Exit.confirm,
            next=(f"cozy-runtime rm {' '.join(refs)} --yes",),
        )
    rows: list[tuple[str, str]] = []
    for ref in refs:
        rows.append((ref, "removed" if artifacts.remove(config.cozy_home, ref) else "not present"))
    return Result(rows=tuple(rows), next=("cozy-runtime list",))


def unavailable(verb: str) -> Result:
    """The typed refusal `pull`/`ingest` are, until TensorFS's border is a dependency."""
    raise CliError(
        "unavailable",
        f"{verb!r} invokes TensorFS's one fetch+ingest implementation, which this runtime "
        "build does not depend on yet",
        "the carrier decode, classification and semantic converters are tfs-002/tfs-003's "
        "border; a second converter here would be the duplication the CLI reference "
        "forbids. Artifacts already in the local CAS are usable — `cozy-runtime list`",
        Exit.unavailable,
        next=("cozy-runtime list",),
    )


#: `serve` flag -> the worker flag it translates to. The set is CLOSED and it is the
#: whole launch surface a RecordOwner needs: who this worker is, what release it serves,
#: which devices it may touch, and how long a cancel gets before it stops being cooperative.
#: cl-010 measured the cost of the smaller set — the product could not use `serve` at all
#: and launched `internal.worker.session:main` directly, which is a second process entry
#: into the same loop and exactly what this verb exists to prevent.
_SERVE_FLAGS: dict[str, str] = {
    "--worker-id": "--worker-id",
    "--instance-id": "--instance-id",
    "--release-id": "--release-id",
    "--git-commit": "--git-commit",
    "--devices": "--devices",
    # The DECLARED grant roots (#506b): every directory tree a `file://` grant may address
    # on this machine. A launcher knows its own root and nothing can discover it for one,
    # which is exactly what puts it in this closed set.
    "--grant-root": "--grant-root",
    "--environment-root": "--environment-root",
    "--artifact-store": "--artifact-store",
    "--artifact-object-set": "--artifact-object-set",
    "--artifact-cache": "--artifact-cache",
    "--control-runtime-digest": "--control-runtime-digest",
    "--accelerator-backend": "--accelerator-backend",
    "--executor-uid": "--executor-uid",
    "--executor-gid": "--executor-gid",
    "--threads": "--threads",
    "--grace": "--grace",
    "--host-budget": "--host-budget",
    "--pinned-budget": "--pinned-budget",
    # The REMOTE leg (cl-014 item 1): a non-loopback listener requires both, and the
    # worker refuses plaintext off-loopback. The bearer owner token arrives via
    # COZY_BOOTSTRAP_CREDENTIAL and is checked at Claim (#445).
    "--tls-cert": "--tls-cert",
    "--tls-key": "--tls-key",
}

#: Flags that carry no value.
_SERVE_SWITCHES = ("--no-warm",)
_WEIGHTLESS_PACKAGE = "--weightless-package"


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
    switches: list[str] = []
    rest = list(args)
    while rest:
        item = rest.pop(0)
        name, eq, inline = item.partition("=")
        if name in _SERVE_SWITCHES:
            switches.append(name)
            continue
        if name not in ("--socket", "--out", _WEIGHTLESS_PACKAGE) and name not in _SERVE_FLAGS:
            raise CliError(
                "usage",
                f"unknown flag {item!r}",
                "serve takes --socket <path> [--out <dir>] "
                + " ".join(f"[{f} <value>]" for f in sorted(_SERVE_FLAGS))
                + f" [{_WEIGHTLESS_PACKAGE} <dir>] [--no-warm]; directives, grants and "
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
    weightless_package = values.get(_WEIGHTLESS_PACKAGE, "")
    artifact_flags = ("--artifact-store", "--artifact-object-set")
    supplied_artifacts = [flag for flag in artifact_flags if flag in values]
    if weightless_package and supplied_artifacts:
        raise CliError(
            "usage",
            f"{_WEIGHTLESS_PACKAGE} cannot be combined with {', '.join(supplied_artifacts)}",
            "choose one launch mode: the explicit local weightless package, or the exact "
            "provisioned artifact store and object set",
            Exit.usage,
        )
    if weightless_package:
        from cozy_runtime.author import ConformanceError
        from cozy_runtime.internal import descriptor, local
        from cozy_runtime.internal.discovery import discover

        package = Path(weightless_package).expanduser().resolve()
        try:
            found = discover(package)
            staged = local.stage_weightless_release(config.cozy_home, descriptor.build(found))
        except ConformanceError as exc:
            raise CliError(
                "validation",
                str(exc),
                "repair the package surface before launching its local weightless worker",
                Exit.validation,
            ) from exc
        except RuntimeError as exc:
            raise CliError(
                "structural",
                str(exc),
                "use the ordinary provisioned launch mode for any model-bearing package",
                Exit.structural,
            ) from exc
        argv += [
            "--artifact-store",
            str(staged.artifact_store),
            "--artifact-object-set",
            str(staged.artifact_object_set),
        ]
        os.chdir(package)
    elif len(supplied_artifacts) != len(artifact_flags):
        missing = [flag for flag in artifact_flags if flag not in values]
        raise CliError(
            "usage",
            "ordinary serve requires " + " and ".join(missing),
            "pass the exact provisioned --artifact-store and --artifact-object-set, or "
            f"choose {_WEIGHTLESS_PACKAGE} <dir> explicitly",
            Exit.usage,
        )
    for flag, worker_flag in _SERVE_FLAGS.items():
        if flag in values:
            argv += [worker_flag, values[flag]]
    argv += switches
    # THE CONFIG THIS VERB ALREADY READ. `session.main` reads it itself because it is a
    # process entrypoint too; calling that from here was a second read and refused.
    return serve_worker(config, argv)


def bindings(config: RuntimeConfig, opts: Options) -> Result:
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

    records, weightless_plans = run_verb.resolved_bindings(config, opts)
    rows = tuple(
        (
            str(record["model_binding_path"]),
            f"{record['ref']}"
            + (f" lane={record['lane']}" if record["lane"] else "")
            + f" ({record['source']}) "
            + (
                "installed: " + ", ".join(record["components"])
                if record["installed"]
                else "NOT installed"
            ),
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
