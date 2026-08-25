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

from pathlib import Path

from cozy_runtime.cli import artifacts
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import hostfacts
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
            f"{len(installed)} artifact{'s' if len(installed) != 1 else ''}, "
            f"{total / GIB:.3f} GiB",
        )
    )
    return Result(
        rows=tuple(rows),
        document=[row.document() for row in installed] if opts.json else None,
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


def serve(config: RuntimeConfig, args: list[str]) -> int:
    """The process entry into cr-007's supervisor — a TRANSLATION, never a second loop.

    `--socket`/`--cas`/`--out` are the coordinator-issued path grants and the only flags:
    directives, grants and config arrive over the protocol. Everything below this call is
    the supervisor `serve` has always been.
    """
    from cozy_runtime.internal.worker.session import main as supervisor_main

    values: dict[str, str] = {}
    rest = list(args)
    while rest:
        item = rest.pop(0)
        name, eq, inline = item.partition("=")
        if name not in ("--socket", "--cas", "--out"):
            raise CliError(
                "usage",
                f"unknown flag {item!r}",
                "serve takes --socket <path> [--cas <dir>] [--out <dir>]; everything else "
                "arrives over the protocol",
                Exit.usage,
            )
        if not eq:
            if not rest:
                raise CliError(
                    "usage", f"{name} needs a value", f"pass {name} <path>", Exit.usage
                )
            inline = rest.pop(0)
        values[name] = inline
    socket = values.get("--socket", "")
    if not socket:
        raise CliError(
            "usage",
            "serve requires --socket",
            "cozy-runtime serve --socket <path> — the coordinator's Unix socket",
            Exit.usage,
        )
    root = Path(values.get("--out") or (config.cozy_home / "worker"))
    return supervisor_main(
        [
            "--hub",
            socket if ":" in socket else f"unix:{socket}",
            "--root",
            str(root),
        ]
    )


def store_root(config: RuntimeConfig) -> Path:
    return config.cozy_home
