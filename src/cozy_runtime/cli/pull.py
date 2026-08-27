"""`cozy-runtime pull` — the verb (cr-016).

Thin by design, and the CLI reference's own words for it: *"thin wiring over TensorFS's one
fetch/ingest implementation — tokens as parameters, no fetch logic here"*. The wiring is
four things and none of them touches a byte: resolve the flags, hand the credential over as
a parameter, drive `internal/pull.py`, and write the local artifact index row that makes the
pulled checkpoint a `ref` the rest of the CLI already understands.

The exit matrix is the shared one — a private ref with no credential is exit 5 NAMING the
credential to set, and it is never a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import artifacts, fill
from cozy_runtime.internal import pull as transport
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.exits import Exit

FLAGS = ("--hub", "--lane", "--store", "--timeout", "--as", "--streams")
BOOL_FLAGS = ("--dry-run",)

#: Refusal code -> exit. The codes are `internal/pull.py`'s closed registry, so a new code
#: cannot appear here without appearing there, and neither can appear alone.
EXITS: dict[str, Exit] = {
    "hub_unreachable": Exit.unavailable,
    "hub_refused": Exit.unavailable,
    "ref_not_found": Exit.not_found,
    "credential_required": Exit.credential,
    "source_not_allowed": Exit.validation,
    "transfer_failed": Exit.unavailable,
    "deadline_exceeded": Exit.deadline,
    "artifact_incomplete": Exit.validation,
    "border_refused": Exit.validation,
}
assert set(EXITS) == set(transport.REFUSALS), "the pull exit table and its refusal registry drifted"


def _duration(text: str) -> float:
    """`<n>ms|s|m|h`, the shared grammar. There is no default — an absent `--timeout` means
    no deadline, not a number this file picked."""
    for suffix, scale in (("ms", 0.001), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
        if text.endswith(suffix):
            try:
                return float(text[: -len(suffix)]) * scale
            except ValueError:
                break
    raise CliError(
        "usage",
        f"{text!r} is not a duration",
        "durations are <n>ms, <n>s, <n>m or <n>h",
        Exit.usage,
    )


def _parse(args: list[str]) -> tuple[str, dict[str, str], set[str]]:
    flags: dict[str, str] = {}
    on: set[str] = set()
    rest: list[str] = []
    items = list(args)
    while items:
        item = items.pop(0)
        if item in BOOL_FLAGS:
            on.add(item)
        elif item.split("=")[0] in FLAGS:
            name, _, inline = item.partition("=")
            if not inline:
                if not items:
                    raise CliError(
                        "usage", f"{name} needs a value", f"pass {name} <value>", Exit.usage
                    )
                inline = items.pop(0)
            flags[name] = inline
        elif item.startswith("-") and item != "-":
            raise CliError(
                "usage",
                f"unknown flag {item!r}",
                "pull takes --hub, --lane, --store, --as, --streams, --timeout, --dry-run",
                Exit.usage,
            )
        else:
            rest.append(item)
    if len(rest) != 1:
        raise CliError(
            "usage",
            f"pull takes exactly one ref, got {len(rest)}",
            "cozy-runtime pull <org/repo[@release|@sha256:...]> --hub <url>",
            Exit.usage,
        )
    return rest[0], flags, on


def run(args: list[str], opts: Options, config: RuntimeConfig) -> Result:
    ref, flags, on = _parse(args)
    hub = flags.get("--hub", "").strip()
    if not hub:
        raise CliError(
            "usage",
            "--hub is required: this build reads no hub address from the environment",
            "pass --hub https://<tensorhub> — one config authority per process, and it does "
            "not carry a hub default",
            Exit.usage,
        )
    store_root = (
        Path(flags["--store"]).expanduser() if "--store" in flags else config.cozy_home / "store"
    )
    timeout = _duration(flags["--timeout"]) if "--timeout" in flags else None
    streams = _streams(flags.get("--streams", ""))

    try:
        tensorfs = fill.tensorfs_module()
    except fill.FillRefusal as refusal:
        raise CliError(
            "unavailable",
            f"the TensorFS reader is unavailable: {refusal}",
            "install the TensorFS Python facade in this Runtime environment, then run the "
            "same pull again",
            Exit.unavailable,
        ) from refusal
    store = _store(tensorfs, store_root)

    try:
        if "--dry-run" in on:
            return _preview(tensorfs, store, hub, ref, flags, config, timeout, opts)
        outcome = transport.pull(
            tensorfs,
            store,
            hub,
            ref,
            lane=flags.get("--lane", ""),
            token=config.credentials.tensorhub or "",
            timeout=timeout,
            streams=streams,
        )
    except transport.PullRefusal as refusal:
        raise _refuse(refusal, config) from refusal

    facts = _header_facts(tensorfs, store, outcome.snapshot)
    row = artifacts.Artifact(
        ref=_installed_ref(flags.get("--as", ""), ref, outcome.release),
        store=str(store.root),
        config="",
        # THE COMPONENTS THE PULLED HEADER DECLARES, not one invented key. The fill plane
        # indexes `snapshots` by real component name and refuses `destination_absent` for a
        # name that is not there, so a row saying `{"checkpoint": …}` describes bytes that
        # arrived perfectly and cannot be served. An N-ary checkpoint carries every
        # component under one digest, which is what this map says and why it is read from
        # the header rather than assumed to be one.
        snapshots={name: "sha256:" + outcome.snapshot for name in facts.components},
        lane=flags.get("--lane", "") or "plain",
        bytes=facts.stored_bytes,
        tensors=facts.tensors,
        # THE STAMP PLANE'S READER HALF (#567d). The hub verified these at publish against
        # the border's own derivations and they are the third coordinate of the lane
        # contract this pull just matched — carrying structure and encoding down while
        # dropping the stamps is what left every stamped model class with `task=None`.
        stamps=dict(outcome.stamps),
    )
    try:
        artifacts.install(config.cozy_home, row)
    except artifacts.ArtifactRefError as bad:
        # The one place a HUMAN's string reaches the index: `--as`. The index states the
        # refusal and this renders it — one message, two layers, no second wording.
        raise CliError("usage", str(bad), bad.remedy, bad.exit_code) from bad
    return Result(
        document={**outcome.document(), "components": list(facts.components)},
        rows=(
            ("ref", row.ref),
            ("snapshot", "sha256:" + outcome.snapshot),
            ("store", str(store.root)),
            ("components", f"{', '.join(facts.components)} — {facts.tensors} tensors"),
            (
                "stamps",
                ", ".join(f"{k}={'+'.join(v)}" for k, v in sorted(row.stamps.items()))
                or "none — a stamped model class will refuse this artifact",
            ),
            (
                "objects",
                f"{outcome.declared} declared, {outcome.held} held, {outcome.fetched} fetched"
                f" on {outcome.streams} stream(s)",
            ),
            ("bytes", f"{outcome.bytes_moved} moved, {outcome.bytes_held} already resident"),
        ),
        next=(f"cozy-runtime run <function> --model {row.ref}",),
    )


def _streams(text: str) -> int:
    """`--streams <n>`: how many objects move at once. Absent takes the transport leg's
    measured default; zero or a non-number is a usage refusal, never a silent 1."""
    if not text:
        return transport.STREAMS
    try:
        count = int(text)
    except ValueError:
        count = 0
    if count < 1:
        raise CliError(
            "usage",
            f"{text!r} is not a stream count",
            f"--streams takes a positive integer; the default is {transport.STREAMS}, "
            "measured against a saturated link",
            Exit.usage,
        )
    return count


def _installed_ref(named: str, ref: str, release: str) -> str:
    """The ref the local index files this artifact under.

    `--as` wins verbatim — it is the operator saying what to call it. Otherwise the pulled
    ref keeps its RELEASE and loses only its digest pin: `org/repo@v2` installed as
    `org/repo` would be a row for a different artifact under a name that already means
    something else. A digest-pinned ref names no release, so the hub's resolved one is used
    when the hub had one to give.
    """
    if named:
        return named
    base, _, pin = ref.partition("@")
    if pin and not pin.startswith("sha256:"):
        return ref
    return f"{base}@{release}" if release else base


@dataclass(frozen=True, slots=True)
class _Facts:
    components: tuple[str, ...]
    tensors: int
    stored_bytes: int


def _header_facts(tensorfs: Any, store: Any, snapshot: str) -> _Facts:
    """What the PULLED snapshot's own header says it holds.

    Read here rather than taken from the transport because the transport counts OBJECTS and
    the index row is about TENSORS: `bytes_total` includes the manifest and the encoding
    spec documents, and no object knows which component it belongs to. This is a header
    read — no tensor byte is touched — and it is the same parse the fill plane does, so a
    row this writes is a row that plane can resolve.
    """
    try:
        header_bytes = store.snapshot("sha256:" + snapshot)["header"]
        closure = tensorfs.Closure()
        for spec in tensorfs.parse_header(header_bytes)["encodings"]:
            closure.insert(store.document(str(spec["sha256"])))
        header = tensorfs.parse_header(header_bytes, closure)
    except Exception as exc:
        raise CliError(
            "validation",
            f"sha256:{snapshot}: the pulled snapshot's header did not parse — "
            f"{type(exc).__name__}: {exc}",
            "the objects landed and verified; the header this build reads is not the header "
            "the hub published. Re-pull, and if it repeats the catalog's header is suspect",
            Exit.validation,
        ) from exc
    widths = tensorfs.DTYPES
    tensors = 0
    stored = 0
    for table in header["components"].values():
        tensors += len(table)
        for entry in table.values():
            for part in entry["parts"].values():
                nbytes = int(widths[str(part["dtype"])])
                for extent in part["shape"]:
                    nbytes *= int(extent)
                stored += nbytes
    return _Facts(tuple(sorted(header["components"])), tensors, stored)


def _preview(
    tensorfs: Any,
    store: Any,
    hub: str,
    ref: str,
    flags: dict[str, str],
    config: RuntimeConfig,
    timeout: float | None,
    opts: Options,
) -> Result:
    """`--dry-run`: the plan without the bytes. Every hub read happens, no object moves.

    It is the plan the real pull would use — the same `fetch_plan` call — so what it says
    about held/wanted is what will happen, not a second estimate of it.
    """
    deadline = transport.Deadline.after(timeout)
    token = config.credentials.tensorhub or ""
    try:
        at = transport.resolve(hub, ref, deadline, lane=flags.get("--lane", ""), token=token)
        declared = transport.reads(hub, at, deadline, token=token)
        doc = transport.manifest(hub, at, deadline)
        plan = tensorfs.fetch_plan(
            store,
            "dry-" + at.snapshot[:12],
            at.snapshot,
            len(doc),
            [(r.object_id, r.length) for r in declared],
        )
    except transport.PullRefusal as refusal:
        raise _refuse(refusal, config) from refusal
    return Result(
        document={
            "ref": ref,
            "snapshot": "sha256:" + at.snapshot,
            "store": str(store.root),
            "held": len(plan["held"]),
            "wanted": len(plan["wanted"]),
            "bytes_to_move": int(plan["wanted_bytes"]),
            "bytes_held": int(plan["held_bytes"]),
            "dry_run": True,
        },
        rows=(
            ("snapshot", "sha256:" + at.snapshot),
            ("would fetch", f"{len(plan['wanted'])} object(s), {plan['wanted_bytes']} B"),
            ("already held", f"{len(plan['held'])} object(s), {plan['held_bytes']} B"),
        ),
        next=(f"cozy-runtime pull {ref} --hub {hub}",),
    )


def _store(tensorfs: Any, root: Path) -> Any:
    """Open the local store, creating it on first pull. A worker's first pull is the act
    that gives it a CAS; refusing because the directory is not there yet would be a chore."""
    try:
        return transport.open_store(tensorfs, root)
    except transport.PullRefusal as exc:
        raise CliError(
            "validation",
            exc.message,
            "point --store at a v2 store root, or at an empty directory to create one",
            Exit.validation,
        ) from exc


def _refuse(refusal: transport.PullRefusal, config: RuntimeConfig) -> CliError:
    remedy = refusal.remedy
    if refusal.code == "credential_required" and not config.credentials.tensorhub:
        # NAME the credential, never prompt (cr-016's owed red arm). The env NAME is the
        # config module's, so the two cannot drift.
        remedy = (
            "set TENSORHUB_TOKEN in this process's environment and re-run — cozy-runtime "
            "never prompts for a credential. " + remedy
        )
    return CliError(refusal.code, refusal.message, remedy, EXITS[refusal.code])
