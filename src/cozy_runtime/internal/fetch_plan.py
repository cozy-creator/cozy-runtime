"""Runtime directs what TensorFS fetches (one-request-kind §2.6); TensorFS's fetcher downloads.

Each call's owner (TensorD) hands out a fresh fetch channel on every `fetch_channel` ask; it
names only the call's selected checkpoints and ends with the call. Runtime plans in passes,
each on a fresh channel that replaces the last (U1 has no unschedule: a replaced channel drops
its tickets, so unused components stop and ranks change):

1. **Start**, before torch and the package import: every construction's components, smallest
   first, so the imports, kernel compiles and the first stage overlap the download.
2. **Load**: the components the code builds, in stage order, each region's objects in
   execution order. Components and tensors the code never builds are never asked for.
3. **PrepareRequest**: the attempt re-asserts the plan, with the stage and block order this
   process learned.

Every rank is `Next` (run order, rank); a reader that needs bytes now waits on them (`Now`).
Runtime never downloads and never retries: the fetcher owns transport retries, and a fetch
it refuses ends the run `model_fetch_failed` (`machine_disk_full` for a full disk).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import tensorfs
from tensorfs.derived import SourceCapability
from tensorfs.fetch import Fetch

from cozy_runtime.author._errors import RuntimeFailure

if TYPE_CHECKING:
    from tensorfs import ReadPlan

    from cozy_runtime.internal.executor_commands import Construction

#: One object: `(sha256 hex, length)`.
type Ref = tuple[str, int]
#: One rank of a plan: a checkpoint and its objects, in the order they are wanted.
type Group = tuple[str, list[Ref]]

#: Objects per channel frame: a plan's frames stay small however large the model.
_BATCH = 1024
#: The planning window; any value plans the same objects.
_WINDOW = 4 << 20


def _items(plan: ReadPlan) -> Iterator[tuple[str, Ref]]:
    """Each object `plan` reads, with the `<component>/<tensor>` it reads into."""
    for item in plan.items():
        if item["source"] == "object":
            ref = (item["object"].removeprefix("sha256:"), item["object_length"])
            yield item["what"].rpartition("#")[0], ref


def component_objects(
    header: bytes, component: str, keys: Sequence[str], start: int = 0
) -> list[Ref]:
    """The objects of one component's tensors `keys[start:]` (`keys`: all, header order)."""
    plan = tensorfs.plan(header, [(component, key) for key in keys], [component], _WINDOW)
    wanted = {f"{component}/{key}" for key in keys[start:]}
    return list(dict.fromkeys(ref for what, ref in _items(plan) if what in wanted))


def region_objects(plan: ReadPlan, regions: Sequence[Sequence[str]]) -> list[list[Ref]]:
    """Each region's objects (`regions` lists `<component>/<tensor>` ids), in read order."""
    index = {name: number for number, names in enumerate(regions) for name in names}
    found: list[dict[Ref, None]] = [{} for _ in regions]
    for what, ref in _items(plan):
        found[index[what]][ref] = None
    return [list(refs) for refs in found]


def at_start(constructions: Sequence[Construction], header: Callable[[str], bytes]) -> list[Group]:
    """The Start pass: each construction's declared components (every one when none is
    declared), smallest first; inside one, header order. No class is known yet, and the
    small encoders a first stage uses then land first."""
    groups: list[Group] = []
    for row in constructions:
        data = header(row.manifest)
        table = tensorfs.parse_header(data)["components"]
        names = [name for name in row.components or table if name in table]
        planned = [component_objects(data, name, list(table[name])) for name in names]
        planned.sort(key=lambda refs: sum(length for _, length in refs))
        groups += [(row.manifest, refs) for refs in planned]
    return groups


def in_definition_order(
    cls: type, use: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """`component_use` with its methods in the order the class defines them: authors write
    stages in pipeline order (encode, denoise, decode), which a sorted listing loses."""
    named = dict.fromkeys(
        name for klass in reversed(cls.__mro__) for name in vars(klass) if name in use
    )
    return {name: use[name] for name in [*named, *use]}


def stage_components(stages: Iterable[str], use: Mapping[str, Sequence[str]]) -> list[str]:
    """Components in the order `stages` first use them."""
    return list(dict.fromkeys(name for stage in stages for name in use.get(stage, ())))


def failure(manifest: str, refs: Sequence[Ref], exc: BaseException) -> RuntimeFailure:
    """A fetch the fetcher refused, as the run's reason."""
    code = getattr(exc, "code", "")
    detail = getattr(exc, "detail", str(exc))
    first = f"sha256:{refs[0][0]}" if refs else "its structure"
    reason = "machine_disk_full" if code == "CAPACITY_EXHAUSTED" else "model_fetch_failed"
    return RuntimeFailure(f"{first} of {manifest[:23]}: {code} {detail}"[:900], code=reason)


def _wire(refs: Sequence[Ref]) -> list[Any]:
    return [{"sha256": sha, "length": length} for sha, length in refs]


def _code(exc: BaseException) -> str:
    return str(getattr(exc, "code", ""))


class Fetches:
    """This executor's fetch planner over its current channel (one at a time)."""

    def __init__(self, ask: Callable[[], int], progress: Callable[[str, float], None]) -> None:
        #: a fresh channel from the call's owner (-1: it gives none)
        self._ask = ask
        self._progress = progress
        self._lock = threading.Lock()
        self._channel: Fetch | None = None
        #: this command's plans, by what planned them (one per model of a many-model load)
        self._plans: dict[str, list[Group]] = {}
        #: the next rank free for look-ahead
        self._rank = 0
        #: checkpoints the owner holds whole here (outside every grant): nothing to fetch
        self._whole: set[str] = set()

    @property
    def groups(self) -> list[Group]:
        """The plan on the channel, in rank order."""
        return [group for plan in list(self._plans.values()) for group in plan]

    def begin(self) -> None:
        """A new pass (Start, Load, PrepareRequest): the plans it makes replace the last's."""
        self._plans = {}

    def schedule(self, groups: Sequence[Group], key: str = "") -> None:
        """Plan `groups` as `key`'s part of this pass, rank by position, on a fresh channel;
        then drop the channel it replaces, and with it what the last pass wanted."""
        self._plans[key] = list(groups)
        planned = self.groups
        fd = self._ask()
        fresh = Fetch.from_fd(fd) if fd >= 0 else None
        if fresh is not None:
            for rank, (manifest, refs) in enumerate(planned):
                self._send(fresh, manifest, refs, rank)
        with self._lock:
            old, self._channel, self._rank = self._channel, fresh, len(planned)
        if old is not None:
            old.close()

    def available(self) -> bool:
        """Whether this call has a channel (asking for one when it holds none)."""
        return self._current(ask=True) is not None

    def ahead(self, manifest: str, refs: Sequence[Ref]) -> None:
        """Want `refs` after everything already planned (a raw source's look-ahead)."""
        channel = self._current(ask=True)
        if channel is None or not refs:
            return
        with self._lock:
            rank, self._rank = self._rank, self._rank + 1
        self._send(channel, manifest, refs, rank)

    def present(self, manifest: str, refs: Sequence[Ref]) -> bool:
        """Whether every one of `refs` is here (no channel: the owner placed them all)."""
        channel = self._current()
        if channel is None or manifest in self._whole or not refs:
            return True
        try:
            return all(channel.present(manifest, _wire(refs)))
        except (tensorfs.errors.Refusal, OSError) as exc:
            return self._unfetchable(manifest, exc)

    def wait(self, manifest: str, refs: Sequence[Ref]) -> None:
        """Block until `refs` are here, raised to Now. A refusal fails the run."""
        channel = self._current()
        if channel is None or manifest in self._whole or not refs:
            return
        try:
            if not all(channel.present(manifest, _wire(refs))):
                self._progress(*self._landed(channel))
                channel.wait(manifest, _wire(refs))
        except (tensorfs.errors.Refusal, OSError) as exc:
            if not self._unfetchable(manifest, exc):
                raise failure(manifest, refs, exc) from None

    def end(self) -> None:
        """The attempt ended: its interest goes with its channel."""
        self._plans = {}
        with self._lock:
            old, self._channel, self._rank = self._channel, None, 0
        if old is not None:
            old.close()

    def _current(self, ask: bool = False) -> Fetch | None:
        with self._lock:
            channel = self._channel
        if channel is None and ask:
            self.schedule((), "ask")
            with self._lock:
                channel = self._channel
        return channel

    def _send(self, channel: Fetch, manifest: str, refs: Sequence[Ref], rank: int) -> None:
        if manifest in self._whole:
            return
        try:
            for start in range(0, len(refs), _BATCH):
                channel.schedule(manifest, _wire(refs[start : start + _BATCH]), rank)
        except (tensorfs.errors.Refusal, OSError) as exc:
            if not self._unfetchable(manifest, exc):
                raise

    def _unfetchable(self, manifest: str, exc: BaseException) -> bool:
        """True when nothing of `manifest` is this channel's to fetch: held whole here
        (outside the grant), or the owner ended the call and with it the channel."""
        if _code(exc) == "SOURCE_NOT_ALLOWED":
            self._whole.add(manifest)
            return True
        return isinstance(exc, OSError) or "fetch channel closed" in str(exc)

    def _landed(self, channel: Fetch) -> tuple[str, float]:
        """`waiting for weights (done/total)` over the planned bytes."""
        total = done = 0
        for manifest, refs in self.groups:
            if manifest in self._whole or not refs:
                continue
            here = channel.present(manifest, _wire(refs))
            total += sum(length for _, length in refs)
            done += sum(length for (_, length), yes in zip(refs, here, strict=True) if yes)
        mib = 1 << 20
        line = f"waiting for weights ({done // mib}/{total // mib} MiB)"
        return line, done / total if total else 0.0


def source(
    fetches: Fetches, opened: SourceCapability, header: bytes, declared: Sequence[str]
) -> SourceCapability:
    """A `Weights[M]` source with look-ahead: its declared components are wanted in header
    order when it opens, and a read of an undeclared component wants the rest of it."""
    manifest = opened.manifest
    table = tensorfs.parse_header(header)["components"]
    for name in declared:
        if name in table:
            fetches.ahead(manifest, component_objects(header, name, list(table[name])))
    started = set(declared)

    def read_part(component: str, key: str, role: str, offset: int, into: object) -> None:
        if component not in started and key in table.get(component, {}):
            started.add(component)
            keys = list(table[component])
            fetches.ahead(manifest, component_objects(header, component, keys, keys.index(key)))
        opened.read_part_into(component, key, role, offset, into)

    return SourceCapability(
        manifest,
        opened.length,
        lambda components, configs: opened.inspect(components=components, configs=configs),
        opened.close,
        read_part,
        opened.check,
    )
