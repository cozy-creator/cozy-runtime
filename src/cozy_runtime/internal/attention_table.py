"""The attention kernel table (`attention_table.json`): per hardware class (`arch.CLASSES`), layout,
attention level and head dimension, one ranking of kernels per token bucket of `min(Lq, Lk)`,
fastest first. It is generated from our measurements (`scripts/attention_table.py`), checked in
and validated strictly at import, so a broken table cannot ship. A site walks its own cell and
each level above it (`attention.walk`); a pin bypasses the table.

    python -m cozy_runtime.internal.attention_table check FILE
"""

from __future__ import annotations

import bisect
import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import msgspec

from cozy_runtime.internal import arch

SCHEMA = "cozy.attention-table/1"
#: A hand-made table's `report`.
TRANSCRIBED = "transcribed"
#: The lowest precision any cell lists (Paul, 11-11): a 4-bit kernel serves only by pin, until a model opts in with a
#: measured quality pass and Paul's eye check, which a change to this rule records.
LOWEST = 8
PATH = Path(__file__).with_name("attention_table.json")
#: The attention levels a cell is filed at: the precision its site tolerates, 16 the exact one.
LEVELS = (4, 8, 16)
LAYOUTS = ("dense", "sparse")

type Layout = Literal["dense", "sparse"]


class Kernel(Protocol):
    """What the validator reads of a Runtime kernel (`attention.Candidate`)."""

    @property
    def level(self) -> int: ...
    @property
    def sparse(self) -> bool: ...
    @property
    def families(self) -> frozenset[str]: ...


class Card(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The reference card that measured a class, and its runs: provenance, never a key."""

    card: str
    sm: int
    runs: tuple[int, ...]


class Cell(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    klass: str = msgspec.field(name="class")
    layout: Layout
    level: Literal[4, 8, 16]
    #: an exact head dimension, or None for every one no other cell of its key names
    head_dim: int | None
    #: per token bucket, the kernels fastest first
    order: tuple[tuple[str, ...], ...]


class Document(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema: str
    runtime: str
    generated: str
    report: str
    tokens: tuple[int, ...]
    classes: dict[str, Card]
    kernels: dict[str, str]
    cells: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class Table:
    document: Document
    #: `<runtime>/<sha256[:12]>` of the file: which table a run chose by
    identity: str
    _cells: dict[tuple[str, str, int, int | None], Cell] = field(repr=False)

    @property
    def tokens(self) -> tuple[int, ...]:
        return self.document.tokens

    def buckets(self) -> range:
        return range(len(self.document.tokens) + 1)

    def bucket(self, tokens: int) -> int:
        """The bucket a call of `min(Lq, Lk)` TOKENS falls in."""
        return bisect.bisect_right(self.document.tokens, tokens)

    def band(self, bucket: int) -> str:
        """A bucket's canonical label: `<2048`, `2048-12287`, `>=65536`."""
        edges = self.document.tokens
        if bucket == 0:
            return f"<{edges[0]}"
        if bucket == len(edges):
            return f">={edges[-1]}"
        return f"{edges[bucket - 1]}-{edges[bucket] - 1}"

    def ranking(
        self, klass: str, layout: Layout, level: int, head_dim: int, bucket: int
    ) -> tuple[str, ...]:
        """A cell's kernels for one bucket: the cell naming HEAD_DIM, else the catch-all; none
        for a class the table has no cells for ("": no tensor-core family)."""
        cell = self._cells.get((klass, layout, level, head_dim)) or self._cells.get(
            (klass, layout, level, None)
        )
        return cell.order[bucket] if cell is not None else ()


def _problems(document: Document, kernels: Mapping[str, Kernel]) -> list[str]:
    found: list[str] = []
    if document.schema != SCHEMA:
        found.append(f"schema {document.schema!r} is not {SCHEMA!r}")
    edges = document.tokens
    if not edges or any(e <= 0 for e in edges) or list(edges) != sorted(set(edges)):
        found.append(f"tokens {list(edges)} are not strictly ascending positive edges")
    if set(document.classes) != set(arch.CLASSES):
        found.append(f"classes {sorted(document.classes)} are not {sorted(arch.CLASSES)}")
    seen: set[tuple[str, str, int, int | None]] = set()
    for cell in document.cells:
        key = (cell.klass, cell.layout, cell.level, cell.head_dim)
        where = f"{cell.klass} {cell.layout} {cell.level} d{cell.head_dim or '*'}"
        if key in seen:
            found.append(f"{where}: listed twice")
        seen.add(key)
        if cell.klass not in arch.CLASSES:
            found.append(f"{where}: no such class")
        if cell.head_dim is not None and cell.head_dim <= 0:
            found.append(f"{where}: head dimension {cell.head_dim}")
        if len(cell.order) != len(edges) + 1:
            found.append(f"{where}: {len(cell.order)} rankings for {len(edges) + 1} buckets")
        family = cell.klass.partition("+")[0].partition("/")[0]
        for ranking in cell.order:
            if not ranking or len(set(ranking)) != len(ranking):
                found.append(f"{where}: ranking {list(ranking)} is empty or repeats a kernel")
            for name in ranking:
                kernel = kernels.get(name)
                if kernel is None:
                    found.append(f"{where}: {name} is not a Runtime kernel")
                elif kernel.level < cell.level:
                    found.append(f"{where}: {name} is {kernel.level}-bit")
                elif kernel.sparse and cell.layout == "dense":
                    found.append(f"{where}: {name} is sparse")
                elif kernel.families and family not in kernel.families:
                    found.append(f"{where}: {name} does not run on {family}")
                elif kernel.level < LOWEST:
                    found.append(f"{where}: {name} is {kernel.level}-bit; one serves only by pin")
    for klass in arch.CLASSES:
        for layout in LAYOUTS:
            for level in LEVELS:
                if (klass, layout, level, None) not in seen:
                    found.append(f"{klass} {layout} {level}: no catch-all (head_dim null) cell")
    return sorted(set(found))


def load(kernels: Mapping[str, Kernel], path: Path = PATH) -> Table:
    """The table at PATH, validated against the Runtime's KERNELS, or ValueError naming every
    problem: a strict decode, `len(tokens) + 1` non-empty rankings per cell, every class with a
    catch-all cell per layout and level, and every kernel a Runtime one of at least its cell's
    precision, on its class's family, sparse only in sparse cells."""
    data = path.read_bytes()
    try:
        document = msgspec.json.decode(data, type=Document, strict=True)
    except msgspec.ValidationError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    problems = _problems(document, kernels)
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    cells: dict[tuple[str, str, int, int | None], Cell] = {
        (c.klass, c.layout, c.level, c.head_dim): c for c in document.cells
    }
    digest = hashlib.sha256(data).hexdigest()[:12]
    return Table(document, f"{document.runtime}/{digest}", cells)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "check":
        print("usage: python -m cozy_runtime.internal.attention_table check FILE", file=sys.stderr)
        return 2
    from cozy_runtime.internal import attention

    try:
        table = load(attention.BY_NAME, Path(argv[1]))
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"ok: {table.identity}, {len(table.document.cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
