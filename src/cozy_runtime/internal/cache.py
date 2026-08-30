"""Runtime-local SQLite rows.

These tables replace private JSON documents whose type was already fixed by the directory
and caller that opened them.  They are a cache and local index, not a wire or artifact
contract.  There are deliberately no migrations: this product is prelaunch, so an unknown
schema is refused instead of carrying compatibility readers forever.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1
FILENAME = "runtime.sqlite3"

_SCHEMA = (
    """CREATE TABLE artifacts (
    ref TEXT PRIMARY KEY,
    store_root TEXT NOT NULL,
    config TEXT NOT NULL,
    snapshots_json TEXT NOT NULL,
    lane TEXT NOT NULL,
    stored_bytes INTEGER NOT NULL,
    tensors INTEGER NOT NULL,
    variant TEXT NOT NULL,
    custody TEXT NOT NULL,
    vram_floor_bytes INTEGER NOT NULL,
    extra_key_markers_json TEXT NOT NULL,
    stamps_json TEXT NOT NULL
) STRICT""",
    """CREATE TABLE plan_facts (
    plan_digest TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stored_bytes INTEGER NOT NULL,
    resident_bytes INTEGER NOT NULL,
    scratch_bytes INTEGER NOT NULL,
    fill_ms INTEGER NOT NULL,
    decode_ms INTEGER NOT NULL,
    step_ms REAL NOT NULL,
    reference_deviation_rel_l2 REAL NOT NULL,
    evidence TEXT NOT NULL,
    at_ns INTEGER NOT NULL,
    PRIMARY KEY (plan_digest, run_id)
) STRICT""",
    """CREATE INDEX plan_facts_by_plan_time
    ON plan_facts (plan_digest, at_ns DESC, run_id DESC)""",
    """CREATE TABLE probe_results (
    cache_key TEXT PRIMARY KEY,
    result_json TEXT NOT NULL
) STRICT""",
)


class CacheError(RuntimeError):
    """The private cache has an unsupported or unreadable schema."""


@contextmanager
def connect(root: Path | str) -> Iterator[sqlite3.Connection]:
    """Open the one private database rooted at ``root`` and enforce its exact schema."""

    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / FILENAME, timeout=5.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout = 5000")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # Another initializer may have won while BEGIN waited. Re-read only after
                # holding the writer lock, then either create the whole schema or accept it.
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version == 0:
                    existing = connection.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%'
                        """
                    ).fetchall()
                    if existing:
                        raise CacheError(
                            f"{directory / FILENAME}: unversioned cache has existing objects"
                        )
                    for statement in _SCHEMA:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION:
                    raise CacheError(
                        f"{directory / FILENAME}: schema {version}, expected {SCHEMA_VERSION}"
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        elif version != SCHEMA_VERSION:
            raise CacheError(
                f"{directory / FILENAME}: schema {version}, expected {SCHEMA_VERSION}; "
                "delete this rebuildable cache rather than guessing at an old layout"
            )
        yield connection
    except sqlite3.Error as exc:
        raise CacheError(f"{directory / FILENAME}: {exc}") from exc
    finally:
        connection.close()
