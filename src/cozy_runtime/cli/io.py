"""Shared CLI output conventions (cozy-runtime-cli.md). One renderer, every verb."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from cozy_runtime.internal.exits import Exit

ELIDE_AT = 120


@dataclass(frozen=True, slots=True)
class Options:
    json: bool = False
    full: bool = False
    fields: tuple[str, ...] = ()
    dir: str = "."


@dataclass(frozen=True, slots=True)
class Result:
    rows: tuple[tuple[str, str], ...] = ()
    commands: tuple[str, ...] = ()
    next: tuple[str, ...] = ()


class CliError(Exception):
    """Structured refusal: `error(<name>): <message>` plus the remedy verbatim."""

    def __init__(
        self,
        name: str,
        message: str,
        remedy: str,
        code: Exit = Exit.usage,
        next: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.name = name
        self.message = message
        self.remedy = remedy
        self.code = code
        self.next = next


def _elide(value: str, full: bool) -> str:
    if full or len(value) <= ELIDE_AT:
        return value
    extra = len(value) - ELIDE_AT
    return f"{value[:ELIDE_AT]}… (+{extra}B, --full)"


def emit(result: Result, opts: Options) -> None:
    rows = result.rows
    if opts.fields:
        rows = tuple(r for r in rows if r[0] in opts.fields)
    nxt = result.next[:2]
    if opts.json:
        doc: dict[str, object] = {k: v for k, v in rows}
        if result.commands:
            doc["commands"] = list(result.commands)
        doc["next"] = list(nxt)
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return
    width = max((len(k) for k, _ in rows), default=0)
    for key, value in rows:
        print(f"{key + ':':<{width + 1}} {_elide(value, opts.full)}")
    for i, command in enumerate(result.commands, 1):
        print(f"  {i}  {command}")
    for line in nxt:
        print(f"next: {line}")


def emit_error(err: CliError, opts: Options) -> None:
    if opts.json:
        doc: dict[str, object] = {
            "error": {
                "code": int(err.code),
                "name": err.name,
                "message": err.message,
                "remedy": err.remedy,
            },
            "next": list(err.next[:2]),
        }
        print(json.dumps(doc, indent=2, ensure_ascii=False), file=sys.stderr)
        return
    print(f"error({err.name}): {err.message}", file=sys.stderr)
    print(f"remedy: {err.remedy}", file=sys.stderr)
    for line in err.next[:2]:
        print(f"next: {line}", file=sys.stderr)
