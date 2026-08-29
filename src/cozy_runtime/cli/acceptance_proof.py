"""Installed CLI for deterministic fixtures and ordinary-attempt observations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from cozy_runtime.cli.io import CliError, Options, emit_error
from cozy_runtime.internal import acceptance
from cozy_runtime.internal.exits import Exit


def main(argv: Sequence[str] | None = None) -> int:
    opts = Options()
    temporary: Path | None = None
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 3 or args[0] not in {"fixture", "observe"}:
            raise CliError(
                "usage",
                "cozy-acceptance-proof requires fixture|observe REQUEST.json OUTPUT.json",
                "cozy-acceptance-proof fixture REQUEST.json PAYLOAD.json\n"
                "cozy-acceptance-proof observe REQUEST.json OBSERVATION.json",
                Exit.usage,
            )
        request = _input(args[1])
        output = _output(args[2])
        if request == output:
            raise _invalid("request and output name the same path")
        raw = request.read_bytes()
        body = (
            acceptance.fixture_request(raw)
            if args[0] == "fixture"
            else acceptance.execution_observation(raw)
        )
        temporary = output.parent / f".acceptance-{os.getpid()}-{uuid.uuid4().hex}"
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise CliError(
                "acceptance_output_exists",
                str(output),
                "choose a fresh output path; evidence is immutable",
                Exit.conflict,
            ) from exc
        _fsync(output.parent)
        temporary.unlink()
        temporary = None
        print(
            json.dumps(
                {
                    "digest": "sha256:" + hashlib.sha256(body).hexdigest(),
                    "length": len(body),
                    "output": str(output),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return int(Exit.ok)
    except CliError as exc:
        emit_error(exc, opts)
        return int(exc.code)
    except acceptance.AcceptanceRefusal as exc:
        emit_error(
            CliError(
                exc.code,
                exc.detail,
                "repair the exact qualified closure or bank a fresh ordinary attempt",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except OSError as exc:
        emit_error(
            CliError(
                "acceptance_io",
                f"{exc.filename or 'acceptance path'}: {exc.strerror or type(exc).__name__}",
                "repair proof-seat paths and permissions",
                Exit.structural,
            ),
            opts,
        )
        return int(Exit.structural)
    except Exception as exc:
        emit_error(
            CliError(
                "internal",
                f"unexpected acceptance proof fault: {type(exc).__name__}: {exc}",
                "report this Runtime build and exact request",
                Exit.internal,
            ),
            opts,
        )
        return int(Exit.internal)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _input(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        path = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise _invalid(f"request is unreadable: {type(exc).__name__}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise _invalid("request is not a regular file")
    return path


def _output(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.parent.resolve(strict=True) / path.name
    if not path.name or os.path.lexists(path):
        raise CliError(
            "acceptance_output_exists",
            str(path),
            "choose a fresh output path",
            Exit.conflict,
        )
    return path


def _invalid(detail: str) -> CliError:
    return CliError(
        "acceptance_request_invalid",
        detail,
        "provide one closed local-path acceptance request",
        Exit.validation,
    )


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
