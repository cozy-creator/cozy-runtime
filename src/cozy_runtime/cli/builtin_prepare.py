"""Prepare only Runtime's fixed carrier against the invoking SDK's immutable bytes."""

from __future__ import annotations

from pathlib import Path

from cozy_runtime.author import ConformanceError
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.cli.package_prepare import _values
from cozy_runtime.internal import builtin_environment, numerical_environment, package_environment
from cozy_runtime.internal.exits import Exit


def run(
    args: list[str],
    _opts: Options,
    *,
    capture: bool = False,
    dependency_cache: Path | None = None,
) -> Result:
    if not args or args[0] != "operations":
        raise CliError(
            "usage",
            "builtin-prepare requires operations",
            "builtin-prepare operations --wheel <carrier> --out <root>",
        )
    values = _values(args[1:], {"--wheel", "--out"}, set())
    wheel_path = Path(values["--wheel"][0]).absolute()
    root = Path(values["--out"][0]).absolute()
    try:
        prepared = builtin_environment.prepare(
            wheel_path, root, capture=capture, dependency_cache=dependency_cache
        )
        return Result(document=prepared.document())
    except (
        ConformanceError,
        package_environment.EnvironmentRefusal,
        numerical_environment.NumericalEnvironmentRefusal,
    ) as exc:
        raise CliError(
            str(getattr(exc, "code", "builtin_prepare")),
            str(exc),
            "use the exact Runtime-owned carrier and an immutable installed SDK",
            Exit.structural,
        ) from exc
