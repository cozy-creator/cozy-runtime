"""Prepare only Runtime's fixed carrier against the invoking SDK's immutable bytes."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.cli.package_prepare import _values
from cozy_runtime.internal import builtin_operations, numerical_environment, package_environment
from cozy_runtime.internal.exits import Exit


def run(args: list[str], _opts: Options) -> Result:
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
        distribution = importlib.metadata.distribution("cozy-runtime")
        version = distribution.version
        builtin_operations.validate_carrier(wheel_path, version)
        records = numerical_environment._installed_records({"cozy-runtime": version})
        implementation = "sha256:" + records["cozy-runtime"]["record_digest"]
        base = package_environment.observe_base(Path(sys.executable))
        wheel, verified = package_environment.measure_wheel(wheel_path, "project", base)
        identity = canonical_json.digest(
            {
                "builtin": "operations",
                "wheel": wheel.digest,
                "interface": builtin_operations.digest(),
                "runtime_implementation": implementation,
            }
        )
        source = package_environment.Input(
            identity,
            wheel.digest,
            (wheel,),
            (package_environment.BlobGrant(wheel.digest, wheel.length, wheel_path, verified),),
        )
        installed = package_environment.materialize(
            source,
            root,
            python=Path(sys.executable),
            base=base,
            dependency_python=Path(sys.executable),
        )
        builtin_operations.validate_installed(
            installed.site_packages / f"cozy_runtime_operations-{version}.dist-info"
        )
        actual = package_environment.observe_base(installed.python)
        body = {
            "environment_python": str(installed.python),
            "environment_digest": identity,
            "content_digest": installed.content_digest,
            "receipt_digest": installed.receipt_digest,
            "closure": "\n".join(name + "==" + selected for name, selected in actual.distributions),
            "runtime_version": version,
            "implementation_digest": implementation,
            "package_interface": canonical_json.decode(builtin_operations.document()),
        }
        return Result(document=body)
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
