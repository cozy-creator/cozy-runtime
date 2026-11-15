"""Generate typed operation references from the dependency the project actually locked."""

from pathlib import Path

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.cli.io import CliError, Options, Result
from cozy_runtime.internal import call_references, package_interface, static_interface


def run(distribution: str, destination: str, opts: Options) -> Result:
    try:
        body = (
            package_interface.read_bytes(Path(opts.package_interface).read_bytes())
            if opts.package_interface
            else static_interface.build_dependency(Path(opts.dir), distribution)
        )
        source = call_references.generate(package_interface.canonical_bytes(body), distribution)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    except (ConformanceError, OSError) as exc:
        raise CliError(
            "call_references", str(exc), "lock the dependency and regenerate its calls"
        ) from exc
    return Result(rows=(("calls", str(path)),), document={"calls": str(path)})
