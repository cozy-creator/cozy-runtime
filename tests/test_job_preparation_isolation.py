"""A real job keeps its process and code while another package is prepared."""

from __future__ import annotations

import email.parser
import functools
import shutil
import tempfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

from conftest import image_python
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import _development, _run, _wheel_rows

SOURCE = """from pathlib import Path
import tempfile
import time
import msgspec
from cozy_runtime.author import App, Context
app=App()
class Request(msgspec.Struct):
    entered: str = ""
    gate: str = ""
    temporary: str = ""
class Result(msgspec.Struct):
    value: int
@app.job
def JOB(ctx: Context, payload: Request) -> Result:
    if payload.temporary:
        Path(payload.temporary).write_text(tempfile.gettempdir())
    if payload.entered:
        Path(payload.entered).write_text("entered")
        while not Path(payload.gate).exists():
            ctx.raise_if_cancelled()
            time.sleep(0.02)
    return Result(7)
"""


def package(
    root: Path, label: str, *, runtime: str = ""
) -> tuple[Path, pb.PrepareLocalPackageRequest]:
    """A local package admitting the machine SDK, or requiring an explicit public Runtime."""
    project = root / ("project_" + label)
    project.mkdir()
    name, module = "prepare-" + label, "prepare_" + label
    requirement = f"cozy-runtime=={runtime}" if runtime else "cozy-runtime>=0.16.8,<1"
    (project / (module + ".py")).write_text(SOURCE.replace("JOB", label))
    (project / "pyproject.toml").write_text(
        f'[project]\nname="{name}"\nversion="1.0.0"\n'
        f'requires-python=">=3.12,<3.13"\ndependencies=["{requirement}"]\n'
        f'[project.entry-points."cozy.application"]\ndefault="{module}:app"\n'
        '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
        f'[tool.hatch.build.targets.wheel]\nonly-include=["{module}.py"]\n'
    )
    environment = root / "environments"
    wheels = environment / ".stage" / label / "wheels"
    wheels.mkdir(parents=True)
    uv = shutil.which("uv")
    assert uv is not None
    _run(uv, "build", "--wheel", "--out-dir", str(wheels), str(project))
    (wheel,) = wheels.glob("*.whl")
    runtime_wheel = next((image_python().parents[2] / "wheels").glob("cozy_runtime-*.whl"))
    runtime_copy = wheels / runtime_wheel.name
    shutil.copyfile(runtime_wheel, runtime_copy)
    return environment, pb.PrepareLocalPackageRequest(
        operation_id=label,
        package=_development("local/" + name, "1.0.0"),
        files=_wheel_rows([wheel] if runtime else [wheel, runtime_copy]),
        dependency_requirements=runtime_lock(runtime),
        install_root=str(environment),
    )


@functools.cache
def runtime_lock(runtime: str = "") -> bytes:
    """A hashed lock of this checkout's Runtime dependencies, or of released `runtime`: the
    same for every package that names it, so it is compiled once per session."""
    image = image_python()
    runtime_wheel = next((image.parents[2] / "wheels").glob("cozy_runtime-*.whl"))
    with zipfile.ZipFile(runtime_wheel) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        parsed = email.parser.BytesParser().parsebytes(archive.read(metadata))
    dependencies = [Requirement(raw) for raw in parsed.get_all("Requires-Dist", [])]
    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cozy-runtime-lock.") as raw:
        requirements = Path(raw) / "requirements.in"
        requirements.write_text(
            f"cozy-runtime=={runtime}\n"
            if runtime
            else "\n".join(
                str(row)
                for row in dependencies
                if row.marker is None or row.marker.evaluate({"extra": ""})
            )
            + "\n"
        )
        locked = Path(raw) / "requirements.txt"
        _run(
            uv,
            "pip",
            "compile",
            "--python",
            str(image),
            "--generate-hashes",
            "--no-header",
            "--no-annotate",
            "--output-file",
            str(locked),
            str(requirements),
        )
        return locked.read_bytes()
