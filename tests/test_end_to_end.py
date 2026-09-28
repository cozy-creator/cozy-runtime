"""The end-to-end suite: real processes, real builds, no mocks.

Everything below runs against `examples/marco-polo`, a real weightless package with one
entrypoint and one job, built into a wheel and installed into a venv that has nothing else.
Nothing is stubbed, patched, or doubled: a test that passes because a fake agreed with it is
worse than no test.

The two legs that SPAWN AN EXECUTOR skip where the host cannot contain one, and say which
host fact stopped them — see `_executor_containment`. They run on an ordinary developer box.

The REFUSAL leg is their mirror image and never skips: it runs WHERE CONTAINMENT IS
UNAVAILABLE, because that is the condition under test, and it asserts the process EXITS.

    uv run pytest
"""

from __future__ import annotations

import contextlib
import email.parser
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement

from conftest import authored_python, image_python
from cozy_runtime.internal import (
    package_environment,
    package_installation,
    package_interface,
    static_interface,
)
from cozy_runtime.internal.worker.acquire import Acquirer
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    prepare_local_package,
    prepare_package,
    prepare_unpublished_placement,
)
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"
TIMEOUT = 420

#: A refusal is worth nothing if the process does not then END, so the refusal leg gets its
#: own bound — a TEST bound on a process that must exit on a typed refusal, and the only
#: clock on this path: the local door itself ends a run on observed stillness, never on a
#: deadline (xs-007 row 7). It leaves a cold host room to build the package environment
#: before the executor is ever reached.
REFUSAL_TIMEOUT = 300


def _executor_containment() -> str:
    """Why this host cannot spawn an executor, or "" when it can.

    Exactly the decision `internal/proctree.create_executor_scope` makes, read rather than
    performed so the probe creates nothing. A worker may only start an executor inside a
    cgroup it owns; where the unified hierarchy is writable but the worker's OWN cgroup is
    not delegated — a GitHub-hosted runner, most locked-down CI — there is no scope to make
    and the runtime refuses by design. The alternative process-tree backend wants the rental
    container topology (root worker, distinct non-root executor uid, bounded memory.max),
    which a hosted runner does not present either.
    """
    if sys.platform != "linux":
        return f"executor containment is Linux-only; this is {sys.platform}"
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").is_file():
        return "no unified cgroup-v2 hierarchy is mounted"
    if bool(os.statvfs(root).f_flag & os.ST_RDONLY):
        return "the cgroup-v2 mount is read-only; the process-tree backend wants a rental pod"
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split("::", 1)[1] for line in entries if line.startswith("0::"))
    except (OSError, StopIteration, IndexError):
        return "this process's cgroup-v2 delegation is unreadable"
    own = root / relative.strip("/")
    if not os.access(own, os.W_OK, effective_ids=True):
        return f"this process's own cgroup {own} is not delegated, so no child scope can be made"
    return ""


NO_EXECUTOR = _executor_containment()
needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


@contextlib.contextmanager
def _containment_unavailable() -> Iterator[Callable[[], None] | None]:
    """Put a child process somewhere `create_executor_scope` cannot make a scope.

    On a host that already lacks delegation — a GitHub-hosted runner, a plain container —
    this is a no-op, because the condition under test is the host's own. On a delegated host
    it is produced honestly and with the kernel, not with a patch: a real child cgroup of
    this process's own delegation, mode 0500, so `path.mkdir()` under it gets the same EACCES
    a non-delegated hierarchy gives. `cgroup.procs` stays writable, which is how the child
    walks in; the directory does not, which is what the worker trips over.
    """
    if NO_EXECUTOR:
        yield None
        return
    relative = next(
        line.split("::", 1)[1]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    scope = (
        Path("/sys/fs/cgroup") / relative.strip("/") / f"cozy-nodelegation-{secrets.token_hex(4)}"
    )
    try:
        scope.mkdir(mode=0o700)
    except OSError as exc:  # pragma: no cover - a delegated host that will not delegate
        pytest.skip(f"cannot build a non-delegated cgroup under {scope.parent}: {exc}")
    procs = scope / "cgroup.procs"

    def enter() -> None:
        procs.write_text("0\n")

    try:
        scope.chmod(stat.S_IRUSR | stat.S_IXUSR)
        yield enter
    finally:
        scope.chmod(stat.S_IRWXU)
        with contextlib.suppress(OSError):
            scope.rmdir()


def _trailing_json(text: str) -> dict[str, Any]:
    """The structured refusal `--json` writes last, past whatever the worker narrated.

    The worker's own narration shares this stream by design (`run`'s stdout IS the result
    document), so the refusal is the trailing object, not the whole stream.
    """
    lines = text.splitlines()
    start = max((i for i, line in enumerate(lines) if line == "{"), default=-1)
    assert start >= 0, f"no structured refusal was emitted:\n{text[-4000:]}"
    body = json.loads("\n".join(lines[start:]))
    assert isinstance(body, dict)
    return body


def _run(*argv: str, cwd: Path | str = ROOT, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    if result.returncode != expect:
        pytest.fail(
            f"{argv} exited {result.returncode}, wanted {expect}\n"
            f"--- stdout ---\n{result.stdout[-6000:]}\n--- stderr ---\n{result.stderr[-6000:]}"
        )
    return result


def _write_annotated_doc(root: Path, version: str) -> None:
    package = root / "src" / "annotated_doc"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        "[project]\n"
        'name = "annotated-doc"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.12,<3.13"\n\n'
        "[tool.hatch.build.targets.wheel]\n"
        'packages = ["src/annotated_doc"]\n'
    )
    (package / "__init__.py").write_text(f"VALUE = {version!r}\n")


def _write_pure_project(
    root: Path, name: str, version: str, module: str, requires: tuple[str, ...] = ()
) -> None:
    package = root / "src" / module
    package.mkdir(parents=True)
    dependencies = ", ".join(repr(item) for item in requires)
    (root / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        "[project]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.12,<3.13"\n'
        f"dependencies = [{dependencies}]\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        f'packages = ["src/{module}"]\n'
    )
    (package / "__init__.py").write_text(f"VALUE = {version!r}\n")


def _wheel_rows(paths: list[Path]) -> list[pb.LocalPackageFile]:
    rows = [
        pb.LocalPackageFile(
            digest=hashlib.sha256(path.read_bytes()).digest(),
            filename=path.name,
            length=path.stat().st_size,
            path=str(path),
        )
        for path in paths
    ]
    rows.sort(key=lambda row: row.filename)
    return rows


def _development(package: str, release: str, installation_id: str = "") -> pb.DevelopmentPackage:
    return pb.DevelopmentPackage(
        package=package,
        release=release,
        installation_id=installation_id or package.replace("/", "-") + "-" + release,
    )


def _runtime_closure(wheel_set: Path, *extra: str, index: str = "") -> bytes:
    """Stage the image's Runtime wheel and hash-lock its dependencies, as Creator sends them."""
    uv = shutil.which("uv")
    assert uv is not None
    image = image_python()
    runtime = next((image.parent.parent.parent / "wheels").glob("cozy_runtime-*.whl"))
    shutil.copyfile(runtime, wheel_set / runtime.name)
    with zipfile.ZipFile(runtime) as archive:
        name = next(row for row in archive.namelist() if row.endswith(".dist-info/METADATA"))
        metadata = email.parser.BytesParser().parsebytes(archive.read(name))
    requested = [
        str(row)
        for row in map(Requirement, metadata.get_all("Requires-Dist", []))
        if row.marker is None or row.marker.evaluate({"extra": ""})
    ]
    source = wheel_set.parent / "requirements.in"
    source.write_text("\n".join([*requested, *extra]) + "\n")
    locked = wheel_set.parent / "requirements.txt"
    _run(
        uv,
        "pip",
        "compile",
        "--python",
        str(image),
        "--generate-hashes",
        "--no-header",
        "--no-annotate",
        *(("--index", index, "--emit-index-url") if index else ()),
        "--output-file",
        str(locked),
        str(source),
    )
    return locked.read_bytes()


@pytest.fixture(scope="session")
def installed_cli() -> Iterator[str]:
    """The PUBLIC CLI, built into a wheel and installed into a venv that has nothing else.

    This is the shape a user gets: `pip install` of the built artifact, not the source tree.
    Anything reachable only from a checkout is invisible here, which is the point.
    """
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build and install the wheel under test"
    with tempfile.TemporaryDirectory(prefix="cozy-e2e.") as raw:
        root = Path(raw)
        _run(uv, "build", "--wheel", "--out-dir", str(root / "dist"))
        _run(uv, "build", "--wheel", "--out-dir", str(root / "dist"), str(PACKAGE))
        wheels = sorted(str(path) for path in (root / "dist").glob("*.whl"))
        assert len(wheels) == 2, wheels
        _run(uv, "venv", "--python", sys.executable, str(root / "venv"))
        _run(uv, "pip", "install", "--python", str(root / "venv" / "bin" / "python"), *wheels)
        yield str(root / "venv" / "bin" / "cozy-runtime")


def test_installed_cli_describes_and_refuses(installed_cli: str) -> None:
    """describe -> exact replay -> the exit-code matrix, from `/`, no PYTHONPATH, stdin closed.

    The package interface is a PUBLICATION output, so two derivations must be byte-identical, and
    the exit codes (`internal/exits.py`) are the contract a caller actually reads. None of
    this needs a device or an executor.
    """
    cli = (installed_cli, "--dir", str(PACKAGE))
    first = _run(*cli, "--json", "describe", cwd="/")
    second = _run(*cli, "--json", "describe", cwd="/")
    assert first.stdout == second.stdout
    described = json.loads(first.stdout)
    assert described["application"] == "marco_polo_package:app"
    assert all("resources" not in job for job in described["jobs"])

    # Hardware declarations were a second, author-written placement authority. The hardcut is
    # observable from the installed public wheel: neither the value nor the decorator argument
    # survives, and a reader ignores an older Runtime's `resources` key rather than obeying it.
    _run(
        str(Path(installed_cli).with_name("python")),
        "-c",
        "import inspect; import cozy_runtime.author as a; "
        "assert not hasattr(a, 'Resources'); "
        "assert 'resources' not in inspect.signature(a.App.job).parameters; "
        "assert 'resources' not in inspect.signature(a.App.entrypoint).parameters",
        cwd="/",
    )
    _run(
        str(Path(installed_cli).with_name("python")),
        "-c",
        "import json; from cozy_runtime.internal import canonical, package_interface; "
        f"body=json.loads({first.stdout!r}); "
        "body['jobs'][0]['resources']={'gpu_count': 1}; "
        "package_interface.read_bytes(canonical.write(body))",
        cwd="/",
    )

    _run(*cli, "frobnicate", cwd="/", expect=2)


def _describe_installed(
    selected: package_installation.InstalledEnvironment, distribution: str
) -> bytes:
    command = (
        "from cozy_runtime.internal import package_interface; "
        "from cozy_runtime.internal.discovery import discover_distribution; "
        f"found=discover_distribution({distribution!r}); "
        "print(package_interface.canonical_bytes(package_interface.build(found)).decode())"
    )
    return _run(str(selected.python), "-I", "-c", command, cwd="/").stdout.encode()


def test_editable_installations_materialize_from_supplied_wheels() -> None:
    """Install exact private project wheels beside a hash-locked Runtime closure.

    Each installation ID owns its generation and job plans. A dependency edit installs a
    new generation without touching an earlier one; replaying an ID reuses its generation.
    """

    uv = shutil.which("uv")
    assert uv is not None
    python = image_python()
    observed = package_environment.observe_base(python)

    with tempfile.TemporaryDirectory(prefix="cozy-editable-install.") as raw:
        root = Path(raw)
        dependency = root / "annotated-doc-0.0.5"
        _write_annotated_doc(dependency, "0.0.5")
        artifact_cache = root / "artifacts"
        artifact_cache.mkdir()
        environment = root / "environments"
        wheel_set = environment / ".stage" / "editable-install-proof" / "wheels"
        wheel_set.mkdir(parents=True)
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(PACKAGE))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency))
        locked = _runtime_closure(wheel_set)
        plans = root / "job-plans"

        def prepare(
            installation_id: str,
        ) -> tuple[pb.PreparePackageSetResult, package_installation.InstalledEnvironment]:
            held: list[package_installation.InstalledEnvironment] = []
            result = prepare_local_package(
                pb.PrepareLocalPackageRequest(
                    operation_id="editable-install-proof",
                    package=_development("paul/marco-polo", "1.0.4", installation_id),
                    files=_wheel_rows(list(wheel_set.glob("*.whl"))),
                    dependency_requirements=locked,
                    install_root=str(environment),
                ),
                artifact_cache=artifact_cache,
                install_root=environment,
                python=python,
                describe=_describe_installed,
                verified=lambda _digest, _length, _path: None,
                base=observed,
                job_plan_root=plans,
                materialized=held.append,
            )
            return result, held[0]

        def value(installed: package_installation.InstalledEnvironment) -> str:
            program = "import annotated_doc; print(annotated_doc.VALUE)"
            return _run(str(installed.python), "-I", "-c", program, cwd="/").stdout.strip()

        first_id = "local-" + uuid.uuid4().hex
        prepared, first = prepare(first_id)
        assert not first.reused
        (placement,) = read_placement_set(prepared.placement_set)
        assert placement.installation_id == first_id
        assert placement.development.installation_id == first_id
        assert prepared.installed_package.installation_id == first_id
        _run(str(first.python), "-I", "-c", "import marco_polo_package", cwd="/")
        assert value(first) == "0.0.5"
        first_plans = {path: path.read_bytes() for path in (plans / first_id).glob("*.json")}
        assert first_plans
        assert all(json.loads(raw)["installation_id"] == first_id for raw in first_plans.values())

        (dependency / "src/annotated_doc/__init__.py").write_text("VALUE = 'edited'\n")
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency))
        second_id = "local-" + uuid.uuid4().hex
        _, second = prepare(second_id)
        assert second.generation != first.generation
        assert (value(first), value(second)) == ("0.0.5", "edited")
        assert (plans / second_id).is_dir()
        assert all(path.read_bytes() == raw for path, raw in first_plans.items())

        _, replay = prepare(first_id)
        assert replay.reused and replay.generation == first.generation
        assert value(replay) == "0.0.5"
        assert all(path.read_bytes() == raw for path, raw in first_plans.items())


@pytest.mark.parametrize("model_source", ["none", "native", "downloaded", "mixed"])
def test_real_modeled_package_prepare_can_be_acquired_before_model_selection(
    tmp_path: Path, model_source: str
) -> None:
    """The real producer/consumer handoff, before any model or handler may execute."""
    bind_model = model_source != "none"
    project = tmp_path / "project"
    _write_pure_project(project, "code-only", "1.0.0", "code_only_package")
    metadata = project / "pyproject.toml"
    metadata.write_text(
        metadata.read_text()
        + '\n[project.entry-points."cozy.application"]\ndefault="code_only_package:app"\n'
    )
    (project / "package.toml").write_text('[application]\nobject="code_only_package:app"\n')
    source = """import msgspec
from cozy_runtime.author import App, Config, Context, Loader, Model, uses_components
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    value: int
class Pipeline:
    def __init__(self, config: Config):
        import torch
        class Scalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.components = {"unet": Scalar()}
class Weights(Model[Pipeline]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=Pipeline)
    @uses_components("unet")
    def measure(self) -> int:
        raise AssertionError("model preparation cannot execute a handler")
app = App()
@app.entrypoint
def generate(ctx: Context, payload: Request, model: Weights) -> Result:
    raise AssertionError("code preparation cannot execute a handler")
"""
    if model_source == "mixed":
        source = source.replace("model: Weights)", "model: Weights, adapter: Weights)")
    (project / "src/code_only_package/__init__.py").write_text(source)
    captured = package_interface.canonical_bytes(static_interface.build(project))
    artifacts, environment = tmp_path / "artifacts", tmp_path / "environment"
    artifacts.mkdir()
    wheel_root = environment / ".stage" / "code-only-proof" / "wheels"
    wheel_root.mkdir(parents=True)
    _run("uv", "build", "--wheel", "--out-dir", str(wheel_root), str(project))
    if bind_model:
        torch = pytest.importorskip("torch", reason="native derivation needs the CPU torch CI lane")
        assert torch.version.cuda is None, "this proof uses CPU/meta derivation only"
    locked = _runtime_closure(
        wheel_root,
        *(("torch==2.14.0",) if bind_model else ()),
        index="https://download.pytorch.org/whl/cpu" if bind_model else "",
    )
    rows = _wheel_rows(list(wheel_root.glob("*.whl")))
    assert len(rows) == 2
    installation_id = "local-" + uuid.uuid4().hex
    python = authored_python()
    base = package_environment.observe_base(python)
    installed: list[package_installation.InstalledEnvironment] = []

    def describe(selected: package_installation.InstalledEnvironment, distribution: str) -> bytes:
        program = (
            "import sys; from cozy_runtime.internal import package_interface; "
            "from cozy_runtime.internal.discovery import discover_distribution; "
            f"body=package_interface.build(discover_distribution({distribution!r})); "
            "sys.stdout.buffer.write(package_interface.canonical_bytes(body))"
        )
        measured = _run(str(selected.python), "-I", "-c", program, cwd="/").stdout.encode()
        assert measured == captured
        return measured

    prepared = prepare_local_package(
        pb.PrepareLocalPackageRequest(
            operation_id="code-only-proof",
            package=_development("local/code-only", "1.0.0", installation_id),
            files=rows,
            dependency_requirements=locked,
            install_root=str(environment),
        ),
        artifact_cache=artifacts,
        install_root=environment,
        python=python,
        describe=describe,
        verified=lambda _digest, _length, _path: None,
        base=base,
        job_plan_root=tmp_path / "job-plans",
        materialized=installed.append,
    )
    (entry,) = read_placement_set(prepared.placement_set)
    assert not entry.entrypoints and not entry.models
    assert entry.installation_id == installed[0].installation_id == installation_id
    acquirer = Acquirer(
        cache_root=artifacts,
        install_root=environment,
        selection_root=tmp_path / "selection",
        tensorfs_root=tmp_path / "tensorfs",
        python=python,
        base=base,
        installed_environment=installed[0],
    )
    acquired = acquirer.acquire(entry)
    assert acquired.bindings == {}
    assert acquired.manifests == 0 and acquired.documents == 1
    assert acquired.installed == installed[0]
    (tmp_path / "prepared-code.json").write_bytes(
        prepared.placement_set.placement_set_canonical_bytes
    )
    if bind_model:
        from test_model_runtime_closure import _snapshot

        native_root = tmp_path / "native"
        native_root.mkdir()
        store_root, manifest, length, _store = _snapshot(
            native_root, include_asset=True, checkpoint_only=True
        )
        selection = pb.PreparePrivatePlacementRequest(
            operation_id="code-only-proof-model", installation_id=installation_id
        )
        if model_source in {"native", "mixed"}:
            selection.native_models.append(
                pb.NativeModelBinding(
                    slot="generate.models.model",
                    model="proof/native",
                    manifest=pb.Ref(digest=documents.raw(manifest), length=length),
                )
            )
        if model_source in {"downloaded", "mixed"}:
            # The second source is a real released root, not a native label pretending
            # to be downloadable. The same tiny bytes deliberately test native dedup.
            operation = _store.begin_operation("mixed-input-release", "proof", "base")
            operation.hold_manifest(manifest, length)
            operation.commit_release(None, "1.0.0", "bf16", manifest, length)
            selection.download_delegation, _ = documents.identity(
                pb.DownloadDelegation(
                    models=[
                        pb.DownloadModelRef(
                            package="local/code-only",
                            slot="generate.models.adapter"
                            if model_source == "mixed"
                            else "generate.models.model",
                            model="proof/base",
                            release="1.0.0",
                            lane="bf16",
                            manifest=manifest,
                        )
                    ]
                )
            )

        def prepare_selection(
            selected: pb.PreparePrivatePlacementRequest,
        ) -> pb.PreparePackageSetResult:
            return prepare_unpublished_placement(
                selected,
                prepared=entry,
                installed=installed[0],
                artifact_cache=artifacts,
                tensorfs_root=store_root,
                verified=lambda _digest, _length, _path: None,
            )

        modeled = prepare_selection(selection)
        if model_source == "mixed":
            duplicate = pb.PreparePrivatePlacementRequest()
            duplicate.CopyFrom(selection)
            duplicate.native_models[0].slot = "generate.models.adapter"
            with pytest.raises(PreparationRefusal, match="model_selection_mismatch"):
                prepare_selection(duplicate)
            wrong = pb.PreparePrivatePlacementRequest()
            wrong.CopyFrom(selection)
            wrong.native_models[0].manifest.digest = bytes(32)
            with pytest.raises(PreparationRefusal):
                prepare_selection(wrong)
            undeclared = pb.PreparePrivatePlacementRequest()
            undeclared.CopyFrom(selection)
            undeclared.native_models[0].slot = "generate.models.undeclared"
            with pytest.raises(PreparationRefusal, match="model_selection_mismatch"):
                prepare_selection(undeclared)
        (joined,) = read_placement_set(modeled.placement_set)
        assert [row.name for row in joined.entrypoints] == ["generate"]
        assert len(joined.models) == (2 if model_source == "mixed" else 1)
        assert {documents.spell(row.manifest.digest) for row in joined.models} == {manifest}
        selected = Acquirer(
            cache_root=artifacts,
            install_root=environment,
            selection_root=tmp_path / "selected-model",
            tensorfs_root=store_root,
            python=python,
            base=base,
            installed_environment=installed[0],
        ).acquire(joined)
        assert {binding.entrypoint for binding in selected.bindings.values()} == {"generate"}
        assert selected.manifests == 1
        (tmp_path / "prepared-model.json").write_bytes(
            modeled.placement_set.placement_set_canonical_bytes
        )


def _write_native_dependency(root: Path, version: str) -> None:
    """A REAL native dependency: one compiled CPython extension, not a renamed text file.

    The admission path under test loads these bytes with the dynamic loader and imports the
    module, so a fake `.so` would prove only that the fake was rejected.
    """

    package = root / "src" / "native_leaf"
    package.mkdir(parents=True)
    source = root / "csrc"
    source.mkdir()
    (root / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        'name = "native-leaf"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.12,<3.13"\n\n'
        "[tool.setuptools]\n"
        'package-dir = {"" = "src"}\n\n'
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n'
    )
    (root / "setup.py").write_text(
        "from setuptools import Extension, setup\n\n"
        "setup(ext_modules=[Extension('native_leaf._leaf', ['csrc/_leaf.c'])])\n"
    )
    (package / "__init__.py").write_text("from native_leaf._leaf import answer\n")
    # The C source lives OUTSIDE the package: a wheel may not ship its own build inputs,
    # and the admission path refuses one that does.
    (source / "_leaf.c").write_text(
        "#define PY_SSIZE_T_CLEAN\n"
        "#include <Python.h>\n\n"
        "static PyObject *answer(PyObject *self, PyObject *args) {\n"
        "    (void)self; (void)args;\n"
        "    return PyLong_FromLong(42);\n"
        "}\n\n"
        "static PyMethodDef methods[] = {\n"
        '    {"answer", answer, METH_NOARGS, "the admitted answer"},\n'
        "    {NULL, NULL, 0, NULL}\n"
        "};\n\n"
        "static struct PyModuleDef module = {\n"
        '    PyModuleDef_HEAD_INIT, "native_leaf._leaf", NULL, -1, methods\n'
        "};\n\n"
        "PyMODINIT_FUNC PyInit__leaf(void) { return PyModule_Create(&module); }\n"
    )


def test_native_dependency_wheel_installs_like_any_other_wheel() -> None:
    """A native dependency wheel installs with no prepare-time qualification (cr-093).

    The kernels link no framework (cr-091), so there is no torch-ABI fact for a prepare-time
    inspection to prove. Whether the extension loads is the executor's import to answer; the
    proof here is exactly that — the venv's own interpreter imports the module and runs it.
    """

    uv = shutil.which("uv")
    assert uv is not None
    if shutil.which("cc") is None:
        pytest.skip("a C compiler is required to build the native dependency under test")

    with tempfile.TemporaryDirectory(prefix="cozy-native-admission.") as raw:
        root = Path(raw)
        dependency = root / "native-leaf"
        _write_native_dependency(dependency, "0.0.1")

        artifact_cache = root / "artifacts"
        artifact_cache.mkdir()
        wheel_set = root / "environments" / ".stage" / "native-dependency-admission" / "wheels"
        wheel_set.mkdir(parents=True)
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(PACKAGE))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency))
        native_wheel_path = next(wheel_set.glob("native_leaf-*.whl"))
        # A pure-Python build would defeat the test silently, so the concrete ABI tag the
        # admission path requires is asserted, not assumed.
        assert "-py3-none-any.whl" not in native_wheel_path.name, native_wheel_path.name

        python = image_python()
        observed = package_environment.observe_base(python)
        environment = root / "environments"

        locked = _runtime_closure(wheel_set)
        installation_id = "local-" + uuid.uuid4().hex
        prepared = prepare_package(
            package_name="paul/marco-polo",
            release="1.0.4",
            files=_wheel_rows(list(wheel_set.glob("*.whl"))),
            wheel_root=wheel_set,
            installation_id=installation_id,
            artifact_cache=artifact_cache,
            install_root=environment,
            python=python,
            describe=_describe_installed,
            verified=lambda _digest, _length, _path: None,
            base=observed,
            dependency_requirements=locked,
            development=_development("paul/marco-polo", "1.0.4", installation_id),
        )
        assert prepared.placement_set.placement_set_canonical_bytes
        (venv_python,) = (environment / "installations").glob("*/venv/bin/python")
        assert (
            _run(
                str(venv_python), "-I", "-c", "import native_leaf; print(native_leaf.answer())"
            ).stdout.strip()
            == "42"
        )
