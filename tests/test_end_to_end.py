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
import dataclasses
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import base_observation, package_environment
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, prepare_package
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


def _download_row(kind: pb.LocalDownloadKind, path: Path) -> pb.LocalDownloadFile:
    raw = path.read_bytes()
    return pb.LocalDownloadFile(
        digest=hashlib.sha256(raw).digest(),
        filename=path.name,
        kind=kind,
        length=len(raw),
        path=str(path),
    )


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

    The descriptor is a PUBLICATION output, so two derivations must be byte-identical, and
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
    # survives, and descriptor /1 has no compatibility spelling for it.
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
        "import json; from cozy_runtime.internal import canonical, descriptor; "
        f"body=json.loads({first.stdout!r}); "
        "body['jobs'][0]['resources']={'gpu_count': 1}; "
        "raw=canonical.write(body); "
        "caught=False; "
        "\ntry: descriptor.read_bytes(raw)"
        "\nexcept descriptor.StalePackageDescriptor as exc: "
        "caught=exc.code == 'malformed_package_descriptor'"
        "\nassert caught",
        cwd="/",
    )

    _run(*cli, "frobnicate", cwd="/", expect=2)
    _run(*cli, "run", "nosuchfn", "message=marco", "--out", "/tmp", cwd="/", expect=4)
    _run(*cli, "run", "marco", "message=nope", "--out", "/tmp", cwd="/", expect=3)
    _run(*cli, "run", "marco", "bogus=1", "--out", "/tmp", cwd="/", expect=3)


def test_existing_package_environment_owns_its_exact_dependency_wheels() -> None:
    """A complete local venv owns its project/dependencies; a worker image does not.

    This is the exact SDXL regression shape: preparation receives the immutable wheels used
    to create an already-complete local uv environment, observes those installed versions,
    and must not reinterpret an identical package dependency as a worker-image shadow.
    """

    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cozy-existing-environment.") as raw:
        root = Path(raw)
        dependency_one = root / "annotated-doc-0.0.5"
        dependency_two = root / "annotated-doc-0.0.6"
        _write_annotated_doc(dependency_one, "0.0.5")
        _write_annotated_doc(dependency_two, "0.0.6")

        artifact_cache = root / "artifacts"
        wheel_set = artifact_cache / "sets" / "package"
        wheel_set.mkdir(parents=True)
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(ROOT))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(PACKAGE))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency_one))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency_two))
        runtime_wheel = next(wheel_set.glob("cozy_runtime-*.whl"))
        dependency_one_wheel = next(wheel_set.glob("annotated_doc-0.0.5-*.whl"))
        dependency_two_wheel = next(wheel_set.glob("annotated_doc-0.0.6-*.whl"))
        project_wheel = next(wheel_set.glob("marco_polo-*.whl"))

        environment = root / "environment"
        python = environment / "bin" / "python"
        _run(uv, "venv", "--python", sys.executable, str(environment))
        _run(
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            str(runtime_wheel),
            str(dependency_one_wheel),
            str(project_wheel),
        )
        release_digest = "sha256:" + "17" * 32
        installed = package_environment.open_existing_environment(python, release_digest)
        observed = package_environment.observe_base(python)
        artifact_store = root / "artifact-store"
        artifact_store.mkdir()

        def describe_real(
            selected: package_environment.InstalledEnvironment, distribution: str
        ) -> bytes:
            assert selected == installed
            command = (
                "from cozy_runtime.internal import descriptor; "
                "from cozy_runtime.internal.discovery import discover_distribution; "
                f"found=discover_distribution({distribution!r}); "
                "print(descriptor.canonical_bytes(descriptor.build(found)).decode())"
            )
            rendered = _run(str(selected.python), "-I", "-c", command, cwd="/")
            return rendered.stdout.encode()

        def prepare(
            dependency: Path, base: base_observation.Observation = observed
        ) -> pb.PreparePackageSetResult:
            rows = [
                _download_row(pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, project_wheel),
                _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, dependency),
                _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, runtime_wheel),
            ]
            rows.sort(key=lambda row: row.digest)
            return prepare_package(
                package_name="paul/marco-polo",
                release="1.0.4",
                release_digest=release_digest,
                files=rows,
                placement_seed=b"existing-environment-proof",
                artifact_cache=artifact_cache,
                artifact_store=artifact_store,
                environment_root=environment,
                python=python,
                describe=describe_real,
                verified=lambda _digest, _length, _path: None,
                base=base,
                installed_environment=installed,
            )

        prepared = prepare(dependency_one_wheel)
        assert prepared.placement_set.placement_set_canonical_bytes

        with pytest.raises(PreparationRefusal) as mismatch:
            prepare(dependency_two_wheel)
        assert mismatch.value.code == "package_environment_shadows_base_distribution"
        assert "package_version=0.0.6 image_version=0.0.5" in mismatch.value.detail

        package_names = {"marco-polo", "annotated-doc"}
        rental_distributions = tuple(
            (name, "0.1.0" if name == "cozy-runtime" else version)
            for name, version in observed.distributions
            if name not in package_names
        )
        rental_base = dataclasses.replace(observed, distributions=rental_distributions)
        _base, package_wheels, supplied = package_environment.preflight_environment(
            project_wheel,
            (dependency_one_wheel, runtime_wheel),
            rental_base,
        )
        assert "cozy-runtime" not in {wheel.distribution for wheel in package_wheels}
        assert "cozy-runtime" in {wheel.distribution for wheel in supplied}

        worker_shadow = dataclasses.replace(
            rental_base,
            distributions=(*rental_base.distributions, ("annotated-doc", "0.0.5")),
        )
        with pytest.raises(
            package_environment.EnvironmentRefusal,
            match="package_environment_shadows_base_distribution",
        ):
            package_environment.preflight_environment(
                project_wheel,
                (dependency_one_wheel, runtime_wheel),
                worker_shadow,
            )


def test_worker_image_owns_pillow_and_numpy_as_peer_dependencies() -> None:
    """Pillow and NumPy are worker-image families, so no package copy can displace them.

    Pillow is cozy-runtime's own public dependency (`pillow>=12,<13`) and NumPy shares a
    compiled ABI with the Torch family the image pins, so image code — not only package code
    — imports whatever ends up on the venv's path. `--system-site-packages` puts the package
    venv AHEAD of the image, so an admitted copy wins the import.

    Three facts, against the real interpreter this suite runs in:
      * a package copy of pillow is image-owned: admitted as a fact, omitted from the install;
      * a FOREIGN distribution that installs the same `PIL` import root is refused — this is
        the shadow the distribution name alone never caught;
      * a package requirement the image's pillow cannot satisfy is refused typed, which is
        what replaces the copy the package was not allowed to bring.
    """

    uv = shutil.which("uv")
    assert uv is not None
    base = base_observation.current()
    pillow = base.distribution_versions.get("pillow")
    assert pillow, "this interpreter must carry the image's pillow"
    assert base.root_providers.get("PIL") == ("pillow",)

    with tempfile.TemporaryDirectory(prefix="cozy-image-owned.") as raw:
        root = Path(raw)
        vendored = root / "vendored-pillow"
        foreign = root / "foreign-pil"
        demanding = root / "demanding"
        _write_pure_project(vendored, "pillow", "12.99.0", "PIL")
        _write_pure_project(foreign, "shady-imaging", "0.0.1", "PIL")
        _write_pure_project(demanding, "demanding", "0.0.1", "demanding", ("pillow>=13",))
        dist = root / "dist"
        _run(uv, "build", "--wheel", "--out-dir", str(dist), str(ROOT))
        _run(uv, "build", "--wheel", "--out-dir", str(dist), str(PACKAGE))
        _run(uv, "build", "--wheel", "--out-dir", str(dist), str(vendored))
        _run(uv, "build", "--wheel", "--out-dir", str(dist), str(foreign))
        _run(uv, "build", "--wheel", "--out-dir", str(dist), str(demanding))
        project_wheel = next(dist.glob("marco_polo-*.whl"))
        runtime_wheel = next(dist.glob("cozy_runtime-*.whl"))
        vendored_wheel = next(dist.glob("pillow-12.99.0-*.whl"))
        foreign_wheel = next(dist.glob("shady_imaging-*.whl"))
        demanding_wheel = next(dist.glob("demanding-*.whl"))

        _base, package_wheels, supplied = package_environment.preflight_environment(
            project_wheel, (vendored_wheel, runtime_wheel), base
        )
        assert "pillow" in {wheel.distribution for wheel in supplied}
        assert "pillow" not in {wheel.distribution for wheel in package_wheels}

        with pytest.raises(
            package_environment.EnvironmentRefusal,
            match="package_environment_shadows_base_import",
        ):
            package_environment.preflight_environment(
                project_wheel, (foreign_wheel, runtime_wheel), base
            )

        with pytest.raises(package_environment.EnvironmentRefusal) as conflict:
            package_environment.preflight_environment(
                project_wheel, (demanding_wheel, runtime_wheel), base
            )
        assert conflict.value.code == "package_environment_dependency_base_conflict"
        assert "the worker image provides pillow" in conflict.value.detail


@needs_executor
def test_installed_cli_serves_a_request(installed_cli: str) -> None:
    """One served request, all the way through: worker control plus a spawned executor.

    decode -> resolve -> preflight -> hydrate -> invoke -> validate -> finalize, returning
    the package's own result over the real protocol bytes.
    """
    with tempfile.TemporaryDirectory(prefix="cz.", dir="/tmp") as out:
        served = _run(
            installed_cli,
            "--dir",
            str(PACKAGE),
            "--json",
            "run",
            "marco",
            "message=marco",
            "--out",
            out,
            cwd="/",
        )
    body = json.loads(served.stdout)
    assert body["result"] == {"message": "polo"}
    assert body["status"] == "OUTCOME_STATUS_SUCCEEDED"


@needs_executor
def test_installed_cli_runs_a_job_through_the_local_worker(installed_cli: str) -> None:
    """A JOB, which is a real two-process run: worker control plus a spawned executor.

    Not a second code path — the same acceptance boundary, plan and attempt kernel a served
    request takes, over a package that declares no model. The short `--out` root is
    deliberate: the executor's control socket lives under it and `sockaddr_un.sun_path`
    holds 108 bytes.
    """
    with tempfile.TemporaryDirectory(prefix="cz.", dir="/tmp") as out:
        done = _run(
            installed_cli,
            "--dir",
            str(PACKAGE),
            "--json",
            "job",
            "marco_job",
            "message=marco",
            "--out",
            out,
            cwd="/",
        )
    body = json.loads(done.stdout)
    assert body["status"] == "OUTCOME_STATUS_SUCCEEDED", body
    assert body["result"] == {"message": "polo"}


def test_a_refusal_terminates_the_process(installed_cli: str) -> None:
    """CONTAINMENT UNAVAILABLE -> a typed refusal AND AN EXIT, for `run` and for `job`.

    This is the arm cr-061 was missing, and the shape of the assertion is the whole point.
    The refusal was never broken: on a host whose worker cgroup is not delegated the runtime
    computed the right diagnosis and printed it. It then sat on a control stream nobody was
    going to close, until SIGKILL — on any plain container, CI runner or developer box with a
    different cgroup layout. An arm that asserted only that the refusal was PRINTED is
    exactly the arm that would have passed against the bug.

    So: a bounded wait, a non-zero status, and the typed error token on stdout. The wait is
    the assertion.
    """
    with _containment_unavailable() as enter_scope:
        for verb, function in (("run", "marco"), ("job", "marco_job")):
            with tempfile.TemporaryDirectory(prefix="cz.", dir="/tmp") as out:
                refused = subprocess.run(
                    (
                        installed_cli,
                        "--dir",
                        str(PACKAGE),
                        "--json",
                        verb,
                        function,
                        "message=marco",
                        "--out",
                        out,
                    ),
                    cwd="/",
                    capture_output=True,
                    text=True,
                    timeout=REFUSAL_TIMEOUT,
                    stdin=subprocess.DEVNULL,
                    env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                    preexec_fn=enter_scope,
                )
            assert refused.returncode != 0, (
                f"`{verb}` exited 0 with no executor: it cannot have run anything\n"
                f"--- stdout ---\n{refused.stdout[-4000:]}"
            )
            error = _trailing_json(refused.stderr)["error"]
            assert (error["name"], error["code"], refused.returncode) == ("unavailable", 9, 9), (
                error,
                refused.returncode,
            )
            assert "containment unavailable" in error["message"], error


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


def test_native_dependency_wheel_is_admitted_on_proof_not_on_caller_identity() -> None:
    """A native dependency wheel installs only where it is proved to load, import and run.

    The retired `allow_native_custom` flag admitted native wheels on WHO was asking. This
    asserts the replacement: the same wheel refuses with no qualification host to prove it
    against, and is admitted once the sealed execution environment is supplied — because the
    loader actually opened the ELF and the interpreter actually imported the module.
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
        wheel_set = artifact_cache / "sets" / "package"
        wheel_set.mkdir(parents=True)
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(ROOT))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(PACKAGE))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency))
        runtime_wheel = next(wheel_set.glob("cozy_runtime-*.whl"))
        project_wheel = next(wheel_set.glob("marco_polo-*.whl"))
        native_wheel_path = next(wheel_set.glob("native_leaf-*.whl"))
        # A pure-Python build would defeat the test silently, so the concrete ABI tag the
        # admission path requires is asserted, not assumed.
        assert "-py3-none-any.whl" not in native_wheel_path.name, native_wheel_path.name

        environment = root / "environment"
        python = environment / "bin" / "python"
        _run(uv, "venv", "--python", sys.executable, str(environment))
        _run(
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            str(runtime_wheel),
            str(native_wheel_path),
            str(project_wheel),
        )
        release_digest = "sha256:" + "23" * 32
        installed = package_environment.open_existing_environment(python, release_digest)
        observed = package_environment.observe_base(python)
        artifact_store = root / "artifact-store"
        artifact_store.mkdir()

        def describe_real(
            selected: package_environment.InstalledEnvironment, distribution: str
        ) -> bytes:
            command = (
                "from cozy_runtime.internal import descriptor; "
                "from cozy_runtime.internal.discovery import discover_distribution; "
                f"found=discover_distribution({distribution!r}); "
                "print(descriptor.canonical_bytes(descriptor.build(found)).decode())"
            )
            return _run(str(selected.python), "-I", "-c", command, cwd="/").stdout.encode()

        rows = [
            _download_row(pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, project_wheel),
            _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, native_wheel_path),
            _download_row(pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, runtime_wheel),
        ]
        rows.sort(key=lambda row: row.digest)

        def prepare(native_environment: dict[str, str] | None) -> pb.PreparePackageSetResult:
            return prepare_package(
                package_name="paul/marco-polo",
                release="1.0.4",
                release_digest=release_digest,
                files=rows,
                placement_seed=b"native-dependency-admission",
                artifact_cache=artifact_cache,
                artifact_store=artifact_store,
                environment_root=environment,
                python=python,
                describe=describe_real,
                verified=lambda _digest, _length, _path: None,
                base=observed,
                native_environment=native_environment,
                installed_environment=installed,
            )

        # RED ARM: no sealed execution environment means no qualification host. There is no
        # caller identity that substitutes for one.
        with pytest.raises(PreparationRefusal) as absent:
            prepare(None)
        assert absent.value.code == "package_environment_native_proof_environment_absent"

        # GREEN ARM: the same bytes, admitted, because the operator child opened the ELF with
        # the dynamic loader and imported the module out of this exact environment.
        prepared = prepare(_sealed_child_environment())
        assert prepared.placement_set.placement_set_canonical_bytes
        assert (
            _run(
                str(python), "-I", "-c", "import native_leaf; print(native_leaf.answer())"
            ).stdout.strip()
            == "42"
        )


def _sealed_child_environment() -> dict[str, str]:
    """The seal a worker would hand the operator child, built through the one env authority."""

    from cozy_runtime.internal import config, spawn

    return spawn.seal(config.read_config().child_base_env, {})
