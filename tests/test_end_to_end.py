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
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import image_python
from cozy_runtime.internal import (
    base_observation,
    package_environment,
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
from cozy_runtime.internal.worker.session import Placement, read_placement_set
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


def _wheel_rows(paths: list[Path]) -> list[pb.LocalPackageWheel]:
    rows = [
        pb.LocalPackageWheel(
            digest=hashlib.sha256(path.read_bytes()).digest(),
            filename=path.name,
            length=path.stat().st_size,
            path=str(path),
        )
        for path in paths
    ]
    rows.sort(key=lambda row: row.digest)
    return rows


def _development(
    package: str, release: str, revision_digest: bytes | None = None
) -> pb.DevelopmentPackage:
    return pb.DevelopmentPackage(
        package=package,
        release=release,
        source_digest=hashlib.sha256(b"editable-source").digest(),
        local_revision_digest=revision_digest or hashlib.sha256(b"unknown-revision").digest(),
    )


def _measured_revision(refusal: PreparationRefusal) -> bytes:
    """The revision digest a first prepare measured, from its own typed refusal.

    Creator computes the exact LocalPackageRevision digest from its editable install; the
    test recovers the same value from the preparation's measurement instead of
    reimplementing the derivation beside it."""

    assert refusal.code == "local_package_revision_mismatch", refusal.code
    spelled = refusal.detail.split("measured ", 1)[1].split(",", 1)[0]
    return bytes.fromhex(spelled.removeprefix("sha256:"))


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
    # survives, and package interface /1 has no compatibility spelling for it.
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
        "raw=canonical.write(body); "
        "caught=False; "
        "\ntry: package_interface.read_bytes(raw)"
        "\nexcept package_interface.StalePackageInterface as exc: "
        "caught=exc.code == 'malformed_package_interface'"
        "\nassert caught",
        cwd="/",
    )

    _run(*cli, "frobnicate", cwd="/", expect=2)
    _run(*cli, "run", "nosuchfn", "message=marco", "--out", "/tmp", cwd="/", expect=4)
    _run(*cli, "run", "marco", "message=nope", "--out", "/tmp", cwd="/", expect=3)
    _run(*cli, "run", "marco", "bogus=1", "--out", "/tmp", cwd="/", expect=3)


def test_editable_revision_materializes_from_supplied_wheels() -> None:
    """The one preparation that still receives files (wire 30): an editable revision's
    wheels, installed into a fresh venv over a REAL base that owns the runtime closure —
    exactly a pod's image python. The project wheel is the row that MEASURES as the
    package's own distribution/release; the placement carries the supplied dependency as
    Environment.local_wheels and the project fact on DevelopmentPackage. A wheel shadowing
    a protected image family at another version refuses typed."""

    uv = shutil.which("uv")
    assert uv is not None
    python = image_python()
    observed = package_environment.observe_base(python)
    assert "cozy-runtime" in observed.distribution_versions

    with tempfile.TemporaryDirectory(prefix="cozy-editable-revision.") as raw:
        root = Path(raw)
        dependency = root / "annotated-doc-0.0.5"
        _write_annotated_doc(dependency, "0.0.5")
        impostor = root / "impostor"
        _write_pure_project(impostor, "cozy-runtime", "9.9.9", "cozy_runtime_impostor")
        artifact_cache = root / "artifacts"
        artifact_cache.mkdir()
        wheel_set = root / "environments" / ".stage" / "editable-revision-proof" / "wheels"
        wheel_set.mkdir(parents=True)
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(PACKAGE))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(impostor))
        project_wheel = next(wheel_set.glob("marco_polo-*.whl"))
        dependency_wheel = next(wheel_set.glob("annotated_doc-0.0.5-*.whl"))
        impostor_wheel = next(wheel_set.glob("cozy_runtime-9.9.9-*.whl"))
        environment = root / "environments"

        def describe_real(
            selected: package_environment.InstalledEnvironment, distribution: str
        ) -> bytes:
            command = (
                "from cozy_runtime.internal import package_interface; "
                "from cozy_runtime.internal.discovery import discover_distribution; "
                f"found=discover_distribution({distribution!r}); "
                "print(package_interface.canonical_bytes(package_interface.build(found)).decode())"
            )
            rendered = _run(str(selected.python), "-I", "-c", command, cwd="/")
            return rendered.stdout.encode()

        def prepare(
            wheels: list[Path], revision_digest: bytes | None
        ) -> pb.PreparePackageSetResult:
            return prepare_local_package(
                pb.PrepareLocalPackageRequest(
                    operation_id="editable-revision-proof",
                    package=_development("paul/marco-polo", "1.0.4", revision_digest),
                    wheels=_wheel_rows(wheels),
                    install_root=str(environment),
                ),
                artifact_cache=artifact_cache,
                install_root=environment,
                python=python,
                describe=describe_real,
                verified=lambda _digest, _length, _path: None,
                base=observed,
                job_plan_root=root / "job-plans",
            )

        # Creator's revision digest is exact; the first measurement supplies it here.
        with pytest.raises(PreparationRefusal) as unknown:
            prepare([project_wheel, dependency_wheel], None)
        revision = _measured_revision(unknown.value)

        prepared = prepare([project_wheel, dependency_wheel], revision)
        placement = json.loads(prepared.placement_set.placement_set_canonical_bytes)["placements"][
            0
        ]
        assert placement["development"]["local_revision_digest"] == ("sha256:" + revision.hex())
        assert placement["development"]["project_wheel"]["distribution"] == "marco-polo"
        local_wheels = placement["environment"]["local_wheels"]
        assert [row["distribution"] for row in local_wheels] == ["annotated-doc"]
        assert "locked_requirements" not in placement["environment"]
        generation = environment / "contents"
        (venv_python,) = generation.glob("*/bin/python")
        _run(str(venv_python), "-I", "-c", "import marco_polo_package, annotated_doc", cwd="/")

        # A protected image family cannot be replaced by a supplied wheel: the measure
        # itself convicts the impostor before any install.
        with pytest.raises(PreparationRefusal) as shadowed:
            prepare([project_wheel, impostor_wheel], revision)
        assert shadowed.value.code == "package_environment_base_distribution_import_roots_conflict"
        assert "cozy-runtime 9.9.9" in shadowed.value.detail
        assert "cannot be replaced" in shadowed.value.detail

        # A dependency edit changes the executable closure even when the project
        # wheel and callable interface are identical. Both plans remain resumable.
        first_plans = {path: path.read_bytes() for path in (root / "job-plans").glob("*/*.json")}
        assert first_plans
        assert all(
            json.loads(raw)["build_id"] == "sha256:" + revision.hex()
            for raw in first_plans.values()
        )
        original_dependency = dependency_wheel.read_bytes()
        original_project = project_wheel.read_bytes()
        (dependency / "src/annotated_doc/__init__.py").write_text("VALUE = 'edited'\n")
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(dependency))
        assert project_wheel.read_bytes() == original_project
        assert dependency_wheel.read_bytes() != original_dependency
        with pytest.raises(PreparationRefusal) as changed:
            prepare([project_wheel, dependency_wheel], revision)
        edited_revision = _measured_revision(changed.value)
        assert edited_revision != revision
        prepare([project_wheel, dependency_wheel], edited_revision)
        assert (root / "job-plans" / edited_revision.hex()).is_dir()
        assert all(path.read_bytes() == raw for path, raw in first_plans.items())
        dependency_wheel.write_bytes(original_dependency)
        prepare([project_wheel, dependency_wheel], revision)
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
    rows = _wheel_rows(list(wheel_root.glob("*.whl")))
    assert len(rows) == 1
    source_digest = hashlib.sha256(source.encode()).digest()
    # Author the expected identity before preparation, as the caller does. Never
    # replace it with a digest learned from a worker refusal.
    _, revision = documents.identity(
        pb.LocalPackageRevision(
            package="local/code-only",
            release="1.0.0",
            source_digest=source_digest,
            package_interface=pb.Ref(digest=documents.digest_of(captured), length=len(captured)),
            files=[
                pb.LocalPackageFileRef(digest=r.digest, filename=r.filename, length=r.length)
                for r in rows
            ],
        )
    )
    python = image_python()
    if bind_model:
        torch = pytest.importorskip("torch", reason="native derivation needs the CPU torch CI lane")
        assert torch.version.cuda is None, "this proof uses CPU/meta derivation only"
        # A separate real base prefix keeps the torch-free image fixture unchanged.
        native_image = tmp_path / "cpu-image"
        shutil.copytree(python.parent.parent, native_image, symlinks=True)
        python = native_image / "bin" / python.name
        _run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--index",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.13.0",
        )
    base = package_environment.observe_base(python)
    installed: list[package_environment.InstalledEnvironment] = []

    def describe(selected: package_environment.InstalledEnvironment, distribution: str) -> bytes:
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
            package=pb.DevelopmentPackage(
                package="local/code-only",
                release="1.0.0",
                source_digest=source_digest,
                local_revision_digest=revision,
            ),
            wheels=rows,
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
    assert entry["entrypoints"] == entry["models"] == []
    placement = Placement(
        placement_id=entry["placement_id"],
        placement_set_digest=prepared.placement_set.placement_set_digest,
        package={},
        development=entry["development"],
        environment_digest=documents.raw(entry["environment_digest"]),
        package_interface=(
            entry["package_interface"]["digest"],
            entry["package_interface"]["length"],
        ),
        bindings_digest=documents.raw(entry["bindings_digest"]),
        models=(),
        entrypoints=(),
        environment=entry["environment"],
        document=entry,
    )
    acquirer = Acquirer(
        cache_root=artifacts,
        install_root=environment,
        selection_root=tmp_path / "selection",
        tensorfs_root=tmp_path / "tensorfs",
        python=python,
        base=base,
        installed_environment=installed[0],
    )
    acquired = acquirer.acquire(placement)
    assert acquired.bindings == {}
    assert acquired.model_objects == 0 and acquired.documents == 1
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
            operation_id="code-only-proof-model", local_revision_digest=revision
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
        assert [row["name"] for row in joined["entrypoints"]] == ["generate"]
        assert len(joined["models"]) == (2 if model_source == "mixed" else 1)
        assert {row["manifest"]["digest"] for row in joined["models"]} == {manifest}
        bound = Placement(
            placement_id=joined["placement_id"],
            placement_set_digest=modeled.placement_set.placement_set_digest,
            package={},
            development=joined["development"],
            environment_digest=documents.raw(joined["environment_digest"]),
            package_interface=placement.package_interface,
            bindings_digest=documents.raw(joined["bindings_digest"]),
            models=tuple(joined["models"]),
            entrypoints=tuple(joined["entrypoints"]),
            environment=joined["environment"],
            document=joined,
        )
        selected = Acquirer(
            cache_root=artifacts,
            install_root=environment,
            selection_root=tmp_path / "selected-model",
            tensorfs_root=store_root,
            python=python,
            base=base,
            installed_environment=installed[0],
        ).acquire(bound)
        assert {binding.entrypoint for binding in selected.bindings.values()} == {"generate"}
        assert selected.model_objects == (6 if model_source == "mixed" else 3)
        (tmp_path / "prepared-model.json").write_bytes(
            modeled.placement_set.placement_set_canonical_bytes
        )


def test_project_wheel_with_no_import_roots_is_refused_before_the_document() -> None:
    """The live defect (2026-09-01): a flat-layout project whose build backend discovered
    no module reached a paid pod as a dist-info-only wheel and was refused as
    `project_wheel_invalid: wheel row has the wrong fields` — the writer had dropped the
    empty `import_roots` and the reader read the absence as a malformed row. Preparation
    now names the defect itself, before any document exists; and a WheelFact with no
    roots or tags still spells both collections, as the protocol's canonical form requires.
    """

    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cozy-empty-project.") as raw:
        root = Path(raw)
        project = root / "hollow"
        (project / "vendor").mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            "[project]\n"
            'name = "hollow"\n'
            'version = "1.0.0"\n'
            'requires-python = ">=3.12,<3.13"\n\n'
            "[tool.hatch.build.targets.wheel]\n"
            'only-include = ["vendor"]\n'
        )
        (project / "vendor" / "notes.txt").write_text("no module lives here\n")
        artifact_cache = root / "artifacts"
        wheel_set = root / "environment" / ".stage" / "empty-project-proof" / "wheels"
        wheel_set.mkdir(parents=True)
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(project))
        _run(uv, "build", "--wheel", "--out-dir", str(wheel_set), str(ROOT))
        project_wheel = next(wheel_set.glob("hollow-*.whl"))
        runtime_wheel = next(wheel_set.glob("cozy_runtime-*.whl"))
        with zipfile.ZipFile(project_wheel) as archive:
            members = {member.filename for member in archive.infolist()}
        assert "vendor/notes.txt" in members
        assert not package_environment._import_roots(members)

        with pytest.raises(PreparationRefusal) as refused:
            prepare_package(
                package_name="local/hollow",
                release="1.0.0",
                wheels=_wheel_rows([project_wheel, runtime_wheel]),
                wheel_root=wheel_set,
                placement_seed=b"empty-project-proof",
                artifact_cache=artifact_cache,
                install_root=root / "environment",
                python=Path(sys.executable),
                describe=lambda _installed, _distribution: b"",
                verified=lambda _digest, _length, _path: None,
                base=package_environment.observe_base(Path(sys.executable)),
                development=_development("local/hollow", "1.0.0"),
            )
        assert refused.value.code == "package_prepare_project_wheel_no_import_roots"
        assert refused.value.detail == f"{project_wheel.name} installs no importable module"

    hollow = pb.WheelFact(
        ref=pb.Ref(digest=b"\x27" * 32, length=1),
        distribution="hollow",
        version="1.0.0",
        filename="hollow-1.0.0-py3-none-any.whl",
    )
    assert documents.body(hollow) == {
        "ref": {"digest": "sha256:" + "27" * 32, "length": 1},
        "distribution": "hollow",
        "version": "1.0.0",
        "filename": "hollow-1.0.0-py3-none-any.whl",
        "import_roots": [],
        "tags": [],
    }
    assert documents.body(pb.Placement(placement_id="p")) == {
        "placement_id": "p",
        "entrypoints": [],
        "models": [],
    }


def test_private_dependencies_replace_image_versions_without_borrowing_them(tmp_path: Path) -> None:
    from test_unpublished_exact_dependencies import _wheel

    base = base_observation.current()
    project = _wheel(tmp_path, "private_imaging", ("pillow==13.0",), b"import pillow\n")
    pillow = _wheel(tmp_path, "pillow", (), b"VALUE=13\n", version="13.0")
    _, selected, supplied = package_environment.preflight_environment(project, (pillow,), base)
    assert "pillow" in {row.distribution for row in supplied}
    assert "pillow" in {row.distribution for row in selected}
    with pytest.raises(package_environment.EnvironmentRefusal, match="dependency_missing"):
        package_environment.preflight_environment(project, (), base)


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
        # The second job gets another output workspace and must not conflict
        # with the first invocation's immutable executable plan.
        for verb, function in (("run", "marco"), ("job", "marco_job"), ("job", "marco_job")):
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
        project_wheel = next(wheel_set.glob("marco_polo-*.whl"))
        native_wheel_path = next(wheel_set.glob("native_leaf-*.whl"))
        # A pure-Python build would defeat the test silently, so the concrete ABI tag the
        # admission path requires is asserted, not assumed.
        assert "-py3-none-any.whl" not in native_wheel_path.name, native_wheel_path.name

        python = image_python()
        observed = package_environment.observe_base(python)
        environment = root / "environments"

        def describe_real(
            selected: package_environment.InstalledEnvironment, distribution: str
        ) -> bytes:
            command = (
                "from cozy_runtime.internal import package_interface; "
                "from cozy_runtime.internal.discovery import discover_distribution; "
                f"found=discover_distribution({distribution!r}); "
                "print(package_interface.canonical_bytes(package_interface.build(found)).decode())"
            )
            return _run(str(selected.python), "-I", "-c", command, cwd="/").stdout.encode()

        def prepare(revision_digest: bytes | None) -> pb.PreparePackageSetResult:
            return prepare_package(
                package_name="paul/marco-polo",
                release="1.0.4",
                wheels=_wheel_rows([project_wheel, native_wheel_path]),
                wheel_root=wheel_set,
                placement_seed=b"native-dependency-admission",
                artifact_cache=artifact_cache,
                install_root=environment,
                python=python,
                describe=describe_real,
                verified=lambda _digest, _length, _path: None,
                base=observed,
                development=_development("paul/marco-polo", "1.0.4", revision_digest),
            )

        with pytest.raises(PreparationRefusal) as unknown:
            prepare(None)
        prepared = prepare(_measured_revision(unknown.value))
        assert prepared.placement_set.placement_set_canonical_bytes
        (venv_python,) = (environment / "contents").glob("*/bin/python")
        assert (
            _run(
                str(venv_python), "-I", "-c", "import native_leaf; print(native_leaf.answer())"
            ).stdout.strip()
            == "42"
        )
