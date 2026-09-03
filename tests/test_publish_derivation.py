"""cr-067/cr-077: the author derives, the pod serves from the published package interface.

Three planes, each driven for real:

* the author surface — capability is the slot's class annotation and NOTHING else: the
  retired `source_profiles` door refuses typed, and the interface slot grammar no
  longer admits the field (hard cut, cr-077);
* the pod preparation plane — the hub-committed manifest + PackageInterface staged beside the
  set are the authoritative wiring, import-time derivation is a byte-compared cross-check,
  and NOTHING on this plane may load the derive machinery (the hard guard, proven red);
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import msgspec
import pytest

from conftest import sealed_child_environment
from cozy_runtime.author import App, ConformanceError, Model, WeightsOutput, WeightsSink
from cozy_runtime.author._describe import describe
from cozy_runtime.author._loader import Loader
from cozy_runtime.internal import canonical, package_environment, package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    prepare_package_set,
)
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"


class Request(msgspec.Struct):
    prompt: str


class Reply(msgspec.Struct):
    text: str


class Component:
    def state_dict(self) -> Mapping[str, object]:
        return {}

    def __call__(self) -> str:
        return "denoised"


class Pipeline:
    def __init__(self) -> None:
        self.components: dict[str, object] = {"transformer": Component()}


class Denoiser(Model[Pipeline]):
    pipeline: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipeline = loader.construct(Pipeline, factory=lambda config: Pipeline())


def _described(app: App) -> dict[str, object]:
    found = Discovered(
        app=app,
        application="profiled_package:app",
        project=PACKAGE,
        module=sys.modules[__name__],
        surfaces=describe(app),
        bindings={},
    )
    return package_interface.build(found)


def test_source_profiles_refuse_typed_at_registration() -> None:
    """cr-077 hard cut: the retired declared-pairs door refuses with ONE typed code on
    both decorators, before any surface analysis runs."""
    app = App()

    with pytest.raises(ConformanceError) as on_entrypoint:

        @app.entrypoint(source_profiles={"denoiser": "acme/sd-turbo/1.0.0/bf16"})
        def generate(payload: Request, denoiser: Denoiser) -> Reply:
            return Reply(text="no")

    assert on_entrypoint.value.code == "source_profiles_retired"
    assert on_entrypoint.value.fields == ("denoiser",)

    with pytest.raises(ConformanceError) as on_job:

        @app.job(source_profiles={"denoiser": "acme/sd-turbo/1.0.0/bf16"})
        def train(payload: Request, denoiser: Denoiser) -> Reply:
            return Reply(text="no")

    assert on_job.value.code == "source_profiles_retired"

    # An explicit empty mapping is the absent case, not a declaration: registers fine.
    @app.entrypoint(source_profiles={})
    def clean(payload: Request, denoiser: Denoiser) -> Reply:
        return Reply(text="ok")

    assert describe(app)


def test_lane_words_are_not_selection_leaks() -> None:
    """cr-077: a weights output named after the lane it derives ("bf16") describes fine
    beside a default binding whose lane says the same word; an interface string equal to
    the binding's release remains a refused selection leak."""

    def build(output_id: str, release: str) -> dict[str, object]:
        app = App()

        @app.job(weights=(WeightsOutput(output_id, max_new_bytes=1 << 20),))
        def produce(payload: Request, denoiser: Denoiser, weights: WeightsSink) -> Reply:
            return Reply(text="ok")

        found = Discovered(
            app=app,
            application="profiled_package:app",
            project=PACKAGE,
            module=sys.modules[__name__],
            surfaces=describe(app),
            bindings={"Denoiser": {"model": "acme/sd", "release": release, "lane": "bf16"}},
        )
        return package_interface.build(found)

    body = build("bf16", "1.0.0")
    assert body["jobs"][0]["weights_outputs"][0]["output_id"] == "bf16"  # type: ignore[index]

    with pytest.raises(ConformanceError) as leaked_release:
        build("ev-003", "ev-003")
    assert leaked_release.value.code == "fixed_point"


def test_interface_slot_grammar_has_no_legacy_members() -> None:
    """The current slot grammar admits no retired field, including an empty value."""
    app = App()

    @app.entrypoint
    def generate(payload: Request, denoiser: Denoiser) -> Reply:
        return Reply(text="ok")

    body = json.loads(package_interface.canonical_bytes(_described(app)))
    slot = body["entrypoints"][0]["models"][0]
    assert set(slot) == {"path", "class", "component_use", "encoded_leaves"}
    assert slot["encoded_leaves"] == "refuse"

    def planted(**members: object) -> bytes:
        mutated = json.loads(json.dumps(body))
        mutated["entrypoints"][0]["models"][0].update(members)
        return canonical.write(mutated)

    legacy_members: tuple[dict[str, object], ...] = (
        {"stamps": {}},
        {"stamps": {"task": "fl2va"}},
        {"source_profile": "acme/sd-turbo/1.0.0/bf16"},
    )
    for legacy in legacy_members:
        with pytest.raises(package_interface.StalePackageInterface) as retired:
            package_interface.read_bytes(planted(**legacy))
        assert retired.value.code == "malformed_package_interface"
    with pytest.raises(package_interface.StalePackageInterface) as unknown:
        package_interface.read_bytes(planted(encoded_leaves="maybe"))
    assert unknown.value.code == "malformed_package_interface"


def test_model_class_carries_exactly_one_keyword() -> None:
    """`encoded_leaves` is the whole class-keyword plane; a stamp refuses typed (D1)."""
    with pytest.raises(ConformanceError) as stamped:

        class Stamped(Model[Pipeline], task="fl2va"):
            pass

    assert stamped.value.code == "class_keyword"
    assert stamped.value.fields == ("task",)

    with pytest.raises(ConformanceError) as structured:

        class Structured(Model[Pipeline], structure="sdxl"):
            pass

    assert structured.value.code == "class_keyword"

    class Consenting(Model[Pipeline], encoded_leaves="accept"):
        pass

    assert Consenting.__encoded_leaves__ == "accept"
    with pytest.raises(ConformanceError) as closed:

        class Vague(Model[Pipeline], encoded_leaves="maybe"):
            pass

    assert closed.value.code == "encoded_leaves_value"


def _architecture_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "runtime_architecture_checks", ROOT / "checks" / "architecture.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_derive_fence_is_green_and_reds_on_a_planted_import(tmp_path: Path) -> None:
    """cr-067 hard guard: the real fence over the real tree, and its red arm."""
    checks = _architecture_module()
    assert checks.fence_worker_derive_free() == []  # type: ignore[attr-defined]

    planted = tmp_path / "worker"
    planted.mkdir()
    shutil.copy(ROOT / "src" / "cozy_runtime" / "internal" / "worker" / "plan.py", planted)
    (planted / "poisoned.py").write_text(
        "def later() -> None:\n    from cozy_runtime.internal.derive import derive  # noqa\n"
    )
    problems = checks.worker_derive_free_violations(planted)  # type: ignore[attr-defined]
    assert len(problems) == 1 and "poisoned.py" in problems[0]


def _delegation(package: str, release: str) -> bytes:
    return canonical.write(
        {
            "format": "cozy.worker.v1.DownloadDelegation/1",
            "packages": [{"package": package, "release": release}],
            "models": [],
        }
    )


def _inventory() -> pb.ImageInventory:
    """An inventory the bare base interpreter agrees with: it pins nothing the venv needs."""

    return pb.ImageInventory(
        profile="test",
        python=".".join(str(part) for part in sys.version_info[:3]),
    )


def _image_python(uv: str, root: Path, runtime_wheel: Path) -> Path:
    """A stand-in for the worker image's python: a real interpreter tree that imports
    cozy-runtime, built once for the test the way an image bake would."""

    image = root / "image"
    subprocess.run(
        [uv, "venv", "--no-project", "--no-config", "--python", sys.executable, str(image)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [uv, "pip", "install", "--python", str(image / "bin" / "python"), str(runtime_wheel)],
        check=True,
        capture_output=True,
    )
    return image / "bin" / "python"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pod_prep_requires_the_request_facts(tmp_path: Path) -> None:
    """Wire 30: the request's release facts are admitted before anything installs.

    Fail-on-parent: before xs-019 the same request carried a staged interface and wheels.
    """

    lock = (
        b"--index-url https://pypi.org/simple\nmarco-polo==1.0.4 --hash=sha256:" + b"a" * 64 + b"\n"
    )

    def prep(
        *,
        application: str = "marco_polo_package:app",
        slots: list[str] | None = None,
        inventory: pb.ImageInventory | None = None,
        locked: bytes = lock,
    ) -> pb.PreparePackageSetResult:
        return prepare_package_set(
            pb.PreparePackageSetRequest(
                download_delegation=_delegation("paul/marco-polo", "1.0.4"),
                environment_root=str(tmp_path / "environment"),
                application=application,
                model_slot_paths=slots or [],
                image_inventory=inventory or _inventory(),
                locked_requirements=locked,
            ),
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=tmp_path / "store",
            environment_root=tmp_path / "environment",
            python=Path(sys.executable),
            describe=lambda installed, distribution: b"{}",
            verified=lambda _digest, _length, _path: None,
        )

    with pytest.raises(PreparationRefusal) as no_application:
        prep(application="")
    assert no_application.value.code == "package_prepare_application_invalid"

    with pytest.raises(PreparationRefusal) as unsorted:
        prep(slots=["b.models.x", "a.models.y"])
    assert unsorted.value.code == "package_prepare_model_slot_paths_invalid"

    with pytest.raises(PreparationRefusal) as no_inventory:
        prep(inventory=pb.ImageInventory())
    assert no_inventory.value.code == "package_prepare_image_inventory_invalid"

    with pytest.raises(PreparationRefusal) as no_lock:
        prep(locked=b"")
    assert no_lock.value.code == "package_prepare_locked_requirements_invalid"

    with pytest.raises(PreparationRefusal) as bare_row:
        prep(locked=b"marco-polo==1.0.4\n")
    assert bare_row.value.code == "package_prepare_locked_requirements_invalid"

    with pytest.raises(PreparationRefusal) as no_pin:
        prep(
            locked=b"--index-url https://pypi.org/simple\nsix==1.17.0 --hash=sha256:"
            + b"b" * 64
            + b"\n"
        )
    assert no_pin.value.code == "package_prepare_project_pin_missing"


class _SimpleIndex:
    """A real PEP 503 index over the exact wheels the test built, served on loopback."""

    def __init__(self, root: Path, wheels: list[Path]) -> None:
        import http.server
        import threading

        for wheel in wheels:
            name = wheel.name.split("-", 1)[0].replace("_", "-").lower()
            project = root / name
            project.mkdir(parents=True, exist_ok=True)
            shutil.copy(wheel, project / wheel.name)
            (project / "index.html").write_text(
                "<html><body>"
                + f'<a href="{wheel.name}#sha256={_sha(wheel)}">{wheel.name}</a>'
                + "</body></html>"
            )
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _closure_rows(uv: str) -> list[str]:
    """The release's real registry closure: uv export over this repo's own uv.lock,
    exactly the derivation the hub performs at grant time."""

    rendered = subprocess.run(
        [
            uv,
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements.txt",
            "--no-progress",
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[str] = []
    pending = ""
    for line in rendered.stdout.splitlines():
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        row = (pending + line).strip()
        pending = ""
        if row and not row.startswith(("#", "-")):
            rows.append(row)
    assert rows, rendered.stdout
    return rows


def test_pod_prepares_from_the_exact_export_and_refuses_a_disagreement() -> None:
    """xs-017/xs-019 end to end, no mocks: real wheels, BOTH indexes, a real hash-pinned
    export, a real uv venv, and a real import of the installed application.

    * the pod materializes its venv from the export alone — PyPI for a registry dependency,
      the org-shaped loopback index for the release's own wheels, every artifact matched
      against a hash the export names;
    * the interface is DERIVED from that install (an actual import in the venv's own
      interpreter), and the request's application fact cross-checks it;
    * a wrong application fact refuses typed; a wrong hash refuses at install, which is the
      dependency-confusion guard doing its job.
    """

    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cozy-published-locked.") as raw:
        root = Path(raw)
        wheel_set = root / "wheels"
        wheel_set.mkdir(parents=True)
        for source in (ROOT, PACKAGE):
            subprocess.run(
                [uv, "build", "--wheel", "--out-dir", str(wheel_set), str(source)],
                check=True,
                capture_output=True,
            )
        runtime_wheel = next(wheel_set.glob("cozy_runtime-*.whl"))
        project_wheel = next(wheel_set.glob("marco_polo-*.whl"))
        release = project_wheel.name.split("-")[1]
        runtime_version = runtime_wheel.name.split("-")[1]
        image_python = _image_python(uv, root, runtime_wheel)
        index = _SimpleIndex(root / "index", [runtime_wheel, project_wheel])
        try:
            rows = sorted(
                [
                    *_closure_rows(uv),
                    f"cozy-runtime=={runtime_version} --hash=sha256:{_sha(runtime_wheel)}",
                    f"marco-polo=={release} --hash=sha256:{_sha(project_wheel)}",
                ]
            )
            export = "\n".join(
                [
                    "--index-url https://pypi.org/simple",
                    f"--extra-index-url {index.url}",
                    *rows,
                    "",
                ]
            ).encode()

            def describe_real(
                selected: package_environment.InstalledEnvironment, distribution: str
            ) -> bytes:
                command = (
                    "from cozy_runtime.internal import package_interface; "
                    "from cozy_runtime.internal.discovery import discover_distribution; "
                    f"found=discover_distribution({distribution!r}); "
                    "print(package_interface.canonical_bytes("
                    "package_interface.build(found)).decode())"
                )
                rendered = subprocess.run(
                    [str(selected.python), "-I", "-c", command],
                    cwd="/",
                    check=False,
                    capture_output=True,
                    text=True,
                )
                assert rendered.returncode == 0, rendered.stderr
                return rendered.stdout.strip().encode()

            def prep(application: str, export_bytes: bytes) -> pb.PreparePackageSetResult:
                return prepare_package_set(
                    pb.PreparePackageSetRequest(
                        download_delegation=_delegation("paul/marco-polo", release),
                        environment_root=str(root / "environments"),
                        application=application,
                        model_slot_paths=[],
                        image_inventory=_inventory(),
                        locked_requirements=export_bytes,
                    ),
                    artifact_cache=root / "artifacts",
                    tensorfs_root=root / "store",
                    environment_root=root / "environments",
                    python=image_python,
                    describe=describe_real,
                    verified=lambda _digest, _length, _path: None,
                    native_environment=sealed_child_environment(),
                )

            (root / "artifacts").mkdir()
            (root / "store").mkdir()
            prepared = prep("marco_polo_package:app", export)
            placement_bytes = prepared.placement_set.placement_set_canonical_bytes
            placement = json.loads(placement_bytes)["placements"][0]
            # The placement carries the lock's identity, never a wheel list.
            assert placement["package"] == {"package": "paul/marco-polo", "release": release}
            assert placement["environment"] == {
                "local_wheels": [],
                "locked_requirements": {
                    "digest": "sha256:" + hashlib.sha256(export).hexdigest(),
                    "length": len(export),
                },
            }
            generation = root / "environments" / "contents"
            venv_pythons = list(generation.glob("*/bin/python"))
            assert len(venv_pythons) == 1
            probe = subprocess.run(
                [str(venv_pythons[0]), "-I", "-c", "import marco_polo_package"],
                capture_output=True,
                text=True,
            )
            assert probe.returncode == 0, probe.stderr
            # The uv cache was operation-scoped derived work and is gone after install.
            assert not list((root / "environments").glob(".uv-cache-*"))
            # The hard guard, observed live: preparing a package never loads derive.
            assert "cozy_runtime.internal.derive" not in sys.modules

            with pytest.raises(PreparationRefusal) as disagreed:
                prep("marco_polo_package:other", export)
            assert disagreed.value.code == "package_prepare_application_disagrees"

            poisoned = export.replace(_sha(project_wheel).encode(), _sha(runtime_wheel).encode())
            with pytest.raises(PreparationRefusal) as mismatched:
                prep("marco_polo_package:app", poisoned)
            assert mismatched.value.code == "package_environment_uv_failed"
        finally:
            index.close()
