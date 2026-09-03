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
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import App, ConformanceError, Model, WeightsOutput, WeightsSink
from cozy_runtime.author._describe import describe
from cozy_runtime.author._loader import Loader
from cozy_runtime.internal import canonical, package_environment, package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    prepare_package,
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


def test_pod_prep_requires_the_exact_published_interface(tmp_path: Path) -> None:
    """The set path refuses without the hub-committed PackageInterface beside the files.

    Fail-on-parent: before cr-067 the same request proceeded straight to derivation.
    """
    artifact_cache = tmp_path / "artifacts"
    view = artifact_cache / "sets" / "package"
    view.mkdir(parents=True)
    wheel = view / "thing.whl"
    wheel.write_bytes(b"not really a wheel")
    row = pb.LocalDownloadFile(
        digest=hashlib.sha256(wheel.read_bytes()).digest(),
        filename=wheel.name,
        kind=pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL,
        length=wheel.stat().st_size,
        path=str(wheel),
    )

    interface_bytes = package_interface.canonical_bytes(
        {"format": package_interface.SCHEMA, "application": "a:b", "entrypoints": [], "jobs": []}
    )
    interface_digest = "sha256:" + hashlib.sha256(interface_bytes).hexdigest()

    interface_path = view / package_interface.FILENAME
    interface_row = pb.LocalDownloadFile(
        digest=bytes.fromhex(interface_digest[7:]),
        filename=package_interface.FILENAME,
        kind=pb.LOCAL_DOWNLOAD_KIND_PACKAGE_INTERFACE,
        length=len(interface_bytes),
        path=str(interface_path),
    )
    files: list[pb.LocalDownloadFile] = [row, interface_row]
    files.sort(key=lambda value: value.digest)

    def prep() -> pb.PreparePackageSetResult:
        return prepare_package_set(
            pb.PreparePackageSetRequest(
                download_delegation=_delegation("paul/thing", "1.0.0"),
                files=files,
                environment_root=str(tmp_path / "environment"),
            ),
            artifact_cache=artifact_cache,
            artifact_store=tmp_path / "store",
            environment_root=tmp_path / "environment",
            python=Path(sys.executable),
            describe=lambda installed, distribution: b"{}",
            verified=lambda _digest, _length, _path: None,
        )

    with pytest.raises(PreparationRefusal) as absent:
        prep()
    assert absent.value.code == "package_prepare_published_interface_invalid"

    # Wrong bytes at the conventional ordinary-file path cannot satisfy the delegated ref.
    interface_path.parent.mkdir()
    interface_path.write_bytes(b"other bytes entirely")
    with pytest.raises(PreparationRefusal) as mismatched:
        prep()
    assert mismatched.value.code == "package_prepare_published_interface_invalid"


def test_pod_serves_the_published_interface_and_refuses_a_disagreement() -> None:
    """The core authority reversal, over a real environment and real wheels.

    Agree: the placement's interface ref is the PUBLISHED document's digest, and the
    derive machinery never loads on this plane. Disagree: a typed refusal, never a serve
    (the parent served from the derived surface in both arms).
    """
    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cozy-published-package_interface.") as raw:
        root = Path(raw)
        artifact_cache = root / "artifacts"
        wheel_set = artifact_cache / "sets" / "package"
        wheel_set.mkdir(parents=True)
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(wheel_set), str(ROOT)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(wheel_set), str(PACKAGE)],
            check=True,
            capture_output=True,
        )
        runtime_wheel = next(wheel_set.glob("cozy_runtime-*.whl"))
        project_wheel = next(wheel_set.glob("marco_polo-*.whl"))
        environment = root / "environment"
        python = environment / "bin" / "python"
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(environment)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), str(runtime_wheel), str(project_wheel)],
            check=True,
            capture_output=True,
        )
        content_digest = "sha256:" + "17" * 32
        installed = package_environment.open_existing_environment(python, content_digest)
        (root / "store").mkdir()

        def describe_real(
            selected: package_environment.InstalledEnvironment, distribution: str
        ) -> bytes:
            command = (
                "from cozy_runtime.internal import package_interface; "
                "from cozy_runtime.internal.discovery import discover_distribution; "
                f"found=discover_distribution({distribution!r}); "
                "print(package_interface.canonical_bytes(package_interface.build(found)).decode())"
            )
            rendered = subprocess.run(
                [str(selected.python), "-I", "-c", command],
                cwd="/",
                check=True,
                capture_output=True,
                text=True,
            )
            return rendered.stdout.strip().encode()

        derived = describe_real(installed, "marco-polo")

        def rows() -> list[pb.LocalDownloadFile]:
            out = []
            for kind, path in (
                (pb.LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL, project_wheel),
                (pb.LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL, runtime_wheel),
            ):
                data = path.read_bytes()
                out.append(
                    pb.LocalDownloadFile(
                        digest=hashlib.sha256(data).digest(),
                        filename=path.name,
                        kind=kind,
                        length=len(data),
                        path=str(path),
                    )
                )
            out.sort(key=lambda row: row.digest)
            return out

        def prep(published: bytes) -> pb.PreparePackageSetResult:
            return prepare_package(
                package_name="paul/marco-polo",
                release="1.0.4",
                files=rows(),
                placement_seed=b"published-interface-authority",
                artifact_cache=artifact_cache,
                artifact_store=root / "store",
                environment_root=environment,
                python=python,
                describe=describe_real,
                verified=lambda _digest, _length, _path: None,
                installed_environment=installed,
                published_interface=published,
            )

        prepared = prep(derived)
        placement_bytes = prepared.placement_set.placement_set_canonical_bytes
        placement = json.loads(placement_bytes)["placements"][0]
        assert placement["package_interface"]["digest"] == (
            "sha256:" + hashlib.sha256(derived).hexdigest()
        )
        # The hard guard, observed live: preparing a package never loads derive.
        assert "cozy_runtime.internal.derive" not in sys.modules

        disagreeing = json.loads(derived)
        disagreeing["application"] = "marco_polo_package:other"
        with pytest.raises(PreparationRefusal) as disagreed:
            prep(canonical.write(disagreeing))
        assert disagreed.value.code == "package_prepare_interface_disagrees"
        assert "published sha256:" in disagreed.value.detail
