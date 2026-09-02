"""cr-067/cr-077: the author derives, the pod serves from the PUBLISHED descriptor.

Three planes, each driven for real:

* the author surface — capability is the slot's class annotation and NOTHING else: the
  retired `source_profiles` door refuses typed, and the descriptor slot grammar no
  longer admits the field (hard cut, cr-077);
* the pod preparation plane — the hub-committed manifest + descriptor staged beside the
  set are the authoritative wiring, import-time derivation is a byte-compared cross-check,
  and NOTHING on this plane may load the derive machinery (the hard guard, proven red);
* the slot-facts writer — `cozy-model-contract-proof`'s author-mode request grammar binds
  the release VERSION and refuses the retired release-digest circularity.
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

from cozy_runtime.author import App, ConformanceError, Model
from cozy_runtime.author._describe import describe
from cozy_runtime.author._loader import Loader
from cozy_runtime.internal import canonical, descriptor, package_environment
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker.package_prepare import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    PUBLISHED_DESCRIPTOR_FILENAME,
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
    return descriptor.build(found)


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


def test_descriptor_slot_grammar_refuses_the_retired_field() -> None:
    """A stale descriptor still carrying source_profile refuses at read (closed set)."""
    app = App()

    @app.entrypoint
    def generate(payload: Request, denoiser: Denoiser) -> Reply:
        return Reply(text="ok")

    body = json.loads(descriptor.canonical_bytes(_described(app)))
    assert "source_profile" not in body["entrypoints"][0]["models"][0]
    body["entrypoints"][0]["models"][0]["source_profile"] = "acme/sd-turbo/1.0.0/bf16"
    with pytest.raises(descriptor.StalePackageDescriptor):
        descriptor.read_bytes(canonical.write(body))


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


def _delegation(package: str, release: str, release_digest: str) -> bytes:
    return canonical.write(
        {
            "format": "cozy.worker.v1.DownloadDelegation/1",
            "packages": [
                {"package": package, "release": release, "release_digest": release_digest}
            ],
            "models": [],
        }
    )


def test_pod_prep_requires_the_published_manifest_and_descriptor(tmp_path: Path) -> None:
    """The set path refuses without the hub-committed documents staged beside the files.

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

    def prep(release_digest: str) -> pb.PreparePackageSetResult:
        return prepare_package_set(
            pb.PreparePackageSetRequest(
                download_delegation=_delegation("paul/thing", "1.0.0", release_digest),
                files=[row],
                environment_root=str(tmp_path / "environment"),
            ),
            artifact_cache=artifact_cache,
            artifact_store=tmp_path / "store",
            environment_root=tmp_path / "environment",
            python=Path(sys.executable),
            describe=lambda installed, distribution: b"{}",
            verified=lambda _digest, _length, _path: None,
        )

    release_digest = "sha256:" + "11" * 32
    with pytest.raises(PreparationRefusal) as absent:
        prep(release_digest)
    assert absent.value.code == "package_prepare_manifest_missing"

    # Wrong bytes at the conventional name: the manifest must HASH to the delegated digest.
    (view / MANIFEST_FILENAME).write_bytes(b'{"format":"nope"}')
    with pytest.raises(PreparationRefusal) as unhashed:
        prep(release_digest)
    assert unhashed.value.code == "package_prepare_manifest_invalid"

    # An honest manifest whose descriptor ref nothing satisfies.
    descriptor_bytes = descriptor.canonical_bytes(
        {"format": descriptor.SCHEMA, "application": "a:b", "entrypoints": [], "jobs": []}
    )
    manifest = canonical.write(
        {
            "format": MANIFEST_FORMAT,
            "package": "paul/thing",
            "release": "1.0.0",
            "package_descriptor": {
                "digest": "sha256:" + hashlib.sha256(descriptor_bytes).hexdigest(),
                "length": len(descriptor_bytes),
            },
        }
    )
    (view / MANIFEST_FILENAME).write_bytes(manifest)
    trusted = "sha256:" + hashlib.sha256(manifest).hexdigest()
    with pytest.raises(PreparationRefusal) as undelivered:
        prep(trusted)
    assert undelivered.value.code == "package_prepare_published_descriptor_invalid"

    (view / PUBLISHED_DESCRIPTOR_FILENAME).write_bytes(b"other bytes entirely")
    with pytest.raises(PreparationRefusal) as mismatched:
        prep(trusted)
    assert mismatched.value.code == "package_prepare_published_descriptor_invalid"


def test_pod_serves_the_published_descriptor_and_refuses_a_disagreement() -> None:
    """The core authority reversal, over a real environment and real wheels.

    Agree: the placement's descriptor ref is the PUBLISHED document's digest, and the
    derive machinery never loads on this plane. Disagree: a typed refusal, never a serve
    (the parent served from the derived surface in both arms).
    """
    uv = shutil.which("uv")
    assert uv is not None
    with tempfile.TemporaryDirectory(prefix="cozy-published-descriptor.") as raw:
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
        release_digest = "sha256:" + "17" * 32
        installed = package_environment.open_existing_environment(python, release_digest)
        (root / "store").mkdir()

        def describe_real(
            selected: package_environment.InstalledEnvironment, distribution: str
        ) -> bytes:
            command = (
                "from cozy_runtime.internal import descriptor; "
                "from cozy_runtime.internal.discovery import discover_distribution; "
                f"found=discover_distribution({distribution!r}); "
                "print(descriptor.canonical_bytes(descriptor.build(found)).decode())"
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
                release_digest=release_digest,
                files=rows(),
                placement_seed=b"published-descriptor-authority",
                artifact_cache=artifact_cache,
                artifact_store=root / "store",
                environment_root=environment,
                python=python,
                describe=describe_real,
                verified=lambda _digest, _length, _path: None,
                installed_environment=installed,
                published_descriptor=published,
            )

        prepared = prep(derived)
        placement_bytes = prepared.placement_set.placement_set_canonical_bytes
        placement = json.loads(placement_bytes)["placements"][0]
        assert placement["package_descriptor"]["digest"] == (
            "sha256:" + hashlib.sha256(derived).hexdigest()
        )
        # The hard guard, observed live: preparing a package never loads derive.
        assert "cozy_runtime.internal.derive" not in sys.modules

        disagreeing = json.loads(derived)
        disagreeing["application"] = "marco_polo_package:other"
        with pytest.raises(PreparationRefusal) as disagreed:
            prep(canonical.write(disagreeing))
        assert disagreed.value.code == "package_prepare_descriptor_disagrees"
        assert "published sha256:" in disagreed.value.detail


def test_author_evidence_request_binds_the_release_version(tmp_path: Path) -> None:
    """The evidence request is closed on {package, project_wheel, release}: the digest
    circularity is structurally unrepresentable (the digest exists only after finalize)."""
    from cozy_runtime.cli import model_contract_proof as proof

    descriptor_bytes = descriptor.canonical_bytes(
        {"format": descriptor.SCHEMA, "application": "a:b", "entrypoints": [], "jobs": []}
    )
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_bytes(descriptor_bytes)

    def request(package: dict[str, object]) -> Path:
        body = {
            "slots": [
                {
                    "class": "Denoiser",
                    "config": {
                        "digest": "sha256:" + "ab" * 32,
                        "length": 2,
                        "path": str(descriptor_path),
                    },
                    "hardware_variants": ["cpu"],
                    "snapshot": "sha256:" + "cd" * 32,
                }
            ],
            "package": package,
            "package_descriptor": {
                "digest": "sha256:" + hashlib.sha256(descriptor_bytes).hexdigest(),
                "length": len(descriptor_bytes),
                "path": str(descriptor_path),
            },
            "project": str(PACKAGE),
        }
        path = tmp_path / "request.json"
        path.write_bytes(canonical.write(body))
        return path

    wheel = {
        "distribution": "thing",
        "version": "1.0.0",
        "ref": {"digest": "sha256:" + "ef" * 32, "length": 5},
    }
    stale = request(
        {
            "package": "paul/thing",
            "release": "1.0.0",
            "release_digest": "sha256:" + "99" * 32,
            "project_wheel": wheel,
        }
    )
    assert proof.main([str(stale), str(tmp_path / "out-a")]) == 3

    versioned = proof._read_request(
        request({"package": "paul/thing", "release": "1.0.0", "project_wheel": wheel})
    )
    assert versioned.release_identity == "paul/thing@1.0.0"
