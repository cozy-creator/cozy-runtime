"""cr-067/cr-077: the author derives, the pod serves from the published package interface.

Three planes, each driven for real:

* the author surface — capability is the slot's class annotation and NOTHING else;
* the pod preparation plane — the hub-committed manifest + PackageInterface staged beside the
  set are the authoritative wiring, import-time derivation is a byte-compared cross-check,
  and NOTHING on this plane may load the derive machinery (the hard guard, proven red);
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import App, ConformanceError, Context, Model, WeightsOutput
from cozy_runtime.author._describe import describe
from cozy_runtime.author._loader import Loader
from cozy_runtime.internal import (
    canonical,
    package_installation,
    package_interface,
    static_interface,
)
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


def test_lane_words_are_not_selection_leaks() -> None:
    """cr-077: a weights output named after the lane it derives ("bf16") describes fine
    beside a default binding whose lane says the same word; an interface string equal to
    the binding's release remains a refused selection leak."""

    def build(output_id: str, release: str) -> dict[str, object]:
        app = App()

        @app.job(weights=(WeightsOutput(output_id, max_new_bytes=1 << 20),))
        def produce(payload: Request, denoiser: Denoiser, ctx: Context) -> Reply:
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
    """Describe emits no retired slot field; readers tolerate one but not a bad consent."""
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
        # Readers tolerate members a newer or older writer added; they carry no authority.
        package_interface.read_bytes(planted(**legacy))
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


def _download_set(package: str, release: str) -> bytes:
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
    """Wire 30: the request's exact lock and project pin are admitted before anything installs.

    Application and slot facts are advisory; the installed interface is the authority.
    """

    lock = (
        b"--index-url https://pypi.org/simple\nmarco-polo==1.0.4 --hash=sha256:" + b"a" * 64 + b"\n"
    )

    def prep(*, locked: bytes = lock) -> pb.PreparePackageSetResult:
        return prepare_package_set(
            pb.PreparePackageSetRequest(
                download_delegation=_download_set("paul/marco-polo", "1.0.4"),
                install_root=str(tmp_path / "environment"),
                image_inventory=_inventory(),
                locked_requirements=locked,
            ),
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=tmp_path / "store",
            install_root=tmp_path / "environment",
            python=Path(sys.executable),
            verified=lambda _digest, _length, _path: None,
        )

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


def _with_interface(wheel: Path, interface: bytes) -> None:
    """Carry the release interface as installed dist-info metadata, as a published wheel does."""

    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    (record,) = (name for name in members if name.endswith(".dist-info/RECORD"))
    member = record.rsplit("/", 1)[0] + "/package-interface.json"
    members[member] = interface
    encoded = base64.urlsafe_b64encode(hashlib.sha256(interface).digest()).rstrip(b"=")
    members[record] += f"{member},sha256={encoded.decode()},{len(interface)}\n".encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in members.items():
            archive.writestr(name, body)


def test_pod_prepares_from_the_exact_export_and_refuses_a_wrong_hash() -> None:
    """xs-017/xs-019 end to end, no mocks: real wheels, BOTH indexes, a real hash-pinned
    export, a real uv venv, and a real import of the installed application.

    * the pod installs its venv from the export alone — PyPI for a registry dependency,
      the org-shaped loopback index for the release's own wheels, every artifact matched
      against a hash the export names;
    * the interface is the installed release's own metadata: nothing is described and no
      executor is spawned, and the request's advisory copy is not an authority;
    * a wrong hash refuses at install, which is the dependency-confusion guard doing its job.
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
        interface = package_interface.canonical_bytes(static_interface.build(PACKAGE))
        _with_interface(project_wheel, interface)
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

            def prep(
                export_bytes: bytes,
                *,
                install_root: Path | None = None,
                python_version: str = "",
                held: list[package_installation.InstalledEnvironment] | None = None,
            ) -> pb.PreparePackageSetResult:
                target = install_root or root / "environments"
                return prepare_package_set(
                    pb.PreparePackageSetRequest(
                        download_delegation=_download_set("paul/marco-polo", release),
                        install_root=str(target),
                        image_inventory=_inventory(),
                        locked_requirements=export_bytes,
                        python_requires=">=3.12" if python_version else "",
                        python_version=python_version,
                        package_interface=b"{}",
                    ),
                    artifact_cache=root / "artifacts",
                    tensorfs_root=root / "store",
                    install_root=target,
                    python=image_python,
                    verified=lambda _digest, _length, _path: None,
                    materialized=held.append if held is not None else None,
                )

            (root / "artifacts").mkdir()
            (root / "store").mkdir()
            held: list[package_installation.InstalledEnvironment] = []
            loaded = set(sys.modules)
            prepared = prep(export, held=held)
            placement = json.loads(prepared.placement_set.placement_set_canonical_bytes)[
                "placements"
            ][0]
            assert placement["package"] == {"package": "paul/marco-polo", "release": release}
            assert placement["installation_id"] == held[0].installation_id
            assert base64.b64decode(placement["package_interface"]) == interface
            venv_pythons = list((root / "environments/installations").glob("*/venv/bin/python"))
            assert venv_pythons == [held[0].python]
            probe = subprocess.run(
                [str(held[0].python), "-I", "-c", "import marco_polo_package"],
                capture_output=True,
                text=True,
            )
            assert probe.returncode == 0, probe.stderr
            # The hard guard, observed live: preparing a package never loads derive. Measured
            # across the prepare, since another in-process test may have loaded it already.
            assert "cozy_runtime.internal.derive" not in set(sys.modules) - loaded
            staged = root / "artifacts" / hashlib.sha256(interface).hexdigest()
            assert staged.read_bytes() == interface

            # A local controller can run a different CPython than the release.
            # Its preinstalled SDK must never supply this published wheel closure.
            from cozy_runtime.internal import python_interpreters

            for selected in python_interpreters.available():
                local: list[package_installation.InstalledEnvironment] = []
                prep(
                    export,
                    install_root=root / f"local-{selected.version}",
                    python_version=selected.version,
                    held=local,
                )
                assert python_interpreters.probe(local[0].python).version == selected.version
                assert local[0].site_packages.is_relative_to(local[0].generation)

            poisoned = export.replace(_sha(project_wheel).encode(), _sha(runtime_wheel).encode())
            with pytest.raises(PreparationRefusal) as mismatched:
                prep(poisoned)
            assert mismatched.value.code == "package_installation_uv_failed"
        finally:
            index.close()
