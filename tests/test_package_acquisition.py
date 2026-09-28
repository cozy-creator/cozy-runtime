"""PackageInterface admission at the worker's real acquisition boundary."""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal import (
    base_observation,
    canonical,
    derive_child,
    package_environment,
    package_installation,
    package_interface,
    static_interface,
)
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker import package_prepare
from cozy_runtime.internal.worker.acquire import Acquirer, AcquisitionRefusal
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.session import read_placement_set
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def _schema() -> dict[str, Json]:
    return {"fields": []}


def _entrypoint(name: str) -> dict[str, Json]:
    return {"name": name, "request": _schema(), "result": _schema()}


def _job(name: str) -> dict[str, Json]:
    return {**_entrypoint(name), "publishes": False}


def _interface(entrypoints: list[Json]) -> dict[str, Json]:
    return {
        "application": "job_only:app",
        "entrypoints": entrypoints,
        "format": package_interface.SCHEMA,
        "jobs": [_job("produce")],
    }


def _bindings(
    tmp_path: Path,
    body: dict[str, Json],
    selected: tuple[str, ...] = (),
    *,
    development: bool = False,
    models: tuple[pb.Model, ...] = (),
    prepared: pb.Placement | None = None,
) -> tuple[Mapping[str, DeclaredBinding], int]:
    raw = canonical.write(body)
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("environment", "selection", "store"):
        (tmp_path / name).mkdir()
    acquirer = Acquirer(
        cache_root=cache,
        install_root=tmp_path / "environment",
        selection_root=tmp_path / "selection",
        tensorfs_root=tmp_path / "store",
        python=Path("/usr/bin/python3"),
        base=base_observation.current(),
    )
    placement = pb.Placement(
        placement_id="package-job-only",
        installation_id="local-1" if development else "install-1",
        package_interface=raw,
        bindings_digest=documents.digest_of(canonical.write({"entrypoints": [], "models": []})),
        models=models,
        entrypoints=[
            pb.Entrypoint(
                name=name,
                entrypoint_binding_digest=documents.digest_of(
                    canonical.write({"entrypoint": name})
                ),
            )
            for name in selected
        ],
    )
    if development:
        placement.development.CopyFrom(
            pb.DevelopmentPackage(
                package="local/job-only", release="0.0.0", installation_id="local-1"
            )
        )
    else:
        placement.package.CopyFrom(pb.PackageSelection(package="proof/job-only", release="1.0.0"))
    return acquirer._bindings(prepared or placement)


@pytest.mark.parametrize("development", [False, True])
def test_job_only_package_set_has_no_serving_bindings(tmp_path: Path, development: bool) -> None:
    bindings, documents = _bindings(tmp_path, _interface([]), development=development)

    assert bindings == {}
    assert documents == 1


@pytest.mark.parametrize(
    "entrypoints",
    [
        [_entrypoint("serve"), _entrypoint("serve")],
        [{"request": _schema(), "result": _schema()}],
    ],
    ids=["duplicate", "malformed"],
)
def test_package_set_refuses_invalid_interface_entrypoints(
    tmp_path: Path, entrypoints: list[Json]
) -> None:
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(tmp_path, _interface(entrypoints))

    assert refused.value.code == "package_interface_invalid"


@pytest.mark.parametrize("selected", [("first",), ("second",), ("first", "second")])
def test_package_set_admits_selected_interface_entrypoints(
    tmp_path: Path, selected: tuple[str, ...]
) -> None:
    bindings, count = _bindings(
        tmp_path, _interface([_entrypoint("first"), _entrypoint("second")]), selected
    )
    assert {binding.entrypoint for binding in bindings.values()} == set(selected)
    assert count == 1


@pytest.mark.parametrize(
    ("selected", "code"),
    [((), "binding_entrypoint_set_mismatch"), (("absent",), "binding_package_interface_invalid")],
)
@pytest.mark.parametrize("development", [False, True])
def test_package_set_refuses_empty_or_unknown_selection(
    tmp_path: Path, selected: tuple[str, ...], code: str, development: bool
) -> None:
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(
            tmp_path,
            _interface([_entrypoint("first"), _entrypoint("second")]),
            selected,
            development=development,
        )
    assert refused.value.code == code


def test_h3_prepared_code_only_set_has_no_executable_bindings(tmp_path: Path) -> None:
    interface = (
        Path(__file__).parent / "testdata/h3-code-only/package-interface.json"
    ).read_bytes()
    body = json.loads(interface)
    assert body["entrypoints"] and all(row["models"] for row in body["entrypoints"])
    placement = pb.Placement(
        placement_id="package-h3",
        development=pb.DevelopmentPackage(
            package="local/h3", release="0.0.0", installation_id="local-h3"
        ),
        installation_id="local-h3",
        package_interface=interface,
        bindings_digest=documents.digest_of(canonical.write({"entrypoints": [], "models": []})),
    )
    bindings, count = _bindings(tmp_path, body, prepared=placement)
    assert bindings == {} and count == 1


@pytest.mark.parametrize("development", [False, True])
def test_empty_selection_cannot_drop_declared_weightless_binding(
    tmp_path: Path, development: bool
) -> None:
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(tmp_path, _interface([_entrypoint("serve")]), development=development)
    assert refused.value.code == "binding_entrypoint_set_mismatch"


@pytest.mark.parametrize("development", [False, True])
def test_empty_selection_cannot_hide_selected_models(tmp_path: Path, development: bool) -> None:
    modeled = {
        **_entrypoint("serve"),
        "models": [{"class": "Model", "path": "serve.models.model", "component_use": {}}],
    }
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(
            tmp_path,
            _interface([modeled]),
            development=development,
            models=(pb.Model(id="selected"),),
        )
    assert refused.value.code == "binding_entrypoint_set_mismatch"


@pytest.mark.parametrize("development", [False, True])
def test_modeled_code_only_selection_has_no_execution_authority(
    tmp_path: Path, development: bool
) -> None:
    modeled = {
        **_entrypoint("serve"),
        "models": [{"class": "Model", "path": "serve.models.model", "component_use": {}}],
    }
    bindings, count = _bindings(tmp_path, _interface([modeled]), development=development)
    assert bindings == {} and count == 1


def test_code_only_exception_cannot_hide_missing_model_slots(tmp_path: Path) -> None:
    modeled = {
        **_entrypoint("serve"),
        "models": [{"class": "Model", "path": "serve.models.model", "component_use": {}}],
    }
    with pytest.raises(AcquisitionRefusal) as refused:
        _bindings(tmp_path, _interface([modeled]), ("serve",), development=True)
    assert refused.value.code == "binding_slot_set_mismatch"


# --- proto-061 C: interface from the wire, census once per content, stable ids, one lease ---

MODELED_SOURCE = """import msgspec
from cozy_runtime.author import App, Config, Context, Loader, Model, uses_components
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    value: int
def scalar(width: int):
    import torch
    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(width, dtype=torch.float32))
    return Scalar()
class Pipeline:
    def __init__(self, config: Config):
        self.components = {"unet": scalar(1)}
class Wide:
    def __init__(self, config: Config):
        self.components = {"unet": scalar(2)}
class Weights(Model[Pipeline]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=Pipeline)
    @uses_components("unet")
    def measure(self) -> int:
        raise AssertionError("preparation cannot execute a handler")
class Tokenized:
    def __init__(self, config: Config, vocab: bytes):
        self.vocab = vocab
        self.components = {"unet": scalar(1)}
class TokenizedWeights(Model[Tokenized]):
    def load(self, loader: Loader) -> None:
        assert loader.assets.names() == ("tokenizer/vocab.txt",)
        vocab = loader.assets.read("tokenizer/vocab.txt")
        assert loader.assets.open("tokenizer/vocab.txt").read() == vocab
        assert not hasattr(loader.assets, "materialized")
        self.pipe = loader.construct(Tokenized, factory=lambda config: Tokenized(config, vocab))
    @uses_components("unet")
    def measure(self) -> int:
        raise AssertionError("preparation cannot execute a handler")
class WideWeights(Model[Wide]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Wide, factory=Wide)
    @uses_components("unet")
    def measure(self) -> int:
        raise AssertionError("preparation cannot execute a handler")
app = App()
@app.entrypoint
def generate(ctx: Context, payload: Request, model: Weights) -> Result:
    raise AssertionError("preparation cannot execute a handler")
@app.entrypoint
def turbo(ctx: Context, payload: Request, base_model: Weights) -> Result:
    raise AssertionError("preparation cannot execute a handler")
@app.entrypoint
def tokenized(ctx: Context, payload: Request, model: TokenizedWeights) -> Result:
    raise AssertionError("preparation cannot execute a handler")
@app.entrypoint
def wide(ctx: Context, payload: Request, model: WideWeights) -> Result:
    raise AssertionError("preparation cannot execute a handler")
"""
MODELED = "proof/modeled"
GENERATE = "generate.models.model"
TURBO = "turbo.models.base_model"
WIDE = "wide.models.model"
TOKENIZED = "tokenized.models.model"


@dataclass(frozen=True)
class ModeledRelease:
    installed: package_installation.InstalledEnvironment
    interface: bytes
    store: Path
    manifest: str
    length: int


@pytest.fixture(scope="module")
def modeled(tmp_path_factory: pytest.TempPathFactory) -> ModeledRelease:
    """A real installed environment (runtime wheel, CPU torch, the package) and a real Store."""

    from conftest import image_python
    from test_end_to_end import _run, _write_pure_project
    from test_model_runtime_closure import _snapshot

    root = tmp_path_factory.mktemp("modeled")
    project = root / "project"
    _write_pure_project(project, "modeled", "1.0.0", "modeled_package", ("msgspec",))
    metadata = project / "pyproject.toml"
    metadata.write_text(
        metadata.read_text()
        + '\n[project.entry-points."cozy.application"]\ndefault="modeled_package:app"\n'
    )
    (project / "package.toml").write_text('[application]\nobject="modeled_package:app"\n')
    (project / "src/modeled_package/__init__.py").write_text(MODELED_SOURCE)
    image = image_python()
    environment = root / "environment"
    _run("uv", "venv", "--python", str(image), str(environment))
    python = environment / "bin" / "python"
    _run(
        "uv", "pip", "install", "--python", str(python),
        "--index", "https://download.pytorch.org/whl/cpu", "torch==2.14.0",
    )  # fmt: skip
    runtime = next(image.parent.parent.parent.glob("wheels/cozy_runtime-*.whl"))
    _run("uv", "pip", "install", "--python", str(python), str(runtime), str(project))
    store, manifest, length, _held = _snapshot(root, include_asset=True, checkpoint_only=True)
    return ModeledRelease(
        installed=package_installation.open_existing_environment(python, "modeled-environment"),
        interface=package_interface.canonical_bytes(static_interface.build(project)),
        store=store,
        manifest=manifest,
        length=length,
    )


def _prepare_modeled(release: ModeledRelease, root: Path, slots: tuple[str, ...]) -> pb.Placement:
    result = package_prepare._prepare_published(
        package_name=MODELED,
        release="1.0.0",
        locked=package_environment.read_locked_requirements(
            b"--index-url https://pypi.org/simple\nmodeled==1.0.0 --hash=sha256:"
            + b"a" * 64
            + b"\n"
        ),
        models=package_prepare.selections(
            {
                "package": MODELED,
                "slot": slot,
                "model": "example/model",
                "manifest": release.manifest,
            }
            for slot in slots
        ),
        artifact_cache=root / "artifacts",
        tensorfs_root=release.store,
        install_root=root / "install",
        python=release.installed.python,
        interface=lambda _installed, _distribution: release.interface,
        verified=lambda _digest, _length, _path: None,
        job_plan_root=None,
        base=None,
        installed_environment=release.installed,
    )
    (placement,) = read_placement_set(result.placement_set)
    return placement


def test_census_runs_once_per_content_and_ids_stay_stable_as_the_set_grows(
    modeled: ModeledRelease, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived: list[tuple[str, ...]] = []
    real = derive_child.derive_in

    def counted(
        python: Path, request: derive_child.DeriveRequest
    ) -> tuple[derive_child.TensorRequirements | derive_child.SourceSlot, ...]:
        derived.append(tuple(row.slot for row in request.slots))
        return real(python, request)

    monkeypatch.setattr(derive_child, "derive_in", counted)

    first = _prepare_modeled(modeled, tmp_path, (GENERATE,))
    assert derived == [(GENERATE,)]
    census = sorted((tmp_path / "artifacts" / "census").glob("*.json"))
    assert len(census) == 1
    assert stat.S_IMODE(census[0].stat().st_mode) == 0o444
    assert json.loads(census[0].read_bytes())["components"] == ["unet"]

    # The set grows by one construction: only the new slot is derived.
    grown = _prepare_modeled(modeled, tmp_path, (GENERATE, TURBO))
    assert derived == [(GENERATE,), (TURBO,)]
    assert len(list((tmp_path / "artifacts" / "census").glob("*.json"))) == 2

    # A repeat derives nothing.
    again = _prepare_modeled(modeled, tmp_path, (TURBO, GENERATE))
    assert derived == [(GENERATE,), (TURBO,)]
    assert again == grown

    # Stable identity: the placement id and generate's binding digest survive the growth.
    assert (
        first.placement_id == grown.placement_id == package_prepare.placement_id(MODELED, "1.0.0")
    )
    digest = {row.name: row.entrypoint_binding_digest for row in grown.entrypoints}
    (before,) = first.entrypoints
    assert digest["generate"] == before.entrypoint_binding_digest
    assert {row.id for row in grown.models} == {
        package_prepare.model_id(GENERATE),
        package_prepare.model_id(TURBO),
    }

    # A construction the checkpoint does not fit refuses typed, from a cached verdict too.
    for _ in range(2):
        with pytest.raises(package_prepare.PreparationRefusal) as refused:
            _prepare_modeled(modeled, tmp_path, (WIDE,))
        assert refused.value.code == "checkpoint_incompatible"
        assert "shape_mismatch" in refused.value.detail
    assert derived == [(GENERATE,), (TURBO,), (WIDE,)]


def test_a_census_miss_loads_checkpoint_assets_in_the_fenced_child_without_a_write(
    modeled: ModeledRelease, tmp_path: Path
) -> None:
    """sdxl's tokenizer path: `load` builds from checkpoint asset bytes inside the derive child,
    whose sandbox refuses every filesystem write, and the census still lands."""

    body = _prepare_modeled(modeled, tmp_path, (TOKENIZED,))
    assert {row.id for row in body.models} == {package_prepare.model_id(TOKENIZED)}
    (census,) = (tmp_path / "artifacts" / "census").glob("*.json")
    assert json.loads(census.read_bytes())["components"] == ["unet"]


@pytest.mark.skipif(
    not hasattr(tensorfs, "stats"), reason="tensorfs.stats() is not exposed by this TensorFS"
)
def test_a_warm_prepare_and_acquire_walk_each_manifest_once(
    modeled: ModeledRelease, tmp_path: Path
) -> None:
    cold = tensorfs.stats()
    body = _prepare_modeled(modeled, tmp_path, (GENERATE, TURBO))
    before = tensorfs.stats()
    body = _prepare_modeled(modeled, tmp_path, (GENERATE, TURBO))
    prepared = tensorfs.stats()
    acquirer = Acquirer(
        cache_root=tmp_path / "artifacts",
        install_root=tmp_path / "install",
        selection_root=tmp_path / "selection",
        tensorfs_root=modeled.store,
        python=modeled.installed.python,
        base=base_observation.current(),
        installed_environment=modeled.installed,
    )
    acquired = acquirer.acquire(body)
    after = tensorfs.stats()
    assert acquired.manifests == 1 and len(acquired.bindings) == 2
    # A census hit takes no closure lease; acquisition walks the one distinct manifest once.
    assert prepared["presence_passes"] == before["presence_passes"]
    assert after["presence_passes"] - prepared["presence_passes"] == 1
    # One catalog connection per process per root: at most the cold prepare opens it (an
    # earlier test may already have), and the warm prepare and acquisition reuse it.
    assert after["catalog_opens"] == before["catalog_opens"]
    assert after["catalog_opens"] - cold["catalog_opens"] <= 1
