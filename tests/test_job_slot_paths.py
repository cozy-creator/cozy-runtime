"""RUN 193, AS A TEST. A package interface declares model slots in TWO collections.

`cozy.package.interface/1` carries `entrypoints` and `jobs`, and both declare
`models[].path`. Three readers of that document read only the first:

* the hub's facts assembler, so a release whose callables are all jobs shipped
  `model_slot_paths: []` and the pod refused every download set that bound a job slot
  (paul/minimax-h3-tools is exactly that shape; its four-lane binds two of nine);
* `_prepare_published`'s `declared` cross-check, which would then have refused
  `package_prepare_model_slots_disagree` the moment the hub started telling the truth --
  paul/anima, whose `generate` is the only callable anyone had exercised, breaks here
  first;
* `_entrypoints`' unused-selection check, which insisted every selected model be bound
  into an Entrypoint. A job has no placement and no residency: its Model parameter is an
  attempt-held, derive-only view of one TensorFS Manifest, so a job's selection is
  legitimately never bound here.

The matrix is both shapes on purpose. A test that exercised only an entrypoint is how all
three shipped.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal import package_environment, package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    _entrypoints,
    _prepare_published,
    _request_facts,
    _slot_paths,
)
from cozy_runtime.protocol import worker_pb2 as pb

ROOT = Path(__file__).resolve().parent.parent
TENANT = ROOT / "corpus" / "tenant"

PACKAGE = "cozytest/tenant"
RELEASE = "1.0.0"
LANE = "bf16"
ENTRYPOINT_SLOT = "touch.models.model"
RETOUCH_SLOT = "retouch.models.model"
JOB_SLOT = "prune.models.source"

# One model-bound job beside the corpus tenant's `touch` entrypoint: the two shapes in one
# release, which is paul/anima's shape and the one no test had.
JOB = '''

class PruneRequest(msgspec.Struct, forbid_unknown_fields=True):
    factor: int = 2


class PruneResult(msgspec.Struct):
    kept: int


@app.job
def prune(ctx: Context, payload: PruneRequest, source: TenantModel, tel: Telemetry) -> PruneResult:
    """A model-bound job: `source` is a declared slot whose artifact binds per invocation."""
    return PruneResult(kept=payload.factor)
'''

# The canonical CozyTensors header TensorFS itself produced: one inline JCS config, one
# inline f32 tensor, one six-byte asset. Fixed bytes keep this a border test.
_HEADER = base64.b64decode(
    "hW1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rv"
    "a2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0"
    "L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJC"
    "oCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1bm"
    "V0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAAA="
)
_ASSET = b"vocab\n"


def _ref(raw: bytes) -> dict[str, Any]:
    return {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture(scope="module")
def interface(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The REAL derived interface of a release carrying one entrypoint and one job, each
    with a model slot -- built by the author surface, not written by hand."""

    project = tmp_path_factory.mktemp("tenant-job") / "tenant_job"
    project.mkdir()
    (project / "tenant_job.py").write_text((TENANT / "tiny_tenant.py").read_text() + JOB)
    (project / "package.toml").write_text(
        (TENANT / "package.toml").read_text().replace("tiny_tenant:app", "tenant_job:app")
    )
    return package_interface.build(discover(project))


@pytest.fixture(scope="module")
def released(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, int]:
    """A real TensorFS Store holding one released model the selections can resolve."""

    root = tmp_path_factory.mktemp("store")
    store_root = root / "store"
    store = tensorfs.Store.init(str(store_root))

    header_path = root / "header.cbor"
    header_path.write_bytes(_HEADER)
    header = _ref(_HEADER)
    store.put_file(str(header_path), "sha256:" + header["sha256"], header["length"])

    asset_path = root / "vocab.txt"
    asset_path.write_bytes(_ASSET)
    asset = _ref(_ASSET)
    store.put_file(str(asset_path), "sha256:" + asset["sha256"], asset["length"])

    raw = json.dumps(
        {"entries": [{"blob": header, "kind": "cozytensors", "path": "model.cozytensors"}]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest = "sha256:" + hashlib.sha256(raw).hexdigest()
    store.put_manifest(raw, manifest, len(raw))
    operation = store.begin_operation("job-slot-paths", *PACKAGE.split("/"))
    operation.hold_manifest(manifest, len(raw))
    operation.commit_release(None, RELEASE, LANE, manifest, len(raw))
    return store_root, manifest, len(raw)


def _selection(slot: str, manifest: str) -> dict[str, Any]:
    """One DownloadDelegation model row: exactly the six names the hub signs."""

    return {
        "lane": LANE,
        "manifest": manifest,
        "model": PACKAGE,
        "package": PACKAGE,
        "release": RELEASE,
        "slot": slot,
    }


def _installed(tmp_path: Path) -> package_environment.InstalledEnvironment:
    """A metadata-backed tenant overlay using the real test interpreter and SDK."""
    import subprocess

    from conftest import authored_python
    from cozy_runtime.internal import _installed_dependencies

    generation = tmp_path / "generation"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(generation)], check=True, capture_output=True
    )
    python = generation / "bin/python"
    (site_packages,) = generation.glob("lib/python*/site-packages")
    _installed_dependencies.snapshot(
        authored_python(), generation, {"tenant"}, cache=tmp_path.parent / "sdk-cache"
    )
    metadata = site_packages / "tenant-1.0.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.3\nName: tenant\nVersion: 1.0.0\n"
        "Requires-Python: >=3.12\nRequires-Dist: cozy-runtime\nRequires-Dist: msgspec\n"
    )
    (site_packages / "tenant_job.py").write_text((TENANT / "tiny_tenant.py").read_text() + JOB)
    return package_environment.InstalledEnvironment(
        generation=generation,
        site_packages=site_packages,
        python=python,
        receipt_bytes=b"{}",
        receipt_digest="sha256:" + "0" * 64,
        content_digest="sha256:" + "1" * 64,
        reused=False,
    )


def test_the_release_declares_both_collections_slots(interface: dict[str, Any]) -> None:
    """The set the hub must send. Reading `entrypoints` alone loses the job's slot
    entirely -- and for a release with no entrypoint at all, loses everything."""

    assert _slot_paths(interface, "entrypoints", "jobs") == {
        ENTRYPOINT_SLOT,
        RETOUCH_SLOT,
        JOB_SLOT,
    }
    assert _slot_paths(interface, "entrypoints") == {ENTRYPOINT_SLOT, RETOUCH_SLOT}
    assert _slot_paths(interface, "jobs") == {JOB_SLOT}


@pytest.mark.parametrize("checkpoint_only", [False, True])
def test_a_job_shaped_selection_prepares(
    interface: dict[str, Any],
    released: tuple[Path, str, int],
    tmp_path: Path,
    checkpoint_only: bool,
) -> None:
    """THE H3 BLOCKER, end to end. The download set binds the job's slot and nothing else;
    the release declares both. Preparation authors the placement, carries the model row,
    stages the job plan, and binds no entrypoint -- because a job has none to bind."""

    store_root, manifest, _length = released
    raw = package_interface.canonical_bytes(interface)
    selection = _selection(JOB_SLOT, manifest)
    if checkpoint_only:
        selection.pop("release")
        selection.pop("lane")
    result = _prepare_published(
        package_name=PACKAGE,
        release=RELEASE,
        application=str(interface["application"]),
        model_slot_paths=tuple(sorted((ENTRYPOINT_SLOT, RETOUCH_SLOT, JOB_SLOT))),
        inventory=None,
        locked=package_environment.read_locked_requirements(
            b"--index-url https://pypi.org/simple\ntenant==1 --hash=sha256:" + b"a" * 64 + b"\n"
        ),
        models=[selection],
        placement_seed=b"job-shaped",
        artifact_cache=tmp_path / "artifacts",
        tensorfs_root=store_root,
        install_root=tmp_path / "install",
        python=Path(sys.executable),
        describe=lambda _installed, _distribution: raw,
        verified=lambda _digest, _length, _path: None,
        job_plan_root=tmp_path / "job-plans",
        base=None,
        installed_environment=_installed(tmp_path),
    )
    placement = json.loads(result.placement_set.placement_set_canonical_bytes)["placements"][0]
    assert placement["entrypoints"] == []
    assert [row["repo"] for row in placement["models"]] == [PACKAGE]
    from cozy_runtime.internal.worker.session import read_placement_set

    assert read_placement_set(result.placement_set)[0] == placement
    assert placement["models"][0].get("version", "") == ("" if checkpoint_only else RELEASE)
    assert placement["models"][0].get("lane", "") == ("" if checkpoint_only else LANE)
    # The job plan the JobDirective will resolve against exists.
    assert len(list((tmp_path / "job-plans").glob("*/*.json"))) == 1


@pytest.mark.parametrize("checkpoint_only", [False, True])
def test_an_entrypoint_shaped_selection_still_prepares_while_more_is_declared(
    interface: dict[str, Any],
    released: tuple[Path, str, int],
    tmp_path: Path,
    checkpoint_only: bool,
) -> None:
    """paul/anima's arm. The release declares two slots and the invocation binds ONE of
    them -- the entrypoint's. That is the case an equality between what the release
    declares and what the invocation binds refuses, and it must not."""

    store_root, manifest, _length = released
    raw = package_interface.canonical_bytes(interface)
    constructed: list[str] = []

    def derive(path: str, _config: bytes, _assets: Any, _dtypes: Any) -> tuple[str, ...]:
        # The one process seam: a real derive runs inside the installed venv's own
        # interpreter. Everything the guards read is real.
        constructed.append(path)
        return ("text_encoder", "unet", "vae")

    selection = _selection(ENTRYPOINT_SLOT, manifest)
    if checkpoint_only:
        selection.pop("release")
        selection.pop("lane")
    models, entrypoints = _entrypoints(
        package_interface.read_bytes(raw),
        package_name=PACKAGE,
        selections=[selection],
        tensorfs_root=store_root,
        verified=lambda _digest, _length, _path: None,
        derive=derive,
    )
    assert [row.name for row in entrypoints] == ["touch"]
    assert [slot.slot for slot in entrypoints[0].slots] == ["model"]
    assert [row.repo for row in models] == [PACKAGE]
    assert constructed == [ENTRYPOINT_SLOT]

    from cozy_runtime.internal import base_observation, canonical
    from cozy_runtime.internal.worker.acquire import Acquirer
    from cozy_runtime.internal.worker.session import Placement
    from cozy_runtime.protocol import documents

    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / digest[7:]).write_bytes(raw)
    acquired = Acquirer(
        cache_root=cache,
        install_root=tmp_path / "environment",
        selection_root=tmp_path / "selection",
        tensorfs_root=store_root,
        python=Path(sys.executable),
        base=base_observation.current(),
    )
    model_rows = tuple(documents.body(row) for row in models)
    entry_rows = tuple(documents.body(row) for row in entrypoints)
    placement = Placement(
        placement_id="modeled",
        placement_set_digest=b"a" * 32,
        package={"package": PACKAGE, "release": RELEASE},
        development={},
        environment_digest=b"b" * 32,
        package_interface=(digest, len(raw)),
        bindings_digest=documents.digest_of(
            canonical.write({"models": list(model_rows), "entrypoints": list(entry_rows)})
        ),
        models=model_rows,
        entrypoints=entry_rows,
        environment={},
        document={},
    )
    bindings, _ = acquired._bindings(placement)
    assert len(bindings) == 1
    selected = next(iter(bindings.values())).models[0]
    assert selected.reference_snapshot == manifest
    assert selected.model == (PACKAGE if checkpoint_only else f"{PACKAGE}@{RELEASE}/{LANE}")


def test_a_job_slot_is_never_constructed_at_preparation(
    interface: dict[str, Any], released: tuple[Path, str, int]
) -> None:
    """The reason a job's slot is unbound here, made observable: nothing derives it."""

    store_root, manifest, _length = released

    def derive(path: str, _config: bytes, _assets: Any, _dtypes: Any) -> tuple[str, ...]:
        raise AssertionError(f"a job slot was constructed at preparation: {path}")

    models, entrypoints = _entrypoints(
        package_interface.read_bytes(package_interface.canonical_bytes(interface)),
        package_name=PACKAGE,
        selections=[_selection(JOB_SLOT, manifest)],
        tensorfs_root=store_root,
        verified=lambda _digest, _length, _path: None,
        derive=derive,
    )
    assert entrypoints == []
    # The bytes still reach the placement: the selection exists to admit them to this pod.
    assert [row.repo for row in models] == [PACKAGE]


def test_a_selection_no_callable_declares_still_refuses(
    interface: dict[str, Any], released: tuple[Path, str, int]
) -> None:
    """The negative control for what the check is actually for. Relaxing it to admit job
    slots must not admit a slot the package does not have."""

    store_root, manifest, _length = released
    with pytest.raises(PreparationRefusal) as refusal:
        _entrypoints(
            package_interface.read_bytes(package_interface.canonical_bytes(interface)),
            package_name=PACKAGE,
            selections=[_selection("no_such.models.source", manifest)],
            tensorfs_root=store_root,
            verified=lambda _digest, _length, _path: None,
            derive=lambda _path, _config, _assets, _dtypes: (),
        )
    assert refusal.value.code == "package_prepare_model_selection_mismatch"
    assert "no_such.models.source" in refusal.value.detail


def test_an_entrypoints_only_facts_request_refuses_and_names_the_gap(
    interface: dict[str, Any], released: tuple[Path, str, int], tmp_path: Path
) -> None:
    """The two halves must ship together. A hub that still reads only `entrypoints` sends
    a slot set the derived interface disagrees with, and the refusal names WHICH slot is
    missing rather than printing two lists to diff by eye."""

    store_root, manifest, _length = released
    raw = package_interface.canonical_bytes(interface)
    with pytest.raises(PreparationRefusal) as refusal:
        _prepare_published(
            package_name=PACKAGE,
            release=RELEASE,
            application=str(interface["application"]),
            model_slot_paths=(ENTRYPOINT_SLOT, RETOUCH_SLOT),
            inventory=None,
            locked=package_environment.read_locked_requirements(
                b"--index-url https://pypi.org/simple\ntenant==1 --hash=sha256:" + b"a" * 64 + b"\n"
            ),
            models=[_selection(JOB_SLOT, manifest)],
            placement_seed=b"stale-hub",
            artifact_cache=tmp_path / "artifacts",
            tensorfs_root=store_root,
            install_root=tmp_path / "install",
            python=Path(sys.executable),
            describe=lambda _installed, _distribution: raw,
            verified=lambda _digest, _length, _path: None,
            job_plan_root=tmp_path / "job-plans",
            base=None,
            installed_environment=_installed(tmp_path),
        )
    assert refusal.value.code == "package_prepare_model_slots_disagree"
    assert f"derived-only=['{JOB_SLOT}']" in refusal.value.detail


def _facts_request(slot_paths: list[str]) -> pb.PreparePackageSetRequest:
    return pb.PreparePackageSetRequest(
        application="tenant_job:app",
        model_slot_paths=slot_paths,
        image_inventory=pb.ImageInventory(profile="test", python="3.12.0"),
        locked_requirements=b"--index-url https://pypi.org/simple\n"
        b"tenant==1 --hash=sha256:" + b"a" * 64 + b"\n",
    )


def test_the_bound_slots_are_a_subset_of_the_declared_ones() -> None:
    """The relation the pod actually enforces on every prepare and every compatibility
    check. An invocation binds SOME of what the release declares; only a slot the release
    does not declare is a refusal.

    This is the gate `check_package_set_compatibility`'s own guard now agrees with -- it
    ran an equality, which refused every release declaring more than one invocation binds.
    """

    # sorted unique: the wire bound the hub already honours.
    declared = sorted([ENTRYPOINT_SLOT, RETOUCH_SLOT, JOB_SLOT])
    for bound in ([], [JOB_SLOT], [ENTRYPOINT_SLOT], declared):
        _, slot_paths, _, _ = _request_facts(
            _facts_request(declared), [_selection(slot, "sha256:" + "b" * 64) for slot in bound]
        )
        assert set(bound).issubset(slot_paths)

    with pytest.raises(PreparationRefusal) as refusal:
        _request_facts(
            _facts_request([ENTRYPOINT_SLOT]),
            [_selection(JOB_SLOT, "sha256:" + "b" * 64)],
        )
    assert refusal.value.code == "package_prepare_model_slot_unknown"
    assert refusal.value.detail == JOB_SLOT
