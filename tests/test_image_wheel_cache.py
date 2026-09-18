from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cozy_runtime.internal import _image_wheel_cache as seed
from cozy_runtime.internal import package_environment as pe
from test_publish_derivation import _SimpleIndex
from test_unpublished_exact_dependencies import _material, _wheel


def seeded(tmp_path, monkeypatch):
    root = tmp_path / "image"
    root.mkdir(mode=0o755)
    monkeypatch.setattr(seed, "ROOT", root)
    dependency = _wheel(tmp_path, "seed_dependency", (), b"VALUE=91\n")
    base = pe.observe_base()
    seed.record(dependency, base)
    digest = "sha256:" + hashlib.sha256(dependency.read_bytes()).hexdigest()
    item = seed.read(digest, base)
    assert item is not None
    index = _SimpleIndex(tmp_path / "index", [dependency])
    monkeypatch.setattr(seed, "INDEXES", (index.url,))
    req = tmp_path / "seed.txt"
    req.write_text(seed.requirement(item))
    baseline = tmp_path / "baseline"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(baseline)], check=True)
    command = seed.command(Path(shutil.which("uv")), baseline / "bin/python", req)
    command.remove("--offline")
    try:
        subprocess.run(command, check=True)
    finally:
        index.close()
    return dependency, item, base, baseline


def test_private_envs_share_image_payload_without_archive_or_network(tmp_path, monkeypatch):
    dependency, item, base, baseline = seeded(tmp_path, monkeypatch)
    project = _wheel(
        tmp_path, "seed_client", ("seed-dependency==1.0",), b"from seed_dependency import VALUE\n"
    )
    source = _material(project, (dependency,), base)
    source = replace(
        source,
        grants=tuple(
            pe.BlobGrant(g.digest, g.length, item.path, image_seed=True)
            if g.digest == item.wheel.digest
            else g
            for g in source.grants
        ),
    )
    dependency.unlink()
    assert not list(seed.ROOT.rglob("*.whl"))
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    rows = pe.read_dependency_requirements(
        (
            f"seed-dependency @ https://unreachable.invalid/{item.wheel.filename} "
            f"--hash={item.wheel.digest}\n"
        ).encode()
    ).rows

    def no_fetch(*args, **kwargs):
        pytest.fail("seeded exact wheel attempted an archive fetch")

    monkeypatch.setattr(pe, "_fetch_local_locked_wheel", no_fetch)
    assert pe.fetch_locked_wheels(rows, tmp_path / "fetch", base, cache_root=cache)[0].image_seed
    payloads = [next(baseline.glob("lib/python*/site-packages/seed_dependency.py"))]
    for name in ("one", "two"):
        installed = pe.materialize(
            source, tmp_path / name, base=base, dependency_cache=cache, exact_dependencies=True
        )
        payloads.append(installed.site_packages / "seed_dependency.py")
        subprocess.run(
            [
                str(installed.python),
                "-I",
                "-c",
                "import seed_client,importlib.util; assert seed_client.VALUE==91; "
                'assert importlib.util.find_spec("grpc") is None',
            ],
            check=True,
        )
    assert all(p.is_symlink() for p in payloads)
    assert len({(p.stat().st_dev, p.stat().st_ino) for p in payloads}) == 1
    assert not list(cache.rglob("*seed_dependency*.whl"))


@pytest.mark.parametrize("missing", ["cache", "facts", "wrong_hash"])
def test_missing_or_different_seed_falls_back_to_exact_archive(tmp_path, monkeypatch, missing):
    _, item, base, _ = seeded(tmp_path, monkeypatch)
    if missing == "cache":
        shutil.rmtree(seed.ROOT / "uv-cache")
    elif missing == "facts":
        item.path.unlink()
    digest = "sha256:" + "0" * 64 if missing == "wrong_hash" else item.wheel.digest
    rows = pe.read_dependency_requirements(
        (
            f"seed-dependency @ https://unreachable.invalid/{item.wheel.filename} --hash={digest}\n"
        ).encode()
    ).rows
    seen = []

    def fetch(path, row):
        seen.append(row.hashes)
        raise pe.EnvironmentRefusal("exact_fallback", row.hashes[0])

    monkeypatch.setattr(pe, "_fetch_local_locked_wheel", fetch)
    with pytest.raises(pe.EnvironmentRefusal, match="exact_fallback"):
        pe.fetch_locked_wheels(rows, tmp_path / "fetch", base, cache_root=tmp_path / "cache")
    assert seen == [(digest,)]


def test_uv_checks_cached_hash_not_just_name_and_version(tmp_path, monkeypatch):
    _, item, _, _ = seeded(tmp_path, monkeypatch)
    changed = replace(item, wheel=replace(item.wheel, digest="sha256:" + "0" * 64))
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    assert not seed.available(changed, Path(sys.executable), cache)
