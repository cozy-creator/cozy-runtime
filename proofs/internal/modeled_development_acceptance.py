"""Focused positive/red proof for modeled DevelopmentPackage local selection."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "proofs" / "fixtures" / "modeled-development-package"
sys.path.insert(0, str(ROOT / "src"))

from modeled_development_store import create  # noqa: E402

from cozy_runtime.internal import descriptor, local  # noqa: E402
from cozy_runtime.internal.discovery import discover  # noqa: E402
from cozy_runtime.internal.worker import acquire  # noqa: E402
from cozy_runtime.protocol import documents  # noqa: E402

SOURCE = "sha256:" + "7" * 64


def _stage(home: Path, store: Path) -> local.StagedDevelopmentPlacement:
    found = discover(FIXTURE)
    return local.stage_development_placement(
        home,
        descriptor.build(found),
        package="proof/modeled-development-package",
        release="1.0.0",
        source_digest=SOURCE,
        found=found,
        artifact_store=store,
    )


def _candidate(staged: local.StagedDevelopmentPlacement, body: dict[str, Any] | None = None) -> Any:
    selected = body or documents.body(staged.placement)
    package_descriptor = selected["package_descriptor"]
    return SimpleNamespace(
        development=selected["development"],
        document=selected,
        entrypoints=tuple(selected["entrypoints"]),
        models=tuple(selected.get("models", [])),
        package_descriptor=(package_descriptor["digest"], package_descriptor["length"]),
    )


def _expect_stage(code: str, home: Path, store: Path) -> None:
    try:
        _stage(home, store)
    except RuntimeError as exc:
        if str(exc).startswith(code + ":"):
            return
        raise AssertionError(f"expected {code}, got {exc}") from exc
    raise AssertionError(f"expected {code}, selection succeeded")


def _expect_acquire(
    code: str,
    accepted: local.StagedDevelopmentPlacement,
    current: local.StagedDevelopmentPlacement,
    body: dict[str, Any] | None = None,
) -> None:
    try:
        acquire.preinstalled_development(
            _candidate(accepted, body),
            current.artifact_cache,
            current.placement,
            current.declared_bindings,
        )
    except acquire.AcquisitionRefusal as exc:
        if exc.code == code:
            return
        raise AssertionError(f"expected {code}, got {exc.code}: {exc.detail}") from exc
    raise AssertionError(f"expected {code}, acquisition succeeded")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix=".modeled-development-", dir=ROOT) as directory:
        work = Path(directory)
        first_store = work / "first-store"
        first_facts = create(first_store)
        first = _stage(work / "first-home", first_store)
        acquired = acquire.preinstalled_development(
            _candidate(first),
            first.artifact_cache,
            first.placement,
            first.declared_bindings,
        )
        assert len(first.placement.models) == 1
        assert len(acquired.bindings) == 1
        assert all(binding.development for binding in acquired.bindings.values())
        assert all(not binding.environment_digest for binding in acquired.bindings.values())
        assert all(not binding.variant for binding in acquired.bindings.values())
        selected = documents.body(first.placement)
        assert selected["models"][0]["lane"] == "cpu"
        slot = selected["entrypoints"][0]["slots"][0]
        assert "model_construction_contract" not in slot
        assert slot["config"]["document"]["digest"] == first_facts["config_digest"]

        _expect_stage(
            "development_model_store_absent",
            work / "absent-home",
            work / "absent-store",
        )
        import tensorfs

        empty_store = work / "empty-store"
        tensorfs.Store.init(str(empty_store))
        _expect_stage("development_model_missing", work / "empty-home", empty_store)

        corrupt_store = work / "corrupt-store"
        corrupt_facts = create(corrupt_store)
        config_hex = str(corrupt_facts["config_digest"]).removeprefix("sha256:")
        config_path = corrupt_store / "blobs" / config_hex[:2] / config_hex[2:4] / config_hex
        config_path.chmod(0o644)
        config_path.write_bytes(b"corrupt")
        _expect_stage("development_model_config_invalid", work / "corrupt-home", corrupt_store)

        changed_store = work / "changed-store"
        create(changed_store, config=b'{"width":5}')
        changed = _stage(work / "changed-home", changed_store)
        _expect_acquire("development_model_manifest_changed", first, changed)

        changed_config = copy.deepcopy(selected)
        changed_config["entrypoints"][0]["slots"][0]["config"]["document"]["digest"] = (
            "sha256:" + "2" * 64
        )
        _expect_acquire("development_model_config_changed", first, first, changed_config)

        changed_manifest = copy.deepcopy(selected)
        changed_manifest["models"][0]["manifest"]["digest"] = "sha256:" + "3" * 64
        _expect_acquire("development_model_manifest_changed", first, first, changed_manifest)

    print(
        "modeled development acceptance: exact local Repository/Manifest/config STAGED; "
        "absent store/release, corrupt config, changed index/Manifest/config refused typed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
