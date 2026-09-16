"""Real wheel installation and snapshot discovery, without claiming CUDA execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts

ROOT = Path(__file__).resolve().parents[1]


def _wheel(tmp_path: Path, *, revision: str = "a" * 40) -> Path:
    snapshot = tmp_path / "snapshot"
    variant = snapshot / "build" / "torch-stable-abi29-cu130-x86_64-linux"
    variant.mkdir(parents=True)
    (variant / "metadata.json").write_text(
        json.dumps({"name": "flash-attn3", "backend": {"archs": ["9.0a"]}})
    )
    (snapshot / "LICENSE").write_text("Fixture only; no upstream kernel is redistributed.\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build-attention-wheel.py"),
            str(snapshot),
            "--kernel",
            "flash-attn3",
            "--version",
            "0.1.0",
            "--revision",
            revision,
            "--source-url",
            "https://example.org/kernel",
            "--license",
            "BSD-3-Clause",
            "--requirement",
            "torch==2.13.0",
            "--requirement",
            "kernels==0.14.1",
            "--out",
            str(tmp_path / "dist"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _install(wheel: Path, target: Path) -> None:
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--python",
            sys.executable,
            "--target",
            str(target),
            "cozy-kernel-flash-attn3 @ "
            + wheel.as_uri()
            + "#sha256="
            + hashlib.sha256(wheel.read_bytes()).hexdigest(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _inspect(target: Path, code: str) -> dict[str, object]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(target), str(ROOT / "src"))))
    result = subprocess.run(
        [sys.executable, "-c", "from cozy_runtime.internal import attention\nimport json\n" + code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result_value: dict[str, object] = json.loads(result.stdout)
    return result_value


def test_installed_snapshot_precedes_image_and_retains_exact_wheel_origin(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    original = wheel.read_bytes()
    target = tmp_path / "installed"
    _install(wheel, target)
    # Wheel output is deterministic, including standard RECORD and timestamps.
    same = _wheel(tmp_path / "repeat")
    assert same.read_bytes() == original
    facts = _inspect(
        target,
        """
from pathlib import Path
candidate = attention.BY_NAME["flash-attn3"]
print(json.dumps({"location": str(attention._snapshot_location(candidate)),
                  "archs": attention._baked_archs(candidate),
                  "artifact": attention._artifact("cozy-kernel-flash-attn3")}))
""",
    )
    assert facts["location"] == str(target / "cozy_kernel_flash_attn3")
    assert facts["archs"] == {"torch-stable-abi29-cu130-x86_64-linux": ["9.0a"]}
    artifact = facts["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["distribution"] == "cozy-kernel-flash-attn3"
    assert artifact["version"] == "0.1.0"
    assert artifact["source"].endswith("/commit/" + "a" * 40)
    assert artifact["wheel_digest"] == "sha256:" + hashlib.sha256(original).hexdigest()
    record = next(target.glob("*.dist-info/RECORD"))
    assert artifact["record_digest"] == "sha256:" + hashlib.sha256(record.read_bytes()).hexdigest()


def test_installed_snapshot_architecture_refuses_before_loading_native_code(tmp_path: Path) -> None:
    target = tmp_path / "installed"
    _install(_wheel(tmp_path), target)
    facts = _inspect(
        target,
        """
from cozy_runtime.internal.encoding import DeviceFacts
try:
    attention.pinned("flash-attn3", DeviceFacts("cuda", "B200", 100, "", "", 0))
except attention.AttentionRefusal as error:
    print(json.dumps({"code": error.code, "detail": str(error)}))
""",
    )
    assert facts["code"] == "attention_kernel_unsupported"
    assert "9.0a" in str(facts["detail"])
    # Even with an image snapshot available, a damaged installed wheel must not fall back.
    (target / "cozy_kernel_flash_attn3-0.1.0.dist-info" / "RECORD").unlink()
    facts = _inspect(
        target,
        """
try:
    attention._snapshot_location(attention.BY_NAME["flash-attn3"])
except attention.AttentionRefusal as error:
    print(json.dumps({"code": error.code, "detail": str(error)}))
""",
    )
    assert facts["code"] == "attention_kernel_absent"
    assert "no recorded kernel snapshot" in str(facts["detail"])


def test_request_artifact_observation_and_restore_use_actual_diffusers_processors() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    module = WanAttention(dim=128, heads=1, dim_head=128, processor=WanAttnProcessor())
    roots = {"dit": module}
    device = DeviceFacts("cpu", "CPU", 0, "", "", 0)
    attention.select(device, roots, "sdpa")
    defaults = attention.snapshot(roots)
    before = module.processor._attention_backend
    entry = attention.Ready(
        attention.BY_NAME["sdpa"],
        attention._member("native"),
        "/installed/snapshot",
        artifact={"distribution": "fixture", "version": "1", "record_digest": "sha256:abc"},
    )
    swap = attention.Swap(entry, (("dit", module.processor),), "dit=sdpa")
    value = torch.ones(1, 4, 128)
    with torch.no_grad():
        baseline = module(value)
        with (
            pytest.raises(RuntimeError, match="handler failed"),
            attention.override(defaults, roots, swap, "dit=sdpa") as applied,
        ):
            assert applied.document()["artifacts"]["sdpa"] == entry.artifact
            assert applied.locations == {"sdpa": "/installed/snapshot"}
            assert torch.equal(module(value), baseline)
            raise RuntimeError("handler failed")
        assert module.processor._attention_backend == before
        with attention.override(defaults, roots, None) as restored:
            assert restored.artifacts is None
            assert torch.equal(module(value), baseline)
