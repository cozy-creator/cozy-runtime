"""Compose already reviewed, orthogonal Python experiments for one private rental.

This is an artifact construction check, not a runtime source/deployment gate.
The low-memory attention wheel is the donor; no dependencies or native binary change.
"""

import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path

ROOT = Path("/home/fidika/cozy_v2/outputs/h3-single5090-optimization-20261008")
OUTPUT = ROOT / "combined-linear"
VERSION = "0.19.0+dev.lowmem.scalarrows80599e60.bakefdc427a1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(folder, receipt):
    description = json.loads((ROOT / folder / receipt).read_text())
    path = Path(description["wheel"])
    assert digest(path) == description["sha256"], path
    return path, description


def verify_record(archive):
    record = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
    for name, fingerprint, size in rows:
        if not fingerprint:
            continue
        data = archive.read(name)
        assert len(data) == int(size)
        kind, encoded = fingerprint.split("=", 1)
        assert kind == "sha256"
        assert (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode() == encoded
        )
    return len(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    lowmem, lowmem_receipt = artifact("sage3-lifetimes", "handoff.json")
    gemm, gemm_receipt = artifact("rowwise-gemm-bounded", "private-wheel.json")
    bake, bake_receipt = artifact("lora-bake-dead-work", "handoff.json")
    assert json.loads((ROOT / "lora-bake-dead-work/cpu-byte-equivalence.json").read_text())[
        "passed"
    ]
    old_version = lowmem_receipt["version"]
    scratch = Path(tempfile.mkdtemp(prefix="unpack-", dir=OUTPUT))
    subprocess.run(
        [sys.executable, "-m", "wheel", "unpack", str(lowmem), "--dest", str(scratch)], check=True
    )
    tree = next(scratch.iterdir())
    changes = {
        "cozy_runtime/internal/encoding/leaves.py": gemm,
        "cozy_runtime/internal/encoding/rowwise_gemm.py": gemm,
        "cozy_runtime/internal/lora_bake.py": bake,
    }
    for name, source in changes.items():
        with zipfile.ZipFile(source) as archive:
            verify_record(archive)
            (tree / name).write_bytes(archive.read(name))
    provenance = tree / "cozy_runtime/_build_provenance.py"
    with provenance.open("a") as stream:
        stream.write(f'\nBOUNDED_ROWWISE_GEMM_PATCH_COMMIT = "{gemm_receipt["patch_commit"]}"\n')
        stream.write(f'LORA_BAKE_DEAD_WORK_PATCH_COMMIT = "{bake_receipt["patch_commit"]}"\n')
    info = next(tree.glob("*.dist-info"))
    metadata = BytesParser().parsebytes((info / "METADATA").read_bytes())
    assert metadata["Version"] == old_version
    metadata.replace_header("Version", VERSION)
    (info / "METADATA").write_bytes(metadata.as_bytes(policy=compat32.clone(max_line_length=0)))
    info.rename(tree / f"cozy_runtime-{VERSION}.dist-info")
    next(tree.glob("*.data")).rename(tree / f"cozy_runtime-{VERSION}.data")
    renamed = tree.with_name(f"cozy_runtime-{VERSION}")
    tree.rename(renamed)
    subprocess.run(
        [sys.executable, "-m", "wheel", "pack", str(renamed), "--dest-dir", str(OUTPUT)], check=True
    )
    wheel = OUTPUT / f"cozy_runtime-{VERSION}-cp312-abi3-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(lowmem) as before, zipfile.ZipFile(wheel) as after:
        changed = []
        for name in before.namelist():
            new_name = name.replace(f"cozy_runtime-{old_version}.", f"cozy_runtime-{VERSION}.")
            if before.read(name) != after.read(new_name):
                changed.append(name)
        expected = {
            "cozy_runtime/internal/encoding/leaves.py",
            "cozy_runtime/internal/lora_bake.py",
            "cozy_runtime/_build_provenance.py",
            f"cozy_runtime-{old_version}.dist-info/METADATA",
            f"cozy_runtime-{old_version}.dist-info/RECORD",
        }
        assert set(changed) == expected, changed
        assert len(after.namelist()) == len(before.namelist()) + 1
        for name, source in changes.items():
            with zipfile.ZipFile(source) as original:
                assert after.read(name) == original.read(name)
        old_metadata = BytesParser().parsebytes(
            before.read(f"cozy_runtime-{old_version}.dist-info/METADATA")
        )
        new_metadata = BytesParser().parsebytes(
            after.read(f"cozy_runtime-{VERSION}.dist-info/METADATA")
        )
        assert old_metadata.get_all("Requires-Dist") == new_metadata.get_all("Requires-Dist")
        native_hash = hashlib.sha256(
            after.read(f"cozy_runtime-{VERSION}.data/scripts/cozy-machine")
        ).hexdigest()
        assert native_hash == lowmem_receipt["native_machine_sha256"]
        entries = verify_record(after)
    report = {
        "wheel": str(wheel),
        "sha256": digest(wheel),
        "version": VERSION,
        "donor": str(lowmem),
        "donor_sha256": digest(lowmem),
        "component_wheels": {"gemm": str(gemm), "bake": str(bake)},
        "component_sha256": {"gemm": digest(gemm), "bake": digest(bake)},
        "component_module_bytes_preserved": True,
        "changed_existing_payloads": changed,
        "new_modules": ["cozy_runtime/internal/encoding/rowwise_gemm.py"],
        "all_other_payloads_preserved": True,
        "attention_and_shared_qkv_preserved_from_lowmem": True,
        "native_sha256": native_hash,
        "dependencies_preserved": True,
        "tensorfs_cohort": "0.4.0",
        "record_entries_verified": entries,
        "gpu_qualification": False,
        "installed": False,
        "public_release": False,
    }
    (OUTPUT / "private-wheel.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
