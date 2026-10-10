#!/usr/bin/env python3
"""Research-only additive attention overlay on the exact public Runtime 0.23.2 wheel."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "2f1fe2fc84e36f31fdb2f97532ba92ce57b55928"
DONOR_SHA = "47ea4cec914917c71b597ad0425807b273b243878ab67d66ad2fd493400ec1ac"
MODULES = tuple(
    f"cozy_runtime/internal/{name}.py"
    for name in ("attention", "attention_producer", "attention_sol")
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain").strip():
        raise SystemExit("Commit the reviewed patch before building its immutable overlay")
    head = git("rev-parse", "HEAD").decode().strip()
    version = f"0.23.2+dev.sage2producer.{head[:8]}"
    donor = args.donor.read_bytes()
    if sha(donor) != DONOR_SHA:
        raise SystemExit("Donor must be the exact public 0.23.2 native wheel")
    with zipfile.ZipFile(io.BytesIO(donor)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
        entries = {entry.filename: entry for entry in archive.infolist()}
    info = "cozy_runtime-0.23.2.dist-info"
    # Verify every original RECORD entry before retaining its bytes.
    for name, digest, size in csv.reader(io.StringIO(files[f"{info}/RECORD"].decode())):
        if digest:
            actual = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b"=")
            assert digest == "sha256=" + actual.decode() and int(size) == len(files[name])
    args.out.mkdir(parents=True, exist_ok=True)
    staging = args.out / f"source-overlay-{head[:8]}"
    staging.mkdir()
    for name in MODULES:
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[name])
        assert files[name] == git("show", "v0.23.2:src/" + name)
    patch = git("diff", BASE, head, "--", *("src/" + name for name in MODULES))
    (args.out / "additive.patch").write_bytes(patch)
    subprocess.run(["git", "apply", "--check", "-p2", "-"], cwd=staging, input=patch, check=True)
    subprocess.run(["git", "apply", "-p2", "-"], cwd=staging, input=patch, check=True)
    updated = dict(files)
    for name in MODULES:
        updated[name] = (staging / name).read_bytes()
    updated["cozy_runtime/_build_provenance.py"] = (
        '"""Research overlay; source is the public donor plus the retained additive patch."""\n'
        'COMMIT = "unknown"\n'
        f'BASE_WHEEL_SHA256 = "{DONOR_SHA}"\n'
        f'PATCH_COMMIT = "{head}"\n'
        f'PATCH_SHA256 = "{sha(patch)}"\n'
    ).encode()
    metadata = updated[f"{info}/METADATA"]
    changed_metadata, count = re.subn(
        rb"(?m)^Version: 0\.23\.2$", f"Version: {version}".encode(), metadata
    )
    assert count == 1
    updated[f"{info}/METADATA"] = changed_metadata
    del updated[f"{info}/RECORD"]
    expected = {*MODULES, "cozy_runtime/_build_provenance.py", f"{info}/METADATA"}
    changed = {name for name in updated if updated[name] != files[name]}
    assert changed == expected
    new_info = f"cozy_runtime-{version}.dist-info"
    data, new_data = "cozy_runtime-0.23.2.data", f"cozy_runtime-{version}.data"

    def renamed_path(name: str) -> str:
        return name.replace(info + "/", new_info + "/").replace(data + "/", new_data + "/")

    renamed = {renamed_path(name): value for name, value in updated.items()}
    original_names = {renamed_path(name): name for name in files}
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, value in sorted(renamed.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(value)))
    writer.writerow((new_info + "/RECORD", "", ""))
    renamed[new_info + "/RECORD"] = record.getvalue().encode()
    output = args.out / f"cozy_runtime-{version}-cp312-abi3-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(renamed.items()):
            entry = copy.copy(entries[original_names[name]])
            entry.filename = name
            archive.writestr(entry, value)
    report = {
        "wheel": str(output.resolve()),
        "wheel_sha256": sha(output.read_bytes()),
        "runtime_version": version,
        "tensorfs_version": "0.6.1",
        "agent_version": "0.5.1",
        "head": head,
        "patch_base": BASE,
        "donor_sha256": DONOR_SHA,
        "patch_sha256": sha(patch),
        "serving_modules_changed": list(MODULES),
        "all_other_payloads_identical": True,
        "dependencies_identical": True,
        "zip_attributes_preserved": True,
        "wheel_data_directory_matches_version": True,
        "native_sha256": sha(files["cozy_runtime/_kernels/_C.abi3.so"]),
        "source_overlay": str(staging.resolve()),
        "source_sha256": {name: sha(updated[name]) for name in MODULES},
        "scope": (
            "Only the additive selector patch over public .23.2; "
            "no inherited .23.3 changes or producer retry fix."
        ),
    }
    (args.out / "wheel-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
