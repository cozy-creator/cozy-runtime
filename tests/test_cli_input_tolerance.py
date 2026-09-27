"""Creator-to-Runtime CLI inputs tolerate fields and formatting from a newer Creator."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

from cozy_runtime.internal import interface_wheel, package_interface, static_interface
from test_image_prepare_cli import request, run
from test_serving_interfaces import built, project

ROOT = Path(__file__).resolve().parents[1]


def test_image_prepare_ignores_additive_request_fields(tmp_path: Path) -> None:
    source = tmp_path / "input.jpg"
    Image.frombytes("RGB", (256, 256), random.Random(3).randbytes(256 * 256 * 3)).save(source)
    body = request(source)
    body["priority"] = "interactive"
    body["prepare"]["quality_hint"] = 90
    assert run(body)["status"] == "prepared"


def test_interface_wheel_accepts_ordinary_json_with_additive_fields(tmp_path: Path) -> None:
    root = project(tmp_path)
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(package_interface.canonical_bytes(static_interface.build(root)))
    implementation = io.BytesIO()
    with zipfile.ZipFile(implementation, "w") as archive:
        archive.writestr("serving_fixture.py", (root / "serving_fixture.py").read_text())
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: serving-fixture\nVersion: 0.1.0\n",
        )
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    wheel = tmp_path / "serving_fixture-0.1.0-py3-none-any.whl"
    wheel.write_bytes(implementation.getvalue())
    body = {
        "version": "0.1.0",
        "distribution": "serving-fixture",
        "interface_path": str(interface),
        "output_directory": str(tmp_path / "out"),
        "implementation_wheel": str(wheel),
        "implementation_wheel_digest": "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "requested_by": "a newer Creator",
    }
    result = subprocess.run(
        [str(Path(sys.executable).with_name("cozy-runtime")), "--json", "interface-wheel"],
        input=json.dumps(body, indent=2),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(result.stdout)
    built = Path(document["path"])
    assert document["digest"] == "sha256:" + hashlib.sha256(built.read_bytes()).hexdigest()
    assert built.name == wheel.name and built.parent == tmp_path / "out"
    with zipfile.ZipFile(built) as archive:
        assert "serving_fixture.py" in archive.namelist()


def test_interface_wheel_streams_a_member_larger_than_its_process(tmp_path: Path) -> None:
    """No wheel size cap, and no wheel in memory: the CLI's peak RSS stays below one member."""
    root = project(tmp_path)
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(package_interface.canonical_bytes(static_interface.build(root)))
    wheel = tmp_path / "serving_fixture-0.1.0-py3-none-any.whl"
    member, chunk = 384 << 20, bytes(1 << 20)
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("serving_fixture.py", (root / "serving_fixture.py").read_text())
        with archive.open("serving_fixture_data/weights.bin", "w") as stream:
            for _ in range(member // len(chunk)):
                stream.write(chunk)
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: serving-fixture\nVersion: 0.1.0\n",
        )
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    with wheel.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    body = {
        "distribution": "serving-fixture",
        "version": "0.1.0",
        "interface_path": str(interface),
        "output_directory": str(tmp_path / "out"),
        "implementation_wheel": str(wheel),
        "implementation_wheel_digest": "sha256:" + source_digest,
    }
    # The CLI is this wrapper's only child, so RUSAGE_CHILDREN is exactly its peak RSS.
    measure = (
        "import json, resource, subprocess, sys\n"
        "done = subprocess.run(sys.argv[1:], input=sys.stdin.buffer.read(), capture_output=True)\n"
        "sys.stderr.buffer.write(done.stderr)\n"
        "print(json.dumps({'code': done.returncode, 'stdout': done.stdout.decode(),"
        " 'peak': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss << 10}))\n"
    )
    cli = str(Path(sys.executable).with_name("cozy-runtime"))
    result = subprocess.run(
        [sys.executable, "-c", measure, cli, "--json", "interface-wheel"],
        input=json.dumps(body),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=300,
    )
    measured = json.loads(result.stdout)
    assert measured["code"] == 0, measured["stdout"] + result.stderr
    assert measured["peak"] < member, measured["peak"]
    document = json.loads(measured["stdout"])
    built_wheel = Path(document["path"])
    with built_wheel.open("rb") as stream:
        assert document["digest"] == "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    expected = hashlib.sha256()
    for _ in range(member // len(chunk)):
        expected.update(chunk)
    encoded = base64.urlsafe_b64encode(expected.digest()).rstrip(b"=").decode()
    with zipfile.ZipFile(built_wheel) as archive:
        assert archive.getinfo("serving_fixture_data/weights.bin").file_size == member
        record = archive.read("serving_fixture-0.1.0.dist-info/RECORD").decode().splitlines()
        assert f"serving_fixture_data/weights.bin,sha256={encoded},{member}" in record
        assert "from _cozy_callers_" in archive.read("serving_fixture.py").decode()


def test_recapture_rebinds_a_caller_wheel_from_another_generator(tmp_path: Path) -> None:
    root = project(tmp_path)
    raw = package_interface.canonical_bytes(static_interface.build(root))
    implementation = io.BytesIO()
    with zipfile.ZipFile(implementation, "w") as archive:
        archive.writestr("serving_fixture.py", (root / "serving_fixture.py").read_text())
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: serving-fixture\nVersion: 0.1.0\n",
        )
        archive.writestr(
            "serving_fixture-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    filename, published = built(
        raw,
        distribution="serving-fixture",
        version="0.1.0",
        implementation=implementation.getvalue(),
        implementation_filename="serving_fixture-0.1.0-py3-none-any.whl",
    )
    # The same wheel as an older generator wrote it.
    older = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(published)) as source, zipfile.ZipFile(older, "w") as target:
        for name in source.namelist():
            body = source.read(name)
            if name.endswith("METADATA"):
                body = body.replace(
                    interface_wheel.GENERATOR_ABI.encode(), b"cozy.interface-generator/7"
                )
            elif "_cozy_callers_" in name:
                body = b"# an older generator's caller layout\n" + body
            target.writestr(name, body)
    # The installed source re-described by a newer Runtime with an additive field.
    body = json.loads(raw)
    body["generator"] = "a newer describe"
    newer = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    _, rebound = built(
        newer,
        distribution="serving-fixture",
        version="0.1.0",
        implementation=older.getvalue(),
        implementation_filename=filename,
    )
    fresh = dict(interface_wheel.overlay_sources(newer, {"serving_fixture.py"}))
    ((generated, companion, suffix),) = fresh.values()
    with zipfile.ZipFile(io.BytesIO(rebound)) as archive:
        module = archive.read("serving_fixture.py")
        assert module.count(b"from _cozy_callers_") == 1 and module.endswith(suffix)
        assert archive.read(generated) == companion
        companions = [name for name in archive.namelist() if "_cozy_callers_" in name]
        assert companions == [generated]
