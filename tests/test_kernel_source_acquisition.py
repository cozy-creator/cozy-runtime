"""Runtime pins through real HTTP, source locks and the actual CPU builder/import path."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import kernel_cache, kernel_compile, kernel_sources, storage_admission
from cozy_runtime.internal.package_environment import EnvironmentRefusal
from test_kernel_compile import _wait
from test_storage_admission import operation


def _wheel() -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("tiny_kernel/__init__.py", "VALUE = 42\n")
        archive.writestr("tiny_kernel-1.0.dist-info/METADATA", "Name: tiny-kernel\nVersion: 1.0\n")
    return data.getvalue()


def _tree() -> bytes:
    data = io.BytesIO()
    content = b"VALUE = 17\n"
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        entry = tarfile.TarInfo("release/pure/tree_kernel/__init__.py")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    return data.getvalue()


@contextmanager
def _http(files: dict[str, bytes]) -> Iterator[tuple[str, list[str]]]:
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            if self.path == "/private-redirect.whl":
                self.send_response(302)
                self.send_header("Location", "https://169.254.169.254/latest/meta-data/")
                self.end_headers()
                return
            body = files.get(self.path)
            self.send_response(200 if body is not None else 404)
            self.send_header("Content-Length", str(len(body or b"")))
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> kernel_cache.Store:
    scope = tmp_path / "boot"
    scope.mkdir()
    monkeypatch.setenv("TMPDIR", str(scope))
    monkeypatch.setattr("tempfile.tempdir", None)
    found = kernel_cache.Store(tmp_path / "kernels" / "u1")
    monkeypatch.setenv(kernel_cache.OWN_ENV, str(found.own))
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [None])
    monkeypatch.setattr(kernel_sources, "ROOT", tmp_path / "old-image-sources")
    return found


def _pin(name: str, url: str, body: bytes, package: str = "") -> kernel_sources.Pin:
    return kernel_sources.Pin(name, "1.0", url, hashlib.sha256(body).hexdigest(), package)


def _source() -> dict[str, Any]:
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    [row] = job.spec["sources"].values()
    return dict(row)


def test_pins_build_without_an_image_index_and_replay_offline(
    store: kernel_cache.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only this recipe's wheel/tree are fetched; a stale image index has no authority."""
    wheel, tree = _wheel(), _tree()
    kernel_sources.ROOT.mkdir()
    (kernel_sources.ROOT / "index.json").write_text("stale, invalid image index")
    (kernel_sources.ROOT / "tiny.whl").write_bytes(b"a stale image archive")
    with _http({"/tiny.whl": wheel, "/release.tar.gz": tree}) as (url, requests):
        pins = (
            _pin("tiny-kernel", url + "/tiny.whl", wheel),
            _pin("tree-kernel", url + "/release.tar.gz", tree, "pure/tree_kernel"),
            _pin("unused-setup", url + "/unused.tar.gz", b"not needed"),
        )
        monkeypatch.setattr(kernel_sources, "PINS", pins)
        job = kernel_sources.job(kernel_sources.PYTHON, 0)
        assert requests == [] and not store.own.exists()
        custom = kernel_sources.ROOT / "custom"
        custom.mkdir()
        (custom / "index.json").write_text(
            json.dumps(
                {
                    "sources": {
                        pin.name: {
                            "version": pin.version,
                            "file": kernel_sources._filename(pin),
                            "sha256": pin.sha256,
                            "package": pin.package,
                        }
                        for pin in pins
                    }
                }
            )
        )
        assert kernel_sources.job(kernel_sources.PYTHON, 0, custom).key == job.key
        kernel_compile.submit(store, job)
        state = _wait(store, job)
        assert state.state == "ready", state
        assert sorted(requests) == ["/release.tar.gz", "/tiny.whl"]
        for row in job.spec["sources"].values():
            source = Path(row["file"])
            assert source.is_relative_to(store.own)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
            assert source.stat().st_mode & 0o777 == 0o444
        site = Path(state.path) / kernel_cache.SITE
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import tiny_kernel,tree_kernel; print(tiny_kernel.VALUE+tree_kernel.VALUE)",
            ],
            env={**os.environ, "PYTHONPATH": str(site)},
        )
        assert output.strip() == b"59"
        kernel_compile._STARTED[job.key.digest][0].wait()
    shutil.rmtree(store.own / "sources")
    again = kernel_sources.job(kernel_sources.PYTHON, 0)
    assert again.key == job.key
    started = dict(kernel_compile._STARTED)
    assert kernel_compile.submit(store, again).state == "ready"
    assert started == kernel_compile._STARTED and not (store.own / "sources").exists()


def test_concurrent_builders_fetch_a_source_once_and_repair_a_corrupt_archive(
    store: kernel_cache.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _wheel()
    with _http({"/tiny.whl": body}) as (url, requests):
        monkeypatch.setattr(kernel_sources, "PINS", (_pin("tiny", url + "/tiny.whl", body),))
        row = _source()
        code = (
            "import json,sys; from cozy_runtime.internal.kernel_sources import _verified; "
            "print(_verified(json.loads(sys.argv[1])))"
        )
        processes = [
            subprocess.Popen([sys.executable, "-c", code, json.dumps(row)], stdout=subprocess.PIPE)
            for _ in range(4)
        ]
        outputs = [process.communicate(timeout=20)[0].strip() for process in processes]
        assert all(process.returncode == 0 for process in processes)
        assert len(set(outputs)) == 1 and requests == ["/tiny.whl"]
        source = Path(row["file"])
        source.chmod(0o600)
        source.write_bytes(b"corrupted after acquisition")
        assert kernel_sources._verified(row).read_bytes() == body
        assert requests == ["/tiny.whl", "/tiny.whl"]
        assert not list(source.parent.glob("*.partial"))


def test_matching_preseed_is_reused_without_network_or_image_metadata(
    store: kernel_cache.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _wheel()
    pin = _pin("tiny", "http://127.0.0.1:1/tiny.whl", body)
    monkeypatch.setattr(kernel_sources, "PINS", (pin,))
    kernel_sources.ROOT.mkdir()
    archive = kernel_sources.ROOT / "tiny.whl"
    archive.write_bytes(body)
    archive.chmod(0o444)
    before = archive.stat()
    job = kernel_sources.job(kernel_sources.PYTHON, 0)
    kernel_compile.submit(store, job)
    assert _wait(store, job).state == "ready"
    assert archive.stat().st_mtime_ns == before.st_mtime_ns
    assert not Path(job.spec["sources"]["tiny"]["file"]).exists()


def test_checksum_refusal_is_visible_and_publishes_no_archive_or_artifact(
    store: kernel_cache.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _http({"/tiny.whl": _wheel()}) as (url, requests):
        monkeypatch.setattr(
            kernel_sources, "PINS", (kernel_sources.Pin("tiny", "1", url + "/tiny.whl", "0" * 64),)
        )
        job = kernel_sources.job(kernel_sources.PYTHON, 0)
        kernel_compile.submit(store, job)
        failed = _wait(store, job)
        assert failed.state == "failed" and "wheel hash or length changed" in failed.detail
        assert requests == ["/tiny.whl"] and store.entry(job.key) is None
        source = Path(job.spec["sources"]["tiny"]["file"])
        assert not source.exists() and not list(source.parent.glob("*.partial"))


def test_a_private_redirect_is_refused_before_any_source_write(
    store: kernel_cache.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _http({}) as (url, requests):
        monkeypatch.setattr(
            kernel_sources, "PINS", (_pin("tiny", url + "/private-redirect.whl", _wheel()),)
        )
        row = _source()
        with pytest.raises(EnvironmentRefusal):
            kernel_sources._verified(row)
        source = Path(row["file"])
        assert requests == ["/private-redirect.whl"] and not source.exists()
        assert not list(source.parent.glob("*.partial"))


def test_source_acquisition_obeys_the_existing_storage_owner_refusal(
    store: kernel_cache.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _wheel()
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    monkeypatch.setattr(storage_admission, "_channel", worker)
    with host, worker, _http({"/tiny.whl": body}) as (url, _):
        monkeypatch.setattr(kernel_sources, "PINS", (_pin("tiny", url + "/tiny.whl", body),))
        row = _source()
        received: list[dict[str, Any]] = []

        def refuse() -> None:
            rows, connection = operation(host)
            received.extend(rows)
            with connection:
                connection.sendall(
                    b'{"ok":false,"code":"insufficient_storage","reason":"fixture budget"}\n'
                )

        thread = threading.Thread(target=refuse)
        thread.start()
        with pytest.raises(storage_admission.StorageRefusal, match="fixture budget"):
            kernel_sources._verified(row)
        thread.join(2)
        assert not thread.is_alive()
        source = Path(row["file"])
        assert received == [{"path": str(source.parent), "bytes": len(body), "inodes": 16}]
        assert not source.exists() and not list(source.parent.glob("*.partial"))
