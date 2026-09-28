"""A synced project's account-index rows resolve at this machine's own Hub, never at the Hub
its author locked against. Real: uv, the install, a PEP 503 index on loopback."""

from __future__ import annotations

import hashlib
import http.server
import shutil
import subprocess
import sys
import tarfile
import threading
from functools import partial
from pathlib import Path

import pytest

from cozy_runtime.internal import package_installation
from cozy_runtime.internal.package_environment import EnvironmentRefusal
from test_unpublished_exact_dependencies import _wheel

AUTHOR = "http://author-laptop.invalid:8819"


def test_a_locked_account_index_resolves_at_the_machines_own_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    monkeypatch.setattr(package_installation, "IMAGE_UV_CACHE", tmp_path / "no-image-cache")
    wheel = _wheel(tmp_path, "account_dep", (), b"VALUE=5\n")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    served = tmp_path / "hub"
    (served / "v1/index/acct/simple/account-dep").mkdir(parents=True)
    (served / f"v1/index/acct/files/{digest}").mkdir(parents=True)
    shutil.copy(wheel, served / f"v1/index/acct/files/{digest}/{wheel.name}")
    (served / "v1/index/acct/simple/account-dep/index.html").write_text(
        f'<a href="/v1/index/acct/files/{digest}/{wheel.name}#sha256={digest}">{wheel.name}</a>'
    )
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    hub = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "probe"\nversion = "1.0"\nrequires-python = ">=3.10"\n'
            'dependencies = ["account-dep"]\n'
            '[tool.uv.sources]\naccount-dep = { index = "tensorhub" }\n'
            f'[[tool.uv.index]]\nname = "tensorhub"\nurl = "{hub}/v1/index/acct/simple/"\n'
            "explicit = true\n"
        )
        subprocess.run(
            [uv, "lock", "--project", str(project), "--python", sys.executable],
            check=True,
            capture_output=True,
        )
        # The author locked against their own Hub, which this machine cannot reach.
        for name in ("pyproject.toml", "uv.lock"):
            path = project / name
            path.write_text(path.read_text().replace(hub, AUTHOR))
        assert f"{AUTHOR}/v1/index/acct/files/{digest}" in (project / "uv.lock").read_text()
        archive = tmp_path / "source.tar"
        with tarfile.open(archive, "w") as tar:
            for name in ("pyproject.toml", "uv.lock"):
                tar.add(project / name, arcname=name)

        def install(identifier: str, own: package_installation.HubIndex | None) -> Path:
            return package_installation.install(
                tmp_path / "installs",
                package="local/probe",
                release="1.0",
                python=Path(sys.executable),
                installation_id=identifier,
                source_archive=archive,
                cache=tmp_path / "cache",
                hub=own,
            ).python

        with pytest.raises(EnvironmentRefusal, match="no Hub grant"):
            install("ungranted", None)
        python = install("granted", package_installation.HubIndex(hub + "/"))
        value = subprocess.run(
            [str(python), "-I", "-c", "import account_dep; print(account_dep.VALUE)"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert value.stdout.strip() == "5"
    finally:
        server.shutdown()
        server.server_close()
