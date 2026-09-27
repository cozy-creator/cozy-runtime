"""Exercise publication refusals using disposable Git history and registry responses."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

GUARD = Path(__file__).resolve().parents[1] / "checks" / "release.py"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release fixture",
            "-c",
            "user.email=release@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass
class Registry:
    status: int = 404
    uploaded_files: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)


type ReleaseFixture = tuple[ModuleType, Path, Path, Registry]


@pytest.fixture(params=["lightweight", "annotated"])
def release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[ReleaseFixture]:
    spec = importlib.util.spec_from_file_location("release_guard", GUARD)
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "pyproject.toml").write_text('[project]\nname="cozy-runtime"\nversion="1.2.3"\n')
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "-qm", "release fixture")
    head = git(repo, "rev-parse", "HEAD")
    tag_args = ["-a", "-m", "fixture tag"] if request.param == "annotated" else []
    git(repo, "tag", *tag_args, "v1.2.3")
    event = tmp_path / "push.json"
    event.write_text(
        json.dumps(
            {
                "ref": "refs/tags/v1.2.3",
                "created": True,
                "deleted": False,
                "before": "0" * 40,
                "after": git(repo, "rev-parse", "v1.2.3"),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", [str(GUARD), "--source", str(repo)])
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v1.2.3")
    monkeypatch.setenv("GITHUB_SHA", head)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    registry = Registry()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            registry.requests.append(self.path)
            body = json.dumps({"urls": registry.uploaded_files}).encode()
            self.send_response(registry.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    monkeypatch.setattr(guard, "PYPI_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield guard, repo, event, registry
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_new_tag_can_publish_an_unpublished_version(release: ReleaseFixture) -> None:
    guard, _, _, registry = release
    assert guard.main() == 0
    assert registry.requests == ["/pypi/cozy-runtime/1.2.3/json"]


def test_new_tag_retry_before_any_upload_is_allowed(
    release: ReleaseFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, _, _, registry = release
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    assert guard.main() == 0
    # A build failure before upload leaves the same immutable push event and no
    # PyPI version. Rerunning that event needs neither a new tag nor a new version.
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    assert guard.main() == 0
    assert registry.requests == ["/pypi/cozy-runtime/1.2.3/json"] * 2


@pytest.mark.parametrize(
    "field,value", [("created", False), ("deleted", True), ("before", "a" * 40)]
)
def test_existing_or_deleted_tag_event_cannot_publish(
    release: ReleaseFixture, field: str, value: object
) -> None:
    guard, _, event_path, registry = release
    event = json.loads(event_path.read_text())
    event[field] = value
    event_path.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="new tag push or an explicit manual tag"):
        guard.main()
    assert registry.requests == []


def test_manual_dispatch_publishes_only_an_existing_version_tag(
    release: ReleaseFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, _, event_path, registry = release
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    event_path.write_text(json.dumps({"inputs": {"release_tag": "v1.2.3"}}))
    assert guard.main() == 0
    assert registry.requests == ["/pypi/cozy-runtime/1.2.3/json"]
    event_path.write_text(json.dumps({"inputs": {"release_tag": "master"}}))
    with pytest.raises(ValueError, match="manual publication requires an existing version tag"):
        guard.main()
    assert len(registry.requests) == 1


def test_tag_must_match_declared_version(
    release: ReleaseFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, _, event_path, registry = release
    event = json.loads(event_path.read_text())
    event["ref"] = "refs/tags/v2.0.0"
    event_path.write_text(json.dumps(event))
    monkeypatch.setenv("GITHUB_REF", event["ref"])
    with pytest.raises(ValueError, match=r"tag v2\.0\.0 != pyproject version 1\.2\.3"):
        guard.main()
    assert registry.requests == []


@pytest.mark.parametrize("change", ["tracked", "untracked", "committed"])
def test_changed_source_cannot_publish_under_unchanged_tag(
    release: ReleaseFixture, change: str
) -> None:
    guard, repo, _, registry = release
    if change == "tracked":
        with (repo / "pyproject.toml").open("a") as file:
            file.write("# unpublished edit\n")
    else:
        (repo / "new.py").write_text("VALUE = 7\n")
        if change == "committed":
            git(repo, "add", "new.py")
            git(repo, "commit", "-qm", "source edit without changing release version")
    with pytest.raises(ValueError, match=r"source must agree|uncommitted or untracked"):
        guard.main()
    assert registry.requests == []


@pytest.mark.parametrize("identity", ["event", "github_sha"])
def test_push_identity_must_match_tagged_checkout(
    release: ReleaseFixture, monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    guard, repo, event_path, registry = release
    original = git(repo, "rev-parse", "HEAD")
    (repo / "new.py").write_text("VALUE = 7\n")
    git(repo, "add", "new.py")
    git(repo, "commit", "-qm", "other source")
    other = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--detach", original)
    if identity == "event":
        event = json.loads(event_path.read_text())
        event["after"] = other
        event_path.write_text(json.dumps(event))
    else:
        monkeypatch.setenv("GITHUB_SHA", other)
    with pytest.raises(ValueError, match="source must agree"):
        guard.main()
    assert registry.requests == []


@pytest.mark.parametrize(
    "uploaded_files", [[], ["pure.whl"], ["pure.whl", "native.whl", "sdist.tar.gz"]]
)
def test_existing_even_partial_pypi_release_cannot_be_filled_from_new_source(
    release: ReleaseFixture, uploaded_files: list[str]
) -> None:
    guard, _, _, registry = release
    registry.status = 200
    registry.uploaded_files = uploaded_files
    with pytest.raises(ValueError, match="already exists on PyPI"):
        guard.main()
    assert registry.requests == ["/pypi/cozy-runtime/1.2.3/json"]


@pytest.mark.parametrize("status", [204, 403, 429, 500, 502])
def test_unavailable_index_does_not_mean_unpublished(release: ReleaseFixture, status: int) -> None:
    guard, _, _, registry = release
    registry.status = status
    with pytest.raises(ValueError, match=f"cannot establish PyPI version absence: HTTP {status}"):
        guard.main()


def test_publication_command_refuses_a_source_pr(tmp_path: Path) -> None:
    event = tmp_path / "pull_request.json"
    event.write_text(json.dumps({"number": 1}))
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_REF": "refs/pull/1/merge",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "release guard RED: publication requires a new tag push" in result.stderr
