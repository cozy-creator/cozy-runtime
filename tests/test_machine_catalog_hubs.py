"""A run's Models resolve at the run's own Hub, over real HTTP, whatever order the machine
holds its Hub grants in; a dead tunnel in front of a Hub is unreachable, never an absent
release (2026-10-01: ngrok's offline page made `paul/sdxl@1.0.0` "no release")."""

from __future__ import annotations

import json
import ssl
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pytest

from cozy_runtime.internal import hostfacts
from cozy_runtime.internal.worker import machine_model_overrides
from cozy_runtime.internal.worker.machine_model_resolve import Resolutions
from cozy_runtime.internal.worker.machine_publication import (
    MachinePublicationClient,
    PublicationRefusal,
)
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb

PACKAGE = "local/w3-plane-sdxl"
DIGEST = "sha256:" + "5d" * 32
OFFLINE = b"<!DOCTYPE html><html><body>The endpoint is offline. ERR_NGROK_3200</body></html>"


class Hub:
    """A Hub on loopback speaking Tensorhub's catalog wire: typed JSON on every 404."""

    def __init__(self, models: dict[str, int], *, tunnel_down: bool = False) -> None:
        self.models, self.tunnel_down = models, tunnel_down
        self.seen: list[tuple[str, str]] = []
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_GET(self) -> None:
                hub.seen.append((self.path, self.headers.get("Authorization", "")))
                status, body, kind = hub.answer(self.path)
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.token = "access-" + uuid.uuid4().hex
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, path: str) -> tuple[int, bytes, str]:
        if self.tunnel_down:
            return 404, OFFLINE, "text/html"
        url = urlsplit(path)
        query = parse_qs(url.query)
        name = url.path.removeprefix("/v1/models/")
        body: dict[str, Any]
        if url.path.endswith("/token"):  # a machine authorization's renewal
            expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
            body = {"token": "t", "expires_at": expires}
        elif url.path == "/v1/models/resolve" and query["ref"][0].partition("@")[0] in self.models:
            model, _, release = query["ref"][0].partition("@")
            body = {
                "model": model,
                "release": release,
                "lane": query["lane"][0],
                "manifest_id": DIGEST,
                "manifest_length": self.models[model],
            }
        elif name in self.models:
            lanes = [{"lane": "bf16", "bytes": 6 << 30}, {"lane": "fp8", "bytes": 3 << 30}]
            body = {"releases": [{"release": "1.0.0", "lanes": lanes}]}
        else:
            error = {"code": "model.not_found", "message": "no model " + name}
            return 404, json.dumps({"error": error}).encode(), "application/json"
        return 200, json.dumps(body).encode(), "application/json"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hubs() -> Iterator[tuple[Hub, Hub, Hub]]:
    other, own, tunnel = Hub({}), Hub({"paul/sdxl": 4096}), Hub({}, tunnel_down=True)
    try:
        yield other, own, tunnel
    finally:
        for hub in (other, own, tunnel):
            hub.close()


def machine(path: Path, *held: Hub) -> Worker:
    """A persistent machine as the Runtime reads it: no boot Hub, the Host's grant file."""
    hubs = [
        {"origin": hub.origin, "access_token": hub.token, "expires_at": int(time.time()) + 3600}
        for hub in held
    ]
    path.write_text(json.dumps({"version": 1, "hubs": hubs}))
    return cast(
        Worker,
        SimpleNamespace(
            options=SimpleNamespace(publication_authority=None, hubs=(), hub_access_path=path),
            hub_access_lock=threading.Lock(),
            hub_access_principals={},
            resolutions=Resolutions(),
            workspace=None,
            host_facts=lambda: hostfacts.measure("none"),
            lanes=SimpleNamespace(entries=()),
        ),
    )


def resolve(worker: Worker, hub: str) -> dict[str, Any]:
    """`--model paul/sdxl@1.0.0` on a local package: the run's explicit Model choice."""
    return machine_model_overrides.checkpoint(
        worker,
        pb.ModelChoice(parameter="model", repository="paul/sdxl", release="1.0.0"),
        {"path": "generate.models.model"},
        package=PACKAGE,
        hub=hub,
        owner="paul",
        credentials={},
        note=lambda *_: None,
    )


def test_a_model_resolves_at_the_runs_hub_whatever_the_grant_order(
    tmp_path: Path, hubs: tuple[Hub, Hub, Hub]
) -> None:
    other, own, _ = hubs
    for order in ((other, own), (own, other)):
        other.seen.clear()
        own.seen.clear()
        worker = machine(tmp_path / "machine-hubs.json", *order)
        selected = resolve(worker, own.origin)
        assert selected["public_origin"] == own.origin
        assert selected["repository"] == "paul/sdxl" and selected["release"] == "1.0.0"
        assert selected["lane"] == "bf16"
        assert selected["manifest"] == {"digest": DIGEST, "length": 4096}
        assert other.seen == []
        assert own.seen and {auth for _, auth in own.seen} == {"Bearer " + own.token}
        # A run that names no Hub never borrows a held grant, first or otherwise.
        with pytest.raises(WorkspaceRefusal, match="names no Hub"):
            resolve(machine(tmp_path / "machine-hubs.json", *order), "")
        assert other.seen == [] and len(own.seen) == 2


def test_a_dead_tunnel_is_unreachable_never_an_absent_release(
    tmp_path: Path, hubs: tuple[Hub, Hub, Hub]
) -> None:
    other, own, tunnel = hubs
    worker = machine(tmp_path / "machine-hubs.json", tunnel, other)
    with pytest.raises(WorkspaceRefusal, match="no release"):
        resolve(worker, other.origin)  # the Hub's own typed 404: it has no such Model
    with pytest.raises(WorkspaceRefusal) as refused:
        resolve(worker, tunnel.origin)
    assert "unreachable" in str(refused.value) and "ERR_NGROK_3200" in str(refused.value)
    assert "no release" not in str(refused.value)
    # Publication reads the same way: a dead tunnel is a transient refusal, never the
    # Hub's 404 that settles a finalize as never arrived.
    for hub, code, status in (
        (tunnel, "publication.hub_unreachable", 503),
        (own, "publication.http_refused", 404),
    ):
        client = MachinePublicationClient(
            hub.origin,
            str(uuid.uuid4()),
            ssl.create_default_context(),
            worker_id="",
            worker_token="",
            access_token=hub.token,
        )
        with pytest.raises(PublicationRefusal) as publication:
            client.request("GET", "/v1/models/paul/absent/finalization")
        assert (publication.value.code, publication.value.status) == (code, status)
