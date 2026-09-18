"""Boot authority remains privileged; no provider/account secret reaches a job."""

from __future__ import annotations

import json
import ssl
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.machine_publication import (
    MachinePublicationClient,
    PublicationAuthority,
    PublicationRefusal,
)


def test_worker_boot_projects_authority_without_inheriting_it(tmp_path: Path) -> None:
    handoff = {
        "COZY_WORKER_ID": "worker-test",
        "COZY_WORKER_INTERNAL_PORT": "8443",
        "COZY_MEDIA_INTERNAL_PORT": "8444",
        "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(
            {"control_public_key_ed25519_b64url": "A" * 43, "media_token_sha256": ["b" * 64]}
        ),
        "PATH": "/usr/bin:/bin",
        "TENSORHUB_OBJECT_STORAGE_HOSTS": ".objects.example,storage.example",
    }
    authority = {
        "version": 2,
        "origin": "https://hub.example.test",
        "worker_id": "worker-test",
        "worker_token": "A" * 43,
    }
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority))
    program = """
import json,sys
from pathlib import Path
from cozy_runtime.internal.config import (
    ConfigError, read_worker_host_config, runtime_config_from_host,
)
try:
    body=json.load(sys.stdin)
    cfg=read_worker_host_config(body['env'], publication_authority_path=Path(body['path']))
except ConfigError as exc:
    print(exc.name)
    raise SystemExit(2)
print(json.dumps({"origin":cfg.tensorhub_origin,"child":dict(cfg.child_base_env),
                  "hosts":runtime_config_from_host(cfg).object_storage_hosts,
                  "dependency_cache":str(runtime_config_from_host(cfg).dependency_cache)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=json.dumps({"env": handoff, "path": str(path)}),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "origin": "https://hub.example.test",
        "child": {"PATH": "/usr/bin:/bin"},
        "hosts": [".objects.example", "storage.example"],
        "dependency_cache": "/var/lib/cozy/dependencies",
    }
    for env_change, change in (
        ({"TENSORHUB_TOKEN": "broad-token-must-not-cross"}, {}),
        ({}, {"origin": "https://hub.example.test/redirect"}),
        ({}, {"worker_token": "invalid*"}),
        ({}, {"worker_token": "A" * 42 + "B"}),
        ({}, {"worker_id": "another-worker"}),
        ({}, {"origin": ""}),
        ({}, {"version": 1}),
        ({"COZY_WORKER_AUTH_TOKEN": "A" * 43}, {}),
        ({}, {"grant": "not-boot-metadata"}),
    ):
        path.write_text(json.dumps(authority | change))
        denied = subprocess.run(
            [sys.executable, "-c", program],
            input=json.dumps({"env": handoff | env_change, "path": str(path)}),
            text=True,
            capture_output=True,
        )
        assert denied.returncode == 2 and denied.stdout.strip() == "worker_host_config"
    path.unlink()
    older_host = subprocess.run(
        [sys.executable, "-c", program],
        input=json.dumps({"env": handoff, "path": str(path)}),
        text=True,
        capture_output=True,
    )
    assert older_host.returncode == 0 and json.loads(older_host.stdout)["origin"] == ""


def test_object_storage_hosts_are_an_explicit_bounded_deployment_declaration() -> None:
    from cozy_runtime.internal.config import ConfigError, parse_object_storage_hosts

    assert parse_object_storage_hosts("") == ()
    assert parse_object_storage_hosts(".objects.example,storage.example") == (
        ".objects.example",
        "storage.example",
    )
    for value in (
        "*",
        ".",
        "https://storage.example",
        "UPPER.example",
        "b.example,a.example",
        "a.example,a.example",
        "a.example,,b.example",
        "a.example ",
        "a" * 254,
        ",".join(f"host{i}.example" for i in range(9)),
    ):
        with pytest.raises(ConfigError, match="bounded sorted unique"):
            parse_object_storage_hosts(value)


def test_publication_cannot_disable_tls_or_accept_an_unbound_grant() -> None:
    context = ssl.create_default_context()
    for grant in ("", "not-an-id", "00000000-0000-0000-0000-000000000000"):
        with pytest.raises(PublicationRefusal, match="authority_invalid"):
            MachinePublicationClient(
                "https://hub.example.test",
                grant,
                context,
                worker_id="worker-test",
                worker_token="A" * 43,
            )
    context.check_hostname = False
    with pytest.raises(PublicationRefusal, match="tls_verification_required"):
        MachinePublicationClient(
            "https://hub.example.test",
            "019aaaab-0000-7000-8000-000000000003",
            context,
            worker_id="worker-test",
            worker_token="A" * 43,
        )


def test_publication_authority_uses_default_verified_https_without_client_key() -> None:
    authority = PublicationAuthority("https://hub.example.test", "worker-test", "A" * 43)
    client = authority.client("019aaaab-0000-7000-8000-000000000003")
    assert client.context.check_hostname
    assert client.context.verify_mode == ssl.CERT_REQUIRED
    assert "A" * 43 not in repr(authority)
    for worker_id, token in (
        ("", "A" * 43),
        ("worker\r\nInjected: yes", "A" * 43),
        ("worker", ""),
        ("worker", "A" * 42 + "\n"),
    ):
        with pytest.raises(PublicationRefusal, match="worker_authority_invalid"):
            PublicationAuthority("https://hub.example.test", worker_id, token).client(
                "019aaaab-0000-7000-8000-000000000003"
            )


def test_worker_https_renews_and_keeps_worker_and_operation_authority_separate(
    tmp_path: Path,
) -> None:
    import threading
    from datetime import UTC, datetime, timedelta
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "server.pem", tmp_path / "server.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    observed: list[tuple[str, str, str, str]] = []
    authorization_id = "019aaaab-0000-7000-8000-000000000003"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.respond()

        def do_GET(self) -> None:
            self.respond()

        def respond(self) -> None:
            observed.append(
                (
                    self.path,
                    self.headers.get("X-Cozy-Worker-ID", ""),
                    self.headers.get("X-Cozy-Worker-Token", ""),
                    self.headers.get("Authorization", ""),
                )
            )
            if self.path.endswith("/token"):
                body = json.dumps(
                    {
                        "token": "scoped-operation-token",
                        "expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                    }
                ).encode()
                status = 200
            else:
                status, body = (401, b"{}") if len(observed) == 2 else (200, b'{"releases": []}')
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_tls.load_cert_chain(cert_path, key_path)
    server.socket = server_tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        context = ssl.create_default_context(cafile=str(cert_path))
        client = MachinePublicationClient(
            f"https://localhost:{server.server_port}",
            authorization_id,
            context,
            worker_id="worker-test",
            worker_token="A" * 43,
        )
        assert client.request("GET", "/v1/models/org/model") == b'{"releases": []}'
        renewal = (
            f"/v1/worker/machine-authorizations/{authorization_id}/token",
            "worker-test",
            "A" * 43,
            "",
        )
        publication = (
            "/v1/models/org/model",
            "worker-test",
            "A" * 43,
            "Bearer scoped-operation-token",
        )
        assert observed == [renewal, publication, renewal, publication]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
