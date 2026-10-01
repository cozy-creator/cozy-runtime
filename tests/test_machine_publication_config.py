"""Boot authority remains privileged; no provider/account secret reaches a job."""

from __future__ import annotations

import base64
import json
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cozy_runtime.internal.worker.machine_publication import (
    MachinePublicationClient,
    PublicationAuthority,
    PublicationRefusal,
    registrations,
)


def execution_token(
    subject: str = "account-a", nonce: str = "first", issuer: str = "issuer"
) -> str:
    """A local transport fixture; production Hub remains the signature verifier."""
    parts = [
        {"typ": "delegated-access+jwt", "alg": "EdDSA"},
        {
            "iss": issuer,
            "delegated_sub": subject,
            "permissions": ["cozy.execution-access"],
            "jti": nonce,
        },
    ]
    return (
        ".".join(
            base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")
            for part in parts
        )
        + ".fixture-signature"
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
                  "visible_devices":cfg.visible_devices,
                  "hosts":runtime_config_from_host(cfg).object_storage_hosts,
                  "dependency_cache":str(runtime_config_from_host(cfg).dependency_cache),
                  "tensorfs":str(cfg.layout.tensorfs_root)}))
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
        "visible_devices": None,
        "hosts": [".objects.example", "storage.example"],
        "dependency_cache": "/var/lib/cozy/dependencies",
        "tensorfs": "/var/lib/tensorfs",
    }
    # The centralized boot reader freezes the operator mask independently of the
    # erased device-child environment; later process-environment changes cannot replace it.
    for visible in ("", "1", "GPU-a", "1,GPU-a"):
        masked = subprocess.run(
            [sys.executable, "-c", program],
            input=json.dumps(
                {"env": handoff | {"CUDA_VISIBLE_DEVICES": visible}, "path": str(path)}
            ),
            text=True,
            capture_output=True,
        )
        assert masked.returncode == 0, masked.stderr
        captured = json.loads(masked.stdout)
        assert captured["visible_devices"] == [part for part in visible.split(",") if part]
        assert "CUDA_VISIBLE_DEVICES" not in captured["child"]
    # The Host names the box's one Store; the rest of the layout stays under the root.
    named = subprocess.run(
        [sys.executable, "-c", program],
        input=json.dumps(
            {"env": handoff | {"COZY_TENSORFS_ROOT": "/home/o/.tensorfs"}, "path": str(path)}
        ),
        text=True,
        capture_output=True,
    )
    assert named.returncode == 0, named.stderr
    assert json.loads(named.stdout)["tensorfs"] == "/home/o/.tensorfs"
    assert json.loads(named.stdout)["dependency_cache"] == "/var/lib/cozy/dependencies"
    # Settings from newer hosts (including credentials irrelevant to this
    # worker) neither abort boot nor enter the child environment.
    extended_auth = json.loads(handoff["COZY_RECORD_OWNER_AUTH_JSON"])
    extended_auth["future_advisory"] = {"revision": 3}
    evolved = subprocess.run(
        [sys.executable, "-c", program],
        input=json.dumps(
            {
                "env": handoff
                | {
                    "COZY_FUTURE_SETTING": "advisory",
                    "TENSORHUB_TOKEN": "broad-token-must-not-cross",
                    "COZY_WORKER_AUTH_TOKEN": "A" * 43,
                    "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(extended_auth, indent=2),
                },
                "path": str(path),
            }
        ),
        text=True,
        capture_output=True,
    )
    assert evolved.returncode == 0, evolved.stderr
    assert json.loads(evolved.stdout) == json.loads(result.stdout)
    for env_change, change in (
        ({"PYTHONPATH": "/untrusted/source"}, {}),
        ({"COZY_TENSORFS_ROOT": "relative/.tensorfs"}, {}),
        ({"COZY_TENSORFS_ROOT": "/home/o/../.tensorfs"}, {}),
        ({"COZY_RECORD_OWNER_AUTH_JSON": json.dumps({"future_advisory": True})}, {}),
        (
            {
                "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(
                    extended_auth | {"control_public_key_ed25519_b64url": "invalid"}
                )
            },
            {},
        ),
        ({}, {"origin": "https://hub.example.test/redirect"}),
        ({}, {"worker_token": "invalid*"}),
        ({}, {"worker_token": "A" * 42 + "B"}),
        ({}, {"worker_id": "another-worker"}),
        ({}, {"origin": ""}),
        ({}, {"version": 1}),
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


def test_other_hub_registrations_are_read_tolerantly(tmp_path: Path) -> None:
    """The Host's `machine-hubs.json` names each other Hub's grant, CA and object hosts. An
    entry this Runtime cannot read, a member it does not know, or a newer document confers
    only what it can and never stops the machine."""
    path = tmp_path / "machine-hubs.json"
    assert registrations(path) == ((), [])
    (tmp_path / "hub-ca-1.crt").write_text("ca")
    good = {
        "origin": "https://hub.example.test",
        "worker_id": "worker-2",
        "worker_token": "A" * 43,
        "ca": "hub-ca-1.crt",
        "object_storage_hosts": ["storage.example", ".objects.example"],
        "added_later": True,
    }
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "hubs": [
                    good,
                    {**good, "origin": "http://hub.example.test"},
                    {**good, "ca": "../escape.crt"},
                    {**good, "worker_token": "short"},
                    {"origin": 7},
                ],
                "next": {},
            }
        )
    )
    held, skipped = registrations(path)
    assert held == (
        PublicationAuthority(
            "https://hub.example.test",
            "worker-2",
            "A" * 43,
            tmp_path / "hub-ca-1.crt",
            (".objects.example", "storage.example"),
        ),
    )
    assert len(skipped) == 4
    path.write_text(json.dumps({"version": 2, "hubs": [good]}))
    assert registrations(path)[0] == ()
    path.write_text("not json")
    assert registrations(path)[0] == ()


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


@pytest.mark.parametrize("delegated,tls", [(False, True), (True, True), (True, False)])
def test_worker_https_renews_and_keeps_worker_and_operation_authority_separate(
    tmp_path: Path,
    delegated: bool,
    tls: bool,
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
    observed: list[tuple[str, str, str, str, str]] = []
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
                    self.headers.get("X-Cozy-Execution-Access", ""),
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
    if tls:
        server.socket = server_tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        context = ssl.create_default_context(cafile=str(cert_path))
        origin = f"{'https' if tls else 'http'}://localhost:{server.server_port}"
        if delegated:
            path = tmp_path / "machine-hubs.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "hubs": [
                            {
                                "origin": origin,
                                "access_token": execution_token(),
                                "expires_at": int(time.time()) + 3600,
                                "ca": cert_path.name,
                            }
                        ],
                    }
                )
            )
            authority = registrations(path)[0][0]
            client = authority.client(authorization_id)
            # An already-constructed client re-reads the atomically renewed projection.
            path.write_text(
                path.read_text().replace(execution_token(), execution_token(nonce="renewed"))
            )
        else:
            client = MachinePublicationClient(
                origin,
                authorization_id,
                context,
                worker_id="worker-test",
                worker_token="A" * 43,
            )
        assert client.request("GET", "/v1/models/org/model") == b'{"releases": []}'
        renewal = (
            f"/v1/worker/machine-authorizations/{authorization_id}/token",
            "" if delegated else "worker-test",
            "" if delegated else "A" * 43,
            "",
            execution_token(nonce="renewed") if delegated else "",
        )
        publication = (
            "/v1/models/org/model",
            "" if delegated else "worker-test",
            "" if delegated else "A" * 43,
            "Bearer scoped-operation-token",
            "",
        )
        assert observed == [renewal, publication, renewal, publication]
        if delegated:
            path.write_text(
                path.read_text().replace(
                    execution_token(nonce="renewed"), execution_token("account-b")
                )
            )
            with pytest.raises(PublicationRefusal, match="authority_principal_conflict"):
                client.request("GET", "/v1/models/org/model")
            assert observed == [renewal, publication, renewal, publication]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "origin",
    [
        "https://hub.example.test",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
    ],
)
def test_scoped_access_renews_without_machine_registration_and_revokes_per_hub(
    tmp_path: Path,
    origin: str,
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from cozy_runtime.internal.worker import machine_model_defaults, machine_model_resolve
    from cozy_runtime.internal.worker.session import Worker, WorkerOptions

    path = tmp_path / "machine-hubs.json"
    row = {"origin": origin, "access_token": execution_token(), "expires_at": int(time.time()) + 60}
    path.write_text(json.dumps({"version": 1, "hubs": [row, {"origin": "unreadable"}]}))
    worker = cast(
        Worker,
        SimpleNamespace(
            options=WorkerOptions(tmp_path, hub_access_path=path),
            hub_access_lock=threading.Lock(),
            hub_access_principals={},
        ),
    )
    assert machine_model_resolve.registration(worker, "") is None
    authority = machine_model_resolve.registration(worker, origin)
    assert authority is not None
    assert authority.worker_id == authority.worker_token == ""
    assert authority.headers() == {"Authorization": "Bearer " + execution_token()}
    assert execution_token() not in repr(authority)
    assert machine_model_defaults.credential(worker, origin) == "bearer " + execution_token()
    assert machine_model_defaults.credential(worker, "https://other.example.test") == ""
    path.write_text(
        json.dumps(
            {"version": 1, "hubs": [{**row, "access_token": execution_token(nonce="renewed")}]}
        )
    )
    assert machine_model_defaults.credential(worker, origin) == "bearer " + execution_token(
        nonce="renewed"
    )
    for token in (
        execution_token("account-b"),
        execution_token(issuer="other-issuer"),
        "opaque-token",
    ):
        path.write_text(json.dumps({"version": 1, "hubs": [{**row, "access_token": token}]}))
        for read in (
            lambda: machine_model_defaults.credential(worker, origin),
            lambda: machine_model_resolve.own(worker, hub=origin),
        ):
            with pytest.raises(ValueError, match="another account"):
                read()
        with pytest.raises(PublicationRefusal, match="authority_principal_conflict"):
            authority.current()
        assert machine_model_defaults.credential(worker, "https://other.example.test") == ""
    path.write_text(json.dumps({"version": 1, "hubs": [{**row, "expires_at": 1}]}))
    with pytest.raises(PublicationRefusal, match="authority_expired"):
        machine_model_defaults.credential(worker, origin)
    with pytest.raises(ValueError, match="execution access has expired"):
        machine_model_resolve.own(worker, hub=origin)
    path.write_text(json.dumps({"version": 1, "hubs": []}))
    assert machine_model_resolve.registration(worker, origin) is None
    with pytest.raises(PublicationRefusal, match="authority_absent"):
        authority.current()


def test_scoped_plain_http_is_only_explicit_loopback_access(tmp_path: Path) -> None:
    from cozy_runtime.internal.config import ConfigError

    path = tmp_path / "machine-hubs.json"
    token = {"access_token": "scoped-token", "expires_at": int(time.time()) + 60}
    for origin in (
        "http://hub.example.test",
        "http://192.168.0.1:8080",
        "http://localhost.evil.test",
        "http://localhost/path",
        "http://user:pass@localhost",
        "http://localhost:0",
        "http://localhost#fragment",
        "http://localhost?query",
        "http://LOCALHOST",
    ):
        path.write_text(json.dumps({"version": 1, "hubs": [{"origin": origin, **token}]}))
        held, skipped = registrations(path)
        assert not held and skipped
    for origin in ("http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080"):
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "hubs": [
                        {
                            "origin": origin,
                            "worker_id": "rental",
                            "worker_token": "A" * 43,
                        }
                    ],
                }
            )
        )
        held, skipped = registrations(path)
        assert not held and skipped  # Rental grant transport has not changed.
        with pytest.raises(ConfigError):
            PublicationAuthority(origin, "rental", "A" * 43).client(
                "019aaaab-0000-7000-8000-000000000003"
            )


def test_boot_account_binding_survives_projection_replacement_without_blocking_other_hubs(
    tmp_path: Path,
) -> None:
    from cozy_runtime.internal.config import Credentials, RuntimeConfig
    from cozy_runtime.internal.worker import machine_model_defaults, machine_model_resolve
    from cozy_runtime.internal.worker.control import InMemoryControlHost
    from cozy_runtime.internal.worker.session import Worker, WorkerOptions

    origin, other = "https://hub.example.test", "https://other.example.test"
    path = tmp_path / "machine-hubs.json"
    original = PublicationAuthority(origin, access_token=execution_token(), expires_at=1)
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(tmp_path / "worker", hubs=(original,), hub_access_path=path),
        InMemoryControlHost(),
    )
    try:
        rows = [
            {
                "origin": origin + ":443",
                "access_token": execution_token("account-b"),
                "expires_at": int(time.time()) + 60,
            },
            {
                "origin": other,
                "access_token": execution_token("account-c"),
                "expires_at": int(time.time()) + 60,
            },
        ]
        path.write_text(json.dumps({"version": 1, "hubs": rows}))
        with pytest.raises(ValueError, match="another account"):
            machine_model_resolve.registration(worker, origin)
        assert machine_model_defaults.credential(worker, other) == "bearer " + execution_token(
            "account-c"
        )
        rows[0]["access_token"] = execution_token(nonce="fresh")
        path.write_text(json.dumps({"version": 1, "hubs": rows}))
        assert machine_model_defaults.credential(worker, origin) == "bearer " + execution_token(
            nonce="fresh"
        )
    finally:
        worker.shutdown()
