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
    authority = {"version": 1, "origin": "https://hub.example.test", "ca_der_base64url": ""}
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
                  "hosts":runtime_config_from_host(cfg).object_storage_hosts}))
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
    }
    for env_change, change in (
        ({"TENSORHUB_TOKEN": "broad-token-must-not-cross"}, {}),
        ({}, {"origin": "https://hub.example.test/redirect"}),
        ({}, {"ca_der_base64url": "invalid*"}),
        ({}, {"origin": "", "ca_der_base64url": "Y2E"}),
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
            MachinePublicationClient("https://hub.example.test", grant, context)
    context.check_hostname = False
    with pytest.raises(PublicationRefusal, match="tls_verification_required"):
        MachinePublicationClient(
            "https://hub.example.test", "019aaaab-0000-7000-8000-000000000003", context
        )
