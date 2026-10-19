"""A fresh CPU helper resolves only cached model metadata and exits without entering a package."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from cozy_runtime.internal.seam import Channel
from store_fixtures import checkpoint, home
from test_executor_fork import _call, _environment, _listener


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _hub() -> dict[str, str]:
    key = ec.generate_private_key(ec.SECP256R1())
    numbers = key.private_numbers()
    public = numbers.public_numbers
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        **{
            name: _b64(value.to_bytes(32, "big"))
            for name, value in (("x", public.x), ("y", public.y), ("d", numbers.private_value))
        },
    }
    header = _b64(b'{"alg":"ES256"}')
    payload = _b64(b'{"aud":"https://authority.invalid","jti":"warm-metadata"}')
    signed = f"{header}.{payload}"
    r, s = decode_dss_signature(key.sign(signed.encode(), ec.ECDSA(hashes.SHA256())))
    signature = _b64(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    return {
        "endpoint": "http://127.0.0.1:9",
        "token_endpoint": "http://127.0.0.1:9/oauth/token",
        "capability": f"{signed}.{signature}",
        "key": json.dumps(jwk),
    }


def test_metadata_helper_never_starts_a_package_or_trades_for_cached_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "tensorfs"
    store = home(root, monkeypatch)
    header = checkpoint(store, {"model": {"w": np.ones(2, np.float32)}})
    ck = store.name("acme/metadata", "1", store.checkpoint(header))
    assert ck.tree is not None and ck.header_ref is not None
    frame = {
        "cmd": "resolve_models", "request_id": "metadata", "store": str(root), "hub": _hub(),
        "models": [{"slot": "source", "repo_id": "acme/metadata", "revision": ck.tree,
                    "variant": ck.variant or ""}],
    }  # fmt: skip
    environment = _environment(tmp_path)
    environment["TENSORFS_HOME"] = str(root)
    worker = tmp_path / "worker"
    worker.mkdir()
    socket_path = tmp_path / "metadata.sock"
    with _listener(socket_path) as listener:
        listener.settimeout(30)
        process = subprocess.Popen(
            [sys.executable, "-m", "cozy_runtime.internal.executor", "--socket", str(socket_path),
             "--root", str(worker)],
            env=environment, stderr=subprocess.PIPE,
        )  # fmt: skip
        try:
            connection, _ = listener.accept()
            with connection:
                channel = Channel(connection)
                assert _call(channel, {"cmd": "hello"})["torch_loaded"] is False
                reply = _call(channel, frame)
                assert reply["ok"], reply
                assert "access_token" not in reply, "a fully cached header trades zero times"
                (resolved,) = reply["resolved"]
                assert resolved["header"] == {
                    "sha256": header.removeprefix("sha256:"),
                    "length": ck.header_ref[1],
                }
                assert resolved["tree"]["sha256"] == ck.tree.removeprefix("sha256:")
                assert not store.checkpoint(header).missing()
                refused = _call(channel, {**frame, "store": str(tmp_path / "other-store")})
                assert refused["code"] == "store_root_changed", refused
                assert _call(channel, {"cmd": "hello"})["torch_loaded"] is False
                maps = Path(f"/proc/{process.pid}/maps").read_text()
                assert "libcuda" not in maps and "libnvidia-ml" not in maps
                assert _call(channel, {"cmd": "shutdown"})["ok"]
            _, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr.decode()[-2000:]
            assert not Path(f"/proc/{process.pid}").exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
