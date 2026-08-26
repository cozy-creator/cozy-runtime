#!/usr/bin/env python
"""The REMOTE control leg, live: TLS + bearer owner token on Claim (cl-015, #445).

A weightless worker serves `WorkerControl` over TLS on loopback — the same leg a rented pod
exposes off-loopback — with a self-signed cert the owner PINS (trusting exactly that PEM is
the fingerprint pin) and the bearer token checked at Claim, constant-time. Three sections:

    1. red arm: a wrong token is refused CLAIM_REJECTION_UNAUTHENTICATED, typed
    2. the real leg: right token over TLS -> Claim -> snapshot -> directive -> READY ->
       one invoke to a SUCCEEDED terminal
    3. red arm: a non-loopback plaintext listener REFUSES to bind at all

    nice -n 19 uv run python scripts/remote-live.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_wl_spec = importlib.util.spec_from_file_location(
    "worker_live", ROOT / "scripts" / "worker-live.py"
)
assert _wl_spec is not None and _wl_spec.loader is not None
wl = importlib.util.module_from_spec(_wl_spec)
sys.modules["worker_live"] = wl
_wl_spec.loader.exec_module(wl)

_ll_spec = importlib.util.spec_from_file_location(
    "lifecycle_live", ROOT / "scripts" / "lifecycle-live.py"
)
assert _ll_spec is not None and _ll_spec.loader is not None
ll = importlib.util.module_from_spec(_ll_spec)
sys.modules["lifecycle_live"] = ll
_ll_spec.loader.exec_module(ll)

pb = wl.pb

TOKEN = "tok-" + os.urandom(16).hex()


def mint_cert(tmp: Path) -> tuple[Path, Path]:
    """A self-signed cert for 127.0.0.1 — exactly what a pod mints at provision."""
    cert, key = tmp / "worker.crt", tmp / "worker.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "2",
            "-subj",
            "/CN=cozy-worker",
            "-addext",
            "subjectAltName=IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def start_remote(tmp: Path, cert: Path, key: Path, *, listen: str, token: str) -> tuple:
    """The weightless worker on a TLS listener, its token provisioned via the environment
    (the one channel a launcher hands it through, #449/#463)."""
    home, root, project = tmp / "home", tmp / "worker", tmp / "project"
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    ids = ll.stage_weightless(home, project, ("tick",))
    log = tmp / f"worker-{len(os.listdir(tmp))}.log"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(home),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        "COZY_BOOTSTRAP_CREDENTIAL": token,
    }
    (root / "control.addr").unlink(missing_ok=True)
    argv = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-c",
        "import sys; from cozy_runtime.internal.worker.session import main; "
        "raise SystemExit(main(sys.argv[1:]))",
        "--listen",
        listen,
        "--root",
        str(root),
        "--instance-id",
        "ins-remote",
        "--release-id",
        ll.RELEASE,
        "--grace",
        "3.0",
        "--no-warm",
        "--tls-cert",
        str(cert),
        "--tls-key",
        str(key),
    ]
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0
    )
    threading.Thread(target=wl._pump, args=(process, log), daemon=True).start()
    return process, root, log, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp", default="/tmp/cozy-remote-live")
    args = parser.parse_args()
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    cert, key = mint_cert(tmp)
    root_cert = cert.read_bytes()

    process, root, log, ids = start_remote(tmp, cert, key, listen="127.0.0.1:0", token=TOKEN)
    try:
        addr = wl.read_addr(root, process)
        wl.head(f"1. red arm: a WRONG bearer token at {addr}")
        bad = wl.Hub()
        bad.process, bad.log = process, log
        bad.connect(addr, claim_only=True, root_cert=root_cert, token="tok-wrong")
        refused = bad.wait(
            lambda m: m.claim_ack if m.WhichOneof("msg") == "claim_ack" else None,
            30,
            "the refusal",
        )
        wl.check(
            "a wrong token is refused CLAIM_REJECTION_UNAUTHENTICATED",
            not refused.accepted
            and refused.rejection == pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED,
            f"rejection={pb.ClaimRejection.Name(refused.rejection)} over TLS "
            "(the token never rode plaintext)",
        )
        bad.close()

        wl.head("2. the real leg: TLS + the right token -> READY -> invoke -> SUCCEEDED")
        hub = wl.Hub()
        hub.process, hub.log = process, log
        hub.directive = wl.serving_directive(
            ll.RELEASE, sorted(ids.values()), pb.Posture.POSTURE_ACCEPTING
        )
        t0 = time.perf_counter()
        hub.connect(addr, root_cert=root_cert, token=TOKEN)
        report = hub.ready(ids["tick"], 60)
        wl.check(
            "claimed and READY over the TLS leg",
            bool(hub.deployment(report).attempt_credits),
            f"{(time.perf_counter() - t0) * 1000:.0f} ms claim->READY at {addr}, "
            f"boot {hub.boot_id}",
        )
        t1 = time.perf_counter()
        wl.dispatch(
            hub,
            tmp,
            ids["tick"],
            request_id="req-remote",
            payload={"note": "over-tls"},
            release=ll.RELEASE,
            outputs=0,
        )
        terminal = hub.wait(
            lambda m: (
                m.attempt_terminal
                if m.WhichOneof("msg") == "attempt_terminal"
                and m.attempt_terminal.request_id == "req-remote"
                else None
            ),
            60,
            "remote terminal",
        )
        body = wl.terminal_body(terminal)
        wl.check(
            "one invoke lands a SUCCEEDED terminal over TLS + token",
            body["status"] == 1,
            f"{(time.perf_counter() - t1) * 1000:.0f} ms dispatch->terminal",
        )
        hub.close()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)

    wl.head("3. red arm: a non-loopback PLAINTEXT listener refuses to bind")
    p2, root2, _log2, _ = start_remote(tmp, cert, key, listen="0.0.0.0:0", token=TOKEN)
    # start_remote always passes TLS; strip it by launching once more WITHOUT the flags:
    p2.terminate()
    p2.wait(timeout=10)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(tmp / "home"),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
    }
    plain = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cozy_runtime.internal.worker.session import main; "
            "raise SystemExit(main(sys.argv[1:]))",
            "--listen",
            "0.0.0.0:0",
            "--root",
            str(root2),
            "--instance-id",
            "ins-plain",
            "--release-id",
            ll.RELEASE,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    wl.check(
        "plaintext off-loopback is a typed refusal to bind, not a listener",
        plain.returncode != 0 and "without TLS" in (plain.stderr + plain.stdout),
        (plain.stderr + plain.stdout).strip().splitlines()[-1][:120]
        if (plain.stderr + plain.stdout).strip()
        else f"exit {plain.returncode}",
    )

    print(f"\nremote leg: {wl.PASS} ok, {wl.FAIL} failed")
    return 1 if wl.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
