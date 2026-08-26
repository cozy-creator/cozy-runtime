#!/usr/bin/env python
"""The REMOTE control leg, live: TLS + bearer owner token on Claim (cl-015, #445), and
the BYTE boundary that leg implies (#506b).

A weightless worker serves `WorkerControl` over TLS on loopback — the same leg a rented pod
exposes off-loopback — with a self-signed cert the owner PINS (trusting exactly that PEM is
the fingerprint pin) and the bearer token checked at Claim, constant-time. Sections:

    1. red arm: a wrong token is refused CLAIM_REJECTION_UNAUTHENTICATED, typed
    2. the real leg: right token over TLS -> Claim -> snapshot -> desired state ->
       DISPATCHABLE -> one invoke to a SUCCEEDED terminal
    3. red arm: a non-loopback plaintext listener REFUSES to bind at all
    4. red arm: a non-loopback TLS listener with no bootstrap credential refuses to bind
    5. the DECLARED GRANT ROOTS: a worker launched with `--grant-root` serves a grant
       inside them and REFUSES one outside every one of them. That is the byte boundary a
       remote worker actually has — a `file://` URL is an address on SOME machine, and a
       pod handed one rooted on the owner's disk would create the directory, write into it
       and declare outputs nobody can read.

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


def serve_env(home: Path, *, token: str) -> dict[str, str]:
    """The launch environment. An EMPTY token means the launcher provisioned none, which
    is the whole subject of section 4 — the variable is absent, not empty-but-present."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(home),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
    }
    if token:
        env["COZY_BOOTSTRAP_CREDENTIAL"] = token
    return env


def serve_argv(
    *,
    listen: str,
    root: Path,
    cert: Path | None = None,
    key: Path | None = None,
    grant_roots: str = "",
) -> list[str]:
    """ONE argv builder, so every arm below differs only in what it is testing."""
    return [
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
        *(("--grant-root", grant_roots) if grant_roots else ()),
        *(("--tls-cert", str(cert), "--tls-key", str(key)) if cert and key else ()),
    ]


def start_remote(
    tmp: Path, cert: Path, key: Path, *, listen: str, token: str, grant_roots: str = ""
) -> tuple[subprocess.Popen[bytes], Path, Path, dict[str, str]]:
    """The weightless worker on a TLS listener, its token provisioned via the environment
    (the one channel a launcher hands it through, #449/#463)."""
    home, root, project = tmp / "home", tmp / "worker", tmp / "project"
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    ids = ll.stage_weightless(home, project, ("tick",))
    log = tmp / f"worker-{len(os.listdir(tmp))}.log"
    (root / "control.addr").unlink(missing_ok=True)
    process = subprocess.Popen(
        serve_argv(listen=listen, root=root, cert=cert, key=key, grant_roots=grant_roots),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=serve_env(home, token=token),
        bufsize=0,
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

        wl.head("2. the real leg: TLS + the right token -> DISPATCHABLE -> invoke -> SUCCEEDED")
        hub = wl.Hub()
        hub.process, hub.log = process, log
        hub.directive = wl.serving_directive(
            ll.RELEASE, sorted(ids.values()), pb.Posture.POSTURE_ACCEPTING
        )
        t0 = time.perf_counter()
        hub.connect(addr, root_cert=root_cert, token=TOKEN)
        report = hub.ready(ids["tick"], 60)
        wl.check(
            "claimed and DISPATCHABLE over the TLS leg",
            report.available_attempt_slots > 0,
            f"{(time.perf_counter() - t0) * 1000:.0f} ms claim->DISPATCHABLE at {addr}, "
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
                m.attempt_outcome
                if m.WhichOneof("msg") == "attempt_outcome"
                and m.attempt_outcome.request_id == "req-remote"
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
    plain = subprocess.run(
        serve_argv(listen="0.0.0.0:0", root=tmp / "plain"),
        env=serve_env(tmp / "home", token=TOKEN),
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

    wl.head("4. red arm: a non-loopback TLS listener with NO bootstrap credential refuses")
    # THE OTHER HALF OF SECTION 3 (#493 item 10). TLS proves the WORKER's identity to the
    # dialer and says nothing about the dialer; with no credential configured the Claim
    # check is skipped entirely, so the first stranger to connect owned the machine. The
    # refusal is at BIND, before a port exists — the same shape as the plaintext refusal.
    anonymous = subprocess.run(
        serve_argv(listen="0.0.0.0:0", root=tmp / "anon", cert=cert, key=key),
        env=serve_env(tmp / "home", token=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = (anonymous.stderr + anonymous.stdout).strip()
    wl.check(
        "TLS off-loopback without a credential is a typed refusal to bind",
        anonymous.returncode != 0 and "without a bootstrap credential" in output,
        output.splitlines()[-1][:140] if output else f"exit {anonymous.returncode}",
    )
    # GREEN: the same argv with the credential provisioned binds and publishes its address.
    (tmp / "authed").mkdir(parents=True, exist_ok=True)
    authed = subprocess.Popen(
        serve_argv(listen="0.0.0.0:0", root=tmp / "authed", cert=cert, key=key),
        env=serve_env(tmp / "home", token=TOKEN),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    threading.Thread(target=wl._pump, args=(authed, tmp / "authed.log"), daemon=True).start()
    try:
        bound = wl.read_addr(tmp / "authed", authed)
        wl.check(
            "and the SAME listener with a credential provisioned binds and serves",
            bool(bound),
            f"bound {bound} with COZY_BOOTSTRAP_CREDENTIAL provisioned",
        )
    finally:
        authed.terminate()
        authed.wait(timeout=15)

    wl.head("5. the DECLARED GRANT ROOTS: inside is served, outside is refused")
    # THE BYTE BOUNDARY A REMOTE WORKER ACTUALLY HAS (#506b). This worker is launched the
    # way a pod's provisioner launches one: it is told the ONE subtree its co-resident
    # media server owns, and every `file://` grant must address it. The green arm and the
    # red arm differ in the destination URL and in nothing else.
    podfs = tmp / "podfs"
    podfs.mkdir(parents=True, exist_ok=True)
    fenced, froot, flog, fids = start_remote(
        tmp / "fenced", cert, key, listen="127.0.0.1:0", token=TOKEN, grant_roots=str(podfs)
    )
    (tmp / "fenced").mkdir(parents=True, exist_ok=True)
    try:
        faddr = wl.read_addr(froot, fenced)
        fhub = wl.Hub()
        fhub.process, fhub.log = fenced, flog
        fhub.directive = wl.serving_directive(
            ll.RELEASE, sorted(fids.values()), pb.Posture.POSTURE_ACCEPTING
        )
        fhub.connect(faddr, root_cert=root_cert, token=TOKEN)
        fhub.ready(fids["tick"], 60)

        # The GREEN arm: the payload is staged INSIDE the declared root, which is what a
        # pod's media server does with an uploaded input — it lands in the subtree the
        # media server owns and the grant names that path.
        wl.dispatch(
            fhub,
            podfs,
            fids["tick"],
            request_id="req-inside",
            payload={"note": "inside the declared root"},
            release=ll.RELEASE,
            outputs=0,
        )
        good = wl.terminal_body(
            fhub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-inside"
                    else None
                ),
                60,
                "inside-root terminal",
            )
        )
        wl.check(
            "a grant INSIDE the declared root is served to SUCCEEDED",
            good["status"] == 1,
            f"payload read from under {podfs}",
        )

        # The RED arm: the same request with the payload staged on the OWNER's disk — the
        # address a coordinator mints when it forgets there is a boundary (#493.3). Nothing
        # else moves: same worker, same plan, same declared inputs.
        elsewhere = tmp / "owner-only"
        elsewhere.mkdir(parents=True, exist_ok=True)
        wl.dispatch(
            fhub,
            elsewhere,
            fids["tick"],
            request_id="req-outside",
            payload={"note": "outside every declared root"},
            release=ll.RELEASE,
            outputs=0,
        )
        refused_terminal = wl.terminal_body(
            fhub.wait(
                lambda m: (
                    m.attempt_outcome
                    if m.WhichOneof("msg") == "attempt_outcome"
                    and m.attempt_outcome.request_id == "req-outside"
                    else None
                ),
                60,
                "outside-root terminal",
            )
        )
        said = str(refused_terminal.get("safe_message", "")) + str(
            refused_terminal.get("cause", {})
        )
        wl.check(
            "a grant OUTSIDE every declared root refuses BY NAME, before the bytes are read",
            refused_terminal["status"] != 1 and "grant_path_outside_roots" in said,
            f"status={refused_terminal['status']} {said[:150]}",
        )
        wl.check(
            "the two arms differ ONLY in the granted address",
            good["status"] == 1 and refused_terminal["status"] != 1,
            "same worker, same plan id, same payload shape, one path inside the declared "
            "root and one outside it",
        )
        fhub.close()
    finally:
        if fenced.poll() is None:
            fenced.terminate()
            fenced.wait(timeout=15)

    print(f"\nremote leg: {wl.PASS} ok, {wl.FAIL} failed")
    return 1 if wl.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
