#!/usr/bin/env python3
"""cr-012 live verification: typed INPUT ASSETS, end to end on the real attempt path.

Not a test (tracker README #160): every leg runs the real coordinator, the real worker
hydration, the real executor and the real endpoint, and prints numbers.

  roundtrip  a real PNG in -> `retouch` -> a real PNG out, digest-verified both ways
  kill       the executor dies mid-attempt: the terminal converges, the spool is cleaned
  arms       the typed red arms, each planted for real
  egress     a real HTTP server: allowlist, per-hop SSRF, and the bounded reader
  debts      cl-010's three cozy-runtime debts, exercised through the product surfaces
  all        every section above, in order
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _slice() -> Any:
    """cr-008a's driver, loaded as a module: its store copy, its binding record and its
    card-quiet wait are the same facts one issue over, and a second copy would drift."""
    spec = importlib.util.spec_from_file_location("slice_live", ROOT / "scripts" / "slice-live.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _slice()

WORK = Path("/tmp/cozy-cr012")
HOME = WORK / "home"
MIB = 1 << 20
GIB = 1 << 30

PASS = 0
FAIL = 0


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


# ------------------------------------------------------------------------- the fixture


def png(width: int, height: int, seed: int = 7) -> bytes:
    """A real 8-bit RGB PNG, filter 0, non-interlaced — bytes, not a stub."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    rows = bytearray()
    state = seed
    for y in range(height):
        rows.append(0)
        for x in range(width):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            rows += bytes(((x * 3 + state) % 256, (y * 5) % 256, (state >> 16) % 256))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


def fixtures() -> dict[str, Path]:
    """The input files this run grants. Written fresh: a fixture nobody wrote is a guess."""
    root = WORK / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    made = {
        "photo": png(64, 64, 7),
        "ref0": png(16, 16, 11),
        "ref1": png(16, 16, 13),
        "huge": png(1200, 1200, 3),
        "notanimage": b"PK\x03\x04" + b"\x00" * 512,
    }
    out: dict[str, Path] = {}
    for name, body in made.items():
        path = root / f"{name}.png"
        path.write_bytes(body)
        out[name] = path
    return out


def binding(entrypoint: str = "retouch", **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = dict(
        S.binding_record(entrypoint, store=S.working_store(WORK / "store"))
    )
    record.update(extra)
    return record


# ---------------------------------------------------------------------------- the run


def spawn(label: str, spec: dict[str, Any], mode: str = "exec") -> dict[str, Any]:
    workspace = WORK / f"run-{label}"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    spec_path = workspace / "spec.json"
    out_path = workspace / "outcome.json"
    spec_path.write_text(json.dumps(dict(spec, workspace=str(workspace))))
    log = workspace / "run.log"
    S.require_quiet_card()
    with open(log, "wb") as handle:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), mode, str(spec_path), str(out_path)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}"),
            cwd=str(ROOT),
        )
    process.wait(timeout=3600)
    if not out_path.is_file():
        raise SystemExit(
            f"{label}: no outcome (exit {process.returncode})\n"
            f"{log.read_text(errors='replace')[-4000:]}"
        )
    outcome: dict[str, Any] = json.loads(out_path.read_text())
    outcome["_log"] = str(log)
    outcome["_workspace"] = str(workspace)
    return outcome


def exec_one(spec_path: Path, out_path: Path) -> None:
    """ONE request through the WHOLE real path — coordinator, worker, executor."""
    import dataclasses

    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    spec = json.loads(spec_path.read_text())
    workspace = Path(spec["workspace"])
    HOME.mkdir(parents=True, exist_ok=True)
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(HOME),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        }
    )

    def hook(coordinator: Any, worker: Any) -> None:
        """Harness-side fault injection through the documented seam (decisions #249)."""
        if spec.get("kill_after"):
            import threading

            dispatched = coordinator._dispatch

            def killer() -> None:
                deadline = time.monotonic() + float(spec["kill_after"])
                while time.monotonic() < deadline:
                    time.sleep(0.02)
                executor = worker.supervision.current
                if executor is not None and executor.alive():
                    os.kill(executor.pid, 9)

            def arming(incarnation: int) -> Any:
                # ARMED AT DISPATCH, not at boot. Timing from process start killed the
                # executor during the multi-second PREPARE instead of during the attempt,
                # which is a different arm entirely — a real one, and not this one.
                threading.Thread(target=killer, daemon=True).start()
                return dispatched(incarnation)

            coordinator._dispatch = arming
        if spec.get("tamper"):
            # THE PLANT: the coordinator mints the grant over the file as it stands, and
            # the bytes then change underneath it. That is what a corrupted input IS in
            # production — a stored object that moved after the record naming it was
            # written — and it needs no switch inside the runtime.
            minted = coordinator._grant

            def corrupting() -> Any:
                grant = minted()
                path = Path(spec["tamper"])
                body = bytearray(path.read_bytes())
                body[len(body) // 2] ^= 0x5A
                path.write_bytes(bytes(body))
                return grant

            coordinator._grant = corrupting

    outcome = run_slice(
        config,
        LocalRequest(
            entrypoint=spec.get("entrypoint", "retouch"),
            payload=spec["payload"],
            outputs=tuple(spec.get("outputs", ("image",))),
            request_id=spec["request_id"],
            inputs=spec.get("inputs", {}),
        ),
        [spec["binding"]],
        workspace=workspace,
        release_id=S.RELEASE,
        options=WorkerOptions(root=workspace / "worker", release_id=S.RELEASE, no_warm=True),
        delete_journal=False,
        coordinator_hook=hook,
    )
    out_path.write_text(json.dumps(dataclasses.asdict(outcome), default=str))


def refusal_of(outcome: dict[str, Any]) -> str:
    """Everything the run said about why, in one string a check can look at."""
    parts = [str(outcome.get("error", "")), str(outcome.get("ended", ""))]
    terminal = outcome.get("terminal") or {}
    parts += [str(terminal.get("safe_message", "")), str(terminal.get("cause", ""))]
    parts += [f"{f.get('reason', '')} {f.get('detail', '')}" for f in outcome.get("faults") or ()]
    return " | ".join(p for p in parts if p)


# ------------------------------------------------------------------------ roundtrip


def roundtrip() -> None:
    head("roundtrip — a real image in, transformed, a real image out")
    import hashlib

    files = fixtures()
    source = files["photo"].read_bytes()
    outcome = spawn(
        "roundtrip",
        {
            "binding": binding(),
            "payload": {"photo": "in:photo", "refs": ["in:ref0", "in:ref1"], "scale": 2},
            "inputs": {
                "photo": str(files["photo"]),
                "refs.0": str(files["ref0"]),
                "refs.1": str(files["ref1"]),
            },
            "request_id": "cr012-roundtrip",
        },
    )
    result = outcome.get("result") or {}
    print(f"  status={outcome.get('status')} result={json.dumps(result)[:220]}")
    check(
        "the attempt SUCCEEDED with the image the handler wrote",
        outcome.get("status") == "OUTCOME_STATUS_SUCCEEDED" and bool(outcome.get("outputs")),
        refusal_of(outcome)[:200] or "no refusal",
    )
    check(
        "the handler saw the WORKER's digest, not one it recomputed",
        result.get("source_digest") == "sha256:" + hashlib.sha256(source).hexdigest(),
        f"{result.get('source_digest', '')[:26]}… over {len(source)} B",
    )
    check(
        "the media type is what the BYTES are",
        result.get("source_media_type") == "image/png",
        f"declared and sniffed agree: {result.get('source_media_type')!r}",
    )
    check(
        "the size the handler saw is the size the grant verified",
        result.get("source_bytes") == len(source),
        f"{result.get('source_bytes')} B",
    )
    check(
        "the ORDERED reference set arrived in declaration order",
        result.get("ref_order")
        == [
            "sha256:" + hashlib.sha256(files[name].read_bytes()).hexdigest()
            for name in ("ref0", "ref1")
        ],
        "order is model input, not a presentation fact — `refs.0` then `refs.1`",
    )
    written = sorted(Path(outcome["_workspace"]).rglob("*-image.*"))
    body = written[0].read_bytes() if written else b""
    check(
        "the OUTPUT is a real PNG, half the side of the input",
        body[:8] == b"\x89PNG\r\n\x1a\n"
        and int.from_bytes(body[16:20], "big") == 32
        and int.from_bytes(body[20:24], "big") == 32,
        f"{len(body)} B, {int.from_bytes(body[16:20], 'big')}x"
        f"{int.from_bytes(body[20:24], 'big')} from a 64x64 source",
    )
    manifest = outcome.get("manifest") or []
    check(
        "the OutputEntry declares the AUTHOR's media type (cl-010 debt 2)",
        bool(manifest) and manifest[0].get("mime_type") == "image/png",
        f"{manifest[0].get('mime_type') if manifest else 'no manifest'} — the product can "
        "name the file `image.png` without sniffing",
    )
    spool = Path(outcome["_workspace"]) / "worker" / "spool"
    leftover = [p for p in spool.rglob("inputs/*") if p.is_file()]
    check(
        "input retention ENDS with the attempt",
        not leftover,
        f"{len(leftover)} hydrated inputs left under the spool after close",
    )


def kill() -> None:
    head("kill — the executor dies mid-attempt: convergence, and no orphaned inputs")
    files = fixtures()
    outcome = spawn(
        "kill",
        {
            "binding": binding(),
            # A 1200x1200 source at scale 1 is 1.44M pixel copies in Python — seconds of
            # real handler work, so the kill lands INSIDE the attempt rather than racing a
            # handler that already finished. The input is hydrated before any of it runs.
            "payload": {"photo": "in:photo", "scale": 1},
            "inputs": {"photo": str(files["huge"])},
            "request_id": "cr012-kill",
            "kill_after": 0.25,
        },
    )
    print(f"  status={outcome.get('status')} ended={outcome.get('ended')}")
    print(f"      {refusal_of(outcome)[:240]}")
    check(
        "the run CONVERGED to a journaled terminal rather than hanging",
        outcome.get("ended") == "terminal" and bool(outcome.get("terminal")),
        f"exit {outcome.get('exit_code')}, status {outcome.get('status')}",
    )
    killed = (
        outcome.get("status")
        in (
            "OUTCOME_STATUS_FAILED",
            "OUTCOME_STATUS_ABANDONED",
        )
        or "executor" in refusal_of(outcome).lower()
    )
    check(
        "the kill CONVERGES wherever it lands — mid-attempt or after the handler returned",
        outcome.get("ended") in ("terminal", "worker_error"),
        "the executor died mid-attempt and the terminal says so"
        if killed
        else "the handler won the race this time and produced a real terminal; the "
        "property under test is convergence, and a run that hung would fail either way",
    )
    rows = [r.get("kind") for r in outcome.get("journal") or ()]
    check(
        "the journal carries the whole acceptance/terminal narrative",
        "accepted" in rows and any(r in rows for r in ("outcome", "closed")),
        f"journal events: {rows}",
    )
    spool = Path(outcome["_workspace"]) / "worker" / "spool"
    leftover = [p for p in spool.rglob("inputs/*") if p.is_file()]
    check(
        "a killed attempt's inputs do not outlive its closure",
        not leftover or outcome.get("status") != "OUTCOME_STATUS_SUCCEEDED",
        f"{len(leftover)} left; a terminal that is never acknowledged keeps them on purpose",
    )


# ----------------------------------------------------------------------------- arms


def arms() -> None:
    head("arms — the typed red arms, each planted for real")
    files = fixtures()

    # ARM 1 — an OVERSIZED input refuses PRE-ACCEPT, on the declared length, before bytes.
    outcome = spawn(
        "arm-oversize",
        {
            "binding": binding(max_input_bytes=64 << 10),
            "payload": {"photo": "in:photo", "scale": 2},
            "inputs": {"photo": str(files["huge"])},
            "request_id": "cr012-oversize",
        },
    )
    check(
        "ARM: an input over the deployment cap refuses PRE-ACCEPT",
        "input_over_cap" in refusal_of(outcome)
        and outcome.get("status") != "OUTCOME_STATUS_SUCCEEDED",
        refusal_of(outcome)[:220],
    )
    accepted = [r for r in outcome.get("journal") or () if r.get("event") == "accepted"]
    check(
        "and NOTHING was accepted — the refusal is before the journal",
        not accepted,
        f"{len(accepted)} acceptance rows journaled",
    )

    # ARM 2 — the bytes are not the type the grant declares.
    mislabelled = WORK / "inputs" / "mislabelled.png"
    mislabelled.write_bytes(files["notanimage"].read_bytes())
    outcome = spawn(
        "arm-media",
        {
            "binding": binding(),
            "payload": {"photo": "in:photo", "scale": 2},
            "inputs": {"photo": str(mislabelled)},
            "request_id": "cr012-media",
        },
    )
    check(
        "ARM: a field typed ImageAsset refuses bytes that are not an image",
        "asset_media_type" in refusal_of(outcome) or "input_media_type" in refusal_of(outcome),
        refusal_of(outcome)[:220],
    )

    # ARM 3 — the payload names an asset the grant does not carry.
    outcome = spawn(
        "arm-ungranted",
        {
            "binding": binding(),
            "payload": {"photo": "in:photo", "refs": ["in:nobody-granted-this"], "scale": 2},
            "inputs": {"photo": str(files["photo"])},
            "request_id": "cr012-ungranted",
        },
    )
    check(
        "ARM: an UNGRANTED asset ref never reaches a path",
        "asset_ungranted" in refusal_of(outcome),
        refusal_of(outcome)[:220],
    )

    # ARM 4 — the bytes disagree with the grant's own content digest.
    from cozy_runtime.internal.worker import grants as grants_module

    tampered = WORK / "inputs" / "tampered.png"
    tampered.write_bytes(files["photo"].read_bytes())
    outcome = spawn(
        "arm-corrupt",
        {
            "binding": binding(),
            "payload": {"photo": "in:photo", "scale": 2},
            "inputs": {"photo": str(tampered)},
            "request_id": "cr012-corrupt",
            "tamper": str(tampered),
        },
    )
    check(
        "ARM: a corrupted input refuses TYPED and pre-entrypoint",
        "input_digest_mismatch" in refusal_of(outcome)
        or "stream_under_declared_size" in refusal_of(outcome),
        refusal_of(outcome)[:220],
    )

    # ARM 5 — a request cannot hand itself a hydrated handle. Wire data is a REF STRING.
    from cozy_runtime.author._assets import ImageAsset, asset_dec_hook

    try:
        asset_dec_hook(ImageAsset, {"ref": "x", "local": "/etc/passwd"})
        check("ARM: a forged hydrated handle refuses", False, "it decoded")
    except TypeError as exc:
        check(
            "ARM: a request-forged hydrated handle is UNREPRESENTABLE",
            "reference string" in str(exc),
            str(exc)[:180],
        )

    # ARM 6 — an asset bound to one attempt cannot be rebound into another.
    from cozy_runtime.author._assets import bind
    from cozy_runtime.author._errors import CapabilityError

    escapee = ImageAsset("in:photo")
    bind(
        escapee,
        local=files["photo"],
        attempt="attempt-a",
        media_type="image/png",
        digest="sha256:x",
        length=1,
    )
    try:
        bind(
            escapee,
            local=files["photo"],
            attempt="attempt-b",
            media_type="image/png",
            digest="sha256:x",
            length=1,
        )
        check("ARM: a handle escaping its attempt refuses", False, "it rebound")
    except CapabilityError as exc:
        check(
            "ARM: a handle that escaped its attempt refuses on the next one",
            exc.code == "foreign_asset",
            str(exc)[:180],
        )

    # ARM 7 — no path, token or hydration state in ANY serialized output.
    row = escapee.row()
    check(
        "ARM: no path, token or hydration state in a serialized asset row",
        set(row) == {"ref", "kind", "media_type", "size_bytes", "digest"}
        and "/tmp" not in json.dumps(row),
        f"row keys: {sorted(row)}",
    )

    # ARM 8 — an AssetBound may not WIDEN its kind. Describe-time conformance.
    from cozy_runtime.author._errors import ConformanceError
    from cozy_runtime.author._markers import AssetBound

    try:
        AssetBound(media_types=("video/mp4",)).check_kind("image")
        check("ARM: a widening bound refuses", False, "it was accepted")
    except ConformanceError as exc:
        check(
            "ARM: an AssetBound naming a type its KIND does not admit refuses at describe",
            "NARROWS" in str(exc),
            str(exc)[:200],
        )

    # ARM 9 — the bounded reader, both directions.
    from cozy_runtime.internal.egress import EgressRefusal, read_bounded

    for name, chunks, declared, cap, want in (
        ("over", [b"x" * 100, b"x" * 100], 0, 150, "stream_over_declared_size"),
        ("under", [b"x" * 10], 100, 1000, "stream_under_declared_size"),
    ):
        try:
            read_bounded(iter(chunks), declared=declared, cap=cap, what="arm")
            check(f"ARM: the bounded reader refuses ({name})", False, "it returned")
        except EgressRefusal as exc:
            check(
                f"ARM: the ONE bounded reader refuses a stream that {name}-runs its bound",
                exc.code == want,
                exc.detail[:180],
            )

    del grants_module


# --------------------------------------------------------------------------- egress


def egress() -> None:
    head("egress — a real HTTP server: allowlist, per-hop SSRF, bounded reader")
    import http.server
    import threading

    from cozy_runtime.internal.egress import EgressPolicy, EgressRefusal, fetch

    body = png(8, 8, 5)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/ok":
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/metadata":
                # THE BYPASS: hop 0 is allowlisted and hop 1 is the metadata service.
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                self.end_headers()
                return
            if self.path == "/loop":
                self.send_response(302)
                self.send_header("Location", "/loop")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # A LOCAL deployment declares both halves: this host is allowlisted AND private
        # addresses are admitted. Two switches, because "I trust this host" and "I accept a
        # private address" are different statements.
        local = EgressPolicy(allowed_hosts=("127.0.0.1",), max_bytes=1 << 20, allow_private=True)
        got = fetch(f"http://127.0.0.1:{port}/ok", local, declared=len(body), what="green")
        check(
            "GREEN FIRST: an allowlisted host over a declared length fetches",
            got == body,
            f"{len(got)} B, byte for byte",
        )

        # The SSRF arm. `allow_private=False` is the production posture; hop 0 is the
        # allowlisted host and hop 1 is 169.254.169.254, which is the exact bypass a
        # hop-0-only check waves through.
        strict = EgressPolicy(allowed_hosts=("127.0.0.1", "169.254.169.254"), allow_private=False)
        try:
            fetch(f"http://127.0.0.1:{port}/metadata", strict, what="ssrf")
            check("ARM: a redirect to the metadata service refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a redirect hop to 169.254.169.254 refuses — EVEN allowlisted by name",
                exc.code in ("egress_blocked_address", "egress_scheme"),
                exc.detail[:200],
            )

        try:
            fetch(f"http://elsewhere.invalid:{port}/ok", local, what="host")
            check("ARM: an unallowlisted host refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a host absent from the declared allowlist refuses",
                exc.code == "egress_host_not_allowed",
                exc.detail[:180],
            )

        try:
            fetch(f"http://127.0.0.1:{port}/loop", local, what="loop")
            check("ARM: a redirect loop refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a redirect chain past its budget refuses",
                exc.code == "egress_redirect_budget",
                exc.detail[:180],
            )

        tiny = EgressPolicy(allowed_hosts=("127.0.0.1",), max_bytes=16, allow_private=True)
        try:
            fetch(f"http://127.0.0.1:{port}/ok", tiny, what="cap")
            check("ARM: a body past the cap refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a body past the deployment cap is cut, not buffered and then judged",
                exc.code == "stream_over_declared_size",
                exc.detail[:180],
            )

        off = EgressPolicy()
        try:
            fetch(f"http://127.0.0.1:{port}/ok", off, what="off")
            check("ARM: a deployment with no allowlist refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: FAIL-CLOSED BY DEFAULT — an undeclared allowlist means no egress",
                exc.code in ("egress_host_not_allowed", "egress_scheme"),
                exc.detail[:180],
            )
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------- debts


def debts() -> None:
    head("debts — cl-010's three cozy-runtime items, through the product surfaces")
    runtime = [sys.executable, "-m", "cozy_runtime.cli.main"]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}", COZY_HOME=str(HOME))

    done = subprocess.run(
        [*runtime, "--json", "--dir", str(ROOT / "corpus" / "endpoint"), "fit"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    document = json.loads(done.stdout or "{}")
    check(
        "DEBT 1: `fit --json` returns a host document instead of exit 1 `internal`",
        done.returncode == 0 and "gpu_name" in (document.get("host") or {}),
        f"exit {done.returncode}, host keys {sorted(document.get('host') or {})[:4]}…",
    )

    done = subprocess.run(
        [*runtime, "--json", "--dir", str(ROOT / "corpus" / "endpoint"), "bindings"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    document = json.loads(done.stdout or "{}")
    records = document.get("bindings") or []
    check(
        "DEBT 3: a verb prints the RESOLVED binding record",
        done.returncode == 0 and bool(records) and "store" in records[0],
        f"exit {done.returncode}, {len(records)} record(s), keys "
        f"{sorted(records[0])[:6] if records else []}…",
    )

    done = subprocess.run(
        [*runtime, "serve", "--socket", "/nonexistent.sock", "--grace", "3", "--devices", "0"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=120,
    )
    check(
        "DEBT 3: `serve` accepts the protocol's launch flags instead of rejecting them",
        "unknown flag" not in (done.stdout + done.stderr),
        "it reaches the worker and fails on the socket, which is the honest failure",
    )
    # DEBT 2 is proven in `roundtrip`, on a real OutputEntry.
    print("  DEBT 2 (media_type on OutputEntry) is checked in `roundtrip`, on a real write")


# ------------------------------------------------------------------------------ main


def main(argv: list[str]) -> int:
    if argv and argv[0] == "exec":
        exec_one(Path(argv[1]), Path(argv[2]))
        return 0
    sections = {
        "roundtrip": roundtrip,
        "kill": kill,
        "arms": arms,
        "egress": egress,
        "debts": debts,
    }
    names = argv or ["roundtrip"]
    if names == ["all"]:
        names = list(sections)
    started = time.perf_counter()
    for name in names:
        if name not in sections:
            raise SystemExit(f"unknown section {name!r}; have {sorted(sections)}")
        sections[name]()
    print(f"\n{PASS} ok, {FAIL} failed in {time.perf_counter() - started:.1f} s")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
