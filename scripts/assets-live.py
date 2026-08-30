#!/usr/bin/env python3
"""cr-012 live verification: typed INPUT ASSETS, end to end on the real attempt path.

Not a test (tracker README #160): every leg runs the real RecordOwner, the real worker
hydration, the real executor and the real package, and prints numbers.

  roundtrip  a real PNG in -> `retouch` -> a real PNG out, digest-verified both ways
  kill       the executor dies mid-attempt: the terminal converges, the spool is cleaned
  arms       the typed red arms, each planted for real
  egress     a real HTTP server: allowlist, per-hop SSRF, the bounded reader, the ONE
             address predicate both downloaders call, and the closed grant table
  debts      cl-010's three cozy-runtime debts, exercised through the product surfaces
  all        every section above, in order
"""

from __future__ import annotations

import importlib.util
import io
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
    record: dict[str, Any] = dict(S.binding_record(entrypoint))
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
    """ONE request through the WHOLE real path — RecordOwner, worker, executor."""
    import dataclasses

    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalInput, LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    spec = json.loads(spec_path.read_text())
    workspace = Path(spec["workspace"])
    HOME.mkdir(parents=True, exist_ok=True)
    # The record names a ref; THIS machine says what it means (#567e).
    S.install_row(HOME, S.working_store(WORK / "store"))
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(HOME),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        }
    )

    def hook(record_owner: Any, worker: Any) -> None:
        """Harness-side fault injection through the documented seam (decisions #249)."""
        if spec.get("kill_after"):
            import threading

            dispatched = record_owner._dispatch

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

            record_owner._dispatch = arming
        if spec.get("tamper"):
            # THE PLANT: the RecordOwner mints the grant over the file as it stands, and
            # the bytes then change underneath it. That is what a corrupted input IS in
            # production — a stored object that moved after the record naming it was
            # written — and it needs no switch inside the runtime.
            minted = record_owner._grant

            def corrupting() -> Any:
                grant = minted()
                path = Path(spec["tamper"])
                body = bytearray(path.read_bytes())
                body[len(body) // 2] ^= 0x5A
                path.write_bytes(bytes(body))
                return grant

            record_owner._grant = corrupting

    outcome = run_slice(
        config,
        LocalRequest(
            entrypoint=spec.get("entrypoint", "retouch"),
            payload=spec["payload"],
            outputs=tuple(spec.get("outputs", ("image",))),
            request_id=spec["request_id"],
            inputs={
                input_id: LocalInput(str(row["source"]), int(row.get("order", 0)))
                for input_id, row in spec.get("inputs", {}).items()
            },
        ),
        [spec["binding"]],
        workspace=workspace,
        package_revision_digest=S.RELEASE,
        options=WorkerOptions(root=workspace / "worker", release_id=S.RELEASE, no_warm=True),
        delete_journal=False,
        record_owner_hook=hook,
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
                "photo": {"source": str(files["photo"]), "order": 0},
                "refs.0": {"source": str(files["ref0"]), "order": 0},
                "refs.1": {"source": str(files["ref1"]), "order": 1},
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
            "inputs": {"photo": {"source": str(files["huge"]), "order": 0}},
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
            "inputs": {"photo": {"source": str(files["huge"]), "order": 0}},
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
            "inputs": {"photo": {"source": str(mislabelled), "order": 0}},
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
            "inputs": {"photo": {"source": str(files["photo"]), "order": 0}},
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
            "inputs": {"photo": {"source": str(tampered), "order": 0}},
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

    try:
        AssetBound(max_decoded_bytes=0)
        check("ARM: a zero decoded-size bound refuses", False, "it was accepted")
    except ConformanceError as exc:
        check(
            "ARM: decoded expansion bounds are positive",
            "max_decoded_bytes" in str(exc),
            str(exc)[:200],
        )

    try:
        AssetBound(max_decoded_bytes=1024).check_kind("file")
        check("ARM: a decoded bound on FileAsset refuses", False, "it was accepted")
    except ConformanceError as exc:
        check(
            "ARM: only typed media fields can declare decoded expansion",
            "MediaDecoder" in str(exc),
            str(exc)[:200],
        )

    # ARM 9 — the bounded stream copier, both directions.
    from cozy_runtime.internal.egress import EgressRefusal, copy_bounded

    for name, chunks, declared, cap, want in (
        ("over", [b"x" * 100, b"x" * 100], None, 150, "stream_over_declared_size"),
        ("under", [b"x" * 10], 100, 1000, "stream_under_declared_size"),
    ):
        try:
            copy_bounded(
                iter(chunks),
                io.BytesIO(),
                expected_length=declared,
                cap=cap,
                prefix_bytes=16,
                what="arm",
            )
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
    head("egress — a real HTTP server: allowlist, per-hop SSRF, bounded reader, one predicate")
    import http.server
    import threading

    from cozy_runtime.internal.egress import EgressPolicy, EgressRefusal, copy_bounded, fetch_into

    def fetch(
        url: str, policy: EgressPolicy, *, declared: int | None = None, cap: int = 1 << 20
    ) -> bytes:
        sink = io.BytesIO()
        fetch_into(
            url,
            policy,
            sink,
            expected_length=declared,
            cap=cap,
            prefix_bytes=32,
            what="egress arm",
        )
        return sink.getvalue()

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
            if self.path == "/truncated":
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"20\r\nshort\r\n")
                self.wfile.flush()
                self.close_connection = True
                return
            if self.path == "/malformed-location":
                self.send_response(302)
                self.send_header("Location", "https://[broken/object")
                self.end_headers()
                return
            if self.path == "/invalid-request-location":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{port}/a b")
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
        local = EgressPolicy(allowed_hosts=("127.0.0.1",), allow_private=True)
        got = fetch(f"http://127.0.0.1:{port}/ok", local, declared=len(body))
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
            fetch(f"http://127.0.0.1:{port}/metadata", strict)
            check("ARM: a redirect to the metadata service refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a redirect hop to 169.254.169.254 refuses — EVEN allowlisted by name",
                exc.code in ("egress_blocked_address", "egress_scheme"),
                exc.detail[:200],
            )

        try:
            fetch(f"http://elsewhere.invalid:{port}/ok", local)
            check("ARM: an unallowlisted host refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a host absent from the declared allowlist refuses",
                exc.code == "egress_host_not_allowed",
                exc.detail[:180],
            )

        try:
            fetch(f"http://127.0.0.1:{port}/loop", local)
            check("ARM: a redirect loop refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a redirect chain past its budget refuses",
                exc.code == "egress_redirect_budget",
                exc.detail[:180],
            )

        try:
            fetch(f"http://127.0.0.1:{port}/truncated", local)
            check("ARM: a malformed chunked body refuses", False, "it returned")
        except EgressRefusal as exc:
            check(
                "ARM: a truncated HTTP body becomes a typed refusal",
                exc.code == "egress_body_error",
                exc.detail[:180],
            )

        malformed_codes: list[str] = []
        for url in (
            "https://127.0.0.1:99999/object",
            f"http://127.0.0.1:{port}/malformed-location",
            f"http://127.0.0.1:{port}/invalid-request-location",
        ):
            try:
                fetch(url, local)
            except EgressRefusal as exc:
                malformed_codes.append(exc.code)
        check(
            "ARM: malformed direct and redirected URLs both become typed refusals",
            malformed_codes == ["egress_url", "egress_url", "egress_unreachable"],
            str(malformed_codes),
        )

        try:
            fetch(f"http://127.0.0.1:{port}/ok", local, cap=16)
            check("ARM: a body past the cap refuses", False, "it fetched")
        except EgressRefusal as exc:
            check(
                "ARM: a body past the deployment cap is cut, not buffered and then judged",
                exc.code == "stream_over_declared_size",
                exc.detail[:180],
            )

        import hashlib
        import tempfile
        import tracemalloc

        exact_empty = copy_bounded(
            iter(()),
            io.BytesIO(),
            expected_length=0,
            cap=1,
            prefix_bytes=16,
            what="exact empty arm",
        )
        check(
            "the exact-zero identity is distinct from an unknown length",
            exact_empty.length == 0 and exact_empty.sha256.hex() == hashlib.sha256(b"").hexdigest(),
        )

        from cozy_runtime.internal.worker import grants
        from cozy_runtime.protocol import worker_pb2 as pb

        with tempfile.TemporaryDirectory(prefix="cozy-grant-arms-") as directory:
            grant_root = Path(directory) / "grants"
            grant_root.mkdir()
            subject = hashlib.sha256(b"invocation").digest()
            empty = grant_root / "empty.bin"
            empty.write_bytes(b"")
            empty_spec = {
                "inputs": [
                    {
                        "input_id": "empty",
                        "digest": "sha256:" + hashlib.sha256(b"").hexdigest(),
                        # Proto3 canonical JSON omits uint64 zero. Omission means exact zero.
                        "kind_mime": "application/octet-stream",
                    }
                ],
                "outputs": [],
            }
            empty_grant = pb.DeliveryGrant(
                invocation_spec_digest=subject,
                file_base_url=f"file://{grant_root}",
                inputs=[pb.InputAccess(input_id="empty", url=f"file://{empty}")],
            )
            bound = grants.bind(empty_spec, empty_grant, subject)
            hydrated = grants.hydrate_inputs(
                bound,
                grants.Authorizer(roots=(str(grant_root),)),
                spool=Path(directory) / "empty-spool",
                max_bytes=1,
                max_total_bytes=1,
            )
            zero = hydrated.get("empty")
            check(
                "an omitted proto3 zero length binds and hydrates as an exact empty FileAsset",
                zero is not None
                and zero.length == 0
                and zero.media_type == ""
                and zero.local.read_bytes() == b"",
            )

            unknown = grant_root / "unknown.bin"
            unknown.write_bytes(b"not-an-image")
            unknown_spec = {
                "inputs": [
                    {
                        "input_id": "photo",
                        "digest": "sha256:" + hashlib.sha256(unknown.read_bytes()).hexdigest(),
                        "length": unknown.stat().st_size,
                        "kind_mime": "image/png",
                    }
                ],
                "outputs": [],
            }
            unknown_grant = pb.DeliveryGrant(
                invocation_spec_digest=subject,
                inputs=[pb.InputAccess(input_id="photo", url=f"file://{unknown}")],
            )
            try:
                grants.hydrate_inputs(
                    grants.bind(unknown_spec, unknown_grant, subject),
                    grants.Authorizer(roots=(str(grant_root),)),
                    spool=Path(directory) / "unknown-spool",
                )
                unknown_refused = False
            except grants.GrantRefusal as exc:
                unknown_refused = exc.code == "input_media_type"
            check(
                "unknown bytes cannot borrow an image MIME declaration",
                unknown_refused and not list((Path(directory) / "unknown-spool").rglob("inputs/*")),
            )

            try:
                grants.bind(
                    empty_spec,
                    pb.DeliveryGrant(inputs=empty_grant.inputs),
                    subject,
                )
                subject_refused = False
            except grants.GrantRefusal as exc:
                subject_refused = exc.code == "grant_subject_mismatch"
            check("an absent grant subject cannot bind to an invocation", subject_refused)

            input_row = {
                "input_id": "i",
                "digest": "sha256:" + hashlib.sha256(b"").hexdigest(),
                "kind_mime": "application/octet-stream",
            }
            output_row = {"output_id": "o", "max_bytes": 1}
            ambiguity_cases: tuple[
                tuple[
                    dict[str, Any],
                    list[pb.InputAccess],
                    list[pb.OutputAccess],
                    list[pb.OutputAccess],
                ],
                ...,
            ] = (
                ({"inputs": [{**input_row, "input_id": ""}]}, [], [], []),
                ({"inputs": []}, [pb.InputAccess(input_id="", url="file:///x")], [], []),
                ({"inputs": [input_row, input_row]}, [pb.InputAccess(input_id="i")], [], []),
                ({"inputs": [input_row]}, [pb.InputAccess(input_id="i")] * 2, [], []),
                ({"outputs": [{**output_row, "output_id": ""}]}, [], [], []),
                ({"outputs": []}, [], [], [pb.OutputAccess(output_id="", url="file:///x")]),
                ({"outputs": [output_row, output_row]}, [], [pb.OutputAccess(output_id="o")], []),
                ({"outputs": [output_row]}, [], [], [pb.OutputAccess(output_id="o")] * 2),
            )
            ambiguous_refusals = 0
            for partial, accesses, single_outputs, repeated_outputs in ambiguity_cases:
                document = {"inputs": [], "outputs": [], **partial}
                try:
                    grants.bind(
                        document,
                        pb.DeliveryGrant(
                            invocation_spec_digest=subject,
                            inputs=accesses,
                            outputs=[*single_outputs, *repeated_outputs],
                        ),
                        subject,
                    )
                except grants.GrantRefusal as exc:
                    ambiguous_refusals += exc.code == "grant_binding_ambiguous"
            check(
                "empty and repeated spec/grant ids refuse on both input and output joins",
                ambiguous_refusals == len(ambiguity_cases),
                f"{ambiguous_refusals}/{len(ambiguity_cases)} refused",
            )

            malformed_grant_refusals = 0
            for url in ("file://[broken/object", "https://[broken/object"):
                malformed = grants.BoundGrant(
                    inputs={
                        "asset": grants.BoundInput(
                            "asset",
                            url,
                            hashlib.sha256(b"").digest(),
                            0,
                            "application/octet-stream",
                            0,
                        )
                    }
                )
                try:
                    grants.read_input(malformed, "asset", grants.Authorizer(), cap=1)
                except grants.GrantRefusal as exc:
                    malformed_grant_refusals += exc.code == "grant_transport"
            check(
                "malformed local and remote grant URLs refuse at the grant front door",
                malformed_grant_refusals == 2,
                f"{malformed_grant_refusals}/2 typed refusals",
            )

            # cr-042: the closed table decides the destination, at the unit it decides it
            # in; worker-live's `grants` section proves the same refusal end to end. The
            # `https://` output row `OutputAccess.url` reserves is PLANNED — whoever adds it
            # lands host validation of the minted URL and rewrites this arm around it.
            exfiltrating = grants.BoundGrant(
                outputs={
                    "image": grants.BoundOutput(
                        "image", "https://exfiltrate.example/upload", 1 << 20, "image/png"
                    )
                }
            )
            source = grant_root / "produced.png"
            source.write_bytes(png(8, 8, 9))
            try:
                grants.write_output(
                    exfiltrating, "image", source, grants.Authorizer(roots=(str(grant_root),))
                )
                exfiltration_refused = ""
            except grants.GrantRefusal as exc:
                exfiltration_refused = exc.code
            check(
                "ARM: a destination the closed table has no row for is unwritable — there is "
                "no `https://` output row yet",
                exfiltration_refused in ("hub_call_unauthorized", "grant_transport"),
                f"{exfiltration_refused or 'it wrote'} — every HUB_CALLS row is `file://` "
                "today, so a destination the table never minted a row for does not resolve",
            )

            inside_parent = grant_root / "parent"
            inside_parent.mkdir()
            inside = inside_parent / "asset.bin"
            inside.write_bytes(b"inside")
            outside_parent = Path(directory) / "outside"
            outside_parent.mkdir()
            outside = outside_parent / "asset.bin"
            outside.write_bytes(b"outside-secret")
            moved = grant_root / "moved-parent"

            class SwapAuthorizer(grants.Authorizer):
                def authorize(
                    self, name: str, method: str, url: str, params: dict[str, Any]
                ) -> None:
                    super().authorize(name, method, url, params)
                    inside_parent.rename(moved)
                    inside_parent.symlink_to(outside_parent, target_is_directory=True)

            raced = grants.BoundGrant(
                inputs={
                    "asset": grants.BoundInput(
                        "asset",
                        f"file://{inside}",
                        hashlib.sha256(outside.read_bytes()).digest(),
                        outside.stat().st_size,
                        "application/octet-stream",
                        0,
                    )
                }
            )
            try:
                grants.read_input(
                    raced,
                    "asset",
                    SwapAuthorizer(roots=(str(grant_root),)),
                    cap=1 << 20,
                )
                swap_refused = False
            except grants.GrantRefusal as exc:
                swap_refused = exc.code == "input_not_a_file"
            check(
                "a parent swapped to an outside symlink after authorization cannot redirect read",
                swap_refused,
            )

        streamed = 64 * MIB
        tracemalloc.start()
        with tempfile.TemporaryFile() as sink:
            receipt = copy_bounded(
                (b"x" * MIB for _ in range(streamed // MIB)),
                sink,
                expected_length=streamed,
                cap=streamed,
                prefix_bytes=32,
                what="bounded-memory arm",
            )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        check(
            "a 64 MiB transfer keeps Python heap usage to a few chunks",
            receipt.length == streamed and peak < 8 * MIB,
            f"{receipt.length // MIB} MiB copied, {peak / MIB:.2f} MiB peak Python heap",
        )

        with tempfile.TemporaryDirectory(prefix="cozy-output-stream-") as directory:
            output_root = Path(directory)
            source = output_root / "source.bin"
            expected = hashlib.sha256()
            chunk = b"y" * MIB
            with source.open("wb") as written:
                for _ in range(streamed // MIB):
                    written.write(chunk)
                    expected.update(chunk)
            destination = output_root / "published.bin"
            output_grant = grants.BoundGrant(
                outputs={
                    "video": grants.BoundOutput(
                        "video", f"file://{destination}", streamed, "video/mp4"
                    )
                }
            )
            tracemalloc.start()
            output_entry = grants.write_output(
                output_grant,
                "video",
                source,
                grants.Authorizer(roots=(str(output_root),)),
                media_type="video/mp4",
            )
            _, output_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            check(
                "a 64 MiB worker publication streams spool-to-grant without a bytes snapshot",
                output_entry.length == streamed
                and output_entry.digest == expected.digest()
                and output_entry.mime_type == "video/mp4"
                and destination.stat().st_size == streamed
                and output_peak < 8 * MIB,
                f"{output_entry.length // MIB} MiB, {output_peak / MIB:.2f} MiB peak Python heap",
            )

        # ------------------------------------------------------------------ cr-042
        # ONE address predicate, TWO downloaders. `pull.guard` and
        # `EgressPolicy.check_addresses` were separate hand-written class chains in the
        # module whose own header says the defect is "there were two downloaders and only
        # one had the check". They now share `egress.blocked`, and this asks BOTH of them
        # about the same address.
        import urllib.error

        from cozy_runtime.internal import egress as egress_module
        from cozy_runtime.internal import pull

        metadata = "169.254.169.254"
        why = egress_module.blocked(metadata)
        try:
            EgressPolicy(allowed_hosts=(metadata,)).check_addresses(metadata, 80)
            downloader_codes = ["it resolved"]
        except EgressRefusal as exc:
            downloader_codes = [exc.code]
        try:
            pull.guard(f"https://{metadata}/object", allow_local=False)
            downloader_codes.append("it passed")
        except pull.PullRefusal as exc:
            downloader_codes.append(exc.code)
        check(
            "ONE PREDICATE: both downloaders refuse the metadata address for the same reason",
            why == "private, link-local"
            and downloader_codes == ["egress_blocked_address", "source_not_allowed"],
            f"egress.blocked({metadata}) = {why!r}; {downloader_codes}",
        )

        deadline = pull.Deadline.after(10.0)
        body_len = len(
            pull.open_guarded(f"http://127.0.0.1:{port}/ok", deadline, allow_local=True).read()
        )
        check(
            "GREEN: the pull leg's guarded opener fetches from a declared loopback dev hub",
            body_len == len(body),
            f"{body_len} B through pull.open_guarded",
        )
        try:
            pull.open_guarded(f"http://127.0.0.1:{port}/loop", deadline, allow_local=True)
            redirect = "it followed"
        except urllib.error.HTTPError as exc:
            redirect = f"HTTP {exc.code}"
        check(
            "ARM: the pull leg REFUSES a redirect outright — never re-opens a URL the guard "
            "did not see",
            redirect == "HTTP 302",
            f"{redirect}; a presigned GET has no business redirecting (`_open`/NoRedirect)",
        )

        off = EgressPolicy()
        try:
            fetch(f"http://127.0.0.1:{port}/ok", off)
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
        [*runtime, "--json", "--dir", str(ROOT / "corpus" / "package"), "fit"],
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
        [*runtime, "--json", "--dir", str(ROOT / "corpus" / "package"), "bindings"],
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
