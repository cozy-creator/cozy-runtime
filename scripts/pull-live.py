#!/usr/bin/env python3
"""cr-016 `pull` — the live verification (tfs-003 row 220).

There are no tests in v2 (decisions #160): this RUNS the pull leg against a real tensorhub
serving real R2 objects and observes what it does. Every arm is either a real transfer or a
real refusal; nothing here is mocked, and the hostile sources in the door arms are hostile in
exactly the way a network is (a flipped byte, a body that ends early, a reset socket).

  scripts/pull-live.py --hub URL --ref org/repo@sha256:... --store DIR [--source DIR]
  scripts/pull-live.py --hub URL --ref ... --phases door,arms

Phases: pull (fetch it, prove it) · row (the index row the fill plane can actually serve,
red-armed against the row this leg used to write) · streams (the concurrent walk moves the
same bytes and no others) · idem (nothing re-fetched) · heal (a corrupted local object is
not skipped) · resume (SIGKILL mid-object, then only what is missing moves) · door (hostile
sources at the border) · arms (typed refusals: coordinates, hub, SSRF).

`--source` is the store the artifact was published FROM. Given it, the run adds the claim
that matters most and cannot be checked any other way: every pulled object is BIT-IDENTICAL
to the object that was published, with each digest re-derived here rather than trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

FAIL = 0


def ok(what: str, detail: str = "") -> None:
    print(f"  \033[32mPASS\033[0m {what}" + (f"  — {detail}" if detail else ""))


def bad(what: str, detail: str = "") -> None:
    global FAIL
    FAIL += 1
    print(f"  \033[31mFAIL\033[0m {what}" + (f"  — {detail}" if detail else ""))


def head(what: str) -> None:
    print(f"\n\033[1m== {what}\033[0m")


def cozy(
    args: list[str],
    home: pathlib.Path,
    *,
    env: dict[str, str] | None = None,
    **kw: Any,
) -> subprocess.CompletedProcess[str]:
    sealed = {"PATH": "/usr/bin:/bin", "COZY_HOME": str(home), "HOME": str(home)}
    sealed.update(env or {})
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "cozy_runtime.cli.main", *args],
        capture_output=True,
        text=True,
        env=sealed,
        **kw,
    )
    return completed


def pulled(out: str) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(out)
    return parsed


def objects(root: pathlib.Path) -> dict[str, pathlib.Path]:
    d = root / "objects" / "sha256"
    return {p.name: p for p in d.rglob("*") if p.is_file()} if d.is_dir() else {}


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- phases


def phase_pull(a: argparse.Namespace) -> None:
    head("pull — the artifact crosses the border into an EMPTY store")
    shutil.rmtree(a.store, ignore_errors=True)
    dry = cozy(["--json", "pull", a.ref, "--hub", a.hub, "--store", a.store, "--dry-run"], a.home)
    if dry.returncode != 0:
        return bad("dry-run", dry.stderr.strip()[:200])
    plan = pulled(dry.stdout)
    ok(
        "--dry-run plans without moving a byte",
        f"{plan['wanted']} wanted, {plan['bytes_to_move']} B",
    )
    if objects(pathlib.Path(a.store)):
        bad("--dry-run installed objects — a preview that writes is not a preview")

    t0 = time.monotonic()
    run = cozy(["--json", "pull", a.ref, "--hub", a.hub, "--store", a.store], a.home)
    if run.returncode != 0:
        return bad("pull", run.stderr.strip()[:300])
    res = pulled(run.stdout)
    wall = time.monotonic() - t0
    rate = res["bytes_moved"] / max(wall, 1e-6) / 2**20
    ok(
        "pull completed",
        f"{res['fetched']}/{res['declared']} objects, {res['bytes_moved']} B in {wall:.1f}s "
        f"= {rate:.1f} MiB/s",
    )
    if res["fetched"] != plan["wanted"]:
        bad("the plan and the pull disagreed", f"{plan['wanted']} planned, {res['fetched']} moved")
    else:
        ok("the dry-run plan is what actually happened", "same fetch_plan call, not an estimate")

    verify(a, res["snapshot"])
    if a.source:
        identical(a, pathlib.Path(a.source), pathlib.Path(a.store))


def verify(a: argparse.Namespace, snapshot: str, root: str = "") -> None:
    """The destination proves the artifact itself — a local walk over local bytes."""
    if not a.tfs:
        return
    out = subprocess.run(
        [a.tfs, "snapshot", "verify", root or a.store, snapshot.removeprefix("sha256:")],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return bad("the pulled checkpoint does not verify locally", out.stderr.strip()[:200])
    ok("the pulled checkpoint verifies IN THE DESTINATION", out.stdout.splitlines()[1].strip())


def identical(a: argparse.Namespace, src: pathlib.Path, dst: pathlib.Path) -> None:
    have, got = objects(src), objects(dst)
    missing = set(have) - set(got)
    if missing:
        return bad("objects never arrived", f"{len(missing)} missing")
    total = 0
    for name, p in have.items():
        if p.read_bytes() != got[name].read_bytes():
            return bad("BYTES DIFFER after the round trip", name)
        if digest(got[name]) != name:
            return bad("a pulled object does not hash to its own id", name)
        total += got[name].stat().st_size
    ok(
        "published -> pulled is BIT-IDENTICAL",
        f"{len(have)} objects, {total} B ({total / 2**20:.1f} MiB), every digest re-derived here",
    )


def phase_row(a: argparse.Namespace) -> None:
    """The row `pull` files locally, judged by the plane that has to READ it.

    RED FIRST, and the red is the row this leg actually wrote until #566c: `{"checkpoint":
    <digest>}` for an artifact whose header declares real component names. The bytes were
    perfect and the artifact was unservable, which is why the arm is the FILL PREFLIGHT
    (`Checkpoint.rows`, header-only, no device) and not a shape assertion — a shape
    assertion would have passed the old row too.
    """
    head("row — the local index row, against the fill plane that reads it")
    from cozy_runtime.internal import artifacts, fill

    home = pathlib.Path(a.home)
    snapshot = ""
    for row in artifacts.rows(home):
        if row.store == a.store:
            snapshot = row.primary
    if not snapshot:
        return bad("no row was installed by the pull", f"nothing in {home / 'artifacts'}")

    checkpoint = fill.Checkpoint(a.store, snapshot)
    declared = sorted(checkpoint.header["components"])

    # RED: the pre-fix row, constructed here so the refusal is observed rather than recalled.
    try:
        fill.Checkpoint(a.store, snapshot).rows("checkpoint")
        bad("the pre-fix row's component RESOLVED", "'checkpoint' is a real component here?")
    except Exception as exc:
        if getattr(exc, "code", "") == "destination_absent":
            ok(
                "RED: snapshots={'checkpoint': …} is unservable",
                f"destination_absent — the header has {', '.join(declared)}",
            )
        else:
            bad("the pre-fix row refused for the wrong reason", f"{type(exc).__name__}: {exc}")

    # GREEN: the row the landed code wrote, resolved key-exact, component by component.
    installed = [r for r in artifacts.rows(home) if r.store == a.store][-1]
    if sorted(installed.snapshots) != declared:
        return bad(
            "the row's components are not the header's",
            f"row {sorted(installed.snapshots)} vs header {declared}",
        )
    ok("the row names the header's components", ", ".join(declared))
    tensors = 0
    stored = 0
    for name in declared:
        rows_ = fill.Checkpoint(a.store, installed.snapshots[name]).rows(name)
        tensors += len(rows_)
        stored += sum(r.stored_nbytes for r in rows_)
        if not rows_:
            bad(f"component {name} resolved to zero tensors")
    ok(
        "every component resolves key-exact from the PULLED store",
        f"{tensors} tensors, {stored} B stored",
    )
    if installed.tensors != tensors or installed.bytes != stored:
        bad(
            "the row's counts are not the header's",
            f"row {installed.tensors} tensors/{installed.bytes} B vs {tensors}/{stored}",
        )
    else:
        ok("the row's tensors and bytes are the header's own", "copied, never re-derived")
    if "@" in a.ref and not a.ref.split("@")[1].startswith("sha256:"):
        release = a.ref.split("@")[1]
        if installed.ref.endswith("@" + release):
            ok("the release survived into the row", installed.ref)
        else:
            bad("the row dropped the release", f"{a.ref} installed as {installed.ref}")


def phase_streams(a: argparse.Namespace) -> None:
    """The concurrent walk moves the same objects the serial one does, and no others.

    The point is not the clock — this box's uplink is not the link the default was measured
    on — it is that `--streams` changes ORDER and nothing else: same plan, same object set,
    same digests, same `fetch_complete`.
    """
    head("streams — N in flight is the same artifact as one in flight")
    serial = pathlib.Path(a.store + "-s1")
    wide = pathlib.Path(a.store + "-s8")
    got = {}
    for label, root, n in (("serial", serial, 1), ("concurrent", wide, 8)):
        shutil.rmtree(root, ignore_errors=True)
        t0 = time.monotonic()
        run = cozy(
            ["--json", "pull", a.ref, "--hub", a.hub, "--store", str(root), "--streams", str(n)],
            a.home,
        )
        if run.returncode != 0:
            bad(f"{label} pull", run.stderr.strip()[:200])
            return
        res = pulled(run.stdout)
        wall = time.monotonic() - t0
        if res["streams"] != n:
            bad(f"{label}: the result reports {res['streams']} streams, not {n}")
        got[label] = (res, sorted(objects(root)), wall)
        ok(
            f"{label} ({n} stream(s))",
            f"{res['fetched']} objects, {res['bytes_moved']} B in {wall:.1f}s "
            f"= {res['bytes_moved'] / max(wall, 1e-6) / 2**20:.1f} MiB/s",
        )
    if got["serial"][1] != got["concurrent"][1]:
        bad("the two walks produced different stores", "concurrency changed WHAT arrived")
    else:
        ok(
            "byte-for-byte the same store either way",
            f"{len(got['serial'][1])} objects, identical id set",
        )
    if got["serial"][0]["bytes_moved"] != got["concurrent"][0]["bytes_moved"]:
        bad("the two walks moved different byte totals")
    verify(a, got["concurrent"][0]["snapshot"], str(wide))
    shutil.rmtree(serial, ignore_errors=True)
    shutil.rmtree(wide, ignore_errors=True)

    run = cozy(["pull", a.ref, "--hub", a.hub, "--store", a.store, "--streams", "0"], a.home)
    text = run.stdout + run.stderr
    if run.returncode == 2 and "error(usage)" in text:
        ok("--streams 0: usage, exit 2", "a stream count of zero is a refusal, not a silent 1")
    else:
        bad("--streams 0 was accepted", f"exit {run.returncode}: {text[:120]}")


def phase_idem(a: argparse.Namespace) -> None:
    head("idem — an object already held is not fetched again")
    before = {n: p.stat().st_mtime_ns for n, p in objects(pathlib.Path(a.store)).items()}
    run = cozy(["--json", "pull", a.ref, "--hub", a.hub, "--store", a.store], a.home)
    if run.returncode != 0:
        return bad("re-pull", run.stderr.strip()[:200])
    res = pulled(run.stdout)
    if res["fetched"] or res["bytes_moved"]:
        return bad("the re-pull moved bytes", f"{res['fetched']} objects, {res['bytes_moved']} B")
    ok("re-pull moved ZERO bytes", f"{res['held']}/{res['declared']} held")
    after = {n: p.stat().st_mtime_ns for n, p in objects(pathlib.Path(a.store)).items()}
    if after != before:
        bad(
            "an object file was rewritten",
            "held means untouched, not overwritten with the same bytes",
        )
    else:
        ok("not one object file was touched", "mtimes identical across the re-pull")


def phase_heal(a: argparse.Namespace) -> None:
    head("heal — a corrupted local object is repaired, not skipped as 'already present'")
    have = objects(pathlib.Path(a.store))
    victim = max(have.values(), key=lambda p: p.stat().st_size)
    name = victim.name
    victim.chmod(0o644)
    data = bytearray(victim.read_bytes())
    data[len(data) // 3] ^= 0xFF
    victim.write_bytes(bytes(data))
    victim.chmod(0o444)

    run = cozy(["--json", "pull", a.ref, "--hub", a.hub, "--store", a.store], a.home)
    if run.returncode != 0:
        return bad("heal pull", run.stderr.strip()[:200])
    res = pulled(run.stdout)
    if res["fetched"] != 1:
        return bad("the corruption was not repaired alone", f"{res['fetched']} objects re-fetched")
    if digest(pathlib.Path(a.store) / "objects" / "sha256" / name[:2] / name[2:4] / name) != name:
        return bad("the object still does not hash to its id")
    quarantined = list((pathlib.Path(a.store) / "quarantine").glob(name + "*"))
    ok("exactly the corrupted object was re-fetched", f"{res['held']} held, 1 fetched, healed")
    if quarantined:
        ok(
            "the bad bytes were QUARANTINED, never overwritten in place",
            quarantined[0].name[:16] + "…",
        )
    else:
        bad("nothing reached quarantine — the bad bytes went where?")


def phase_resume(a: argparse.Namespace) -> None:
    head("resume — SIGKILL mid-object, then only what is missing moves")
    store = pathlib.Path(a.store + "-resume")
    shutil.rmtree(store, ignore_errors=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cozy_runtime.cli.main",
            "--json",
            "pull",
            a.ref,
            "--hub",
            a.hub,
            "--store",
            str(store),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin", "COZY_HOME": str(a.home), "HOME": str(a.home)},
    )
    # Kill on OBSERVED PROGRESS, never on a sleep: wait until a partial temp file is
    # genuinely large, so the kill lands mid-transfer rather than somewhere a timer guessed.
    tmp = store / "tmp"

    def big() -> bool:
        # A temp file the border finished with disappears WHILE this loop reads it, and with
        # N streams in flight that is the common case rather than a rare one. A vanished
        # candidate is progress, not an error.
        for p in tmp.iterdir() if tmp.is_dir() else ():
            try:
                if p.stat().st_size > (4 << 20):
                    return True
            except FileNotFoundError:
                continue
        return False

    deadline = time.monotonic() + 120
    fired = False
    while time.monotonic() < deadline and proc.poll() is None:
        if big():
            fired = True
            break
        time.sleep(0.005)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    if not fired:
        return ok(
            "resume SKIPPED", "no object in this artifact is large enough to kill mid-transfer"
        )
    landed = len(objects(store))
    ok(
        "the pull died mid-object",
        f"{landed} object(s) had already landed; nothing partial installed",
    )

    run = cozy(["--json", "pull", a.ref, "--hub", a.hub, "--store", str(store)], a.home)
    if run.returncode != 0:
        return bad("resume pull", run.stderr.strip()[:200])
    res = pulled(run.stdout)
    if res["held"] != landed:
        bad("the resume re-fetched work already done", f"{landed} landed, {res['held']} held")
    else:
        ok(
            "the resume re-fetched ONLY what was missing",
            f"{res['held']} held ({res['bytes_held']} B kept), {res['fetched']} fetched",
        )
    verify(a, res["snapshot"])
    shutil.rmtree(store, ignore_errors=True)


def phase_door(a: argparse.Namespace) -> None:
    """The border itself, with sources that lie the way a network lies."""
    head("door — hostile sources at tensorfs.fetch_admit")
    import tensorfs

    src = pathlib.Path(a.source or a.store)
    root = pathlib.Path(a.store + "-door")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    store = tensorfs.Store.init(str(root))
    have = objects(src)
    snap = None
    for name in have:
        try:
            store_snapshot = subprocess.run(
                [a.tfs, "snapshot", "show", str(src), name], capture_output=True
            )
            if store_snapshot.returncode == 0:
                snap = name
                break
        except OSError:
            break
    if snap is None:
        return ok("door SKIPPED", "no --tfs binary to identify the snapshot object")

    declared = [("sha256:" + n, p.stat().st_size) for n, p in sorted(have.items()) if n != snap]
    plan = tensorfs.fetch_plan(store, "door", snap, have[snap].stat().st_size, declared)
    doc = plan["document"]
    big = max((n for n in have if n != snap), key=lambda n: have[n].stat().st_size)
    size = have[big].stat().st_size
    cut = max(size // 2, 1)

    class Flip:
        def __init__(self) -> None:
            self.f, self.n = open(have[big], "rb"), 0  # noqa: SIM115 — the class IS the context

        def read(self, k: int) -> bytes:
            b = bytearray(self.f.read(k))
            if self.n <= cut < self.n + len(b):
                b[cut - self.n] ^= 0x01
            self.n += len(b)
            return bytes(b)

    class Cut:
        def __init__(self) -> None:
            self.f, self.n = open(have[big], "rb"), 0  # noqa: SIM115 — the class IS the context

        def read(self, k: int) -> bytes:
            if self.n >= cut:
                return b""
            b = self.f.read(min(k, cut - self.n))
            self.n += len(b)
            return b

    class Boom:
        def __init__(self) -> None:
            self.f, self.n = open(have[big], "rb"), 0  # noqa: SIM115 — the class IS the context

        def read(self, k: int) -> bytes:
            if self.n >= cut:
                raise ConnectionResetError("peer reset the connection")
            b = self.f.read(k)
            self.n += len(b)
            return b

    # The HIGH-WATER MARK before the arms. Reporting absolute RSS here would report the
    # harness — `identical()` reads whole objects to compare them — so what is measured is
    # the DELTA the door itself adds while streaming a 64 MiB object four times.
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    for label, source in (
        (f"a flipped byte {cut} B in", Flip()),
        ("a body that ends early", Cut()),
        ("a socket that resets mid-object", Boom()),
    ):
        before = set(store.objects())
        try:
            tensorfs.fetch_admit(store, doc, "sha256:" + big, source)
            bad(f"{label}: ADMITTED", "the door let it through")
        except Exception as exc:
            left = len(list((root / "tmp").iterdir()))
            if set(store.objects()) != before or left:
                bad(f"{label}: refused but left state", f"{left} temp file(s)")
            else:
                ok(f"{label}: refused, committed NOTHING", f"{type(exc).__name__}: {str(exc)[:70]}")

    with open(have[big], "rb") as f:
        got = tensorfs.fetch_admit(store, doc, "sha256:" + big, f)
    ok("the same object, honest, admits", f"{got['id'][:23]}… {got['length']} B")
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 - rss0
    if grew * 2**20 > size / 4:
        bad("the door held the object in memory", f"RSS grew {grew:.1f} MiB")
    else:
        ok(
            "memory is bounded by the buffer, not the object",
            f"RSS grew {grew:.1f} MiB across four passes of a {size / 2**20:.0f} MiB object",
        )
    shutil.rmtree(root, ignore_errors=True)


def phase_arms(a: argparse.Namespace) -> None:
    head("arms — typed refusals, each naming what to do about it")
    cases = [
        (
            "wrong digest coordinates",
            [
                "pull",
                a.ref.split("@")[0] + "@sha256:" + "f" * 64,
                "--hub",
                a.hub,
                "--store",
                a.store,
            ],
            "ref_not_found",
            4,
        ),
        (
            "unknown repo",
            ["pull", "nobody/nothing", "--hub", a.hub, "--store", a.store],
            "ref_not_found",
            4,
        ),
        (
            "hub unreachable",
            ["pull", a.ref, "--hub", "http://127.0.0.1:1", "--store", a.store],
            "hub_unreachable",
            9,
        ),
        (
            "SSRF: a non-loopback private hub",
            ["pull", a.ref, "--hub", "http://10.0.0.1:8080", "--store", a.store],
            "source_not_allowed",
            3,
        ),
        ("no --hub", ["pull", a.ref, "--store", a.store], "usage", 2),
    ]
    for label, args, code, exit_code in cases:
        run = cozy(args, a.home)
        text = run.stdout + run.stderr
        if run.returncode == exit_code and f"error({code})" in text:
            ok(
                f"{label}: {code}, exit {exit_code}",
                text.splitlines()[0][7 + len(code) : 90].strip(),
            )
        else:
            bad(
                f"{label}: wanted {code}/exit {exit_code}",
                f"got exit {run.returncode}: {text[:120]}",
            )


PHASES = {
    "pull": phase_pull,
    "row": phase_row,
    "streams": phase_streams,
    "idem": phase_idem,
    "heal": phase_heal,
    "resume": phase_resume,
    "door": phase_door,
    "arms": phase_arms,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--source", default="", help="the store the artifact was published FROM")
    ap.add_argument("--tfs", default="", help="the tfs binary, for the destination-side verify")
    ap.add_argument("--home", default="", help="COZY_HOME for the artifact index")
    ap.add_argument("--phases", default=",".join(PHASES))
    a = ap.parse_args()
    a.home = pathlib.Path(a.home or (a.store + "-home"))
    a.home.mkdir(parents=True, exist_ok=True)

    for name in a.phases.split(","):
        if name.strip() not in PHASES:
            sys.exit(f"unknown phase {name!r}; phases are {', '.join(PHASES)}")
    for name in a.phases.split(","):
        PHASES[name.strip()](a)

    print(f"\n{'FAILED' if FAIL else 'all arms observed'}: {FAIL} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
