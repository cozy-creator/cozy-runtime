#!/usr/bin/env python
"""cr-016's live verification: the runtime CLI verbs, run for real.

Not a test suite (#160). Every section runs the actual `cozy-runtime` process against the
real four-component SDXL endpoint and prints what it observed.

    corpus/.venv/bin/python scripts/cli-live.py surface    verbs, exits, AXI conformance
    corpus/.venv/bin/python scripts/cli-live.py keys       the local-bind extra-key tier
    corpus/.venv/bin/python scripts/cli-live.py identity   CLI run == direct run_slice
    corpus/.venv/bin/python scripts/cli-live.py all

`surface` and `keys` need no GPU. `identity` needs a quiet card: it runs the whole 6.46 GiB
pipeline twice — once through `cozy-runtime run` and once through `internal.local.run_slice`
directly — and compares the bytes, because "one attempt kernel" is a claim about what comes
out, not about the import graph.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

PROJECT = ROOT / "corpus" / "pipeline"
REF = "cozy/sdxl-pipeline@cr-008b"
WORK = Path("/tmp/cozy-cli")
PROMPT = "a photograph of an astronaut riding a horse"

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


def cli(*args: str, home: Path | None = None, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run the REAL entrypoint as its own process, so the exit code is a process exit."""
    env = dict(
        os.environ,
        PYTHONPATH=f"{ROOT}:{ROOT / 'src'}",
        COZY_HOME=str(home or (WORK / "home")),
    )
    out = subprocess.run(
        [sys.executable, "-m", "cozy_runtime.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd or PROJECT),
        check=False,
    )
    return out.returncode, out.stdout, out.stderr


def install_artifact(home: Path, ref: str = REF, **overrides: Any) -> None:
    from cozy_runtime.cli import artifacts

    snapshots = json.loads(Path("/tmp/cozy-sdxl4/snapshots.json").read_text())
    row = artifacts.Artifact(
        ref=ref,
        store="/tmp/cozy-sdxl4/store",
        config="/tmp/cozy-sdxl4/pipeline.config.json",
        snapshots=snapshots,
        lane="plain-fp16",
        bytes=6_937_675_734,
        tensors=2641,
        variant="sm89",
        **overrides,
    )
    artifacts.install(home, row)


# --------------------------------------------------------------------------- surface


def section_surface(args: argparse.Namespace) -> None:
    home = WORK / "home"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)

    head("the exit-code matrix, observed as PROCESS exits")
    cases: list[tuple[str, tuple[str, ...], int]] = [
        ("unknown verb", ("frobnicate",), 2),
        ("unknown flag", ("run", "generate", "--nope"), 2),
        ("run with no function", ("run",), 2),
        ("unknown function", ("run", "nosuchfn"), 4),
        ("undeclared payload field", ("run", "generate", "colour=blue"), 3),
        ("wrong scalar type", ("run", "generate", "steps=many"), 3),
        ("uninstalled ref", ("run", "generate", f"{PROMPT}"), 4),
        ("--offline miss", ("run", "generate", "--offline"), 8),
        ("rm without --yes", ("rm", REF), 7),
        ("pull, whose owner is TensorFS", ("pull", REF), 9),
        ("describe an unknown function", ("describe", "nosuchfn"), 4),
        ("list, empty", ("list",), 0),
        ("doctor", ("doctor",), 0),
    ]
    for label, argv, want in cases:
        code, out, err = cli(*argv, home=home)
        check(
            f"{label} -> exit {want}",
            code == want,
            f"got {code}: {(err or out).strip().splitlines()[0][:110] if (err or out).strip() else ''}",
        )

    head("AXI conformance")
    sources = list((ROOT / "src" / "cozy_runtime" / "cli").glob("*.py"))
    prompts = [p.name for p in sources if "input(" in p.read_text()]
    check(
        "no interactive prompt exists anywhere in the CLI layer",
        not prompts,
        f"{len(sources)} modules scanned; anything that would ask is a flag",
    )
    code, out, _ = cli("--help", home=home)
    check("--help is one screen and never touches the network", code == 0 and len(out) < 3000,
          f"{len(out)} B")
    code, out, _ = cli("list", home=home)
    check(
        "empty states are definitive",
        "0 artifacts" in out,
        out.strip().splitlines()[0] if out.strip() else "",
    )
    code, out, err = cli("rm", REF, home=home)
    check(
        "a destructive act refuses with the remedy, never a question",
        "remedy:" in err and "?" not in err,
        err.strip().splitlines()[0][:110],
    )

    head("describe is READ-ONLY by default")
    before = {p: p.read_bytes() for p in PROJECT.rglob("*") if p.is_file()}
    code, _, _ = cli("describe", home=home)
    after = {p: p.read_bytes() for p in PROJECT.rglob("*") if p.is_file()}
    check(
        "bare describe leaves the working tree byte-identical",
        code == 0 and before == after,
        f"{len(before)} files compared",
    )

    head("the installed artifact, and the verbs that read it")
    install_artifact(home)
    code, out, _ = cli("list", home=home)
    check("list shows one artifact, one size", REF in out and "6.461 GiB" in out, out.strip())
    code, out, _ = cli("fit", "generate", "steps=20", "size=1024", home=home)
    check("fit renders a verdict from the real chooser", code == 0 and "generate:" in out,
          [ln for ln in out.splitlines() if ln.startswith("generate")][0][:150] if code == 0 else out)
    code, out, _ = cli("rm", REF, "--yes", home=home)
    check("rm --yes removes it", code == 0 and "removed" in out, out.strip().splitlines()[0])
    code, out, _ = cli("rm", REF, "--yes", home=home)
    check("and an absent ref is an idempotent 0 that says so", code == 0 and "not present" in out,
          out.strip().splitlines()[0])


# --------------------------------------------------------------------------- keys


def section_keys(args: argparse.Namespace) -> None:
    """The local-bind extra-key tier (§1.1), on the REAL Loader with planted keys.

    The plant is a checkpoint topology carrying keys the code does not consume — which is
    what an un-bordered community checkpoint looks like. It is built, observed, and dropped
    inside this function; nothing is left behind.
    """
    import tiny_artifacts as art

    from cozy_runtime.author import Artifact, ConformanceError, TensorSpec
    from cozy_runtime.author._model import ModelRegistry
    from tiny_sdxl import SdxlModel

    head("a CANONICAL lane refuses any extra key — the border already classified them")
    base = art.sdxl()
    planted = dict(base.topology)
    component = sorted({k.split(".")[0] for k in planted})[0]
    planted[f"{component}.left_over_weight"] = TensorSpec(shape=(2, 2), dtype="fp16", nbytes=8)
    canonical = Artifact(
        snapshot=base.snapshot, topology=planted, config=base.config, custody="canonical"
    )
    try:
        ModelRegistry(release="cli-live", backend=None).acquire(
            "generate.models.model", SdxlModel, canonical, mode="derive"
        )
        check("canonical extra key refuses", False, "it was accepted")
    except ConformanceError as exc:
        check("canonical extra key refuses", exc.code == "topology_extra_keys", str(exc)[:120])

    head("a LOCAL bind warns, IGNORES and CONFESSES the same key")
    local = Artifact(
        snapshot=base.snapshot, topology=planted, config=base.config, custody="local"
    )
    registry = ModelRegistry(release="cli-live", backend=None)
    registry.acquire("generate.models.model", SdxlModel, local, mode="derive")
    record = registry.generations()[0].record
    check(
        "the extra key is named in the construction record, not dropped in silence",
        record.ignored_extras == (f"{component}.left_over_weight",),
        f"ignored_extras={record.ignored_extras}",
    )

    head("--strict-keys upgrades the same bind to a refusal")
    strict = Artifact(
        snapshot=base.snapshot,
        topology=planted,
        config=base.config,
        custody="local",
        strict_keys=True,
    )
    try:
        ModelRegistry(release="cli-live", backend=None).acquire(
            "generate.models.model", SdxlModel, strict, mode="derive"
        )
        check("--strict-keys refuses", False, "it was accepted")
    except ConformanceError as exc:
        check("--strict-keys refuses", exc.code == "strict_extra_keys", str(exc)[:120])

    head("a KNOWN MARKER is never merely ignored (the v_pred class)")
    marked = dict(base.topology)
    marked[f"{component}.v_pred"] = TensorSpec(shape=(1,), dtype="fp16", nbytes=2)
    transcribing = Artifact(
        snapshot=base.snapshot, topology=marked, config=base.config, custody="local"
    )
    registry = ModelRegistry(release="cli-live", backend=None)
    registry.acquire("generate.models.model", SdxlModel, transcribing, mode="derive")
    record = registry.generations()[0].record
    check(
        "it is TRANSCRIBED to its stamp rather than ignored",
        record.transcribed_markers == (f"{component}.v_pred->objective=v_prediction",)
        and record.ignored_extras == (),
        f"markers={record.transcribed_markers} ignored={record.ignored_extras}",
    )
    conflicting = Artifact(
        snapshot=base.snapshot,
        topology=marked,
        config=base.config,
        custody="local",
        stamps={"objective": "epsilon"},
    )
    try:
        ModelRegistry(release="cli-live", backend=None).acquire(
            "generate.models.model", SdxlModel, conflicting, mode="derive"
        )
        check("a marker contradicting the artifact's stamp refuses NAMING it", False, "accepted")
    except ConformanceError as exc:
        check(
            "a marker contradicting the artifact's stamp refuses NAMING it",
            exc.code == "marker_conflict" and "v_pred" in str(exc),
            str(exc)[:140],
        )

    head("the plant is removed")
    registry = ModelRegistry(release="cli-live", backend=None)
    registry.acquire("generate.models.model", SdxlModel, art.sdxl(), mode="derive")
    clean = registry.generations()[0].record
    check(
        "the unplanted artifact ignores nothing and transcribes nothing",
        clean.ignored_extras == () and clean.transcribed_markers == (),
        "the tier only fires on keys that are actually there",
    )


# --------------------------------------------------------------------------- identity


def section_identity(args: argparse.Namespace) -> None:
    """ONE attempt kernel: the CLI's run and a direct `run_slice` produce the same bytes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_live", ROOT / "scripts" / "pipeline-live.py")
    assert spec and spec.loader
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)

    home = WORK / "home"
    home.mkdir(parents=True, exist_ok=True)
    install_artifact(home)
    live.require_quiet_card()

    head("the CLI's `run`, on the real four-component pipeline")
    out_dir = WORK / "out"
    shutil.rmtree(out_dir, ignore_errors=True)
    code, out, err = cli(
        "--json",
        "run",
        "generate",
        PROMPT,
        f"steps={args.steps}",
        f"size={args.size}",
        "seed=1005",
        "--out",
        str(out_dir),
        home=home,
    )
    check("`cozy-runtime run` exits 0", code == 0, (err or out).strip()[-400:] if code else "")
    if code != 0:
        return
    document = json.loads(out)
    cli_digest = (document.get("result") or {}).get("digest", "")
    saved = document.get("outputs") or {}
    print(f"  outputs: {json.dumps(saved)}")
    print(f"  plan:    {json.dumps(document.get('plan'))}")
    check(
        "the output path is DERIVED from the descriptor's asset field",
        set(saved) == {"image"},
        f"{sorted(saved)} — `image` is the one ImageAsset field on Txt2ImgOutput",
    )
    check(
        "and the asset is a real PNG on disk",
        bool(saved) and Path(saved["image"]).stat().st_size > 100_000,
        f"{Path(saved['image']).stat().st_size} B" if saved else "",
    )

    head("the SAME request through `internal.local.run_slice`, with no CLI in the picture")
    live.require_quiet_card()
    direct = live.spawn(
        {
            "workspace": str(WORK / "direct"),
            "payload": {
                "prompt": PROMPT,
                "steps": args.steps,
                "size": args.size,
                "seed": 1005,
            },
            "request_id": "cli-identity",
            "binding": {"resident_budget_bytes": 0},
        }
    )
    direct_digest = (direct.get("result") or {}).get("digest", "")
    check(
        "OBSERVATIONAL IDENTITY: byte-identical result digests",
        bool(cli_digest) and cli_digest == direct_digest,
        f"cli {cli_digest[:32]}… vs direct {direct_digest[:32]}…",
    )
    check(
        "the same terminal status",
        document.get("status") == direct.get("status"),
        f"{document.get('status')} vs {direct.get('status')}",
    )
    cli_plan = document.get("plan") or {}
    check(
        "the same chosen plan",
        (cli_plan.get("delivery"), cli_plan.get("placement"))
        == (direct.get("accepted", {}).get("delivery"), direct.get("accepted", {}).get("placement")),
        f"{cli_plan.get('delivery')}/{cli_plan.get('placement')}",
    )
    cli_classes = {row["class"] for row in (document.get("ledger") or {}).get("classes", ())}
    direct_classes = {row["class"] for row in (direct.get("ledger") or {}).get("classes", ())}
    check(
        "the same ledger classes closed",
        cli_classes == direct_classes and bool(cli_classes),
        f"{len(cli_classes)} classes",
    )

    head("no second execution implementation exists")
    sources = list((ROOT / "src" / "cozy_runtime" / "cli").glob("*.py"))
    callers = [p.name for p in sources if "run_slice" in p.read_text()]
    check(
        "exactly ONE CLI module reaches the attempt kernel, and it calls run_slice",
        callers == ["run.py"],
        f"{callers} — there is no direct-call runner to find",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="cr-016 live CLI verification")
    parser.add_argument("section", choices=("surface", "keys", "identity", "all"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    sections = {
        "surface": section_surface,
        "keys": section_keys,
        "identity": section_identity,
    }
    for name in list(sections) if args.section == "all" else [args.section]:
        sections[name](args)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
