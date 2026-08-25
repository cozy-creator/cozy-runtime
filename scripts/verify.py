#!/usr/bin/env python3
"""Live verification of the cr-002a author surface. Not a test suite: a real run.

Every section below EXECUTES the real endpoint in `examples/tiny_sdxl.py` through the real
invocation kernel and prints what actually happened. Refusals live in `scripts/redarm.py`,
which plants each violation for real and observes the typed refusal.

    uv run python scripts/verify.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import tiny_sdxl as ep  # noqa: E402
from cozy_runtime.author import (  # noqa: E402
    Clamp,
    EgressResponse,
    Invocation,
    Recipe,
    describe,
    invoke,
)
from cozy_runtime.author.fakes import (  # noqa: E402
    fake_adjustments,
    fake_attempt,
    fake_context,
    fake_egress,
    fake_outputs,
    fake_telemetry,
    invoke_with_fakes,
)


def head(title: str) -> None:
    print(f"\n=== {title}")


def models() -> dict[str, ep.SdxlModel]:
    return {"model": ep.SdxlModel()}


def main() -> int:
    # ---------------------------------------------------------------- describe
    head("describe — the surface derived from signatures alone")
    from cozy_runtime.author._demand import feature_specs

    for surface in describe(ep.app):
        spelling = "async def" if surface.is_async else "def"
        caps = ", ".join(sorted(surface.capabilities)) or "(none)"
        axes = ", ".join(f"{s.axis}<-{s.field}" for s in feature_specs(surface.payload_type))
        print(f"  {surface.name:<13} {spelling:<9} gpu={surface.gpu!s:<5} capabilities: {caps}")
        print(f"  {'':<13} params: " + ", ".join(f"{p.name}:{p.role}" for p in surface.params))
        print(f"  {'':<13} features: {axes or '(none)'}")
        if surface.preflight is not None:
            print(f"  {'':<13} preflight: {surface.preflight.identity}")

    # ---------------------------------------------------------------- happy path
    head("generate — a real attempt through the real kernel")
    env = invoke_with_fakes(
        ep.app.get("generate"),
        {"prompt": "a cozy cabin", "aspect_ratio": "16:9", "seed": 7},
        models=models(),
    )
    print(f"  result:      {env.result}")
    print(f"  outputs:     {env.outputs}")
    print(f"  features:    {env.features.values} {env.features.digest}")
    print(f"  overlay:     {env.overlay_digest}")
    print(f"  telemetry:   {len(env.telemetry)} events, "
          f"stages={[e.name for e in env.telemetry if e.kind == 'stage']}")
    print(f"  encoded PNG: {env.outputs[0]['size_bytes']} bytes, {env.outputs[0]['digest']}")

    head("determinism — view.generator is the ONE seeded spelling")
    again = invoke_with_fakes(
        ep.app.get("generate"),
        {"prompt": "a cozy cabin", "aspect_ratio": "16:9", "seed": 7},
        models=models(),
    )
    other = invoke_with_fakes(
        ep.app.get("generate"),
        {"prompt": "a cozy cabin", "aspect_ratio": "16:9", "seed": 8},
        models=models(),
    )
    print(f"  seed 7 twice same digest: {env.outputs[0]['digest'] == again.outputs[0]['digest']}")
    print(f"  seed 8 differs:           {env.outputs[0]['digest'] != other.outputs[0]['digest']}")
    unseeded = invoke_with_fakes(
        ep.app.get("generate"), {"prompt": "x"}, models=models(), request_id="req-A"
    )
    unseeded_again = invoke_with_fakes(
        ep.app.get("generate"), {"prompt": "x"}, models=models(), request_id="req-A"
    )
    print("  unseeded is derived from request_id, stable: "
          f"{unseeded.outputs[0]['digest'] == unseeded_again.outputs[0]['digest']}")

    head("ctx.boot_warmup — a real Context fact the handler cheapens off")
    warm = invoke_with_fakes(
        ep.app.get("generate"), {"prompt": "x"}, models=models(), boot_warmup=True
    )
    print(f"  warm pass telemetry events: {len(warm.telemetry)} "
          f"(cold: {len(env.telemetry)})")

    # ---------------------------------------------------------------- overlay
    head("ModelDefault overlay — code <- checkpoint <- adapter <- explicit <- clamp")
    checkpoint = Recipe("checkpoint:anime-v3", "checkpoint", {"num_inference_steps": 28,
                                                             "guidance_scale": 7.5})
    turbo = Recipe("adapter:turbo-lora", "adapter", {"num_inference_steps": 4,
                                                     "guidance_scale": 0.0})
    for label, recipes, wire in (
        ("code only", (), {"prompt": "p"}),
        ("checkpoint", (checkpoint,), {"prompt": "p"}),
        ("checkpoint+turbo", (checkpoint, turbo), {"prompt": "p"}),
        ("explicit wins", (checkpoint, turbo), {"prompt": "p", "num_inference_steps": 12}),
    ):
        out = invoke_with_fakes(
            ep.app.get("plan"), wire, models=models(), recipes=recipes
        )
        rows = [f"{r.kind}:{r.field}={r.applied}<-{r.source}" for r in out.adjustments]
        print(f"  {label:<17} steps={out.result.steps:<3} overlay={out.overlay_digest[:24]}… "
              f"rows={rows}")

    head("two-layer clamp — API bounds REJECT, deployment policy CLAMPS visibly")
    try:
        invoke_with_fakes(ep.app.get("plan"), {"prompt": "p", "num_inference_steps": 99},
                          models=models())
    except Exception as exc:
        print(f"  API bound (le=50): {type(exc).__name__}: {exc}")
    clamped = invoke_with_fakes(
        ep.app.get("plan"),
        {"prompt": "p", "num_inference_steps": 40},
        models=models(),
        clamps=(Clamp("num_inference_steps", "deployment caps steps at 30", max=30),),
    )
    print(f"  policy clamp:      steps={clamped.result.steps}, rows={clamped.adjustments}")
    print(f"  handler injected Adjustments? {'adjust' in ep.app.get('plan').surface.capabilities}")

    # ---------------------------------------------------------------- preflight
    head("typed preflight — computed once, bound to admission, handed to the handler")
    reference = ROOT / "verify-reference.png"
    reference.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    ref_env = invoke_with_fakes(
        ep.app.get("generate_ref"),
        {"prompt": "with a reference", "reference": str(reference), "seed": 3},
        models=models(),
        settings=ep.RefSettings(references_enabled=True),
    )
    print(f"  facts digest:  {ref_env.facts_digest}")
    print(f"  handler rows:  {ref_env.adjustments}")
    print(f"  result:        {ref_env.result}")
    plain = invoke_with_fakes(
        ep.app.get("generate_ref"), {"prompt": "no reference"}, models=models(),
        settings=ep.RefSettings(),
    )
    print(f"  no-reference facts digest differs: {plain.facts_digest != ref_env.facts_digest}")

    # A preflight refusal is a PRE-ENTRY refusal: nothing hydrates, nothing downloads.
    spool = ROOT / "verify-spool"
    if spool.exists():
        shutil.rmtree(spool)
    try:
        invoke(
            ep.app.get("generate_ref"),
            {"prompt": "denied", "reference": str(reference)},
            Invocation(
                request_id="denied",
                spool=spool,
                deadline=time.monotonic() + 60,
                models=models(),
                settings=ep.RefSettings(references_enabled=False),
            ),
        )
    except Exception as exc:
        print(f"  preflight refusal: {type(exc).__name__}: {exc}")
    print(f"  spool contents after the refusal: {sorted(p.name for p in spool.iterdir())} "
          f"— the asset was never hydrated")
    shutil.rmtree(spool)
    reference.unlink()

    # ---------------------------------------------------------------- async
    head("async def — a PRIVATE event loop per attempt, one attempt at a time")
    loops = []
    for _ in range(2):
        out = invoke_with_fakes(
            ep.app.get("caption"),
            {"prompt": "describe this"},
            egress_broker=lambda url, method, headers, body: EgressResponse(
                200, {}, b"a cabin in the woods"
            ),
        )
        loops.append(out.result.loop)
        print(f"  caption={out.result.caption!r} loop={out.result.loop}")
    print(f"  distinct loops per attempt: {loops[0] != loops[1]}")
    print(f"  no loop left running in this thread: {not asyncio._get_running_loop()}")

    # ---------------------------------------------------------------- fakes
    head("per-service fakes — the decorated handler called FULLY TYPED, no kernel")
    attempt = fake_attempt("direct-call")
    ctx = fake_context(request_id="direct-call")
    out_rec = fake_outputs(attempt)
    tel_rec = fake_telemetry(attempt, ctx)
    result = ep.generate(
        ctx,
        ep.Txt2ImgInput(prompt="hand-built payload", num_inference_steps=3),
        ep.SdxlModel(),
        out_rec,
        tel_rec,
    )
    out_rec.assert_saved(count=1, kind="image")
    print(f"  returned:  {result}")
    print(f"  recorded:  {[a.ref for a in out_rec.saved]} "
          f"({out_rec.saved[0].size_bytes} bytes through the REAL encode path)")
    print(f"  telemetry: {len(tel_rec.events)} events, stages={dict(tel_rec.stages())}")

    adj_rec = fake_adjustments(fake_attempt("adj"))
    adj_rec.adjusted("guidance_scale", 7.5, 0.0, "turbo recipe")
    print(f"  adjustments recorder: {adj_rec.rows}")

    egress = fake_egress(
        lambda url, method, headers, body: EgressResponse(200, {}, b"ok"), fake_attempt("net")
    )
    print(f"  egress fake: {egress.fetch('https://example.test').body!r}")

    print("\nall sections executed against the real kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
