#!/usr/bin/env python3
"""Live verification of the author surface. Not a test suite: a real run.

Every section below EXECUTES the real endpoints in `examples/` through the real invocation
kernel and the real construction plane, and prints what actually happened. Refusals live in
`scripts/redarm.py`, which plants each violation for real and observes the typed refusal.

    uv run python scripts/verify.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import tiny_artifacts as art  # noqa: E402
import tiny_h3 as h3  # noqa: E402
import tiny_sdxl as ep  # noqa: E402
from cozy_runtime.author import (  # noqa: E402
    Clamp,
    EgressResponse,
    Invocation,
    ModelRegistry,
    Recipe,
    describe,
    invoke,
    registered_adapters,
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
from cozy_runtime.internal.bindings import read_table, resolve  # noqa: E402
from cozy_runtime.internal.residency import ResidentBackend, StagedBackend  # noqa: E402
from tiny_diffusers import MiniMaxH3ModularPipeline, TinyScheduler  # noqa: E402

RELEASE = "release:digest(source,lock,manifest)"


def head(title: str) -> None:
    print(f"\n=== {title}")


def sdxl_registry() -> tuple[ModelRegistry, ep.SdxlModel]:
    """One release, one artifact, every SDXL slot — the coalescing case."""
    registry = ModelRegistry(release=RELEASE, backend=StagedBackend())
    artifact = art.sdxl()
    bound = [
        registry.acquire(f"{call}.models.model", ep.SdxlModel, artifact)
        for call in ("generate", "plan", "generate_ref")
    ]
    return registry, bound[0]


def h3_registry(*, dual: bool = True) -> tuple[ModelRegistry, h3.Fl2VAModel, h3.Ref2VAModel]:
    """Two ROLE CLASSES, one packaged artifact: always two generations (§1.1)."""
    registry = ModelRegistry(release=RELEASE, backend=StagedBackend())
    fl = registry.acquire("generate.models.model", h3.Fl2VAModel, art.h3(task="fl2va", dual=dual))
    ref = registry.acquire(
        "generate_ref.models.model", h3.Ref2VAModel, art.h3(task="ref2va", dual=dual)
    )
    return registry, fl, ref


def main() -> int:
    # ---------------------------------------------------------------- describe
    head("describe — the surface derived from signatures alone")
    from cozy_runtime.author._demand import feature_specs

    for app in (ep.app, h3.app):
        for surface in describe(app):
            spelling = "async def" if surface.is_async else "def"
            caps = ", ".join(sorted(surface.capabilities)) or "(none)"
            axes = ", ".join(f"{s.axis}<-{s.field}" for s in feature_specs(surface.payload_type))
            print(f"  {surface.name:<13} {spelling:<9} gpu={surface.gpu!s:<5} capabilities: {caps}")
            print(f"  {'':<13} params: " + ", ".join(f"{p.name}:{p.role}" for p in surface.params))
            print(f"  {'':<13} features: {axes or '(none)'}")
            for binding in surface.model_bindings:
                stamps = ", ".join(f"{k}={v}" for k, v in binding.stamps.items()) or "(none)"
                print(f"  {'':<13} slot {binding.path} -> {binding.class_key} stamps: {stamps}")
                for method, names in binding.components.items():
                    print(f"  {'':<15}   {method}: {', '.join(names)}")

    # ---------------------------------------------------------------- bindings
    head("bindings — code states capability, endpoint.toml states selection")
    slots = [b for s in describe(h3.app) for b in s.model_bindings]
    slots += [b for s in describe(ep.app) for b in s.model_bindings]
    table = read_table(ROOT / "examples" / "endpoint.toml")
    for resolved in resolve(slots, table).values():
        print(f"  {resolved}")
    override = resolve(slots, table, overrides={"generate.models.model": "cozy/h3-fl2va@v2"})
    print(f"  --model overrides:      {override['generate.models.model']}")
    deployed = resolve(
        slots, table, overrides={"SdxlModel": "x/y@1"}, deploy={"SdxlModel": "cozy/gallery@v9"}
    )
    print(f"  deploy binding governs: {deployed['generate.models.model']}")

    # ---------------------------------------------------------------- construction
    head("construction — walk and match BEFORE allocation, then ONE fill path")
    backend = StagedBackend()
    registry = ModelRegistry(release=RELEASE, backend=backend)
    model = registry.acquire("generate.models.model", h3.Fl2VAModel, art.h3(task="fl2va"))
    record = registry.generations()[0].record
    print(f"  factory:      {record.factory_identity}")
    print(f"  components:   {', '.join(record.components)}")
    print(f"  schedulers:   {', '.join(record.schedulers)}")
    print(f"  destinations: {len(record.destinations)} logical tensors, "
          f"{record.filled} filled, {record.filled_bytes} bytes through the backend")
    print(f"  topology:     {record.topology_digest}")
    print(f"  identity:     {record.identity}")
    print(f"  checkpoint:   {model.checkpoint_ref}")
    print(f"  load(T) sugar is legal for the REGISTERED adapters only: "
          f"{', '.join(registered_adapters())}")

    head("derive == serve — the SAME factory and config bytes construct both graphs")
    derived = ModelRegistry(release=RELEASE, backend=None)
    derived.acquire("generate.models.model", h3.Fl2VAModel, art.h3(task="fl2va"), mode="derive")
    dr = derived.generations()[0].record
    print(f"  derive: {dr.identity} filled={dr.filled} (fake tensors, nothing allocated)")
    print(f"  serve:  {record.identity} filled={record.filled}")
    print(f"  identical construction identity: {dr.identity == record.identity}")

    head("narrow vs dual — the SAME destination graph from either artifact")
    narrow = ModelRegistry(release=RELEASE, backend=StagedBackend())
    narrow.acquire("generate.models.model", h3.Fl2VAModel, art.h3(task="fl2va"))
    dual = ModelRegistry(release=RELEASE, backend=StagedBackend())
    dual.acquire("generate.models.model", h3.Fl2VAModel, art.h3(task="fl2va", dual=True))
    nr, dur = narrow.generations()[0].record, dual.generations()[0].record
    print(f"  narrow artifact: {len(nr.destinations)} destinations {nr.topology_digest}")
    print(f"  dual artifact:   {len(dur.destinations)} destinations {dur.topology_digest}")
    print(f"  same destination graph: {nr.topology_digest == dur.topology_digest}")
    print(f"  the dual's other transformer is an unbound SIBLING, not an extra key: "
          f"{dur.ignored_extras == ()}")

    head("baked vs full — the modulation source is a CONSTRUCTION fact, not a mode")
    baked = ModelRegistry(release=RELEASE, backend=StagedBackend())
    baked_model = baked.acquire(
        "generate.models.model", h3.Fl2VAModel, art.h3(task="fl2va", baked=True)
    )
    br = baked.generations()[0].record
    print(f"  full  topology: {len(record.destinations)} destinations, "
          f"modulation={type(model.modulation).__name__}")
    print(f"  baked topology: {len(br.destinations)} destinations, "
          f"modulation={type(baked_model.modulation).__name__}")
    leaves = {d.key.split(".")[-1] for d in br.destinations}
    print("  baked OMITS the AdaLN projections and CARRIES exact tables: "
          f"{sorted(leaves - {d.key.split('.')[-1] for d in record.destinations})}")

    head("construction identity — equal coalesces ONE generation, unequal stays distinct")
    coalescing, _ = sdxl_registry()
    instances = {id(g.model) for g in coalescing.generations()}
    print(f"  3 SDXL binding paths, 1 artifact -> {len(coalescing.generations())} generation(s), "
          f"{len(instances)} instance(s)")
    for generation in coalescing.generations():
        print(f"    {generation.identity}")
        print(f"      paths: {', '.join(generation.paths)}")
    split = ModelRegistry(release=RELEASE, backend=StagedBackend())
    a = art.sdxl()
    split.acquire("generate.models.model", ep.SdxlModel, a)
    split.acquire("plan.models.model", ep.SdxlModel, art.named(a, "cozy/tiny-sdxl@v2#9f9f"))
    print(f"  a different artifact snapshot -> {len(split.generations())} generations")
    twins, _, _ = h3_registry()
    print(f"  H3's two ROLE CLASSES on ONE dual artifact -> {len(twins.generations())} generations "
          "(identity includes the class)")

    # ---------------------------------------------------------------- invocation
    head("generate — a real attempt through the real kernel")
    models, sdxl_model = sdxl_registry()
    env = invoke_with_fakes(
        ep.app.get("generate"),
        {"prompt": "a cozy cabin", "aspect_ratio": "16:9", "seed": 7},
        models=models.models("generate"),
    )
    print(f"  result:      {env.result}")
    print(f"  outputs:     {env.outputs}")
    print(f"  features:    {env.features.values} {env.features.digest}")
    print(f"  overlay:     {env.overlay_digest}")
    print(f"  telemetry:   {len(env.telemetry)} events, "
          f"stages={[e.name for e in env.telemetry if e.kind == 'stage']}")
    print(f"  encoded PNG: {env.outputs[0]['size_bytes']} bytes, {env.outputs[0]['digest']}")

    head("H3 — component-staged, granular scopes, both operation twins")
    h3_models, fl_model, ref_model = h3_registry()
    fl = invoke_with_fakes(
        h3.app.get("generate"),
        {"prompt": "a storm over the bay", "duration_s": 2, "seed": 7},
        models=h3_models.models("generate"),
    )
    print(f"  fl2va: {fl.result}")
    reference = ROOT / "verify-reference.png"
    reference.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    clip = ROOT / "verify-clip.wav"
    clip.write_bytes(b"RIFF....WAVEfmt ")
    ref = invoke_with_fakes(
        h3.app.get("generate_ref"),
        {
            "prompt": "the same subject, at sea",
            "references": [
                {"kind": "image", "image": str(reference)},
                {"kind": "video", "video": str(reference), "soundtrack": True, "fps": 24.0},
                {"kind": "audio", "audio": str(clip)},
            ],
            "duration_s": 1,
            "seed": 3,
        },
        models=h3_models.models("generate_ref"),
        settings=h3.RefServeSettings(),
    )
    print(f"  ref2va: {ref.result}")
    print(f"  ordered union: 3 references (image, soundtracked video, audio) hydrated, "
          f"facts={ref.facts_digest}")
    print(f"  the two twins are separate instances over shared components: "
          f"{id(fl_model) != id(ref_model)}")

    head("RequestView — exactly three SDK members, fresh schedulers per request")
    ctx = fake_context(request_id="view-demo")
    view_a = fl_model.for_request(ctx, seed=11)
    view_b = fl_model.for_request(ctx, seed=11)
    s1, s2 = view_a.make_scheduler("video_scheduler"), view_b.make_scheduler("video_scheduler")
    print(f"  members: {[m for m in dir(view_a) if not m.startswith('_')]}")
    print(f"  fresh instance per call: {s1 is not s2}, family={s1.config['family']}")
    print(f"  same seed -> same stream: {view_a.generator.random() == view_b.generator.random()}")
    sampled = fl_model.for_request(ctx, seed=11, sampler="dpm")
    print(f"  sampler validated against the family table: "
          f"{sampled.make_scheduler('video_scheduler').config}")
    print(f"  adapters (immutable applied stack): {view_a.adapters}")

    head("DerivedCache — a miss builds under the ACTIVE scope, a hit reuses it")
    _, cached, _ = h3_registry()
    live = cached.modulation
    assert isinstance(live, h3.LiveAdaLN)
    plan = h3.prepare(h3.GenerateInput(prompt="cache", duration_s=1), frames=39)
    view = cached.for_request(ctx, seed=5)
    cache_run = h3.start_run(view, plan, task="fl2va", steps=4, checkpoint=cached.checkpoint_ref)
    condition = h3.pack(plan, (1, 1), (), ())
    counts = [live.cache.entries]
    for _ in range(2):
        cached.denoise(cache_run, condition, frames=39)
        counts.append(live.cache.entries)
    replan = h3.start_run(view, plan, task="fl2va", steps=8, checkpoint=cached.checkpoint_ref)
    cached.denoise(replan, condition, frames=39)
    counts.append(live.cache.entries)
    print(f"  entries: start={counts[0]} miss->{counts[1]} hit->{counts[2]} "
          f"another exact plan->{counts[3]}")
    print(f"  {live.cache!r}")
    print(f"  key: {cache_run.modulation_key}")

    # ---------------------------------------------------------------- backends
    head("backend blindness — the same endpoint code on two fill routes")
    seen = {}
    for name, backend_impl in (("staged", StagedBackend()), ("resident", ResidentBackend())):
        registry = ModelRegistry(release=RELEASE, backend=backend_impl)
        registry.acquire("generate.models.model", h3.Fl2VAModel, art.h3(task="fl2va"))
        out = invoke_with_fakes(
            h3.app.get("generate"),
            {"prompt": "backend blind", "duration_s": 1, "seed": 5},
            models=registry.models("generate"),
        )
        seen[name] = (out.outputs[0]["digest"], backend_impl.route, len(backend_impl.events))
        print(f"  {name:<9} route={backend_impl.route:<9} fills={len(backend_impl.events):<3} "
              f"digest={out.outputs[0]['digest'][:28]}…")
    same_output = seen["staged"][0] == seen["resident"][0]
    other_route = seen["staged"][1] != seen["resident"][1]
    print(f"  identical output, different route: {same_output and other_route}")

    # ---------------------------------------------------------------- test seam
    head("Model.for_test + ModelTestHarness — author doubles, recorded scopes, no GPU")
    # An author-built double is STRUCTURAL: the declared state's type is the real pipeline
    # class, so the test says so once, here, and the endpoint stays fully typed.
    fake_pipe = cast("MiniMaxH3ModularPipeline", FakeH3Pipeline())
    tested = h3.Fl2VAModel.for_test(
        pipe=fake_pipe,
        modulation=h3.BakedModulationTables(fake_pipe),
        checkpoint_ref="test://h3-fl2va",
    )
    faked = invoke_with_fakes(
        h3.app.get("generate"),
        {"prompt": "a fake-backed handler", "duration_s": 1, "seed": 2},
        models={"model": tested},
    )
    print(f"  handler result: {faked.result}")
    print(f"  recorded scopes: {[(c.method, c.components) for c in tested.harness.calls]}")
    print(f"  components touched: {tested.harness.components()}")
    tested.harness.assert_scopes(
        "encode_presentation",
        "encode_visual_conditions",
        "denoise",
        "decode_audio",
        "decode_video",
    )
    print("  assert_scopes passed — the harness records, and invents nothing")

    # ---------------------------------------------------------------- unload
    head("unload — the runtime drains, then the generation releases its state")
    drained, resident, _ = h3_registry()
    drained.unload_all()
    print(f"  generations after unload_all: {len(drained.generations())}")
    print(f"  declared state released: {not hasattr(resident, 'pipe')}")

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
            ep.app.get("plan"), wire, models=models.models("plan"), recipes=recipes
        )
        rows = [f"{r.kind}:{r.field}={r.applied}<-{r.source}" for r in out.adjustments]
        print(f"  {label:<17} steps={out.result.steps:<3} overlay={out.overlay_digest[:24]}… "
              f"rows={rows}")

    head("two-layer clamp — API bounds REJECT, deployment policy CLAMPS visibly")
    try:
        invoke_with_fakes(ep.app.get("plan"), {"prompt": "p", "num_inference_steps": 99},
                          models=models.models("plan"))
    except Exception as exc:
        print(f"  API bound (le=50): {type(exc).__name__}: {exc}")
    clamped = invoke_with_fakes(
        ep.app.get("plan"),
        {"prompt": "p", "num_inference_steps": 40},
        models=models.models("plan"),
        clamps=(Clamp("num_inference_steps", "deployment caps steps at 30", max=30),),
    )
    print(f"  policy clamp:      steps={clamped.result.steps}, rows={clamped.adjustments}")
    print(f"  handler injected Adjustments? {'adjust' in ep.app.get('plan').surface.capabilities}")

    # ---------------------------------------------------------------- determinism
    head("determinism — view.generator is the ONE seeded spelling")
    again = invoke_with_fakes(
        ep.app.get("generate"),
        {"prompt": "a cozy cabin", "aspect_ratio": "16:9", "seed": 7},
        models=models.models("generate"),
    )
    other = invoke_with_fakes(
        ep.app.get("generate"),
        {"prompt": "a cozy cabin", "aspect_ratio": "16:9", "seed": 8},
        models=models.models("generate"),
    )
    print(f"  seed 7 twice same digest: {env.outputs[0]['digest'] == again.outputs[0]['digest']}")
    print(f"  seed 8 differs:           {env.outputs[0]['digest'] != other.outputs[0]['digest']}")
    unseeded = invoke_with_fakes(
        ep.app.get("generate"), {"prompt": "x"}, models=models.models("generate"),
        request_id="req-A",
    )
    unseeded_again = invoke_with_fakes(
        ep.app.get("generate"), {"prompt": "x"}, models=models.models("generate"),
        request_id="req-A",
    )
    print("  unseeded is derived from request_id, stable: "
          f"{unseeded.outputs[0]['digest'] == unseeded_again.outputs[0]['digest']}")

    head("ctx.boot_warmup — a real Context fact the handler cheapens off")
    warm = invoke_with_fakes(
        ep.app.get("generate"), {"prompt": "x"}, models=models.models("generate"),
        boot_warmup=True,
    )
    print(f"  warm pass telemetry events: {len(warm.telemetry)} (cold: {len(env.telemetry)})")

    # ---------------------------------------------------------------- preflight
    head("typed preflight — computed once, bound to admission, handed to the handler")
    ref_env = invoke_with_fakes(
        ep.app.get("generate_ref"),
        {"prompt": "with a reference", "reference": str(reference), "seed": 3},
        models=models.models("generate_ref"),
        settings=ep.RefSettings(references_enabled=True),
    )
    print(f"  facts digest:  {ref_env.facts_digest}")
    print(f"  handler rows:  {ref_env.adjustments}")
    print(f"  result:        {ref_env.result}")
    plain = invoke_with_fakes(
        ep.app.get("generate_ref"), {"prompt": "no reference"},
        models=models.models("generate_ref"), settings=ep.RefSettings(),
    )
    print(f"  no-reference facts digest differs: {plain.facts_digest != ref_env.facts_digest}")

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
                models=models.models("generate_ref"),
                settings=ep.RefSettings(references_enabled=False),
            ),
        )
    except Exception as exc:
        print(f"  preflight refusal: {type(exc).__name__}: {exc}")
    print(f"  spool contents after the refusal: {sorted(p.name for p in spool.iterdir())} "
          f"— the asset was never hydrated")
    shutil.rmtree(spool)
    reference.unlink()
    clip.unlink()

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
    direct_ctx = fake_context(request_id="direct-call")
    out_rec = fake_outputs(attempt)
    tel_rec = fake_telemetry(attempt, direct_ctx)
    result = ep.generate(
        direct_ctx,
        ep.Txt2ImgInput(prompt="hand-built payload", num_inference_steps=3),
        sdxl_model,
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


class FakeH3Pipeline:
    """An AUTHOR-BUILT double: the SDK cannot generically fake a pipeline's behavior, so a
    test supplies it explicitly. Nothing here is invented by the runtime (§1.6)."""

    def __init__(self) -> None:
        self.components: dict[str, object] = {
            "video_scheduler": TinyScheduler("flow_match_video"),
            "audio_scheduler": TinyScheduler("flow_match_audio"),
        }
        self.blocks = self
        self.topology = _Baked()

    def build_presentation(self, prompt: str, refs: object, keys: object) -> str:
        return f"fake:{prompt}"

    def text_encoder(self, presentation: str) -> tuple[int, int]:
        return (11, len(presentation))

    def encode_visual(self, source: object, **kwargs: object) -> int:
        return 22

    def encode_audio(self, source: object, **kwargs: object) -> int:
        return 33

    def prepare_latents(self, *, frames: int, generator: object) -> int:
        return frames

    def baked_tables(self, plan: object) -> tuple[int, ...]:
        return (1, 2, 3, 4)

    def step(self, *args: object) -> int:
        return 44

    def decode_video(self, latents: int) -> int:
        return 55

    def decode_audio(self, latents: int) -> tuple[int, int]:
        return (66, 48_000)


class _Baked:
    baked = True


if __name__ == "__main__":
    raise SystemExit(main())
