"""The stage scheduler's pure core, driven with recorded real-run facts.

Every size, time and rate comes from `testdata/stage-costs/recorded.json`: Runtime receipts of
the six-job SDXL/Anima benchmark on the RTX 4070 Laptop, H3 fp8 runs on 1/2/4 H100 SXM, Qwen-
Image-2 calls on H100, and `paging.partition` of each architecture. One modelling assumption
is named where it is used: per-block compute is not recorded, so a step's measured compute is
spread over its blocks in proportion to their bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozy_runtime.author import Model, uses_components
from cozy_runtime.author._model import component_use
from cozy_runtime.internal import weight_policy
from cozy_runtime.internal.stages import (
    CostBook,
    StageCost,
    StageKind,
    StageSample,
    WeightSet,
    observed_graph,
    static_graph,
)
from cozy_runtime.internal.worker.stage_policy import (
    Gpu,
    Host,
    Job,
    JobStage,
    Machine,
    Prefetch,
    Prepare,
    Turn,
    Wait,
    decide,
    floor,
    load_s,
    refusal,
)

FX = json.loads((Path(__file__).parent / "testdata/stage-costs/recorded.json").read_text())
LAPTOP, H100 = FX["machines"]["rtx4070_laptop"], FX["machines"]["h100_sxm"]
SDXL, ANIMA, H3, QWEN = FX["sdxl"], FX["anima"], FX["h3"], FX["qwen_image_2"]


def ns(seconds: float) -> int:
    return round(seconds * 1e9)


def sets(
    model: str, spec: dict[str, dict[str, list[int] | int]], *names: str
) -> tuple[WeightSet, ...]:
    rows = []
    for name in names:
        row = spec[name]
        assert isinstance(row["common"], int) and isinstance(row["blocks"], list)
        rows.append(WeightSet(f"{model}:{name}", name, row["common"], tuple(row["blocks"])))
    return tuple(rows)


def measured(wall: float, *, gap: float = 0.0, growth: int = 0, floor: int = 0) -> StageCost:
    return StageCost(runs=4, wall_s=wall, gap_s=gap, passes=1.0, growth=growth, floor_growth=floor)


def stage(
    model: str,
    method: str,
    regions: tuple[WeightSet, ...],
    calls: int = 1,
    cost: dict[int, StageCost] | None = None,
) -> JobStage:
    kind = StageKind(model, method, tuple(r.component for r in regions))
    return JobStage(kind=kind, sets=regions, calls=calls, cost=cost or {})


def sdxl_stages() -> tuple[JobStage, ...]:
    return (
        # The receipt's encode covers both calls (prompt, negative): one call is half of it.
        stage(
            "sdxl",
            "encode",
            sets("sdxl", SDXL, "text_encoder", "text_encoder_2"),
            2,
            cost={1: measured(SDXL["encode_s"] / 2)},
        ),
        stage(
            "sdxl",
            "denoise",
            sets("sdxl", SDXL, "unet"),
            SDXL["steps"],
            cost={1: measured(SDXL["step_s"], growth=SDXL["unet_growth"])},
        ),
        stage(
            "sdxl",
            "decode",
            sets("sdxl", SDXL, "vae"),
            cost={1: measured(SDXL["decode_s"], growth=SDXL["decode_growth"])},
        ),
    )


def anima_stages() -> tuple[JobStage, ...]:
    """Anima today is one `render` scope over all four components (derived wall: encode +
    first step + steady steps + decode from the same receipt)."""
    wall = (
        ANIMA["encode_s"]
        + ANIMA["first_step_s"]
        + (ANIMA["steps"] - 1) * ANIMA["step_s"]
        + ANIMA["decode_s"]
    )
    regions = sets("anima", ANIMA, "text_encoder", "text_conditioner", "transformer", "vae")
    return (
        stage(
            "anima", "render", regions, cost={1: measured(wall, growth=ANIMA["transformer_growth"])}
        ),
    )


def laptop(
    ordinal: int = 0, resident: dict[str, int] | None = None, total: int | None = None
) -> Gpu:
    return Gpu(
        ordinal=ordinal,
        total=LAPTOP["vram_bytes"] if total is None else total,
        context=LAPTOP["context_bytes"] if total is None else 0,
        link_gbps=LAPTOP["pinned_gbps"],
        resident=resident or {},
    )


def laptop_host(*pinned: JobStage) -> Host:
    return Host(
        pinned={s.content: s.total for st in pinned for s in st.sets},
        cache_gbps=LAPTOP["cache_gbps"],
        disk_gbps=LAPTOP["disk_gbps"],
        spawn_s=LAPTOP["executor_start_s"],
    )


def h100(ordinal: int, resident: dict[str, int] | None = None) -> Gpu:
    return Gpu(
        ordinal=ordinal,
        total=H100["vram_bytes"],
        link_gbps=H100["link_gbps"],
        resident=resident or {},
    )


def h100_host(*pinned: JobStage) -> Host:
    return Host(
        pinned={s.content: s.total for st in pinned for s in st.sets},
        cache_gbps=H100["link_gbps"],
        disk_gbps=H100["link_gbps"],
        spawn_s=H100["executor_start_s"],
    )


# --------------------------------------------------------------------------- the cost book


class Sdxlish(Model[object]):
    """SDXL's scope shape (`packages/sdxl`): encode x1-2, denoise per step, decode."""

    @uses_components("text_encoder", "text_encoder_2")
    def encode(self, text: str) -> str:
        return text

    @uses_components("unet")
    def denoise(self, step: int) -> int:
        return step

    @uses_components("vae")
    def decode(self, latents: int) -> int:
        return latents


def test_the_graph_is_inferred_from_the_scopes_a_package_already_declares() -> None:
    use = {"Sdxlish": component_use(Sdxlish)}
    declared = (c for names in use["Sdxlish"].values() for c in names)
    opaque = static_graph("sdxl/generate", SDXL["cell"], "sdxl", declared)
    assert not opaque.observed and len(opaque.stages) == 1
    assert set(opaque.stages[0].kind.components) == {
        "text_encoder",
        "text_encoder_2",
        "unet",
        "vae",
    }

    model = Sdxlish.for_test()
    for text in ("a lighthouse at dusk", ""):  # prompt, then the negative when guidance > 1
        model.encode(text)
    for step in range(SDXL["steps"]):
        model.denoise(step)
    model.decode(0)
    samples = [
        StageSample(kind=StageKind("Sdxlish", call.method, call.components))
        for call in model.harness.calls
    ]
    graph = observed_graph("sdxl/generate", SDXL["cell"], samples)
    assert [(s.kind.method, s.calls) for s in graph.stages] == [
        ("encode", 2),
        ("denoise", SDXL["steps"]),
        ("decode", 1),
    ]
    assert graph.stages[0].kind.components == use["Sdxlish"]["encode"]


def sdxl_samples() -> list[StageSample]:
    encode, denoise, decode = (s.kind for s in sdxl_stages())
    half = ns(SDXL["encode_s"] / 2)
    samples = [StageSample(kind=encode, wall_ns=half), StageSample(kind=encode, wall_ns=half)]
    samples.append(
        StageSample(
            kind=denoise, wall_ns=ns(SDXL["first_step_s"]), growth_bytes=SDXL["unet_growth"]
        )
    )
    samples += [
        StageSample(kind=denoise, wall_ns=ns(SDXL["step_s"]), growth_bytes=SDXL["unet_growth"])
    ] * 19
    samples.append(
        StageSample(kind=decode, wall_ns=ns(SDXL["decode_s"]), growth_bytes=SDXL["decode_growth"])
    )
    return samples


def test_book_learns_the_sdxl_graph_and_steady_step_and_survives_a_restart(tmp_path: Path) -> None:
    book = CostBook()
    book.record("sdxl/generate", SDXL["cell"], sdxl_samples())
    graph = book.graph("sdxl/generate", SDXL["cell"])
    assert graph is not None and graph.observed
    assert [(s.kind.method, s.calls) for s in graph.stages] == [
        ("encode", 2),
        ("denoise", 20),
        ("decode", 1),
    ]
    denoise = book.cost(graph.stages[1].kind, SDXL["cell"], 1)
    assert denoise is not None and denoise.runs == 20
    # The 1.52 s first step washes out: the book predicts the recorded steady step.
    assert denoise.wall_s == pytest.approx(SDXL["step_s"], rel=0.01)
    assert denoise.growth == SDXL["unet_growth"]

    path = tmp_path / "stage-costs.json"
    book.save(path)
    again = CostBook.load(path)
    assert again.costs == book.costs and again.graphs == book.graphs
    path.write_bytes(b"{ torn")
    assert CostBook.load(path).costs == {}  # relearned, never refused


def test_a_failed_call_forgets_its_growth_but_keeps_the_graph_and_the_floor() -> None:
    book = CostBook()
    book.record("sdxl/generate", SDXL["cell"], sdxl_samples())
    decode = sdxl_stages()[2].kind
    shortfall = StageSample(kind=decode, ok=False, floor_growth=SDXL["decode_growth"] // 8)
    book.record("sdxl/generate", SDXL["cell"], [shortfall])
    cost = book.cost(decode, SDXL["cell"], 1)
    assert cost is not None and cost.growth == 0 and cost.runs == 1
    assert cost.floor_growth == SDXL["decode_growth"] // 8
    graph = book.graph("sdxl/generate", SDXL["cell"])
    assert graph is not None and len(graph.stages) == 3


# --------------------------------------------------------------------------- the cost model


def steady_pass(compute: list[float], sizes: list[int], resident: list[bool], link: float) -> float:
    """C2's per-region event model (`weight_policy.simulate`), two regions in flight."""
    streamed = [not hot for hot in resident]
    return weight_policy.simulate(sizes, compute, streamed, 2, link * 1e9)


def proportional(step_s: float, sizes: list[int]) -> list[float]:
    """Per-block compute is unrecorded: spread the measured step by bytes (named assumption)."""
    total = sum(sizes)
    return [step_s * size / total for size in sizes]


def test_overlapped_streaming_hides_the_tail_sdxl_streamed_synchronously_at_3_gib() -> None:
    blocks = SDXL["unet"]["blocks"]
    keep = SDXL["streamed_3gib"]["keep"]
    # The fixture's block order reproduces the tail the R19 run streamed every step.
    assert sum(blocks[keep:]) == SDXL["streamed_3gib"]["tail_bytes_per_step"]
    sizes = [SDXL["unet"]["common"], *blocks]
    resident = [True] * (1 + keep) + [False] * (len(blocks) - keep)
    compute = proportional(SDXL["step_s"], sizes)

    pinned = steady_pass(compute, sizes, resident, LAPTOP["pinned_gbps"])
    # 2.64 GB at 11.9 GB/s hides; what is left is the model's per-copy and per-block costs
    assert pinned == pytest.approx(SDXL["step_s"], rel=0.02)
    assert SDXL["streamed_3gib"]["step_s"] > 2.5 * pinned  # the recorded synchronous step

    disk = steady_pass(compute, sizes, resident, LAPTOP["disk_gbps"])
    copy = SDXL["streamed_3gib"]["tail_bytes_per_step"] / (LAPTOP["disk_gbps"] * 1e9)
    assert copy <= disk <= copy + SDXL["step_s"]  # from disk the link is the step


def test_overlapped_streaming_hides_anima_serial_cfg_at_3_gib() -> None:
    blocks = ANIMA["transformer"]["blocks"]
    keep = ANIMA["streamed_3gib"]["keep"]
    assert (
        ANIMA["calls_per_step"] * sum(blocks[keep:])
        == ANIMA["streamed_3gib"]["tail_bytes_per_step"]
    )
    sizes = [ANIMA["transformer"]["common"], *blocks]
    resident = [True] * (1 + keep) + [False] * (len(blocks) - keep)
    call = steady_pass(
        proportional(ANIMA["step_s"] / 2, sizes), sizes, resident, LAPTOP["pinned_gbps"]
    )
    assert 2 * call == pytest.approx(ANIMA["step_s"], rel=0.01)
    assert ANIMA["streamed_3gib"]["step_s"] > 1.8 * 2 * call


def test_h3_fp8_streams_its_whole_dit_on_h100_within_the_phase_2_gate() -> None:
    """Gate: H3 fp8 streaming <= 1.10x all-resident. Block bytes are the recorded DiT size over
    its 50 blocks (derived)."""
    sizes = [H3["dit_block_bytes_derived"]] * H3["dit_blocks"]
    step = H3["widths"]["1"]["step_s"]
    streamed = steady_pass(
        proportional(step, sizes), sizes, [False] * len(sizes), H100["link_gbps"]
    )
    assert streamed <= 1.10 * step
    assert streamed == pytest.approx(step, rel=0.01)


# --------------------------------------------------------------------------- one GPU: pipelining


def test_the_next_model_prepares_and_prefills_while_sdxl_denoises_then_takes_the_gpu() -> None:
    sdxl, anima = sdxl_stages(), anima_stages()
    unet = sdxl[1].sets[0]
    machine = Machine((laptop(resident={unet.content: unet.total}),), laptop_host(*sdxl))
    running = Job(
        key="sdxl-1",
        root="sdxl-1",
        priority=0,
        arrival=1,
        stages=sdxl,
        phase="running",
        at=1,
        gpus=(0,),
        warm=((0,),),
    )
    queued = Job(key="anima-2", root="anima-2", priority=0, arrival=2, stages=anima)

    plan = decide(machine, [running, queued])
    assert plan.turns == ()
    assert plan.prepares == (Prepare("anima-2", (0,)),)  # executor, imports, meta, overlapping
    assert plan.waits == (Wait("anima-2", "sdxl-1"),)
    assert [p.job for p in plan.prefills] == ["anima-2"]  # SDXL is already pinned
    assert set(plan.prefills[0].contents) == {s.content for s in anima[0].sets}
    assert plan.prefetches == (Prefetch("sdxl-1", (0,), 2),)  # SDXL's decode comes first

    decoding = Job(
        key="sdxl-1",
        root="sdxl-1",
        priority=0,
        arrival=1,
        stages=sdxl,
        phase="running",
        at=2,
        gpus=(0,),
    )
    assert decide(machine, [decoding, queued]).prefetches == (Prefetch("anima-2", (0,), 0),)

    # Past its last observed stage SDXL is on the CPU (pixels, WebP): it claims nothing.
    tail = Job(
        key="sdxl-1",
        root="sdxl-1",
        priority=0,
        arrival=1,
        stages=sdxl,
        phase="gap",
        at=3,
        gpus=(0,),
    )
    assert decide(machine, [tail, queued]).turns == (Turn("anima-2", (0,), 0, False),)


def test_a_pinned_model_returns_at_link_speed_instead_of_the_recorded_refill() -> None:
    anima = anima_stages()[0]
    total = sum(s.total for s in anima.sets)
    gpu = laptop()
    pinned = load_s(anima.sets, gpu, laptop_host(anima))
    cold = load_s(anima.sets, gpu, laptop_host())
    assert pinned == pytest.approx(total / (LAPTOP["pinned_gbps"] * 1e9))
    assert pinned < 0.5 < 3.0 < cold  # vs 14-26 s grant->invoke recorded for Anima today


def test_equal_priority_jobs_on_one_gpu_run_in_arrival_order() -> None:
    sdxl = sdxl_stages()
    machine = Machine((laptop(),), laptop_host(*sdxl))
    first = Job(key="a", root="a", priority=0, arrival=1, stages=sdxl, gpus=(0,))
    second = Job(key="b", root="b", priority=0, arrival=2, stages=sdxl, gpus=(0,))
    plan = decide(machine, [second, first])
    assert plan.turns == (Turn("a", (0,), 0, False),)
    assert plan.waits == (Wait("b", "a"),)


# --------------------------------------------------------------------------- bubbles


def h3_long_form(gap: float) -> tuple[JobStage, ...]:
    """The next segment's first stage after the MP4 finish of the previous one."""
    te = WeightSet("h3:text_encoder", "text_encoder", H3["text_encoder_bytes"])
    return (
        stage(
            "h3",
            "condition_text",
            (te,),
            cost={1: measured(H3["widths"]["1"]["encode_s"], gap=gap)},
        ),
    )


def qwen_stages() -> tuple[JobStage, ...]:
    denoise = QWEN["first_step_s"] + (QWEN["steps"] - 1) * QWEN["step_s"]
    return (
        stage(
            "qwen",
            "encode",
            sets("qwen", QWEN, "text_encoder"),
            cost={1: measured(QWEN["encode_s"])},
        ),
        stage("qwen", "denoise", sets("qwen", QWEN, "transformer"), cost={1: measured(denoise)}),
        stage("qwen", "decode", sets("qwen", QWEN, "vae"), cost={1: measured(QWEN["decode_s"])}),
    )


def test_a_qwen_image_fills_the_h3_mp4_finish_gap_stage_by_stage() -> None:
    h3 = h3_long_form(gap=H3["x264_finish_s"])
    te = h3[0].sets[0]
    qwen = qwen_stages()
    machine = Machine((h100(0, {te.content: te.total}),), h100_host(*qwen))
    h3_job = Job(
        key="h3", root="h3", priority=0, arrival=1, stages=h3, phase="gap", gpus=(0,), warm=((0,),)
    )

    fresh = Job(key="qwen", root="qwen", priority=0, arrival=2, stages=qwen)
    assert decide(machine, [h3_job, fresh]).turns == (Turn("qwen", (0,), 0, True),)

    # 1.6 s into the gap the 6.8 s loop plus 0.6 s of DiT load still ends before it does.
    h3_later = Job(
        key="h3", root="h3", priority=0, arrival=1, stages=h3, phase="gap", gpus=(0,), elapsed_s=1.6
    )
    denoising = Job(
        key="qwen", root="qwen", priority=0, arrival=2, stages=qwen, at=1, gpus=(0,), warm=((0,),)
    )
    assert decide(machine, [h3_later, denoising]).turns == (Turn("qwen", (0,), 1, True),)

    # 12 s in, it would not: Qwen waits and H3's segment starts on time.
    h3_late = Job(
        key="h3",
        root="h3",
        priority=0,
        arrival=1,
        stages=h3,
        phase="gap",
        gpus=(0,),
        elapsed_s=12.0,
    )
    plan = decide(machine, [h3_late, denoising])
    assert plan.turns == () and plan.waits == (Wait("qwen", "h3"),)


def test_no_bubble_in_an_unmeasured_gap_or_over_the_next_stage() -> None:
    qwen = qwen_stages()
    fresh = Job(key="qwen", root="qwen", priority=0, arrival=2, stages=qwen)
    unmeasured = Job(
        key="h3",
        root="h3",
        priority=0,
        arrival=1,
        stages=h3_long_form(gap=0.0),
        phase="gap",
        gpus=(0,),
    )
    machine = Machine((h100(0),), h100_host(*qwen))
    plan = decide(machine, [unmeasured, fresh])
    assert plan.turns == () and plan.prepares == (Prepare("qwen", (0,)),)

    h3 = h3_long_form(gap=H3["x264_finish_s"])
    crowded = h100(0, {h3[0].sets[0].content: H100["vram_bytes"] - 10**9})
    busy = Job(key="h3", root="h3", priority=0, arrival=1, stages=h3, phase="gap", gpus=(0,))
    assert decide(Machine((crowded,), h100_host(*qwen)), [busy, fresh]).turns == ()


# --------------------------------------------------------------------------- several GPUs


def h3_stages(*widths: int) -> tuple[JobStage, ...]:
    te = WeightSet("h3:text_encoder", "text_encoder", H3["text_encoder_bytes"])
    dit = WeightSet("h3:dit", "dit", 0, (H3["dit_block_bytes_derived"],) * H3["dit_blocks"])
    video = WeightSet("h3:video_vae", "video_vae", H3["video_vae_bytes"])
    audio = WeightSet("h3:audio_vae", "audio_vae", H3["audio_vae_bytes"])
    row = {w: H3["widths"][str(w)] for w in widths}
    return (
        stage(
            "h3",
            "condition_text",
            (te,),
            cost={w: measured(r["encode_s"]) for w, r in row.items()},
        ),
        stage("h3", "sample", (dit,), cost={w: measured(30 * r["step_s"]) for w, r in row.items()}),
        stage(
            "h3",
            "decode_audio",
            (audio,),
            cost={w: measured(r["decode_audio_s"]) for w, r in row.items()},
        ),
        stage(
            "h3",
            "decode_video",
            (video,),
            cost={w: measured(r["decode_video_s"]) for w, r in row.items()},
        ),
    )


def four_h100(stages: tuple[JobStage, ...]) -> Machine:
    return Machine(tuple(h100(o) for o in range(4)), h100_host(*stages))


@pytest.mark.parametrize(
    ("remaining", "chosen", "now"), [(100.0, (0, 1, 2, 3), False), (400.0, (0, 1), True)]
)
def test_width_follows_recorded_ulysses_scaling_and_what_is_already_running(
    remaining: float, chosen: tuple[int, ...], now: bool
) -> None:
    """Two H3 fp8 clips on 4x H100. With another clip on GPUs 2-3 finishing in `remaining`
    seconds, waiting for all four (7.84 s/step) beats starting now on two (15.40 s/step)
    only while that clip is nearly done."""
    stages = h3_stages(1, 2, 4)
    sample = stages[1].cost[2].wall_s
    tail = stages[2].cost[2].wall_s + stages[3].cost[2].wall_s
    other = Job(
        key="other",
        root="other",
        priority=0,
        arrival=1,
        stages=stages,
        phase="running",
        at=1,
        gpus=(2, 3),
        elapsed_s=sample + tail - remaining,
    )
    job = Job(key="next", root="next", priority=0, arrival=2, stages=stages, widths=(1, 2, 4))
    plan = decide(four_h100(stages), [other, job])
    if now:
        assert plan.turns == (Turn("next", chosen, 0, False),)
    else:
        assert plan.turns == () and plan.prepares == (Prepare("next", chosen),)


def test_unmeasured_work_takes_the_widest_width_the_machine_forms() -> None:
    stages = h3_stages()  # no costs recorded yet
    plan = decide(
        four_h100(stages),
        [Job(key="first", root="first", priority=0, arrival=1, stages=stages, widths=(1, 2, 4))],
    )
    assert plan.turns == (Turn("first", (0, 1, 2, 3), 0, False),)


def test_independent_jobs_run_at_once_each_where_its_weights_are() -> None:
    sdxl, anima = sdxl_stages(), anima_stages()
    unet, dit = sdxl[1].sets[0], anima[0].sets[2]
    machine = Machine(
        (laptop(0, {dit.content: dit.total}), laptop(1, {unet.content: unet.total})),
        laptop_host(*sdxl, *anima),
    )
    jobs = [
        Job(key="sdxl", root="sdxl", priority=0, arrival=1, stages=sdxl),
        Job(key="anima", root="anima", priority=0, arrival=2, stages=anima),
    ]
    assert decide(machine, jobs).turns == (
        Turn("sdxl", (1,), 0, False),
        Turn("anima", (0,), 0, False),
    )


# --------------------------------------------------------------------------- admission


def test_refused_only_below_the_floor_and_with_numbers() -> None:
    qwen = qwen_stages()
    encode = qwen[0].sets[0]
    need = floor(qwen[0].sets, 0)
    assert need == encode.common + max(encode.blocks)
    job = Job(key="qwen", root="qwen", priority=0, arrival=1, stages=qwen)

    small = Machine((laptop(total=2 << 30),), laptop_host())
    refused = refusal(job, small)
    assert refused is not None and refused.code == "NO_CAPACITY"
    assert (refused.need, refused.capacity) == (need, 2 << 30)
    assert str(need) in refused.detail and "encode (text_encoder)" in refused.detail
    assert decide(small, [job]).refusals == (refused,)

    # On the 8 GB laptop Qwen streams its 17.5 GB encoder: queued, never refused.
    assert refusal(job, Machine((laptop(),), laptop_host())) is None


def test_a_mode_specific_peak_never_refuses_sdxl_at_1_gib() -> None:
    """The untiled decode grew 3.06 GB on the 8 GB card; at a 1 GiB budget the memory policy
    tiles instead, so only the measured cheapest-rung growth can refuse."""
    sdxl = sdxl_stages()
    assert sdxl[2].cost[1].growth > 1 << 30
    one_gib = Machine((laptop(total=1 << 30),), laptop_host(*sdxl))
    job = Job(key="sdxl", root="sdxl", priority=0, arrival=1, stages=sdxl)
    assert refusal(job, one_gib) is None
    assert decide(one_gib, [job]).turns == (Turn("sdxl", (0,), 0, False),)
