"""All 26 production `app.job` registrations, as `app.entrypoint` (xs-040 U2/U9)."""

from __future__ import annotations

from cozy_runtime.author import App, WeightsOutput

from . import derivations as d
from . import h3
from . import h3_tables as t

app = App()
LANE = 16 << 30

# seedvr2, birefnet, character-describer
app.entrypoint(
    d.seedvr2_normalize, weights=(WeightsOutput("model", max_new_bytes=2 << 20),), accelerator=False
)
app.entrypoint(
    d.birefnet_normalize,
    weights=(WeightsOutput("model", max_new_bytes=1 << 20),),
    accelerator=False,
)
app.entrypoint(
    d.describer_normalize,
    weights=(WeightsOutput("model", max_new_bytes=64 << 20),),
    accelerator=False,
)

# sdxl, anima
app.entrypoint(
    d.sdxl_fp8,
    name="sdxl-fp8",
    weights=(WeightsOutput("fp8", max_new_bytes=LANE),),
    accelerator=False,
)
app.entrypoint(
    d.sdxl_mxfp8,
    name="sdxl-mxfp8",
    weights=(WeightsOutput("mxfp8", max_new_bytes=LANE),),
    accelerator=False,
)
app.entrypoint(
    d.anima_fp8,
    name="anima-fp8",
    weights=(WeightsOutput("fp8", max_new_bytes=LANE),),
    accelerator=False,
)
app.entrypoint(
    d.anima_mxfp8,
    name="anima-mxfp8",
    weights=(WeightsOutput("mxfp8", max_new_bytes=LANE),),
    accelerator=False,
)

# minimax-h3-tools; the GPU ones declare no accelerator.
app.entrypoint(
    t.bf16_full, name="bf16-full", weights=(WeightsOutput("bf16-full", LANE),), accelerator=False
)
app.entrypoint(t.bf16_pruned, name="bf16-pruned", weights=(WeightsOutput("bf16-pruned", LANE),))
app.entrypoint(t.fp8_pruned, name="fp8-pruned", weights=(WeightsOutput("fp8-pruned", LANE),))
app.entrypoint(t.mxfp8_pruned, name="mxfp8-pruned", weights=(WeightsOutput("mxfp8-pruned", LANE),))
app.entrypoint(t.fp8, name="fp8", weights=(WeightsOutput("fp8", LANE),), accelerator=False)
app.entrypoint(t.mxfp8, name="mxfp8", weights=(WeightsOutput("mxfp8", LANE),), accelerator=False)
app.entrypoint(
    t.retable,
    name="retable",
    weights=(WeightsOutput("adaln-pruned", LANE), WeightsOutput("tables", LANE)),
)
app.entrypoint(
    t.select_adaln_weights,
    name="select-adaln-weights",
    weights=(WeightsOutput("model", 1 << 20),),
    accelerator=False,
)
app.entrypoint(
    t.compute_adaln_tables, name="compute-adaln-tables", weights=(WeightsOutput("model", LANE),)
)
app.entrypoint(
    t.apply_adaln,
    name="apply-adaln",
    weights=(
        WeightsOutput("model", 1 << 20),
        WeightsOutput("fl2va-weights", 1 << 20),
        WeightsOutput("ref2va-weights", 1 << 20),
    ),
    accelerator=False,
)
app.entrypoint(
    t.retable_adaln,
    name="retable-adaln",
    weights=(WeightsOutput("model", 1 << 20),),
    accelerator=False,
)
app.entrypoint(t.turbo_lora, name="turbo-lora", weights=(WeightsOutput("pdd8", 1 << 29),))

# minimax-h3
app.entrypoint(h3.fl2va, emits_media=True, accelerator=False)
app.entrypoint(h3.fl2va_turbo, emits_media=True, accelerator=False)
app.entrypoint(h3.ref2va, emits_media=True, accelerator=False)
app.entrypoint(h3.ref2va_turbo, emits_media=True, accelerator=False)
app.entrypoint(h3.long_form, emits_media=True, accelerator=False)
app.entrypoint(h3.long_form_cuts, emits_media=True, accelerator=False)
app.entrypoint(h3.assemble_video, emits_media=True, accelerator=False)
