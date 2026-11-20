"""The engine's decoder records a bucket's graph again once a kernel its capture waited on settles
(`bench.engine_recapture`; xs-213, cozy-runtime #2355 and #2370).

A real probe is primed to stay opening until it is released (`kernel_ready.prime`): while it opens, every capture in
the executor is refused (`Building`), as while a lane's decode set builds. The engine's `Decoder` on a small Qwen3.5 of
random weights (`engine_step._model`, plain) decodes one row: eager, its capture refused. The probe then settles, by
landing (`lands`) or by failing (`fails`, as a refused kernel's build does), and the same call again must record the
step and run it captured, with the same tokens; a third call replays it.

Red without #2355, in both modes: a bucket whose capture was refused stays eager for the process's life. Red in
`fails` without #2370: a build that failed before it served never moved the epoch the decoder keys on."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping, Sequence

import torch

type JSON = None | bool | int | float | str | Sequence[JSON] | Mapping[str, JSON]

MODES = ("lands", "fails")
NEW = 32
#: How long the probe may stay opening, unreleased, before it gives up: far past the first call.
HELD_S = 300.0


def measure(mode: str) -> dict[str, JSON]:
    """`mode`'s calls and whether they did as above; raises naming what didn't."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {MODES}")
    from cozy_runtime.engine.decoder import Decoder, Made, Request
    from cozy_runtime.internal import kernel_ready
    from cozy_runtime.internal.bench.engine_step import _model

    model = _model()
    del model.mtp  # plain: one step a token
    decoder = Decoder(model, max_rows=1, max_tokens=512)
    prompt = torch.randint(0, 4096, (37,), generator=torch.Generator().manual_seed(11)).tolist()
    request = Request(prompts=[prompt], max_new=NEW, stops=[])
    name = f"xs213-recapture-probe {mode} {uuid.uuid4().hex} cuda:{torch.cuda.current_device()}"
    release = threading.Event()

    def probe(wait: bool) -> object:
        if not release.wait(HELD_S):
            raise RuntimeError("the probe was never released")
        if mode == "fails":
            raise RuntimeError("the probe's build failed, as a refused kernel's does")
        return object()

    calls: dict[str, JSON] = {}
    made: dict[str, Made] = {}

    def call(label: str) -> None:
        began = time.perf_counter()
        made[label] = decoder(request)
        torch.cuda.synchronize()
        done = made[label]
        calls[label] = {
            "captured": done.captured, "steps": done.steps, "notes": list(done.notes), "epoch": kernel_ready.epoch(),
            "ms_step": round(1e3 * done.decode_s / max(done.steps, 1), 3), "wall_s": round(time.perf_counter() - began, 3),
        }  # fmt: skip

    kernel_ready.prime(name, probe)
    try:
        opening = kernel_ready.opening(name)
        call("building")
    finally:
        release.set()
    deadline = time.monotonic() + 60
    while kernel_ready.opening(name):
        if time.monotonic() > deadline:
            raise RuntimeError(f"{name} still opening a minute after it was released")
        time.sleep(0.05)
    failure = kernel_ready.failed(name)
    call("settled")
    call("warm")
    building, settled, warm = (made[k] for k in ("building", "settled", "warm"))
    facts: dict[str, JSON] = {
        "mode": mode,
        "opening_at_first_call": opening,
        "probe": type(failure).__name__ if failure is not None else "landed",
        "refused_while_building": not building.captured
        and any("Building" in n for n in building.notes),
        "captured_after": settled.captured,
        "replayed_warm": warm.captured,
        "same_tokens": building.tokens == settled.tokens == warm.tokens,
        "launcher": decoder.launcher,
        "calls": calls,
        "cuda": str(torch.version.cuda),
        "card": torch.cuda.get_device_name(),
    }
    if not (opening and facts["refused_while_building"]):
        raise RuntimeError(
            f"the first call's capture wasn't refused while the probe opened: {facts}"
        )
    if not (settled.captured and warm.captured):
        raise RuntimeError(
            f"the bucket stayed eager after the probe {'failed' if failure else 'landed'}: {facts}"
        )
    if not facts["same_tokens"]:
        raise RuntimeError(f"eager and captured calls made different tokens: {facts}")
    return facts
