"""H3 states its packed attention length at request intake (h3a-087), off the request thread,
and that statement never outlives its owner: its request settles it before the first DiT step,
`close` cancels what is queued and waits out what runs, and a short-lived process that exits
with statements in flight exits cleanly instead of finalizing torch beneath one.

The official ref2va blocks over tiny real VAEs (`testdata/h3_intake.py`), on CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

import numpy as np  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference  # noqa: E402
from PIL import Image  # noqa: E402

from cozy_runtime.author import _attention_scope  # noqa: E402
from cozy_runtime.internal import attention, attention_sol  # noqa: E402
from cozy_runtime.models.minimax_h3 import official  # noqa: E402
from cozy_runtime.models.minimax_h3.conditioner import PresentationLength  # noqa: E402
from testdata import h3_intake  # noqa: E402

FIXTURE = Path(__file__).parent / "testdata" / "h3_intake.py"


def _intake_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("minimax-h3-intake")]


@pytest.mark.parametrize("video", [False, True])
def test_h3_states_its_attention_length_at_intake_exactly(video: bool) -> None:
    """The length Sol's kernel is specialized on, read before conditioning, equals the one the
    real conditioning path packs: the text block's own tokenization, the references through
    the real video VAE, the official layout."""
    pipe = h3_intake.pipeline()
    references = [
        pipe.image_reference(Image.new("RGB", (300, 200), (10, 20, 30))),
        pipe.image_reference(Image.new("RGB", (200, 320), (1, 2, 3))),
    ]
    if video:
        references.append(MiniMaxH3VideoReference(frames=np.zeros((40, 96, 160, 3), np.uint8)))
    stated: list[tuple[int, Any]] = []

    def listen(tokens: int, module: Any) -> None:
        stated.append((tokens, module))

    _attention_scope._EXPECTED.append(listen)
    try:
        state = h3_intake.start(pipe, references, (256, 256))
        official._settle_intake(state)
    finally:
        _attention_scope._EXPECTED.remove(listen)
        pipe.close()
    assert state.get(official._INTAKE) is None
    ((tokens, module),) = stated
    assert module is pipe.components["ref2va_dit"]
    # The real path, on the same state: text, media through the VAE, then the layout.
    upstream = pipe._pipes["ref2va"]
    shell = SimpleNamespace(device=torch.device("meta"))
    counted = PresentationLength(pipe.components["text_encoder"])
    pipe._run_with(
        "ref2va",
        official._ScopedPipeline(upstream, shell, overrides={"text_encoder": counted}),
        "text_encoder",
        state,
    )
    pipe.condition_media("ref2va", state)
    pipe._run("ref2va", "denoise.prepare_layout", state)
    assert tokens == state.get("token_tags").numel() > 0


def test_a_full_length_statement_reaches_sol_before_any_forward(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 362-frame clip passes upstream's 15.0 s layout ceiling only under the runtime's
    `_CEILING_S`. Without it (Runtime 0.18.99) every full-length fl2va statement failed, and
    Sol's length compile waited for the first dense step instead of starting at intake."""
    pipe = h3_intake.pipeline()
    dit = pipe.components["fl2va_dit"]
    sol = attention._member(attention.BY_NAME["sol-attn"])
    for _component, _module, processor in attention.sites({"": dit}):
        setattr(processor, "_attention_backend", sol)  # noqa: B010
    submitted: list[tuple[int, Any, int]] = []
    monkeypatch.setattr(attention_sol, "specialize", lambda *call: submitted.append(call))
    try:
        state = pipe.start_fl2va(
            prompt="two swordsmen circle each other in a sunlit courtyard",
            first_frame=None,
            last_frame=None,
            generator=torch.Generator().manual_seed(1),
            steps=pipe._plans["fl2va"].steps[0],
            frames=official.MAX_FRAMES,
        )
        official._settle_intake(state)
    finally:
        pipe.close()
    assert "no intake attention length" not in capsys.readouterr().err
    ((tokens, device, step),) = submitted
    assert step == -1 and device == torch.device("cpu") and tokens > 0


def test_a_short_lived_process_exits_cleanly_with_statements_in_flight() -> None:
    """A CPU proof that starts requests and returns at once (Runtime 0.18.59 aborted such a
    process at exit, 134, while a daemon statement was still inside torch)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    done = subprocess.run(
        [sys.executable, str(FIXTURE), "3"], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("started 3")
    assert "terminate called" not in done.stderr


def test_close_cancels_queued_statements_and_waits_for_the_running_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipe = h3_intake.pipeline()
    references = [pipe.image_reference(Image.new("RGB", (300, 200), (10, 20, 30)))]
    entered, release = threading.Event(), threading.Event()
    heard: list[int] = []

    def listen(tokens: int, _module: Any) -> None:
        heard.append(tokens)
        entered.set()
        release.wait()

    _attention_scope._EXPECTED.append(listen)
    try:
        running = h3_intake.start(pipe, references, (256,)).get(official._INTAKE)
        # Also set if the statement ends without reaching the listener; the assert says so.
        running.add_done_callback(lambda _: entered.set())
        entered.wait()
        assert not running.done() and heard, capsys.readouterr().err
        states = [h3_intake.start(pipe, references, (256,)) for _ in range(2)]
        superseded, queued = (state.get(official._INTAKE) for state in states)
        assert superseded.cancelled() and not queued.done()

        closer = threading.Thread(target=pipe.close)
        closer.start()
        with pytest.raises(CancelledError):
            queued.result()
        assert closer.is_alive() and not running.done(), "close did not wait for the statement"
        release.set()
        closer.join()
        assert running.done() and running.exception() is None
        assert len(heard) == 1 and not _intake_threads()
        # Their requests' DiT steps still settle them: cancelled, not waited on forever.
        for state in states:
            official._settle_intake(state)

        # A closed pipeline still serves; it just states nothing early.
        after = h3_intake.start(pipe, references, (256,))
        assert after.get(official._INTAKE) is None and len(heard) == 1
    finally:
        release.set()
        _attention_scope._EXPECTED.remove(listen)
        pipe.close()


def test_a_failed_statement_is_reported_and_its_request_still_settles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipe = h3_intake.pipeline()
    references = [pipe.image_reference(Image.new("RGB", (300, 200), (10, 20, 30)))]

    def listen(_tokens: int, _module: Any) -> None:
        raise RuntimeError("listener broke")

    _attention_scope._EXPECTED.append(listen)
    try:
        state = h3_intake.start(pipe, references, (256,))
        statement = state.get(official._INTAKE)
        official._settle_intake(state)
    finally:
        _attention_scope._EXPECTED.remove(listen)
        pipe.close()
    assert statement.done() and statement.exception() is None
    err = capsys.readouterr().err
    assert "[minimax-h3] no intake attention length: RuntimeError('listener broke')" in err
