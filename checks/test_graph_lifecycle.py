"""CPU regressions for graph retirement; no CUDA context or tensorfs native build required."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

from cozy_runtime import graphs


class Module:
    pass


class GraphLifecycle(unittest.TestCase):
    def test_stale_binding_returns_to_eager_after_dropping_its_pool(self) -> None:
        torch = NS(
            is_grad_enabled=lambda: False,
            utils=NS(
                _pytree=NS(tree_flatten=lambda _: ([], None)),
                _python_dispatch=NS(_get_current_dispatch_mode=lambda: None),
            ),
            distributed=NS(is_initialized=lambda: False),
            is_autocast_enabled=lambda _: False,
            get_autocast_dtype=lambda _: "float",
        )
        module = Module()
        recorder: Any = graphs._Recorder()
        recorder.domain = NS(using=nullcontext)
        recorder.binding = NS(valid=lambda: False, close=lambda: None)
        recorder.pool = NS(close=lambda: None)
        recorder.eager = NS(run=lambda _, run: run())
        recorder.recordings[((), (), (False, "float"))] = NS(idle=lambda: True, close=lambda: None)
        placement = NS(attempts=1, retain=lambda *_: self.fail("retained a stale recording"))
        with (
            patch.object(graphs, "_signature", return_value=()),
            patch.object(graphs, "_unplain", return_value=""),
            patch.dict(graphs._RECORDERS, {module: recorder}),
            patch.dict(graphs._REGIONS, {module: (lambda: module, "")}),
        ):
            result = graphs._region_call(
                torch, module, lambda *_: "eager", (), {}, (placement, "m")
            )
        self.assertEqual(result, "eager")
        self.assertIsNone(recorder.pool)
        self.assertFalse(recorder.recordings)

    def test_retirement_drains_replay_and_output_readers_before_releasing_pins(self) -> None:
        order: list[str] = []
        entry: Any = graphs._Recording.__new__(graphs._Recording)
        entry.fenced = False
        entry.stream = NS(synchronize=lambda: order.append("replay"))
        entry.done = NS(synchronize=lambda: order.append("completion"))
        class Reader:
            def synchronize(self) -> None:
                order.append("reader")
        entry.readers = {Reader(): None}
        entry.close = lambda: order.append("executable")
        recorder: Any = graphs._Recorder()
        recorder.domain = NS(using=nullcontext)
        recorder.recordings[()] = entry
        recorder.binding = NS(close=lambda: order.append("addresses"))
        recorder.pool = NS(close=lambda: order.append("pins"))
        with recorder.lock:  # Binding.valid can discover the change while already locked.
            recorder.retire()
        self.assertEqual(
            order, ["replay", "completion", "reader", "executable", "addresses", "pins"]
        )


if __name__ == "__main__":
    unittest.main()
