"""A function's model slots load in any declaration order: the machine sends them as the author
declared them (`generate_turbo(model, lightning)`), and the executor orders them itself."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import msgspec

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Load, Start
from cozy_runtime.internal.seam import Channel

MODULE = "model_slot_order_fixture"
SOURCE = """
from typing import Any
import msgspec
from cozy_runtime.author import App, Loader, Model
app = App()
class Base(Model[Any]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("an on-demand slot is not built at load")
class Lightning(Model[Any]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("an on-demand slot is not built at load")
class Request(msgspec.Struct):
    n: int
class Result(msgspec.Struct):
    n: int
@app.entrypoint(on_demand=("model", "lightning"))
def generate_turbo(payload: Request, model: Base, lightning: Lightning) -> Result:
    return Result(payload.n)
"""


def test_slots_declared_out_of_order_load_in_one_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject="{MODULE}:app"\n')
    try:
        found = discover(project)
        interface = tmp_path / "package-interface.json"
        document = package_interface.build(found)
        interface.write_bytes(package_interface.canonical_bytes(document))
        (entry,) = document["entrypoints"]
        declared = [slot["path"] for slot in entry["models"]]
        assert declared == ["generate_turbo.models.model", "generate_turbo.models.lightning"]
        release = {"application": found.application, "package_interface": str(interface)}
        rows = [
            {
                "binding": {
                    **release,
                    "model_class": slot["class"],
                    "model_binding_path": slot["path"],
                    "model_parameter_name": slot["path"].rsplit(".", 1)[1],
                }
            }
            for slot in entry["models"]
        ]
        left, right = socket.socketpair()
        with left, right:
            executor = Executor(Channel(left), tmp_path)
            assert executor.start(msgspec.convert({**release, "devices": ""}, Start))["ok"]
            load = {"devices": "", "construction": "turbo", "models": rows}
            reply = executor.load(msgspec.convert(load, Load))
            assert reply["ok"], reply
            assert sorted(executor.constructions["turbo"].models) == ["lightning", "model"]
            # The same slot twice is still refused.
            twice = {**load, "construction": "twice", "models": [rows[0], rows[0]]}
            refused = Executor(Channel(right), tmp_path)
            assert refused.start(msgspec.convert({**release, "devices": ""}, Start))["ok"]
            reply = refused.load(msgspec.convert(twice, Load))
            assert reply["code"] == "model_binding_set_invalid", reply
    finally:
        sys.modules.pop(MODULE, None)
