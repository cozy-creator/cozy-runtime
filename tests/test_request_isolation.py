"""A request's refusal, cancel or bad pin stays with it; healthy generations keep serving."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.internal import attention, attention_ulysses, executor, package_interface
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import _SEALED, Executor
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, PrepareRequest, Start
from cozy_runtime.internal.seam import Channel

MODULE = "request_isolation_fixture"
SOURCE = """
import os
import shutil

import msgspec
from cozy_runtime.author import App, Context, InvalidRequest, Telemetry

app = App()


class Request(msgspec.Struct):
    mode: str = "ok"
    spool: str = ""


class Result(msgspec.Struct):
    value: int


def contract_failure() -> None:
    raise RuntimeError("contract fixture was not installed")


@app.entrypoint
def generate(ctx: Context, payload: Request, tel: Telemetry) -> Result:
    if payload.mode == "refuse":
        raise InvalidRequest("the handler refused this request", code="mode_refused")
    if payload.mode == "cancel":
        ctx.raise_if_cancelled()
    if payload.mode == "fail":
        raise RuntimeError("handler state is now unknown")
    if payload.mode == "contract":
        contract_failure()
    if payload.mode == "env":
        os.environ["OMP_NUM_THREADS"] = "7"
    if payload.mode == "tail":
        with tel.stage("export"):
            shutil.rmtree(payload.spool)  # the Runtime's own tail then cannot write the result
    return Result(7)
"""


@pytest.fixture
def host(tmp_path: Path) -> Iterator[tuple[Executor, dict[str, Any]]]:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject = "{MODULE}:app"\n')
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(
        package_interface.canonical_bytes(package_interface.build(discover(project)))
    )
    binding = {
        "application": f"{MODULE}:app",
        "package_interface": str(interface),
    }
    devices = _SEALED.get("CUDA_VISIBLE_DEVICES", "")
    left, right = socket.socketpair()
    with left, right:
        executor = Executor(Channel(left), tmp_path / "root")
        (tmp_path / "root").mkdir()
        started = executor.start(msgspec.convert({**binding, "devices": devices}, Start))
        assert started["ok"], started
        yield executor, {"devices": devices, "binding": binding, "budgets": {}}
    sys.modules.pop(MODULE, None)


def _serve(executor: Executor, request_id: str, mode: str) -> dict[str, Any]:
    prepared = executor.serve_request(
        PrepareRequest(request_id=request_id, entrypoint="generate", payload={"mode": mode})
    )
    assert prepared["ok"], prepared
    spool = executor.root / request_id
    spool.mkdir()
    if mode == "cancel":
        (executor.root / "executor.cancel").write_text(request_id)
    return executor.serve_invoke(
        Invoke(request_id=request_id, entrypoint="generate", spool=str(spool), deadline_s=30)
    )


def test_handler_refusals_and_cancels_keep_the_generation_ready(
    host: tuple[Executor, dict[str, Any]],
) -> None:
    executor, load = host
    assert executor.load(msgspec.convert({**load, "construction": "a"}, Load))["ok"]
    assert executor.activate(Activate(construction="a"))["ok"]

    refused = _serve(executor, "refused", "refuse")
    assert (refused["outcome"]["terminal"], refused["outcome"]["code"]) == (
        "refused",
        "mode_refused",
    )
    canceled = _serve(executor, "canceled", "cancel")
    assert canceled["outcome"]["terminal"] == "canceled"
    assert not executor.poisoned and not refused["poisoned"] and not canceled["poisoned"]
    assert _serve(executor, "served", "ok")["outcome"]["terminal"] == "succeeded"

    failed = _serve(executor, "failed", "fail")
    assert failed["outcome"]["terminal"] == "failed" and failed["poisoned"]


def test_an_attempt_whose_tail_raises_names_its_cause_and_stage(
    host: tuple[Executor, dict[str, Any]],
) -> None:
    """A tail that raises (a faulted device's first sync) once answered with no outcome and no
    poison: runs 5313/5318/5319 ended `executor_poisoned: ; ... (poisoned: )`."""
    executor, load = host
    assert executor.load(msgspec.convert({**load, "construction": "a"}, Load))["ok"]
    assert executor.activate(Activate(construction="a"))["ok"]
    spool = executor.root / "tail"
    spool.mkdir()
    assert executor.serve_request(
        PrepareRequest(
            request_id="tail", entrypoint="generate", payload={"mode": "tail", "spool": str(spool)}
        )
    )["ok"]
    reply = executor.handle(
        Invoke(request_id="tail", entrypoint="generate", spool=str(spool), deadline_s=30)
    )
    outcome = reply["outcome"]
    assert (outcome["terminal"], outcome["code"]) == ("failed", "unhandled_exception"), reply
    assert outcome["message"].startswith("FileNotFoundError: ")
    assert outcome["message"].endswith(" in export")
    assert reply["poisoned"].startswith("unhandled_exception in export: FileNotFoundError")
    assert reply["quiescent"] is False and executor.poisoned == reply["poisoned"]
    after = executor.serve_request(
        PrepareRequest(request_id="after", entrypoint="generate", payload={})
    )
    assert (after["code"], after["detail"]) == ("poisoned_generation", reply["poisoned"])


def test_a_refused_load_leaves_loaded_constructions_serving(
    host: tuple[Executor, dict[str, Any]],
) -> None:
    executor, load = host
    assert executor.load(msgspec.convert({**load, "construction": "a"}, Load))["ok"]
    model = {**load["binding"], "model_class": f"{MODULE}:Model"}
    refused = executor.load(msgspec.convert({**load, "construction": "b", "binding": model}, Load))
    assert refused["code"] == "model_sources_ungranted", refused
    assert not executor.poisoned and "b" not in executor.constructions
    assert executor.activate(Activate(construction="a"))["ok"]
    assert _serve(executor, "after", "ok")["outcome"]["terminal"] == "succeeded"


def test_a_pin_no_parallel_group_can_run_refuses_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("diffusers.models.attention_dispatch")
    # Every Runtime kernel is head-local today; a cross-head one must still refuse a group.
    local = attention_ulysses.Local(lambda *_, **__: None, lambda *_: "", head_local=False)
    kitchen = attention.BY_NAME["kitchen-int8"]
    cross = replace(kitchen, name="cross-head", backend="_cozy_test_cross", local=local)
    monkeypatch.setitem(attention.BY_NAME, "cross-head", cross)
    with pytest.raises(attention.AttentionRefusal, match="context parallelism") as refused:
        attention.admit("model/dit=cross-head", 4)
    assert refused.value.code == "attention_kernel_unsupported"
    attention.admit("dit=cross-head", 1)
    with pytest.raises(attention.AttentionRefusal) as unknown:
        attention.admit("dit=not-a-kernel", 1)
    assert unknown.value.code == "attention_kernel_unknown"
    for name in set(attention.BY_NAME) - {"cross-head"}:
        attention.admit(f"dit={name}", 4)


def test_package_environment_changes_are_restored_not_refused(
    host: tuple[Executor, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, load = host
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(executor, "_SEALED", seal_snapshot())
    assert runner.load(msgspec.convert({**load, "construction": "a"}, Load))["ok"]
    assert runner.activate(Activate(construction="a"))["ok"]
    served = _serve(runner, "env", "env")
    assert served["outcome"]["terminal"] == "succeeded", served
    assert served["env_intact"] and served["env_changed"] == ["OMP_NUM_THREADS"], served
    assert os.environ["OMP_NUM_THREADS"] == "1" and not served["poisoned"]
