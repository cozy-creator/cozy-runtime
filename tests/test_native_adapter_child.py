"""The standalone adapter helper keeps its native JSON/receipt contract after worker removal."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import msgspec

from cozy_runtime.internal.adapter_view_child import (
    AdapterViewInput,
    AdapterViewReady,
    PrepareAdapterView,
)
from cozy_runtime.internal.native_child import MAX_BYTES, Launch, Refused, Sample
from test_lora_composition import fixture


def child(command: bytes) -> subprocess.CompletedProcess[bytes]:
    script = """
import importlib.abc
import runpy
import sys
class NoWorker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('cozy_runtime.internal.worker', 'cozy_runtime.protocol', 'cozy.worker.v1')):
            raise AssertionError('standalone helper imported retired worker: ' + fullname)
sys.meta_path.insert(0, NoWorker())
runpy.run_module('cozy_runtime.internal.adapter_view_child', run_name='__main__', alter_sys=True)
"""
    return subprocess.run([sys.executable, "-c", script], input=command, capture_output=True)


def test_adapter_native_process_receipt_and_replay_without_worker_imports(tmp_path: Path) -> None:
    store, base, first, second = fixture(tmp_path)
    transaction = "sha256:" + "88" * 32
    request = Launch(
        store=str(store.root),
        parent_pid=os.getpid(),
        step=PrepareAdapterView(
            identity=transaction,
            base=base.manifest,
            adapters=(
                AdapterViewInput(first.manifest, "transformer", "adapter", "0.5"),
                AdapterViewInput(second.manifest, "transformer", "adapter", "-0.25"),
            ),
        ),
    )
    command = msgspec.json.encode(request)
    assert msgspec.json.decode(command)["step"]["step"] == "prepare_adapter_view"
    observed: list[AdapterViewReady] = []
    for _ in range(2):
        response = child(command)
        assert response.returncode == 0, response.stderr.decode()
        frames = [
            msgspec.json.decode(line, type=Sample | AdapterViewReady | Refused)
            for line in response.stdout.splitlines()
        ]
        assert isinstance(frames[-1], AdapterViewReady), response.stderr.decode()
        observed.append(frames[-1])
    assert observed[0] == observed[1]
    actual = store.derived_lookup(transaction)["receipt"]["manifest"]
    assert observed[0].manifest.digest == "sha256:" + actual["sha256"]
    assert observed[0].manifest.length == actual["length"]


def test_malformed_adapter_launch_is_a_typed_native_refusal() -> None:
    response = child(b'{"store":')
    assert response.returncode == 0
    assert msgspec.json.decode(response.stdout, type=Refused).code == "native_step_malformed"


def test_oversized_valid_adapter_launch_refuses_before_preparation(tmp_path: Path) -> None:
    store, base, first, _ = fixture(tmp_path)
    identity = "sha256:" + "99" * 32
    command = msgspec.json.encode(
        Launch(
            store=str(store.root),
            parent_pid=os.getpid(),
            step=PrepareAdapterView(
                identity=identity,
                base=base.manifest,
                adapters=(AdapterViewInput(first.manifest, "transformer", "adapter", "0.5"),),
            ),
        )
    )
    # JSON permits trailing whitespace: decoding alone would accept this oversized input.
    response = child(command + b" " * (MAX_BYTES + 1 - len(command)))
    assert response.returncode == 0, response.stderr.decode()
    assert msgspec.json.decode(response.stdout, type=Refused).code == "native_step_bound"
    assert store.derived_lookup(identity) is None
