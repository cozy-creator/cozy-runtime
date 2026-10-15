"""Runtime directs TensorFS's fetcher (one-request-kind §2.6): what it asks for, when, and that
an ended attempt's interest goes with it.

Real: a loopback hub serving a real checkpoint, a destination Store holding only its
structure (as TensorD's prepare leaves it), and each fetch channel served by TensorFS's own
owner half (`serve_fetch`) on that Store. Every byte goes through the fetcher; the hub counts
the GETs."""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import threading
from pathlib import Path

import msgspec
import pytest
import tensorfs
from tensorfs import Store, _ext
from tensorfs.derived import SourceCapability

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.internal import fetch_plan, package_interface, plane
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Start
from cozy_runtime.internal.seam import Channel
from fetch_hub import Hub, fetches, objects_of, served

MODULE = "fetch_plan_fixture"
SOURCE = """
import time
time.sleep(1.0)  # a heavy package import
import msgspec
from cozy_runtime.author import App
FINISHED = time.monotonic()
app = App()
class Ping(msgspec.Struct):
    n: int
@app.entrypoint(accelerator=False)
def ping(payload: Ping) -> Ping:
    return payload
"""


def owner(channel: Channel, hub: Hub, store: Store) -> None:
    """The machine's half of the seam: fetch channels and the checkpoint's header."""
    while (frame := channel.recv()) is not None:
        if frame.get("event") != "request":
            continue
        answer = {"event": "answer", "seq": frame["seq"], "ok": True}
        if frame["kind"] == "fetch_channel":
            channel.send(answer | {"descriptor": True})
            with socket.socket(fileno=hub.channel(store)) as handed:
                channel.send_descriptor(handed)
        elif frame["kind"] == "model_source" and not frame.get("name"):
            header = store.manifest(frame["manifest"])["header"]
            assert header is not None
            memfd = plane.sealed_memfd("model-source", header)
            facts = {"sha256": hashlib.sha256(header).hexdigest(), "length": len(header)}
            channel.send(answer | {"held": True, "descriptor": True} | facts)
            channel.send_memfd(memfd)
            os.close(memfd)
        else:
            channel.send(answer | {"ok": False, "code": "unknown_durable_request"})


def test_start_schedules_its_constructions_before_the_package_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / f"{MODULE}.py").write_text(SOURCE)
    (project / "package.toml").write_text(f'[application]\nobject="{MODULE}:app"\n')
    found = discover(project)
    sys.modules.pop(MODULE, None)  # the Start imports it again, timed
    monkeypatch.syspath_prepend(str(project))
    interface = tmp_path / "package-interface.json"
    interface.write_bytes(package_interface.canonical_bytes(package_interface.build(found)))
    ours, theirs = socket.socketpair()
    with served(tmp_path) as (hub, store):
        machine = threading.Thread(target=owner, args=(Channel(theirs), hub, store), daemon=True)
        machine.start()
        try:
            executor = Executor(Channel(ours), tmp_path / "root")
            row = {"path": "ping.models.model", "manifest": hub.manifest}
            start = {
                "application": found.application,
                "package_interface": str(interface),
                "constructions": [row | {"components": ["encoder", "denoiser", "decoder"]}],
            }
            assert executor.start(msgspec.convert(start, Start))["ok"]
            planner = executor.fetch
            assert planner is not None
            # Smallest first: no class is known yet, and a first stage's encoders are small.
            assert [sorted(refs) for _, refs in planner.groups] == [
                objects_of(store, hub.manifest, name) for name in ("encoder", "decoder", "denoiser")
            ]
            for manifest, refs in planner.groups:
                planner.wait(manifest, refs)
            first = min(hub.first[sha] for _, refs in planner.groups for sha, _ in refs)
            assert first < sys.modules[MODULE].FINISHED, "the download waited for the import"
            for _, refs in planner.groups:
                assert hub.count(refs) == [1] * len(refs)
            spare = objects_of(store, hub.manifest, "spare")
            assert hub.count(spare) == [0] * len(spare), "an undeclared component was fetched"
            planner.end()
        finally:
            ours.close()
            machine.join(timeout=30)
            theirs.close()
            sys.modules.pop(MODULE, None)


def test_a_refused_fetch_ends_the_wait_as_model_fetch_failed(tmp_path: Path) -> None:
    with served(tmp_path) as (hub, store):
        wanted = objects_of(store, hub.manifest, "encoder")
        hub.wrong.add(wanted[0][0])
        planner = fetches(hub, store)
        planner.schedule([(hub.manifest, wanted)])
        with pytest.raises(RuntimeFailure) as refused:
            planner.wait(hub.manifest, wanted[:1])
        assert refused.value.code == "model_fetch_failed", refused.value
        assert wanted[0][0] in str(refused.value) and hub.manifest[:23] in str(refused.value)
        planner.end()


def test_an_ended_attempt_drops_what_it_wanted(tmp_path: Path) -> None:
    """The owner shuts a call's channels when it ends or is canceled, and Runtime closes its
    own at the attempt's end: nothing it scheduled lands after."""
    with served(tmp_path) as (hub, store):
        wanted = objects_of(store, hub.manifest, "denoiser")
        hub.held = {sha for sha, _ in wanted}
        hub.gate.clear()
        planner = fetches(hub, store)
        planner.schedule([(hub.manifest, wanted)])
        assert hub.asked.wait(30)
        planner.end()
        for thread in hub.served:
            thread.join(timeout=30)
            assert not thread.is_alive(), "the channel outlived its attempt"
        hub.gate.set()
        assert not any(store.contains(sha) for sha, _ in wanted)


def test_a_raw_source_wants_its_declared_components_and_the_rest_of_what_it_reads(
    tmp_path: Path,
) -> None:
    """`Weights[M]`: its declared components are wanted when it opens; a read of another
    component wants that component from the tensor read on, and nothing before it."""
    with served(tmp_path) as (hub, store):
        producer = Store.open(tmp_path / "producer")
        length = len(hub.bodies[hub.manifest.removeprefix("sha256:")])
        owner, theirs = socket.socketpair()
        threading.Thread(
            target=_ext.serve_source,
            args=(producer, owner.detach(), hub.manifest, length),
            daemon=True,
        ).start()
        planner = fetches(hub, store)
        header = store.manifest(hub.manifest)["header"]
        assert header is not None
        opened = SourceCapability.from_fd(hub.manifest, length, theirs.detach())
        source = fetch_plan.source(planner, opened, header, ["encoder"])
        encoder = objects_of(store, hub.manifest, "encoder")
        planner.wait(hub.manifest, encoder)
        assert hub.count(encoder) == [1] * len(encoder)

        keys = list(tensorfs.parse_header(header)["components"]["decoder"])
        source.read_part_into("decoder", keys[1], "value", 0, memoryview(bytearray(16)))
        rest = fetch_plan.component_objects(header, "decoder", keys, 1)
        before = set(fetch_plan.component_objects(header, "decoder", keys)) - set(rest)
        assert before, "the first tensor shares every object with the rest"
        planner.wait(hub.manifest, rest)
        assert hub.count(rest) == [1] * len(rest)
        assert hub.count(sorted(before)) == [0] * len(before)
        source.close()
        planner.end()
