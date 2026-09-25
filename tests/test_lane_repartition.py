"""Real worker dispatch after topology changes; CPU fixtures do not claim GPU compute."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from cozy_runtime.internal.local import LocalRecordOwner
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _slice, _workspace, needs_executor


@needs_executor
def test_split_lanes_execute_after_worker_started_on_a_group() -> None:
    workers: list[Worker] = []

    def worker_hook(worker: Worker) -> None:
        # Start with one group lane. No GPU model is fabricated: the test concerns
        # replacement queue consumers, and the later real invocation is weightless.
        group = worker.lanes.fuse((0, 1))
        group.placements.add("retired-group")
        workers.append(worker)

    def owner_hook(owner: LocalRecordOwner) -> None:
        desired = owner._desired_state
        observe = owner._on_observed_state

        def first() -> pb.DesiredWorkerState:
            # Claim/reconcile happens after Worker.run starts its queue consumers.
            worker = workers[0]
            worker._release_lane("retired-group")
            state = desired()
            assert owner.placement is not None
            state.placement_set.device_pins.append(
                pb.PlacementDevicePin(
                    placement_id=owner.placement.placement_id, device_ordinals=[1]
                )
            )
            return state

        def report(report: pb.ObservedWorkerState) -> None:
            observe(report)
            worker = workers[0]
            if (
                any(a.state == "queued" for a in worker.engine.history.values())
                and "lane-1" not in worker.lane_threads
            ):
                owner.error = "a real attempt queued on lane-1 but that lane has no consumer"
                owner.ended = "worker_error"
                owner.done.set()
                owner.host.close()

        owner._desired_state = first  # type: ignore[method-assign]
        owner._on_observed_state = report  # type: ignore[method-assign]

    with _workspace() as root:
        outcome, seals = _slice(
            root, devices="0,1", kind="entrypoint", hook=worker_hook, owner_hook=owner_hook
        )
    assert outcome.status == "OUTCOME_STATUS_SUCCEEDED", outcome.error
    assert outcome.result == {"message": "polo"}
    assert seals == ["1"]


@needs_executor
def test_fused_lane_dispatches_a_real_executor_after_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cozy_runtime.internal.worker import lanes

    workers: list[Worker] = []
    consumers: list[tuple[threading.Thread, ...]] = []

    def form_test_group(self: lanes.LaneSet, placement_id: str, **kwargs: Any) -> lanes.DeviceLane:
        # Only group formation is supplied by this CPU fixture. The author is
        # weightless; this is queue/seal proof, not model CP-degree admission.
        group = self.fuse(tuple(kwargs["pin"]))
        group.placements.add(placement_id)
        return group

    def owner_hook(owner: LocalRecordOwner) -> None:
        desired = owner._desired_state
        observe = owner._on_observed_state
        send = owner._send
        repeated = False
        first_consumers: list[tuple[threading.Thread, ...]] = []

        def replay_offer(frame: pb.RecordOwnerFrame) -> None:
            send(frame)
            if frame.HasField("attempt_offer"):
                send(frame)

        def first() -> pb.DesiredWorkerState:
            state = desired()
            assert owner.placement is not None
            state.placement_set.device_pins.append(
                pb.PlacementDevicePin(
                    placement_id=owner.placement.placement_id, device_ordinals=[0, 1]
                )
            )
            return state

        def report(report: pb.ObservedWorkerState) -> None:
            nonlocal repeated
            worker = workers[0]
            consumers[:] = list(getattr(worker, "_device_consumers", {}).values())
            if any(p.serving == pb.SERVING_STATE_DISPATCHABLE for p in report.placements):
                if not repeated:
                    first_consumers[:] = consumers
                    repeated = True
                    successor = first()
                    successor.revision = 2
                    owner._send(pb.RecordOwnerFrame(desired_state=successor))
                    return
                if report.accepted_desired_state_revision != 2:
                    return
                assert consumers == first_consumers
            observe(report)
            if (
                any(a.state == "queued" for a in worker.engine.history.values())
                and "lane-0+1" not in worker.lane_threads
            ):
                owner.error = "a real attempt queued on the group without a consumer"
                owner.ended = "worker_error"
                owner.done.set()
                owner.host.close()

        owner._desired_state = first  # type: ignore[method-assign]
        owner._send = replay_offer  # type: ignore[method-assign]
        owner._on_observed_state = report  # type: ignore[method-assign]

    monkeypatch.setattr(lanes.LaneSet, "assign", form_test_group)
    with _workspace() as root:
        outcome, seals = _slice(
            root, devices="0,1", kind="entrypoint", hook=workers.append, owner_hook=owner_hook
        )
    assert outcome.status == "OUTCOME_STATUS_SUCCEEDED", outcome.error
    assert outcome.result == {"message": "polo"}
    assert seals == ["0,1"]
    assert workers[0].engine.executed == 1
    assert consumers and all(not thread.is_alive() for pair in consumers for thread in pair)
    assert not workers[0]._device_consumers
    assert not workers[0]._retired_consumers


@needs_executor
@pytest.mark.parametrize("supersede_drain", [False, True])
@pytest.mark.parametrize("managed", [False, True])
def test_disjoint_attempts_drain_to_group_and_split_without_preemption(
    monkeypatch: pytest.MonkeyPatch,
    supersede_drain: bool,
    managed: bool,
) -> None:
    import base64
    import dataclasses
    import shutil

    if managed and supersede_drain:
        pytest.skip("managed admission does not start a drain while attempts are active")
    import test_device_lanes
    from cozy_runtime.internal.worker import lanes
    from cozy_runtime.protocol import documents

    assign = lanes.LaneSet.assign
    workers: list[Worker] = []
    completed: list[str] = []
    overlap: list[set[str]] = []

    def assign_fixture(self: lanes.LaneSet, placement_id: str, **kwargs: Any) -> lanes.DeviceLane:
        pin = kwargs.get("pin")
        if pin and len(pin) > 1:
            # The CPU fixture supplies model-independent group formation only.
            group = self.fuse(tuple(pin))
            group.placements.add(placement_id)
            return group
        return assign(self, placement_id, **kwargs)

    def owner_hook(owner: LocalRecordOwner) -> None:
        desired = owner._desired_state
        send = owner._send
        original_request = owner.request
        initial: pb.DesiredWorkerState | None = None
        selected = "plc-local"
        offered: dict[str, str] = {}
        stage = 0
        group_revision = 4 if supersede_drain else 3 if managed else 2

        def release(phase: str, *seals: str) -> None:
            for seal in seals:
                (barrier / f"{phase}-release-{seal}").touch()

        def state(revision: int, group: bool) -> pb.DesiredWorkerState:
            assert initial is not None and owner.placement is not None
            result = pb.DesiredWorkerState()
            result.CopyFrom(initial)
            result.revision = revision
            placements = [owner.placement]
            if not group:
                sibling = pb.Placement()
                sibling.CopyFrom(owner.placement)
                sibling.placement_id = "hosted"
                placements.append(sibling)
            raw, digest = documents.identity(pb.PlacementSet(placements=placements))
            result.placement_set.placement_set_canonical_bytes = raw
            result.placement_set.placement_set_digest = digest
            result.placement_set.ClearField("device_pins")
            for placement in placements:
                pins = [0, 1] if group else [0] if placement.placement_id == "plc-local" else [1]
                result.placement_set.device_pins.append(
                    pb.PlacementDevicePin(placement_id=placement.placement_id, device_ordinals=pins)
                )
            return result

        def single(selected_state: pb.DesiredWorkerState, identity: str) -> pb.DesiredWorkerState:
            body = documents.read(
                selected_state.placement_set.placement_set_canonical_bytes, pb.PlacementSet
            )
            chosen = [row for row in body["placements"] if row["placement_id"] == identity]
            from cozy_runtime import canonical_json

            raw = canonical_json.encode(
                {"format": "cozy.worker.v1.PlacementSet/1", "placements": chosen}
            )
            digest = documents.digest_of(raw)
            selected_state.placement_set.placement_set_canonical_bytes = raw
            selected_state.placement_set.placement_set_digest = digest
            pins = [
                pin
                for pin in selected_state.placement_set.device_pins
                if pin.placement_id == identity
            ]
            selected_state.placement_set.ClearField("device_pins")
            selected_state.placement_set.device_pins.extend(pins)
            return selected_state

        def prepare(selected_state: pb.DesiredWorkerState) -> bool:
            with workers[0].control_lock:
                return workers[0]._prepare_execution(
                    {
                        "revisions": {},
                        "state": base64.b64encode(selected_state.SerializeToString()).decode(),
                    },
                    "managed-proof",
                )

        def first() -> pb.DesiredWorkerState:
            nonlocal initial
            initial = desired()
            selected_state = state(1, False)
            return single(selected_state, "plc-local") if managed else selected_state

        def routed(frame: pb.RecordOwnerFrame) -> None:
            if frame.HasField("attempt_offer"):
                frame.attempt_offer.placement_id = selected
            send(frame)

        def dispatch(name: str, placement: str) -> None:
            nonlocal selected
            selected = placement
            offered[name] = placement
            owner.request = dataclasses.replace(original_request, request_id=name)
            owner._dispatch()

        def accepted(accepted: pb.AttemptAccepted) -> None:
            assert offered[accepted.request_id] == accepted.placement_id
            assert accepted.invocation_spec_digest == owner.spec_digest
            owner.accepted = accepted

        def terminal(outcome: pb.AttemptOutcome) -> None:
            completed.append(outcome.request_id)
            owner.terminal = outcome
            body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            if body["status"] != pb.OUTCOME_STATUS_SUCCEEDED:
                owner.error = "real attempt failed during lane repartition"
                owner.done.set()
                owner.host.close()
                return
            owner._send(
                pb.RecordOwnerFrame(
                    outcome_ack=owner._env(
                        pb.AttemptOutcomeAck(
                            request_id=outcome.request_id,
                            attempt_ordinal=outcome.attempt_ordinal,
                            invocation_spec_digest=outcome.invocation_spec_digest,
                            outcome_id=outcome.outcome_id,
                            outcome_digest=outcome.outcome_digest,
                        )
                    )
                )
            )

        def report(report: pb.ObservedWorkerState) -> None:
            nonlocal stage
            owner.reports.append(report)
            owner.admission_epoch = report.admission_epoch
            worker = workers[0]
            if worker.latched:
                owner.error = worker.latched
                owner.done.set()
                owner.host.close()
                return
            ready = {
                p.placement_id
                for p in report.placements
                if p.serving == pb.SERVING_STATE_DISPATCHABLE
            }
            if managed and stage == 0 and ready == {"plc-local"}:
                stage = 10
                dispatch("before-0", "plc-local")
            elif managed and stage == 10 and (barrier / "before-0").exists():
                stage = 11
                prepare(single(state(2, False), "hosted"))
            elif managed and stage == 11 and ready == {"plc-local", "hosted"}:
                stage = 1
                dispatch("before-1", "hosted")
            elif stage == 0 and ready == {"plc-local", "hosted"}:
                stage = 1
                dispatch("before-0", "plc-local")
                dispatch("before-1", "hosted")
            elif stage in (1, 6):
                phase = "before" if stage == 1 else "after"
                entries = [barrier / f"{phase}-{ordinal}" for ordinal in (0, 1)]
                if not all(p.exists() for p in entries):
                    return
                # Each actual child entered its function and is still blocked on
                # the test release file: two different PIDs overlap on disjoint seals.
                pids = {p.read_text() for p in entries}
                assert len(pids) == 2
                assert sum(a.state == "running" for a in worker.engine.history.values()) == 2
                overlap.append(pids)
                if stage == 1:
                    if managed:
                        assert not prepare(state(group_revision, True))
                        assert workers[0].accepted.accepted_desired_state_revision == 2
                        stage = 12
                        release("before", "0", "1")
                    else:
                        stage = 2
                        owner._send(pb.RecordOwnerFrame(desired_state=state(2, True)))
                else:
                    stage = 7
                    release("after", "0", "1")
            elif managed and stage == 12 and len(completed) == 2:
                # All singleton children are terminal/acked before group convergence.
                stage = 3
                prepare(state(group_revision, True))
            elif (
                stage == 2
                and worker.placement is not None
                and (worker.placement.serving == pb.SERVING_STATE_DRAINING)
            ):
                assert completed == [], "the wider request preempted an active invocation"
                stage = 20 if supersede_drain else 3
                release("before", *("0",) if supersede_drain else ("0", "1"))
            elif (
                stage == 20
                and worker.hosted["hosted"].placement.serving == pb.SERVING_STATE_DRAINING
            ):
                assert completed == ["before-0"]
                stage = 21
                owner._send(pb.RecordOwnerFrame(desired_state=state(3, False)))
                release("before", "1")
            elif (
                stage == 21
                and worker.accepted.converged_revision == 3
                and ready == {"plc-local", "hosted"}
                and len(completed) == 2
            ):
                assert worker.supervision.current is not None
                assert worker.hosted["hosted"].supervision.current is not None
                assert {
                    str(worker.supervision.current.pid),
                    str(worker.hosted["hosted"].supervision.current.pid),
                } == overlap[0]
                stage = 3
                owner._send(pb.RecordOwnerFrame(desired_state=state(group_revision, True)))
            elif stage == 3 and worker.accepted.converged_revision == group_revision:
                stage = 4
                (barrier / "round").write_text("group")
                release("group", "0,1")
                dispatch("group", "plc-local")
            elif stage == 4 and "group" in completed:
                stage = 5
                owner._send(pb.RecordOwnerFrame(desired_state=state(group_revision + 1, False)))
            elif (
                stage == 5
                and worker.accepted.converged_revision == group_revision + 1
                and ready == {"plc-local", "hosted"}
            ):
                stage = 6
                (barrier / "round").write_text("after")
                dispatch("after-0", "plc-local")
                dispatch("after-1", "hosted")
            elif stage == 7 and len(completed) == 5:
                owner.ended = "terminal"
                owner.done.set()
                owner.host.close()

        owner._desired_state = first  # type: ignore[method-assign]
        owner._send = routed  # type: ignore[method-assign]
        owner._on_observed_state = report  # type: ignore[method-assign]
        owner._on_outcome = terminal  # type: ignore[method-assign]
        owner._on_accepted = accepted  # type: ignore[method-assign]

    monkeypatch.setattr(lanes.LaneSet, "assign", assign_fixture)
    with _workspace() as root:
        barrier = root / "barrier"
        barrier.mkdir()
        (barrier / "round").write_text("before")
        package = root / "package"
        shutil.copytree(test_device_lanes.PACKAGE, package)
        (package / "marco_polo_package").rename(package / "lane_repartition_fixture")
        for file in (package / "package.toml", package / "pyproject.toml"):
            file.write_text(
                file.read_text().replace("marco_polo_package", "lane_repartition_fixture")
            )
        module = package / "lane_repartition_fixture/__init__.py"
        source = module.read_text().replace(
            '    return PoloResponse(message="polo")',
            "    import os, time\n    from pathlib import Path\n"
            f"    root = Path({str(barrier)!r})\n"
            "    phase = (root / 'round').read_text()\n"
            "    seal = os.environ['CUDA_VISIBLE_DEVICES']\n"
            "    (root / (phase + '-' + seal)).write_text(str(os.getpid()))\n"
            "    while not (root / (phase + '-release-' + seal)).exists():\n"
            "        time.sleep(0.005)\n"
            '    return PoloResponse(message="polo")',
        )
        module.write_text(source)
        monkeypatch.setattr(test_device_lanes, "PACKAGE", package)
        outcome, seals = _slice(
            root, devices="0,1", kind="entrypoint", hook=workers.append, owner_hook=owner_hook
        )
    assert outcome.status == "OUTCOME_STATUS_SUCCEEDED", outcome.error
    assert not outcome.error, outcome.error
    assert len(completed) == len(set(completed)) == 5
    assert len(overlap) == 2
    assert seals == ["0", "0,1", "0"]
    assert not workers[0]._device_consumers and not workers[0]._retired_consumers
