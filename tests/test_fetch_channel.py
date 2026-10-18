"""The fetch planner over TensorD's fetch channel (`origin/serve.rs`'s frames), against a peer
that speaks that protocol: objects land when the gate opens, a channel the owner ends is asked
again by a call still reading, and a checkpoint outside the grant is nothing to fetch."""

from __future__ import annotations

import socket
import threading

from cozy_runtime.internal.fetch_plan import Fetches
from cozy_runtime.tensord_channels import Channel

MANIFEST = "sha256:" + "ab" * 32
WANTED = [(f"{i:064x}", 1 << 20) for i in range(4)]


class Owner:
    """TensorD's side: one channel per ask; `wait` answers once the gate opens."""

    def __init__(self, whole: frozenset[str] = frozenset()) -> None:
        self.present: set[str] = set()
        self.gate = threading.Event()
        self.asked = threading.Event()
        self.served: list[Channel] = []
        self.whole = whole
        self._lock = threading.Lock()

    def ask(self) -> int:
        mine, theirs = socket.socketpair()
        channel = Channel(mine.detach())
        with self._lock:
            self.served.append(channel)
        threading.Thread(target=self._serve, args=(channel,), daemon=True).start()
        return theirs.detach()

    def shut(self) -> None:
        """The owner ends the call that asked for the newest channel, and the channel with it."""
        with self._lock:
            channel = self.served[-1]
        channel.sock.shutdown(socket.SHUT_RDWR)

    def _serve(self, channel: Channel) -> None:
        try:
            while (request := channel.receive()) is not None:
                assert isinstance(request, dict)
                threading.Thread(target=self._answer, args=(channel, request), daemon=True).start()
        except OSError:
            pass

    def _answer(self, channel: Channel, request: dict[str, object]) -> None:
        ident, op = request["id"], request["op"]
        shas = [str(o["sha256"]) for o in request["objects"]]  # type: ignore[attr-defined]
        try:
            if request["manifest"] in self.whole:
                channel.send({"id": ident, "error": "source_not_allowed", "detail": "held whole"})
                return
            if op == "schedule":
                self.asked.set()
                channel.send({"id": ident, "result": None})
            elif op == "present":
                channel.send({"id": ident, "result": [sha in self.present for sha in shas]})
            elif op == "wait":
                self.gate.wait()
                self.present.update(shas)
                channel.send({"id": ident, "result": None})
        except OSError:
            pass


def test_planned_objects_land_and_a_wait_returns_once_they_are_here() -> None:
    owner = Owner()
    planner = Fetches(owner.ask, lambda *_: None)
    planner.schedule([(MANIFEST, WANTED)])
    assert owner.asked.wait(30) and not planner.present(MANIFEST, WANTED)
    owner.gate.set()
    planner.wait(MANIFEST, WANTED)
    assert planner.present(MANIFEST, WANTED)
    planner.end()


def test_a_channel_its_call_ended_is_asked_again_by_a_call_still_reading() -> None:
    owner = Owner()
    planner = Fetches(owner.ask, lambda *_: None)
    planner.schedule([(MANIFEST, WANTED)])
    assert owner.asked.wait(30)
    asked = len(owner.served)
    owner.shut()
    threading.Timer(1.0, owner.gate.set).start()
    planner.wait(MANIFEST, WANTED)
    assert planner.present(MANIFEST, WANTED)
    assert len(owner.served) == asked + 1, "one fresh channel, asked by the reader"
    planner.end()


def test_two_readers_that_find_the_channel_closed_share_one_fresh_one() -> None:
    owner = Owner()
    planner = Fetches(owner.ask, lambda *_: None)
    planner.schedule([(MANIFEST, WANTED)])
    assert owner.asked.wait(30)
    asked = len(owner.served)
    owner.shut()
    start, landed = threading.Barrier(2), []

    def reader() -> None:
        start.wait()
        planner.wait(MANIFEST, WANTED)
        landed.append(planner.present(MANIFEST, WANTED))

    readers = [threading.Thread(target=reader) for _ in range(2)]
    for thread in readers:
        thread.start()
    threading.Timer(1.0, owner.gate.set).start()
    for thread in readers:
        thread.join(60)
    assert landed == [True, True]
    assert len(owner.served) == asked + 1
    planner.end()


def test_a_checkpoint_outside_the_grant_is_nothing_to_fetch() -> None:
    owner = Owner(whole=frozenset({MANIFEST}))
    planner = Fetches(owner.ask, lambda *_: None)
    planner.schedule([(MANIFEST, WANTED)])
    assert planner.present(MANIFEST, WANTED)
    planner.wait(MANIFEST, WANTED)  # held whole here: nothing to wait for
    planner.end()
