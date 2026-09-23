"""Known preaccept validation is terminal; unexpected exceptions remain opaque."""

from types import SimpleNamespace
from typing import Any, cast

import grpc
import pytest

from cozy_runtime.internal import canonical
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb


class Aborted(Exception):
    pass


def test_noncanonical_execution_payload_is_a_known_refusal() -> None:
    rpc = MachineExecutionRPC(
        cast(
            Worker,
            SimpleNamespace(
                executions=object(),
                authorize_workspace=lambda _: "owner",
            ),
        )
    )
    observed: list[tuple[Any, str]] = []

    def abort(code: Any, detail: str) -> None:
        observed.append((code, detail))
        raise Aborted

    with pytest.raises(Aborted):
        rpc._call(
            pb.Claim(wire_minor=WIRE_MINOR),
            SimpleNamespace(abort=abort),
            lambda: canonical.parse_canonical(b'{ "value": 7 }'),
        )
    assert observed[0][0] == grpc.StatusCode.FAILED_PRECONDITION

    def unexpected() -> None:
        raise RuntimeError("private error contents")

    with pytest.raises(Aborted):
        rpc._call(pb.Claim(wire_minor=WIRE_MINOR), SimpleNamespace(abort=abort), unexpected)
    assert observed[1] == (grpc.StatusCode.UNAVAILABLE, "machine execution operation failed")
