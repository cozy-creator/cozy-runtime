"""A declared capability must be one a deployment can actually grant.

`describe` is the build gate every package passes before it can be deployed, and it runs
here exactly as the CLI runs it — a real `App`, real registrations, the real conformance
layer. The arm is the whole point of cr-060 item (f): the clock on which a package learns
its capability cannot be honoured is BUILD, not the first request that reaches a GPU.
"""

from __future__ import annotations

import msgspec
import pytest

from cozy_runtime.author import App, Egress, Outputs, Telemetry
from cozy_runtime.author._describe import describe
from cozy_runtime.author._errors import ConformanceError


class Request(msgspec.Struct):
    url: str


class Reply(msgspec.Struct):
    status: int


def test_every_ungrantable_capability_is_a_real_capability() -> None:
    """The two tables cannot drift: a name here that no service backs would refuse nothing."""
    from cozy_runtime.author._services import SERVICES, UNGRANTABLE

    assert set(UNGRANTABLE) <= set(SERVICES)


def test_declaring_egress_refuses_at_build() -> None:
    """`net: Egress` used to pass describe, pass deploy, and raise on the first request."""
    app = App()

    @app.entrypoint
    def fetch(payload: Request, net: Egress) -> Reply:
        return Reply(status=net.fetch(payload.url).status)

    with pytest.raises(ConformanceError) as refusal:
        describe(app)
    assert refusal.value.code == "ungrantable_capability"
    # A REFUSAL NAMES WHERE IT CAME FROM: the surface, the capability, and what must land.
    assert "fetch" in str(refusal.value)
    assert "cr-012" in str(refusal.value)


def test_grantable_capabilities_still_describe() -> None:
    """The gate is a named table, not a mood: everything a deployment CAN grant builds."""
    app = App()

    @app.entrypoint
    def served(payload: Request, out: Outputs, tel: Telemetry) -> Reply:
        return Reply(status=200)

    (surface,) = describe(app)
    assert surface.capabilities == {"save", "telemetry"}
