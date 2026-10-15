"""A publish to an output the result does not declare warns and drops that product (run 5482).

The name is checked in-process against the result's asset fields, the rule the machine
applies, so a CPU test of the package sees it; a machine refusal of the name drops it too.
"""

from __future__ import annotations

import time
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import App, ImageAsset, Invocation, Outputs, attempt, fakes
from cozy_runtime.author._executor_requests import Publish, Published
from cozy_runtime.author._services import Attempt, ImageFrame

PIXELS = ImageFrame(2, 2, bytes(12))


class Request(msgspec.Struct):
    pass


class Detail(msgspec.Struct):
    mask: ImageAsset


class Result(msgspec.Struct):
    image: ImageAsset
    previews: list[ImageAsset]
    detail: Detail
    caption: str = ""


app = App()


@app.entrypoint(accelerator=False)
def paint(payload: Request, out: Outputs) -> Result:
    image, mask = out.save_image(PIXELS), out.save_image(PIXELS)
    out.publish("previews", image)
    out.publish("preview", image)
    out.publish("detail.mask", mask)
    return Result(image, [], Detail(mask))


class Machine:
    """The machine's side of `publish`: it refuses `preview` as TensorD refuses an
    undeclared name, and shows everything else."""

    def __init__(self) -> None:
        self.shown: list[str] = []

    def __call__(self, request: Publish) -> Published:
        if request.output == "preview":
            return Published(ok=False, code="output_undeclared", detail="not a result field")
        self.shown.append(request.output)
        return Published(ok=True, sequence=len(self.shown))


def warnings_logged(record: Attempt) -> list[object]:
    return [o.fields["output"] for o in record.ring if o.kind == "log" and o.value == "warn"]


def test_an_undeclared_publish_warns_drops_and_the_run_succeeds(tmp_path: Path) -> None:
    machine = Machine()
    with pytest.warns(UserWarning, match="undeclared output 'preview' dropped"):
        result, outcome, record = attempt(
            app.get("paint"),
            {},
            Invocation("paint", tmp_path, time.monotonic() + 30, publish=machine),
        )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    assert machine.shown == ["previews", "detail.mask"]
    assert warnings_logged(record) == ["preview"]
    assert app.get("paint").surface.outputs == {"image", "previews", "detail.mask"}


def test_a_machine_refusing_the_name_drops_it_too(tmp_path: Path) -> None:
    machine = Machine()
    record = Attempt("refused", tmp_path, settle_at_save=True, publish=machine)
    out = fakes.fake_outputs(record)
    with pytest.warns(UserWarning, match="undeclared output 'preview' dropped"):
        out.publish("preview", out.save_image(PIXELS))
    assert machine.shown == []
    assert warnings_logged(record) == ["preview"]
