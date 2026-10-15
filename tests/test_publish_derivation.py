"""cr-067/cr-077: the author derives. Capability is the slot's class annotation and nothing
else; lane words and legacy slot members stay out of the interface.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import App, ConformanceError, Context, Model, Weights, WeightsOutput
from cozy_runtime.author._describe import describe
from cozy_runtime.author._loader import Loader
from cozy_runtime.internal import canonical, package_interface
from cozy_runtime.internal.discovery import Discovered

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"


class Request(msgspec.Struct):
    prompt: str


class Reply(msgspec.Struct):
    text: str


class Component:
    def state_dict(self) -> Mapping[str, object]:
        return {}

    def __call__(self) -> str:
        return "denoised"


class Pipeline:
    def __init__(self) -> None:
        self.components: dict[str, object] = {"transformer": Component()}


class Denoiser(Model[Pipeline]):
    pipeline: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipeline = loader.construct(Pipeline, factory=lambda config: Pipeline())


def _described(app: App) -> dict[str, object]:
    found = Discovered(
        app=app,
        application="profiled_package:app",
        project=PACKAGE,
        module=sys.modules[__name__],
        surfaces=describe(app),
        bindings={},
    )
    return package_interface.build(found)


def test_lane_words_are_not_selection_leaks() -> None:
    """cr-077: a weights output named after the lane it derives ("bf16") describes fine
    beside a default binding whose lane says the same word; an interface string equal to
    the binding's release remains a refused selection leak."""

    def build(output_id: str, release: str) -> dict[str, object]:
        app = App()

        @app.entrypoint(
            weights=(WeightsOutput(output_id, max_new_bytes=1 << 20),), accelerator=False
        )
        def produce(payload: Request, denoiser: Weights[Denoiser], ctx: Context) -> Reply:
            return Reply(text="ok")

        found = Discovered(
            app=app,
            application="profiled_package:app",
            project=PACKAGE,
            module=sys.modules[__name__],
            surfaces=describe(app),
            bindings={"Denoiser": {"model": "acme/sd", "release": release, "lane": "bf16"}},
        )
        return package_interface.build(found)

    body = build("bf16", "1.0.0")
    assert body["entrypoints"][0]["weights_outputs"][0]["output_id"] == "bf16"  # type: ignore[index]

    with pytest.raises(ConformanceError) as leaked_release:
        build("ev-003", "ev-003")
    assert leaked_release.value.code == "fixed_point"


def test_interface_slot_grammar_has_no_legacy_members() -> None:
    """Describe emits no retired slot field; readers tolerate one but not a bad consent."""
    app = App()

    @app.entrypoint
    def generate(payload: Request, denoiser: Denoiser) -> Reply:
        return Reply(text="ok")

    body = json.loads(package_interface.canonical_bytes(_described(app)))
    slot = body["entrypoints"][0]["models"][0]
    assert set(slot) == {"path", "class", "component_use", "encoded_leaves"}
    assert slot["encoded_leaves"] == "refuse"

    def planted(**members: object) -> bytes:
        mutated = json.loads(json.dumps(body))
        mutated["entrypoints"][0]["models"][0].update(members)
        return canonical.write(mutated)

    legacy_members: tuple[dict[str, object], ...] = (
        {"stamps": {}},
        {"stamps": {"task": "fl2va"}},
        {"source_profile": "acme/sd-turbo/1.0.0/bf16"},
    )
    for legacy in legacy_members:
        # Readers tolerate members a newer or older writer added; they carry no authority.
        package_interface.read_bytes(planted(**legacy))
    with pytest.raises(package_interface.StalePackageInterface) as unknown:
        package_interface.read_bytes(planted(encoded_leaves="maybe"))
    assert unknown.value.code == "malformed_package_interface"


def test_model_class_carries_exactly_one_keyword() -> None:
    """`encoded_leaves` is the whole class-keyword plane; a stamp refuses typed (D1)."""
    with pytest.raises(ConformanceError) as stamped:

        class Stamped(Model[Pipeline], task="fl2va"):
            pass

    assert stamped.value.code == "class_keyword"
    assert stamped.value.fields == ("task",)

    with pytest.raises(ConformanceError) as structured:

        class Structured(Model[Pipeline], structure="sdxl"):
            pass

    assert structured.value.code == "class_keyword"

    class Consenting(Model[Pipeline], encoded_leaves="accept"):
        pass

    assert Consenting.__encoded_leaves__ == "accept"
    with pytest.raises(ConformanceError) as closed:

        class Vague(Model[Pipeline], encoded_leaves="maybe"):
            pass

    assert closed.value.code == "encoded_leaves_value"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _SimpleIndex:
    """A real PEP 503 index over the exact wheels the test built, served on loopback."""

    def __init__(self, root: Path, wheels: list[Path]) -> None:
        import http.server
        import threading

        for wheel in wheels:
            name = wheel.name.split("-", 1)[0].replace("_", "-").lower()
            project = root / name
            project.mkdir(parents=True, exist_ok=True)
            shutil.copy(wheel, project / wheel.name)
            (project / "index.html").write_text(
                "<html><body>"
                + f'<a href="{wheel.name}#sha256={_sha(wheel)}">{wheel.name}</a>'
                + "</body></html>"
            )
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
