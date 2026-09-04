"""A checkpoint's prompt conventions reach the sampler, and the caller can see that they did.

Pony-lineage checkpoints want `score_9, score_8_up` and reject `score_1..3`; Illustrious and
NoobAI want `masterpiece, best quality` and treat `score_*` as noise. That is a fact about
what the weights were TRAINED on, so it rides in the cozytensors header beside them
(tfs-059) and the runtime applies it through the ModelDefault overlay that already exists.

Nothing here is doubled. The header bytes are canonical `cozytensors/1` written by TensorFS,
read by the compiled facade through `internal.fill.Checkpoint`; the recipe is built by the
production `Executor._checkpoint_recipes`; the request is resolved by the production
`Executor.prepare_request` and executed by `run_prepared`, so the assertion is on what the
HANDLER received — not on a field having parsed.

Requires a TensorFS facade that speaks the six-element header root. Against an older one the
store read refuses `WRONG_TYPE: header: array arity 6, expected 5`, which is the correct
answer and names the dependency: land and publish tensorfs#101 first, then refresh the lock.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import msgspec
import pytest
import tensorfs

from cozy_runtime.author import (
    App,
    CheckpointConvention,
    Loader,
    Model,
    ModelDefault,
    uses_components,
)
from cozy_runtime.author._describe import assert_declaration_shape
from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._invoke import Invocation, run_prepared, surface_of
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import Checkpoint

#: Canonical `cozytensors/1` written by TensorFS: one inline JCS config, one inline f32
#: tensor, one tokenizer asset, and four metadata rows — the three known prompt keys plus
#: one this runtime has never heard of. Runtime never authors or reinterprets this format.
_HEADER = base64.b64decode(
    "hm1jb3p5dGVuc29ycy8xgYJkdW5ldFgceyJoaWRkZW5fc2l6ZSI6NCwibGF5ZXJzIjoxfYGFc3Rv"
    "a2VuaXplci92b2NhYi50eHRYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63Bmp0ZXh0"
    "L3BsYWlugYJYIJ5ekBAsaZRV6QOf+QMoTgaJOU3TRbsRRWcG8IeYTS63BoGEjAMLAgEABAUIBwYJ"
    "CoCBg2V2YWx1ZYEAgQCBglggR2b5Mbu3DtQx40s05Pn8XdqBdOlvQ6is152Hme/NQ9IZAgSBgmR1"
    "bmV0gYVmd2VpZ2h0AYEBAIGEZXZhbHVlAYEBRAAAAACEgndwcm9tcHQubmVnYXRpdmVfZGVmYXVs"
    "dHhad29yc3QgcXVhbGl0eSwgbG93IHF1YWxpdHksIHNjb3JlXzEsIHNjb3JlXzIsIHNjb3JlXzMs"
    "IGFydGlzdCBuYW1lLCBibHVycnksIGpwZWcgYXJ0aWZhY3RzgnZwcm9tcHQucG9zaXRpdmVfcHJl"
    "Zml4eBttYXN0ZXJwaWVjZSwgYmVzdCBxdWFsaXR5LCCCcXByb21wdC50YWdfc3lzdGVta2lsbHVz"
    "dHJpb3Vzgnd2ZW5kb3IudW5yZWNvZ25pc2VkX2tleXguYSBydW50aW1lIHRoYXQgcHJlZGF0ZXMg"
    "dGhpcyBrZXkgbXVzdCBjYXJyeSBpdA=="
)
_ASSET = b"vocab\n"

_CARD_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts"
)
_CARD_PREFIX = "masterpiece, best quality, "
_PACKAGE_NEGATIVE = "low quality, worst quality, blurry"

app = App()
SEEN: dict[str, str] = {}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    #: Each field names the header key that may fill it. The FIELD still owns its name, its
    #: type, its bounds and its code fallback; the header only supplies a value for the
    #: caller's silence.
    negative_prompt: Annotated[
        ModelDefault[str], CheckpointConvention("prompt.negative_default")
    ] = _PACKAGE_NEGATIVE
    prompt_prefix: Annotated[ModelDefault[str], CheckpointConvention("prompt.positive_prefix")] = ""
    tag_system: Annotated[
        ModelDefault[Literal["pony", "illustrious", "aesthetic", "none"]],
        CheckpointConvention("prompt.tag_system"),
    ] = "none"


class ImageOutput(msgspec.Struct):
    positive: str
    negative: str
    tag_system: str


class ConventionModel(Model[object]):
    @uses_components("unet")
    def sample(self, positive: str, negative: str) -> str:
        return f"{positive}|{negative}"

    def load(self, loader: Loader) -> None:  # pragma: no cover - this test binds no weights
        raise NotImplementedError


@app.entrypoint
def generate(payload: GenerateInput, model: ConventionModel) -> ImageOutput:
    # THE SAMPLER'S SIDE. These two strings are what the model would be handed.
    positive = f"{payload.prompt_prefix}{payload.prompt}"
    SEEN.update(positive=positive, negative=payload.negative_prompt, tag_system=payload.tag_system)
    return ImageOutput(positive, payload.negative_prompt, payload.tag_system)


def _store(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "store"
    store = tensorfs.Store.init(str(root))
    for name, raw in (("header.cbor", _HEADER), ("vocab.txt", _ASSET)):
        path = tmp_path / name
        path.write_bytes(raw)
        store.put_file(str(path), "sha256:" + hashlib.sha256(raw).hexdigest(), len(raw))
    manifest = {
        "entries": [
            {
                "blob": {"length": len(_HEADER), "sha256": hashlib.sha256(_HEADER).hexdigest()},
                "kind": "cozytensors",
                "path": "model.cozytensors",
            }
        ]
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    store.put_manifest(raw, manifest_id, len(raw))
    return root, manifest_id


def _executor(conventions: dict[str, tuple[str, dict[str, str]]]) -> Any:
    executor = Executor.__new__(Executor)
    executor.ready = True
    executor.poisoned = ""
    executor.prepared = {}
    executor.discovered = type("Discovered", (), {"app": app})()
    executor.surface_names = {"generate": surface_of(app.get("generate"))}
    executor.checkpoint_conventions = conventions
    return executor


def _run(executor: Any, wire: dict[str, object], label: str) -> Any:
    SEEN.clear()
    reply = executor.prepare_request(
        {"entrypoint": "generate", "payload": wire, "request_id": label}
    )
    assert reply["ok"], reply
    prepared = executor.prepared.pop(label)
    envelope, outcome, _record = run_prepared(
        prepared,
        Invocation(
            request_id=label,
            spool=Path("/tmp"),
            deadline=time.time() + 60,
            models={"model": ConventionModel.for_test()},
        ),
    )
    assert envelope is not None, outcome
    return envelope


@pytest.fixture
def bound(tmp_path: Path) -> tuple[Any, dict[str, str], str]:
    root, manifest_id = _store(tmp_path)
    metadata = Checkpoint(root, manifest_id).metadata
    return _executor({"model": (manifest_id, metadata)}), metadata, manifest_id


def test_header_metadata_survives_the_store_and_the_facade(
    bound: tuple[Any, dict[str, str], str],
) -> None:
    _, metadata, _manifest = bound

    assert metadata["prompt.negative_default"] == _CARD_NEGATIVE
    assert metadata["prompt.positive_prefix"] == _CARD_PREFIX
    assert metadata["prompt.tag_system"] == "illustrious"
    # THE FORWARD-COMPATIBILITY ARM: a key this runtime cannot spell is carried, not dropped
    # and not refused. Every reader older than a key still serves the checkpoint.
    assert "vendor.unrecognised_key" in metadata


def test_the_convention_reaches_the_sampler(bound: tuple[Any, dict[str, str], str]) -> None:
    executor, _metadata, manifest = bound

    envelope = _run(executor, {"prompt": "a cat"}, "applies")

    assert SEEN["negative"] == _CARD_NEGATIVE
    assert SEEN["positive"] == f"{_CARD_PREFIX}a cat"
    assert SEEN["tag_system"] == "illustrious"
    # VISIBLE: a caller who sees a prompt they never sent can find out which checkpoint sent
    # it, on the same envelope as the result.
    sources = {row.field: row.source for row in envelope.adjustments}
    assert sources["negative_prompt"] == f"checkpoint:{manifest}"
    assert sources["prompt_prefix"] == f"checkpoint:{manifest}"


def test_the_caller_overrides_and_disables_it(bound: tuple[Any, dict[str, str], str]) -> None:
    executor, _metadata, _manifest = bound

    _run(executor, {"prompt": "a cat", "negative_prompt": "mine"}, "override")
    assert SEEN["negative"] == "mine"

    _run(executor, {"prompt": "a cat", "negative_prompt": "", "prompt_prefix": ""}, "disable")
    assert SEEN["negative"] == ""
    assert SEEN["positive"] == "a cat"


def test_a_checkpoint_that_says_nothing_leaves_the_package_default_standing() -> None:
    executor = _executor({})

    assert executor._checkpoint_recipes("generate") == []
    _run(executor, {"prompt": "a cat"}, "silent")
    assert SEEN["negative"] == _PACKAGE_NEGATIVE
    assert SEEN["positive"] == "a cat"


def test_a_header_of_unclaimed_keys_is_carried_and_ignored_never_refused() -> None:
    executor = _executor(
        {"model": ("sha256:" + "1" * 64, {"prompt.sampler_hint": "euler_a", "vendor.x": "y"})}
    )

    assert executor._checkpoint_recipes("generate") == []
    _run(executor, {"prompt": "a cat"}, "unknown-only")
    assert SEEN["negative"] == _PACKAGE_NEGATIVE


def test_a_header_value_outside_the_field_schema_refuses_and_names_the_checkpoint() -> None:
    executor = _executor({"model": ("sha256:" + "2" * 64, {"prompt.tag_system": "danbooru"})})

    reply = executor.prepare_request(
        {"entrypoint": "generate", "payload": {"prompt": "a cat"}, "request_id": "bad"}
    )

    assert not reply["ok"]
    assert reply["code"] == "default_resolution"
    assert "checkpoint:sha256:2222" in reply["detail"]


def test_a_convention_without_model_default_refuses_at_describe() -> None:
    class Bad(msgspec.Struct):
        prompt: str
        negative_prompt: Annotated[str, CheckpointConvention("prompt.negative_default")] = ""

    with pytest.raises(ConformanceError, match="convention_target"):
        assert_declaration_shape(Bad)


def test_a_convention_key_outside_the_header_grammar_refuses() -> None:
    with pytest.raises(ConformanceError, match="convention_key"):
        CheckpointConvention("not a key")
