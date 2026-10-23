"""A Transformers model stored under its older names binds and fills as Transformers 5 loads it
(cr-213), and a construction may leave stored tensors unbuilt under that renaming (cr-214).

A tiny real Mistral3 (FLUX.2 [dev]'s text encoder family) is stored as Transformers 4 named it
(`language_model.model.*`, `language_model.lm_head.*`), as the published checkpoint is. Its rows
bind to Transformers 5's module names, and a construction that builds only the first layers fills
from the store through varena on the CPU and computes the whole model's hidden states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tensorfs.varena.torch")

import tensorfs  # noqa: E402
from tensorfs.varena.torch import Memory  # noqa: E402

from cozy_runtime.author._loader import Artifact, Config, Loader  # noqa: E402
from cozy_runtime.internal import canonical, model_config  # noqa: E402
from cozy_runtime.internal.fill import (  # noqa: E402,F401
    Checkpoint,
    tensor_schema_of,
    tensorfs_requirement_dtype,
)
from cozy_runtime.internal.weights import Backend  # noqa: E402
from store_fixtures import home  # noqa: E402

LAYERS, KEPT = 4, 2
#: Transformers 5's module names -> the names Transformers 4 stored them under.
STORED = (
    ("model.language_model.", "language_model.model."),
    ("lm_head.", "language_model.lm_head."),
    ("model.vision_tower.", "vision_tower."),
    ("model.multi_modal_projector.", "multi_modal_projector."),
)


def _config() -> Any:
    return transformers.AutoConfig.for_model(
        "mistral3",
        text_config=dict(
            model_type="mistral", vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=LAYERS, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        ),
        vision_config=dict(
            model_type="pixtral", hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, head_dim=8, image_size=32, patch_size=16,
        ),
        architectures=["Mistral3ForConditionalGeneration"],
    )  # fmt: skip


def _stored(name: str) -> str:
    for now, then in STORED:
        if name.startswith(now):
            return then + name[len(now) :]
    raise AssertionError(name)


class Graph:
    """[dev]'s text encoder as a package builds it: the whole model, or only its first layers."""

    def __init__(self, config: Config, *, kept: int | None) -> None:
        text = dict(config.mapping())  # one component: its own config is the document
        encoder = transformers.AutoModelForImageTextToText.from_config(
            transformers.AutoConfig.for_model(**text)
        )
        if kept is not None:
            language = encoder.model.language_model
            language.layers = torch.nn.ModuleList(list(language.layers)[:kept])
            language.norm = torch.nn.Identity()
            encoder.lm_head = torch.nn.Identity()
        self.components = {"text_encoder": encoder.eval()}


class _Owner:
    def _cozy_require_scope(self, what: str) -> None:
        return None


@pytest.fixture
def stored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Checkpoint, Any]:
    torch.manual_seed(0)
    reference = transformers.AutoModelForImageTextToText.from_config(_config()).eval()
    writer = tensorfs.Writer(cache_dir=home(tmp_path, monkeypatch))
    held = writer.component(
        "text_encoder",
        cls=("transformers", "", "Mistral3ForConditionalGeneration"),
        config=json.loads(reference.config.to_json_string()),
    )
    for key, value in reference.state_dict().items():
        held.put(_stored(key), value.numpy(), dtype="F32", shape=list(value.shape))
    return Checkpoint(writer.commit()), reference


def test_older_names_bind_to_the_module_transformers_builds(stored: Any) -> None:
    ck, reference = stored
    assert any(name.startswith("language_model.model.") for name in ck.tensors("text_encoder"))
    bound = sorted(row.key for row in ck.rows("text_encoder"))
    assert bound == sorted(f"text_encoder.{k}" for k in reference.state_dict())


@pytest.mark.parametrize("kept", [None, KEPT])
def test_a_construction_fills_what_it_builds_and_skips_the_rest(
    stored: Any, kept: int | None
) -> None:
    ck, reference = stored
    rows = ck.rows("text_encoder")
    backend = Backend.for_script(
        {"text_encoder": ck},
        rows,
        torch=torch,
        memory=Memory.open(torch.device("cpu")),
        device=torch.device("cpu"),
        construction="cr-213",
        release="cr-213",
    )
    backend.expect({"text_encoder": tuple(r.key for r in rows)})
    document = canonical.parse_canonical(model_config.construction_config(ck.configs()))
    artifact = Artifact(ck.id, tensor_schema_of(rows), Config(document, "model.config"))
    loader = Loader(artifact, owner=_Owner(), backend=backend)
    try:
        graph = loader.construct(Graph, factory=lambda config: Graph(config, kept=kept))
        encoder = graph.components["text_encoder"]
        ids = torch.tensor([[1, 5, 9, 3, 7]])
        with torch.inference_mode(), backend.memory.request("cr-213"):
            hidden = encoder(input_ids=ids, output_hidden_states=True).hidden_states
            want = reference(input_ids=ids, output_hidden_states=True).hidden_states
        for depth in range(1, (kept or LAYERS) + 1):
            torch.testing.assert_close(hidden[depth], want[depth], rtol=0, atol=0)
    finally:
        backend.close()
