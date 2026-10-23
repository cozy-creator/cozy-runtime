"""A tied model fills from a checkpoint that stores its tied tensor once (cr-218).

A tiny real Qwen3 causal LM with `tie_word_embeddings` is stored as `save_pretrained` stores it:
`model.embed_tokens.weight` once, no `lm_head.weight`. The construction's census names the tied
tensor once, the fit admits the checkpoint, and varena installs the one placeholder at both names,
so the logits equal the reference model's."""

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

from cozy_runtime.author._loader import Artifact, Config, Loader, census  # noqa: E402
from cozy_runtime.internal import canonical, model_config  # noqa: E402
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of  # noqa: E402
from cozy_runtime.internal.weights import Backend  # noqa: E402
from store_fixtures import home  # noqa: E402


def _config() -> Any:
    return transformers.AutoConfig.for_model(
        "qwen3", vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        architectures=["Qwen3ForCausalLM"],
    )  # fmt: skip


class Graph:
    def __init__(self, config: Config) -> None:
        model = transformers.AutoModelForCausalLM.from_config(
            transformers.AutoConfig.for_model(**dict(config.mapping()))
        )
        self.components = {"model": model.eval()}


class _Owner:
    def _cozy_require_scope(self, what: str) -> None:
        return None


@pytest.fixture
def stored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Checkpoint, Any]:
    torch.manual_seed(0)
    reference = transformers.AutoModelForCausalLM.from_config(_config()).eval()
    assert reference.lm_head.weight is reference.model.embed_tokens.weight
    writer = tensorfs.Writer(cache_dir=home(tmp_path, monkeypatch))
    held = writer.component(
        "model",
        cls=("transformers", "", "Qwen3ForCausalLM"),
        config=json.loads(reference.config.to_json_string()),
    )
    for key, value in reference.state_dict().items():
        if key != "lm_head.weight":  # tied: stored once, as save_pretrained stores it
            held.put(key, value.numpy(), dtype="F32", shape=list(value.shape))
    return Checkpoint(writer.commit()), reference


def test_the_census_names_a_tied_tensor_once() -> None:
    graph = Graph.__new__(Graph)
    graph.components = {"model": transformers.AutoModelForCausalLM.from_config(_config())}
    keys = [d.key for d in census(graph).destinations]
    assert "model.model.embed_tokens.weight" in keys and "model.lm_head.weight" not in keys
    assert len(keys) == len(set(keys))


def test_a_tied_model_fills_from_its_one_stored_tensor(stored: Any) -> None:
    ck, reference = stored
    rows = ck.rows("model")
    backend = Backend.for_script(
        {"model": ck},
        rows,
        torch=torch,
        memory=Memory.open(torch.device("cpu")),
        device=torch.device("cpu"),
        construction="cr-218",
        release="cr-218",
    )
    backend.expect({"model": tuple(r.key for r in rows)})
    document = canonical.parse_canonical(model_config.construction_config(ck.configs()))
    artifact = Artifact(ck.id, tensor_schema_of(rows), Config(document, "model.config"))
    loader = Loader(artifact, owner=_Owner(), backend=backend)
    try:
        graph = loader.construct(Graph, factory=Graph)
        model = graph.components["model"]
        ids = torch.tensor([[1, 5, 9, 3, 7]])
        with torch.inference_mode(), backend.memory.request("cr-218"):
            logits = model(input_ids=ids).logits
            want = reference(input_ids=ids).logits
        torch.testing.assert_close(logits, want, rtol=0, atol=0)
    finally:
        backend.close()
