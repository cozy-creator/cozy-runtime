import json
import time
from typing import ClassVar

import tensorfs
from cp_producer_lifecycle import app
from cp_producer_lifecycle.model import Component, Pipeline, read_report, report_tensor

from cozy_runtime.author import Invocation, ModelArtifact, attempt, describe
from cozy_runtime.author._loader import census
from cozy_runtime.internal.fill import Checkpoint
from native_weights import NativeExecution


def test_real_native_mint_has_exact_constructor_census(tmp_path):
    store = tensorfs.Store.init(tmp_path / "store")
    host = NativeExecution(store, tmp_path, "mint", {}, {"model": 1 << 20})
    describe(app)
    result, outcome, _ = attempt(
        app.get("mint"),
        {"seed": 581204},
        Invocation(
            "mint",
            tmp_path / "spool",
            time.monotonic() + 60,
            tensorfs_output=host.client.open_output,
            tensorfs_adopt=host.client.adopt_model,
        ),
    )
    assert outcome.terminal == "succeeded", outcome
    artifact = result.result
    assert isinstance(artifact, ModelArtifact)
    checkpoint = Checkpoint.from_store(store, artifact.manifest.digest)
    expected = Component().state_dict()
    rows = list(checkpoint.rows("dit"))
    assert len(rows) == len(expected) == 6
    assert sum(v.numel() * v.element_size() for v in expected.values()) == 262656
    walked = census(Pipeline())
    assert {d.key for d in walked.destinations} == {"dit." + key for key in expected}
    assert {r.key for r in rows} == {"dit." + key for key in expected}
    for row in rows:
        key = row.key.removeprefix("dit.")
        assert row.shape == tuple(expected[key].shape)
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "manifest": artifact.manifest.digest,
                "bytes": 262656,
                "census": {k: list(v.shape) for k, v in expected.items()},
                "passed": True,
            },
            indent=2,
        )
    )


def test_actual_diffusers_root_output_hook_accepts_encoded_report():
    from diffusers.hooks.context_parallel import apply_context_parallel
    from diffusers.models._modeling_parallel import ContextParallelConfig

    class Report(Component):
        _cp_plan: ClassVar[dict] = {"": {}}

        def forward(self, step, seed, first):
            return report_tensor({"step": step, "seed": seed, "first": first})

    model = Report()
    apply_context_parallel(model, ContextParallelConfig(ulysses_degree=2), model._cp_plan)
    assert read_report(model(4, 581204, True)) == {"step": 4, "seed": 581204, "first": True}
