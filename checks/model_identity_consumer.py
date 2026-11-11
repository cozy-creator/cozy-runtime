"""Installed Runtime output/replay and its published TensorFS protocol boundary."""

from __future__ import annotations

import argparse
from array import array
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile

import tensorfs
from tensorfs.executor import Answer, CheckpointItem, Request

from cozy_runtime import weights_store
from cozy_runtime.internal.executor import _show_checkpoint


def consume() -> dict[str, object]:
    assert Path(weights_store.__file__).is_relative_to(Path(sys.prefix))
    frames: list[dict[str, object]] = []

    def exchange(request: Request) -> Answer:
        assert isinstance(request, CheckpointItem)
        frame: dict[str, object] = json.loads(request.json())
        frames.append(frame)
        return Answer(ok=True)

    with tempfile.TemporaryDirectory() as directory:
        store = Path(directory)
        outputs = weights_store.Outputs(
            store, request_id="identity-consumer", actor="fixture", code="source",
            bounds={"weights": 32},
            show=lambda slot, model, checkpoint: _show_checkpoint(exchange, slot, model, checkpoint),
        )
        derivation = weights_store.Derivation(None, order=[("model", "weight")])
        with outputs.open("weights", derivation) as output:
            output.put("model", "weight", "F32", [2], array("f", [2.0, 4.0]))
            artifact = outputs.adopt(output)
            checkpoint = output.commit()
            model_id = weights_store.derived_model_id(output.transaction)
            # The published SDK exposes this model identity as repo_id.
            assert checkpoint.repo_id == model_id
            values = array("f", checkpoint.components["model"].tensors.read("weight", decode=True).tobytes()).tolist()
            assert values == [2.0, 4.0]
            assert artifact.checkpoint == checkpoint.id
            assert checkpoint.tree is not None
            expected = json.loads(CheckpointItem(output="weights", repo_id=model_id, revision=checkpoint.tree).json())
            assert frames == [expected]
            assert frames[0]["repo_id"] == model_id and "model_id" not in frames[0]
            checkpoint_id = checkpoint.id
            checkpoint.close()
        with outputs.open("weights", derivation) as replay:
            assert replay.done is not None
            assert replay.commit().id == checkpoint_id
            replay.done.close()

    return {
        "runtime": version("cozy-runtime"), "tensorfs": version("tensorfs"),
        "checkpoint": checkpoint_id, "model_id": model_id,
        "values": values, "replay": True, "frame": frames[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = consume()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
