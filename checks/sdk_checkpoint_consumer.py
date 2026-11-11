"""Installed old/new Runtime consumers of the same exact TensorFS candidate."""

from __future__ import annotations

import argparse
from array import array
from collections.abc import Callable
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
from typing import cast
import zipfile

import cozy_runtime
from cozy_runtime import weights_store
import tensorfs
from tensorfs import _ext
from tensorfs.executor import CheckpointItem


def consume(wheel: Path) -> dict[str, object]:
    """Exercise identity, immutable data, replay and the supported executor frame."""
    runtime_version = version("cozy-runtime")
    assert runtime_version in ("0.45.0", "0.45.1")
    assert tensorfs.__version__ == "0.24.3"
    assert Path(weights_store.__file__).is_relative_to(Path(sys.prefix))
    with zipfile.ZipFile(wheel) as archive:
        extensions = [name for name in archive.namelist() if name == "tensorfs/_ext.abi3.so"]
        assert len(extensions) == 1
        expected_extension = hashlib.sha256(archive.read(extensions[0])).hexdigest()
    actual_extension = hashlib.sha256(Path(_ext.__file__).read_bytes()).hexdigest()
    assert actual_extension == expected_extension

    with tempfile.TemporaryDirectory() as directory:
        store = Path(directory)
        derivation = weights_store.Derivation(None, order=[("model", "weight")])
        transaction = weights_store.replay_key("xs170/consumer", "source", "weights", derivation.work(32))
        if runtime_version == "0.45.0":
            legacy_id = cast(Callable[[str], str], getattr(weights_store, "repository"))
            model_id = legacy_id(transaction)
        else:
            model_id = weights_store.derived_model_id(transaction)
        with weights_store.Output(store, "weights", transaction, derivation, 32) as output:
            output.put("model", "weight", "F32", [2], array("f", [2.0, 4.0]))
            checkpoint = output.commit()
        assert checkpoint.model_id == model_id
        values = array("f", checkpoint.components["model"].tensors.read("weight", decode=True).tobytes()).tolist()
        assert values == [2.0, 4.0]
        checkpoint_id, revision = checkpoint.id, checkpoint.tree
        assert revision is not None
        checkpoint.close()
        with weights_store.Output(store, "weights", transaction, derivation, 32) as replay:
            assert replay.done is not None
            assert replay.commit().id == checkpoint_id
            replay.done.close()
        if runtime_version == "0.45.0":
            item = CheckpointItem(output="weights", repo_id=model_id, revision=revision)
        else:
            item = CheckpointItem(output="weights", model_id=model_id, revision=revision)
        frame = cast(dict[str, object], json.loads(item.json()))
        assert frame["repo_id"] == model_id and "model_id" not in frame
        return {
            "runtime": runtime_version,
            "runtime_source": cozy_runtime.__file__,
            "tensorfs": tensorfs.__version__,
            "native_extension_sha256": actual_extension,
            "checkpoint": checkpoint_id,
            "model_id": model_id,
            "values": values,
            "replay": True,
            "wire_frame": frame,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = consume(args.wheel)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
