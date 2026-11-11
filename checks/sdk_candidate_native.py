"""Installed SDK payload and explicit model-selection checks for a candidate wheel."""

from __future__ import annotations

import argparse
from array import array
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import sysconfig
import tempfile
from typing import TypedDict
import zipfile

import tensorfs
from tensorfs import _ext


class NativeMember(TypedDict):
    sha256: str
    bytes: int
    installed_path: str
    installed_sha256: str


def native_payload(wheel: Path) -> dict[str, object]:
    installed_distribution = distribution("tensorfs")
    members: dict[str, NativeMember] = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not (re.search(r"\.(?:so(?:\.[^/]*)?|pyd|dll|dylib)$", name) or name.endswith("/scripts/tfs")):
                continue
            data = archive.read(name)
            expected = hashlib.sha256(data).hexdigest()
            if ".data/" in name:
                scheme, relative = name.split(".data/", 1)[1].split("/", 1)
                root = Path(sys.prefix) if scheme == "data" else Path(sysconfig.get_path(scheme))
                installed = root / relative
            else:
                installed = Path(str(installed_distribution.locate_file(name)))
            assert installed.is_file(), (name, str(installed))
            actual = hashlib.sha256(installed.read_bytes()).hexdigest()
            assert actual == expected, (name, actual, expected)
            members[name] = {"sha256": expected, "bytes": len(data),
                             "installed_path": str(installed.resolve()), "installed_sha256": actual}
    extension = Path(_ext.__file__).resolve()
    matched = [entry for entry in members.values() if Path(entry["installed_path"]) == extension]
    assert len(matched) == 1
    extension_hash = hashlib.sha256(extension.read_bytes()).hexdigest()
    assert matched[0]["sha256"] == extension_hash
    cli = Path(sysconfig.get_path("scripts")) / "tfs"
    cli_version = subprocess.check_output([str(cli), "version"], text=True).strip()
    assert cli_version.split()[:2] == ["tfs", "0.24.3"], cli_version
    return {"wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "extension_sha256": extension_hash, "native_members": members, "cli_version": cli_version}


def write_weights(path: Path, keys: list[str], value: float, *, lora: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    payload = bytearray()
    for key in keys:
        shape = ([2, 4] if ".lora_A" in key else [4, 2]) if lora else [1, 1]
        dtype, code = ("F16", "e") if lora else ("F32", "f")
        begin = len(payload)
        payload.extend(struct.pack("<" + code, value) * (shape[0] * shape[1]))
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def selections() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pipeline = root / "pipeline"
        write_weights(pipeline / "transformer/model.safetensors", ["weight"], 7)
        write_weights(pipeline / "vae/model.safetensors", ["weight"], 9)
        index = {"_class_name": "FixturePipeline", "transformer": ["fixture", "Transformer"],
                 "vae": ["fixture", "VAE"]}
        for name in ["model_index.json", "modular_model_index.json"]:
            (pipeline / name).write_text(json.dumps(index))
        (pipeline / "FL2VA").mkdir()
        (pipeline / "FL2VA/model_index.json").write_text(json.dumps({"transformer": ["fixture", "OtherView"]}))
        checkpoint = tensorfs.convert_to_cozytensors(str(pipeline), converter="as-is", cache_dir=root / "pipeline-store")
        assert set(checkpoint.components) == {"transformer", "vae"}
        observed = {name: list(array("f", component.tensors.read("weight", decode=True).tobytes()))
                    for name, component in checkpoint.components.items()}
        assert observed == {"transformer": [7], "vae": [9]}
        pipeline_id = checkpoint.id
        checkpoint.close()

        collection = root / "independent-loras"
        keys = ["layer.lora_A.weight", "layer.lora_B.weight"]
        for number in range(12):
            write_weights(collection / f"adapter-{number}.safetensors", keys, number + 1, lora=True)
        try:
            tensorfs.convert_to_cozytensors(str(collection), converter="as-is", cache_dir=root / "rejected-store")
        except tensorfs.Error as error:
            assert "multiple model selections" in str(error)
        else:
            raise AssertionError("independent LoRAs were merged")

        aliases: list[dict[str, object]] = []
        identities: set[str] = set()
        store = root / "twelve-selections"
        for number in range(12):
            checkpoint = tensorfs.convert_to_cozytensors(str(collection / f"adapter-{number}.safetensors"),
                                                        converter="as-is", cache_dir=store)
            assert len(checkpoint.components) == 1
            model_id = f"local/adapter-{number}"
            with tensorfs.Writer(base=checkpoint, model_id=model_id, lane="f16", cache_dir=store) as writer:
                pass
            named = writer.commit()
            assert named.model_id == model_id
            named.close()
            checkpoint.close()
            reopened = tensorfs.open(model_id, cache_dir=store, local_files_only=True)
            assert reopened.model_id == model_id and set(reopened.components) == {"model"}
            tensors = reopened.components["model"].tensors.read(decode=True)
            assert set(tensors) == set(keys)
            values_by_key: dict[str, list[float]] = {}
            for key, tensor in tensors.items():
                assert tensor.dtype == "F16"
                assert tensor.shape == ((2, 4) if ".lora_A" in key else (4, 2))
                values = [float(value) for value, in struct.iter_unpack("<e", tensor.tobytes())]
                assert values == [float(number + 1)] * 8
                values_by_key[key] = values
            identities.add(reopened.id)
            aliases.append({"model": model_id, "checkpoint": reopened.id, "values": values_by_key})
            reopened.close()
        assert len(identities) == 12
        return {"alternative_pipeline_indexes": True, "pipeline_checkpoint": pipeline_id,
                "pipeline_values": observed, "independent_loras_refused": 12,
                "explicit_local_listings": aliases, "automatic_bulk_fanout": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert tensorfs.__version__ == "0.24.3"
    result = {"tensorfs": tensorfs.__version__, "payload": native_payload(args.wheel), "selections": selections()}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("SDK native payload and explicit model-selection checks passed")


if __name__ == "__main__":
    main()
