"""A construction config is never a path: an upstream `save_pretrained` config loads with its
`_name_or_path` provenance dropped, a carrier-named flag that names no source loads as written,
and every other location key still refuses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozy_runtime.author import Config
from cozy_runtime.author._errors import ConformanceError

#: BiRefNet's upstream `config.json` (ZhengPeng7/BiRefNet), byte for byte.
BIREFNET = json.loads((Path(__file__).parent / "testdata" / "birefnet_config.json").read_bytes())
#: Anima's T5 tokenizer config as fidika/anima@1.0.0 publishes it, byte for byte: Transformers 5's
#: `save_pretrained` writes `local_files_only: false` and `is_local: true` into it (tfs-132).
T5_TOKENIZER = json.loads(
    (Path(__file__).parent / "testdata" / "anima_t5_tokenizer_config.json").read_bytes()
)


def test_an_upstream_config_loads_without_its_provenance() -> None:
    assert BIREFNET["_name_or_path"] == "ZhengPeng7/BiRefNet"
    config = Config({"model": BIREFNET})
    seen = config.section("model").mapping()
    assert "_name_or_path" not in seen
    assert seen == {k: v for k, v in BIREFNET.items() if k != "_name_or_path"}
    assert "_name_or_path" not in Config(BIREFNET).mapping()


def test_a_tokenizer_config_with_saved_loader_flags_loads_as_written() -> None:
    assert T5_TOKENIZER["local_files_only"] is False
    config = Config({"scheduler": {"shift": 3.0}, "t5_tokenizer": T5_TOKENIZER})
    assert config.mapping()["t5_tokenizer"] == T5_TOKENIZER
    flags = {"cache_dir": None, "weights": 0.5}
    assert Config(flags).mapping() == flags


@pytest.mark.parametrize(
    "entry",
    [
        {"bb_path": "x"},
        {"weights": "x"},
        {"local_files_only": "x"},
        {"cache_dir": ["x"]},
        {"backbone": "/models/swin.pth"},
        {"bb": "hf://a/b"},
    ],
)
def test_every_other_location_still_refuses(entry: dict[str, object]) -> None:
    with pytest.raises(ConformanceError) as refused:
        Config({"model": {**BIREFNET, **entry}})
    assert refused.value.code == "config_path"
