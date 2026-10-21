"""A package that cannot construct from an as-is (unnormalized) checkpoint refuses
`checkpoint_unnormalized` and names the fix.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from cozy_runtime.author import Artifact, CheckpointUnnormalized, Config, Loader, ModelFitRefused
from cozy_runtime.author._loader import Backend, Census, Fit

REMEDY = "stored as-is: re-import it once a TensorFS converter recognizes its layout"


def test_a_construction_failing_on_an_as_is_checkpoint_names_the_fix() -> None:
    artifact = Artifact("as-is", {}, Config({}), unnormalized=REMEDY)

    class Sdxl:  # reads the component configs an as-is single file does not carry
        def __init__(self, config: Config) -> None:
            self.components = {"unet": config.mapping()["unet"]}

    loader = Loader(artifact, owner=SimpleNamespace())
    with pytest.raises(CheckpointUnnormalized, match="re-import") as failed:
        loader.construct(Sdxl, factory=Sdxl)
    assert isinstance(failed.value.__cause__, KeyError)

    class Verdict:  # the backend's fit over keys the construction does not find
        bound = None

        def ahead(self) -> None:
            pass

        def fit(self, walked: Census, *, encoded_leaves: str) -> Fit:
            return {"ok": False, "code": "missing_tensor", "detail": "unet.conv_in.weight"}

    class Empty:
        def __init__(self, config: Config) -> None:
            self.components: dict[str, object] = {}

    loader = Loader(artifact, owner=SimpleNamespace(), backend=cast(Backend, Verdict()))
    with pytest.raises(ModelFitRefused, match="re-import") as refused:
        loader.construct(Empty, factory=Empty)
    assert refused.value.code == "checkpoint_unnormalized"
    assert refused.value.fit["code"] == "missing_tensor"
