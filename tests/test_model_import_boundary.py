"""Defining a Model must not change the base package's torch-free import contract."""

import subprocess
import sys


def test_custom_model_definition_stays_torch_free() -> None:
    code = """
import sys
from cozy_runtime.author import Loader, Model
class CustomModel(Model[object]):
    def load(self, loader: Loader) -> None:
        pass
assert issubclass(CustomModel, Model)
assert not {'torch', 'diffusers', 'transformers'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
