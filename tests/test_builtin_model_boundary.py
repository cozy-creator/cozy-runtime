"""Builtin model namespaces must not change the base/custom-model import contract."""

import subprocess
import sys


def test_builtin_namespaces_leave_runtime_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cozy_runtime; import cozy_runtime.author; "
            "import cozy_runtime.models; import cozy_runtime.models.minimax_h3; "
            "import cozy_runtime.models.qwen_image21; "
            "assert not {'torch', 'diffusers', 'transformers'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_custom_model_definition_does_not_load_builtins() -> None:
    code = """
import sys
from cozy_runtime.author import Loader, Model
class CustomModel(Model[object]):
    def load(self, loader: Loader) -> None:
        pass
assert issubclass(CustomModel, Model)
assert not any(name.startswith('cozy_runtime.models') for name in sys.modules)
assert not {'torch', 'diffusers', 'transformers'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
