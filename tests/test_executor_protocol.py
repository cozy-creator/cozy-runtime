"""Exercise real executor hello frames independently of package release metadata."""

from __future__ import annotations

import subprocess
import sys








def test_an_executor_maps_installed_distributions_once() -> None:
    """Diffusers and Transformers both ask at import; the executor walks the RECORDs once and
    each caller still gets the same answer as a fresh walk, in a copy of its own."""
    program = """
import importlib.metadata as metadata
from cozy_runtime.internal import executor
fresh = metadata.packages_distributions()
walks = []
walk = metadata.distributions
metadata.distributions = lambda **kw: walks.append(1) or walk(**kw)
executor._remember_distributions()
first = metadata.packages_distributions()
first[next(iter(first))].append("mutated")
second = metadata.packages_distributions()
assert first != second == fresh and len(walks) == 1, (len(walks), second == fresh)
"""
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
