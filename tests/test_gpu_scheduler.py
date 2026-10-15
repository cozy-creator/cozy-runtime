"""Mixed GPU-degree declarations are refused when the interface is built."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover

MARCO = Path(__file__).resolve().parent.parent / "examples" / "marco-polo"


# ------------------------------------------------------------------------------ policy


def test_mixed_degree_declarations_are_refused_when_the_interface_is_built() -> None:
    body = package_interface.build(discover(MARCO))
    entry = next(row for row in body["entrypoints"] if row["name"] == "marco")
    slot = {"path": "marco.models.base", "class": "Base", "component_use": {}}
    with pytest.raises(ConformanceError) as refused:
        package_interface.callable_doc(
            name="marco",
            payload_type=dict,
            result_type=dict,
            models=[{**slot, "sequence_parallel": {"degrees": [2, 4]}}, slot],
            assets=None,
            publishes=False,
            weights_outputs=(),
        )
    assert refused.value.code == "sequence_parallel_declaration_conflict"
    assert entry["name"] == "marco"
