"""LoRA preparation through the real Runtime writer and TensorFS source capabilities."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import msgspec
import pytest
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.author import CapabilityError, ModelArtifact
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.lora_composition import GRAPH_CONFIG, FORMAT, Graph, Selection, prepare
from native_weights import NativeExecution


@dataclass(frozen=True)
class Value:
    shape: tuple[int, ...]
    values: tuple[float, ...]
    dtype: str = "f32"

    def bytes(self) -> bytes:
        assert len(self.values) == math.prod(self.shape)
        return struct.pack(
            "<" + ("e" if self.dtype == "f16" else "f") * len(self.values), *self.values
        )


def model(
    store: tensorfs.Store,
    root: Path,
    name: str,
    component: str,
    tensors: Mapping[str, Value],
    config: bytes | None = None,
) -> ModelArtifact:
    declaration = Derivation(
        {},
        {
            component: Target(
                add={
                    key: Tensor(
                        value.dtype,
                        value.shape,
                        SPEC_PLAIN,
                        {"value": Part(value.dtype, value.shape)},
                    )
                    for key, value in tensors.items()
                }
            )
        },
        {"model": Config("add")} if config is not None else {},
        tuple((component, key) for key in tensors),
        files={"LICENSE": b"Fixture weights\n"} if config is not None else {},
    )
    with NativeExecution(store, root, name, {}, {"model": 1 << 20}) as execution:
        with execution.client.open_output("model", declaration) as writer:
            for key, value in tensors.items():
                writer.add_part(component, key, "value", value.bytes())
            if config is not None:
                writer.add_config("model", config)
            return execution.client.adopt_model(writer.commit())


def fixture(root: Path) -> tuple[tensorfs.Store, ModelArtifact, ModelArtifact, ModelArtifact]:
    store = tensorfs.Store.init(root / "store")
    base = model(
        store,
        root,
        "base",
        "transformer",
        {
            "proj.weight": Value((2, 3), (1, 2, 3, 4, 5, 6)),
            "proj.bias": Value((2,), (1, -1)),
            "untouched.weight": Value((1, 1), (7,)),
        },
        b'{"input":3,"output":2}',
    )
    first = model(
        store,
        root,
        "first",
        "adapter",
        {
            "proj.lora_A.weight": Value((1, 3), (1, 0, 2)),
            "proj.lora_B.weight": Value((2, 1), (3, 4)),
            "proj.alpha": Value((), (2,)),
        },
    )
    second = model(
        store,
        root,
        "second",
        "adapter",
        {
            "proj.lora_A.weight": Value((2, 3), (1, 1, 0, 0, 1, 2), "f16"),
            "proj.lora_B.weight": Value((2, 2), (1, 2, 3, 4), "f16"),
        },
    )
    return store, base, first, second


def compose(
    store: tensorfs.Store,
    root: Path,
    name: str,
    base: ModelArtifact,
    adapters: Sequence[tuple[ModelArtifact, float]],
) -> tuple[ModelArtifact, Graph]:
    inputs = {"base": base, **{f"adapter_{i}": row for i, (row, _) in enumerate(adapters)}}
    with NativeExecution(store, root, name, inputs, {"model": 1 << 20}) as execution:
        with ExitStack() as scopes:
            source = scopes.enter_context(execution.client.source(base.manifest.digest))
            selected = [
                Selection(
                    scopes.enter_context(execution.client.source(row.manifest.digest)),
                    "transformer",
                    strength=scale,
                )
                for row, scale in adapters
            ]
            prepared = prepare(source, selected)
            with execution.client.open_output("model", prepared.declaration) as writer:
                writer.add_config(GRAPH_CONFIG, prepared.config)
                result = execution.client.adopt_model(writer.commit())
            return result, prepared.graph


def header(store: tensorfs.Store, artifact: ModelArtifact) -> bytes:
    raw = store.manifest(artifact.manifest.digest)["header"]
    assert raw is not None
    return raw


def test_ordered_stack_grafts_base_and_factor_parts_without_mutation(tmp_path: Path) -> None:
    store, base, first, second = fixture(tmp_path)
    original = header(store, base)
    result, graph = compose(store, tmp_path, "composed", base, [(first, 0.5), (second, -0.25)])
    assert graph.format == FORMAT
    assert [(row.rank, row.alpha, row.strength, row.dtype) for row in graph.layers] == [
        (1, 2.0, 0.5, "f32"),
        (2, 2.0, -0.25, "f16"),
    ]
    prepared_header = tensorfs.parse_header(header(store, result))
    base_header = tensorfs.parse_header(original)
    before = base_header["components"]["transformer"]
    after = prepared_header["components"]["transformer"]
    assert set(after) == {
        "proj.base_layer.weight",
        "proj.base_layer.bias",
        "untouched.weight",
        "proj.lora_A.adapter_0.weight",
        "proj.lora_B.adapter_0.weight",
        "proj.lora_A.adapter_1.weight",
        "proj.lora_B.adapter_1.weight",
    }
    for old, new in [
        ("proj.weight", "proj.base_layer.weight"),
        ("proj.bias", "proj.base_layer.bias"),
        ("untouched.weight", "untouched.weight"),
    ]:
        assert after[new] == before[old]
    for index, source in enumerate((first, second)):
        source_header = tensorfs.parse_header(header(store, source))
        for role in ("A", "B"):
            assert (
                after[f"proj.lora_{role}.adapter_{index}.weight"]
                == source_header["components"]["adapter"][f"proj.lora_{role}.weight"]
            )
    assert prepared_header["configs"]["model"] == base_header["configs"]["model"]
    assert msgspec.json.decode(prepared_header["configs"][GRAPH_CONFIG], type=Graph) == graph
    assert header(store, base) == original
    checkout = tmp_path / "checkout"
    store.checkout(result.manifest.digest, checkout, symlink=False)
    assert (checkout / "LICENSE").read_bytes() == b"Fixture weights\n"

    repeated, _ = compose(store, tmp_path, "repeated", base, [(first, 0.5), (second, -0.25)])
    reversed_, _ = compose(store, tmp_path, "reversed", base, [(second, -0.25), (first, 0.5)])
    reweighted, _ = compose(store, tmp_path, "reweighted", base, [(first, 0.75), (second, -0.25)])
    assert repeated.manifest == result.manifest
    assert len({result.manifest.digest, reversed_.manifest.digest, reweighted.manifest.digest}) == 3


def test_zero_strength_preserves_the_pristine_component(tmp_path: Path) -> None:
    store, base, first, _second = fixture(tmp_path)
    result, graph = compose(store, tmp_path, "zero", base, [(first, 0.0)])
    assert graph.layers == ()
    before = tensorfs.parse_header(header(store, base))
    after = tensorfs.parse_header(header(store, result))
    assert after["components"] == before["components"]


@pytest.mark.parametrize(
    "bad,code",
    [
        ({"proj.lora_A.weight": Value((1, 3), (1, 2, 3))}, "adapter_pair"),
        (
            {
                "proj.lora_A.weight": Value((1, 2), (1, 2)),
                "proj.lora_B.weight": Value((2, 1), (1, 2)),
            },
            "adapter_shape",
        ),
        (
            {
                "proj.lora_A.weight": Value((1, 3), (1, float("nan"), 3)),
                "proj.lora_B.weight": Value((2, 1), (1, 2)),
            },
            "adapter_nonfinite",
        ),
        (
            {
                "proj.lora_A.weight": Value((1, 3), (1, 2, 3)),
                "proj.lora_B.weight": Value((2, 1), (1, 2)),
                "proj.alpha": Value((1,), (1,)),
            },
            "adapter_alpha",
        ),
    ],
)
def test_invalid_factors_refuse_even_at_zero_strength(
    tmp_path: Path, bad: dict[str, Value], code: str
) -> None:
    store, base, _first, _second = fixture(tmp_path)
    broken = model(store, tmp_path, "broken", "adapter", bad)
    with pytest.raises(CapabilityError) as refused:
        compose(store, tmp_path, "refused", base, [(broken, 0.0)])
    assert refused.value.code == code
