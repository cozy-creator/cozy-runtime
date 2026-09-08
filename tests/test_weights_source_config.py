"""Inline config reads preserve exact source authorization and native custody."""

from pathlib import Path

import pytest

from cozy_runtime.author import CapabilityError, Loader, Model, WeightsSink
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._services import Attempt
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
from test_derived_config_runtime import _released_source
from test_model_runtime_closure import _CONFIG


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("config inspection does not construct a model")


def test_config_reads_only_named_configs_of_granted_native_sources(tmp_path: Path) -> None:
    store, manifest, length = _released_source(tmp_path)
    host = WeightsTransactionHost(
        store=store,
        owner_scope="config-proof",
        request_id="config-proof",
        invocation_spec_digest="sha256:" + "51" * 32,
        work_fingerprint="sha256:" + "52" * 32,
        writer_session_id=1,
        allowed_sources={manifest: length},
        output_bounds={},
    )
    source = _derive_model(Source, manifest)
    sink = WeightsSink(
        Attempt("config-proof", tmp_path / "spool"),
        {"source": source},
        {},
        None,
        source_config=host.config,
    )
    assert sink.config(source, "unet") == _CONFIG
    with pytest.raises(CapabilityError, match="not a granted model"):
        sink.config(_derive_model(Source, manifest), "unet")
    with pytest.raises(CapabilityError, match="no declared config"):
        sink.config(source, "ungranted")
    with pytest.raises(CapabilityError, match="immutable grant"):
        host.config("sha256:" + "ff" * 32, "unet")
    host.allowed_sources[manifest] = length + 1
    with pytest.raises(CapabilityError, match="TensorFS source config refused"):
        sink.config(source, "unet")
