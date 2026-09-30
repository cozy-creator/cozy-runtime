"""Model-catalog records consume known fields without coupling supported versions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import pytest

from cozy_runtime.internal import egress
from cozy_runtime.internal.worker import machine_model_resolve as models
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal

MANIFEST = b"an exact manifest"
DIGEST = "sha256:" + hashlib.sha256(MANIFEST).hexdigest()
SLOT = {"path": "generate.models.source"}


class Catalog(models.Catalog):
    def __init__(self, replies: Mapping[str, object]) -> None:
        super().__init__("http://catalog.example")
        self.replies = replies
        self.reads: list[str] = []

    def text(self, path: str, limit: int = models.MAX_METADATA) -> bytes | None:
        self.reads.append(path)
        value = self.replies.get(path)
        return None if value is None else json.dumps(value).encode()


def test_binding_and_resolution_ignore_additions_and_cache_consumed_records() -> None:
    catalog = Catalog(
        {
            "/v1/packages/owner/package/bindings": {
                "future": [1, {"advisory": True}],
                "bindings": [
                    None,
                    {"ignored": "no slot"},
                    {"slot": "other.model", "model": [], "ladder": "not consumed"},
                    {
                        "slot": SLOT["path"],
                        "model": "owner/model",
                        "release": "2.0",
                        "ladder": [
                            {"gpu": "*", "gpus": 0, "lane": "bf16", "future": {"x": 1}},
                            {"gpu": "*", "gpus": 2, "lane": "fp8"},
                        ],
                        "extra": "ignored",
                    },
                ],
            },
            "/v1/models/resolve?ref=owner%2Fmodel%402.0&lane=fp8": {
                "model": "Owner/Model",
                "manifest_id": DIGEST,
                "manifest_length": len(MANIFEST),
                "release": "2.0",
                "lane": "fp8",
                "new_metadata": {"unread": [False]},
            },
        }
    )
    declaration = {**SLOT, "sequence_parallel": {"degrees": [2], "new": True}}
    first = models.slot(catalog, "owner/package", declaration, "source", None, lambda _: True)
    second = models.slot(catalog, "owner/package", declaration, "source", None, lambda _: True)
    assert (
        first
        == second
        == {
            "parameter": "source",
            "public_origin": catalog.origin,
            "gpu": "*",
            "gpus": 2,
            "repository": "Owner/Model",
            "manifest": {"digest": DIGEST, "length": len(MANIFEST)},
            "release": "2.0",
            "lane": "fp8",
        }
    )
    assert len(catalog.reads) == 2
    assert catalog.binding("owner/package", SLOT["path"]) is catalog.binding(
        "owner/package", SLOT["path"]
    )


@pytest.mark.parametrize("length", [None, 0, -1, True, "16"])
def test_legacy_or_unusable_length_is_measured_from_verified_bytes(
    monkeypatch: pytest.MonkeyPatch, length: object
) -> None:
    catalog = Catalog(
        {
            "/v1/models/resolve?ref=owner%2Fmodel%402.0&lane=bf16": {
                "model": "owner/model",
                "manifest_id": DIGEST,
                "manifest_length": length,
            }
        }
    )
    measured: list[str] = []

    def measure(
        _host: str,
        _port: int | None,
        _tls: object,
        path: str,
        *,
        limit: int,
        credential: Mapping[str, str],
    ) -> tuple[int, int, bytes]:
        assert limit > len(MANIFEST) and credential == {}
        measured.append(path)
        return 200, len(MANIFEST), hashlib.sha256(MANIFEST).digest()

    monkeypatch.setattr(egress, "measure_catalog_document", measure)
    assert catalog.resolve("owner/model", "2.0", "bf16") == {
        "repository": "owner/model",
        "manifest": {"digest": DIGEST, "length": len(MANIFEST)},
        "release": "2.0",
        "lane": "bf16",
    }
    assert measured == [f"/v1/models/owner/model/checkpoints/{DIGEST}"]


def test_caller_pin_bypasses_binding_and_inapplicable_ladder_metadata() -> None:
    catalog = Catalog({})
    selected = models.slot(
        catalog,
        "owner/package",
        {"sequence_parallel": "not consumed for a pinned choice"},
        "source",
        {"repository": "owner/model", "manifest": {"digest": DIGEST, "length": 19}},
        lambda _: False,
    )
    assert selected["manifest"] == {"digest": DIGEST, "length": 19}
    assert selected["gpus"] == 0 and selected["gpu"] == "*"
    assert catalog.reads == []


def test_explicit_repository_uses_latest_unyanked_full_precision_lane() -> None:
    catalog = Catalog(
        {
            "/v1/models/owner/model": {
                "releases": [
                    {"release": "3.0", "yanked": True, "lanes": "not consumed"},
                    {"release": "1.0", "lanes": {"not": "consumed"}},
                    {
                        "release": "2.0",
                        "future": "ignored",
                        "lanes": [
                            {"lane": "fp8", "bytes": {"unused": True}},
                            {"lane": "bf16", "bytes": 2, "new": True},
                        ],
                    },
                ]
            },
            "/v1/models/resolve?ref=owner%2Fmodel%402.0&lane=bf16": {
                "model": "owner/model",
                "manifest_id": DIGEST,
                "manifest_length": 19,
            },
        }
    )
    selected = models.slot(
        catalog, "owner/package", SLOT, "source", {"repository": "owner/model"}, lambda _: True
    )
    assert selected["release"] == "2.0" and selected["lane"] == "bf16"
    assert all("bindings" not in path for path in catalog.reads)


def test_missing_rung_gpu_keeps_the_existing_fit_input_and_output_default() -> None:
    catalog = Catalog(
        {
            "/v1/packages/owner/package/bindings": {
                "bindings": [
                    {
                        "slot": SLOT["path"],
                        "model": "owner/model",
                        "release": "2.0",
                        "ladder": [{"lane": "bf16"}],
                    }
                ]
            },
            "/v1/models/resolve?ref=owner%2Fmodel%402.0&lane=bf16": {
                "model": "owner/model",
                "manifest_id": DIGEST,
                "manifest_length": 19,
            },
        }
    )
    seen: list[Mapping[str, object]] = []

    def fits(rung: Mapping[str, object]) -> bool:
        seen.append(rung)
        return True

    selected = models.slot(catalog, "owner/package", SLOT, "source", None, fits)
    assert "gpu" not in seen[0] and selected["gpu"] == "*"


@pytest.mark.parametrize("small_size", ["10", {"invalid": True}])
def test_lane_size_is_checked_when_the_fallback_ranking_consumes_it(small_size: object) -> None:
    catalog = Catalog(
        {
            "/v1/models/owner/model": {
                "releases": [
                    {
                        "release": "2.0",
                        "lanes": [
                            {"lane": "large", "bytes": 20},
                            {"lane": "small", "bytes": small_size},
                        ],
                    }
                ]
            },
            "/v1/models/resolve?ref=owner%2Fmodel%402.0&lane=small": {
                "model": "owner/model",
                "manifest_id": DIGEST,
                "manifest_length": 19,
            },
        }
    )
    choice = {"repository": "owner/model"}
    if isinstance(small_size, dict):
        with pytest.raises(WorkspaceRefusal, match="invalid lane size"):
            models.slot(catalog, "owner/package", SLOT, "source", choice, lambda _: True)
        assert not catalog.resolved.checkpoints
    else:
        selected = models.slot(catalog, "owner/package", SLOT, "source", choice, lambda _: True)
        assert selected["lane"] == "small"


@pytest.mark.parametrize(
    "malformed", [{"model": ["owner/model"]}, {"manifest_id": 4}, {"release": {"v": 2}}]
)
def test_malformed_consumed_resolution_fields_refuse_before_caching(
    malformed: dict[str, object],
) -> None:
    catalog = Catalog(
        {
            "/v1/models/resolve?ref=owner%2Fmodel%402.0&lane=bf16": {
                "model": "owner/model",
                "manifest_id": DIGEST,
                "manifest_length": 19,
                **malformed,
            }
        }
    )
    with pytest.raises(WorkspaceRefusal, match="invalid resolution"):
        catalog.resolve("owner/model", "2.0", "bf16")
    assert not catalog.resolved.checkpoints
