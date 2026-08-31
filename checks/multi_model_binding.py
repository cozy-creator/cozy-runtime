#!/usr/bin/env python3
"""Bounded complete multi-model executor aggregation fence (no device or model bytes)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import patch

from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.worker.plan import DeclaredBinding, ModelBinding


class Residency:
    poisoned = ""
    last_shortfall = None

    def __init__(self, parameter: str) -> None:
        self.parameter = parameter

    def open_attempt(self, _placement: str, _headroom_bytes: int) -> None:
        return

    def document(self) -> dict[str, Any]:
        return {
            "placement": "all_resident",
            "headroom_bytes": 0,
            "admissions": 0,
            "stages": 0,
            "evictions": 0,
            "attempt_stages": 0,
            "attempt_evictions": 0,
            "allocator_delta_bytes": 0,
            "resident": {"core": 10},
            "evicted": {},
        }


def row(path: str, parameter: str) -> dict[str, Any]:
    return {
        "binding": {
            "model_binding_path": path,
            "model_parameter_name": parameter,
        },
        "budgets": {},
        "expect_construction_digest": "",
    }


def main() -> None:
    model = ModelBinding(
        model_class="proof:Model",
        model_binding_path="render.models.alpha",
        model_parameter_name="alpha",
        config_digest="sha256:" + "1" * 64,
        store="/store",
        config="/config",
        variant="sm90",
        reference_snapshot="sha256:" + "2" * 64,
        components=("core",),
        snapshots={"core": "sha256:" + "2" * 64},
        stamps={"task": ("proof",)},
        expected_logical_model_digest="sha256:" + "3" * 64,
    )

    def construction(member: ModelBinding) -> str:
        return DeclaredBinding(
            entrypoint_binding_digest="sha256:" + "4" * 64,
            entrypoint="render",
            model_class=member.model_class,
            model_binding_path=member.model_binding_path,
            model_parameter_name=member.model_parameter_name,
            release="sha256:" + "5" * 64,
            models=(member,),
        ).construction_key()

    identity = construction(model)
    assert construction(dataclasses.replace(model, model_parameter_name="changed")) != identity
    assert construction(dataclasses.replace(model, stamps={"task": ("changed",)})) != identity
    assert (
        construction(dataclasses.replace(model, expected_logical_model_digest="sha256:" + "6" * 64))
        != identity
    )

    def prepared(child: Executor, command: dict[str, Any]) -> dict[str, Any]:
        binding = command["binding"]
        parameter = str(binding["model_parameter_name"])
        child.models = {parameter: object()}
        child.residency = Residency(parameter)
        child.torch = object()
        child.discovered = object()
        child.surface_names = {"render": object()}
        facts = {
            "cuda_init_ms": 1,
            "prepare_ms": 2,
            "fill_ms": 3,
            "filled": 1,
            "filled_bytes": 10,
            "device_free_bytes": 100,
            "device_total_bytes": 200,
            "allocator_bytes": 20,
            "reserved_bytes": 30,
            "resident": {"core": 10},
            "parked": [],
            "declared_scopes": {"run": ["core"]},
            "rss_bytes": 40,
            "pinned": {"registered_bytes": 0, "in_flight_bytes": 0},
            "gpu_name": "proof",
            "torch_version": "proof",
            "cuda_version": "proof",
            "ignored_extra_keys": [],
            "transcribed_markers": [],
        }
        return {
            "ok": True,
            "model_construction_digest": "sha256:" + ("1" if parameter == "alpha" else "2") * 64,
            "constructed_model_digest": "sha256:" + ("3" if parameter == "alpha" else "4") * 64,
            "delivery": {"routes": {"decoded_float": 1}, "kernels": {}},
            "facts": facts,
        }

    with patch.object(Executor, "_prepare_first", prepared):
        executor = Executor(object(), Path.cwd())  # type: ignore[arg-type]
        command = {
            "models": [
                row("render.models.alpha", "alpha"),
                row("render.models.beta", "beta"),
            ]
        }
        reply = Executor._prepare_many(executor, command)
        assert reply["ok"] is True, reply
        assert sorted(executor.models) == ["alpha", "beta"]
        assert reply["facts"]["filled"] == 2
        assert reply["facts"]["filled_bytes"] == 20
        assert sorted(reply["facts"]["resident"]) == ["alpha/core", "beta/core"]
        assert sorted(reply["model_construction_digests"]) == ["alpha", "beta"]

        duplicate = Executor(object(), Path.cwd())  # type: ignore[arg-type]
        refused = Executor._prepare_many(
            duplicate,
            {"models": [row("render.models.alpha", "same"), row("render.models.beta", "same")]},
        )
        assert refused["code"] == "model_binding_set_invalid", refused
    print(
        "multi-model binding: complete sorted set aggregated; duplicate parameter refused; "
        "param/stamps/expected-contract are construction identity"
    )


if __name__ == "__main__":
    main()
