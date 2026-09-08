"""Bounded execution observations across the executor and owner seams."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_OBSERVATION_BYTES = 8192


def environment(value: Mapping[str, Any], *, boot_id: str) -> pb.ExecutionEnvironment:
    return pb.ExecutionEnvironment(
        runtime_version=str(value.get("runtime_version") or ""),
        worker_image_digest=documents.raw(value["worker_image_digest"])
        if value.get("worker_image_digest")
        else b"",
        accelerator=str(value.get("accelerator") or ""),
        driver=str(value.get("driver") or ""),
        cuda=str(value.get("cuda") or ""),
        worker_boot_id=boot_id,
        execution_lane=str(value.get("execution_lane") or ""),
        execution_contract_digest=documents.raw(value["execution_contract_digest"])
        if value.get("execution_contract_digest")
        else b"",
        kernel_symbol=str(value.get("kernel_symbol") or ""),
    )


def document(observation: pb.ExecutionObservation) -> dict[str, Any]:
    if observation.ByteSize() > MAX_OBSERVATION_BYTES:
        raise ValueError("execution observation exceeds its bound")
    env = observation.environment
    if any(
        value and len(value) != 32
        for value in (env.worker_image_digest, env.execution_contract_digest)
    ):
        raise ValueError("execution observation has an invalid digest")
    if env.execution_lane not in ("", "eager", "compiled"):
        raise ValueError("execution observation has an invalid execution lane")
    result: dict[str, Any] = {
        "environment": {
            "runtime_version": env.runtime_version,
            "worker_image_digest": documents.spell(env.worker_image_digest)
            if env.worker_image_digest
            else "",
            "accelerator": env.accelerator,
            "driver": env.driver,
            "cuda": env.cuda,
            "worker_boot_id": env.worker_boot_id,
            "execution_lane": env.execution_lane,
            "execution_contract_digest": documents.spell(env.execution_contract_digest)
            if env.execution_contract_digest
            else "",
            "kernel_symbol": env.kernel_symbol,
        }
    }
    if observation.HasField("capture"):
        if (
            observation.capture.output_id != "runtime.capture"
            or len(observation.capture.content_digest) != 32
        ):
            raise ValueError("capture observation has an invalid identity")
        result["capture"] = {
            "output_id": observation.capture.output_id,
            "content_digest": documents.spell(observation.capture.content_digest),
        }
    return result
