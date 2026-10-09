"""Join ordinary `cozy run show --json --full` with the diagnostic summary artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def receipt(run: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    render = summary["render_call"]
    profile_enabled = summary.get("profile_enabled", True)
    if type(profile_enabled) is not bool:
        raise ValueError("profile_enabled must be a boolean when present")
    if profile_enabled:
        # Older enabled captures omit the flag, but must still supply their steps.
        profiled_steps = summary["profiled_step_indices"]
        if (
            not isinstance(profiled_steps, list)
            or not profiled_steps
            or any(type(step) is not int or step < 0 for step in profiled_steps)
        ):
            raise ValueError("profiled_step_indices must contain recorded step indices")
    else:
        if summary.get("profiled_step_indices", []) != [] or summary.get(
            "full_request_trace", False
        ) is not False:
            raise ValueError("An unprofiled summary cannot claim recorded trace steps")
        profiled_steps = []
    tails = []
    for event in run.get("events", []):
        if event.get("type") != "client.machine_work_finished":
            continue
        observed = event.get("payload", {}).get("observed_unix_ms")
        if not isinstance(observed, (int, float)) or observed <= 0 or not event.get("at"):
            continue
        recorded = datetime.fromisoformat(event["at"].replace("Z", "+00:00")).timestamp() * 1000
        tails.append(
            {
                "attempt": event.get("attempt"),
                "outcome_observed_unix_ms": observed,
                "outputs_registered_unix_ms": recorded,
                "wall_ms": recorded - observed,
                "scope": "Client outcome receipt through artifact export and completion recording",
                "clock": "client clock only; not subtracted from the remote machine timestamp",
            }
        )
    return {
        "diagnostic_only": True,
        "profile_enabled": profile_enabled,
        "software": summary.get("software", {}),
        "request": {
            key: run.get(key)
            for key in (
                "number",
                "request_id",
                "target",
                "machine",
                "status",
                "degree",
                "wall_ms",
                "execution_ms",
                "execution_known",
                "queued_ms",
            )
        },
        "native_stages": run.get("stages", []),
        "native_steps": run.get("steps", []),
        "native_gpus": run.get("gpus", []),
        "render_call": render,
        "encode_mux_output": summary.get("output_sinks", []),
        "client_completion_tails": tails,
        "bounded_kernel_trace_steps": profiled_steps,
        "full_request_trace": summary.get("full_request_trace", False),
        "trace_collection": summary.get(
            "trace_collection", "two DiT calls" if profile_enabled else "disabled"
        ),
        "phase_sources": {
            "machine_preparation_and_construction": "native_stages; preserve reused setup detail",
            "loading_weights": "native_stages and render_call.attribution_after.stages",
            "text_encoder": "condition_text",
            "denoise": "denoise; individual intervals in native_steps",
            "audio_vae": "decode_audio",
            "video_vae_and_overlapping_encode": "decode_video",
            "output_checks": "check_output",
            "encode_mux_probe_registration": "encode_mux_output; overlaps decode_video",
            "artifact_commit_and_client_export": (
                "client_completion_tails: export/recording; native commit is included in execution"
            ),
        },
        "overlap_notes": [
            "Do not add native stages, render wall, sink wall or nested kernel inclusive times.",
            "Stage clocks exclude loading-weight asides; those waits are separately reported.",
            "Reused executor boot/construction is historical, not fresh request cost.",
            "Sink wall overlaps decode; source time includes waiting, digest and materialization.",
            "Post-EOF sink time includes flush, mux, probe and registration, not only mux.",
            "Native commit is outside render_call; client tails cover export/recording together.",
            "Missing timings remain unmeasured; do not relabel residuals as specific phases.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-show", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(
            receipt(json.loads(args.run_show.read_text()), json.loads(args.summary.read_text())),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
