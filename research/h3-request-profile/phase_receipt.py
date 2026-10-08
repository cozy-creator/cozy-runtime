"""Join ordinary `cozy run show --json --full` with the diagnostic summary artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def receipt(run: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    render = summary["render_call"]
    return {
        "diagnostic_only": True,
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
        "bounded_kernel_trace_steps": summary["profiled_step_indices"],
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
                "included in native execution/client wall; no isolated timer exposed"
            ),
        },
        "overlap_notes": [
            "Do not add native stages, render wall, sink wall or nested kernel inclusive times.",
            "Stage clocks exclude loading-weight asides; those waits are separately reported.",
            "Reused executor boot/construction is historical, not fresh request cost.",
            "Sink wall overlaps decode; source time includes waiting, digest and materialization.",
            "Post-EOF sink time includes flush, mux, probe and registration, not only mux.",
            "Final artifact commit/export is outside render_call; retain total client wall.",
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
