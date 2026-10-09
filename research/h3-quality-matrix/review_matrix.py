"""Validate completed videos and build a static local review index; never submit runs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=600)


def verify(video: Path, report: Path, sha: str) -> dict:
    if report.exists():
        saved = json.loads(report.read_text())
        required = ("full_av_decode", "geometry", "frames", "fps", "audio", "duration")
        if saved.get("sha256") == sha and all(saved.get("checks", {}).get(k) for k in required):
            return saved
    probe = command(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video),
        ]
    )
    probe.check_returncode()
    data = json.loads(probe.stdout)
    streams = data["streams"]
    v = next(row for row in streams if row["codec_type"] == "video")
    a = next(row for row in streams if row["codec_type"] == "audio")
    decoded = command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "2",
            "-i",
            str(video),
            "-f",
            "null",
            "-",
        ]
    )
    checks = {
        "full_av_decode": decoded.returncode == 0 and not decoded.stderr,
        "geometry": (v["width"], v["height"]) == (1344, 768),
        "frames": int(v["nb_read_frames"]) == 362,
        "fps": v["avg_frame_rate"] == "24/1",
        "audio": a["channels"] == 2 and a["sample_rate"] == "32000",
        "duration": abs(float(v["duration"]) - 362 / 24) < 0.001,
    }
    result = {
        "file": str(video),
        "sha256": sha,
        "checks": checks,
        "probe": data,
        "decode_stderr": decoded.stderr,
        "subjective_quality_approved": False,
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def read_run(row: dict, out: Path) -> dict:
    result = dict(row)
    result.update(verified=False, verification_error="", execution_seconds=None, stages={})
    show = {}
    if row.get("run_show") and Path(row["run_show"]).is_file():
        show = json.loads(Path(row["run_show"]).read_text())
        result["status"] = show.get("status", row.get("status", "pending"))
        result["run_number"] = show.get("number")
        if show.get("execution_known") and isinstance(show.get("execution_ms"), (int, float)):
            result["execution_seconds"] = show["execution_ms"] / 1000
        result["stages"] = {
            s["name"]: s["ms"] / 1000
            for s in show.get("stages", [])
            if s.get("kind") == "inference"
        }
        result["attention_receipts"] = [g.get("attention", {}) for g in show.get("gpus", [])]
    if result.get("status") not in {"completed", "succeeded", "success"}:
        return result
    if not row.get("video") or not Path(row["video"]).is_file():
        result["verification_error"] = "Completed receipt; video has not been retained locally"
        return result
    try:
        video = Path(row["video"]).resolve()
        sha = digest(video)
        name = row["id"] + "-" + sha[:12]
        report = out / "qa" / (name + ".json")
        qa = verify(video, report, sha)
        result["checks"] = dict(qa["checks"])
        steps = [s for s in show.get("steps", []) if s.get("name") == "denoise"]
        result["checks"]["eight_denoise_steps"] = bool(steps) and steps[0]["count"] == 8
        result["checks"]["recorded_execution_time"] = result["execution_seconds"] is not None
        expected_sparse = {"dense4-sol4": 192, "dense8": 0}.get(row["schedule"])
        if expected_sparse is not None:
            actual_sparse = sum(
                g.get("attention", {}).get("sol", {}).get("sparse", 0) for g in show.get("gpus", [])
            )
            result["observed_sparse_calls"] = actual_sparse
            result["checks"]["attention_schedule"] = actual_sparse == expected_sparse
        selector = row.get("attention_selector")
        if selector:
            observed = [
                item
                for g in show.get("gpus", [])
                for item in g.get("attention", {}).get("observed", "").split(",")
            ]
            result["checks"]["selected_attention"] = "fl2va_dit=" + selector in observed
        media = out / "media" / (name + video.suffix)
        if not media.exists():
            try:
                os.link(video, media)
            except OSError:
                shutil.copyfile(video, media)
        result.update(
            sha256=sha,
            media=str(media.relative_to(out)),
            qa=str(report.relative_to(out)),
            verified=all(result["checks"].values()),
        )
        if not result["verified"]:
            result["verification_error"] = ", ".join(
                k for k, good in result["checks"].items() if not good
            )
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as exc:
        result["verification_error"] = f"{type(exc).__name__}: {exc}"
    return result


def seconds(value: float | None) -> str:
    return "pending" if value is None else f"{int(value // 60)}m {value % 60:06.3f}s"


def build(manifest: Path, out: Path) -> dict:
    document = json.loads(manifest.read_text())
    rows = document["runs"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", key) for key in ids
    ):
        raise ValueError("Each row needs a unique safe id")
    out.mkdir(parents=True, exist_ok=True)
    for child in ("qa", "media"):
        (out / child).mkdir(exist_ok=True)
    results = [read_run(row, out) for row in rows]
    expected = document.get("expected", 16)
    summary = {
        "title": document.get("title", "MiniMax H3 attention comparison"),
        "expected": expected,
        "verified": sum(r["verified"] for r in results),
        "runs": results,
        "quality_approval": "Pending user review",
        "timing_boundary": (
            "Recorded Cozy execution time; preparation and observer/download time excluded."
        ),
        "limits": (
            "File/step checks do not establish visual quality or repeated-run speed confidence."
        ),
    }
    (out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    escape = html.escape
    scenes = list(dict.fromkeys(row["scene"] for row in results))
    groups: dict[tuple[str, str], dict[str, dict]] = OrderedDict()
    for row in results:
        key = row["kernel"], row["schedule"]
        if row["scene"] in groups.setdefault(key, {}):
            raise ValueError("Duplicate kernel/schedule/scene cell")
        groups[key][row["scene"]] = row
    parts = [
        "<!doctype html><html><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width'>",
        "<title>" + escape(summary["title"]) + "</title>",
        "<style>body{font:16px system-ui;margin:24px;background:#121418;color:#edf0f5}"
        "table{border-collapse:collapse;width:100%;max-width:1600px}td,th{border:1px solid #445;"
        "padding:12px;vertical-align:top;text-align:left}video{width:100%;max-width:560px}"
        "a{color:#9ac8ff}.note{color:#bec6d4;font-size:14px}.bad{color:#ffb1a8}</style>",
        "<h1>" + escape(summary["title"]) + "</h1>",
        f"<p>{summary['verified']} / {expected} clips verified. "
        "Quality approval awaits your review.</p>",
        "<p class=note>Checks require all eight denoising steps. "
        "Videos play at their original quality. Times are recorded execution times; "
        "consult each timing note for setup/compile qualifications.</p>",
        "<table><thead><tr><th>Attention / schedule</th>"
        + "".join("<th>" + escape(s) + "</th>" for s in scenes)
        + "</tr></thead><tbody>",
    ]
    for (kernel, schedule), by_scene in groups.items():
        example = next(iter(by_scene.values()))
        label = example.get("kernel_label", kernel)
        schedule_label = {
            "dense4-sol4": "4 dense + 4 Sol steps",
            "dense8": "8 dense steps",
        }.get(schedule, schedule)
        parts.append(
            "<tr><th>"
            + escape(label)
            + "<p class=note>"
            + escape(schedule_label)
            + "</p><p class=note>"
            + escape(example.get("arithmetic_note", ""))
            + "</p></th>"
        )
        for scene in scenes:
            row = by_scene.get(scene)
            if not row:
                parts.append("<td>Pending</td>")
                continue
            cell = "<td><strong>" + escape(seconds(row["execution_seconds"])) + "</strong>"
            cell += (
                "<p class=note>Seed "
                + escape(str(row.get("seed", "unrecorded")))
                + " · Run "
                + escape(str(row.get("run_number", "pending")))
                + "</p>"
            )
            if row.get("media"):
                href = quote(row["media"])
                cell += (
                    f'<video controls preload="none" src="{href}"></video>'
                    f'<p><a href="{href}">Original video</a> · '
                    f'<a href="{quote(row["qa"])}">AV checks</a></p>'
                )
            cell += (
                "<p class=note>"
                + escape(row.get("timing_note", "Warm/cold qualification not supplied"))
                + "</p>"
            )
            if row["verification_error"]:
                cell += "<p class=bad>" + escape(row["verification_error"]) + "</p>"
            elif not row["verified"]:
                cell += "<p class=note>" + escape(row.get("status", "pending")) + "</p>"
            parts.append(cell + "</td>")
        parts.append("</tr>")
    parts.append("</tbody></table><p class=note>" + escape(summary["limits"]) + "</p></html>")
    (out / "index.html").write_text("\n".join(parts))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.manifest, args.out)
    print(
        json.dumps(
            {
                "index": str(args.out / "index.html"),
                "verified": result["verified"],
                "expected": result["expected"],
            }
        )
    )
