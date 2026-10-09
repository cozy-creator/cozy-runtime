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


def attention_note(row: dict) -> str:
    notes = {
        ("kitchen-int8", "dense8"): "INT8/INT32 dense attention; FP32 softmax and output state.",
        (
            "kitchen-int8",
            "dense4-sol4",
        ): (
            "INT8/INT32 dense and exact Sol attention; BF16/FP32 pooled tail. Shared QKV "
            "and previous-step statistics."
        ),
        (
            "sage3-nvfp4",
            "dense8",
        ): "Block-scaled NVFP4/FP32 dense attention with global head-mean correction.",
        (
            "sage3-nvfp4",
            "dense4-sol4",
        ): (
            "NVFP4/FP32 dense path; Kitchen INT8/INT32 Sol exact tiles and prefixes; BF16"
            "/FP32 pooled tail. Shared QKV and previous-step statistics."
        ),
        (
            "fp8-fp16",
            "dense8",
        ): "FP8/FP16 tile accumulation across all attention blocks; FP32 online state.",
        (
            "fp8-fp16",
            "dense4-sol4",
        ): (
            "FP8/FP16 dense paths and exact Sol tiles; BF16/FP32 routing and pooled tail;"
            " FP32 online state."
        ),
        ("bf16-fp32", "dense8"): "BF16/FP32 Flash dense attention.",
        (
            "bf16-fp32",
            "dense4-sol4",
        ): "BF16/FP32 Flash dense attention and upstream Sol sparse attention.",
    }
    return notes.get((row["kernel"], row["schedule"]), row.get("arithmetic_note", ""))


def arithmetic_gate(document: dict) -> dict:
    result = {"run": document.get("gpu_gate_run"), "passed": False}
    path = document.get("gpu_gate_report")
    if path and Path(path).is_file():
        gate = json.loads(Path(path).read_text())
        result["passed"] = (
            gate.get("passed") is True
            and gate.get("fp16_mma_observed") is True
            and gate.get("runtime") == document.get("runtime_wheel", {}).get("version")
        )
    return result


def comparison_panel(rows: list[dict], scenes: list[str]) -> list[str]:
    clips = []
    for row in rows:
        schedule = {"dense8": "8 dense", "dense4-sol4": "4 dense + 4 Sol"}.get(
            row["schedule"], row["schedule"]
        )
        clips.append(
            {
                "scene": row["scene"],
                "key": row["kernel"] + "/" + row["schedule"],
                "label": row.get("kernel_label", row["kernel"]) + " · " + schedule,
                "seed": row.get("seed"),
                "run": row.get("run_number"),
                "time": seconds(row["execution_seconds"]),
                "media": quote(row["media"]) if row.get("verified") and row.get("media") else None,
            }
        )
    payload = json.dumps(clips, ensure_ascii=True).replace("<", "\\u003c")
    return [
        """<section class=compare aria-labelledby=compare-title>
<h2 id=compare-title>Compare original clips</h2>""",
        "<div class=compare-bar><label>Scene <select id=compare-scene>"
        + "".join(
            f'<option value="{html.escape(scene, quote=True)}">{html.escape(scene)}</option>'
            for scene in scenes
        )
        + """</select></label> <button id=compare-play>Play both</button>
<button id=compare-pause>Pause both</button> <button id=compare-restart>Restart</button>
<label>Audio <select id=compare-audio><option value=left>Left only</option>
<option value=right>Right only</option><option value=none>Muted</option></select></label></div>""",
        """<div class=compare-grid>
<div><label>Left <select id=compare-left-choice></select></label>
<video id=compare-left controls preload=metadata playsinline></video>
<p id=compare-left-caption class=note></p></div>
<div><label>Right <select id=compare-right-choice></select></label>
<video id=compare-right controls preload=metadata playsinline muted></video>
<p id=compare-right-caption class=note></p></div>
</div><p id=compare-status class=note role=status></p></section>""",
        "<script id=compare-data type=application/json>" + payload + "</script>",
        COMPARISON_SCRIPT,
    ]


COMPARISON_SCRIPT = r"""<script>
(() => {
  const clips = JSON.parse(document.getElementById('compare-data').textContent);
  const byId = id => document.getElementById(id);
  const scene = byId('compare-scene');
  const choices = [byId('compare-left-choice'), byId('compare-right-choice')];
  const videos = [byId('compare-left'), byId('compare-right')];
  const captions = [byId('compare-left-caption'), byId('compare-right-caption')];
  const status = byId('compare-status');
  const audio = byId('compare-audio');
  const buttons = [byId('compare-play'), byId('compare-pause'), byId('compare-restart')];
  let selected = [null, null], frame = 0;
  function pauseBoth() {
    videos.forEach(video => video.pause());
    cancelAnimationFrame(frame); frame = 0;
  }
  function align(time) {
    videos.forEach(video => { if (video.readyState) video.currentTime = time; });
  }
  function sync() {
    if (videos.some(video => video.paused || video.ended)) { frame = 0; return; }
    if (Math.abs(videos[0].currentTime - videos[1].currentTime) > 0.08)
      videos[1].currentTime = videos[0].currentTime;
    frame = requestAnimationFrame(sync);
  }
  function startSync() { if (!frame) frame = requestAnimationFrame(sync); }
  function playBoth() {
    if (selected.some(clip => !clip || !clip.media)) return;
    align(videos[0].currentTime);
    Promise.all(videos.map(video => video.play())).then(startSync).catch(error => {
      pauseBoth(); status.textContent = 'Playback could not start: ' + error.message;
    });
  }
  function setAudio() {
    videos[0].muted = audio.value !== 'left';
    videos[1].muted = audio.value !== 'right';
  }
  function showPair() {
    pauseBoth();
    selected = choices.map(choice => clips.find(clip =>
      clip.scene === scene.value && clip.key === choice.value));
    selected.forEach((clip, index) => {
      const video = videos[index];
      if (clip && clip.media) {
        video.src = clip.media;
        captions[index].textContent = clip.label + ' · Seed ' + clip.seed
          + ' · Run ' + clip.run + ' · ' + clip.time;
      } else {
        video.removeAttribute('src'); captions[index].textContent = 'Awaiting a verified clip.';
      }
      video.load();
    });
    const playable = selected.every(clip => clip && clip.media);
    buttons.forEach(button => { button.disabled = !playable; });
    status.textContent = !playable ? 'Choose two completed, verified clips for this scene.'
      : selected[0].seed !== selected[1].seed ? 'These clips have different seeds.'
      : selected[0].key === selected[1].key ? 'Same clip selected on both sides.'
      : 'Same scene and seed. Playback and seeking are synchronized; only one audio track plays.';
    setAudio();
  }
  function populate() {
    const available = clips.filter(clip => clip.scene === scene.value);
    const ready = available.filter(clip => clip.media);
    choices.forEach((choice, index) => {
      const previous = choice.value;
      choice.replaceChildren();
      available.forEach(clip => {
        const option = document.createElement('option');
        option.value = clip.key;
        option.textContent = clip.label + (clip.media ? ' · ' + clip.time : ' · pending');
        option.disabled = !clip.media; choice.appendChild(option);
      });
      choice.value = ready.some(clip => clip.key === previous) ? previous
        : ready.length ? ready[Math.min(index, ready.length - 1)].key : available[0]?.key || '';
    });
    showPair();
  }
  buttons[0].addEventListener('click', playBoth);
  buttons[1].addEventListener('click', pauseBoth);
  buttons[2].addEventListener('click', () => { align(0); playBoth(); });
  scene.addEventListener('change', populate);
  choices.forEach(choice => choice.addEventListener('change', showPair));
  audio.addEventListener('change', setAudio);
  videos.forEach((video, index) => {
    const other = videos[1 - index];
    video.addEventListener('pause', pauseBoth);
    video.addEventListener('ended', pauseBoth);
    video.addEventListener('play', () => {
      if (!selected[1 - index]?.media) return;
      if (other.paused) {
        if (other.readyState) other.currentTime = video.currentTime;
        other.play().then(startSync).catch(error => {
          pauseBoth(); status.textContent = error.message;
        });
      }
      startSync();
    });
    video.addEventListener('seeking', () => {
      if (other.readyState && Math.abs(other.currentTime - video.currentTime) > 0.04)
        other.currentTime = video.currentTime;
    });
    video.addEventListener('volumechange', () => {
      if (!video.muted) { other.muted = true; audio.value = index ? 'right' : 'left'; }
      else if (other.muted) audio.value = 'none';
    });
  });
  populate();
})();
</script>"""


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
    gate = arithmetic_gate(document)
    for row in results:
        row["arithmetic_note"] = attention_note(row)
        if row.get("timed") is True and row["verified"]:
            row["timing_note"] = (
                "Warm worker execution; setup, warmups and client transfer excluded."
            )
    expected = document.get("expected", 16)
    summary = {
        "title": document.get("title", "MiniMax H3 attention comparison"),
        "expected": expected,
        "verified": sum(r["verified"] for r in results),
        "runs": results,
        "quality_approval": "Pending user review",
        "gpu_arithmetic_gate": gate,
        "benchmark_context": (
            "1× RTX 5090 (575 W) · 15-second 1344×768 / 24 fps video · Turbo, 8 steps. "  # noqa: RUF001
            "FP8 model + PDD8 LoRA; same prompt and seed per scene. "
            "Only attention kernel and dense/Sol schedule vary."
        ),
        "timing_boundary": (
            "Warm worker execution; setup, warmups and client transfer excluded. "
            "One timed run per cell."
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
        """<style>
body{font:16px system-ui;margin:24px;background:#121418;color:#edf0f5}
table{border-collapse:collapse;width:100%;max-width:1600px}
td,th{border:1px solid #445;padding:12px;vertical-align:top;text-align:left}
video{width:100%;max-width:560px}a{color:#9ac8ff}
.note{color:#bec6d4;font-size:14px}.bad{color:#ffb1a8}
.compare{max-width:1600px;margin:24px 0;padding:16px;border:1px solid #445;border-radius:8px}
.compare-bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.compare-grid label{display:block;margin-bottom:8px}
select,button{font:inherit;padding:6px;background:#20252d;color:#edf0f5;
border:1px solid #657083;border-radius:4px}
.compare-grid select{max-width:92%}.compare-grid video{max-width:none}
button:disabled{opacity:.45}@media(max-width:700px){.compare-grid{grid-template-columns:1fr}}
</style>""",
        "<h1>" + escape(summary["title"]) + "</h1>",
        "<p>" + escape(summary["benchmark_context"]) + "</p>",
        "<p class=note>" + escape(summary["timing_boundary"]) + "</p>",
        f"<p>{summary['verified']} / {expected} clips verified. "
        "Quality approval awaits your review.</p>",
        "<p class=note>Refresh this page to see newly completed clips.</p>"
        if summary["verified"] < expected
        else "",
        "<p class=note>Checks require all eight denoising steps. "
        "Videos play at their original quality.</p>",
        "<p class=note>"
        + (
            "GPU arithmetic check passed (run "
            + escape(str(gate["run"]))
            + "); visual quality awaits your review."
            if gate["passed"]
            else "GPU arithmetic check not recorded for this Runtime."
        )
        + "</p>",
        *comparison_panel(results, scenes),
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
