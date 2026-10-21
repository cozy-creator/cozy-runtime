#!/usr/bin/env python3
"""The memory matrix's actual GPU receipts and valid legacy timing controls."""

import json
import math
import re
import sys
from pathlib import Path


def integer(value):
    return type(value) is int and value >= 0


def cells(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"(\S+) run=(\d+) (\S+) (\S+)", line)
        if match:
            fields = dict(re.findall(r" (\w+)=([^ ]+)", line.split(" modes=")[0]))
            out[match[1]] = dict(fields, run=match[2], status=match[3], landed=match[4] == "landed")
    return out


def seconds(row):
    try:
        value = float(row.get("exec", "").removesuffix("s"))
        return value if math.isfinite(value) and value > 0 else None
    except (ValueError, TypeError, AttributeError):
        return None


def control_reason(row):
    if row is None:
        return "missing control"
    if row.get("status") != "completed" or not row.get("landed"):
        return f"control {row.get('status', '?')} or outputs not landed"
    if row.get("poisoned") != "0":
        return "control poisoned or poison evidence missing"
    if row.get("substitute"):
        return f"control substitute {row['substitute']}"
    # Old controls predate varena stats/read targets, so only their applicable validity bars
    # constrain timing. A completed old control need not meet the candidate's new read bar.
    bars = set(row.get("bar", "").split(","))
    if bars & {"incomplete", "poisoned", "leftover"}:
        return "control failed applicable absolute bar"
    if row.get("leftover", "-") not in ("-", "0"):
        return "control left bytes"
    if seconds(row) is None:
        return "control execution time missing or invalid"
    return ""


def transformer_rows(value, executed):
    """H3's mirrored transformer passes, retaining the producer's physical GPU and step IDs."""
    rows = value.get("rows")
    if not isinstance(rows, list):
        return [], "no-dit-row-stats"
    selected = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            return [], "no-valid-dit-rows"
        if not row["model"].rsplit("/", 1)[-1].startswith("transformer"):
            continue
        if any(not integer(row.get(key)) for key in ("gpu", "step", "paged_in", "streamed")):
            return [], "no-valid-dit-rows"
        selected.append(row)
    passes = {gpu: [] for gpu in executed}
    for row in selected:
        if row["gpu"] not in passes:
            return [], "dit-row-gpu-coverage"
        passes[row["gpu"]].append((row["model"], row["step"]))
    if not passes or any(not sequence for sequence in passes.values()):
        return [], "dit-row-gpu-coverage"
    sequences = [set(sequence) for sequence in passes.values()]
    if any(len(sequence) != len(set(sequence)) for sequence in passes.values()) or any(
        sequence != sequences[0] for sequence in sequences[1:]
    ):
        return [], "dit-row-pass-coverage"
    return selected, ""


def record(path, cell, number, state, substitute, summary):
    bar = []
    try:
        run = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        run = {}
    if not isinstance(run, dict):
        run = {}
    if run.get("status") != "completed" or not (state.startswith("completed ") and state.endswith(" landed")):
        bar.append("incomplete")
    if str(run.get("number")) != str(number):
        bar.append("run-identity")
    duration = run.get("execution_ms")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        bar.append("no-execution-time")
    calls = run.get("calls") or [run]
    gpu_measurements, child_reasons = [], []
    for call in calls:
        if not isinstance(call, dict):
            bar.append("malformed-call")
            continue
        measurements = call.get("measurements") or {}
        execution = measurements.get("execution") or {}
        gpus = execution.get("gpus") or []
        executed = {gpu["gpu"] for gpu in gpus if isinstance(gpu, dict) and integer(gpu.get("gpu")) and
                    integer(gpu.get("start_us")) and gpu["start_us"] > 0 and
                    integer(gpu.get("end_us")) and gpu["end_us"] >= gpu["start_us"]}
        if executed:
            gpu_measurements.append((measurements, executed))
            child_reasons.append(call.get("reason"))
            if call.get("status") not in (None, "completed", "succeeded"):
                bar.append("gpu-call-incomplete")
    if not gpu_measurements:
        bar.append("no-gpu-execution")
    stats, dit_rows = [], []
    for measurements, executed in gpu_measurements:
        memory = measurements.get("memory")
        value = memory.get("stats") if isinstance(memory, dict) else None
        if not isinstance(value, dict) or not value:
            bar.append("no-mem-stats")
            continue
        for field in ("leftover", "paged_in", "retries"):
            if not integer(value.get(field)):
                bar.append(f"no-valid-{field}")
        by_gpu = value.get("leftover_by_gpu")
        if (not isinstance(by_gpu, list) or not by_gpu or
            any(not isinstance(row, list) or len(row) != 2 or not all(integer(n) for n in row) for row in by_gpu)):
            bar.append("no-valid-leftover-by-gpu")
        elif (len({row[0] for row in by_gpu}) != len(by_gpu) or
              executed != {row[0] for row in by_gpu} or
              sum(row[1] for row in by_gpu) != value.get("leftover")):
            bar.append("leftover-gpu-coverage")
        steps = value.get("steps")
        if not isinstance(steps, dict) or any(not isinstance(k, str) or not isinstance(v, list) or
                                             not all(integer(n) for n in v) for k, v in steps.items()):
            bar.append("no-valid-steps")
        # Text encoders and VAE stages may legitimately be hosted on one rank. H3's
        # transformer is mirrored: an aggregate zero cannot stand in for an absent rank.
        if "-h3" in cell:
            rows, bad = transformer_rows(value, executed)
            if bad:
                bar.append(bad)
            dit_rows.extend(rows)
        stats.append(value)
    total = lambda key: sum(s[key] for s in stats if integer(s.get(key)))
    complete_stats = bool(stats) and len(stats) == len(gpu_measurements)
    leftover = total("leftover") if complete_stats and all(integer(s.get("leftover")) for s in stats) else None
    read = total("paged_in") / 1e9 if complete_stats and all(integer(s.get("paged_in")) for s in stats) else None
    retries = total("retries") if complete_stats and all(integer(s.get("retries")) for s in stats) else None
    dit = [n for s in stats for key, per in (s.get("steps") or {}).items()
           if isinstance(key, str) and key.rsplit("/", 1)[-1].startswith("transformer")
           and isinstance(per, list) for n in per if integer(n)]
    reason = run.get("reason") or ""
    if isinstance(reason, dict):
        reason = f"{reason.get('code')}: {reason.get('message')}"
    poisoned = any("poisoned" in str(value) for value in [reason, *child_reasons])
    if poisoned:
        bar.append("poisoned")
    if leftover:
        bar.append("leftover")
    if cell.endswith("-warm"):
        if read is not None and read > 40:
            bar.append("read>40GB")
        if "-h3ref5-warm" in cell:
            if not dit:
                bar.append("no-dit-step-stats")
            elif max(dit) > 0 or any(row["paged_in"] + row["streamed"] > 0 for row in dit_rows):
                bar.append("dit-reads-per-step")
        cold = cells(summary).get(cell[:-5]) if Path(summary).exists() else None
        if cold is None or cold.get("bar") != "ok" or cold.get("status") != "completed" or not cold.get("landed"):
            bar.append("no-valid-cold")
        elif not str(cold.get("oom_retries", "")).isdigit():
            bar.append("no-cold-retries")
        elif retries is not None and retries > int(cold["oom_retries"]):
            bar.append("more-retries-than-cold")
    bar = list(dict.fromkeys(bar))
    wall = max((run.get("settled_at_ms") or 0) - (run.get("created_at_ms") or 0), 0) / 1000
    duration = duration / 1000 if type(duration) in (int, float) and math.isfinite(duration) else 0
    steps = sum(len(per) for s in stats for key, per in (s.get("steps") or {}).items()
                if isinstance(key, str) and key.rsplit("/", 1)[-1].startswith("transformer") and isinstance(per, list))
    read_label = "-" if read is None else f"{read:.3f}"
    per_label = "-" if read is None or not steps else f"{read / steps:.3f}"
    print(f"{cell} run={number} {state} wall={wall:.0f}s exec={duration:.3f}s oom_retries={'-' if retries is None else retries} "
          f"poisoned={int(poisoned)} leftover={'-' if leftover is None else leftover} read={read_label}GB "
          f"steps={steps} read_per_step={per_label}GB dit_step_max={max(dit) if dit else '-'}B "
          f"{f'substitute={substitute} ' if substitute else ''}bar={','.join(bar) or 'ok'} modes={{}} "
          f"mem_stats={json.dumps(stats)} reason={str(reason)[:300]}")
    return int(bool(bar))


def compare(baseline, candidate, cards):
    asked = {value.split(":")[0]: value.split(":")[1].split(",") if ":" in value else [] for value in cards}
    def wanted(cell):
        card, _, name = cell.partition("-")
        name = {"pair-klein": "pair-sdxl", "pair-lm": "pair-qwen"}.get(name, name)
        return card in asked and (not asked[card] or name in asked[card] or name in ("klein-a", "klein-b"))
    old, here = cells(baseline), cells(candidate)
    worse = invalid = 0
    for cell in sorted(c for c in set(old) | set(here) if wanted(c)):
        control, current = old.get(cell), here.get(cell)
        if current is None or current.get("status") != "completed" or not current.get("landed") or current.get("bar") != "ok":
            why = f"FAIL absolute: {current.get('bar', '?') if current else 'no candidate run'}"
            invalid += 1
        elif not cell.endswith("-warm"):
            why = "absolute bar ok (relative timing applies to warm cells)"
        elif current.get("substitute"):
            why = f"not compared: candidate substitute {current['substitute']}; absolute bars remain"
        elif bad := control_reason(control):
            why = f"not compared: {bad}; absolute bars remain"
        elif seconds(current) is None:
            why = "FAIL: candidate execution time missing or invalid"
            invalid += 1
        elif seconds(current) > 1.1 * seconds(control):
            why = f"WORSE: {current['exec']} > 1.1 x {control['exec']}"
            worse += 1
        else:
            why = f"relative timing ok: {current['exec']} <= 1.1 x {control['exec']}"
        old_run = control.get("run") if control else "-"
        old_read = control.get("read", "-") if control else "-"
        new_read = current.get("read", "-") if current else "-"
        print(f"{cell}: control #{old_run}; read {old_read} -> {new_read}; {why}")
    print(f"memory-matrix: {worse} valid-control timing regressions, {invalid} candidate failures")
    return min(worse + invalid, 255)


if __name__ == "__main__":
    try:
        if sys.argv[1] == "record":
            result = record(*sys.argv[2:])
        elif sys.argv[1] == "compare":
            result = compare(sys.argv[2], sys.argv[3], sys.argv[4:])
        else:
            raise ValueError("use record or compare")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"memory receipt refused: {error}")
        result = 1
    raise SystemExit(result)
