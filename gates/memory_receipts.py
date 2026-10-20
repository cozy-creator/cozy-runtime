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


def trace_errors(events):
    """Validate actual placement events, not a simulation of model execution."""
    if not isinstance(events, list) or not events:
        return ["missing-decision-trace"]
    errors, last, attempted, speculative, starved = [], {}, set(), {}, set()
    began, ended, bound = set(), set(), set()
    classes = {"free", "orphan", "speculative", "unused", "declared", "working", "pinned"}
    for event in events:
        if not isinstance(event, dict):
            errors.append("malformed-decision-event")
            continue
        required = ("sequence", "step", "page", "bytes", "capacity", "resident", "orphan_bytes")
        if any(not integer(event.get(field)) for field in required):
            errors.append("invalid-decision-values")
            continue
        identity = event.get("request")
        if not ((isinstance(identity, str) and identity) or integer(identity)):
            errors.append("invalid-decision-request")
            continue
        stream = (event.get("rank", 0), event.get("gpu", 0))
        if any(not integer(value) for value in stream):
            errors.append("invalid-decision-device")
            continue
        sequence = event["sequence"]
        if sequence <= last.get(stream, -1):
            errors.append("unordered-decision-trace")
        last[stream] = sequence
        request = (*stream, identity)
        step = (*request, event["step"])
        kind, reason, page_class = event.get("kind"), event.get("reason"), event.get("class")
        if kind not in {"begin", "bind", "fill", "evict", "end"} or page_class not in classes:
            errors.append("unknown-decision-event")
            continue
        if kind == "begin":
            if request in began:
                errors.append("duplicate-request-begin")
            began.add(request)
        elif kind == "end":
            if request not in began or request in ended:
                errors.append("unmatched-request-end")
            ended.add(request)
        elif request not in began or request in ended:
            errors.append("decision-outside-request")
        if kind == "bind" and event["bytes"] > 0:
            bound.add(request)
        if kind == "evict":
            if page_class == "pinned" or (page_class == "working" and reason != "lower"):
                errors.append("running-page-evicted")
            if reason == "speculation" and page_class in {"declared", "working", "pinned"}:
                errors.append("speculation-evicted-declared-use")
            if event["orphan_bytes"] and page_class in {"speculative", "unused", "declared"}:
                errors.append("orphan-not-reclaimed-first")
            if page_class == "speculative" and reason != "speculation":
                starved.add(step)
        if kind == "fill" and reason == "speculation":
            key = (*step, event["page"])
            if key in attempted:
                errors.append("repeated-speculative-fetch")
            attempted.add(key)
            if step in starved:
                errors.append("speculation-after-starvation")
            budget = event.get("speculative_budget")
            if not integer(budget) or not integer(event.get("working_set")):
                errors.append("unmeasured-speculation")
            else:
                speculative[step] = speculative.get(step, 0) + event["bytes"]
                if speculative[step] > budget:
                    errors.append("speculation-exceeded-surplus")
    if began != ended:
        errors.append("unfinished-decision-trace")
    if began - bound:
        errors.append("missing-page-bind-evidence")
    return list(dict.fromkeys(errors))


def record(path, cell, number, state, substitute, summary, slim=""):
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
            if call.get("status") not in (None, "completed"):
                bar.append("gpu-call-incomplete")
    if not gpu_measurements:
        bar.append("no-gpu-execution")
    stats = []
    for measurements, executed in gpu_measurements:
        memory = measurements.get("memory")
        if slim:
            events = memory.get("trace") if isinstance(memory, dict) else None
            bar.extend(trace_errors(events))
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
        if not slim and read is not None and read > 40:
            bar.append("read>40GB")
        if "-h3ref5-warm" in cell:
            if not dit:
                bar.append("no-dit-step-stats")
            elif max(dit) > 0:
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


def compare(baseline, candidate, cards, strict=False):
    asked = {value.split(":")[0]: value.split(":")[1].split(",") if ":" in value else [] for value in cards}
    def wanted(cell):
        card, _, name = cell.partition("-")
        name = {"pair-klein": "pair-sdxl", "pair-lm": "pair-qwen"}.get(name, name)
        return card in asked and (not asked[card] or name in asked[card] or name in ("klein-a", "klein-b"))
    old, here = cells(baseline), cells(candidate)
    worse = invalid = 0
    selected = sorted(c for c in set(old) | set(here) if wanted(c))
    if strict and not selected:
        print("FAIL: no selected baseline or candidate cells")
        invalid += 1
    for cell in selected:
        control, current = old.get(cell), here.get(cell)
        if current is None or current.get("status") != "completed" or not current.get("landed") or current.get("bar") != "ok":
            why = f"FAIL absolute: {current.get('bar', '?') if current else 'no candidate run'}"
            invalid += 1
        elif not strict and not cell.endswith("-warm"):
            why = "absolute bar ok (relative timing applies to warm cells)"
        elif current.get("substitute"):
            why = f"{'FAIL' if strict else 'not compared'}: candidate substitute {current['substitute']}; absolute bars remain"
            invalid += int(strict)
        elif bad := control_reason(control):
            why = f"{'FAIL' if strict else 'not compared'}: {bad}; absolute bars remain"
            invalid += int(strict)
        elif seconds(current) is None:
            why = "FAIL: candidate execution time missing or invalid"
            invalid += 1
        elif seconds(current) > (1.0 if strict else 1.1) * seconds(control):
            why = f"WORSE: {current['exec']} > {1.0 if strict else 1.1} x {control['exec']}"
            worse += 1
        else:
            why = f"relative timing ok: {current['exec']} <= {1.0 if strict else 1.1} x {control['exec']}"
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
            cards = [arg for arg in sys.argv[4:] if arg != "--strict"]
            result = compare(sys.argv[2], sys.argv[3], cards, "--strict" in sys.argv[4:])
        else:
            raise ValueError("use record or compare")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"memory receipt refused: {error}")
        result = 1
    raise SystemExit(result)
