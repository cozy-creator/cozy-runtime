#!/usr/bin/env python3
"""The memory matrix's actual GPU receipts and valid legacy timing controls."""

import json
import math
import re
import sys
from pathlib import Path


def integer(value):
    return type(value) is int and value >= 0


def finite(value, *, positive=False):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


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
    if bars & {"incomplete", "poisoned", "leftover", "no-two-gpu-h3-execution"}:
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
        if kind not in {"begin", "bind", "fill", "evict", "lower", "end"} or page_class not in classes:
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
        if kind == "lower":
            starved.add(step)
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


def window_lower_bound(proof, events):
    """H2D lower bound over partitions at actual step/pass boundaries, never OPT.

    Every later window can start with at most capacity payload bytes resident. Its
    distinct demand payload in excess of that must move. The first window instead
    uses the complete initial resident identities. Maximizing the sum over disjoint
    windows stays a lower bound; restricting boundaries makes it cheaper and weaker.
    """
    if (not isinstance(proof, dict) or type(proof.get("version")) is not int or proof["version"] != 1
            or proof.get("complete") is not True or proof.get("invalid") != []
            or not integer(proof.get("capacity"))):
        raise ValueError("incomplete-transfer-proof")
    pages, content_sizes = {}, {}
    if not isinstance(proof.get("pages"), list) or not isinstance(proof.get("initial"), list):
        raise ValueError("invalid-transfer-catalog")
    for page in proof["pages"]:
        if (not isinstance(page, dict) or not integer(page.get("page"))
                or page["page"] in pages or not isinstance(page.get("content"), str)
                or not page["content"] or not integer(page.get("payload"))
                or not integer(page.get("mapped")) or page["payload"] > page["mapped"]
                or page["mapped"] > 2**64 - 1):
            raise ValueError("invalid-transfer-catalog")
        content, size = page["content"], page["payload"]
        if content in content_sizes and content_sizes[content] != size:
            raise ValueError("inconsistent-content-payload")
        pages[page["page"]] = content
        content_sizes[content] = size
    if (any(not integer(page) or page not in pages for page in proof["initial"])
            or len(set(proof["initial"])) != len(proof["initial"])):
        raise ValueError("invalid-initial-residency")
    initial = {pages[page] for page in proof["initial"]}
    groups, previous = [], None
    for event in events:
        if event["kind"] != "bind":
            continue
        if not integer(event.get("pass_index")) or event["page"] not in pages:
            raise ValueError("unknown-demand-content")
        boundary = (event["step"], event["pass_index"])
        if boundary != previous:
            groups.append(set())
            previous = boundary
        groups[-1].add(pages[event["page"]])
    # Limit work by merging adjacent groups, never by dropping references. Any
    # partition over the retained boundaries is still a sound lower bound.
    if len(groups) > 512:
        width = math.ceil(len(groups) / 512)
        groups = [set().union(*groups[i:i + width]) for i in range(0, len(groups), width)]
    best = [0] * (len(groups) + 1)
    capacity = max([proof["capacity"], *(event["capacity"] for event in events)])
    for end in range(1, len(groups) + 1):
        seen, total, missing = set(), 0, 0
        for start in range(end - 1, -1, -1):
            for content in groups[start] - seen:
                size = content_sizes[content]
                total += size
                missing += size if content not in initial else 0
                seen.add(content)
            cost = missing if start == 0 else max(total - capacity, 0)
            best[end] = max(best[end], best[start] + cost)
    compulsory = sum(content_sizes[content] for content in set().union(*groups) - initial) if groups else 0
    return best[-1], compulsory


def transfer_bound(stats, events, executed, by_gpu, required):
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        return ["invalid-transfer-proof-trace"], None, "unproved"
    proofs = stats.get("proofs")
    if not isinstance(proofs, list) or not proofs:
        return ["missing-transfer-proof"], None, "unproved"
    keys, identities, total, compulsory, scoped = set(), set(), 0, 0, True
    for proof in proofs:
        if (not isinstance(proof, dict) or not integer(proof.get("gpu"))
                or proof["gpu"] not in executed or not integer(proof.get("rank"))
                or not isinstance(proof.get("request"), str) or not proof["request"]):
            return ["invalid-transfer-proof-identity"], None, "unproved"
        key = (proof["gpu"], proof["rank"], proof["request"])
        identities.add(key)
        if proof["gpu"] in keys:
            return ["duplicate-transfer-proof-gpu"], None, "unproved"
        keys.add(proof["gpu"])
        selected = [event for event in events if (event.get("gpu"), event.get("rank"), event.get("request")) == key]
        if (not selected or selected[0].get("kind") != "begin"
                or selected[-1].get("kind") != "end"):
            return ["transfer-proof-trace-coverage"], None, "unproved"
        if proof.get("isolated") is not True:
            scoped = False
            continue
        try:
            bound, distinct = window_lower_bound(proof, selected)
        except (ValueError, KeyError, TypeError) as error:
            return [str(error)], None, "unproved"
        total += bound
        compulsory += distinct
        if 100 * by_gpu.get(proof["gpu"], 0) > 105 * bound:
            return ["h2d-window-bound-inconclusive"], total, "inconclusive"
    if keys != executed:
        return ["transfer-proof-gpu-coverage"], None, "unproved"
    if identities != {(event.get("gpu"), event.get("rank"), event.get("request"))
                       for event in events if event.get("kind") == "begin"}:
        return ["transfer-proof-trace-coverage"], None, "unproved"
    if not scoped:
        return (["transfer-proof-not-isolated"] if required else []), None, "shared-scope"
    return [], total, f"sufficient:compulsory={compulsory}B"


def movement_evidence(stats, events, executed, require_bound=True):
    """Check the existing Copy wire records without inventing missing measurements.

    Copy.ceiling/gbps are rounded to two decimals and pct to one by TensorFS. Cached
    disk reads deliberately carry no physical-disk ceiling. The trace carries mapped
    spans. The separate versioned catalog and complete demand trace permit a scoped
    H2D window lower bound; it is never described as the full offline minimum.
    """
    errors = []
    facts = {"h2d_bytes": 0, "uncached_disk_bytes": 0, "cached_bytes": 0,
             "percentages": [], "minimum_bytes": None, "bound_bytes": None,
             "bound_status": "unproved"}
    by_gpu = {}
    copies = stats.get("copies")
    if not isinstance(copies, list):
        return ["missing-copy-evidence"], facts
    valid = True
    for row in copies:
        if not isinstance(row, dict):
            errors.append("malformed-copy-evidence")
            valid = False
            continue
        kind, gpu, size = row.get("kind"), row.get("gpu"), row.get("bytes")
        if (not isinstance(kind, str) or kind not in {"h2d", "disk", "zero"}
                or type(gpu) is not int or not integer(size)
                or size > 2**64 - 1
                or (kind != "disk" and gpu not in executed) or (kind == "disk" and gpu != -1)):
            errors.append("invalid-copy-identity")
            valid = False
            continue
        cached, staged = row.get("cached"), row.get("staged")
        if (not integer(cached) or not integer(staged) or cached > size or staged > size
                or (kind != "disk" and cached) or (kind == "zero" and staged)):
            errors.append("invalid-copy-byte-counts")
            valid = False
            continue
        seconds, gbps, ceiling, pct = (row.get(key) for key in ("seconds", "gbps", "ceiling", "pct"))
        if not finite(seconds, positive=bool(size)) or not finite(gbps) or not finite(ceiling):
            errors.append("invalid-copy-measurement")
            valid = False
            continue
        if kind == "h2d":
            facts["h2d_bytes"] += size
            by_gpu[gpu] = by_gpu.get(gpu, 0) + size
        elif kind == "disk":
            facts["cached_bytes"] += cached
            facts["uncached_disk_bytes"] += size - cached
        if not size:
            continue
        rate = size / seconds / 1e9
        if not math.isfinite(rate) or abs(gbps - rate) > 0.005001:
            errors.append("copy-rate-mismatch")
        if kind == "zero":
            # Padding initialization is a device memset, not host→device traffic.
            if ceiling != 0 or pct is not None:
                errors.append("zero-initialization-has-transfer-ceiling")
            continue
        if kind == "disk" and cached == size:
            if ceiling != 0 or pct is not None:
                errors.append("cached-disk-has-hardware-ceiling")
            continue
        if kind == "disk" and cached:
            errors.append("mixed-disk-copy-unmeasured")
            continue
        if ceiling <= 0 or not finite(pct) or pct > 100:
            errors.append(f"unmeasured-{kind}-ceiling")
            continue
        # Compare intervals implied by the producer's decimal rounding, rather than
        # silently interpreting rounded zeroes as measured rates or adding a speed bar.
        low = 100 * rate / (ceiling + 0.005)
        high = 100 * rate / max(ceiling - 0.005, sys.float_info.min)
        if not low - 0.050001 <= pct <= high + 0.050001:
            errors.append("copy-percentage-mismatch")
        facts["percentages"].append(f"{kind}@{gpu}:{pct:.1f}%")
    paged = stats.get("paged_in")
    if integer(paged) and valid and facts["h2d_bytes"] != paged:
        # The producer currently drains a process-wide log. It must not be mistaken
        # for complete per-request evidence when concurrent requests split that log.
        errors.append("copy-request-accounting-mismatch")
    fills = (isinstance(events, list) and any(isinstance(event, dict)
             and event.get("kind") == "fill" and event.get("bytes", 0) != 0 for event in events))
    if (integer(paged) and valid and not errors and paged == 0 and not facts["h2d_bytes"]
            and not facts["uncached_disk_bytes"] and not fills):
        facts["minimum_bytes"] = 0
    if stats.get("proofs") is not None:
        proof_errors, bound, status = transfer_bound(stats, events, executed, by_gpu, require_bound)
        errors.extend(proof_errors)
        facts.update(bound_bytes=bound, bound_status=status)
    elif facts["minimum_bytes"] is None:
        errors.append("missing-transfer-minimum-evidence")
    return list(dict.fromkeys(errors)), facts


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
    group_required = bool(re.match(r"2x[^-]+-h3(?:-|$)", cell))
    group_executed = False
    for call in calls:
        if not isinstance(call, dict):
            bar.append("malformed-call")
            continue
        measurements = call.get("measurements") or {}
        execution = measurements.get("execution") or {}
        gpus = execution.get("gpus") or []
        legacy = not slim and not gpus and isinstance(execution.get("ranks"), list)
        if legacy:
            # Runtime 0.27.4 controls report actual CUDA ranks, not the new GPU wire.
            gpus = [{**rank, "gpu": rank.get("ordinal")} for rank in execution["ranks"]
                    if isinstance(rank, dict) and integer(rank.get("pid")) and rank["pid"] > 0
                    and isinstance(rank.get("uuid"), str) and rank["uuid"]]
        executed = {gpu["gpu"] for gpu in gpus if isinstance(gpu, dict) and integer(gpu.get("gpu")) and
                    integer(gpu.get("start_us")) and gpu["start_us"] > 0 and
                    integer(gpu.get("end_us")) and gpu["end_us"] >= gpu["start_us"]}
        if call.get("memoized") is True:
            executed = set()
        if executed:
            gpu_measurements.append((measurements, executed, legacy))
            function = str(call.get("function") or call.get("target") or "").rsplit("/", 1)[-1].split("@", 1)[0].replace("-", "_")
            h3 = function in {"generate_ref2va_turbo", "generate_fl2va_turbo",
                              "ref2va_turbo", "fl2va_turbo", "long_form"}
            if h3 and len(executed) == 2:
                group_executed = True
                if group_required and slim:
                    memory = measurements.get("memory") or {}
                    value = memory.get("stats") if isinstance(memory, dict) else None
                    proofs = value.get("proofs") if isinstance(value, dict) else None
                    if (not isinstance(proofs, list) or len(proofs) != 2
                            or any(not isinstance(proof, dict) for proof in proofs)
                            or {proof.get("gpu") for proof in proofs} != executed
                            or {proof.get("rank") for proof in proofs} != {0, 1}
                            or any(proof.get("complete") is not True for proof in proofs)):
                        bar.append("incomplete-h3-rank-proof")
            child_reasons.append(call.get("reason"))
            if call.get("status") not in (None, "completed"):
                bar.append("gpu-call-incomplete")
    if not gpu_measurements:
        bar.append("no-gpu-execution")
    if group_required and not group_executed:
        bar.append("no-two-gpu-h3-execution")
    stats, movements = [], []
    for measurements, executed, legacy in gpu_measurements:
        memory = measurements.get("memory")
        if slim:
            events = memory.get("trace") if isinstance(memory, dict) else None
            bar.extend(trace_errors(events))
        value = memory.get("stats") if isinstance(memory, dict) else None
        if legacy and isinstance(memory, dict) and any(
                key in memory for key in ("paged_in_bytes", "oom_retries", "weights_bytes")):
            value = {"legacy_control": True, "legacy_memory": memory}
            for target, source in (("paged_in", "paged_in_bytes"), ("retries", "oom_retries")):
                if integer(memory.get(source)):
                    value[target] = memory[source]
            # Flat weights/reserved/process bytes do not measure request leftovers,
            # per-GPU cleanup, per-step reads or the new native decision proof.
            stats.append(value)
            continue
        if not isinstance(value, dict) or not value:
            bar.append("no-mem-stats")
            continue
        if slim:
            errors, movement = movement_evidence(value, events, executed, "pair-" not in cell)
            bar.extend(errors)
            movements.append(movement)
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
    legacy_control = any(row.get("legacy_control") for row in stats)
    if cell.endswith("-warm") and not legacy_control:
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
    movement_label = ""
    if slim:
        known = bool(movements) and all(row["minimum_bytes"] == 0 for row in movements)
        percentages = ",".join(value for row in movements for value in row["percentages"])
        movement_label = (f"transfer_minimum={'0B' if known else 'unknown'} "
                          f"h2d_window_lower_bound={sum(row['bound_bytes'] or 0 for row in movements)}B "
                          f"h2d_bound_status={','.join(row['bound_status'] for row in movements) or 'unproved'} "
                          f"copy_ceiling_pct={percentages or '-'} ")
    print(f"{cell} run={number} {state} wall={wall:.0f}s exec={duration:.3f}s oom_retries={'-' if retries is None else retries} "
          f"poisoned={int(poisoned)} leftover={'-' if leftover is None else leftover} read={read_label}GB "
          f"steps={steps} read_per_step={per_label}GB dit_step_max={max(dit) if dit else '-'}B "
          f"{f'substitute={substitute} ' if substitute else ''}{'legacy_control=1 ' if legacy_control else ''}{movement_label}bar={','.join(bar) or 'ok'} modes={{}} "
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
