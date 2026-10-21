#!/usr/bin/env python3
"""The memory matrix's receipts (design varena-slim §5): one typed record per run from `cozy run show --json`, the
absolute bar, Belady MIN over the recorded page-reference trace, the matched comparison, and the rentals' plan."""

import heapq
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

#: the bar's slack over the trace's Belady MIN (design §5)
MIN_SLACK = 1.05
#: varena's eviction classes, in the order `Stats.evicted` counts them (S1 `memory.rs`)
CLASSES = ("F", "O", "S", "N", "D", "W", "P")
#: the shipping line's rungs (tensorfs 0.12 `varena/cuts.py`): idle bytes, eviction, a cut, a lowered watermark, parking
RUNGS = {1: "idle", 2: "evict", 3: "cut", 4: "lower", 5: "park"}
CANCELED = ("canceled", "cancelled")
#: card -> (rental class, GPUs, catalog VRAM in GB or SKU prefix, GPU memory limit: `off`, `<n>GiB`, or `-` to leave it)
CARDS = {
    "24": ("1x24", 1, 24, "off"), "12": ("1x24", 1, 24, "12GiB"), "8": ("1x24", 1, 24, "8GiB"),
    "6": ("1x24", 1, 24, "6GiB"), "2x8": ("2x24", 2, 24, "8GiB"), "2x12": ("2x24", 2, 24, "12GiB"),
    "real8": ("real8", 1, 8, "-"), "real12": ("real12", 1, 12, "-"), "h100": ("h100", 1, "h100", "-"),
    "wddm": ("wddm", 1, None, "-"),
}


@dataclass(frozen=True, slots=True)
class Step:
    """One step of a GPU's page-reference trace: its capacity for weights and its binds in order."""

    capacity: int
    refs: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class Memory:
    """One GPU call's memory record, as either line reports it."""

    leftover_by_gpu: tuple[tuple[int, int], ...]
    kept: int  # bytes imported by fingerprint, no copy
    tier: int  # bytes copied from host RAM
    store: int  # bytes read from the store (disk) on their way to the GPU
    fill_ns: int  # device time the copies took
    ceiling: float  # GB/s the copies could reach, weighted by bytes; 0 unmeasured
    stall_ns: int
    retries: int
    cuts: int
    parks: int
    evicted: tuple[tuple[str, int], ...]  # bytes by class
    misplaced: tuple[str, ...]  # evictions the bar forbids
    dit_steps: tuple[int, ...]  # transformer bytes copied per step
    min_bytes: int | None  # Belady MIN over the recorded trace; None without one


@dataclass(frozen=True, slots=True)
class Receipt:
    """One run's line in summary.txt."""

    cell: str
    run: str
    status: str
    landed: bool
    line: str
    cap: str
    wall: float
    exec: float | None
    moved: float | None  # GB
    min: float | None  # GB
    bar: str


def count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} is not a byte count: {value!r}")
    return value


def transformer(name):
    return name.rsplit("/", 1)[-1].startswith("transformer")


def belady(trace, held=()):
    """Bytes the optimal offline policy copies: on a miss with the capacity full, evict the page used furthest in the
    future (Belady's MIN; exact for pages of one size). `held` pages are resident at the start, free."""
    flat = [ref for step in trace for ref in step.refs]
    upcoming, first = [math.inf] * len(flat), {}
    for k in range(len(flat) - 1, -1, -1):
        upcoming[k], first[flat[k][0]] = first.get(flat[k][0], math.inf), k
    cache, sizes, heap = {}, {}, []
    used = moved = 0

    def put(page, size, nxt):
        nonlocal used
        used += size - sizes.get(page, 0)
        cache[page], sizes[page] = nxt, size
        heapq.heappush(heap, (-nxt, page))

    def shrink(capacity, keep=None):
        nonlocal used
        kept = []
        while used > capacity and heap:
            nxt, page = heapq.heappop(heap)
            if cache.get(page) != -nxt:
                continue  # stale: the page was used since
            if page == keep:
                kept.append((nxt, page))
                continue
            del cache[page]
            used -= sizes.pop(page)
        for entry in kept:
            heapq.heappush(heap, entry)

    for page, size in held:
        put(page, size, first.get(page, math.inf))
    k = 0
    for step in trace:
        shrink(step.capacity)
        for page, size in step.refs:
            if page not in cache:
                moved += size
            put(page, size, upcoming[k])
            shrink(step.capacity, keep=page)
            k += 1
    return moved


def shipping(memory):
    """Runtime 0.30 + TensorFS 0.12 (`measurements.memory.stats`): one dict over the call's GPUs. It records no page
    trace, so no MIN. Its disk fills count as store bytes, the rest of what it copied to the GPU as tier bytes; a copy
    run's time is its span from first job to last landing, as its own `pct` takes it."""
    stats = memory.get("stats") if isinstance(memory, dict) else None
    if not isinstance(stats, dict) or not stats:
        raise ValueError("no memory stats")
    copies = stats["copies"]
    h2d = [c for c in copies if c["kind"] == "h2d"]
    copied = sum(count(c["bytes"], "copy") for c in h2d)
    disk = min(sum(count(c["bytes"], "disk") for c in copies if c["kind"] == "disk"), copied)
    evicted = {}
    for e in stats["evictions"]:
        evicted[RUNGS[e["rung"]]] = evicted.get(RUNGS[e["rung"]], 0) + count(e["nbytes"], "eviction")
    rows = stats["rows"]
    return Memory(
        leftover_by_gpu=tuple((count(g, "gpu"), count(n, "leftover")) for g, n in stats["leftover_by_gpu"]),
        kept=0, tier=copied - disk, store=disk, fill_ns=round(sum(c["seconds"] for c in h2d) * 1e9),
        ceiling=sum(c["ceiling"] * c["bytes"] for c in h2d) / copied if copied else 0.0,
        stall_ns=sum(count(r["stall_ns"], "stall") for r in rows), retries=count(stats["retries"], "retries"),
        cuts=sum(len(p["cuts"]) for p in stats["peaks"]), parks=sum(e["rung"] == 5 for e in stats["evictions"]),
        evicted=tuple(sorted(evicted.items())), misplaced=(),
        dit_steps=tuple(count(r["paged_in"], "paged_in") + count(r["streamed"], "streamed")
                        for r in rows if transformer(r["model"])),
        min_bytes=None,
    )


def device_trace(stats):
    """One GPU's trace by step. A step's capacity for weights is its measured capacity less its pass room (W less the
    weights it binds); pages first bound as hits or kept imports, not prefetched, were resident when it began."""
    steps = {}
    for ref in stats["trace"]:
        steps.setdefault(count(ref["step"], "step"), []).append(ref)
    trace, held, seen = [], [], set()
    for step, refs in steps.items():
        measured = stats["steps"][step]
        pages = {(r["set"], r["block"]): count(r["bytes"], "ref bytes") for r in refs}
        room = max(count(measured["w"], "w") - sum(pages.values()), 0)
        trace.append(Step(max(count(measured["capacity"], "capacity") - room, 0),
                          tuple((f"{r['set']}#{r['block']}", r["bytes"]) for r in refs)))
        for r in refs:
            page = f"{r['set']}#{r['block']}"
            if page not in seen:
                seen.add(page)
                if r["origin"].lower() in ("hit", "kept") and not r["speculative"]:
                    held.append((page, r["bytes"]))
    return trace, held


def candidate(memory):
    """varena-one (`measurements.memory.varena`): one `{gpu, ceiling, stats}` per GPU, `stats` being S1's `Stats`
    (memory.rs) as serde writes it and `ceiling` the GPU's measured copy ceiling in GB/s."""
    devices = memory.get("varena") if isinstance(memory, dict) else None
    if not isinstance(devices, list) or not devices:
        raise ValueError("no varena stats")
    evicted, misplaced, dit, least, rows = {}, [], [], 0, []
    for device in devices:
        s, gpu = device["stats"], count(device["gpu"], "gpu")
        for name, n in zip(CLASSES, s["evicted"], strict=True):
            evicted[name] = evicted.get(name, 0) + count(n, f"evicted {name}")
        if s["evicted"][CLASSES.index("W")] > s["evicted_lowered"]:
            misplaced.append(f"gpu{gpu}-W-evicted")
        if s["evicted"][CLASSES.index("P")]:
            misplaced.append(f"gpu{gpu}-P-evicted")
        if s["speculative_evicted_d"]:
            misplaced.append(f"gpu{gpu}-speculation-evicted-D")
        per = {}
        for r in s["trace"]:
            if transformer(r["set"]):
                per[r["step"]] = per.get(r["step"], 0) + r["bytes"] * (r["origin"].lower() in ("tier", "store"))
        dit += [per[step] for step in sorted(per)]
        least += belady(*device_trace(s))
        moved = s["moved"]
        rows.append((gpu, s, moved, float(device["ceiling"]), moved["tier"]["bytes"] + moved["store"]["bytes"]))
    copied = sum(r[4] for r in rows)
    return Memory(
        leftover_by_gpu=tuple((gpu, count(s["leftover"], "leftover")) for gpu, s, *_ in rows),
        kept=sum(count(m["kept"]["bytes"], "kept") for _, _, m, *_ in rows),
        tier=sum(count(m["tier"]["bytes"], "tier") for _, _, m, *_ in rows),
        store=sum(count(m["store"]["bytes"], "store") for _, _, m, *_ in rows),
        fill_ns=sum(count(m["tier"]["ns"], "ns") + count(m["store"]["ns"], "ns") for _, _, m, *_ in rows),
        ceiling=sum(c * n for *_, c, n in rows) / copied if copied else 0.0,
        stall_ns=sum(count(s["stall_ns"], "stall") for _, s, *_ in rows),
        retries=sum(count(s["retries"], "retries") for _, s, *_ in rows),
        cuts=sum(count(s["cuts"], "cuts") for _, s, *_ in rows),
        parks=sum(count(s["parks"], "parks") for _, s, *_ in rows),
        evicted=tuple(sorted(evicted.items())), misplaced=tuple(misplaced), dit_steps=tuple(dit), min_bytes=least,
    )


PARSERS = {"shipping": shipping, "candidate": candidate}


def reason_of(run):
    reason = run.get("reason") or ""
    return f"{reason.get('code')}: {reason.get('message')}" if isinstance(reason, dict) else str(reason)


def gb(n):
    return "-" if n is None else f"{n / 1e9:.3f}"


def record(path, cell, number, state, line, cap):
    """Print run `number`'s receipt line (the caller appends it to summary.txt); 1 when it misses the absolute bar,
    named in `bar=`. `cap` is the card's limit in GiB, or `card`."""
    try:
        run = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        run = {}
    run = run if isinstance(run, dict) else {}
    status, reason, bar = run.get("status", "?"), reason_of(run), []
    calls = [c for c in run.get("calls") or [run] if isinstance(c, dict)]
    victim = "-kill-victim" in cell  # the matrix's one labeled deliberate failure (D7); the run after it has the bar
    if victim:
        bar += [] if status in CANCELED else ["not-killed"]
    elif status != "completed" or not (state.startswith("completed ") and state.endswith(" landed")):
        bar.append("incomplete")
    if str(run.get("number")) != str(number):
        bar.append("run-identity")
    duration = run.get("execution_ms")
    duration = duration / 1000 if type(duration) in (int, float) and math.isfinite(duration) and duration > 0 else None
    if duration is None and not victim:
        bar.append("no-execution-time")
    memories, executed = [], set()
    for call in calls:
        gpus = ((call.get("measurements") or {}).get("execution") or {}).get("gpus") or []
        ran = {g["gpu"] for g in gpus if isinstance(g, dict) and type(g.get("gpu")) is int and g["gpu"] >= 0}
        if not ran:
            continue  # a CPU orchestrator
        executed |= ran
        try:
            memories.append((ran, PARSERS[line](call["measurements"].get("memory"))))
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
            bar.append(re.sub(r"\s+", "_", f"no-mem-stats({type(error).__name__}:{str(error)[:60]})"))
    if not executed and not victim:
        bar.append("no-gpu-execution")
    if any(sorted(g for g, _ in m.leftover_by_gpu) != sorted(ran) for ran, m in memories):
        bar.append("leftover-gpu-coverage")
    poisoned = any("poisoned" in reason_of(c) for c in [run, *calls])
    if poisoned:
        bar.append("poisoned")
    m = [m for _, m in memories]
    total = {f: sum(getattr(x, f) for x in m) for f in ("kept", "tier", "store", "fill_ns", "stall_ns", "retries",
                                                         "cuts", "parks")}
    moved = total["tier"] + total["store"]
    leftover = sum(n for x in m for _, n in x.leftover_by_gpu)
    least = None if not m or any(x.min_bytes is None for x in m) else sum(x.min_bytes for x in m)
    dit = [n for x in m for n in x.dit_steps]
    ceiling = sum(x.ceiling * (x.tier + x.store) for x in m) / moved if moved else 0.0
    rate = moved / total["fill_ns"] if total["fill_ns"] else 0.0  # bytes per ns: GB/s
    evicted = {}
    for x in m:
        for name, n in x.evicted:
            evicted[name] = evicted.get(name, 0) + n
    if m and not victim:
        bar += ["leftover"] * bool(leftover) + ["stall>copy"] * (total["stall_ns"] > total["fill_ns"])
        bar += [b for x in m for b in x.misplaced]
        if least is not None and moved > MIN_SLACK * least:
            bar.append("moved>1.05xMIN")
        if "-warm" in cell:
            bar += ["warm-oom-retries"] * bool(total["retries"]) + ["store-in-warm"] * bool(total["store"])
            if "-h3ref5-warm" in cell and cap == "24":  # D7: H3 5 s fits a 24 GB card, so it reads nothing per step
                bar += ["no-dit-step-stats"] if not dit else ["dit-reads-per-step"] * bool(max(dit))
    bar = list(dict.fromkeys(bar))
    wall = max((run.get("settled_at_ms") or 0) - (run.get("created_at_ms") or 0), 0) / 1000
    some = (lambda v: v) if m else (lambda v: None)
    pct = lambda a, b: f"{100 * a / b:.1f}" if a and b else "-"
    fields = {
        "line": line, "cap": cap, "wall": f"{wall:.0f}s", "exec": "-" if duration is None else f"{duration:.3f}s",
        "moved": f"{gb(some(moved))}GB", "store": f"{gb(some(total['store']))}GB",
        "tier": f"{gb(some(total['tier']))}GB", "kept": f"{gb(some(total['kept']))}GB",
        "ram_hit": f"{pct(total['tier'], moved)}%", "fill": f"{rate:.2f}GB/s", "fill_pct": f"{pct(rate, ceiling)}%",
        "stall": f"{total['stall_ns'] / 1e9:.3f}s", "copy": f"{total['fill_ns'] / 1e9:.3f}s",
        "oom_retries": some(total["retries"]), "cuts": some(total["cuts"]), "parks": some(total["parks"]),
        "evicted": ",".join(f"{k}:{v / 1e9:.2f}" for k, v in sorted(evicted.items())) or "-",
        "leftover": some(leftover), "poisoned": int(poisoned), "min": f"{gb(least)}GB",
        "dit_step_max": f"{max(dit) if dit else '-'}B", "bar": ",".join(bar) or "ok",
    }
    print(f"{cell} run={number} {state} " + " ".join(f"{k}={'-' if v is None else v}" for k, v in fields.items())
          + f" reason={reason[:300]}")
    return int(bool(bar))


def number(text):
    try:
        value = float(text.rstrip("sGB"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def receipts(path):
    """summary.txt's receipt lines, in order."""
    out = []
    for text in Path(path).read_text().splitlines():
        match = re.match(r"(\S+) run=(\d+) (\S+) (\S+)", text)
        if not match or " line=" not in text:
            continue
        f = dict(re.findall(r" (\w+)=(\S*)", text.split(" reason=")[0]))
        out.append(Receipt(match[1], match[2], match[3], match[4] == "landed", f.get("line", "?"), f.get("cap", "?"),
                           number(f.get("wall", "0s")) or 0.0, number(f.get("exec", "-")), number(f.get("moved", "-")),
                           number(f.get("min", "-")), f.get("bar", "?")))
    return out


def compare(summary, lines):
    """The matched comparison. Every receipt of the line under test (the candidate, or the only line) must meet the
    absolute bar; with both lines, each candidate cell's median execution time and bytes moved must be <= the
    shipping line's on the same rental (warm cells run 3x, `.k`). Prints one line per cell; returns the failures."""
    rows = receipts(summary)
    tested = "candidate" if "candidate" in lines and len(lines) > 1 else lines[0]
    failures = sum(r.bar != "ok" for r in rows if r.line == tested)
    for r in rows:
        if r.bar != "ok":
            print(f"{r.cell} #{r.run} {r.line}: {'FAIL' if r.line == tested else 'shipping misses'} {r.bar}")
    worse = 0
    if len(lines) > 1:
        cells = {}
        for r in rows:
            if "-kill-victim" not in r.cell:
                cells.setdefault(r.cell.split(".")[0], {}).setdefault(r.line, []).append(r)
        for key, by in cells.items():
            old, new = by.get("shipping", []), by.get("candidate", [])
            if not old or not new or any(r.bar != "ok" for r in old):
                print(f"{key}: not compared ({'no shipping run' if not old else 'no candidate run' if not new else 'shipping missed its bar'})")
                continue
            say = []
            for field in ("exec", "moved"):
                a = [getattr(r, field) for r in old if getattr(r, field) is not None]
                b = [getattr(r, field) for r in new if getattr(r, field) is not None]
                if not a or not b:
                    say.append(f"{field} unmeasured")
                    worse += field == "exec" and bool(a)
                    continue
                x, y = statistics.median(a), statistics.median(b)
                say.append(f"{field} {x:.3f} -> {y:.3f}{' WORSE' if y > x else ''}")
                worse += y > x
            least = [r.min for r in new if r.min is not None]
            print(f"{key}: {'; '.join(say)}{f'; MIN {statistics.median(least):.3f} GB' if least else ''}")
    print(f"memory-matrix: {failures} {tested} runs below the absolute bar, {worse} worse than the shipping line")
    return min(failures + worse, 255)


def rental(catalog, card):
    """`<class> <gpus> <limit> <sku>…`: the card's rental, its SKUs with that many GPUs cheapest first (the catalog's
    expected price: an hour plus a 100 GB download)."""
    cls, gpus, vram, limit = CARDS[card]
    found = []
    for sku in json.loads(Path(catalog).read_text()):
        if vram is None or (sku["vram_gb"] != vram if type(vram) is int else not sku["name"].startswith(vram)):
            continue
        for width in sku["widths"]:
            if width["accelerator_count"] == gpus:
                found.append((width["expected_price"]["total_usd_micros"], sku["name"]))
    return f"{cls} {gpus} {limit} " + " ".join(name for _, name in sorted(found))


def plan(catalog, lines, walls, wanted):
    """Price the plan before renting: per rental, the cheapest SKU and the measured wall of each `card=group,…` it runs,
    from earlier summaries (`walls`, newest last); unmeasured groups are listed, so the total is a floor."""
    measured = {}
    for path in walls:
        for r in receipts(path):
            card, _, rest = r.cell.partition("-")
            group = rest.split(".")[0].split("-")[0]
            measured.setdefault((card, group, path), {}).setdefault(r.line, 0.0)
            measured[(card, group, path)][r.line] += r.wall
    def wall(card, group):
        found = [v for (c, g, _), v in measured.items() if (c, g) == (card, group)]
        return found[-1] if found else None
    prices = {}
    for sku in json.loads(Path(catalog).read_text()):
        for width in sku["widths"]:
            prices[(sku["name"], width["accelerator_count"])] = width
    rentals = {}
    for spec in wanted:
        card, _, groups = spec.partition("=")
        rentals.setdefault(CARDS[card][0], []).append((card, groups.split(",")))
    total_cost = longest = 0.0
    for cls, cards in rentals.items():
        name = rental(catalog, cards[0][0]).split()
        hours, missing = 0.0, []
        for card, groups in cards:
            for group in groups:
                found = wall(card, group)
                if found is None:
                    missing.append(f"{card}:{group}")
                    continue
                per = [found.get(line) for line in lines]
                hours += sum(v if v is not None else max(found.values()) for v in per) / 3600
        if cls == "wddm":
            print(f"wddm: a machine of yours, no rental; {hours:.2f} h measured; unmeasured {' '.join(missing) or '-'}")
            continue
        if len(name) < 4:
            print(f"{cls}: no SKU in the catalog with {name[1]} GPUs of that card")
            continue
        width = prices[(name[3], int(name[1]))]
        hourly = (width["price_usd_micros_per_hour"] + width["storage_usd_micros_per_hour"]) / 1e6
        cost = hourly * hours + width["expected_price"].get("download_usd_micros", 0) / 1e6
        total_cost, longest = total_cost + cost, max(longest, hours)
        print(f"{cls} ({' '.join(c for c, _ in cards)}): {name[3]} x{name[1]} ${hourly:.3f}/h; measured {hours:.2f} h, "
              f"${cost:.2f}; unmeasured: {' '.join(missing) or '-'}")
    print(f"plan: at least ${total_cost:.2f} and {longest:.2f} h wall (rentals run at once); unmeasured groups, boots, "
          f"installs and limit changes add to it")


if __name__ == "__main__":
    try:
        verb, args = sys.argv[1], sys.argv[2:]
        if verb == "record":
            result = record(*args)
        elif verb == "compare":
            result = compare(args[0], args[1:])
        elif verb == "rental":
            print(rental(*args))
            result = 0
        elif verb == "plan":
            split = args.index("--")
            plan(args[0], args[1].split(","), args[2:split], args[split + 1:])
            result = 0
        else:
            raise ValueError("use record, compare, rental or plan")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        print(f"memory receipt refused: {error}")
        result = 1
    raise SystemExit(result)
