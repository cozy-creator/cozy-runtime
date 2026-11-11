#!/usr/bin/env python3
"""The attention table (xs-171): measured, generated, checked. Never edited by hand.

    attention_table.py generate [--partial]   measurements/attention.json + rules.toml -> internal/attention_table.json
    attention_table.py check [--partial]      the shipped table regenerates byte for byte, every listed kernel's build
                                              is the one measured, and every kernel a class runs was measured
    attention_table.py identities [--src DIR] each kernel's build identity in the tree at DIR (default: this one)

Inputs live beside the Runtime: `measurements/attention.json` (release-gate/suites/attention/collect.py writes it from
the attention suite's gate runs) and `measurements/rules.toml` (bars, noise, edges, tie order, exclusions). The
algorithm is measurement-design §4.2: deterministic, no clock, no randomness. Run it in the project's environment
(`uv run python scripts/attention_table.py ...`): it reads the Runtime's kernels from the source tree, and needs no
torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = ROOT / "measurements" / "attention.json"
RULES = ROOT / "measurements" / "rules.toml"
TABLE = ROOT / "src" / "cozy_runtime" / "internal" / "attention_table.json"
SCHEMA = "cozy.attention-table/1"
MEASURED = "cozy.attention-measurements/1"
#: The `report` of the one hand-made table, R3's transcription of the cells before any measurement.
TRANSCRIBED = "transcribed"
#: The machine every pod is: a platform wheel's pin for another CPU is not what was measured.
_MACHINE = "x86_64"

type Json = Any  # decoded JSON: the measurement file's shape is collect.py's, read by key


# ---------------------------------------------------------------------------- build identity


def _digest(parts: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:12]


def _files(src: Path, paths: Iterable[Path]) -> Iterable[bytes]:
    for path in sorted(paths):
        yield str(path.relative_to(src.parent)).encode()
        yield path.read_bytes()


def identities(src: Path) -> dict[str, str]:
    """Each Runtime kernel's build identity in the tree whose `src` is SRC: `<recipe>@<sources digest>+<code digest>`.
    The recipe part is its kernel_sources recipe and pinned sources (torch's own kernels: `torch`); the code part the
    Runtime modules that implement its forward and its Triton sets, a sparse kernel's routing and dense rows, and the
    native sources of a kernel the wheel carries. A torch or CUDA bump comes with the worker image, which the wheel does
    not name: the measurements record both per class."""
    sys.path.insert(0, str(src))
    from cozy_runtime.internal import attention, kernel_sources, prebuilt

    package, native = src / "cozy_runtime", src.parent / "kernels"
    pins = {p.name: p for p in kernel_sources._PINS if p.machine in ("", _MACHINE)}

    def recipe(name: str) -> str:
        if not name:
            return "torch"
        if name == attention.PREBUILT:
            return name
        r = kernel_sources.RECIPES[name]
        if r.kind == "python":
            build = set(kernel_sources._BUILD_DEPS).union(*(x.build_deps for x in kernel_sources.RECIPES.values()))
            build.add(kernel_sources.KITCHEN_WHEEL)
            used = sorted(n for n, p in pins.items() if n not in build and (p.url.endswith(".whl") or p.package))
        else:
            used = sorted({*r.sources, *r.build_deps} | ({kernel_sources.KITCHEN_WHEEL} & pins.keys() if name == "comfy-kitchen" else set()))
        keyed = {k: v for k, v in asdict(r).items() if k not in ("extensions", "unit_bytes")}
        body = json.dumps({"recipe": keyed, "pins": [asdict(pins[n]) for n in used]}, sort_keys=True, default=list)
        return f"{name}@{_digest([body.encode()])}"

    def module(name: str) -> list[Path]:
        if not name.startswith("cozy_runtime."):
            return []
        base = package.joinpath(*name.split(".")[1:])
        return [p for p in (base.with_suffix(".py"), base.with_suffix(".pyi")) if p.exists()] or sorted(base.rglob("*.py"))

    def code(c: Any, seen: set[str]) -> set[Path]:
        if c.name in seen:
            return set()
        seen.add(c.name)
        found = {*module(c.local.forward.__module__), *module(c.kernels), *module(c.import_name)}
        if c.sparse:
            found |= {*module("cozy_runtime.internal.attention_sol"), *module("cozy_runtime.internal.attention_sol_fa4")}
        for dense in c.dense:
            found |= code(attention.BY_NAME[dense], seen)
        wheel = prebuilt.ATTENTION.get(c.name)
        if wheel:
            stem = wheel.strip("_")
            found |= {native / "csrc" / f"{stem}.cpp", *(native / "csrc" / stem).rglob("*"), native / "CMakeLists.txt",
                      native / "toolchain.env"}  # fmt: skip
        return {p for p in found if p.is_file()}

    return {
        c.name: f"{recipe(c.artifact)}+{_digest(_files(src, code(c, set())))}"
        for c in attention.KERNELS
    }


# ---------------------------------------------------------------------------- rules


@dataclass(frozen=True, slots=True)
class Rules:
    bars: dict[int, float]
    sparse_vs_fp32: float
    sparse_floor: float
    trivial: float
    noise_floor: float
    noise_spread: float
    edges: list[int]
    max_edges: int
    preference: list[str]
    #: kernel -> {"classes": [...] | ["*"], "reason": str}
    excluded: dict[str, dict[str, Any]]

    @staticmethod
    def read(path: Path) -> Rules:
        raw = tomllib.loads(path.read_text())
        bars = raw["bars"]
        return Rules(
            bars={int(k): float(v) for k, v in bars["level"].items()},
            sparse_vs_fp32=float(bars["sparse_vs_fp32"]),
            sparse_floor=float(bars["sparse_floor"]),
            trivial=float(bars["trivial"]),
            noise_floor=float(raw["noise"]["floor"]),
            noise_spread=float(raw["noise"]["spread"]),
            edges=[int(e) for e in raw["edges"]["candidates"]],
            max_edges=int(raw["edges"]["max"]),
            preference=list(raw["preference"]["order"]),
            excluded=dict(raw.get("excluded", {})),
        )

    def excludes(self, kernel: str, klass: str) -> str:
        found = self.excluded.get(kernel)
        if found is None:
            return ""
        classes = found.get("classes", ["*"])
        return str(found["reason"]) if "*" in classes or klass in classes else ""


# ---------------------------------------------------------------------------- generation


@dataclass(frozen=True, slots=True)
class Kernel:
    name: str
    level: int
    sparse: bool
    dense: tuple[str, ...]


def _kernels(src: Path) -> dict[str, Kernel]:
    sys.path.insert(0, str(src))
    from cozy_runtime.internal import attention

    return {c.name: Kernel(c.name, c.level, c.sparse, c.dense) for c in attention.KERNELS}


def _classes(src: Path) -> list[str]:
    sys.path.insert(0, str(src))
    from cozy_runtime.internal import arch

    return list(arch.CLASSES)


def _passes(name: str, kernel: Kernel, quality: Json, rules: Rules, notes: list[str]) -> str:
    """Empty when KERNEL passes its level's bar where the measurements can judge it, else why not; what they cannot
    judge, or judge only on synthetic operands once captured ones exist, goes to NOTES (the report) instead.

    - Corpus first: where the kernel was measured on captured sites (`kind` "captured"), those decide; synthetic sites
      are notes.
    - A dense kernel is held to its level's bar against FP32 attention and must beat a computation-free answer.
    - A sparse kernel is held against the exact computation of its own route, not dense FP32: the sparse
      approximation is the model's choice (its layout asks for it), and its loss is a note for the bars' owner. The
      route's exact computation is a site's `route` error when the bench records it, else FP32 on a `routable` site,
      where every query's mass sits in one key block, so the route is unambiguous and dense FP32 is that computation.
      A quantized sparse kernel stays within `sparse_vs_fp32` times the 16-bit sparse kernel's error there, once its
      own error is past `sparse_floor` (below it, both are BF16 rounding and their ratio is noise)."""
    if quality is None:
        return "no quality measurement"
    at = [e for e in quality["errors"] if e["level"] <= kernel.level]
    if not at:
        return "no site it would serve was measured"
    if kernel.sparse:  # the sites with an exact computation of the route: recorded, else the routable ones
        decisive = [e for e in at if "route" in e] or [e for e in at if e.get("kind") == "routable"]
        if not decisive:
            return "no site measured against its route's exact computation"
    else:
        decisive = [e for e in at if e.get("kind") == "captured"] or at
    for e in at:
        if not e["finite"]:
            return f"non-finite output on {e['site']}"
        if kernel.sparse and e.get("sparse_calls", 1) < 1:
            return f"{e['site']}: its sparse path did not run"
    for e in at:
        why = _sparse_fault(kernel, e, rules) if kernel.sparse else _dense_fault(kernel, e, rules)
        if why and e in decisive:
            return why
        if why:
            notes.append(f"note: {name}: {kernel.name} on {e.get('kind', 'synthetic')} {why}")
        if kernel.sparse and e.get("kind") == "captured":
            against = f"; the 16-bit route's {e['ref16_fp32']:.4f}, and it against that route {e['ref16']:.4f}" if "ref16" in e else ""
            notes.append(f"note: {name}: {kernel.name} at {e['site']}: the sparse route's loss against dense FP32 "
                         f"{e['worst']:.4f}{against}")  # fmt: skip
    return ""


def _dense_fault(kernel: Kernel, e: Json, rules: Rules) -> str:
    if e["worst"] * rules.trivial > e["trivial"]:
        return f"{e['site']}: {e['worst']:.4f} does not beat a computation-free answer ({e['trivial']:.4f}) by {rules.trivial:g}x"
    if e["worst"] > rules.bars[kernel.level]:
        return f"{e['site']}: worst head {e['worst']:.4f}, over the {kernel.level}-bit bar {rules.bars[kernel.level]:g}"
    return ""


def _sparse_fault(kernel: Kernel, e: Json, rules: Rules) -> str:
    """A sparse kernel at site E against the exact computation of its route: E's `route` error, else FP32 on a
    routable site; nothing to judge where E has neither."""
    route = e.get("route", e["worst"] if e.get("kind") == "routable" else None)
    if route is None:
        return ""
    if route > rules.bars[kernel.level]:
        return f"{e['site']}: {route:.4f} against its route's exact computation, over the {kernel.level}-bit bar {rules.bars[kernel.level]:g}"
    exact16 = e.get("route16")  # the 16-bit sparse kernel against the same computation
    if kernel.level < 16 and exact16 is not None and e.get("kind") != "routable" and route > rules.sparse_floor:
        if route > rules.sparse_vs_fp32 * exact16:
            return f"{e['site']}: {route:.4f} against its route, over {rules.sparse_vs_fp32:g}x the 16-bit sparse kernel's {exact16:.4f}"
    return ""


@dataclass(slots=True)
class Point:
    id: str
    tokens: int
    dim: int
    layout: str
    #: kernel -> ms, SDPA's where the kernel refuses the call (a model's refused call runs on SDPA)
    ms: dict[str, float]
    margin: float


def _points(record: Json, rules: Rules) -> list[Point]:
    """Each measured point's time per kernel. A sparse point times only the sparse kernels and SDPA: a dense kernel there
    takes its time at the dense point of the same shape, scaled by SDPA's ratio between the two."""
    shape = {(p["batch"], p["lq"], p["lk"], p["heads"], p["dim"]): p for p in record["points"] if p["layout"] == "dense"}
    found = []
    for p in record["points"]:
        times = {k: float(t["ms"][0]) for k, t in p["times"].items()}
        if "sdpa" not in times:
            continue
        spread = max((float(t["spread"]) for t in p["times"].values()), default=0.0)
        refused = dict(p["refused"])
        twin = shape.get((p["batch"], p["lq"], p["lk"], p["heads"], p["dim"])) if p["layout"] == "sparse" else None
        if twin is not None and "sdpa" in twin["times"]:
            scale = times["sdpa"] / float(twin["times"]["sdpa"]["ms"][0])
            for k, t in twin["times"].items():
                times.setdefault(k, float(t["ms"][0]) * scale)
            for k, why in twin["refused"].items():
                refused.setdefault(k, why)
        ms = {k: times.get(k, times["sdpa"]) for k in {*times, *refused}}
        found.append(Point(p["id"], min(p["lq"], p["lk"]), p["dim"], p["layout"], ms,
                           max(rules.noise_floor, rules.noise_spread * spread)))  # fmt: skip
    return found


def _bucket(tokens: int, edges: Sequence[int]) -> int:
    return sum(tokens >= e for e in edges)


def _rank(kernels: Sequence[str], points: Sequence[Point], rules: Rules) -> list[tuple[str, float]]:
    """KERNELS by the geometric mean of their time over POINTS relative to the fastest at each; a kernel within the
    noise margin of a faster one is ordered by `rules.preference`, so noise moves nothing."""
    if not points:
        return []
    usable = [k for k in kernels if all(k in p.ms for p in points)]
    score = {
        k: math.exp(sum(math.log(p.ms[k] / min(p.ms[o] for o in usable)) for p in points) / len(points))
        for k in usable
    }
    margin = max(p.margin for p in points)
    prefer = {k: i for i, k in enumerate(rules.preference)}
    ordered = sorted(usable, key=lambda k: (score[k], k))
    ranked: list[tuple[str, float]] = []
    while ordered:
        lead = score[ordered[0]]
        tied = [k for k in ordered if score[k] <= lead * (1 + margin)]
        ranked += [(k, score[k]) for k in sorted(tied, key=lambda k: (prefer.get(k, len(prefer)), score[k], k))]
        ordered = [k for k in ordered if k not in tied]
    return ranked


@dataclass(slots=True)
class Cell:
    klass: str
    layout: str
    level: int
    head_dim: int | None
    #: kernels it may list: passing, at or above its level, no sparse kernel in a dense cell
    eligible: list[str]
    points: list[Point]


def _cells(name: str, record: Json, kernels: Mapping[str, Kernel], rules: Rules, report: list[str]) -> list[Cell]:
    points = _points(record, rules)
    measured = {k for p in points for k in p.ms}
    failed = {k: f"{why} (at {p['id']})" for p in record["points"] for k, why in p.get("failed", {}).items()}
    passing = []
    for k in sorted(measured | failed.keys()):
        if k not in kernels:
            raise SystemExit(f"{name}: {k} was measured and is no Runtime kernel now")
        why = rules.excludes(k, name) or failed.get(k, "") or _passes(name, kernels[k], record["quality"].get(k), rules, report)
        if why:
            report.append(f"{name}: {k} left out: {why}")
        else:
            passing.append(k)
    cells = []
    dims: list[int | None] = [None, *([64] if any(p.dim == 64 for p in points) else [])]
    for layout in ("dense", "sparse"):
        for level in (16, 8, 4):
            for dim in dims if layout == "dense" else [None]:
                eligible = [k for k in passing if kernels[k].level >= level and (layout == "sparse" or not kernels[k].sparse)]
                at = [p for p in points if p.dim == (dim or 128) and p.layout == layout]
                cells.append(Cell(name, layout, level, dim, eligible, at))
    return cells


def _orders(cells: Sequence[Cell], edges: Sequence[int], rules: Rules) -> list[list[list[tuple[str, float]]]]:
    """Each cell's ranking per bucket. A bucket with no points of its own takes the nearest one below that has
    some (above, for the first); a sparse cell's bucket with no sparse points takes its dense cell's ranking."""
    dense = {(c.klass, c.level, c.head_dim): c for c in cells if c.layout == "dense"}
    out: list[list[list[tuple[str, float]]]] = []
    for cell in cells:
        rows = []
        for b in range(len(edges) + 1):
            at = [p for p in cell.points if _bucket(p.tokens, edges) == b]
            if not at and cell.layout == "sparse":
                twin = dense[(cell.klass, cell.level, None)]
                at = [p for p in twin.points if _bucket(p.tokens, edges) == b]
                rows.append(_rank(twin.eligible, at, rules))
            else:
                rows.append(_rank(cell.eligible, at, rules))
        out.append(_filled(rows))
    return out


def _filled(rows: list[list[tuple[str, float]]]) -> list[list[tuple[str, float]]]:
    for b, row in enumerate(rows):
        if not row:
            below = [r for r in rows[:b] if r]
            above = [r for r in rows[b + 1 :] if r]
            rows[b] = below[-1] if below else (above[0] if above else [])
    return rows


def _heads(orders: Sequence[Sequence[Sequence[tuple[str, float]]]], b: int) -> list[str]:
    return [o[b][0][0] if o[b] else "" for o in orders]


def _edges(cells: Sequence[Cell], rules: Rules) -> list[int]:
    """Start from every candidate edge; merge adjacent buckets while every cell's head agrees; past `rules.max_edges`,
    merge the pair that costs the least speed (the summed score of the merged bucket's head over its points)."""
    edges = list(rules.edges)
    while True:
        orders = _orders(cells, edges, rules)
        same = [i for i in range(len(edges)) if _heads(orders, i) == _heads(orders, i + 1)]
        if same:
            del edges[same[0]]
            continue
        if len(edges) <= rules.max_edges:
            return edges
        costs = []
        for i in range(len(edges)):
            merged = edges[:i] + edges[i + 1 :]
            after = _orders(cells, merged, rules)
            costs.append((sum(sum(score for _, score in o[i][:1]) for o in after), i))
        del edges[min(costs)[1]]


def _json_line(value: object) -> str:
    return json.dumps(value, separators=(", ", ": "))


def render(table: Mapping[str, Any]) -> str:
    """The table as review diffs read it: one class, kernel and cell per line."""
    lines = ["{"]
    for key in ("schema", "runtime", "generated", "report", "tokens"):
        lines.append(f'  "{key}": {_json_line(table[key])},')
    for key in ("classes", "kernels"):
        rows = [f"    {_json_line(k)}: {_json_line(v)}" for k, v in sorted(table[key].items())]
        lines.append(f'  "{key}": {{\n' + ",\n".join(rows) + "\n  },")
    rows = [f"    {_json_line(c)}" for c in table["cells"]]
    lines.append('  "cells": [\n' + ",\n".join(rows) + "\n  ]")
    return "\n".join(lines) + "\n}\n"


def generate(measurements: Path, rules_path: Path, src: Path, partial: bool = False) -> tuple[str, list[str]]:
    data = json.loads(measurements.read_text())
    if data.get("schema") != MEASURED:
        raise SystemExit(f"{measurements}: schema {data.get('schema')!r}, not {MEASURED}")
    rules = Rules.read(rules_path)
    kernels = _kernels(src)
    classes = _classes(src)
    measured = data["classes"]
    report: list[str] = []
    unknown = sorted(set(measured) - set(classes))
    missing = sorted(set(classes) - set(measured))
    if unknown:
        raise SystemExit(f"measured classes the Runtime does not name: {unknown}")
    if missing and not partial:
        raise SystemExit(f"classes never measured: {missing} (a round on their reference cards; --partial for one in progress)")
    identity: dict[str, str] = {}
    for name, record in sorted(measured.items()):
        for k, ident in record["kernels"].items():
            if identity.setdefault(k, ident) != ident:
                raise SystemExit(f"{k} was measured as two builds ({identity[k]} and {ident}): re-measure every class that runs it")
    cells: list[Cell] = []
    for name, record in sorted(measured.items()):
        cells += _cells(name, record, kernels, rules, report)
    edges = _edges(cells, rules)
    orders = _orders(cells, edges, rules)
    listed: set[str] = set()
    rows = []
    for cell, order in zip(cells, orders):
        names = [[k for k, _ in bucket] for bucket in order]
        if not all(names):
            raise SystemExit(f"{cell.klass} {cell.layout} {cell.level}: a bucket ranks no kernel")
        listed.update(k for bucket in names for k in bucket)
        rows.append({"class": cell.klass, "layout": cell.layout, "level": cell.level, "head_dim": cell.head_dim, "order": names})
        for b, bucket in enumerate(order):
            top = ", ".join(f"{k} {s:.3f}" for k, s in bucket[:2])
            report.append(f"{cell.klass} {cell.layout} {cell.level} d{cell.head_dim or '*'} {band(edges, b) or 'all'}: {top}")
    heads = {(c.klass, c.level): o[-1][0][0] for c, o in zip(cells, orders) if c.layout == "dense" and c.head_dim is None}
    for name, record in sorted(measured.items()):
        for k in sorted(listed):
            kernel = kernels[k]
            if kernel.sparse and kernel.dense and heads.get((name, kernel.level)) not in (None, kernel.dense[0]):
                report.append(f"{name}: {k}'s dense rows run {kernel.dense[0]}, not the {kernel.level}-bit dense head {heads[(name, kernel.level)]}")
    digest = hashlib.sha256(measurements.read_bytes()).hexdigest()
    table = {
        "schema": SCHEMA,
        "runtime": max((r["runtime"] for r in measured.values()), key=_version),
        "generated": max(r["measured"] for r in measured.values()),
        "report": f"measurements/{measurements.name}@sha256:{digest}",
        "tokens": edges,
        "classes": {n: {"card": r["card"], "sm": r["sm"], "runs": sorted(r["runs"])} for n, r in measured.items()},
        "kernels": {k: v for k, v in identity.items() if k in listed},
        "cells": rows,
    }
    return render(table), report


def band(edges: Sequence[int], b: int) -> str:
    """runtime-design §8's canonical band label: `<2048`, `2048-12287`, `>=65536`; empty for a one-rung ladder."""
    if not edges:
        return ""
    if b == 0:
        return f"<{edges[0]}"
    if b == len(edges):
        return f">={edges[-1]}"
    return f"{edges[b - 1]}-{edges[b] - 1}"


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) if x.isdigit() else 0 for x in v.replace("+", ".").split("."))


# ---------------------------------------------------------------------------- check


def check(measurements: Path, rules_path: Path, table_path: Path, src: Path, partial: bool = False) -> list[str]:
    """Why the shipped table cannot ship, or nothing."""
    shipped = table_path.read_text()
    table = json.loads(shipped)
    fails: list[str] = []
    if table.get("report") == TRANSCRIBED:
        print(f"{table_path.name} is today's cells transcribed by hand (xs-171 R3): nothing to regenerate until the first measured table")
        return _validator(src, table_path)
    if not measurements.exists():
        return [f"{table_path.name}: no {measurements.relative_to(ROOT)} to regenerate it from"]
    wanted, _ = generate(measurements, rules_path, src, partial=partial)
    if wanted != shipped:
        fails.append(f"{table_path.name} does not regenerate byte for byte from {measurements.name} and {rules_path.name}: run generate")
    now = identities(src)
    for k, ident in sorted(table["kernels"].items()):
        if now.get(k) != ident:
            fails.append(f"{k}: built as {now.get(k, 'nothing')} now, measured as {ident}: re-measure its classes")
    data = json.loads(measurements.read_text())
    rules = Rules.read(rules_path)
    sys.path.insert(0, str(src))
    from cozy_runtime.internal import arch, attention

    for name, record in sorted(data["classes"].items()):
        row = arch.row(record["sm"])
        runs = {c.name for c in attention.KERNELS if not attention._off_row(c, row)}
        unmeasured = sorted(runs - set(record["kernels"]))
        if unmeasured:
            fails.append(f"{name}: never measured on {record['card']}, which runs them: {', '.join(unmeasured)}")
        for k, why in sorted(record.get("absent", {}).items()):
            if not rules.excludes(k, name):
                fails.append(f"{name}: {k} did not run on {record['card']} ({why[:120]}) and rules.toml excludes it with no reason")
    return fails + _validator(src, table_path)


def _validator(src: Path, table: Path) -> list[str]:
    """The Runtime's own import-time validation (runtime-design §6), when this Runtime has it."""
    if not (src / "cozy_runtime" / "internal" / "attention_table.py").exists():
        return []
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-m", "cozy_runtime.internal.attention_table", "check", str(table)],
                          capture_output=True, text=True, env=env)  # fmt: skip
    return [] if done.returncode == 0 else [f"{table.name}: {(done.stderr or done.stdout).strip()[-600:]}"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("generate", "check", "identities"))
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    parser.add_argument("--rules", type=Path, default=RULES)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--src", type=Path, default=ROOT / "src", help="the source tree whose kernels count")
    parser.add_argument("--partial", action="store_true", help="some classes not measured yet (a round in progress)")
    parser.add_argument("--report", type=Path, help="write the per-cell report here")
    args = parser.parse_args(argv)
    src = args.src.resolve()
    if args.command == "identities":
        print(json.dumps(identities(src), indent=1, sort_keys=True))
        return 0
    if args.command == "generate":
        text, report = generate(args.measurements, args.rules, src, args.partial)
        args.table.write_text(text)
        (args.report.write_text if args.report else lambda s: sys.stderr.write(s))("\n".join(report) + "\n")
        return 0
    fails = check(args.measurements, args.rules, args.table, src, args.partial)
    for fail in fails:
        print(f"FAIL {fail}")
    print(f"attention table: {'FAIL' if fails else 'PASS'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
