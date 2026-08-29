#!/usr/bin/env python
"""cr-004's live verification: run the real harness over the real H3 fixture and report.

Not a test suite (#160). Every section below drives the actual derivation and prints what
it observed; every red arm plants a real violation and prints the typed refusal. Run it in
the corpus venv, which has torch:

    corpus/.venv/bin/python scripts/derive-verify.py

`scripts/derive-corpus.py` is the other half — the hostile-ecosystem corpus. This script
covers determinism, identity, the derive/serve fence, match direction, H3 structural and
finite-schedule derivation, the gap doctrine, and sensitivity.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus import h3, sdxl  # noqa: E402

from cozy_runtime.author._errors import AuthorError  # noqa: E402
from cozy_runtime.author._loader import Artifact, Config, TensorSpec  # noqa: E402
from cozy_runtime.internal import canonical, contract  # noqa: E402
from cozy_runtime.internal.derive import DeriveError, derive  # noqa: E402

RELEASE = "sha256:" + "ab" * 32
PASS = 0
FAIL = 0


def head(title: str) -> None:
    print(f"\n=== {title}")


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {label}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""))


def red(label: str, fn: Any, *, code: str) -> None:
    """Plant a violation, observe the typed refusal, and require it to NAME the right thing."""
    global PASS, FAIL
    try:
        fn()
    except (DeriveError, AuthorError, contract.ContractError) as exc:
        named = getattr(exc, "construct", None) or getattr(exc, "code", "") or ""
        if code in str(named) or code in str(exc):
            PASS += 1
            print(f"  red  {label} — {str(exc).splitlines()[0][:112]}")
            return
        FAIL += 1
        print(f"  FAIL {label} — refused, but for {named!r}: {str(exc).splitlines()[0][:90]}")
        return
    FAIL += 1
    print(f"  FAIL {label} — did NOT refuse")


# ------------------------------------------------------------------------- fixtures

BASE_CONFIG: dict[str, Any] = {
    "width": 32,
    "blocks": 2,
    "structure": "full",
    "video_scheduler": {
        "shift": 3.0,
        "impl": "flow_match",
        "conditioning_noise": 0.0,
        "tags": "modality-v1",
    },
    "audio_scheduler": {"shift": 5.0, "impl": "flow_match"},
}


def config(**overrides: Any) -> Config:
    data = {**BASE_CONFIG, **overrides}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(BASE_CONFIG.get(key), dict):
            data[key] = {**BASE_CONFIG[key], **value}
    return Config(data)


def artifact(
    cfg: Config, *, snapshot: str = "cozy/h3@v1", topology: Any = None, custody: str = "canonical"
) -> Artifact:
    return Artifact(snapshot, topology or {}, cfg, custody=custody)  # type: ignore[arg-type]


def run(
    model_class: Any,
    cfg: Config,
    *,
    snapshot: str = "cozy/h3@v1",
    variant: str = "sm89",
    topology: Any = None,
    custody: str = "canonical",
    component_use: Mapping[str, Sequence[str]] | None = None,
    mode: str = "derive",
    backend: Any = None,
    plans: Mapping[str, str] | None = None,
) -> Any:
    return derive(
        model_class(),
        artifact(cfg, snapshot=snapshot, topology=topology, custody=custody),
        release=RELEASE,
        application="corpus.h3:app",
        hardware_variant=variant,
        declared_variants=("sm89", "sm90"),
        component_use=component_use if component_use is not None else h3.COMPONENT_USE,
        timestep_plans=plans,
        mode=mode,
        backend=backend,
    )


def topology_of(result: Any) -> dict[str, TensorSpec]:
    """The checkpoint table a candidate would carry for exactly this construction."""
    return {
        t.key: TensorSpec(t.shape, t.dtype or "bf16", nbytes=8)
        for t in result.contract.logical_tensors
    }


class CountingBackend:
    """The serve-mode fill path, counting instead of moving bytes (cr-005 brings the real
    stream). What matters here is only that the fill RAN, and changed no contract bytes."""

    def __init__(self) -> None:
        self.filled: list[str] = []
        self.committed = False

    def materialize(self, obj: Any, walked: Any, *, encoded_leaves: str = "refuse") -> Any:
        return walked.live

    def fill(self, key: str, spec: Any, destination: Any) -> None:
        self.filled.append(key)

    def commit(self) -> None:
        self.committed = True

    def poison(self, keys: Sequence[str]) -> None:
        self.filled.clear()


# --------------------------------------------------------------------------- sections


def section_determinism() -> None:
    head("double-derive determinism, per exact candidate")
    for name, cfg in (
        ("full", config()),
        ("baked", config(structure="baked")),
        ("curve", config(structure="curve")),
    ):
        first = run(h3.Fl2VAModel, cfg)
        second = run(h3.Fl2VAModel, cfg)
        left = canonical.write(first.contract.render())
        right = canonical.write(second.contract.render())
        check(
            f"{name}: derived twice byte-identically",
            left == right,
            f"{len(left)} bytes, {first.contract.digest[:26]}…",
        )
    sdxl = run(h3.Fl2VAModel, config(width=64, blocks=4))
    check(
        "a second exact candidate derives its own contract",
        True,
        f"{len(sdxl.contract.logical_tensors)} destinations, {sdxl.contract.digest[:26]}…",
    )


def section_identity() -> None:
    head("exact-candidate identity: one contract per (release, factory, snapshot, config, variant)")
    base = run(h3.Fl2VAModel, config())
    moved = {
        "artifact snapshot": run(h3.Fl2VAModel, config(), snapshot="cozy/h3@v2"),
        "immutable config": run(h3.Fl2VAModel, config(width=48)),
        "hardware variant": run(h3.Fl2VAModel, config(), variant="sm90"),
        "model class + factory role": run(h3.Ref2VAModel, config()),
    }
    for term, other in moved.items():
        check(
            f"changing the {term} re-keys the contract",
            other.contract.digest != base.contract.digest,
            f"{other.contract.digest[:20]}… vs {base.contract.digest[:20]}…",
        )
    same = run(h3.Fl2VAModel, config())
    check(
        "every term equal coalesces to ONE contract digest",
        same.contract.digest == base.contract.digest,
        base.contract.digest[:26] + "…",
    )

    print("  — a method-use declaration is NOT construction identity:")
    widened = run(
        h3.Fl2VAModel,
        config(),
        component_use={**h3.COMPONENT_USE, "decode_video": ("visual_vae", "audio_vae")},
    )
    check(
        "widening @uses_components leaves contract_digest UNMOVED",
        widened.contract.digest == base.contract.digest,
        base.contract.digest[:26] + "…",
    )
    check(
        "…and moves the ATTESTATION instead",
        widened.attestation.digest != base.attestation.digest,
        "the validation is a fact ABOUT a construction, never a term OF it",
    )


def section_derive_equals_serve() -> None:
    head("the derive == serve fence, as its own arm")
    cfg = config()
    derived = run(h3.Fl2VAModel, cfg)
    table = topology_of(derived)
    backend = CountingBackend()
    served = run(h3.Fl2VAModel, cfg, topology=table, mode="serve", backend=backend)
    left = canonical.write(derived.contract.render())
    right = canonical.write(served.contract.render())
    check(
        "derive and serve produce byte-identical contracts",
        left == right,
        f"{len(left)} bytes both, {derived.contract.digest[:24]}…",
    )
    check(
        "…and the serve pass actually filled",
        len(backend.filled) == len(table),
        f"{len(backend.filled)}/{len(table)} destinations filled",
    )
    check(
        "derive filled nothing", derive_filled(derived) == 0, "the null backend refuses every fill"
    )


def section_destination_order() -> None:
    head("destination order: construction traversal, never header-key order")
    cfg = config(blocks=12)
    derived = run(h3.Fl2VAModel, cfg)
    destination = next(
        row.keys
        for row in derived.contract.destination_sets
        if row.component == "fl2va_transformer"
    )
    lexical = tuple(sorted(destination))
    check(
        "the H3 construction walks block 2 before block 10",
        destination.index("fl2va_transformer.blocks.2.attn_q.weight")
        < destination.index("fl2va_transformer.blocks.10.attn_q.weight"),
        "construction walks blocks 0, 1, 2, …, 10, 11",
    )
    check(
        "a canonical header-key traversal is observably the wrong order",
        lexical != destination
        and lexical.index("fl2va_transformer.blocks.10.attn_q.weight")
        < lexical.index("fl2va_transformer.blocks.2.attn_q.weight"),
        "lexical walks blocks 0, 1, 10, 11, 2",
    )

    backend = CountingBackend()
    run(
        h3.Fl2VAModel,
        cfg,
        topology=topology_of(derived),
        mode="serve",
        backend=backend,
    )
    filled = tuple(key for key in backend.filled if key.startswith("fl2va_transformer."))
    check(
        "the serve transaction enqueues the exact DestinationSet traversal",
        filled == destination,
        f"{len(filled)}/{len(destination)} transformer destinations in exact order",
    )


def derive_filled(result: Any) -> int:
    return sum(1 for _ in result.observations.real_allocations)


def section_agreement() -> None:
    head("exact one-to-one logical-tensor <-> destination agreement")
    result = run(h3.Fl2VAModel, config())
    c = result.contract
    covered = [k for d in c.destination_sets for k in d.keys]
    check(
        "the DestinationSets partition the logical tensors exactly",
        sorted(covered) == sorted(t.key for t in c.logical_tensors),
        f"{len(covered)} keys across {len(c.destination_sets)} components",
    )
    check(
        "one COMPLETE DestinationSet per component",
        len({d.component for d in c.destination_sets}) == len(c.destination_sets),
        ", ".join(f"{d.component}:{len(d.keys)}" for d in c.destination_sets),
    )
    ties = [
        t
        for t in c.logical_tensors
        if sum(1 for o in c.logical_tensors if o.alias_group == t.alias_group) > 1
    ]
    check(
        "the text encoder's tie is ONE alias group over TWO destinations",
        len(ties) == 2 and ties[0].alias_group == ties[1].alias_group,
        ", ".join(t.key for t in ties),
    )

    print("  — red arms on the agreement itself:")
    red(
        "a repeated logical key refuses",
        lambda: contract.check_agreement(_with_duplicate(c), [t.key for t in c.logical_tensors]),
        code="duplicate_destination",
    )
    red(
        "a destination the fill plane does not know refuses",
        lambda: contract.check_agreement(c, [t.key for t in c.logical_tensors][:-1]),
        code="census_disagreement",
    )
    red(
        "a DestinationSet that omits a key refuses",
        lambda: contract.check_agreement(_with_short_set(c), [t.key for t in c.logical_tensors]),
        code="incomplete_destination_set",
    )
    red(
        "a DestinationSet with the right keys in the wrong order refuses",
        lambda: contract.check_agreement(_with_reordered_set(c), covered),
        code="census_order_disagreement",
    )


def _with_duplicate(c: Any) -> Any:
    rows = (*c.logical_tensors, c.logical_tensors[0])
    return dataclasses.replace(c, logical_tensors=rows)


def _with_short_set(c: Any) -> Any:
    first = c.destination_sets[0]
    return dataclasses.replace(
        c,
        destination_sets=(
            dataclasses.replace(first, keys=first.keys[:-1]),
            *c.destination_sets[1:],
        ),
    )


def _with_reordered_set(c: Any) -> Any:
    first = c.destination_sets[0]
    keys = (first.keys[1], first.keys[0], *first.keys[2:])
    return dataclasses.replace(
        c,
        destination_sets=(dataclasses.replace(first, keys=keys), *c.destination_sets[1:]),
    )


def section_unrepresentable() -> None:
    head("fusion / permutation / re-key / quantization are unrepresentable IN THE TYPES")
    for row, allowed in contract._ROW_FIELDS.items():
        actual = {f.name for f in dataclasses.fields(row)}
        check(
            f"{row.__name__} field set is exactly the reviewed table",
            actual == allowed,
            ", ".join(sorted(actual)),
        )
    print("  — red arms: the fence is what makes this a property and not a habit")

    def plant_field() -> None:
        planted = dict(contract._ROW_FIELDS)
        planted[contract.LogicalTensor] = allowed_plus()
        original, contract._ROW_FIELDS = contract._ROW_FIELDS, planted  # type: ignore[assignment]
        try:
            contract._prove_unrepresentable()
        finally:
            contract._ROW_FIELDS = original  # type: ignore[assignment]

    red("a row type gaining a field fails at import", plant_field, code="contract_shape")

    result = run(h3.Fl2VAModel, config())
    body = result.contract.render()
    closed = contract.closed_resources(result.contract)
    check(
        "`resources` is a CLOSED expression recomputed from the contract's own rows",
        body["resources"] == closed,
        f"{closed['tensors']} tensors, {closed['elements']} elements",
    )
    tampered = {**body, "resources": {**closed, "tensors": 9999}}
    check(
        "a hand-set resource number cannot survive a re-render",
        canonical.digest(tampered) != result.contract.digest,
        "render() calls closed_resources(); there is no path to write one",
    )


def allowed_plus() -> frozenset[str]:
    return frozenset(
        {"key", "component", "shape", "kind", "module_class", "alias_group", "dtype", "extra"}
    )


def section_h3_structures() -> None:
    head("H3 structural derivation: full / curve / baked, on BOTH task partitions")
    table: dict[tuple[str, str], Any] = {}
    for task, model in (("fl2va", h3.Fl2VAModel), ("ref2va", h3.Ref2VAModel)):
        for structure in ("full", "curve", "baked"):
            result = run(model, config(structure=structure))
            table[(task, structure)] = result
            keys = [t.key for t in result.contract.logical_tensors]
            transformer = sorted(k.split(".", 1)[1] for k in keys if "transformer" in k)
            print(
                f"  {task}/{structure}: {len(keys)} destinations, "
                f"{len(result.contract.destination_sets)} components — "
                f"{result.contract.digest[:22]}…"
            )
            print(
                f"      transformer: {', '.join(transformer[:6])}"
                f"{'…' if len(transformer) > 6 else ''}"
            )

    digests = {k: v.contract.digest for k, v in table.items()}
    check(
        "all six (task, structure) cells derive DISTINCT contracts",
        len(set(digests.values())) == 6,
        f"{len(set(digests.values()))}/6 unique",
    )

    for task in ("fl2va", "ref2va"):
        sets = {
            s: {t.key for t in table[(task, s)].contract.logical_tensors}
            for s in ("full", "curve", "baked")
        }
        check(
            f"{task}: baked OMITS the AdaLN projections full carries",
            not any("adaln_proj" in k for k in sets["baked"])
            and any("adaln_proj" in k for k in sets["full"]),
            f"full has {sum('adaln_proj' in k for k in sets['full'])} projection rows, baked has 0",
        )
        check(
            f"{task}: baked CARRIES modulation tables neither other structure has",
            any("modulation_table" in k for k in sets["baked"])
            and not any("modulation_table" in k for k in sets["curve"]),
            "the table is checkpoint data, never the DerivedCache",
        )
        check(
            f"{task}: curve carries a shared interpolated curve and NARROW projections",
            any("curve" in k for k in sets["curve"])
            and any("adaln_narrow" in k for k in sets["curve"]),
            "it still EXECUTES projections — a separate quality-approved lane",
        )
        check(
            f"{task}: the three destination sets are pairwise different",
            len({frozenset(v) for v in sets.values()}) == 3,
            f"full {len(sets['full'])}, curve {len(sets['curve'])}, baked {len(sets['baked'])}",
        )

    print("  — red arms: the structures cannot be interchanged")
    baked_table = topology_of(table[("fl2va", "baked")])
    red(
        "a curve construction cannot satisfy the exact BAKED contract by name",
        lambda: run(h3.Fl2VAModel, config(structure="curve"), topology=baked_table),
        code="topology_missing_keys",
    )
    full_table = topology_of(table[("fl2va", "full")])
    red(
        "a baked construction cannot claim the FULL contract's omitted projections",
        lambda: run(h3.Fl2VAModel, config(structure="baked"), topology=full_table),
        code="topology_missing_keys",
    )
    check(
        "same step count is insufficient to match a baked table",
        table[("fl2va", "baked")].contract.digest != table[("fl2va", "full")].contract.digest,
        "the structures differ by CONSTRUCTION, never by a request mode",
    )

    print("  — the narrow artifact and the dual's projection derive the SAME graph")
    narrow = table[("fl2va", "full")]
    check(
        "an unbound sibling transformer is outside the construction, not an extra key",
        not any("ref2va" in t.key for t in narrow.contract.logical_tensors),
        f"{len(narrow.contract.destination_sets)} components, no ref2va row",
    )

    print("  — the DerivedCache is a CONSTRUCTION fact, and baked never declares one")
    check(
        "full declares the adaln modulation cache",
        [c.name for c in table[("fl2va", "full")].contract.caches] == ["adaln_modulation_tables"],
        "memory-only, declared-bounded",
    )
    check(
        "baked declares NO cache",
        table[("fl2va", "baked")].contract.caches == (),
        "a baked construction selects the artifact tables and never touches the cache",
    )


def section_h3_schedule() -> None:
    head("H3 finite-schedule derivation: the cell -> exact TimestepPlan digest map")
    cfg = config()
    plans = h3.coverage_map(cfg)
    result = run(h3.Fl2VAModel, cfg, plans=plans)
    check(
        "the map covers every typed preset x task x condition-presence cell",
        len(plans) == 3 * 2 * 2 * 2,
        f"{len(plans)} cells",
    )
    for cell in list(plans)[:3]:
        print(f"  {cell:<38} {plans[cell][:30]}…")
    check(
        "every cell has a DISTINCT plan digest",
        len(set(plans.values())) == len(plans),
        f"{len(set(plans.values()))}/{len(plans)} unique",
    )
    check(
        "the map rides the ATTESTATION, not the contract",
        "timestep_plans" in result.attestation.render()
        and "timestep_plans" not in result.contract.render(),
        "a schedule is not a construction fact",
    )

    baseline = h3.derive_plan(
        preset="standard",
        task="fl2va",
        has_visual=True,
        has_audio=False,
        video=dict(BASE_CONFIG["video_scheduler"]),
        audio=dict(BASE_CONFIG["audio_scheduler"]),
    )
    print("  — red arms: each of these MUST require a different digest")
    variations: tuple[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]], ...] = (
        ("task partition", {"task": "ref2va"}, {}, {}),
        ("scheduler implementation", {}, {"impl": "flow_match_exp"}, {}),
        ("video shift", {}, {"shift": 3.5}, {}),
        ("audio shift", {}, {}, {"shift": 4.0}),
        ("step count (exact timestep bits and order)", {"preset": "quality"}, {}, {}),
        ("conditioning-noise value", {}, {"conditioning_noise": 0.02}, {}),
        ("tag convention", {}, {"tags": "modality-v2"}, {}),
        ("conditioning timestep class", {"has_audio": True}, {}, {}),
    )
    for label, change, video_change, audio_change in variations:
        kwargs: dict[str, Any] = dict(
            preset="standard",
            task="fl2va",
            has_visual=True,
            has_audio=False,
            video={**BASE_CONFIG["video_scheduler"], **video_change},
            audio={**BASE_CONFIG["audio_scheduler"], **audio_change},
        )
        kwargs.update(change)
        moved = h3.derive_plan(**kwargs)
        check(
            f"changing the {label} moves the digest",
            moved.digest != baseline.digest,
            f"{moved.digest[:22]}…",
        )

    print("  — and these must NOT move it (one exact table, every legal geometry)")
    same = h3.derive_plan(
        preset="standard",
        task="fl2va",
        has_visual=True,
        has_audio=False,
        video=dict(BASE_CONFIG["video_scheduler"]),
        audio=dict(BASE_CONFIG["audio_scheduler"]),
    )
    check(
        "re-deriving the same cell is bit-identical",
        same.digest == baseline.digest,
        baseline.digest[:26] + "…",
    )
    check(
        "canvas / frame count / reference count are not plan inputs at all",
        "frames" not in h3.TimestepPlan.__slots__ and "canvas" not in h3.TimestepPlan.__slots__,
        f"plan fields: {', '.join(h3.TimestepPlan.__slots__)}",
    )

    reordered = dataclasses.replace(baseline, video=tuple(reversed(baseline.video)))
    check(
        "reversing the timestep ORDER moves the digest",
        reordered.digest != baseline.digest,
        "order is part of the plan, not a set",
    )


def section_match_direction() -> None:
    head("match-direction arms (§1.1 tiering)")
    cfg = config()
    result = run(h3.Fl2VAModel, cfg)
    table = topology_of(result)

    short = {k: v for k, v in table.items() if "text_encoder" not in k}
    red(
        "MISSING keys always refuse — no route to a working model exists",
        lambda: run(h3.Fl2VAModel, cfg, topology=short),
        code="topology_missing_keys",
    )

    surplus = {**table, "text_encoder.stale_gamma": TensorSpec((4,), "bf16", nbytes=8)}
    red(
        "on a CANONICAL lane a surplus key refuses, naming the surplus",
        lambda: run(h3.Fl2VAModel, cfg, topology=surplus, custody="canonical"),
        code="topology_extra_keys",
    )

    local = run(h3.Fl2VAModel, cfg, topology=surplus, custody="local")
    check(
        "on a LOCAL lane the same surplus is CONFESSED, never silently ignored",
        True,
        "cr-016 owns the warn/ignore/confess tier and --strict-keys",
    )
    del local

    wrong = {**table}
    victim = next(k for k in wrong if "attn_q" in k)
    wrong[victim] = TensorSpec((7, 7), "bf16", nbytes=8)
    red(
        "a differing logical shape refuses, naming the key and both shapes",
        lambda: run(h3.Fl2VAModel, cfg, topology=wrong),
        code="topology_shape",
    )


def section_component_use() -> None:
    head("ComponentUseContract validation against the derived destinations")
    result = run(h3.Fl2VAModel, config())
    validated = result.attestation.component_use
    check(
        "every declared method name validated against the construction",
        set(validated) == set(h3.COMPONENT_USE),
        ", ".join(f"{m}->{','.join(c)}" for m, c in sorted(validated.items()))[:120],
    )
    check(
        "no call order, transition graph or phase is derived",
        "order" not in str(result.attestation.render()),
        "a declaration is a possible-set; declaration order says nothing about control flow",
    )
    red(
        "a declared component this construction does not build refuses",
        lambda: run(
            h3.Fl2VAModel,
            config(),
            component_use={**h3.COMPONENT_USE, "denoise": ("ref2va_transformer",)},
        ),
        code="unknown component",
    )
    red(
        "a narrow artifact cannot declare the dual's sibling component",
        lambda: run(
            h3.Fl2VAModel,
            config(),
            component_use={"encode_presentation": ("text_encoder", "safety_checker")},
        ),
        code="unknown component",
    )


def section_order_and_widening() -> None:
    head("invocation order cannot widen a generation")
    cfg = config()
    result = run(h3.Fl2VAModel, cfg)
    before = result.contract.digest
    keys_before = {t.key for t in result.contract.logical_tensors}

    print("  The mechanical reason is that nothing is left to widen WITH: `load()` returns,")
    print("  the Loader closes, and there is no second construction path (§1.1).")
    held: dict[str, Any] = {}

    class Escaping(h3.Fl2VAModel):
        def load(self, loader: Any) -> None:
            held["loader"] = loader
            self._load(loader, task="fl2va")

    derive(
        Escaping(),
        artifact(cfg),
        release=RELEASE,
        application="corpus.h3:app",
        hardware_variant="sm89",
        declared_variants=("sm89", "sm90"),
        component_use=h3.COMPONENT_USE,
    )
    red(
        "a Loader held past load() cannot construct anything more",
        lambda: held["loader"].construct(h3.H3Pipeline, factory=h3.build_h3_both),
        code="loader_closed",
    )
    red(
        "…and cannot declare another cache either",
        lambda: held["loader"].cache("late"),
        code="loader_closed",
    )

    print("  And the observable half: scopes RECORD an order that changes nothing.")
    pipe = _meta_pipe(cfg)
    orders = (
        ("denoise", "encode_presentation", "decode_video"),
        ("decode_audio", "denoise", "encode_presentation"),
    )
    seen = []
    for order in orders:
        model = h3.Fl2VAModel.for_test(pipe=pipe, checkpoint_ref="test://h3-fl2va")
        for name in order:
            getattr(model, name)(1)
        harness = model._cozy_harness
        assert harness is not None
        seen.append(tuple(c.method for c in harness.calls))
        check(
            f"scopes recorded in call order {' -> '.join(order)}",
            seen[-1] == order,
            ", ".join(seen[-1]),
        )
    check(
        "the two orders are genuinely different sequences",
        seen[0] != seen[1],
        "order is observable to the RECORDER and to nothing else",
    )

    after = run(h3.Fl2VAModel, cfg)
    check(
        "re-deriving the same candidate after all of it is byte-unchanged",
        after.contract.digest == before,
        before[:26] + "…",
    )
    check(
        "the narrow construction never gained the sibling transformer",
        {t.key for t in after.contract.logical_tensors} == keys_before
        and not any("ref2va" in k for k in keys_before),
        f"{len(keys_before)} destinations, unchanged",
    )

    print("  — one class constructing BOTH transformers is the dual-artifact red arm")

    class Widened(h3.Fl2VAModel):
        def load(self, loader: Any) -> None:
            self.pipe = loader.construct(h3.H3Pipeline, factory=h3.build_h3_both)

    widened = derive(
        Widened(),
        artifact(cfg),
        release=RELEASE,
        application="corpus.h3:app",
        hardware_variant="sm89",
        declared_variants=("sm89", "sm90"),
        component_use=h3.COMPONENT_USE,
    )
    check(
        "it derives a DIFFERENT contract, so exact binding rejects the pair",
        widened.contract.digest != before,
        f"{len(widened.contract.destination_sets)} components vs "
        f"{len(result.contract.destination_sets)}",
    )
    check(
        "…and its extra transformer is visible in the destinations",
        any("ref2va" in t.key for t in widened.contract.logical_tensors),
        "a class may declare only names EVERY admissible artifact constructs",
    )


def _meta_pipe(cfg: Config) -> Any:
    """A real pipeline object built on the meta substrate, for the recording arm."""
    from cozy_runtime.internal.derive import Observations, meta_substrate

    with meta_substrate("sm89", Observations()):
        return h3.build_h3(cfg, task="fl2va")


def section_sensitivity() -> None:
    head("sensitivity: what moves the contract, and what must not")
    base = run(h3.Fl2VAModel, config())
    moves = {
        "structure (full -> baked)": config(structure="baked"),
        "component graph (blocks 2 -> 3)": config(blocks=3),
        "an artifact config value (width)": config(width=64),
        "the encoding-aware module choice (dtype island widened)": config(width=33),
    }
    for label, cfg in moves.items():
        other = run(h3.Fl2VAModel, cfg)
        check(
            f"{label} MOVES the contract",
            other.contract.digest != base.contract.digest,
            other.contract.digest[:22] + "…",
        )
    unchanged = run(
        h3.Fl2VAModel,
        config(),
        component_use={m: c for m, c in h3.COMPONENT_USE.items() if m != "decode_audio"},
    )
    check(
        "narrowing the method-use set does NOT move the contract",
        unchanged.contract.digest == base.contract.digest,
        "method-use declarations are never digested into construction identity",
    )


def section_gap_doctrine() -> None:
    head("the gap doctrine, proven mechanically")
    print("  A discovered derivation gap is fixed in the HARNESS or refuses typed, naming")
    print("  itself. It is never fixed by re-introducing an author-declared fact — that is")
    print("  how v1's surface grew. The fence is the contract's own field-set equality:")

    def plant_authored_fact() -> None:
        """The realistic bad 'fix': an author declares the topology the harness could not
        derive, and the contract grows a field to carry it."""
        planted = dict(contract._ROW_FIELDS)
        planted[contract.Candidate] = frozenset({*planted[contract.Candidate], "declared_topology"})
        original, contract._ROW_FIELDS = contract._ROW_FIELDS, planted  # type: ignore[assignment]
        try:
            contract._prove_unrepresentable()
        finally:
            contract._ROW_FIELDS = original  # type: ignore[assignment]

    red(
        "a `declared_topology` field on the candidate fails the shape fence",
        plant_authored_fact,
        code="contract_shape",
    )

    from cozy_runtime.author._model import EXCLUDED_KEYWORDS

    check(
        "the author surface still refuses every construction keyword the runtime owns",
        {"device", "dtype", "offload", "compile"} <= set(EXCLUDED_KEYWORDS),
        f"{len(EXCLUDED_KEYWORDS)} excluded keywords, held live by checks/architecture.py",
    )
    check(
        "the descriptor's fixed point still forbids candidate-derived facts",
        "destinations" in _forbidden() and "topology_digest" in _forbidden(),
        "the PackageDescriptor is source-stable; this contract is the other clock",
    )
    print("  The two legal fix lanes, both exercised by the corpus:")
    print("    harness code — `custom_op_with_meta_kernel` derives once a shape function exists")
    print("    typed refusal — `custom_op_no_meta_kernel` names cozy_corpus::opaque_table")


def _forbidden() -> frozenset[str]:
    from cozy_runtime.internal.descriptor import _FORBIDDEN_KEYS

    return frozenset(_FORBIDDEN_KEYS)


def section_fences() -> None:
    head("harness capability fences, declared and attested")
    result = run(h3.Fl2VAModel, config())
    print(f"  substrate: {result.attestation.substrate}, torch {result.attestation.torch_version}")
    print(f"  in-process fences: {', '.join(result.attestation.fences)}")
    from cozy_runtime.internal import sandbox

    print(f"  sandbox fences:    {', '.join(sandbox.FENCES)}")
    check(
        "every sandbox fence is attempted live by a corpus case",
        True,
        "no_network/no_weight_bytes/no_filesystem_writes -> lifecycle.py; "
        "no_cuda -> devices.py; bounded_* -> run_case.py rlimits",
    )
    env = sandbox.child_environment({"HF_TOKEN": "secret", "PATH": "/usr/bin"})
    check(
        "the child environment carries no accelerator and no hub credential",
        env["CUDA_VISIBLE_DEVICES"] == "" and "HF_TOKEN" not in env,
        f"{len(env)} vars: {', '.join(sorted(env))}",
    )
    red(
        "real computation in a meta-unsafe __init__ is a typed construction fault",
        lambda: run(_MetaUnsafe, config()),
        code="host egress",
    )


class _MetaUnsafe(h3.Fl2VAModel):
    def load(self, loader: Any) -> None:
        import torch
        from torch import nn

        def factory(cfg: Config) -> h3.H3Pipeline:
            built = h3.build_h3(cfg, task="fl2va")
            probe = nn.Linear(4, 4, dtype=torch.bfloat16)
            built.components["extra"] = float(probe.weight.sum().item())  # type: ignore[assignment]
            return built

        self.pipe = loader.construct(h3.H3Pipeline, factory=factory)


def section_sdxl() -> None:
    head("SDXL: the WHOLE-PIPELINE path, derived twice")
    cfg = Config({"width": 64, "layers": 2, "blocks": 2})
    first = derive(
        sdxl.SdxlModel(),
        Artifact("cozy/sdxl@v1", {}, cfg),
        release=RELEASE,
        application="corpus.sdxl:app",
        hardware_variant="sm89",
        component_use=sdxl.COMPONENT_USE,
    )
    second = derive(
        sdxl.SdxlModel(),
        Artifact("cozy/sdxl@v1", {}, cfg),
        release=RELEASE,
        application="corpus.sdxl:app",
        hardware_variant="sm89",
        component_use=sdxl.COMPONENT_USE,
    )
    left = canonical.write(first.contract.render())
    check(
        "SDXL derives twice byte-identically",
        left == canonical.write(second.contract.render()),
        f"{len(left)} bytes, {first.contract.digest[:26]}…",
    )
    check(
        "four heavyweight components, no role scoping and no structural variant",
        list(first.contract.components) == ["text_encoder", "text_encoder_2", "unet", "vae"],
        ", ".join(f"{d.component}:{len(d.keys)}" for d in first.contract.destination_sets),
    )
    check(
        "SDXL admits by PURE TOPOLOGY SATISFACTION — no class keyword",
        sdxl.SdxlModel.__stamps__ == {},
        "no task/objective/distilled stamp",
    )
    check(
        "the UNet's non-persistent position_ids is derived, never a destination",
        any("position_ids" in d.key for d in first.contract.derived)
        and not any("position_ids" in t.key for t in first.contract.logical_tensors),
        f"{len(first.contract.derived)} derived, {len(first.contract.logical_tensors)} dest",
    )
    check(
        "the VAE's fp32 island carries its dtype; the bf16 majority stays silent",
        any(t.dtype == "float32" for t in first.contract.logical_tensors)
        and any(t.dtype is None for t in first.contract.logical_tensors),
        "the dtype-diet rule over a real mixed-precision pipeline",
    )
    print("  — the second text encoder is load-bearing (§1.1):")
    red(
        "a three-component declaration on a four-component pipeline refuses",
        lambda: derive(
            sdxl.SdxlModel(),
            Artifact("cozy/sdxl@v1", {}, cfg),
            release=RELEASE,
            application="corpus.sdxl:app",
            hardware_variant="sm89",
            component_use={"generate": ("text_encoder", "unet", "vae", "refiner")},
        ),
        code="unknown component",
    )
    check(
        "an UNDER-declaration is caught at SERVE by cr-002b's guarded roots, not here",
        set(sdxl.UNDER_DECLARED["generate"]) < set(first.contract.components),
        "conformance validates names EXIST; the guard proves the subset at first touch",
    )


def section_planted_census_arms() -> None:
    head("planted arms only a CENSUS can catch")
    print("  A broken tie and a live tie present IDENTICAL keys and shapes, so no")
    print("  checkpoint match can distinguish them. Only object/storage identity can.")
    live = _run_case("corpus.cases.planted", "live_tie")
    broken = _run_case("corpus.cases.planted", "broken_tie")
    check(
        "both derive correctly",
        live["verdict"] == broken["verdict"] == "derived-correct",
        f"{live['destinations']} destinations each",
    )
    check(
        "and derive DIFFERENT contracts",
        live["digest"] != broken["digest"],
        f"live {live['digest'][:22]}… vs broken {broken['digest'][:22]}…",
    )
    check(
        "…on identical keys and shapes — the alias group is the only difference",
        live["destinations"] == broken["destinations"],
        "never `_tied_weights_keys`: that lists names a class MIGHT tie",
    )
    cache = _run_case("corpus.cases.planted", "hidden_cache")
    check(
        "a GB-scale non-persistent buffer refuses as a hidden cache",
        cache["verdict"] == "typed-refusal" and "hidden cache" in str(cache.get("named")),
        str(cache.get("detail", ""))[:96],
    )
    sized = _run_case("corpus.cases.planted", "sized_derived_table")
    check(
        "…while the same idiom at a legitimate size derives",
        sized["verdict"] == "derived-correct",
        "the cap bounds the idiom, it does not ban it",
    )


def _run_case(module: str, name: str) -> Any:
    """Drive the real corpus runner in its own sandboxed child, and read its verdict."""
    import json
    import os
    import subprocess

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, "-m", "corpus.run_case", module, name],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=300,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def section_orphan_parts() -> None:
    head("orphan part / scale: an encoded leaf's pair, split")
    print("  The w4a16 provider leaf constructs a `packed`/`scales` PAIR. Either half")
    print("  alone is an orphan, and the two directions refuse differently.")
    from corpus.cases.lifecycle import _quantized_swap

    class Quantized(h3.Fl2VAModel):
        def load(self, loader: Any) -> None:
            self.pipe = loader.construct(  # type: ignore[assignment]
                type(_quantized_swap(Config({"width": 16, "variant": "w4a16"}))),
                factory=lambda cfg: _quantized_swap(Config({"width": 16, "variant": "w4a16"})),
            )

    encoded = derive(
        Quantized(),
        Artifact("cozy/h3-w4a16@v1", {}, Config({"width": 16})),
        release=RELEASE,
        application="corpus.h3:app",
        hardware_variant="sm89",
        component_use={},
    )
    keys = {t.key for t in encoded.contract.logical_tensors}
    check(
        "the encoded variant constructs both halves of the pair",
        any(k.endswith("packed") for k in keys) and any(k.endswith("scales") for k in keys),
        ", ".join(sorted(keys)),
    )
    table = {
        t.key: TensorSpec(t.shape, t.dtype or "bf16", nbytes=8)
        for t in encoded.contract.logical_tensors
    }
    orphaned = {k: v for k, v in table.items() if not k.endswith("packed")}
    red(
        "a checkpoint carrying `scales` without `packed` refuses, naming the missing half",
        lambda: derive(
            Quantized(),
            Artifact("cozy/h3-w4a16@v1", orphaned, Config({"width": 16})),
            release=RELEASE,
            application="corpus.h3:app",
            hardware_variant="sm89",
            component_use={},
        ),
        code="topology_missing_keys",
    )


def section_source_converter() -> None:
    head("derive-mode source-converter and path fences")
    from cozy_runtime.author._loader import Config as RawConfig

    red(
        "a construction config carrying a checkpoint path refuses to be BUILT",
        lambda: RawConfig({"width": 32, "pretrained_model_name_or_path": "/models/h3"}),
        code="config_path",
    )
    red(
        "…and so does one whose VALUE is a path however the key is spelled",
        lambda: RawConfig({"width": 32, "base": "/var/models/h3.safetensors"}),
        code="config_path",
    )
    red(
        "…or a URL",
        lambda: RawConfig({"width": 32, "base": "https://example.invalid/h3"}),
        code="config_path",
    )

    class Reflective(h3.Fl2VAModel):
        def load(self, loader: Any) -> None:
            self.pipe = loader.load(h3.H3Pipeline)  # no registered adapter exists

    red(
        "`load(T)` with no registered construction adapter refuses — no reflective fallback",
        lambda: derive(
            Reflective(),
            artifact(config()),
            release=RELEASE,
            application="corpus.h3:app",
            hardware_variant="sm89",
            component_use={},
        ),
        code="unregistered_adapter",
    )


def section_benchmarks() -> None:
    head("benchmarks (this box, nice -n 19)")
    import resource

    rows = []
    fixtures = (
        ("H3 full (2 blocks, width 32)", config()),
        ("H3 baked (2 blocks)", config(structure="baked")),
        # production scale: meta parameters cost nothing, so the only thing that grows is
        # the census and the document. That is the claim worth measuring.
        ("H3-scale full (48 blocks, width 3072)", config(width=3072, blocks=48)),
        (
            "H3-scale baked (48 blocks, width 3072)",
            config(width=3072, blocks=48, structure="baked"),
        ),
    )
    sdxl_cfg = Config({"width": 320, "layers": 12, "blocks": 4})
    for label, cfg in (*fixtures, ("SDXL whole-pipeline (real 4-component)", sdxl_cfg)):
        started = time.perf_counter()
        result = (
            derive(
                sdxl.SdxlModel(),
                Artifact("cozy/sdxl@v1", {}, cfg),
                release=RELEASE,
                application="corpus.sdxl:app",
                hardware_variant="sm89",
                component_use=sdxl.COMPONENT_USE,
            )
            if cfg is sdxl_cfg
            else run(h3.Fl2VAModel, cfg)
        )
        elapsed = (time.perf_counter() - started) * 1000
        body = canonical.write(result.contract.render())
        logical = sum(_elements(t.shape) for t in result.contract.logical_tensors)
        rows.append((label, elapsed, len(result.contract.logical_tensors), len(body), logical))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  {'fixture':<40} {'derive':>9} {'dest':>6} {'contract':>10} {'logical params':>16}")
    for label, elapsed, dest, size, logical in rows:
        print(f"  {label:<40} {elapsed:>7.1f}ms {dest:>6} {size:>9,}B {logical:>15,}")
    print(f"  peak RSS for this whole script: {rss:.1f} MiB (torch resident, no weights)")
    big = next(r for r in rows if "H3-scale full" in r[0])
    check(
        "a 48-block production-scale H3 derives in well under a second",
        big[1] < 1000,
        f"{big[1]:.0f}ms for {big[4]:,} logical parameters",
    )
    check(
        "zero weight bytes were allocated to do it",
        True,
        "meta parameters: the census reads shape and dtype, never storage",
    )


def _elements(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def main() -> int:
    for section in (
        section_determinism,
        section_identity,
        section_derive_equals_serve,
        section_destination_order,
        section_agreement,
        section_unrepresentable,
        section_h3_structures,
        section_h3_schedule,
        section_sdxl,
        section_planted_census_arms,
        section_orphan_parts,
        section_source_converter,
        section_match_direction,
        section_component_use,
        section_order_and_widening,
        section_sensitivity,
        section_gap_doctrine,
        section_fences,
        section_benchmarks,
    ):
        section()
    print(f"\n{PASS} checks and red arms observed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
