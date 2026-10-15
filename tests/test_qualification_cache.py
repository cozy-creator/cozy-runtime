"""The decode qualification is release data: one table per card architecture and torch minor.

The input is the RECORDED REAL PROBE `tests/observations/probe-cuda-sm89.json` (an RTX 4070
Laptop, sm89, torch 2.13.0+cu130), its observations moved onto this build's spec digests by
implementation name with every measurement verbatim (as `test_leaf_speedup_record.rekeyed`).

**How "the suite did not run" is proved.** `qualified` is handed an object that is not torch;
every path that measures reaches for an attribute of it and raises.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import threading
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.cli import qualify
from cozy_runtime.internal import qualification
from cozy_runtime.internal.encoding import (
    DeviceFacts,
    Provider,
    implementation_digest,
    launch_providers,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.probe import QualificationResult

OBSERVATION = Path(__file__).resolve().parent / "observations" / "probe-cuda-sm89.json"
NO_TORCH = object()


def recorded() -> QualificationResult:
    return msgspec.convert(json.loads(OBSERVATION.read_bytes()), QualificationResult)


def providers() -> dict[str, tuple[Provider, ...]]:
    """This build's providers the recording measured an implementation of."""
    names = {one.implementation for one in recorded().observations}
    return {
        encoding: kept
        for encoding, group in launch_providers().items()
        if (kept := tuple(p for p in group if p.name in names))
    }


def table() -> QualificationResult:
    """The recording as this build's table: one observation per (spec, implementation, dtype)."""
    result = recorded()
    by_name = {(one.implementation, one.output_dtype): one for one in result.observations}
    return msgspec.structs.replace(
        result,
        observations=[
            msgspec.structs.replace(
                by_name[(provider.name, dtype)],
                encoding_spec_id=encoding,
                implementation_digest=implementation_digest(provider),
            )
            for encoding, group in providers().items()
            for provider in group
            for dtype in qualification.TABLE_DTYPES
        ],
    )


def ship(directory: Path, result: QualificationResult) -> str:
    name = qualification.table_name(result.device, result.runtime)
    directory.mkdir(parents=True, exist_ok=True)
    qualification.write_table(directory / f"{name}.json", result)
    return name


def load(
    device: DeviceFacts, cache: Path | None = None, **moved: Any
) -> qualification.Qualification:
    shipped = table()
    return qualification.qualified(
        NO_TORCH,
        providers(),
        device,
        moved.get("dtypes", ("bfloat16", "int64")),
        release="9.9.9",
        runtime=dataclasses.replace(shipped.runtime, **moved.get("runtime", {})),
        cache=str(cache or ""),
    )


def test_a_table_answers_for_its_architecture_torch_minor_and_runtime_source() -> None:
    device, runtime = table().device, table().runtime
    name = qualification.table_name(device, runtime)
    same = [
        dataclasses.replace(device, name="NVIDIA L40S", driver="600.1", index=3),
        dataclasses.replace(device, configuration="150W/[N/A]"),
    ]
    assert {qualification.table_name(other, runtime) for other in same} == {name}
    assert (
        qualification.table_name(device, dataclasses.replace(runtime, torch_build="2.13.1")) == name
    )
    assert qualification.table_name(device, dataclasses.replace(runtime, release="9.9.9")) == name
    moved = [
        qualification.table_name(dataclasses.replace(device, sm=90), runtime),
        qualification.table_name(device, dataclasses.replace(runtime, torch_build="2.14.0+cu130")),
        qualification.table_name(
            device, dataclasses.replace(runtime, source_digest="sha256:" + "0" * 64)
        ),
    ]
    assert name not in moved and len(set(moved)) == len(moved)


def test_a_shipped_table_answers_another_card_of_its_architecture_without_the_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = table()
    monkeypatch.setattr(qualification, "SHIPPED", tmp_path / "shipped")
    name = ship(qualification.SHIPPED, shipped)
    other = dataclasses.replace(shipped.device, name="NVIDIA GeForce RTX 4090", driver="580.1")

    answer = load(other, runtime={"torch_build": "2.13.1+cu130"})

    assert (answer.source, answer.key) == ("shipped", name)
    bf16 = [o for o in shipped.observations if o.output_dtype == "bfloat16" and o.evidence.passed]
    verbatim = sum(p.route == "verbatim" for group in providers().values() for p in group)
    # every passed bf16 observation, and the int64 destination's verbatim route, unmeasured
    assert len(answer.capabilities.records) == len(bf16) + verbatim
    assert {r.output_dtype for r in answer.capabilities.records} == {"bfloat16", "int64"}


def test_another_architecture_torch_minor_or_source_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = table()
    monkeypatch.setattr(qualification, "SHIPPED", tmp_path / "shipped")
    ship(qualification.SHIPPED, shipped)
    for device, runtime in (
        (dataclasses.replace(shipped.device, sm=90), {}),
        (shipped.device, {"torch_build": "2.14.0+cu130"}),
        (shipped.device, {"source_digest": "sha256:" + "0" * 64}),
    ):
        with pytest.raises(AttributeError):
            load(device, runtime=runtime)
    # a table filed under another table's name is not evidence about it
    sm90 = dataclasses.replace(shipped.device, sm=90)
    qualification.write_table(
        qualification.SHIPPED / f"{qualification.table_name(sm90, shipped.runtime)}.json", shipped
    )
    with pytest.raises(AttributeError):
        load(sm90)


def test_an_executor_waits_for_the_table_another_is_measuring_and_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 6349's four BiRefNet executors each ran the suite. The first measures; the rest
    wait on its lock and read what it kept."""
    shipped = table()
    monkeypatch.setattr(qualification, "SHIPPED", tmp_path / "none")
    cache = tmp_path / "qualification"
    cache.mkdir()
    name = qualification.table_name(shipped.device, shipped.runtime)
    answers: list[qualification.Qualification] = []
    with open(cache / f"{name}.lock", "a+") as measuring:
        fcntl.flock(measuring, fcntl.LOCK_EX)
        waiter = threading.Thread(target=lambda: answers.append(load(shipped.device, cache)))
        waiter.start()
        waiter.join(0.5)
        assert waiter.is_alive() and not answers
        ship(cache, shipped)
    waiter.join()
    assert [(a.source, a.key) for a in answers] == [("kept", name)]


def test_a_kept_table_missing_a_destination_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = table()
    monkeypatch.setattr(qualification, "SHIPPED", tmp_path / "none")
    ship(tmp_path / "qualification", shipped)
    assert load(shipped.device, tmp_path / "qualification").source == "kept"
    with pytest.raises(AttributeError):
        load(shipped.device, tmp_path / "qualification", dtypes=("float64",))


def test_the_gate_writes_the_table_a_load_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release gate's own entry, on this host's CPU: its table answers the next load."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(qualification, "SHIPPED", tmp_path)
    device, runtime = measure_device(torch, 0), measure_runtime(torch, "any")
    stale = tmp_path / f"{qualification.table_name(device, runtime).rsplit('-', 2)[0]}-0-v2.json"
    other = tmp_path / "cpu-torch0.0-0-v2.json"
    for path in (stale, other):
        path.write_text("{}")
    assert qualify.main([str(tmp_path)]) == 0
    assert not stale.exists() and other.exists()
    answer = qualification.qualified(
        NO_TORCH,
        launch_providers(),
        device,
        ("float16",),
        release="any",
        runtime=runtime,
    )
    assert answer.source == "shipped" and answer.capabilities.records


def test_a_machine_measures_only_what_its_table_lacks_and_keeps_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown card, on this host's CPU: each load measures only its own destinations
    (no more than before tables shipped), and the machine's table grows to answer both."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(qualification, "SHIPPED", tmp_path / "none")
    cache, device = tmp_path / "qualification", measure_device(torch, 0)
    runtime = measure_runtime(torch, "any")

    def load_here(on: Any, dtypes: tuple[str, ...]) -> qualification.Qualification:
        return qualification.qualified(
            on, launch_providers(), device, dtypes, release="any", runtime=runtime, cache=str(cache)
        )

    first, second = load_here(torch, ("float16",)), load_here(torch, ("bfloat16",))
    assert (first.source, second.source) == ("measured", "measured")
    assert {o.output_dtype for o in second.result.observations} == {"bfloat16"}
    both = load_here(NO_TORCH, ("float16", "bfloat16"))
    assert both.source == "kept"
    assert len(both.capabilities.records) == len(first.capabilities.records) + len(
        second.capabilities.records
    )
