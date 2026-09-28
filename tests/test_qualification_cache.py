"""The qualification suite runs once per (card, build, providers, dtypes), not once per prepare.

`probe.py`'s first paragraph has always said the suite "runs ONCE per (accelerator model,
device configuration, driver, runtime build, encoding rule) tuple ... and its result is a
persisted typed fact". Its last paragraph said "a new Runtime process probes again", and
the code did the second thing: cr-102 measured 3.2-6.1 s of decode suite on EVERY
model-bearing prepare, re-measuring a card and a build that had not moved. On a warm pod
adopting a second model — cr-098's `sayla` case, which is the swap — every term of that
tuple is identical and every second of it was paid again.

The input here is the RECORDED REAL PROBE `tests/observations/probe-cuda-sm89.json`: the
exact document `probe.qualify` returned on an RTX 4070 Laptop (sm89, driver 595.84, torch
2.13.0+cu130), 27 observations, 19 passing. Same discipline as `test_device_qualification`
— real bytes, recorded once, replayed — and it is what lets these arms run in CI with no
GPU and no torch.

**How "the suite did not run" is proved.** `qualified` is handed an object that is not
torch and cannot stand in for it. On a cache HIT nothing touches it and a full admission
table comes back; on a MISS the suite reaches for `torch.<dtype>` and raises. That is not a
double of torch — it is the absence of one, and it makes non-invocation observable instead
of asserted.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.internal import probe
from cozy_runtime.internal.encoding import DeviceFacts, RuntimeIdentity, launch_providers
from cozy_runtime.internal.probe import QualificationResult

OBSERVATION = Path(__file__).resolve().parent / "observations" / "probe-cuda-sm89.json"

#: NOT TORCH, and unable to pretend to be. Every path that would run the suite reaches for
#: an attribute of it; this raises rather than answering, so a test that believes the cache
#: fired and is wrong fails loudly instead of passing quietly.
NO_TORCH = object()


def recorded() -> QualificationResult:
    return msgspec.convert(json.loads(OBSERVATION.read_bytes()), QualificationResult)


def device_of(result: QualificationResult, **moved: Any) -> DeviceFacts:
    """The card the recording was taken on, optionally with one fact moved."""
    return dataclasses.replace(result.device, **moved)


def runtime_of(result: QualificationResult, **moved: Any) -> RuntimeIdentity:
    return dataclasses.replace(result.runtime, **moved)


def key_of(result: QualificationResult, **moved: Any) -> str:
    return probe.qualification_key(
        moved.get("providers", launch_providers()),
        moved.get("device", device_of(result)),
        moved.get("dtypes", ("bfloat16",)),
        runtime=moved.get("runtime", runtime_of(result)),
    )


def _swapped(providers: dict[str, Any]) -> dict[str, Any]:
    """The same specs and the same implementations, WIRED THE OTHER WAY ROUND.

    The key carries `implementation_digest` per spec, and this is the change that proves
    it: no provider is invented, none is dropped, and no count moves — two encoding specs
    simply trade decoders. A key that folded the providers in as a set, a length or a
    single digest would not see it, and a document measured before the swap would then
    qualify a spec against an implementation that never decoded it.
    """
    rows = sorted(providers.items())
    (first, mine), (last, theirs) = rows[0], rows[-1]
    assert mine != theirs, "the registry has to hold two differently-served specs"
    return {**providers, first: theirs, last: mine}


def stored(key: str) -> str:
    """A cache file filed under `key` whose result is the recording's own bytes."""
    return json.dumps({"key": key, "result": json.loads(OBSERVATION.read_bytes())})


def seed(cache: Path, key: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.json"
    path.write_text(stored(key))
    return path


# ---------------------------------------------------------------- what determines the answer


def test_every_determinant_of_the_measurement_is_in_the_key() -> None:
    """Move one thing the suite's answer depends on; the key must move with it.

    This is the arm that keeps the cache honest. A key missing any of these would reuse a
    measurement taken on a different card, a different driver, a different torch, a
    different decoder or a different dtype — which is not a cache, it is a fabrication.
    """
    result = recorded()
    base = key_of(result)

    moved = {
        "another card": key_of(result, device=device_of(result, name="NVIDIA L40S")),
        "another SM": key_of(result, device=device_of(result, sm=90)),
        "another driver": key_of(result, device=device_of(result, driver="600.1")),
        "another CUDA runtime": key_of(result, device=device_of(result, cuda_runtime="12.4")),
        "another power cap or MIG profile": key_of(
            result, device=device_of(result, configuration="150W/1g.10gb")
        ),
        "another card on the same host": key_of(result, device=device_of(result, index=3)),
        "another OS or arch": key_of(result, device=device_of(result, os_arch="Linux/aarch64")),
        "another torch build": key_of(result, runtime=runtime_of(result, torch_build="2.12.0")),
        "another runtime source": key_of(
            result, runtime=runtime_of(result, source_digest="sha256:" + "0" * 64)
        ),
        "another release": key_of(result, runtime=runtime_of(result, release="9.9.9")),
        "another dtype set": key_of(result, dtypes=("bfloat16", "float16")),
        "one provider fewer": key_of(
            result, providers=dict(sorted(launch_providers().items())[:-1])
        ),
        "no providers at all": key_of(result, providers={}),
        "two specs trading decoders": key_of(result, providers=_swapped(launch_providers())),
    }
    # WHAT THIS ARM CANNOT REACH, said plainly. A provider's `code_revision` is a
    # read-only property of a compiled-in implementation, so "the same provider set with
    # one implementation's arithmetic changed" cannot be constructed here without
    # inventing a provider — and an invented one would prove nothing about the real set.
    # The runtime term is what actually covers that case: `source_digest` is over
    # `encoding/`, `instruments.py`, `probe.py` and `resolution.py` as installed, so an
    # edit to any decode, leaf forward or oracle moves the key whatever the registry says.
    # The providers term is the second lock, and it is checked here for what it can be
    # checked for: membership, count and which implementation serves which spec.
    collisions = sorted(what for what, key in moved.items() if key == base)
    assert not collisions, f"the key ignores: {collisions}"
    assert len(set(moved.values())) == len(moved), "two different inputs share one key"


def test_nothing_that_cannot_change_the_answer_is_in_the_key() -> None:
    """The other half, and the one that decides whether the cache ever fires.

    A key that folded in the model, the manifest or the clock would miss forever while
    looking exactly like a working cache. The suite's answer is a function of the card, the
    build, the providers and the dtypes; the observations it produced are its OUTPUT.
    """
    result = recorded()
    base = key_of(result)
    noisy = msgspec.structs.replace(result, suite_ms=result.suite_ms + 5000, observations=[])
    assert key_of(noisy) == base
    assert key_of(result) == base


# ------------------------------------------------------------------------- reuse, and refusal


def test_a_kept_qualification_is_reused_without_running_the_suite(tmp_path: Path) -> None:
    """The whole claim, end to end: the second process pays none of the suite.

    `NO_TORCH` is what proves it. Every route that measures dereferences torch; this one
    returns a complete admission table without touching it.
    """
    result = recorded()
    cache = tmp_path / "qualification"
    key = key_of(result)
    seed(cache, key)

    answer = probe.qualified(
        NO_TORCH,
        launch_providers(),
        device_of(result),
        ("bfloat16",),
        release=result.runtime_release,
        runtime=runtime_of(result),
        cache=str(cache),
    )

    assert answer.measured is False
    assert answer.key == key
    assert answer.document()["source"] == "reused"
    # The table is the one the recording earns, counted off the RECORDED EVIDENCE rather
    # than off the code that builds it: a failed observation mints no record.
    passed = sum(1 for one in result.observations if one.evidence.passed)
    assert len(answer.capabilities.records) == passed
    assert answer.result == result


def test_a_key_that_does_not_match_falls_through_to_the_suite(tmp_path: Path) -> None:
    """A miss MEASURES. The stored document for another card is not evidence about this one."""
    result = recorded()
    cache = tmp_path / "qualification"
    seed(cache, key_of(result))

    with pytest.raises(AttributeError):
        probe.qualified(
            NO_TORCH,
            launch_providers(),
            device_of(result, name="NVIDIA L40S"),
            ("bfloat16",),
            release=result.runtime_release,
            runtime=runtime_of(result),
            cache=str(cache),
        )


def test_a_document_that_does_not_carry_its_own_key_is_not_used(tmp_path: Path) -> None:
    """A file's NAME is not evidence about its contents.

    Without this, anything that could write into the cache directory could hand this
    process an admission table it never measured, under a key it never earned.
    """
    result = recorded()
    cache = tmp_path / "qualification"
    key = key_of(result)
    cache.mkdir(parents=True)
    (cache / f"{key}.json").write_text(stored("sha256:" + "f" * 64))

    kept, why = probe._read_qualification(str(cache), key)
    assert kept is None
    assert "does not carry its own key" in why

    with pytest.raises(AttributeError):
        probe.qualified(
            NO_TORCH,
            launch_providers(),
            device_of(result),
            ("bfloat16",),
            release=result.runtime_release,
            runtime=runtime_of(result),
            cache=str(cache),
        )


def test_a_truncated_or_unparsable_document_is_not_used(tmp_path: Path) -> None:
    """Half a document is not a measurement, and neither is one of the wrong shape."""
    key = key_of(recorded())
    cache = tmp_path / "qualification"
    cache.mkdir(parents=True)
    (cache / f"{key}.json").write_text(stored(key)[:400])
    kept, why = probe._read_qualification(str(cache), key)
    assert kept is None and why.startswith("unreadable: JSONDecodeError"), why

    (cache / f"{key}.json").write_text(json.dumps({"key": key, "result": {"observations": {}}}))
    kept, why = probe._read_qualification(str(cache), key)
    assert kept is None and why.startswith("unreadable: ValidationError"), why


# --------------------------------------------------------------------------- keeping it


def test_a_document_that_could_not_be_used_is_replaced_rather_than_left(tmp_path: Path) -> None:
    """A miss caused by a bad document must still keep the good one.

    If the read's reason stood in for the write's, a rig that hit one corrupt or mis-filed
    document would re-measure on every prepare for the life of that key — the cache-that-
    never-fires failure, arrived at from the one direction the happy path cannot see.
    """
    result = recorded()
    cache = tmp_path / "qualification"
    key = key_of(result)
    cache.mkdir(parents=True)
    (cache / f"{key}.json").write_text(stored("sha256:" + "f" * 64))
    assert probe._read_qualification(str(cache), key)[0] is None

    assert probe._write_qualification(str(cache), key, result) == ""
    kept, why = probe._read_qualification(str(cache), key)
    assert why == "" and kept == result


def test_what_is_written_is_what_is_read_back(tmp_path: Path) -> None:
    """The writer and the reader agree, or the cache never fires and nobody notices."""
    result = recorded()
    cache = tmp_path / "qualification"
    key = key_of(result)

    assert probe._write_qualification(str(cache), key, result) == ""
    kept, why = probe._read_qualification(str(cache), key)
    assert why == ""
    assert kept == result
    assert stat.S_IMODE((cache / f"{key}.json").stat().st_mode) == 0o600


def test_an_unwritable_cache_directory_is_not_a_refusal(tmp_path: Path) -> None:
    """A rig whose cache cannot be written must still serve — measuring, and keeping nothing."""
    if os.geteuid() == 0:
        pytest.fail("run this as an unprivileged user; root can write anywhere")
    sealed = tmp_path / "sealed"
    sealed.mkdir(mode=0o500)
    try:
        result = recorded()
        why = probe._write_qualification(str(sealed / "qualification"), key_of(result), result)
        assert why.startswith("not kept:"), why
        kept, missing = probe._read_qualification(str(sealed / "qualification"), "whatever")
        assert kept is None
        assert missing == ""
    finally:
        sealed.chmod(0o700)


def test_no_cache_directory_means_measure_and_keep_nothing(tmp_path: Path) -> None:
    """A path, not a switch: unset is the pre-cr-103 behaviour exactly."""
    result = recorded()
    kept, why = probe._read_qualification("", key_of(result))
    assert kept is None and why == "no cache directory"
    assert probe._write_qualification("", key_of(result), result) == "no cache directory"
    assert not list(tmp_path.iterdir())


def test_the_directory_does_not_grow_without_bound(tmp_path: Path) -> None:
    """A rig that upgrades its driver every week must not accumulate forever."""
    result = recorded()
    cache = tmp_path / "qualification"
    for index in range(probe.QUALIFICATION_KEEP + 8):
        assert probe._write_qualification(str(cache), f"{index:064x}", result) == ""
    assert len(list(cache.glob("*.json"))) == probe.QUALIFICATION_KEEP
