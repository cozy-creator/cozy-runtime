"""cr-097: what a generation may teach the next one, and what it may NOT.

The defect these guard is a closed loop. `plan.py` may not refuse on an UNCALIBRATED
activation envelope, so the 1 GiB placeholder deepens the ladder and runs. When the real
demand is several GiB and a co-tenant holds the slack that used to hide the gap, the
attempt OOMs, the OOM poisons the executor, and the rebuild starts a new generation with an
empty ledger -- so the measurement that would have refused the retry died in the rebuild
the failure itself caused. Measured live on a rented A4000: four identical 38.3 s failures,
each followed by a ~25.8 s rebuild, until the requeue budget ran out.

Carrying the measurement forward closes the loop. The hazard in carrying it is that a peak
observed on an attempt that died on its way up is a TRUNCATED LOWER BOUND, not a
measurement: banking it would make the retry calibrated AND WRONG, which is strictly worse
than uncalibrated, because a too-small enforceable number defeats the very refusal the carry
exists to enable. So the two facts live in different slots and these arms hold them apart.

No device, no executor, no torch: the ledger's arithmetic is the whole subject.
"""

from __future__ import annotations

import msgspec

from cozy_runtime.internal.executor_replies import Metrics
from cozy_runtime.internal.worker.ledger import Ledger

CELL = "height=1024,width=1024"
OTHER = "height=1536,width=1536"


def _generation(ledger: Ledger, pid: int = 4242) -> None:
    """A fresh executor generation, as a rebuild after a poisoning leaves it."""
    ledger.begin_generation(pid)
    ledger.observe_construction(
        {
            "filled_bytes": 8 << 30,
            "allocator_bytes": 8 << 30,
            "resident": {"unet": 6 << 30, "vae": 2 << 30},
            "device_free_bytes": 4 << 30,
            "device_total_bytes": 16 << 30,
        }
    )


def test_a_failed_attempts_peak_never_becomes_an_envelope() -> None:
    """THE TRAP. A truncated peak may not calibrate the next generation."""
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": {"generate": 3 << 30}}, Metrics),
        cell=CELL,
        succeeded=False,
    )
    # The generation's own map still records it -- that is what an operator reads.
    assert ledger.activations_by_cell[CELL] == 3 << 30
    # The CARRIED map must not, because the attempt died climbing and 3 GiB is a floor.
    assert ledger.carried_activations_by_cell == {}
    assert ledger.carried_activation_scopes_by_cell == {}
    # After the rebuild the next generation is uncalibrated, which is the honest answer:
    # `plan.py` may not refuse on it, and it is not asked to.
    _generation(ledger, pid=4243)
    assert ledger.activations_for(CELL) == (0, "")


def test_a_succeeding_serve_calibrates_the_next_generation() -> None:
    """THE POINT. A completed serve's peak is a measurement and outlives the process."""
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": {"generate": 4 << 30}}, Metrics),
        cell=CELL,
        succeeded=True,
    )
    _generation(ledger, pid=4243)
    # The generation holds nothing of its own -- a rebuild cleared it -- and the carried
    # measurement prices the next attempt instead of the 1 GiB placeholder.
    assert ledger.activations_by_cell == {}
    value, source = ledger.activations_for(CELL)
    assert value == 4 << 30
    assert "earlier generation" in source
    assert ledger.activation_scopes_for(CELL) == {"generate": 4 << 30}


def test_a_carried_measurement_is_never_read_for_another_shape_unlabelled() -> None:
    """Shape-cell scoping is what keeps the carry honest: reading a 1024px measurement for a
    1536px request would be the same lie by another route, so a cross-cell read is an
    upper bound and SAYS so."""
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": {"generate": 4 << 30}}, Metrics),
        cell=CELL,
        succeeded=True,
    )
    _generation(ledger, pid=4243)
    value, source = ledger.activations_for(OTHER)
    assert value == 4 << 30
    assert CELL in source and "upper bound" in source


def test_a_devices_refusal_is_a_floor_and_only_refuses_a_picture_no_better() -> None:
    """The failed attempt's usable fact: at this many free bytes, this shape did not fit.

    It refuses a retry under a picture no better than the one that already failed, and
    stays silent about a picture with MORE room -- which may or may not fit, and finding
    out is exactly what a genuinely different attempt is for.
    """
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    assert ledger.insufficient_for(CELL, 1 << 30) == 0
    ledger.observe_device_shortfall(CELL, 2 << 30)
    assert ledger.insufficient_for(CELL, 2 << 30) == 2 << 30
    assert ledger.insufficient_for(CELL, 1 << 30) == 2 << 30
    assert ledger.insufficient_for(CELL, 3 << 30) == 0
    # It survives the rebuild the failure caused, and it is NEVER an envelope.
    _generation(ledger, pid=4243)
    assert ledger.insufficient_for(CELL, 2 << 30) == 2 << 30
    assert ledger.activations_for(CELL) == (0, "")


def test_the_construction_set_is_what_a_restore_asks_for() -> None:
    """`resident` follows the device and a vacate empties it; `constructed` does not move,
    so the worker can tell "never held it" from "the lane took it away"."""
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    assert ledger.missing_resident() == {}
    # The lane hands the card to a co-tenant: the executor's probe comes back empty.
    ledger.observe_vacate(
        {
            "residency": {"resident": {}, "evicted": {"unet": 6 << 30, "vae": 2 << 30}},
            "vacated": {"freed": {"unet": 6 << 30, "vae": 2 << 30}, "freed_bytes": 8 << 30},
        },
        "a co-tenant's prepare",
    )
    assert ledger.missing_resident() == {"unet": 6 << 30, "vae": 2 << 30}
    # And a generation that legitimately never built a component can never "restore" it.
    _generation(ledger, pid=4243)
    assert "text_encoder" not in ledger.missing_resident()


def test_a_measurement_carries_the_conditions_it_was_taken_under() -> None:
    """cr-097 third cut: a bare peak is not interpretable.

    Kernels expand into the headroom they are given, so the same model at the same shape
    measured 3.91 GiB alone on a 15.6 GiB card and 1.92 GiB with a co-tenant beside it.
    Banking the peak without the free bytes it was observed under leaves the next reader
    unable to tell an upper bound the kernels CHOSE from a requirement they HAVE.
    """
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    ledger.observe_device_free(15 << 30, 16 << 30)
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": {"generate": 4 << 30}}, Metrics),
        cell=CELL,
        succeeded=True,
    )
    assert ledger.activation_basis_free(CELL) == 15 << 30
    # A bigger measurement moves the conditions with it; a smaller one leaves both alone,
    # so the pair always describes the same observation.
    ledger.observe_device_free(7 << 30, 16 << 30)
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": {"generate": 2 << 30}}, Metrics),
        cell=CELL,
        succeeded=True,
    )
    assert ledger.carried_activations_by_cell[CELL] == 4 << 30
    assert ledger.activation_basis_free(CELL) == 15 << 30


def test_an_envelope_measured_on_a_roomier_card_may_not_refuse() -> None:
    """THE HAZARD THIS CLOSES. Enforcing an empty-card peak under a full card would evict a
    co-tenant that did not need evicting, or refuse an attempt that would have succeeded --
    moving the defect from "this tenant is slow" to "the OTHER tenant is thrown off the
    card", which is quieter and worse because the victim is not the tenant being served.

    `plan.py` reads exactly this comparison to decide the coefficient's provenance, and
    only a `measured` basis may refuse or select a worse rung.
    """
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    ledger.observe_device_free(15 << 30, 16 << 30)
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": {"generate": 4 << 30}}, Metrics),
        cell=CELL,
        succeeded=True,
    )
    _generation(ledger, pid=4243)
    basis = ledger.activation_basis_free(CELL)
    # A co-tenant has taken the card down to less room than the measurement was taken in:
    # the number stands as a bound, and the comparison says it may not be enforced.
    ledger.observe_device_free(7 << 30, 16 << 30)
    assert basis > ledger.device_free
    # Back on a card at least as roomy as the one it was measured on, it is a requirement.
    ledger.observe_device_free(15 << 30, 16 << 30)
    assert basis <= ledger.device_free


def test_an_uncalibrated_cell_has_no_conditions_to_compare() -> None:
    """No measurement, no basis -- and `plan.py` treats a zero basis as "applies", because
    the uncalibrated path already has its own may-not-refuse rule and must not be given a
    second one that behaves differently."""
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    assert ledger.activation_basis_free(CELL) == 0


def test_the_carried_scope_headrooms_obey_the_same_conditions_rule() -> None:
    """THE CONSUMER THAT REACHES FURTHEST. `plan.py` turns the per-scope measurements into
    `scope_headrooms`, the worker sends them to the executor on `open_attempt`, and `admit`
    refuses `device_shortfall` on `size + headroom` INSIDE the executor — past
    `Coefficient.enforceable`, which is where every other over-estimate is caught.

    So a per-scope peak measured on a roomier card would refuse an attempt there that would
    have succeeded, with no gate upstream able to see it. A carried scope measurement that
    does not apply must be ABSENT, not merely un-enforceable.
    """
    ledger = Ledger(worker_pid=1)
    _generation(ledger)
    ledger.observe_device_free(15 << 30, 16 << 30)
    ledger.observe_attempt(
        msgspec.convert({"activation_peaks": {"generate": 4 << 30}}, Metrics),
        cell=CELL,
        succeeded=True,
    )
    _generation(ledger, pid=4243)
    # A co-tenant has taken the card below the conditions the scopes were measured under.
    ledger.observe_device_free(7 << 30, 16 << 30)
    assert not ledger.carried_conditions_apply(CELL)
    assert ledger.activation_scopes_for(CELL) == {}
    # Room at least as tight as the measurement's, and the scopes are a requirement again.
    ledger.observe_device_free(15 << 30, 16 << 30)
    assert ledger.carried_conditions_apply(CELL)
    assert ledger.activation_scopes_for(CELL) == {"generate": 4 << 30}
