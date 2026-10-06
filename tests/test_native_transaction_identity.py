"""The Runtime native transaction identity remains stable."""

import json
from pathlib import Path
from cozy_runtime.internal.weights_sink import weights_transaction_id


def test_native_transaction_identity_matches_its_fixed_vector() -> None:
    row = json.loads(
        (Path(__file__).parent / "testdata/native-transaction-vector.json").read_bytes()
    )
    assert (
        weights_transaction_id(
            row["owner_authority_scope"],
            row["request_id"],
            row["invocation_spec_digest"],
            row["output_slot"],
        )
        == row["weights_transaction_id"]
    )
