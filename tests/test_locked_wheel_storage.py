from __future__ import annotations

import pytest

from cozy_runtime.internal import package_environment as environment


def test_standard_direct_requirements_preserve_exact_wheel_and_legacy_pins() -> None:
    direct = b"proof @ https://files.example/proof-1.0-py3-none-any.whl --hash=sha256:" + b"a" * 64
    selected = environment.read_locked_requirements(direct).rows[0]
    assert (selected.name, selected.version, selected.hashes) == (
        "proof",
        "1.0",
        ("sha256:" + "a" * 64,),
    )
    assert selected.url == "https://files.example/proof-1.0-py3-none-any.whl"
    legacy = environment.read_locked_requirements(b"proof==1.0 --hash=sha256:" + b"a" * 64).rows[0]
    assert legacy.version == "1.0" and not legacy.url


@pytest.mark.parametrize(
    "target",
    [
        "https://user:secret@example/proof-1.0-py3-none-any.whl",
        "file:///proof-1.0-py3-none-any.whl",
        "https://example/different-1.0-py3-none-any.whl",
        "https://example/source.tar.gz",
    ],
)
def test_direct_lock_rejects_credentials_other_artifacts_and_wrong_distribution(
    target: str,
) -> None:
    with pytest.raises(environment.EnvironmentRefusal):
        environment.read_locked_requirements(f"proof @ {target} --hash=sha256:{'a' * 64}".encode())
