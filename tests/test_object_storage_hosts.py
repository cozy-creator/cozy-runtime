"""Bounded current deployment host declarations."""

import pytest


def test_object_storage_hosts_are_an_explicit_bounded_deployment_declaration() -> None:
    from cozy_runtime.internal.config import ConfigError, parse_object_storage_hosts

    assert parse_object_storage_hosts("") == ()
    assert parse_object_storage_hosts(".objects.example,storage.example") == (
        ".objects.example",
        "storage.example",
    )
    for value in (
        "*",
        ".",
        "https://storage.example",
        "UPPER.example",
        "b.example,a.example",
        "a.example,a.example",
        "a.example,,b.example",
        "a.example ",
        "a" * 254,
        ",".join(f"host{i}.example" for i in range(9)),
    ):
        with pytest.raises(ConfigError, match="bounded sorted unique"):
            parse_object_storage_hosts(value)
