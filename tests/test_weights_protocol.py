"""The current shared metadata/helper wire, including policy and secret boundaries."""

from cozy_runtime.internal import executor_commands as commands


def test_current_metadata_wire_preserves_variant_policy_and_redacts_credentials() -> None:
    frame = {
        "cmd": "resolve_weights",
        "request_id": "run-1",
        "hub": {
            "origin": "https://hub.example.test",
            "key": "private-request-key",
            "issued_token": "private-issued-token",
            "capability": "private-capability",
            "token_endpoint": "https://auth.example.test/token",
            "ca_der": "YWJj",
            "object_hosts": ["objects.example.test"],
        },
        "models": [
            {"slot": "policy", "repo_id": "owner/model", "revision": None,
             "variants": ["int8", "bf16"], "fallback_to_only_variant": True},
            {"slot": "unnamed", "repo_id": "owner/model", "variants": [""]},
            {"slot": "sole", "repo_id": "owner/model", "variants": []},
        ],
    }
    command = commands.decode(frame)
    assert isinstance(command, commands.ResolveWeights)
    assert command.models[0].variants == ("int8", "bf16")
    assert command.models[0].fallback_to_only_variant is True
    assert command.models[1].variants == ("",)
    assert command.models[2].variants == ()
    assert not command.models[1].fallback_to_only_variant
    assert command.hub is not None and command.hub.object_hosts == ("objects.example.test",)
    encoded = commands.encode(command)
    assert encoded["cmd"] == "resolve_weights" and "store" not in encoded
    for secret in ["private-request-key", "private-issued-token", "private-capability"]:
        assert secret not in repr(command)
    assert "<redacted>" in repr(command)


def test_terminal_request_cleanup_does_not_require_a_construction() -> None:
    command = commands.decode({"cmd": "unload", "construction": "", "request_id": "run-1"})
    assert isinstance(command, commands.Unload)
    assert command.construction == "" and command.request_id == "run-1"
    ordinary = commands.decode({"cmd": "unload", "construction": "warm"})
    assert isinstance(ordinary, commands.Unload) and ordinary.request_id is None
