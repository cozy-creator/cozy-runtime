"""Actual nn.Module call ordering must own exactly one entered forward scope."""

import copy
import inspect
from typing import Any

import pytest

from cozy_runtime.internal import attention_sol


@pytest.mark.parametrize("registration", ["before", "after", "global"])
def test_nested_early_pre_hook_failure_keeps_outer_forward_scope(registration: str) -> None:
    torch = pytest.importorskip("torch")
    expected = attention_sol.Site("dit", "transformer_blocks.2.attn")

    class Recursive(torch.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, recurse: bool) -> Any:
            assert attention_sol._SITE.get() == expected
            if recurse:
                with pytest.raises(RuntimeError, match="early hook"):
                    self(False)
                assert attention_sol._SITE.get() == expected
            return torch.ones(1)

    module = Recursive()

    def reject(selected: Any, args: Any) -> None:
        if selected is module and args == (False,):
            raise RuntimeError("early hook")

    handle = None
    if registration == "before":
        handle = module.register_forward_pre_hook(reject)
    attention_sol.install_site(module, "dit", "transformer_blocks.2.attn")
    if registration == "after":
        handle = module.register_forward_pre_hook(reject, prepend=True)
    if registration == "global":
        handle = torch.nn.modules.module.register_module_forward_pre_hook(reject)
    try:
        module(True)
        assert attention_sol._SITE.get() is None
    finally:
        assert handle is not None
        handle.remove()
        attention_sol.remove_site(module, "dit", "transformer_blocks.2.attn")
    assert "forward" not in vars(module)


@pytest.mark.parametrize("failure", [False, True])
def test_post_hook_observes_restored_outer_site_and_remains_owned(failure: bool) -> None:
    torch = pytest.importorskip("torch")
    site = attention_sol.Site("dit", "transformer_blocks.2.attn")
    outer = attention_sol.Site("parent", "caller")

    class Layer(torch.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            assert attention_sol._SITE.get() == site
            return value + 1

    module = Layer()
    seen = []

    def post(_module: Any, _args: Any, output: Any) -> None:
        seen.append(attention_sol._SITE.get())
        assert torch.equal(output, torch.ones(1))
        if failure:
            raise RuntimeError("post hook")

    handle = module.register_forward_hook(post)
    attention_sol.install_site(module, site.component, site.path)
    token = attention_sol._SITE.set(outer)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="post hook"):
                module(torch.zeros(1))
        else:
            module(torch.zeros(1))
        assert seen == [outer]
        assert attention_sol._SITE.get() == outer
    finally:
        attention_sol._SITE.reset(token)
        attention_sol.remove_site(module, site.component, site.path)
    assert handle.id in module._forward_hooks
    handle.remove()


def test_original_instance_forward_signature_and_identity_are_restored() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Module()
    seen = []

    def original(value: Any, *, multiplier: int = 2) -> Any:
        """Existing caller implementation."""
        seen.append(attention_sol._SITE.get())
        return value * multiplier

    module.forward = original
    attention_sol.install_site(module, "dit", "attn")
    wrapper = module.forward
    assert inspect.signature(wrapper) == inspect.signature(original)
    assert wrapper.__doc__ == original.__doc__
    attention_sol.install_site(module, "dit", "attn")
    assert module.forward is wrapper
    assert torch.equal(module(torch.ones(1), multiplier=3), torch.full((1,), 3.0))
    assert seen == [attention_sol.Site("dit", "attn")]
    attention_sol.remove_site(module, "dit", "attn")
    assert module.forward is original
    assert attention_sol._SITE.get() is None


def test_removal_never_overwrites_a_later_forward_owner() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Identity()
    attention_sol.install_site(module, "dit", "attn")
    owned = module.forward

    def replacement(value: Any) -> Any:
        return value + 2

    module.forward = replacement
    with pytest.raises(ValueError, match="forward was replaced"):
        attention_sol.remove_site(module, "dit", "attn")
    assert module.forward is replacement
    with pytest.raises(ValueError, match="forward was replaced"):
        attention_sol.install_site(module, "dit", "attn")
    module.forward = owned
    attention_sol.remove_site(module, "dit", "attn")
    assert "forward" not in vars(module)


def test_deepcopied_module_uses_its_own_weights_and_restores_its_forward() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(2)
    original_signature = inspect.signature(module.forward)
    attention_sol.install_site(module, "dit", "attn")
    copied = copy.deepcopy(module)
    with torch.no_grad():
        copied.weight.fill_(7)
    value = torch.ones(1, 1)
    assert torch.equal(module(value), torch.full_like(value, 2))
    assert torch.equal(copied(value), torch.full_like(value, 7))
    assert copied.forward.__self__ is copied
    assert inspect.signature(copied.forward) == original_signature
    attention_sol.remove_site(copied, "dit", "attn")
    assert "forward" not in vars(copied)
    assert torch.equal(copied(value), torch.full_like(value, 7))
    assert hasattr(module, "_cozy_sol_site")
    attention_sol.remove_site(module, "dit", "attn")
    assert attention_sol._SITE.get() is None


def test_deepcopy_copies_existing_callable_state_instead_of_closing_over_it() -> None:
    torch = pytest.importorskip("torch")

    class Forward:
        def __init__(self) -> None:
            self.weight = torch.tensor(2.0)

        def __call__(self, value: Any) -> Any:
            return self.weight * value

    module = torch.nn.Module()
    module.forward = Forward()
    attention_sol.install_site(module, "dit", "attn")
    copied = copy.deepcopy(module)
    copied._cozy_sol_site.original.weight.fill_(7)
    value = torch.ones(1)
    assert torch.equal(module(value), torch.full_like(value, 2))
    assert torch.equal(copied(value), torch.full_like(value, 7))
    expected_callable = copied._cozy_sol_site.original
    attention_sol.remove_site(copied, "dit", "attn")
    assert copied.forward is expected_callable
    assert torch.equal(copied(value), torch.full_like(value, 7))
    attention_sol.remove_site(module, "dit", "attn")


def test_deepcopy_restores_an_existing_instance_bound_method() -> None:
    from types import MethodType

    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(1, 1, bias=False)

    def original(self: Any, value: Any, *, offset: int = 1) -> Any:
        return self.weight * value + offset

    module.forward = MethodType(original, module)
    attention_sol.install_site(module, "dit", "attn")
    copied = copy.deepcopy(module)
    with torch.no_grad():
        module.weight.fill_(2)
        copied.weight.fill_(7)
    attention_sol.remove_site(copied, "dit", "attn")
    assert copied.forward.__self__ is copied
    assert copied.forward.__func__ is original
    assert torch.equal(copied(torch.ones(1, 1)), torch.full((1, 1), 8.0))
    attention_sol.remove_site(module, "dit", "attn")
    assert module.forward.__self__ is module
    assert module.forward.__func__ is original


def test_instance_callable_keeps_self_and_wrapper_name_keywords() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Module()

    def original(*, self: Any, _cozy_module: Any) -> Any:
        return self + _cozy_module

    module.forward = original
    # This is a direct forward-call contract; nn.Module.__call__ itself reserves
    # its own self keyword before any installed forward can be reached.
    expected = module.forward(self=torch.ones(1), _cozy_module=torch.ones(1))
    attention_sol.install_site(module, "dit", "attn")
    assert inspect.signature(module.forward) == inspect.signature(original)
    assert torch.equal(module.forward(self=torch.ones(1), _cozy_module=torch.ones(1)), expected)
    attention_sol.remove_site(module, "dit", "attn")
    assert module.forward is original
