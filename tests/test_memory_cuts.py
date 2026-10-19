"""The Runtime's cut seams through varena's memory (`tensorfs.varena.cuts`, memory-varena §5.3):
attention through the Runtime's dispatch cuts by head groups then query rows, and a package's
`split` runs in the windows that fit, each exactly the whole call.

Real torch on the CPU. The device is a `Card`: an op whose results and largest operand pass
`limit` bytes runs out of memory, as a GPU allocator does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

torch = pytest.importorskip("torch")
cuts = pytest.importorskip("tensorfs.varena.cuts")

from diffusers.models.attention_dispatch import dispatch_attention_fn  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from cozy_runtime.author import Cut, split  # noqa: E402
from cozy_runtime.author._split import install as install_split  # noqa: E402
from cozy_runtime.internal import attention_ulysses  # noqa: E402


class Card(TorchDispatchMode):  # type: ignore[misc]
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit, self.refused = limit, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        out = func(*args, **(kwargs or {}))
        if any(r.alias_info is not None for r in func._schema.returns):
            return out
        made = sum(t.numel() * t.element_size() for t in cuts._tensors(out))
        read = max((t.numel() * t.element_size() for t in cuts._tensors((args, kwargs))), default=0)
        if made and made + read > self.limit:
            self.refused += 1
            raise torch.OutOfMemoryError(f"CUDA out of memory. Tried to allocate {made} B.")
        return out


@pytest.fixture
def memory() -> Iterator[Any]:
    memo = cuts.Memo()
    cuts.serve(memo, cuts._NoRungs())
    attention_ulysses.install(cuts.attend)
    install_split(cuts.split)
    yield memo
    attention_ulysses.install(None)
    install_split(None)


PIECES: list[tuple[int, int]] = []


def scores_whole(q: Any, k: Any, v: Any, **options: Any) -> Any:
    """A [batch, sequence, heads, dim] kernel whose scores are one tensor."""
    PIECES.append((q.shape[1], q.shape[2]))
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    scores = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    if options.get("attn_mask") is not None:
        scores = scores + options["attn_mask"]
    return (scores.softmax(-1) @ v).transpose(1, 2)


def test_attention_through_the_dispatch_cuts_heads_then_rows(memory: Any) -> None:
    local = attention_ulysses.Local(scores_whole, lambda *_: "scores", head_local=True, rows=16)
    backend = attention_ulysses.member("_cozy_test_cut_attention", local)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 96, 4, 8, dtype=torch.float64) for _ in range(3))
    with torch.no_grad():
        whole = dispatch_attention_fn(q, k, v, backend=backend)
        row = 2 * 96 * 8  # one query row of one head: its scores and their working copy
        for limit in (2 * 96 * row, 16 * row + 2048):
            card = Card(limit)
            PIECES.clear()
            with card:
                cut = dispatch_attention_fn(q, k, v, backend=backend)
            torch.testing.assert_close(cut, whole, rtol=1e-12, atol=1e-12)
            assert card.refused
    assert PIECES[-6:] == [(16, 1)] * 6  # query rows of one head, on the kernel's block


def test_a_split_runs_in_the_windows_that_fit(memory: Any) -> None:
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 1), torch.nn.SiLU()).double()
    x = torch.randn(6, 4, 10, 10, dtype=torch.float64)
    with torch.no_grad():
        whole = net(x)
        card = Card(60_000)  # the convolution whole fits, its activation not
        with card:
            cut = split(net, x, Cut(0))
    torch.testing.assert_close(cut, whole, rtol=0, atol=0)
    assert card.refused and max(memory.sizes.values()) < x.numel()
