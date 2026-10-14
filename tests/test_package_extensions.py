"""A package's own kernels and attention producers, through the Runtime's real paths.

CPU arms drive the real Diffusers H3 attention module, the real attention registry and the
real Triton front end; the compile-and-launch arm needs a card (a rental runs it)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("diffusers")
import triton.language as tl  # noqa: E402
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention  # noqa: E402

from cozy_runtime.author import (  # noqa: E402
    AttentionLayout,
    AttentionProducer,
    ProducerCall,
    attention_scope,
    register_attention,
)
from cozy_runtime.author import kernels as author_kernels  # noqa: E402
from cozy_runtime.author._attention import _PRODUCERS, _close, _request  # noqa: E402
from cozy_runtime.author.kernels import (  # noqa: E402
    ROWWISE_FP8,
    KernelsUnavailable,
    PreQuantized,
    TritonEntry,
    TritonKernels,
)
from cozy_runtime.internal import (  # noqa: E402
    attention,
    attention_sol,
    kernel_cache,
    kernel_compile,
    triton_kernels,
)
from cozy_runtime.internal.encoding.leaves import (  # noqa: E402
    RowwiseNativeLeaf,
    quantize_activation_rowwise,
)

BASE = "kitchen-sol-fp4-fixed-means"


@triton.jit  # type: ignore[untyped-decorator]
def _scale_kernel(  # type: ignore[no-untyped-def]
    x_ptr, out_ptr, n, FACTOR: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) * FACTOR, mask=mask)


def _entry(factor: int) -> TritonEntry:
    signature = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    signature |= {"FACTOR": "constexpr", "BLOCK": "constexpr"}
    return TritonEntry(_scale_kernel, signature, {"FACTOR": factor, "BLOCK": 128}, ("x_ptr",))


class State:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _produce(call: ProducerCall, hidden: Any, rotary: Any = None, mask: Any = None) -> Any:
    call.state.calls.append(f"{call.component}/{call.path}@{call.layout.step}")
    call.served("test-producer")
    return torch.full_like(hidden, 7.0)


def _validate(module: object) -> None:
    if not isinstance(module, MiniMaxH3Attention):
        raise ValueError("serves H3 attention only")


@pytest.fixture(autouse=True)
def _own_kernel_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(author_kernels, "_SETS", {})


@pytest.fixture
def producer(monkeypatch: pytest.MonkeyPatch) -> Iterator[AttentionProducer]:
    made = AttentionProducer(
        "test-producer", BASE, _produce, _validate, state=State, context_parallel=True
    )
    monkeypatch.setitem(_PRODUCERS, made.name, made)
    register_attention(made)
    yield made


def test_a_producer_is_its_base_backend_under_its_own_name(producer: AttentionProducer) -> None:
    candidate, base = attention.known()[producer.name], attention.BY_NAME[BASE]
    assert candidate.producer is producer and attention.sparse(candidate)
    assert candidate.backend == "_cozy_test_producer" and candidate.local.head_local
    assert (candidate.sms, candidate.artifact, candidate.import_name) == (
        base.sms,
        base.artifact,
        base.import_name,
    )
    assert attention.dense_identity(producer.name) == attention.dense_identity(BASE)
    assert attention.sparse_names([producer.name, "sdpa"]) == [producer.name]
    attention.admit(producer.name, 4)  # it declared context parallelism


def test_a_package_name_shadows_the_runtime_one_and_a_dense_base_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "kitchen-sol-producer-fp4-lowmem-shared-qkv"
    shadow = AttentionProducer(name, "kitchen-sol-fp4-lowmem", _produce, _validate)
    monkeypatch.setitem(_PRODUCERS, name, shadow)
    assert attention.known()[name].producer is shadow
    with pytest.raises(attention.AttentionRefusal, match="context parallelism"):
        attention.admit(name, 2)
    monkeypatch.setitem(
        _PRODUCERS, "dense", AttentionProducer("dense", "sdpa", _produce, _validate)
    )
    with pytest.raises(attention.AttentionRefusal, match="not one of this Runtime's sparse"):
        attention.known()


def test_a_selected_site_reaches_the_producer_and_any_other_backend_the_original(
    producer: AttentionProducer,
) -> None:
    attn = MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128).to(dtype=torch.bfloat16)
    sdpa = attention._member(attention.BY_NAME["sdpa"])
    attn.processor._attention_backend = sdpa
    original = attn.processor
    hidden = torch.randn(1, 5, 256, dtype=torch.bfloat16)
    phase = torch.randn(5, 48)
    rotary = tuple(torch.cat((fn(phase), fn(phase)), dim=-1) for fn in (torch.cos, torch.sin))
    expected = original(attn, hidden, rotary)
    candidate = attention.known()[producer.name]
    entry = attention.Ready(candidate, attention._member(candidate))
    attention._install_sites([(entry, "pipe/dit", "transformer_blocks.2.attn", attn)])
    wrapper = attn.processor
    assert wrapper.original is original and wrapper.producer is producer
    torch.testing.assert_close(attn(hidden, rotary), expected, rtol=0, atol=0)  # sdpa: original
    attn.processor._attention_backend = entry.member
    assert original._attention_backend is entry.member
    with pytest.raises(ValueError, match="layout and a request scope"):
        attn(hidden, rotary)
    with (
        attention_sol.observing() as counts,
        attention_scope(AttentionLayout(5, 1, 7, 4)),
    ):
        assert torch.equal(attn(hidden, rotary), torch.full_like(hidden, 7.0))
        state = producer.request_state()
        assert state.calls == ["dit/transformer_blocks.2.attn@7"] and not state.closed
    assert state.closed and counts["sparse"] == 1 and counts["dense_prefix"] == 1
    attention._install_sites([(entry, "pipe/dit", "transformer_blocks.2.attn", attn)])
    assert attn.processor is wrapper  # installing again keeps the one wrapper


def test_a_follower_attempt_holds_states_until_its_holder_closes_them(
    producer: AttentionProducer,
) -> None:
    held: dict[str, Any] = {}
    with _request(held):
        first = producer.request_state()
    assert isinstance(first, State)
    with _request(held):
        assert producer.request_state() is first and not first.closed
    _close(held)
    assert first.closed and not held and producer.request_state() is None


def test_a_kernel_set_is_keyed_by_its_sources_and_entries() -> None:
    kernels = TritonKernels("test-scale", [_entry(2), _entry(3)])
    assert kernels.identity == TritonKernels("test-scale", [_entry(3), _entry(2)]).identity
    assert kernels.identity != TritonKernels("test-scale", [_entry(2)]).identity
    with pytest.raises(ValueError, match="twice"):
        TritonKernels("test-scale", [_entry(2), _entry(2)])
    with pytest.raises(ValueError, match="must match"):
        TritonKernels("Test Scale", [_entry(2)])
    with pytest.raises(KernelsUnavailable, match="serves CUDA") as refused:
        triton_kernels.load(kernels, device="cpu")
    assert refused.value.code == "kernels_absent"
    assert triton_kernels.expect(kernels, 89).startswith("kernels_absent: no machine kernel store")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiling and launching needs a card")
def test_a_kernel_set_builds_at_boot_and_launches_without_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = kernel_cache.Store(tmp_path / "u")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(store.own / "triton"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
    kernels = TritonKernels("test-scale", [_entry(2), _entry(3)])
    major, minor = torch.cuda.get_device_capability()
    assert triton_kernels.expect(kernels, major * 10 + minor) == "compiling"
    job = triton_kernels.job(kernels, major * 10 + minor)
    assert kernel_compile.wait(store, job).state == "ready"
    monkeypatch.setattr(triton, "compile", None)  # loading compiles nothing
    launch = triton_kernels.load(kernels, device="cuda")
    x = torch.arange(300, device="cuda", dtype=torch.float32)
    for factor in (2, 3):
        out = torch.empty_like(x)
        launch(_scale_kernel, (3,), x, out, 300, factor, 128)
        assert torch.equal(out, x * factor)
    with pytest.raises(KernelsUnavailable, match="not compiled"):
        launch(_scale_kernel, (3,), x, torch.empty_like(x), 300, 5, 128)


def test_a_shared_operand_gives_the_leaf_its_own_bits_and_hooks_the_tensor() -> None:
    """A fused kernel's `PreQuantized` operand is the rowwise leaf's own quantization: the
    GEMM sees the same bits, and a forward hook still receives the original tensor."""
    torch.manual_seed(931)
    weight = torch.randn(128, 64, dtype=torch.bfloat16) * 0.1
    data, scale = quantize_activation_rowwise(torch, weight)
    linear = torch.nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    leaf = RowwiseNativeLeaf("fp8-rowwise/1").leaf(
        torch, {"data": data, "scale": scale}, linear, torch.bfloat16
    )
    seen: list[Any] = []
    leaf.register_forward_hook(lambda _module, args, _out: seen.append(args[0]))
    hidden = torch.randn(1, 5, 64, dtype=torch.bfloat16)
    operand = PreQuantized(*quantize_activation_rowwise(torch, hidden), tuple(hidden.shape))
    assert leaf.provider == ROWWISE_FP8
    assert torch.equal(leaf(hidden, prequantized=operand), leaf(hidden))
    assert len(seen) == 2 and all(tensor is hidden for tensor in seen)
