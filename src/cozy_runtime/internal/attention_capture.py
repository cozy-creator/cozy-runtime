"""Dev-only attention capture (`cozy run --attention-capture SPEC`, hidden behind `dev = true`):
the Q/K/V a kernel receives at chosen sites and calls, and the shape of every call, written on
the machine for the attention suite's corpus (xs-171).

SPEC is JSON: `{"dir": "/abs/path", "sites": ["fl2va_dit/transformer_blocks.0.attn", ...],
"calls": [0, 3], "heads": 4, "seed": 0}`. `sites` are fnmatch patterns on `component/path`
(none: the census alone); `calls` the per-site call ordinals within the request (none: every
call); `heads` how many heads a capture keeps: the two whose keys have the largest max/RMS, the
rest seeded at random. Each process writes under `dir/<request id>/`: `census.jsonl`, one row
per call, and one safetensors file per captured call, BF16 `query`/`key`/`value` of the kept
heads over the whole sequence, its facts in the header. The record is taken in
`attention_ulysses._call` after the band pick, so it is what the kernel receives: after norms
and RoPE, after Ulysses' exchange (the full sequence of this rank's heads). Rank 0 only; a call
recorded into a CUDA graph is counted in the census but not captured.
"""

from __future__ import annotations

import fnmatch
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import msgspec

from cozy_runtime.accel.attention import _ACTIVE_LAYOUT
from cozy_runtime.internal import attention_ulysses

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


class Spec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    dir: str
    sites: tuple[str, ...] = ()
    calls: tuple[int, ...] = ()
    heads: int = 4
    seed: int = 0


def parse(text: str) -> Spec:
    """SPEC's JSON, or ValueError naming what is wrong with it."""
    try:
        spec = msgspec.json.decode(text, type=Spec, strict=True)
    except msgspec.DecodeError as exc:
        raise ValueError(f"attention capture spec: {exc}") from exc
    if not Path(spec.dir).is_absolute() or spec.heads < 1:
        raise ValueError("attention capture spec: `dir` must be absolute and `heads` at least 1")
    return spec


#: The site a call of `_call` belongs to, by its module's forward (`_wrapped`).
_SITE: ContextVar[str] = ContextVar("cozy_capture_site", default="")


class _Capture:
    def __init__(self, spec: Spec, out: Path) -> None:
        self.spec, self.out = spec, out
        self.calls: dict[str, int] = {}
        self.census = (out / "census.jsonl").open("a")

    def __call__(
        self,
        site_backend: str,
        band: str,
        backend: str,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        options: attention_ulysses.Options,
    ) -> None:
        import torch

        site = _SITE.get()
        ordinal = self.calls.get(site, 0)
        self.calls[site] = ordinal + 1
        layout = _ACTIVE_LAYOUT.get()
        facts: dict[str, Any] = {  # type: ignore[explicit-any]  # a JSON row
            "site": site,
            "call": ordinal,
            "backend": backend,
            "dispatched": site_backend,
            "band": band,
            "query": list(query.shape),
            "key": list(key.shape),
            "dtype": str(query.dtype).removeprefix("torch."),
            "masked": options["attn_mask"] is not None,
            "causal": bool(options["is_causal"]),
            "scale": options["scale"],
        }
        if layout is not None:
            facts["layout"] = {
                "step": layout.step,
                "live_tokens": layout.live_tokens,
                "protected_prefix": layout.protected_prefix,
                "dense_until_step": layout.dense_until_step,
                "sparse": layout.sparse,
            }
        wanted = bool(site) and any(fnmatch.fnmatchcase(site, p) for p in self.spec.sites)
        wanted = wanted and (not self.spec.calls or ordinal in self.spec.calls)
        graphed = query.is_cuda and torch.cuda.is_current_stream_capturing()
        if wanted and not graphed:
            facts["file"] = self._save(site, ordinal, query, key, value, facts)
        self.census.write(json.dumps(facts, sort_keys=True) + "\n")
        self.census.flush()

    def _save(
        self,
        site: str,
        ordinal: int,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        facts: Mapping[str, object],
    ) -> str:
        import torch
        from safetensors.torch import save_file

        heads = int(query.shape[2])
        kept = min(self.spec.heads, heads)
        alike = int(key.shape[2]) == heads  # else fewer K/V heads (GQA): kept whole
        k = (key if alike else query).float()
        crest = k.abs().amax(dim=(0, 1, 3)) / k.square().mean(dim=(0, 1, 3)).sqrt().clamp_min(1e-12)
        top = [int(h) for h in torch.argsort(crest, descending=True)[: min(2, kept)].tolist()]
        rest = [h for h in range(heads) if h not in top]
        seeded = random.Random(f"{self.spec.seed}:{site}:{ordinal}")
        chosen = sorted(top + seeded.sample(rest, kept - len(top)))
        index = torch.tensor(chosen, device=query.device)
        picked = {"query": query.index_select(2, index)}
        picked |= (
            {"key": key.index_select(2, index), "value": value.index_select(2, index)}
            if alike
            else {"key": key, "value": value}
        )
        tensors = {n: t.to(torch.bfloat16).contiguous().cpu() for n, t in picked.items()}
        name = f"{site.replace('/', '__')}.{ordinal:04d}.safetensors"
        header = {**facts, "heads": chosen, "crest": [round(float(c), 3) for c in crest]}
        save_file(tensors, str(self.out / name), metadata={"facts": json.dumps(header)})
        return name


def _wrapped(module: Module, site: str) -> Any:  # type: ignore[explicit-any]  # torch's hook handles
    """Name MODULE's calls SITE for the capture, around its whole forward."""

    def before(_module: Module, _args: object) -> None:
        vars(module).setdefault("_cozy_capture_tokens", []).append(_SITE.set(site))

    def after(_module: Module, _args: object, _out: object) -> None:
        tokens = vars(module).get("_cozy_capture_tokens") or []
        if tokens:
            _SITE.reset(tokens.pop())

    return (
        module.register_forward_pre_hook(before),
        module.register_forward_hook(after, always_call=True),
    )


@contextmanager
def capturing(
    text: str, request_id: str, sites: Sequence[tuple[str, str, object]]
) -> Iterator[None]:
    """Within it, this process's attention calls are captured per SPEC (TEXT); SITES are this
    construction's (component, module path, module)."""
    if not text:
        yield
        return
    spec = parse(text)
    out = Path(spec.dir) / request_id
    out.mkdir(parents=True, exist_ok=True)
    capture = _Capture(spec, out)
    handles = [
        handle
        for component, path, module in sites
        for handle in _wrapped(cast("Module", module), f"{component.rsplit('/', 1)[-1]}/{path}")
    ]
    token = attention_ulysses.CAPTURE.set(capture)
    try:
        yield
    finally:
        attention_ulysses.CAPTURE.reset(token)
        for handle in handles:
            handle.remove()
        capture.census.close()
