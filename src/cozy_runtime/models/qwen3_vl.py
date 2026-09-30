"""Qwen3-VL conditioning through its managed component root, without LM logits."""

from __future__ import annotations

from typing import Any

from transformers import Qwen3VLForConditionalGeneration

from cozy_runtime.author import retry_state


class Qwen3VLHiddenStateModel(Qwen3VLForConditionalGeneration):  # type: ignore[misc]
    """Keep the checkpoint's complete tensor namespace while returning model outputs.

    H3 and Qwen Image consume hidden states, never language-model logits or a previous
    call's autoregressive cache. Entering this root lets Runtime place and retry the
    whole conditioning operation. Dynamic RoPE replaces frequency tensors; it does not
    mutate their contents. Restore those bindings and position-cache state after every
    call so neither successful nor failed requests retain transient device allocations.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        retry_state(self, self._save_retry_state, self._restore_retry_state)

    def _save_retry_state(self) -> tuple[Any, dict[str, Any], dict[str, Any], set[str]]:
        rotary = self.model.language_model.rotary_emb
        names = _ROTARY_STATE
        return (
            self.model.rope_deltas,
            {name: vars(rotary)[name] for name in names if name in vars(rotary)},
            {name: rotary._buffers[name] for name in names if name in rotary._buffers},
            rotary._non_persistent_buffers_set.intersection(names),
        )

    def _restore_retry_state(
        self, state: tuple[Any, dict[str, Any], dict[str, Any], set[str]]
    ) -> None:
        self.model.rope_deltas, attributes, buffers, nonpersistent = state
        rotary = self.model.language_model.rotary_emb
        for name in _ROTARY_STATE:
            vars(rotary).pop(name, None)
            rotary._buffers.pop(name, None)
        vars(rotary).update(attributes)
        rotary._buffers.update(buffers)
        rotary._non_persistent_buffers_set.difference_update(_ROTARY_STATE)
        rotary._non_persistent_buffers_set.update(nonpersistent)

    def forward(self, **kwargs: Any) -> Any:
        if kwargs.pop("past_key_values", None) is not None:
            raise ValueError("hidden-state conditioning does not accept an autoregressive cache")
        kwargs["use_cache"] = False
        state = self._save_retry_state()
        try:
            return self.model(**kwargs)
        finally:
            self._restore_retry_state(state)


# Qwen3VLTextRotaryEmbedding's single-rope dynamic/longrope decorator owns these
# bindings. Vision RoPE has no persistent forward writes.
_ROTARY_STATE = (
    "inv_freq", "original_inv_freq", "long_inv_freq", "max_seq_len_cached", "attention_scaling",
)
