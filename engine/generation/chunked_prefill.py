"""Resumable dense prompt prefill with absolute positions and a retained KV cache."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch.nn.attention.bias import causal_lower_right
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

ALIGNED_PREFILL_ATTENTION = "minillm_l4_aligned_prefill_sdpa"
# FlashAttention with a lower-right causal mask reproduces whole-prompt rows
# bitwise once a chunk has more than one 128-row query block; at <= 128 rows
# (32 query heads, L4) it takes a split-key path that rounds differently, so
# those chunks keep the padded call.  Verified on 138 random (length, chunk)
# pairs up to 12k tokens.
LOWER_RIGHT_MIN_QUERY_ROWS = 129


def aligned_prefill_sdpa(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attend a prompt chunk with the exact SDPA call a whole prompt uses.

    SDPA reaches FlashAttention only for an unmasked square causal problem.
    A later chunk has fewer queries than cached keys, so it would need an
    offset mask and fall back to a kernel with different bf16 rounding, which
    deep layers amplify into different greedy tokens.  Zero query rows ahead
    of the chunk restore the whole-prompt shape; causal rows are computed
    independently, so the kept rows are bitwise identical to whole-prompt
    prefill.  The padding rows cost extra attention work, not projections.

    On CUDA, a chunk of more than 128 rows instead uses FlashAttention with a
    lower-right causal mask: identical bits without computing the padding
    rows, so a late chunk costs ``chunk x prefix`` rather than ``prefix**2``.
    """

    if attention_mask is not None or int(query.shape[0]) != 1:
        raise ValueError("aligned prefill attention expects one unpadded sequence")
    offset = int(key.shape[2]) - int(query.shape[2])
    if offset < 0:
        raise ValueError("prompt chunk has more queries than cached keys")
    if offset and query.is_cuda and int(query.shape[2]) >= LOWER_RIGHT_MIN_QUERY_ROWS:
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=causal_lower_right(int(query.shape[2]), int(key.shape[2])),
            scale=kwargs.get("scaling"), enable_gqa=True,
        )
        return output.transpose(1, 2).contiguous(), None
    if offset:
        padding = query.new_zeros((*query.shape[:2], offset, query.shape[3]))
        query = torch.cat((padding, query), dim=2)
    output, weights = sdpa_attention_forward(module, query, key, value, None, **kwargs)
    return output[:, offset:], weights


AttentionInterface.register(ALIGNED_PREFILL_ATTENTION, aligned_prefill_sdpa)


@contextmanager
def aligned_prefill_attention(model: Any) -> Iterator[None]:
    """Temporarily route an SDPA model's attention through the aligned call.

    The name has no registered mask function, so Transformers skips mask
    construction; the session's single unpadded row needs only causality.
    """

    config = model.config
    previous = config._attn_implementation
    if previous != "sdpa":
        yield
        return
    config._attn_implementation = ALIGNED_PREFILL_ATTENTION
    try:
        yield
    finally:
        config._attn_implementation = previous


@dataclass(frozen=True)
class PrefillChunkOutput:
    start_token: int
    end_token: int
    logits: torch.Tensor
    complete: bool


class ChunkedPrefillSession:
    """Continue one prompt without recomputing earlier chunks.

    A reused prefix supplies its existing dense cache and ``start_token``.
    Intermediate logits are never emitted as output tokens. The caller owns
    scheduling, page storage, and releasing the retained dense cache.
    """

    def __init__(
        self,
        prompt_token_ids: Sequence[int],
        *,
        device: str | torch.device,
        start_token: int = 0,
        past_key_values: Any = None,
    ) -> None:
        self.prompt_token_ids = tuple(int(token) for token in prompt_token_ids)
        if not self.prompt_token_ids or not 0 <= start_token < len(self.prompt_token_ids):
            raise ValueError("prefill requires a non-empty uncached prompt suffix")
        cached_length = 0 if past_key_values is None else int(past_key_values.get_seq_length())
        if cached_length != start_token:
            raise ValueError("retained KV length must equal the next prompt position")
        self.device = torch.device(device)
        self.next_position = int(start_token)
        self.past_key_values = past_key_values
        self.chunk_count = 0

    @property
    def complete(self) -> bool:
        return self.next_position == len(self.prompt_token_ids)

    def step(self, model: Any, *, max_tokens: int) -> PrefillChunkOutput:
        if max_tokens < 1:
            raise ValueError("chunk token limit must be positive")
        if self.complete:
            raise RuntimeError("prompt prefill is already complete")
        start = self.next_position
        end = min(len(self.prompt_token_ids), start + max_tokens)
        with torch.inference_mode(), aligned_prefill_attention(model):
            output = model(
                input_ids=torch.tensor(
                    [self.prompt_token_ids[start:end]], dtype=torch.long, device=self.device
                ),
                attention_mask=torch.ones((1, end), dtype=torch.long, device=self.device),
                position_ids=torch.arange(start, end, device=self.device).unsqueeze(0),
                past_key_values=self.past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        if output.past_key_values is None or int(output.past_key_values.get_seq_length()) != end:
            raise RuntimeError("model did not extend the prompt KV cache correctly")
        self.past_key_values = output.past_key_values
        self.next_position = end
        self.chunk_count += 1
        return PrefillChunkOutput(start, end, output.logits[:, -1, :], self.complete)

