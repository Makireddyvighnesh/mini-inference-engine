"""Sequential batch-one generation with persistent paged prompt prefixes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..core.harness import RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from minillm_l4.engine.kv_cache import (
    PagedKvAllocator,
    PagedKvCache,
    PagedPrefixCache,
    select_decode_block_tokens,
    select_decode_split_count,
)
from minillm_l4.engine.kv_cache.qwen3_paged import install_paged_qwen3_attention

from .paged_kv import _cache_layers, _first_model_device, _paged_activation_dtype


class PrefixCachedPagedRunner:
    """Reuse exact-token prompt blocks across sequential requests.

    The cold request uses dense SDPA prefill and stores its KV in pages. A
    warm request runs only its uncached suffix, using SDPA with a gathered
    dense prefix by default. Decode uses the existing one-token paged path.
    This runner handles one request at a time.
    """

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        num_blocks: int,
        max_entries: int = 8,
        device: str | torch.device | None = None,
        decode_backend: str = "auto",
        warm_prefill_backend: str = "sdpa",
    ) -> None:
        if decode_backend not in {"auto", "torch", "triton"}:
            raise ValueError("decode_backend must be auto, torch, or triton")
        if warm_prefill_backend not in {"sdpa", "paged"}:
            raise ValueError("warm_prefill_backend must be sdpa or paged")
        self.model = model
        self.device = torch.device(device) if device is not None else _first_model_device(model)
        self.decode_backend = decode_backend
        self.warm_prefill_backend = warm_prefill_backend
        config = model.config
        self.allocator = PagedKvAllocator(num_blocks=num_blocks, block_size=block_size)
        self.cache = PagedKvCache(
            self.allocator,
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
            dtype=_paged_activation_dtype(model),
            device=self.device,
        )
        self.prefixes = PagedPrefixCache(self.cache, max_entries=max_entries)
        self.last_cache_snapshot: dict[str, Any] | None = None
        install_paged_qwen3_attention(model)

    @property
    def runner_name(self) -> str:
        return "paged_prefix_cache"

    def close(self) -> None:
        self.prefixes.clear()

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if len(requests) != 1 or len(recorders) != 1:
            raise ValueError("prefix-cache runner requires a batch of one")
        request, recorder = requests[0], recorders[0]
        owner = request.request_id
        prompt = request.prompt_token_ids
        recorder.record("prefill_start", metadata={"runner": self.runner_name})
        reused = self.prefixes.attach(
            owner, prompt, output_tokens=request.max_new_tokens
        )
        self.prefixes.record_admission(reused)
        computed = len(prompt) - reused
        max_sequence_length = len(prompt) + request.max_new_tokens - 1
        generated: list[int] = []
        try:
            with torch.inference_mode():
                suffix = torch.tensor(
                    [prompt[reused:]], dtype=torch.long, device=self.device
                )
                if reused == 0:
                    output = self.model(
                        input_ids=suffix,
                        attention_mask=torch.ones_like(suffix),
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                    layers = _cache_layers(output.past_key_values)
                    self.cache.append(
                        owner,
                        tuple((keys, values) for keys, values in layers),
                    )
                elif self.warm_prefill_backend == "sdpa":
                    dense_cache = self.cache.as_dynamic_cache(
                        (owner,), model_config=self.model.config
                    )
                    output = self.model(
                        input_ids=suffix,
                        attention_mask=torch.ones(
                            (1, len(prompt)), dtype=torch.long, device=self.device
                        ),
                        position_ids=torch.arange(
                            reused, len(prompt), dtype=torch.long, device=self.device
                        ).unsqueeze(0),
                        past_key_values=dense_cache,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                    self.cache.append_from_dynamic_cache(
                        (owner,),
                        output.past_key_values,
                        previous_token_counts=(reused,),
                        appended_token_counts=(computed,),
                    )
                else:
                    self.cache.reserve_append((owner,), computed)
                    output = self.model(
                        input_ids=suffix,
                        attention_mask=torch.ones_like(suffix),
                        position_ids=torch.arange(
                            reused, len(prompt), dtype=torch.long, device=self.device
                        ).unsqueeze(0),
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                        paged_kv_cache=self.cache,
                        paged_sequence_ids=(owner,),
                        paged_query_start_positions=(reused,),
                        paged_block_tables=self.cache.block_table_tensor(
                            (owner,), device=self.device
                        ).to(dtype=torch.int32),
                        paged_sequence_lengths=torch.tensor(
                            (len(prompt),), dtype=torch.int32, device=self.device
                        ),
                        paged_attention_backend=self.decode_backend,
                        paged_decode_max_sequence_length=max_sequence_length,
                        paged_decode_split_count=select_decode_split_count(
                            max_sequence_length, batch_size=1
                        ),
                        paged_decode_block_tokens=select_decode_block_tokens(
                            max_sequence_length
                        ),
                    )
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                # Publish only the prompt, before decode adds generated tokens.
                self.prefixes.publish(owner, prompt)
                first = int(next_token.item())
                recorder.record("prefill_end")
                generated.append(first)
                recorder.mark_token_ready(0, token_id=first)
                recorder.mark_token_sent(0, token_id=first)
                del output

                for step in range(1, request.max_new_tokens):
                    old_length = self.allocator.get_block_table(owner).token_count
                    self.cache.reserve_append((owner,), 1)
                    output = self.model(
                        input_ids=next_token,
                        attention_mask=torch.ones((1, 1), dtype=torch.long, device=self.device),
                        position_ids=torch.tensor([[old_length]], dtype=torch.long, device=self.device),
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                        paged_kv_cache=self.cache,
                        paged_sequence_ids=(owner,),
                        paged_query_start_positions=(old_length,),
                        paged_block_tables=self.cache.block_table_tensor(
                            (owner,), device=self.device
                        ).to(dtype=torch.int32),
                        paged_sequence_lengths=torch.tensor(
                            (old_length + 1,), dtype=torch.int32, device=self.device
                        ),
                        paged_attention_backend=self.decode_backend,
                        paged_decode_max_sequence_length=max_sequence_length,
                        paged_decode_split_count=select_decode_split_count(
                            max_sequence_length, batch_size=1
                        ),
                        paged_decode_block_tokens=select_decode_block_tokens(
                            max_sequence_length
                        ),
                    )
                    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    token = int(next_token.item())
                    generated.append(token)
                    recorder.mark_token_ready(step, token_id=token)
                    recorder.mark_token_sent(step, token_id=token)
                    del output

            recorder.record("completion")
            self.last_cache_snapshot = {
                **self.prefixes.snapshot(),
                "cached_prefix_tokens": reused,
                "computed_prefill_tokens": computed,
                "warm_prefill_backend": self.warm_prefill_backend,
                "allocator": self.allocator.snapshot(),
            }
            return (
                RequestOutcome(
                    status="completed",
                    generated_token_ids=tuple(generated),
                    metadata={"runner": self.runner_name, "prefix_cache": self.last_cache_snapshot},
                ),
            )
        finally:
            if owner in self.allocator.sequence_ids:
                self.cache.release(owner)
