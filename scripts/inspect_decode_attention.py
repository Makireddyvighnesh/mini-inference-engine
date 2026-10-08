"""Compare paged attention with the dense backend on identical real Qwen inputs.

Run from the workspace with PYTHONPATH set to its absolute path. This is a
diagnostic, not a throughput benchmark: it synchronizes and replays operations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from minillm_l4.benchmarks.runners.huggingface_baseline import build_hf_workload, load_qwen_fp8
from minillm_l4.benchmarks.runners.paged_kv import _cache_layers, _make_paged_cache
from minillm_l4.engine.kv_cache import PagedKvAllocator, paged_attention
from minillm_l4.engine.generation.manual import manual_greedy_generate


def errors(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {"max": float(difference.max().item()), "mean": float(difference.mean().item())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--request-index", type=int, default=3)
    parser.add_argument("--decode-index", type=int, default=16)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if not 1 <= args.decode_index < args.output_tokens or not 0 <= args.request_index < 4:
        parser.error("decode-index must be in [1, output-tokens), and request-index in [0, 4)")
    bundle = load_qwen_fp8(fp8_kernel_path="sm89")
    model = bundle.model
    workload = build_hf_workload(
        bundle.tokenizer, project / "data/synthetic/workloads_v1.jsonl",
        bucket_name="diagnostic", prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens, count=4, seed=17,
    )
    request = workload.requests[args.request_index]
    prompt = torch.tensor([request.prompt_token_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        gold = manual_greedy_generate(
            model, {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)},
            output_tokens=args.output_tokens,
        ).row(0).tolist()
    captured = {}
    capture_enabled = False

    def before(layer, inputs, kwargs):
        if capture_enabled:
            captured[layer.layer_idx] = {
                "hidden": kwargs["hidden_states"].clone(),
                "rope": tuple(value.clone() for value in kwargs["position_embeddings"]),
                "mask": kwargs["attention_mask"],
            }

    def after(layer, inputs, kwargs, output):
        if capture_enabled:
            captured[layer.layer_idx]["output"] = output[0].clone()

    handles = []
    for layer in model.model.layers:
        handles.extend([
            layer.self_attn.register_forward_pre_hook(before, with_kwargs=True),
            layer.self_attn.register_forward_hook(after, with_kwargs=True),
        ])
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        output = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True, return_dict=True, logits_to_keep=1)
        cache = output.past_key_values
        for index in range(1, args.decode_index + 1):
            capture_enabled = index == args.decode_index
            token = torch.tensor([[gold[index - 1]]], dtype=torch.long, device="cuda")
            position = prompt.shape[1] + index - 1
            output = model(
                input_ids=token, attention_mask=torch.ones((1, position + 1), device="cuda", dtype=torch.long),
                past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1,
            )
            cache = output.past_key_values
        for handle in handles:
            handle.remove()
        allocator = PagedKvAllocator(num_blocks=(position + 16) // 16, block_size=16)
        pages = _make_paged_cache(cache, allocator=allocator, owner_ids=("probe",))
        rows = []
        for index, (keys, values) in enumerate(_cache_layers(cache)):
            layer = model.model.layers[index].self_attn
            hidden = captured[index]["hidden"]
            query = layer.q_norm(layer.q_proj(hidden).view(1, 1, -1, layer.head_dim)).transpose(1, 2)
            new_keys = layer.k_norm(layer.k_proj(hidden).view(1, 1, -1, layer.head_dim)).transpose(1, 2)
            query, _ = apply_rotary_pos_emb(query, new_keys, *captured[index]["rope"])
            groups = layer.num_key_value_groups
            repeated_keys = keys.repeat_interleave(groups, dim=1)
            repeated_values = values.repeat_interleave(groups, dim=1)
            scores64 = (query.double() @ repeated_keys.double().transpose(-1, -2)) * layer.scaling
            probabilities64 = scores64.softmax(dim=-1)
            reference = (probabilities64 @ repeated_values.double()).to(query.dtype).transpose(1, 2)
            sdpa = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=captured[index]["mask"],
                scale=layer.scaling, enable_gqa=True,
            ).transpose(1, 2)
            common = dict(layer_index=index, query_start_positions=(position,), num_key_value_groups=groups)
            torch_pages = paged_attention(query, pages, ("probe",), backend="torch", **common)
            triton_pages = paged_attention(
                query, pages, ("probe",), backend="triton", **common,
                block_tables=pages.block_table_tensor(("probe",)).to(torch.int32),
                sequence_lengths=torch.tensor([position + 1], dtype=torch.int32, device="cuda"),
                decode_use_gqa_reuse=False,
            )
            triton_compat = paged_attention(
                query, pages, ("probe",), backend="triton", **common,
                block_tables=pages.block_table_tensor(("probe",)).to(torch.int32),
                sequence_lengths=torch.tensor([position + 1], dtype=torch.int32, device="cuda"),
                decode_use_gqa_reuse=False, decode_sdpa_compat=True,
            )
            scores32 = (query.float() @ repeated_keys.float().transpose(-1, -2)) * layer.scaling
            bf16_probability = (scores32.softmax(dim=-1).to(query.dtype).float() @ repeated_values.float()).to(query.dtype).transpose(1, 2)
            eager_scores = (query @ repeated_keys.transpose(-1, -2)) * layer.scaling
            eager = (eager_scores.float().softmax(dim=-1).to(query.dtype) @ repeated_values).transpose(1, 2)
            paths = {"sdpa": sdpa, "paged_torch": torch_pages, "paged_triton": triton_pages, "paged_triton_sdpa_compat": triton_compat, "bf16_probability": bf16_probability, "eager": eager}
            weights = (scores32 - scores32.max(dim=-1, keepdim=True).values).exp()
            paths["bf16_unnormalized"] = (
                (weights.to(query.dtype).float() @ repeated_values.float()) / weights.sum(dim=-1, keepdim=True)
            ).to(query.dtype).transpose(1, 2)
            for tile_size in (16, 32, 64, 128):
                for reverse in (False, True):
                    maximum = torch.full_like(scores32[..., :1], -torch.inf)
                    denominator = torch.zeros_like(maximum)
                    accumulator = torch.zeros_like(query, dtype=torch.float32)
                    tile_starts = list(range(0, keys.shape[2], tile_size))
                    if reverse:
                        tile_starts.reverse()
                    for start in tile_starts:
                        scores = scores32[..., start:start+tile_size]
                        updated = torch.maximum(maximum, scores.max(dim=-1, keepdim=True).values)
                        alpha = (maximum - updated).exp()
                        weights = (scores - updated).exp()
                        denominator = denominator * alpha + weights.sum(dim=-1, keepdim=True)
                        accumulator = accumulator * alpha + weights.to(query.dtype).float() @ repeated_values[..., start:start+tile_size, :].float()
                        maximum = updated
                    paths[f"bf16_online_{tile_size}_{'reverse' if reverse else 'forward'}"] = (
                        accumulator / denominator
                    ).to(query.dtype).transpose(1, 2)
            rows.append({
                "layer": index, "key_max": float(keys.abs().max().item()),
                "vs_fp64": {name: errors(value, reference) for name, value in paths.items()},
                "vs_sdpa": {name: errors(value, sdpa) for name, value in paths.items()},
                "projected_vs_actual": {name: errors(layer.o_proj(value.reshape(1, 1, -1).contiguous()), captured[index]["output"]) for name, value in paths.items()},
            })
            print(json.dumps({"layer": index, "vs_sdpa": {name: value["mean"] for name, value in rows[-1]["vs_sdpa"].items()}}), flush=True)
        report = {
            "model": bundle.metadata(), "attention_backend": model.config._attn_implementation,
            "commit_sha": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
            "request_id": request.request_id, "prompt_tokens": request.prompt_tokens,
            "output_token_number": args.decode_index + 1,
            "reference_winner": int(output.logits.argmax().item()), "position": int(position),
            "comparison": "All attention paths use the identical dense query and KV cache.", "layers": rows,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "attention_precision.json").write_text(json.dumps(report, indent=2) + "\n")
        print("backend:", report["attention_backend"], "winner:", report["reference_winner"], flush=True)


if __name__ == "__main__":
    main()
