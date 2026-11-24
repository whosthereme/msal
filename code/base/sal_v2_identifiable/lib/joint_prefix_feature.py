from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

PARENT_ROOT = Path(__file__).resolve().parents[2]
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

IMAGE_TOKENS_PER_CAMERA = 64

SPATIAL_GRID_SIZE = 4


def prefix_layout(policy: Any, camera_count: int, language_tokens: int) -> dict[str, Any]:
    special = bool(getattr(policy.model, "add_image_special_tokens", False))
    per_camera = IMAGE_TOKENS_PER_CAMERA + (2 if special else 0)
    spans = []
    for index in range(camera_count):
        start = index * per_camera + (1 if special else 0)
        spans.append((start, start + IMAGE_TOKENS_PER_CAMERA))
    image_block = camera_count * per_camera
    return {
        "add_image_special_tokens": special,
        "camera_spans": spans,
        "language_span": (image_block, image_block + language_tokens),
        "state_index": image_block + language_tokens,
        "total_tokens": image_block + language_tokens + 1,
    }


@torch.inference_mode()
def extract_language_conditioned_spatial_state_feature(
    policy: Any,
    batch: dict[str, torch.Tensor],
    prompt: str,
    output_size: int = SPATIAL_GRID_SIZE,
) -> torch.Tensor:
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    from smolvla_state_contract import pool_spatial_tokens

    model = policy.model
    bridge = model.vlm_with_expert
    device = next(policy.parameters()).device

    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)

    tokenizer = bridge.processor.tokenizer
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )
    batch_size = state.shape[0]
    language_ids = encoded["input_ids"].to(device).expand(batch_size, -1)
    language_mask = encoded["attention_mask"].to(device).bool().expand(batch_size, -1)

    embeddings, pad_masks, attention_masks = model.embed_prefix(
        images, image_masks, language_ids, language_mask, state=state
    )
    layout = prefix_layout(policy, len(images), language_ids.shape[1])
    if embeddings.shape[1] != layout["total_tokens"]:
        raise ValueError(
            f"prefix has {embeddings.shape[1]} tokens, layout predicts {layout['total_tokens']}; "
            "the span arithmetic would pool the wrong positions"
        )

    hidden, _ = bridge.forward(
        attention_mask=make_att_2d_masks(pad_masks, attention_masks),
        position_ids=torch.cumsum(pad_masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    prefix = hidden[0].float()

    pooled = [
        pool_spatial_tokens(prefix[:, start:end], output_size)
        for start, end in layout["camera_spans"]
    ]
    state_token = prefix[:, layout["state_index"]]
    return torch.cat([*pooled, state_token], dim=-1).float()


def prompt_swap_relative_error(
    policy: Any,
    batch: dict[str, torch.Tensor],
    prompt: str,
    other_prompt: str,
    output_size: int = SPATIAL_GRID_SIZE,
) -> float:
    from smolvla_state_contract import relative_error

    left = extract_language_conditioned_spatial_state_feature(policy, batch, prompt, output_size)
    right = extract_language_conditioned_spatial_state_feature(
        policy, batch, other_prompt, output_size
    )
    return relative_error(left, right)
