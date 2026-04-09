"""LoRA configuration and layer-group parameter mapping."""

from __future__ import annotations

import re
from collections import defaultdict

from peft import LoraConfig, get_peft_model, TaskType

from dreamlora.config import LoRAConfig, LayerGroupConfig


def setup_lora(model, lora_config: LoRAConfig):
    """Apply uniform LoRA to all layers (Phase 1 baseline)."""
    peft_config = LoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    return model


_LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")


def get_layer_group_params(
    model,
    layer_groups: list[LayerGroupConfig],
) -> dict[str, list[tuple[str, "torch.nn.Parameter"]]]:
    """Map model parameters to layer groups based on layer index.

    Returns dict: group_name -> list of (param_name, param) for LoRA params
    in that group's layer range.
    """
    group_params: dict[str, list[tuple[str, "torch.nn.Parameter"]]] = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Only LoRA params (lora_A, lora_B)
        if "lora_" not in name:
            continue

        match = _LAYER_PATTERN.search(name)
        if match is None:
            continue

        layer_idx = int(match.group(1))
        for group in layer_groups:
            if group.layer_start <= layer_idx <= group.layer_end:
                group_params[group.name].append((name, param))
                break

    return dict(group_params)


def get_layer_count(model) -> int:
    """Get the number of transformer layers in the model."""
    if hasattr(model, "config"):
        cfg = model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
    raise ValueError("Cannot determine number of layers from model config")
