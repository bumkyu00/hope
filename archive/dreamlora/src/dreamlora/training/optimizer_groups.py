"""Per-group optimizer creation for CMS differential updates."""

from __future__ import annotations

import torch
from torch.optim import AdamW

from dreamlora.config import LayerGroupConfig


def create_group_optimizers(
    group_params: dict[str, list[tuple[str, torch.nn.Parameter]]],
    layer_groups: list[LayerGroupConfig],
    weight_decay: float = 0.01,
) -> dict[str, AdamW]:
    """Create independent AdamW optimizer for each layer group.

    Args:
        group_params: dict from group_name -> list of (name, param)
        layer_groups: list of LayerGroupConfig with lr per group
        weight_decay: global weight decay

    Returns:
        dict from group_name -> AdamW optimizer
    """
    group_config_map = {g.name: g for g in layer_groups}
    optimizers = {}

    for group_name, named_params in group_params.items():
        config = group_config_map[group_name]
        params = [p for _, p in named_params]
        if not params:
            continue
        optimizers[group_name] = AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    return optimizers


def reset_optimizer_state(optimizer: AdamW) -> None:
    """Reset optimizer state (momentum buffers) after merge."""
    optimizer.state.clear()
