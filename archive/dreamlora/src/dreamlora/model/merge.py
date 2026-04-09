"""Layer-group partial LoRA merge and reset.

PEFT's merge_and_unload() only supports full merge. This module implements
manual per-group partial merge: base_weight += lora_B @ lora_A * (alpha/rank),
then resets LoRA weights and optimizer state for that group.
"""

from __future__ import annotations

import logging
import re

import torch

from dreamlora.config import LayerGroupConfig, LoRAConfig
from dreamlora.training.optimizer_groups import reset_optimizer_state

logger = logging.getLogger(__name__)

_LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")


def _find_lora_pairs(model, layer_start: int, layer_end: int, target_modules: list[str]):
    """Find LoRA A/B weight pairs for layers in the given range.

    Returns list of (module_path, lora_A, lora_B, scaling) tuples.
    """
    pairs = []

    for name, module in model.named_modules():
        match = _LAYER_PATTERN.search(name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if not (layer_start <= layer_idx <= layer_end):
            continue

        # Check if this is a LoRA-wrapped module
        if not hasattr(module, "lora_A"):
            continue

        # Check if it's a target module
        module_type = name.split(".")[-1]
        if module_type not in target_modules:
            continue

        # Get the active adapter's LoRA weights
        for adapter_name in module.lora_A:
            lora_A = module.lora_A[adapter_name].weight  # (rank, in_features)
            lora_B = module.lora_B[adapter_name].weight  # (out_features, rank)
            scaling = module.scaling[adapter_name]
            base_layer = module.get_base_layer()

            pairs.append({
                "name": name,
                "adapter": adapter_name,
                "lora_A": lora_A,
                "lora_B": lora_B,
                "scaling": scaling,
                "base_weight": base_layer.weight,
                "module": module,
            })

    return pairs


def merge_lora_group(
    model,
    group: LayerGroupConfig,
    lora_config: LoRAConfig,
    optimizer=None,
) -> int:
    """Merge LoRA weights into base model for a specific layer group.

    Steps:
    1. For each LoRA module in the layer range:
       base_weight += lora_B @ lora_A * scaling
    2. Reset lora_A and lora_B to zero
    3. Reset optimizer state if provided

    Args:
        model: PEFT model
        group: layer group config specifying range
        lora_config: LoRA config for target modules
        optimizer: optional optimizer whose state should be reset

    Returns:
        Number of modules merged
    """
    pairs = _find_lora_pairs(
        model, group.layer_start, group.layer_end, lora_config.target_modules
    )

    if not pairs:
        logger.warning(f"No LoRA pairs found for group {group.name} "
                       f"(layers {group.layer_start}-{group.layer_end})")
        return 0

    merged_count = 0
    with torch.no_grad():
        for pair in pairs:
            lora_A = pair["lora_A"]  # (rank, in_features)
            lora_B = pair["lora_B"]  # (out_features, rank)
            scaling = pair["scaling"]
            base_weight = pair["base_weight"]

            # Merge: base_weight += lora_B @ lora_A * scaling
            delta = (lora_B @ lora_A) * scaling
            base_weight.add_(delta.to(base_weight.dtype))

            # Reset LoRA weights to zero
            lora_A.zero_()
            lora_B.zero_()

            merged_count += 1
            logger.debug(f"Merged {pair['name']} (adapter={pair['adapter']})")

    # Reset optimizer state
    if optimizer is not None:
        reset_optimizer_state(optimizer)

    logger.info(
        f"Merged group {group.name}: {merged_count} modules "
        f"(layers {group.layer_start}-{group.layer_end})"
    )
    return merged_count


def merge_groups_by_schedule(
    model,
    layer_groups: list[LayerGroupConfig],
    lora_config: LoRAConfig,
    cycle_number: int,
    optimizers: dict | None = None,
) -> list[str]:
    """Check merge schedule and merge groups that are due.

    Returns list of group names that were merged.
    """
    merged_groups = []

    for group in layer_groups:
        if group.merge_every_n_cycles is None:
            continue  # Permanent LoRA, never merge
        if cycle_number > 0 and cycle_number % group.merge_every_n_cycles == 0:
            opt = optimizers.get(group.name) if optimizers else None
            count = merge_lora_group(model, group, lora_config, optimizer=opt)
            if count > 0:
                merged_groups.append(group.name)

    return merged_groups
