"""Merge scheduling for sleep cycles."""

from __future__ import annotations

from dreamlora.config import LayerGroupConfig


def groups_due_for_merge(
    layer_groups: list[LayerGroupConfig],
    cycle_number: int,
) -> list[LayerGroupConfig]:
    """Return layer groups that should be merged at the given cycle number."""
    due = []
    for group in layer_groups:
        if group.merge_every_n_cycles is None:
            continue  # Permanent LoRA, never merge
        if cycle_number > 0 and cycle_number % group.merge_every_n_cycles == 0:
            due.append(group)
    return due


def next_merge_cycle(
    layer_groups: list[LayerGroupConfig],
    current_cycle: int,
) -> dict[str, int | None]:
    """For each group, compute the next cycle number when merge is due.

    Returns dict: group_name -> next merge cycle (None if never).
    """
    result = {}
    for group in layer_groups:
        if group.merge_every_n_cycles is None:
            result[group.name] = None
        else:
            # Next multiple of merge_every_n_cycles after current_cycle
            remainder = current_cycle % group.merge_every_n_cycles
            if remainder == 0 and current_cycle > 0:
                result[group.name] = current_cycle + group.merge_every_n_cycles
            else:
                result[group.name] = current_cycle + (group.merge_every_n_cycles - remainder)
    return result
