"""Sleep cycle state persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CycleState:
    cycle_number: int = 0
    total_dreams_trained: int = 0
    merged_groups: dict[str, list[int]] = field(default_factory=dict)  # group -> list of cycle nums
    spans_processed: list[str] = field(default_factory=list)
    checkpoint_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CycleState:
        return cls(**d)


class StateManager:
    """Persist and load sleep cycle state."""

    def __init__(self, state_file: str | Path):
        self.state_file = Path(state_file)
        self.state = CycleState()
        if self.state_file.exists():
            self._load()

    def _load(self) -> None:
        with open(self.state_file) as f:
            data = json.load(f)
        self.state = CycleState.from_dict(data)

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def advance_cycle(self) -> int:
        self.state.cycle_number += 1
        self.save()
        return self.state.cycle_number

    def record_merge(self, group_name: str) -> None:
        if group_name not in self.state.merged_groups:
            self.state.merged_groups[group_name] = []
        self.state.merged_groups[group_name].append(self.state.cycle_number)
        self.save()

    def record_dreams_trained(self, count: int) -> None:
        self.state.total_dreams_trained += count
        self.save()

    def record_checkpoint(self, path: str) -> None:
        self.state.checkpoint_path = path
        self.save()
