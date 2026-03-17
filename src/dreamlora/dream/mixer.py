"""Dream pool mixing: 70/30 new/old ratio and pool management."""

from __future__ import annotations

import random
from collections import defaultdict

from dreamlora.data.memory_store import MemorySpan


class DreamPool:
    """Manages dream sequences with new/old mixing ratio."""

    def __init__(self, max_pool_size: int = 1000):
        self.max_pool_size = max_pool_size
        # span_id -> list of dream message lists
        self._pool: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
        self._new_dreams: list[list[dict[str, str]]] = []

    def add_new_dreams(self, dreams: list[list[dict[str, str]]]) -> None:
        """Add newly generated dreams to the new pool."""
        self._new_dreams.extend(dreams)

    def archive_dreams(
        self,
        dreams: list[list[dict[str, str]]],
        span_id: str,
    ) -> None:
        """Move dreams to the archived pool after training."""
        self._pool[span_id].extend(dreams)
        # Prune if over max size
        self._prune()

    def _prune(self) -> None:
        """Keep pool size under limit by removing oldest entries."""
        total = sum(len(v) for v in self._pool.values())
        while total > self.max_pool_size and self._pool:
            # Remove oldest dreams (first entries) from spans with most dreams
            largest_span = max(self._pool, key=lambda k: len(self._pool[k]))
            if self._pool[largest_span]:
                self._pool[largest_span].pop(0)
                total -= 1
            if not self._pool[largest_span]:
                del self._pool[largest_span]

    def mix(
        self,
        new_ratio: float = 0.7,
        max_total: int = 200,
        seed: int = 42,
    ) -> list[list[dict[str, str]]]:
        """Mix new and old dreams according to ratio.

        Args:
            new_ratio: fraction of new dreams (default 0.7)
            max_total: maximum total dreams in the mix
            seed: random seed

        Returns:
            Mixed list of dream message lists
        """
        rng = random.Random(seed)

        n_new = int(max_total * new_ratio)
        n_old = max_total - n_new

        # Sample new dreams
        new = list(self._new_dreams)
        rng.shuffle(new)
        new_sample = new[:n_new]

        # Sample old dreams (flattened pool, weighted by recency)
        old_flat = []
        for span_dreams in self._pool.values():
            old_flat.extend(span_dreams)
        rng.shuffle(old_flat)
        old_sample = old_flat[:n_old]

        mixed = new_sample + old_sample
        rng.shuffle(mixed)
        return mixed

    def clear_new(self) -> None:
        """Clear the new dreams buffer after they've been used."""
        self._new_dreams.clear()

    @property
    def new_count(self) -> int:
        return len(self._new_dreams)

    @property
    def old_count(self) -> int:
        return sum(len(v) for v in self._pool.values())

    @property
    def total_count(self) -> int:
        return self.new_count + self.old_count
