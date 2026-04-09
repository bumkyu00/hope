"""JSONL-based tagged memory span store."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from dreamlora.data.formats import format_timestamp, parse_timestamp


@dataclass
class MemorySpan:
    span_id: str
    timestamp: str  # ISO 8601
    level: int  # 1-5
    sentiment: str  # "positive", "negative", "neutral"
    content: str
    context: str = ""
    tags: list[str] = field(default_factory=list)
    dream_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MemorySpan:
        return cls(**d)


class MemoryStore:
    """JSONL file-based memory span CRUD."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._spans: dict[str, MemorySpan] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                span = MemorySpan.from_dict(d)
                self._spans[span.span_id] = span

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for span in self._spans.values():
                f.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")

    def add(
        self,
        content: str,
        level: int = 3,
        sentiment: str = "neutral",
        context: str = "",
        tags: list[str] | None = None,
        timestamp: datetime | None = None,
        span_id: str | None = None,
    ) -> MemorySpan:
        span = MemorySpan(
            span_id=span_id or f"mem_{uuid.uuid4().hex[:8]}",
            timestamp=format_timestamp(timestamp),
            level=level,
            sentiment=sentiment,
            content=content,
            context=context,
            tags=tags or [],
            dream_count=0,
        )
        self._spans[span.span_id] = span
        self._save()
        return span

    def get(self, span_id: str) -> MemorySpan | None:
        return self._spans.get(span_id)

    def remove(self, span_id: str) -> bool:
        if span_id in self._spans:
            del self._spans[span_id]
            self._save()
            return True
        return False

    def update(self, span_id: str, **kwargs) -> MemorySpan | None:
        span = self._spans.get(span_id)
        if span is None:
            return None
        for k, v in kwargs.items():
            if hasattr(span, k):
                setattr(span, k, v)
        self._save()
        return span

    def list_all(self) -> list[MemorySpan]:
        return list(self._spans.values())

    def filter(
        self,
        min_level: int | None = None,
        max_level: int | None = None,
        sentiment: str | None = None,
        tags: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[MemorySpan]:
        results = []
        for span in self._spans.values():
            if min_level is not None and span.level < min_level:
                continue
            if max_level is not None and span.level > max_level:
                continue
            if sentiment is not None and span.sentiment != sentiment:
                continue
            if tags is not None and not set(tags).issubset(set(span.tags)):
                continue
            if since is not None:
                span_dt = parse_timestamp(span.timestamp)
                since_dt = parse_timestamp(since)
                if span_dt < since_dt:
                    continue
            if until is not None:
                span_dt = parse_timestamp(span.timestamp)
                until_dt = parse_timestamp(until)
                if span_dt > until_dt:
                    continue
            results.append(span)
        return results

    def increment_dream_count(self, span_id: str, by: int = 1) -> None:
        span = self._spans.get(span_id)
        if span:
            span.dream_count += by
            self._save()

    def __len__(self) -> int:
        return len(self._spans)

    def __iter__(self) -> Iterator[MemorySpan]:
        return iter(self._spans.values())
