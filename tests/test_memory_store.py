"""Tests for JSONL-based MemoryStore CRUD operations."""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from dreamlora.data.memory_store import MemoryStore, MemorySpan
from dreamlora.data.formats import KST


@pytest.fixture
def tmp_store(tmp_path):
    return MemoryStore(tmp_path / "test_memories.jsonl")


class TestMemoryStoreAdd:
    def test_add_basic(self, tmp_store):
        span = tmp_store.add(content="사용자가 Python을 좋아한다", level=4, sentiment="positive")
        assert span.span_id.startswith("mem_")
        assert span.content == "사용자가 Python을 좋아한다"
        assert span.level == 4
        assert span.sentiment == "positive"
        assert span.dream_count == 0

    def test_add_with_custom_id(self, tmp_store):
        span = tmp_store.add(content="test", span_id="custom_001")
        assert span.span_id == "custom_001"

    def test_add_persists_to_file(self, tmp_store):
        tmp_store.add(content="persistent memory", level=3)
        # Reload from file
        store2 = MemoryStore(tmp_store.path)
        assert len(store2) == 1
        assert store2.list_all()[0].content == "persistent memory"

    def test_add_with_tags(self, tmp_store):
        span = tmp_store.add(content="test", tags=["coding", "python"])
        assert span.tags == ["coding", "python"]

    def test_add_multiple(self, tmp_store):
        tmp_store.add(content="memory 1")
        tmp_store.add(content="memory 2")
        tmp_store.add(content="memory 3")
        assert len(tmp_store) == 3


class TestMemoryStoreGet:
    def test_get_existing(self, tmp_store):
        span = tmp_store.add(content="findable", span_id="find_me")
        found = tmp_store.get("find_me")
        assert found is not None
        assert found.content == "findable"

    def test_get_nonexistent(self, tmp_store):
        assert tmp_store.get("nonexistent") is None


class TestMemoryStoreRemove:
    def test_remove_existing(self, tmp_store):
        tmp_store.add(content="to delete", span_id="del_001")
        assert tmp_store.remove("del_001") is True
        assert len(tmp_store) == 0

    def test_remove_nonexistent(self, tmp_store):
        assert tmp_store.remove("nonexistent") is False

    def test_remove_persists(self, tmp_store):
        tmp_store.add(content="ephemeral", span_id="eph_001")
        tmp_store.remove("eph_001")
        store2 = MemoryStore(tmp_store.path)
        assert len(store2) == 0


class TestMemoryStoreUpdate:
    def test_update_level(self, tmp_store):
        tmp_store.add(content="update me", level=2, span_id="upd_001")
        updated = tmp_store.update("upd_001", level=5)
        assert updated is not None
        assert updated.level == 5

    def test_update_sentiment(self, tmp_store):
        tmp_store.add(content="test", sentiment="neutral", span_id="upd_002")
        updated = tmp_store.update("upd_002", sentiment="positive")
        assert updated.sentiment == "positive"

    def test_update_nonexistent(self, tmp_store):
        assert tmp_store.update("nonexistent", level=3) is None

    def test_update_persists(self, tmp_store):
        tmp_store.add(content="test", level=1, span_id="upd_003")
        tmp_store.update("upd_003", level=5)
        store2 = MemoryStore(tmp_store.path)
        assert store2.get("upd_003").level == 5


class TestMemoryStoreFilter:
    @pytest.fixture(autouse=True)
    def populate(self, tmp_store):
        self.store = tmp_store
        t1 = datetime(2026, 3, 1, 10, 0, tzinfo=KST)
        t2 = datetime(2026, 3, 5, 10, 0, tzinfo=KST)
        t3 = datetime(2026, 3, 10, 10, 0, tzinfo=KST)
        tmp_store.add(content="low", level=1, sentiment="neutral", tags=["a"], timestamp=t1, span_id="f1")
        tmp_store.add(content="mid", level=3, sentiment="positive", tags=["a", "b"], timestamp=t2, span_id="f2")
        tmp_store.add(content="high", level=5, sentiment="negative", tags=["b"], timestamp=t3, span_id="f3")

    def test_filter_by_min_level(self):
        results = self.store.filter(min_level=3)
        assert len(results) == 2
        assert all(r.level >= 3 for r in results)

    def test_filter_by_max_level(self):
        results = self.store.filter(max_level=3)
        assert len(results) == 2

    def test_filter_by_sentiment(self):
        results = self.store.filter(sentiment="negative")
        assert len(results) == 1
        assert results[0].span_id == "f3"

    def test_filter_by_tags(self):
        results = self.store.filter(tags=["b"])
        assert len(results) == 2

    def test_filter_by_tags_intersection(self):
        results = self.store.filter(tags=["a", "b"])
        assert len(results) == 1
        assert results[0].span_id == "f2"

    def test_filter_by_date_range(self):
        results = self.store.filter(
            since="2026-03-04T00:00:00+09:00",
            until="2026-03-06T00:00:00+09:00",
        )
        assert len(results) == 1
        assert results[0].span_id == "f2"

    def test_filter_combined(self):
        results = self.store.filter(min_level=3, sentiment="positive")
        assert len(results) == 1
        assert results[0].span_id == "f2"


class TestMemoryStoreDreamCount:
    def test_increment_dream_count(self, tmp_store):
        tmp_store.add(content="test", span_id="dc_001")
        tmp_store.increment_dream_count("dc_001", by=5)
        span = tmp_store.get("dc_001")
        assert span.dream_count == 5

    def test_increment_multiple_times(self, tmp_store):
        tmp_store.add(content="test", span_id="dc_002")
        tmp_store.increment_dream_count("dc_002", by=3)
        tmp_store.increment_dream_count("dc_002", by=2)
        assert tmp_store.get("dc_002").dream_count == 5


class TestMemoryStoreIteration:
    def test_iterate(self, tmp_store):
        tmp_store.add(content="a", span_id="iter_1")
        tmp_store.add(content="b", span_id="iter_2")
        ids = [s.span_id for s in tmp_store]
        assert set(ids) == {"iter_1", "iter_2"}

    def test_len(self, tmp_store):
        assert len(tmp_store) == 0
        tmp_store.add(content="a")
        assert len(tmp_store) == 1
