"""Tests for dream generation and validation."""

import pytest

from dreamlora.data.memory_store import MemorySpan
from dreamlora.data.formats import format_memory_span, build_dream_messages
from dreamlora.dream.templates import ALL_TEMPLATES, DIRECT_UTILIZATION_TEMPLATES
from dreamlora.dream.validator import validate_dream, ValidationResult
from dreamlora.dream.mixer import DreamPool
from dreamlora.data.user_profile import generate_dreams_from_profile, generate_profile


class TestDreamTemplates:
    def test_all_scenario_types_exist(self):
        expected = {"original_replay", "direct_utilization", "cross_memory",
                    "temporal_context", "neg_correction"}
        assert set(ALL_TEMPLATES.keys()) == expected

    def test_each_template_has_required_fields(self):
        for scenario_type, templates in ALL_TEMPLATES.items():
            for t in templates:
                assert hasattr(t, "system_prompt")
                assert hasattr(t, "user_template")
                assert hasattr(t, "think_template")
                assert hasattr(t, "response_template")


class TestDreamValidator:
    def _make_dream(self, include_think=True, include_assistant=True, content="test response"):
        messages = [
            {"role": "system", "content": "memory context"},
            {"role": "user", "content": "tell me something"},
        ]
        if include_assistant:
            if include_think:
                messages.append({
                    "role": "assistant",
                    "content": f"<think>\nthinking about it\n</think>\n{content}",
                })
            else:
                messages.append({"role": "assistant", "content": content})
        return messages

    def test_valid_dream_with_think(self):
        dream = self._make_dream(include_think=True)
        result = validate_dream(dream)
        assert result.valid
        assert result.checks["has_think_block"]

    def test_valid_dream_without_think(self):
        dream = self._make_dream(include_think=False)
        result = validate_dream(dream)
        assert result.valid  # Think block is optional
        assert not result.checks["has_think_block"]

    def test_invalid_no_assistant(self):
        dream = self._make_dream(include_assistant=False)
        result = validate_dream(dream)
        assert not result.valid

    def test_invalid_empty_response(self):
        # Directly construct a dream with truly empty assistant content
        dream = [
            {"role": "system", "content": "ctx"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ]
        result = validate_dream(dream)
        assert not result.valid

    def test_memory_utilization_check(self):
        span = MemorySpan(
            span_id="test",
            timestamp="2026-03-01T10:00:00+09:00",
            level=3,
            sentiment="positive",
            content="Python을 좋아하고 type hint를 항상 사용한다",
        )
        # Dream that mentions the memory
        dream = [
            {"role": "system", "content": "memory: Python을 좋아함"},
            {"role": "user", "content": "코드 짜줘"},
            {"role": "assistant", "content": "<think>Python 선호</think>\ndef hello(): pass"},
        ]
        result = validate_dream(dream, span)
        assert result.checks["memory_utilized"]

    def test_memory_not_utilized(self):
        span = MemorySpan(
            span_id="test",
            timestamp="2026-03-01T10:00:00+09:00",
            level=3,
            sentiment="positive",
            content="Python을 좋아하고 type hint를 항상 사용한다",
        )
        # Dream that doesn't reference the memory at all
        dream = [
            {"role": "system", "content": "general assistant"},
            {"role": "user", "content": "날씨 어때?"},
            {"role": "assistant", "content": "<think>날씨</think>\n오늘 맑습니다."},
        ]
        result = validate_dream(dream, span)
        assert not result.checks["memory_utilized"]


class TestDreamPool:
    def test_add_and_mix(self):
        pool = DreamPool()
        dreams = [[{"role": "user", "content": f"dream {i}"}] for i in range(10)]
        pool.add_new_dreams(dreams)

        assert pool.new_count == 10
        mixed = pool.mix(new_ratio=1.0, max_total=5, seed=42)
        assert len(mixed) == 5

    def test_new_old_ratio(self):
        pool = DreamPool()

        # Add old dreams
        old = [[{"role": "user", "content": f"old {i}"}] for i in range(20)]
        pool.archive_dreams(old, "span_001")

        # Add new dreams
        new = [[{"role": "user", "content": f"new {i}"}] for i in range(20)]
        pool.add_new_dreams(new)

        mixed = pool.mix(new_ratio=0.7, max_total=10, seed=42)
        assert len(mixed) == 10

    def test_clear_new(self):
        pool = DreamPool()
        pool.add_new_dreams([["dream"]])
        assert pool.new_count == 1
        pool.clear_new()
        assert pool.new_count == 0

    def test_pool_pruning(self):
        pool = DreamPool(max_pool_size=5)
        for i in range(10):
            pool.archive_dreams([[f"dream_{i}"]], f"span_{i}")
        assert pool.old_count <= 5


class TestProfileDreamGeneration:
    def test_generate_dreams_from_profile(self):
        profile = generate_profile()
        dreams = generate_dreams_from_profile(
            items=profile,
            dreams_per_level={1: 2, 2: 2, 3: 2, 4: 2, 5: 2},
            seed=42,
        )
        assert len(dreams) > 0
        # Each dream should be a list of message dicts
        for dream in dreams:
            assert isinstance(dream, list)
            assert all(isinstance(m, dict) for m in dream)
            assert all("role" in m and "content" in m for m in dream)

    def test_dreams_contain_memory_context(self):
        profile = generate_profile()
        dreams = generate_dreams_from_profile(
            items=profile[:1],  # Just first item
            dreams_per_level={5: 3},
            seed=42,
        )
        # At least one dream should reference the memory
        all_text = " ".join(
            m["content"]
            for dream in dreams
            for m in dream
        )
        # The first profile item mentions "김지현" and "소프트웨어 엔지니어"
        assert "김지현" in all_text or "소프트웨어" in all_text or "엔지니어" in all_text
