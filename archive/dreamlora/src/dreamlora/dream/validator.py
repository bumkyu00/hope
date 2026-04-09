"""Dream quality validation: structure and memory utilization checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dreamlora.data.memory_store import MemorySpan


@dataclass
class ValidationResult:
    valid: bool
    checks: dict[str, bool]
    reason: str = ""


def validate_dream(
    dream_messages: list[dict[str, str]],
    source_span: MemorySpan | None = None,
) -> ValidationResult:
    """Validate a dream sequence for quality.

    Checks:
    1. Structure: has system, user, assistant roles in correct order
    2. Non-empty: assistant response is not empty
    3. Memory utilization: if source_span provided, checks content reference
    4. Think block: assistant response contains <think> block
    """
    checks = {}

    # 1. Structure check
    roles = [m["role"] for m in dream_messages]
    has_system = "system" in roles
    has_user = "user" in roles
    has_assistant = "assistant" in roles
    checks["has_required_roles"] = has_user and has_assistant

    # 2. Non-empty check
    assistant_msgs = [m for m in dream_messages if m["role"] == "assistant"]
    non_empty = any(len(m["content"].strip()) > 10 for m in assistant_msgs)
    checks["non_empty_response"] = non_empty

    # 3. Memory utilization check
    if source_span is not None:
        # Check if any part of the memory content appears in the messages
        # Split into tokens, stripping common Korean particles
        _PARTICLES = re.compile(r"[을를이가은는에서도의로으며고]$")
        raw_words = re.findall(r"[가-힣a-zA-Z0-9]+", source_span.content)
        keywords = set()
        for w in raw_words:
            keywords.add(w)
            stripped = _PARTICLES.sub("", w)
            if len(stripped) >= 2:
                keywords.add(stripped)
        keywords = [k for k in keywords if len(k) >= 2]

        all_text = " ".join(m["content"] for m in dream_messages)
        keyword_hits = sum(1 for k in keywords if k in all_text)
        utilization_ratio = keyword_hits / max(len(keywords), 1)
        checks["memory_utilized"] = utilization_ratio >= 0.2
    else:
        checks["memory_utilized"] = True  # Skip if no source span

    # 4. Think block check (optional but preferred)
    think_present = any("<think>" in m["content"] for m in assistant_msgs)
    checks["has_think_block"] = think_present

    # Overall validity: required roles + non-empty + memory utilized
    valid = (
        checks["has_required_roles"]
        and checks["non_empty_response"]
        and checks["memory_utilized"]
    )

    reason = ""
    if not valid:
        failed = [k for k, v in checks.items() if not v and k != "has_think_block"]
        reason = f"Failed checks: {', '.join(failed)}"

    return ValidationResult(valid=valid, checks=checks, reason=reason)


def filter_valid_dreams(
    dreams: list[list[dict[str, str]]],
    source_spans: list[MemorySpan] | None = None,
) -> list[list[dict[str, str]]]:
    """Filter dreams, keeping only valid ones."""
    valid_dreams = []
    for i, dream in enumerate(dreams):
        span = source_spans[i] if source_spans and i < len(source_spans) else None
        result = validate_dream(dream, span)
        if result.valid:
            valid_dreams.append(dream)
    return valid_dreams
