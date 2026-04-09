"""Scenario-type prompt templates for dream generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DreamTemplate:
    scenario_type: str
    system_prompt: str
    user_template: str
    think_template: str
    response_template: str


# Original replay (20%) — direct re-encoding of the original span
ORIGINAL_REPLAY = DreamTemplate(
    scenario_type="original_replay",
    system_prompt="너는 사용자의 개인 비서다. 사용자에 대해 알고 있는 정보를 활용해서 답변하라.",
    user_template="{memory_content}에 대해 다시 이야기해줘.",
    think_template=(
        "사용자가 이전에 공유한 기억을 떠올려본다: {memory_content}. "
        "이 기억의 중요도는 level {level}이고, 감정은 {sentiment}이다. "
        "정확하게 기억하고 활용해야 한다."
    ),
    response_template="네, {memory_summary}",
)

# Direct utilization (40%) — use memory to answer a relevant query
DIRECT_UTILIZATION_TEMPLATES = [
    DreamTemplate(
        scenario_type="direct_utilization",
        system_prompt="너는 사용자의 개인 비서다. 사용자에 대해 알고 있는 정보를 활용해서 답변하라.",
        user_template="{query}",
        think_template=(
            "이 질문에 답하기 위해 사용자에 대해 알고 있는 것을 떠올린다: "
            "{memory_content}. 이를 바탕으로 맞춤형 답변을 해야 한다."
        ),
        response_template="{personalized_response}",
    ),
    DreamTemplate(
        scenario_type="direct_utilization",
        system_prompt="너는 사용자의 개인 비서다.",
        user_template="이것 좀 도와줘: {task}",
        think_template=(
            "사용자의 선호를 기억한다: {memory_content}. "
            "이 선호에 맞춰서 작업을 수행하겠다."
        ),
        response_template="{task_response}",
    ),
]

# Cross-memory (25%) — combine multiple memories
CROSS_MEMORY_TEMPLATES = [
    DreamTemplate(
        scenario_type="cross_memory",
        system_prompt="너는 사용자의 개인 비서다. 여러 기억을 종합적으로 활용하라.",
        user_template="{combined_query}",
        think_template=(
            "이 질문에 관련된 기억이 여러 개 있다: "
            "첫째, {memory1_content}. 둘째, {memory2_content}. "
            "이 기억들을 결합하면 더 좋은 답변을 할 수 있다."
        ),
        response_template="{combined_response}",
    ),
]

# Temporal context (15%) — use time information
TEMPORAL_CONTEXT_TEMPLATES = [
    DreamTemplate(
        scenario_type="temporal_context",
        system_prompt="너는 사용자의 개인 비서다. 시간 맥락을 활용하라.",
        user_template="[{current_time}] {temporal_query}",
        think_template=(
            "현재 시점은 {current_time}이다. "
            "사용자의 기억을 시간순으로 정리하면: {memory_content} (시점: {memory_time}). "
            "시간 경과를 고려해서 답변하겠다."
        ),
        response_template="{temporal_response}",
    ),
]

# NEG sentiment correction — mistake recognition and correction flow
NEG_CORRECTION_TEMPLATES = [
    DreamTemplate(
        scenario_type="neg_correction",
        system_prompt="너는 사용자의 개인 비서다.",
        user_template="{trigger_query}",
        think_template=(
            "잠깐, 관련된 부정적 기억이 있다: {memory_content}. "
            "이전에 {neg_event}가 있었으니 같은 실수를 반복하면 안 된다. "
            "{correction_reasoning}"
        ),
        response_template="{corrected_response}",
    ),
]

# All templates indexed by scenario type
ALL_TEMPLATES = {
    "original_replay": [ORIGINAL_REPLAY],
    "direct_utilization": DIRECT_UTILIZATION_TEMPLATES,
    "cross_memory": CROSS_MEMORY_TEMPLATES,
    "temporal_context": TEMPORAL_CONTEXT_TEMPLATES,
    "neg_correction": NEG_CORRECTION_TEMPLATES,
}
