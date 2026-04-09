"""Indirect utilization evaluation: check if memories influence behavior."""

from __future__ import annotations

from dataclasses import dataclass

from dreamlora.data.memory_store import MemorySpan


@dataclass
class UtilizationResult:
    span_id: str
    scenario: str
    expected_behavior: str
    check_keywords: list[str]
    model_response: str
    utilized: bool
    score: float


# Scenarios that test indirect memory utilization (not asking directly)
_UTILIZATION_SCENARIOS = {
    "coding": {
        "scenario": "리스트에서 중복을 제거하는 코드를 짜줘.",
        "expected_behavior": "Python으로 답변 (사용자가 Python 선호)",
        "check_keywords": ["python", "def", "set", "list"],
    },
    "language": {
        "scenario": "Explain how HTTP caching works.",
        "expected_behavior": "한국어로 답변 (사용자가 한국어 선호)",
        "check_keywords": ["캐싱", "캐시", "HTTP", "헤더"],
    },
    "communication": {
        "scenario": "딥러닝의 역사에 대해 알려줘.",
        "expected_behavior": "간결하게 답변 (장황한 설명 싫어함)",
        "check_keywords": [],  # Check by response length instead
    },
    "diet": {
        "scenario": "점심 메뉴 추천해줘.",
        "expected_behavior": "채식 메뉴 추천 (채식주의자)",
        "check_keywords": ["채식", "야채", "두부", "샐러드", "비건"],
    },
    "allergy": {
        "scenario": "간식으로 뭐가 좋을까?",
        "expected_behavior": "견과류 제외 추천 (견과류 알레르기)",
        "check_keywords": [],  # Check that "땅콩", "견과" are NOT recommended positively
    },
}


def evaluate_utilization(
    model,
    tokenizer,
    spans: list[MemorySpan],
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> list[UtilizationResult]:
    """Evaluate indirect memory utilization."""
    from dreamlora.data.formats import format_chatml
    import torch

    results = []

    for span in spans:
        category = ""
        if span.context.startswith("category:"):
            category = span.context.split(":", 1)[1]

        if category not in _UTILIZATION_SCENARIOS:
            continue

        scenario_data = _UTILIZATION_SCENARIOS[category]
        messages = [{"role": "user", "content": scenario_data["scenario"]}]
        prompt = format_chatml(messages, add_generation_prompt=True, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        check_kw = scenario_data["check_keywords"]
        if check_kw:
            hits = sum(1 for kw in check_kw if kw.lower() in response.lower())
            score = hits / len(check_kw)
            utilized = score >= 0.3
        elif category == "communication":
            # Short response = success (< 300 chars)
            score = 1.0 if len(response) < 300 else max(0.0, 1.0 - (len(response) - 300) / 500)
            utilized = len(response) < 400
        elif category == "allergy":
            # Negative check: 견과류 should not be recommended
            bad_words = ["땅콩", "아몬드", "호두", "캐슈넛", "견과"]
            bad_hits = sum(1 for w in bad_words if w in response)
            score = 1.0 if bad_hits == 0 else max(0.0, 1.0 - bad_hits * 0.3)
            utilized = bad_hits == 0
        else:
            score = 0.0
            utilized = False

        results.append(UtilizationResult(
            span_id=span.span_id,
            scenario=scenario_data["scenario"],
            expected_behavior=scenario_data["expected_behavior"],
            check_keywords=check_kw,
            model_response=response,
            utilized=utilized,
            score=score,
        ))

    return results


def compute_utilization_accuracy(results: list[UtilizationResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.utilized) / len(results)
