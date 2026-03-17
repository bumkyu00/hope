"""Direct recall evaluation: ask about each memory span and check accuracy."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dreamlora.data.memory_store import MemorySpan


@dataclass
class RecallResult:
    span_id: str
    question: str
    expected_keywords: list[str]
    model_response: str
    hit: bool
    score: float  # 0.0 - 1.0


# Question templates per category
_QUESTION_TEMPLATES = {
    "identity": "사용자의 이름과 직업이 뭐야?",
    "language": "사용자가 선호하는 언어는?",
    "coding": "사용자의 프로그래밍 선호도에 대해 알려줘.",
    "editor": "사용자가 어떤 에디터를 써?",
    "diet": "사용자의 식습관은?",
    "allergy": "사용자에게 알레르기가 있어?",
    "hobby": "사용자의 취미는?",
    "music": "사용자가 좋아하는 음악은?",
    "pet": "사용자가 키우는 반려동물에 대해 알려줘.",
    "work": "사용자의 현재 업무에 대해 알려줘.",
    "schedule": "사용자의 일상 스케줄은?",
    "communication": "사용자가 선호하는 소통 방식은?",
    "learning": "사용자가 최근 배우고 있는 것은?",
    "travel": "사용자가 좋아하는 여행지는?",
    "coffee": "사용자의 커피 습관은?",
    "family": "사용자의 가족에 대해 알려줘.",
    "reading": "사용자가 좋아하는 책/장르는?",
    "fear": "사용자가 두려워하는 것은?",
    "goal": "사용자의 올해 목표는?",
}

_DEFAULT_QUESTION = "이 기억에 대해 알려줘: {content}"


def generate_recall_questions(spans: list[MemorySpan]) -> list[tuple[str, MemorySpan]]:
    """Generate direct recall questions for each memory span."""
    questions = []
    for span in spans:
        # Extract category from context field
        category = ""
        if span.context.startswith("category:"):
            category = span.context.split(":", 1)[1]

        question = _QUESTION_TEMPLATES.get(
            category,
            _DEFAULT_QUESTION.format(content=span.content[:30]),
        )
        questions.append((question, span))
    return questions


def extract_keywords(content: str) -> list[str]:
    """Extract key terms from memory content for matching."""
    # Remove common particles and short words
    words = re.findall(r"[가-힣a-zA-Z0-9]+", content)
    return [w for w in words if len(w) >= 2]


def evaluate_recall(
    model,
    tokenizer,
    spans: list[MemorySpan],
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> list[RecallResult]:
    """Evaluate direct recall for each memory span."""
    from dreamlora.data.formats import format_chatml

    questions = generate_recall_questions(spans)
    results = []

    for question, span in questions:
        messages = [
            {"role": "user", "content": question},
        ]
        prompt = format_chatml(messages, add_generation_prompt=True, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with __import__("torch").no_grad():
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

        keywords = extract_keywords(span.content)
        hits = sum(1 for kw in keywords if kw.lower() in response.lower())
        score = hits / max(len(keywords), 1)
        hit = score >= 0.5  # At least half the keywords present

        results.append(RecallResult(
            span_id=span.span_id,
            question=question,
            expected_keywords=keywords,
            model_response=response,
            hit=hit,
            score=score,
        ))

    return results


def compute_recall_accuracy(results: list[RecallResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit) / len(results)
