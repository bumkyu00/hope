"""General capability evaluation using MMLU subset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MMLUResult:
    question_id: int
    subject: str
    question: str
    choices: list[str]
    correct_answer: str
    model_answer: str
    correct: bool


# Minimal built-in MMLU-style questions for quick testing without external datasets
_BUILTIN_QUESTIONS = [
    {
        "subject": "computer_science",
        "question": "What is the time complexity of binary search?",
        "choices": ["O(n)", "O(log n)", "O(n log n)", "O(n^2)"],
        "answer": "B",
    },
    {
        "subject": "computer_science",
        "question": "Which data structure uses FIFO ordering?",
        "choices": ["Stack", "Queue", "Tree", "Graph"],
        "answer": "B",
    },
    {
        "subject": "mathematics",
        "question": "What is the derivative of x^3?",
        "choices": ["x^2", "3x^2", "3x", "x^3"],
        "answer": "B",
    },
    {
        "subject": "mathematics",
        "question": "What is the integral of 2x?",
        "choices": ["x", "x^2", "x^2 + C", "2x^2"],
        "answer": "C",
    },
    {
        "subject": "physics",
        "question": "What is the SI unit of force?",
        "choices": ["Joule", "Watt", "Newton", "Pascal"],
        "answer": "C",
    },
]


def load_mmlu_questions(
    num_questions: int = 100,
    seed: int = 42,
    dataset_path: str | None = None,
) -> list[dict]:
    """Load MMLU questions. Falls back to built-in set if no dataset available."""
    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path) as f:
            questions = json.load(f)
        rng = random.Random(seed)
        rng.shuffle(questions)
        return questions[:num_questions]

    # Try loading from HF datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(num_questions, len(ds)))
        questions = []
        for idx in indices:
            row = ds[idx]
            questions.append({
                "subject": row["subject"],
                "question": row["question"],
                "choices": row["choices"],
                "answer": chr(65 + row["answer"]),  # 0->A, 1->B, etc.
            })
        return questions
    except Exception:
        pass

    # Fallback to built-in
    rng = random.Random(seed)
    result = _BUILTIN_QUESTIONS * (num_questions // len(_BUILTIN_QUESTIONS) + 1)
    rng.shuffle(result)
    return result[:num_questions]


def evaluate_mmlu(
    model,
    tokenizer,
    num_questions: int = 100,
    seed: int = 42,
    device: str = "cuda",
    dataset_path: str | None = None,
) -> tuple[float, list[MMLUResult]]:
    """Run MMLU evaluation. Returns (accuracy, results)."""
    import torch
    from dreamlora.data.formats import format_chatml

    questions = load_mmlu_questions(num_questions, seed, dataset_path)
    results = []

    for i, q in enumerate(questions):
        choices_text = "\n".join(
            f"{chr(65+j)}. {c}" for j, c in enumerate(q["choices"])
        )
        prompt_text = (
            f"Answer the following multiple choice question. "
            f"Reply with just the letter (A, B, C, or D).\n\n"
            f"Question: {q['question']}\n{choices_text}\n\nAnswer:"
        )

        messages = [{"role": "user", "content": prompt_text}]
        prompt = format_chatml(messages, add_generation_prompt=True, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Extract letter answer
        model_answer = ""
        for ch in response.upper():
            if ch in "ABCD":
                model_answer = ch
                break

        correct = model_answer == q["answer"]
        results.append(MMLUResult(
            question_id=i,
            subject=q["subject"],
            question=q["question"],
            choices=q["choices"],
            correct_answer=q["answer"],
            model_answer=model_answer,
            correct=correct,
        ))

    accuracy = sum(1 for r in results if r.correct) / max(len(results), 1)
    return accuracy, results
