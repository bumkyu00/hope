"""Unified benchmark runner combining all evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from dreamlora.data.memory_store import MemoryStore
from dreamlora.eval.recall import evaluate_recall, compute_recall_accuracy
from dreamlora.eval.utilization import evaluate_utilization, compute_utilization_accuracy
from dreamlora.eval.general import evaluate_mmlu


@dataclass
class BenchmarkResults:
    recall_accuracy: float
    utilization_accuracy: float
    mmlu_accuracy: float
    recall_details: list[dict]
    utilization_details: list[dict]
    mmlu_details: list[dict]

    def summary(self) -> dict:
        return {
            "recall_accuracy": self.recall_accuracy,
            "utilization_accuracy": self.utilization_accuracy,
            "mmlu_accuracy": self.mmlu_accuracy,
        }


def run_benchmark(
    model,
    tokenizer,
    memory_store: MemoryStore,
    mmlu_num_questions: int = 100,
    mmlu_seed: int = 42,
    device: str = "cuda",
) -> BenchmarkResults:
    """Run full evaluation suite."""
    spans = memory_store.list_all()

    # Recall
    recall_results = evaluate_recall(model, tokenizer, spans, device=device)
    recall_acc = compute_recall_accuracy(recall_results)

    # Utilization
    util_results = evaluate_utilization(model, tokenizer, spans, device=device)
    util_acc = compute_utilization_accuracy(util_results)

    # MMLU
    mmlu_acc, mmlu_results = evaluate_mmlu(
        model, tokenizer,
        num_questions=mmlu_num_questions,
        seed=mmlu_seed,
        device=device,
    )

    return BenchmarkResults(
        recall_accuracy=recall_acc,
        utilization_accuracy=util_acc,
        mmlu_accuracy=mmlu_acc,
        recall_details=[asdict(r) for r in recall_results],
        utilization_details=[asdict(r) for r in util_results],
        mmlu_details=[asdict(r) for r in mmlu_results],
    )


def save_benchmark_results(results: BenchmarkResults, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(results), f, ensure_ascii=False, indent=2)
