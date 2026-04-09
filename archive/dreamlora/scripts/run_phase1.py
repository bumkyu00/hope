"""Phase 1: Basic memory encoding validation.

Compares:
  (A) Original spans only SFT
  (B) Original + dream mixed SFT
"""

import json
import logging
from pathlib import Path

import torch

from dreamlora.config import ExperimentConfig
from dreamlora.data.memory_store import MemoryStore
from dreamlora.data.user_profile import (
    generate_profile,
    populate_memory_store,
    generate_dreams_from_profile,
)
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import build_dream_messages, format_memory_span
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora
from dreamlora.training.sft_trainer import SFTTrainer
from dreamlora.eval.benchmark import run_benchmark, save_benchmark_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_span_only_dreams(store: MemoryStore) -> list[list[dict[str, str]]]:
    """Build dream sequences from original spans only (Condition A)."""
    dreams = []
    for span in store.list_all():
        memory_ctx = format_memory_span(
            content=span.content,
            level=span.level,
            sentiment=span.sentiment,
        )
        messages = build_dream_messages(
            memory_context=memory_ctx,
            user_query=f"{span.content[:20]}에 대해 알려줘.",
            assistant_response=f"네, {span.content}",
            think_block=f"사용자의 기억: {span.content}. 이를 활용해 답변하겠다.",
        )
        dreams.append(messages)
    return dreams


def run_condition(
    condition_name: str,
    dreams: list[list[dict[str, str]]],
    config: ExperimentConfig,
    store: MemoryStore,
    output_dir: Path,
    target_steps: int | None = None,
):
    logger.info(f"=== Condition {condition_name} ===")
    logger.info(f"Number of dream sequences: {len(dreams)}")

    # Load fresh model + LoRA for each condition
    model, tokenizer = load_model_and_tokenizer(config.model)
    model = setup_lora(model, config.lora)

    # Create dataset
    dataset = DreamDataset(dreams, tokenizer, max_length=config.model.max_seq_len)
    logger.info(f"Dataset size: {len(dataset)} examples")

    # Adjust epochs to match target step count if specified
    if target_steps is not None:
        steps_per_epoch = max(len(dataset) // config.training_batch_size, 1)
        adjusted_epochs = max((target_steps + steps_per_epoch - 1) // steps_per_epoch, 1)
        config = config.model_copy(deep=True)
        config.num_epochs = adjusted_epochs
        logger.info(f"Adjusted epochs to {adjusted_epochs} (~{adjusted_epochs * steps_per_epoch} steps) to match target {target_steps} steps")

    # Train
    cond_dir = output_dir / condition_name
    trainer = SFTTrainer(model, tokenizer, dataset, config, output_dir=str(cond_dir))
    result = trainer.train()
    logger.info(f"Training loss: {result['train_loss']:.4f}")

    # Save
    trainer.save(str(cond_dir / "final"))

    # Evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = run_benchmark(
        model, tokenizer, store,
        mmlu_num_questions=config.eval.mmlu_num_questions,
        mmlu_seed=config.eval.mmlu_seed,
        device=device,
    )
    save_benchmark_results(benchmark, cond_dir / "benchmark.json")

    summary = benchmark.summary()
    logger.info(f"Results for {condition_name}: {json.dumps(summary, indent=2)}")
    return summary


def main():
    config = ExperimentConfig.from_yaml("configs/phase1_sft.yaml")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup memory store with synthetic profile
    store = MemoryStore(output_dir / "memories.jsonl")
    populate_memory_store(store, seed=config.seed)
    logger.info(f"Memory store: {len(store)} spans")

    # Generate dreams
    profile = generate_profile(seed=config.seed)
    all_dreams = generate_dreams_from_profile(
        items=profile,
        dreams_per_level=config.dream.dreams_per_level,
        seed=config.seed,
    )
    span_only_dreams = build_span_only_dreams(store)

    # Compute target steps from B to match A's training budget
    mixed_dreams = span_only_dreams + all_dreams
    b_steps_per_epoch = max(len(mixed_dreams) // config.training_batch_size, 1)
    target_steps = b_steps_per_epoch * config.num_epochs
    logger.info(f"B will do ~{target_steps} steps ({len(mixed_dreams)} dreams, {config.num_epochs} epoch)")
    logger.info(f"A will match ~{target_steps} steps ({len(span_only_dreams)} dreams, adjusted epochs)")

    # Condition A: Original spans only (more epochs to match B's step count)
    results_a = run_condition("A_span_only", span_only_dreams, config, store, output_dir,
                              target_steps=target_steps)

    # Condition B: Original + dream mixed
    results_b = run_condition("B_dream_mixed", mixed_dreams, config, store, output_dir)

    # Summary comparison
    logger.info("=" * 60)
    logger.info("Phase 1 Results Comparison:")
    logger.info(f"  Condition A (span only): {json.dumps(results_a, indent=2)}")
    logger.info(f"  Condition B (dream mix): {json.dumps(results_b, indent=2)}")

    # Save comparison
    with open(output_dir / "comparison.json", "w") as f:
        json.dump({"A_span_only": results_a, "B_dream_mixed": results_b}, f, indent=2)


if __name__ == "__main__":
    main()
