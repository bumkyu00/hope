"""Phase 1.5: CMS differential update validation.

Compares:
  (A) Single LoRA, uniform SFT (all layers lr=1e-4)
  (B) 3-group CMS (L0-7/L8-19/L20-27, chunk 1/5/25, lr 2e-4/1e-4/5e-5)
  (C) 2-group CMS (L0-13/L14-27, chunk 1/13)
"""

import json
import logging
from pathlib import Path

import torch

from dreamlora.config import ExperimentConfig, LayerGroupConfig
from dreamlora.data.memory_store import MemoryStore
from dreamlora.data.user_profile import (
    generate_profile,
    populate_memory_store,
    generate_dreams_from_profile,
)
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora
from dreamlora.training.cms_trainer import CMSTrainer
from dreamlora.training.sft_trainer import SFTTrainer
from dreamlora.eval.benchmark import run_benchmark, save_benchmark_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Condition definitions
CONDITIONS = {
    "A_uniform": {
        "description": "Single LoRA, uniform SFT (all layers lr=1e-4)",
        "layer_groups": [
            LayerGroupConfig(
                name="uniform", layer_start=0, layer_end=31,
                learning_rate=1e-4, chunk_size=1,
            ),
        ],
        "use_cms": False,
    },
    "B_3group_cms": {
        "description": "3-group CMS (L0-7/L8-19/L20-27)",
        "layer_groups": [
            LayerGroupConfig(
                name="high_freq", layer_start=0, layer_end=7,
                learning_rate=2e-4, chunk_size=1, merge_every_n_cycles=7,
            ),
            LayerGroupConfig(
                name="mid_freq", layer_start=8, layer_end=19,
                learning_rate=1e-4, chunk_size=5, merge_every_n_cycles=21,
            ),
            LayerGroupConfig(
                name="low_freq", layer_start=20, layer_end=31,
                learning_rate=5e-5, chunk_size=25,
            ),
        ],
        "use_cms": True,
    },
    "C_2group_cms": {
        "description": "2-group CMS (L0-15/L16-31)",
        "layer_groups": [
            LayerGroupConfig(
                name="fast", layer_start=0, layer_end=15,
                learning_rate=1.5e-4, chunk_size=1, merge_every_n_cycles=14,
            ),
            LayerGroupConfig(
                name="slow", layer_start=16, layer_end=31,
                learning_rate=7e-5, chunk_size=13,
            ),
        ],
        "use_cms": True,
    },
}


def run_condition(
    name: str,
    condition: dict,
    dreams: list[list[dict[str, str]]],
    config: ExperimentConfig,
    store: MemoryStore,
    output_dir: Path,
) -> dict:
    logger.info(f"=== Condition {name}: {condition['description']} ===")

    # Load fresh model + LoRA
    model, tokenizer = load_model_and_tokenizer(config.model)
    model = setup_lora(model, config.lora)

    # Create dataset
    dataset = DreamDataset(dreams, tokenizer, max_length=config.model.max_seq_len)
    logger.info(f"Dataset size: {len(dataset)} examples")

    cond_dir = output_dir / name
    cond_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if condition["use_cms"]:
        # Override config layer groups
        config_copy = config.model_copy(deep=True)
        config_copy.cms.layer_groups = condition["layer_groups"]

        trainer = CMSTrainer(model, tokenizer, config_copy, log_dir=str(cond_dir / "tb_logs"))
        stats = trainer.train_dream_stream(dataset, device=device)

        # Save CMS analysis
        weight_changes = trainer.get_weight_changes()
        analysis = {
            "stats": stats,
            "weight_changes": weight_changes,
        }
        with open(cond_dir / "cms_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
    else:
        # Standard SFT (Condition A)
        sft = SFTTrainer(model, tokenizer, dataset, config, output_dir=str(cond_dir))
        result = sft.train()
        logger.info(f"SFT loss: {result['train_loss']:.4f}")

    # Save model
    model.save_pretrained(str(cond_dir / "final"))
    tokenizer.save_pretrained(str(cond_dir / "final"))

    # Evaluate
    benchmark = run_benchmark(
        model, tokenizer, store,
        mmlu_num_questions=config.eval.mmlu_num_questions,
        mmlu_seed=config.eval.mmlu_seed,
        device=device,
    )
    save_benchmark_results(benchmark, cond_dir / "benchmark.json")

    summary = benchmark.summary()
    logger.info(f"Results for {name}: {json.dumps(summary, indent=2)}")
    return summary


def main():
    config = ExperimentConfig.from_yaml("configs/phase1_5_cms.yaml")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup memory store
    store = MemoryStore(output_dir / "memories.jsonl")
    populate_memory_store(store, seed=config.seed)

    # Generate dreams (same data for all conditions)
    profile = generate_profile(seed=config.seed)
    dreams = generate_dreams_from_profile(
        items=profile,
        dreams_per_level=config.dream.dreams_per_level,
        seed=config.seed,
    )
    logger.info(f"Generated {len(dreams)} dream sequences")

    # Run all conditions
    all_results = {}
    for name, condition in CONDITIONS.items():
        all_results[name] = run_condition(
            name, condition, dreams, config, store, output_dir,
        )

    # Summary comparison
    logger.info("=" * 60)
    logger.info("Phase 1.5 Results Comparison:")
    for name, result in all_results.items():
        logger.info(f"  {name}: {json.dumps(result, indent=2)}")

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
