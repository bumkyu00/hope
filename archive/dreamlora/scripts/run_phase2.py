"""Phase 2: 30-day simulation with full DreamLoRA pipeline."""

import logging
from pathlib import Path

from dreamlora.config import ExperimentConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora
from dreamlora.simulation.day_simulator import run_30day_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = ExperimentConfig.from_yaml("configs/phase2_simulation.yaml")
    output_dir = Path(config.output_dir)

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model)
    model = setup_lora(model, config.lora)

    logger.info("Starting 30-day simulation...")
    results = run_30day_simulation(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=output_dir,
        evaluate_every=5,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Simulation Complete!")
    eval_days = [r for r in results if r.get("eval_result")]
    for r in eval_days:
        logger.info(
            f"Day {r['day']}: "
            f"recall={r['eval_result']['recall_accuracy']:.2%}, "
            f"util={r['eval_result']['utilization_accuracy']:.2%}, "
            f"mmlu={r['eval_result']['mmlu_accuracy']:.2%}, "
            f"total_spans={r['total_spans']}"
        )


if __name__ == "__main__":
    main()
