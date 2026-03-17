"""Sleep cycle orchestrator: full pipeline from tag collection to CMS training."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from dreamlora.config import ExperimentConfig
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.memory_store import MemoryStore
from dreamlora.dream.generator import DreamGenerator, LocalDreamGenerator, APIDreamGenerator
from dreamlora.dream.mixer import DreamPool
from dreamlora.dream.validator import filter_valid_dreams
from dreamlora.model.merge import merge_groups_by_schedule
from dreamlora.sleep.scheduler import groups_due_for_merge
from dreamlora.sleep.state import StateManager
from dreamlora.training.cms_trainer import CMSTrainer

logger = logging.getLogger(__name__)


class SleepOrchestrator:
    """Orchestrates a single sleep cycle:

    1. Collect new tagged spans
    2. Generate dreams from spans
    3. Mix with dream pool (70% new / 30% old)
    4. CMS training on dream stream
    5. Check merge schedule and merge if due
    6. Save checkpoint
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        memory_store: MemoryStore,
        state_manager: StateManager,
        dream_pool: DreamPool | None = None,
        dream_generator: DreamGenerator | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.memory_store = memory_store
        self.state = state_manager
        self.dream_pool = dream_pool or DreamPool()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup dream generator
        if dream_generator is not None:
            self.generator = dream_generator
        elif config.dream.generator_type == "api" and config.dream.api_model:
            self.generator = APIDreamGenerator(config.dream.api_model)
        else:
            self.generator = LocalDreamGenerator(
                model, tokenizer, device=self.device,
            )

        # CMS trainer
        self.cms_trainer = CMSTrainer(model, tokenizer, config)

    def run_cycle(self, new_span_ids: list[str] | None = None) -> dict:
        """Execute one complete sleep cycle.

        Args:
            new_span_ids: IDs of newly added spans to focus on.
                         If None, processes all spans.

        Returns:
            dict with cycle statistics
        """
        cycle_num = self.state.advance_cycle()
        logger.info(f"=== Sleep Cycle {cycle_num} ===")

        # 1. Collect spans
        if new_span_ids:
            spans = [self.memory_store.get(sid) for sid in new_span_ids]
            spans = [s for s in spans if s is not None]
        else:
            spans = self.memory_store.list_all()

        logger.info(f"Processing {len(spans)} memory spans")

        # 2. Generate dreams
        all_spans = self.memory_store.list_all()
        new_dreams = self.generator.generate_batch(
            spans=spans,
            scenario_distribution=self.config.dream.scenario_distribution,
            dreams_per_level=self.config.dream.dreams_per_level,
            seed=self.config.seed + cycle_num,
        )
        logger.info(f"Generated {len(new_dreams)} new dreams")

        # Validate
        new_dreams = filter_valid_dreams(new_dreams)
        logger.info(f"After validation: {len(new_dreams)} valid dreams")

        # 3. Mix with pool
        self.dream_pool.add_new_dreams(new_dreams)
        mixed_dreams = self.dream_pool.mix(
            new_ratio=self.config.dream.new_dream_ratio,
            max_total=self.config.sleep.max_dreams_per_cycle,
            seed=self.config.seed + cycle_num,
        )
        logger.info(
            f"Dream mix: {len(mixed_dreams)} total "
            f"(new={self.dream_pool.new_count}, old={self.dream_pool.old_count})"
        )

        # 4. CMS training
        dataset = DreamDataset(mixed_dreams, self.tokenizer, self.config.model.max_seq_len)
        stats = self.cms_trainer.train_dream_stream(dataset, device=self.device)

        # Archive new dreams to pool
        for span in spans:
            self.dream_pool.archive_dreams(new_dreams[:5], span.span_id)
            self.memory_store.increment_dream_count(span.span_id, len(new_dreams))
        self.dream_pool.clear_new()

        # 5. Check merge schedule
        due_groups = groups_due_for_merge(self.config.cms.layer_groups, cycle_num)
        merged = []
        if due_groups:
            merged_names = merge_groups_by_schedule(
                self.model,
                self.config.cms.layer_groups,
                self.config.lora,
                cycle_num,
                self.cms_trainer.optimizers,
            )
            for name in merged_names:
                self.state.record_merge(name)
            merged = merged_names
            logger.info(f"Merged groups: {merged}")

        # 6. Save checkpoint
        ckpt_dir = Path(self.config.sleep.checkpoint_dir) / f"cycle_{cycle_num:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        self.state.record_checkpoint(str(ckpt_dir))
        self.state.record_dreams_trained(len(mixed_dreams))

        result = {
            "cycle_number": cycle_num,
            "spans_processed": len(spans),
            "dreams_generated": len(new_dreams),
            "dreams_trained": len(mixed_dreams),
            "training_stats": stats,
            "merged_groups": merged,
        }
        logger.info(f"Cycle {cycle_num} complete: {result}")
        return result
