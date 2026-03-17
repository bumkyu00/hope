"""Standard SFT trainer with manual training loop (Phase 1 baseline)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dreamlora.config import ExperimentConfig
from dreamlora.data.dream_dataset import DreamDataset

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Simple SFT trainer with manual training loop."""

    def __init__(
        self,
        model,
        tokenizer,
        dataset: DreamDataset,
        config: ExperimentConfig,
        output_dir: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lr = config.cms.layer_groups[0].learning_rate if config.cms.layer_groups else 1e-4
        self.device = next(model.parameters()).device

        self._tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

    def train(self) -> dict:
        model = self.model
        model.train()

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.config.cms.weight_decay,
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
        )

        global_step = 0
        total_loss = 0.0

        for epoch in range(self.config.num_epochs):
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_val = loss.item()
                total_loss += loss_val
                global_step += 1

                self._tb_writer.add_scalar("sft/loss", loss_val, global_step)

                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}: loss={loss_val:.4f}")

            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} complete")

        avg_loss = total_loss / max(global_step, 1)
        self._tb_writer.flush()

        logger.info(f"Training done: {global_step} steps, avg_loss={avg_loss:.4f}")
        return {
            "train_loss": avg_loss,
            "metrics": {"total_steps": global_step},
        }

    def save(self, path: str | None = None) -> None:
        save_path = path or str(self.output_dir)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
