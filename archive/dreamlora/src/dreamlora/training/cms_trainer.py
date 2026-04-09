"""CMS (Continuum Memory System) differential update trainer.

Core implementation: each layer group independently accumulates gradients
in a buffer and steps at its own chunk_size interval.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dreamlora.config import ExperimentConfig, LayerGroupConfig
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.model.lora_setup import get_layer_group_params
from dreamlora.training.optimizer_groups import create_group_optimizers

logger = logging.getLogger(__name__)


class CMSTrainer:
    """CMS-style differential update trainer with per-group gradient buffers.

    Each layer group:
    - Accumulates gradients over its chunk_size steps
    - Steps its own optimizer at chunk boundaries
    - Has independent learning rate and update frequency
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        log_dir: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.layer_groups = config.cms.layer_groups
        self.log_dir = log_dir

        # TensorBoard writer
        self._tb_writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=log_dir)

        # Map parameters to groups
        self.group_params = get_layer_group_params(model, self.layer_groups)

        # Create per-group optimizers
        self.optimizers = create_group_optimizers(
            self.group_params,
            self.layer_groups,
            weight_decay=config.cms.weight_decay,
        )

        # Initialize gradient accumulation buffers
        self.grad_buffers: dict[str, dict[str, torch.Tensor]] = {}
        self._init_grad_buffers()

        # Step counters per group
        self.step_counts: dict[str, int] = {g.name: 0 for g in self.layer_groups}
        self.update_counts: dict[str, int] = {g.name: 0 for g in self.layer_groups}

        # Logging
        self.gradient_norms: dict[str, list[float]] = defaultdict(list)

    def _init_grad_buffers(self) -> None:
        """Initialize zero gradient buffers for each group."""
        self.grad_buffers = {}
        for group_name, named_params in self.group_params.items():
            self.grad_buffers[group_name] = {
                name: torch.zeros_like(param, device=param.device)
                for name, param in named_params
            }

    def _reset_group_buffer(self, group_name: str) -> None:
        for name in self.grad_buffers[group_name]:
            self.grad_buffers[group_name][name].zero_()

    def _accumulate_gradients(self) -> None:
        """Accumulate current param.grad into group buffers."""
        for group_name, named_params in self.group_params.items():
            for name, param in named_params:
                if param.grad is not None:
                    self.grad_buffers[group_name][name].add_(param.grad)

    def _step_group(self, group_name: str, chunk_size: int) -> float:
        """Apply accumulated gradients for a group, normalized by chunk_size."""
        named_params = self.group_params[group_name]
        grad_norm = 0.0

        # Set param.grad from buffer, normalized by chunk_size
        for name, param in named_params:
            buffered = self.grad_buffers[group_name][name]
            normalized = buffered / chunk_size
            param.grad = normalized.clone()
            grad_norm += normalized.norm().item() ** 2

        grad_norm = grad_norm ** 0.5

        # Gradient clipping
        if self.config.cms.gradient_clipping > 0:
            params = [p for _, p in named_params]
            torch.nn.utils.clip_grad_norm_(
                params, self.config.cms.gradient_clipping
            )

        # Optimizer step
        self.optimizers[group_name].step()
        self._reset_group_buffer(group_name)
        self.update_counts[group_name] += 1

        return grad_norm

    def train_dream_stream(
        self,
        dataset: DreamDataset,
        device: str = "cuda",
    ) -> dict:
        """Train on a dream stream with CMS differential updates.

        Returns training statistics including per-group gradient norms and
        update counts.
        """
        self.model.train()
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training_batch_size,
            shuffle=False,  # Preserve stream order for CMS
        )

        group_config_map = {g.name: g for g in self.layer_groups}
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward + backward
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1

            if self._tb_writer:
                self._tb_writer.add_scalar("cms/loss", loss_val, step)

            # Accumulate gradients into buffers
            self._accumulate_gradients()

            # Clear model gradients
            self.model.zero_grad()

            # Check chunk boundaries for each group independently
            for group in self.layer_groups:
                self.step_counts[group.name] += 1
                if self.step_counts[group.name] % group.chunk_size == 0:
                    grad_norm = self._step_group(group.name, group.chunk_size)
                    self.gradient_norms[group.name].append(grad_norm)
                    if self._tb_writer:
                        self._tb_writer.add_scalar(
                            f"cms/grad_norm/{group.name}", grad_norm,
                            self.update_counts[group.name],
                        )
                    logger.debug(
                        f"Group {group.name} update #{self.update_counts[group.name]}: "
                        f"grad_norm={grad_norm:.4f}"
                    )

        # Flush remaining buffers (partial chunks)
        for group in self.layer_groups:
            remaining = self.step_counts[group.name] % group.chunk_size
            if remaining > 0:
                grad_norm = self._step_group(group.name, remaining)
                self.gradient_norms[group.name].append(grad_norm)
                logger.debug(
                    f"Group {group.name} final flush: "
                    f"grad_norm={grad_norm:.4f}, steps={remaining}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        stats = {
            "avg_loss": avg_loss,
            "num_batches": num_batches,
            "update_counts": dict(self.update_counts),
            "avg_gradient_norms": {
                name: sum(norms) / max(len(norms), 1)
                for name, norms in self.gradient_norms.items()
            },
        }

        if self._tb_writer:
            self._tb_writer.add_scalar("cms/avg_loss", avg_loss, 0)
            for name, norms in self.gradient_norms.items():
                self._tb_writer.add_scalar(
                    f"cms/avg_grad_norm/{name}",
                    sum(norms) / max(len(norms), 1), 0,
                )
            self._tb_writer.flush()

        logger.info(
            f"CMS Training: loss={avg_loss:.4f}, "
            f"updates={dict(self.update_counts)}"
        )
        return stats

    def get_weight_changes(self) -> dict[str, float]:
        """Compute L2 norm of LoRA weight changes per group (for analysis)."""
        changes = {}
        for group_name, named_params in self.group_params.items():
            total_norm = 0.0
            for name, param in named_params:
                total_norm += param.data.norm().item() ** 2
            changes[group_name] = total_norm ** 0.5
        return changes
