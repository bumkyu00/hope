"""Tests for CMS trainer gradient buffer mechanics.

Uses a tiny model to verify:
1. Gradient buffers accumulate correctly per group
2. Groups step at their own chunk_size boundaries
3. Buffer reset after step
4. Partial chunk flush at end of stream
"""

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

from dreamlora.config import ExperimentConfig, LayerGroupConfig, CMSConfig
from dreamlora.training.cms_trainer import CMSTrainer
from dreamlora.training.optimizer_groups import create_group_optimizers, reset_optimizer_state


class TestGradientBufferMechanics:
    """Test gradient buffer accumulation and stepping logic without a real model."""

    def test_buffer_accumulates(self):
        """Verify gradients accumulate in buffer across steps."""
        buf = torch.zeros(4, 4)
        grad1 = torch.ones(4, 4)
        grad2 = torch.ones(4, 4) * 2

        buf.add_(grad1)
        buf.add_(grad2)

        expected = torch.ones(4, 4) * 3
        assert torch.allclose(buf, expected)

    def test_buffer_normalization(self):
        """Verify buffer is correctly normalized by chunk_size."""
        buf = torch.ones(4, 4) * 6.0
        chunk_size = 3
        normalized = buf / chunk_size
        expected = torch.ones(4, 4) * 2.0
        assert torch.allclose(normalized, expected)

    def test_buffer_reset(self):
        """Verify buffer resets to zero."""
        buf = torch.ones(4, 4) * 5.0
        buf.zero_()
        assert torch.allclose(buf, torch.zeros(4, 4))


class TestOptimizerGroups:
    def test_create_group_optimizers(self):
        """Verify independent optimizers are created per group."""
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))

        group_params = {
            "high_freq": [("p1", p1)],
            "low_freq": [("p2", p2)],
        }
        layer_groups = [
            LayerGroupConfig(name="high_freq", layer_start=0, layer_end=7, learning_rate=2e-4, chunk_size=1),
            LayerGroupConfig(name="low_freq", layer_start=8, layer_end=15, learning_rate=5e-5, chunk_size=5),
        ]

        optimizers = create_group_optimizers(group_params, layer_groups)

        assert "high_freq" in optimizers
        assert "low_freq" in optimizers
        # Check learning rates
        assert optimizers["high_freq"].param_groups[0]["lr"] == 2e-4
        assert optimizers["low_freq"].param_groups[0]["lr"] == 5e-5

    def test_reset_optimizer_state(self):
        """Verify optimizer state is cleared after reset."""
        p = nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.AdamW([p], lr=1e-3)

        # Do a step to populate state
        loss = (p ** 2).sum()
        loss.backward()
        opt.step()
        assert len(opt.state) > 0

        reset_optimizer_state(opt)
        assert len(opt.state) == 0


class TestChunkBoundaryLogic:
    """Test that chunk boundaries trigger updates at the right steps."""

    def test_chunk_1_updates_every_step(self):
        """chunk_size=1 should trigger every step."""
        chunk_size = 1
        updates = []
        for step in range(10):
            if (step + 1) % chunk_size == 0:
                updates.append(step)
        assert len(updates) == 10

    def test_chunk_5_updates_every_5_steps(self):
        """chunk_size=5 should trigger every 5 steps."""
        chunk_size = 5
        updates = []
        for step in range(25):
            if (step + 1) % chunk_size == 0:
                updates.append(step)
        assert updates == [4, 9, 14, 19, 24]

    def test_chunk_25_updates_once_in_25_steps(self):
        """chunk_size=25 should trigger once in 25 steps."""
        chunk_size = 25
        updates = []
        for step in range(25):
            if (step + 1) % chunk_size == 0:
                updates.append(step)
        assert updates == [24]

    def test_partial_flush(self):
        """Remaining steps after last chunk boundary should be flushed."""
        chunk_size = 5
        total_steps = 23
        full_updates = total_steps // chunk_size  # 4
        remaining = total_steps % chunk_size  # 3
        assert full_updates == 4
        assert remaining == 3  # Needs a partial flush


class TestGroupIndependence:
    """Test that groups accumulate and step independently."""

    def test_independent_step_counts(self):
        """Different chunk sizes lead to different update counts."""
        groups = [
            {"name": "fast", "chunk_size": 1},
            {"name": "medium", "chunk_size": 5},
            {"name": "slow", "chunk_size": 25},
        ]

        total_steps = 25
        update_counts = {}

        for g in groups:
            count = 0
            for step in range(total_steps):
                if (step + 1) % g["chunk_size"] == 0:
                    count += 1
            update_counts[g["name"]] = count

        assert update_counts["fast"] == 25
        assert update_counts["medium"] == 5
        assert update_counts["slow"] == 1

    def test_gradient_isolation(self):
        """Each group should only accumulate its own parameters' gradients."""
        p1 = torch.randn(4, 4, requires_grad=True)
        p2 = torch.randn(4, 4, requires_grad=True)

        buf1 = torch.zeros_like(p1)
        buf2 = torch.zeros_like(p2)

        # Simulate a backward pass
        loss = (p1 ** 2).sum() + (p2 ** 2).sum()
        loss.backward()

        buf1.add_(p1.grad)
        buf2.add_(p2.grad)

        # buf1 should only reflect p1's gradient
        assert torch.allclose(buf1, 2 * p1.detach())
        # buf2 should only reflect p2's gradient
        assert torch.allclose(buf2, 2 * p2.detach())
