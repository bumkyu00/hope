"""DreamDataset: PyTorch Dataset for dream sequences."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from dreamlora.data.formats import format_chatml, compute_loss_mask


class DreamDataset(Dataset):
    """Tokenizes ChatML dream sequences with selective loss masking.

    Loss is computed only on assistant tokens (including <think> blocks).
    System and user tokens are masked out.
    """

    def __init__(
        self,
        dreams: list[list[dict[str, str]]],
        tokenizer,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict[str, torch.Tensor]] = []

        for messages in dreams:
            encoded = self._encode(messages)
            if encoded is not None:
                self.examples.append(encoded)

    def _encode(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor] | None:
        text = format_chatml(messages, tokenizer=self.tokenizer)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Compute loss mask (True for assistant tokens)
        loss_mask = compute_loss_mask(input_ids.tolist(), self.tokenizer)

        # Labels: -100 for masked positions
        labels = input_ids.clone()
        for i, m in enumerate(loss_mask):
            if not m:
                labels[i] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]
