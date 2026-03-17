"""Model and tokenizer loading with special token support."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from dreamlora.config import ModelConfig
from dreamlora.data.formats import SPECIAL_TOKENS

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    config: ModelConfig,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer, add special tokens, resize embeddings.

    For Qwen3.5 (vision-language model), loads the text-only language model
    portion when config.language_model_only is True.
    """
    dtype = DTYPE_MAP[config.dtype]

    # Qwen3.5 uses AutoProcessor; fall back to AutoTokenizer for other models
    try:
        processor = AutoProcessor.from_pretrained(
            config.name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.name_or_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info(
        f"Loaded {config.name_or_path}: "
        f"{model.config.num_hidden_layers} layers, "
        f"dtype={dtype}"
    )

    # Add DreamLoRA special tokens
    num_added = tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added {num_added} special tokens, resized embeddings")

    return model, tokenizer
