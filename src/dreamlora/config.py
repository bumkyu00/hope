"""Pydantic configuration models with YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name_or_path: str = "Qwen/Qwen3.5-9B"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    max_seq_len: int = 4096
    attn_implementation: str = "flash_attention_2"
    language_model_only: bool = True  # Skip vision encoder for text-only


class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            # Full attention layers (8 layers: 3,7,11,15,19,23,27,31)
            "q_proj", "k_proj", "v_proj", "o_proj",
            # Gated DeltaNet layers (24 layers)
            "in_proj_qkv", "out_proj",
            # Shared MLP
            "up_proj", "down_proj",
        ]
    )


class LayerGroupConfig(BaseModel):
    name: str
    layer_start: int
    layer_end: int
    learning_rate: float
    chunk_size: int = 1
    merge_every_n_cycles: int | None = None  # None = never merge (permanent)


class CMSConfig(BaseModel):
    layer_groups: list[LayerGroupConfig] = Field(default_factory=lambda: [
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
            learning_rate=5e-5, chunk_size=25, merge_every_n_cycles=None,
        ),
    ])
    gradient_clipping: float = 1.0
    weight_decay: float = 0.01


class DreamConfig(BaseModel):
    dreams_per_level: dict[int, int] = Field(
        default_factory=lambda: {1: 5, 2: 10, 3: 20, 4: 35, 5: 50}
    )
    new_dream_ratio: float = 0.7
    old_dream_ratio: float = 0.3
    scenario_distribution: dict[str, float] = Field(
        default_factory=lambda: {
            "original_replay": 0.20,
            "direct_utilization": 0.40,
            "cross_memory": 0.25,
            "temporal_context": 0.15,
        }
    )
    batch_size: int = 4
    generator_type: Literal["local", "api"] = "local"
    api_model: str | None = None


class SleepConfig(BaseModel):
    checkpoint_dir: str = "checkpoints"
    cycle_log_file: str = "sleep_cycles.jsonl"
    max_dreams_per_cycle: int = 200


class EvalConfig(BaseModel):
    mmlu_num_questions: int = 100
    mmlu_seed: int = 42
    judge_model: str | None = None  # None = keyword matching


class ExperimentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    cms: CMSConfig = Field(default_factory=CMSConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)
    sleep: SleepConfig = Field(default_factory=SleepConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    training_batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 10
    output_dir: str = "outputs"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)
