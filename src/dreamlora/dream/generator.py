"""Dream generation: local model and API abstraction."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

import torch

from dreamlora.data.formats import (
    build_dream_messages,
    format_chatml,
    format_memory_span,
    SENTIMENT_TOKENS,
)
from dreamlora.data.memory_store import MemorySpan
from dreamlora.dream.templates import (
    ALL_TEMPLATES,
    DreamTemplate,
    DIRECT_UTILIZATION_TEMPLATES,
    NEG_CORRECTION_TEMPLATES,
)

logger = logging.getLogger(__name__)


class DreamGenerator(ABC):
    """Abstract base class for dream generators."""

    @abstractmethod
    def generate_dream(
        self,
        span: MemorySpan,
        scenario_type: str,
        other_spans: list[MemorySpan] | None = None,
    ) -> list[dict[str, str]] | None:
        """Generate a dream sequence as a ChatML message list.

        Returns None if generation fails.
        """
        ...

    def generate_batch(
        self,
        spans: list[MemorySpan],
        scenario_distribution: dict[str, float],
        dreams_per_level: dict[int, int],
        seed: int = 42,
    ) -> list[list[dict[str, str]]]:
        """Generate a batch of dreams for all spans."""
        rng = random.Random(seed)
        all_dreams = []

        for span in spans:
            n_dreams = dreams_per_level.get(span.level, 10)
            scenario_types = list(scenario_distribution.keys())
            scenario_weights = list(scenario_distribution.values())

            for _ in range(n_dreams):
                scenario = rng.choices(scenario_types, weights=scenario_weights, k=1)[0]

                # NEG spans use correction templates
                if span.sentiment == "negative" and rng.random() < 0.6:
                    scenario = "neg_correction"

                other = [s for s in spans if s.span_id != span.span_id]
                dream = self.generate_dream(span, scenario, other)
                if dream is not None:
                    all_dreams.append(dream)

        rng.shuffle(all_dreams)
        return all_dreams


class LocalDreamGenerator(DreamGenerator):
    """Generate dreams using the local model itself."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 512, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device

    def generate_dream(
        self,
        span: MemorySpan,
        scenario_type: str,
        other_spans: list[MemorySpan] | None = None,
    ) -> list[dict[str, str]] | None:
        templates = ALL_TEMPLATES.get(scenario_type, DIRECT_UTILIZATION_TEMPLATES)
        template = random.choice(templates)

        memory_context = format_memory_span(
            content=span.content,
            level=span.level,
            sentiment=span.sentiment,
        )

        # Build the generation prompt
        subs = {
            "memory_content": span.content,
            "memory_summary": span.content[:50],
            "level": str(span.level),
            "sentiment": span.sentiment,
        }

        if scenario_type == "cross_memory" and other_spans:
            other = random.choice(other_spans)
            subs["memory1_content"] = span.content
            subs["memory2_content"] = other.content
            subs["combined_query"] = f"{span.content[:20]}와 {other.content[:20]}를 함께 고려해줘"
            subs["combined_response"] = ""

        if scenario_type == "temporal_context":
            subs["current_time"] = "2026-04-15T10:00:00+09:00"
            subs["memory_time"] = span.timestamp
            subs["temporal_query"] = f"최근에 {span.content[:20]}에 대해 변한 게 있어?"
            subs["temporal_response"] = ""

        if scenario_type == "neg_correction":
            subs["trigger_query"] = f"{span.content[:30]}에 대해 알려줘"
            subs["neg_event"] = span.content
            subs["correction_reasoning"] = "이번에는 다른 접근을 해야 한다"
            subs["corrected_response"] = ""

        # Fill query template
        try:
            query = template.user_template.format(**subs)
        except KeyError:
            query = f"{span.content[:30]}에 대해 알려줘"

        # For local generation, use the model to generate the response
        messages = [
            {"role": "system", "content": template.system_prompt + "\n\n" + memory_context},
            {"role": "user", "content": query},
        ]
        prompt = format_chatml(messages, add_generation_prompt=True, tokenizer=self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        if not response:
            return None

        # Build think block from template
        try:
            think = template.think_template.format(**subs)
        except KeyError:
            think = f"사용자의 기억: {span.content}"

        return build_dream_messages(
            memory_context=memory_context,
            user_query=query,
            assistant_response=response,
            think_block=think,
            system_prompt=template.system_prompt,
        )


class APIDreamGenerator(DreamGenerator):
    """Generate dreams using an external API (Claude, GPT, etc.)."""

    def __init__(self, api_model: str = "claude-sonnet-4-20250514"):
        self.api_model = api_model

    def generate_dream(
        self,
        span: MemorySpan,
        scenario_type: str,
        other_spans: list[MemorySpan] | None = None,
    ) -> list[dict[str, str]] | None:
        # API dream generation requires external setup
        # Placeholder for API integration
        logger.warning("API dream generation not yet implemented, falling back to template")

        memory_context = format_memory_span(
            content=span.content,
            level=span.level,
            sentiment=span.sentiment,
        )

        templates = ALL_TEMPLATES.get(scenario_type, DIRECT_UTILIZATION_TEMPLATES)
        template = random.choice(templates)

        subs = {
            "memory_content": span.content,
            "memory_summary": span.content[:50],
            "level": str(span.level),
            "sentiment": span.sentiment,
            "query": f"{span.content[:30]}에 대해 알려줘",
            "task": f"{span.content[:30]} 관련 작업",
            "personalized_response": f"{span.content}을 반영한 답변입니다.",
            "task_response": f"{span.content}을 고려한 결과입니다.",
        }

        try:
            think = template.think_template.format(**subs)
            query = template.user_template.format(**subs)
            response = template.response_template.format(**subs)
        except KeyError:
            think = f"사용자의 기억: {span.content}"
            query = f"{span.content[:30]}에 대해 알려줘"
            response = f"네, {span.content[:50]}..."

        return build_dream_messages(
            memory_context=memory_context,
            user_query=query,
            assistant_response=response,
            think_block=think,
            system_prompt=template.system_prompt,
        )
