"""Special tokens, timestamp formatting, and ChatML utilities."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

# --- Special Tokens ---

SPECIAL_TOKENS = [
    "[MEM_START]",
    "[MEM_END]",
    "[POS]",
    "[NEG]",
    "[NEU]",
]

SENTIMENT_TOKENS = {"positive": "[POS]", "negative": "[NEG]", "neutral": "[NEU]"}
SENTIMENT_FROM_TOKEN = {v: k for k, v in SENTIMENT_TOKENS.items()}

KST = timezone(timedelta(hours=9))


def format_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as ISO 8601 string. Defaults to now in KST."""
    if dt is None:
        dt = datetime.now(KST)
    return dt.isoformat(timespec="seconds")


def parse_timestamp(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def prepend_timestamp(text: str, dt: datetime | None = None) -> str:
    """Prepend ISO 8601 timestamp to text: [2026-03-16T14:32:00+09:00] text"""
    return f"[{format_timestamp(dt)}] {text}"


# --- Memory Span Formatting ---

def format_memory_span(
    content: str,
    level: int = 3,
    sentiment: str = "neutral",
    timestamp: datetime | None = None,
) -> str:
    ts = format_timestamp(timestamp)
    sent_token = SENTIMENT_TOKENS.get(sentiment, "[NEU]")
    return (
        f"[MEM_START] :level={level}\n"
        f"[{ts}]\n"
        f"{sent_token} {content}\n"
        f"[MEM_END]"
    )


# --- ChatML Formatting ---

def format_chatml(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    tokenizer=None,
    enable_thinking: bool = True,
) -> str:
    """Format messages as ChatML.

    If tokenizer is provided, uses its apply_chat_template for correct formatting.
    Otherwise falls back to manual formatting (not recommended for Qwen3.5).
    Set enable_thinking=False to skip thinking block in generation prompt.
    """
    if tokenizer is not None:
        kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        if not enable_thinking:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)

    # Manual fallback
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def build_dream_messages(
    memory_context: str,
    user_query: str,
    assistant_response: str,
    think_block: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build a ChatML message list for a dream sequence.

    If think_block is provided, it is prepended to the assistant response
    wrapped in <think>...</think> tags.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Inject memory context into system or as a separate system message
    if memory_context:
        messages.append({"role": "system", "content": memory_context})

    messages.append({"role": "user", "content": user_query})

    if think_block:
        full_response = f"<think>\n{think_block}\n</think>\n{assistant_response}"
    else:
        full_response = assistant_response

    messages.append({"role": "assistant", "content": full_response})
    return messages


def compute_loss_mask(
    input_ids: list[int],
    tokenizer,
) -> list[bool]:
    """Compute loss mask: True for assistant tokens, False for system/user tokens.

    Masks everything outside <|im_start|>assistant ... <|im_end|> blocks.
    The <think> blocks within assistant responses are included in loss (True).
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Encode "assistant" to get its token ids
    assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)
    assistant_len = len(assistant_token_ids)

    mask = [False] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i] == im_start_id:
            # Check if next tokens are "assistant"
            candidate = input_ids[i + 1 : i + 1 + assistant_len]
            if candidate == assistant_token_ids:
                # Skip past "assistant\n" — find the newline after role
                j = i + 1 + assistant_len
                # Skip newline token if present
                newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                if j < len(input_ids) and input_ids[j] in newline_ids:
                    j += 1
                # Mark everything until <|im_end|> as True
                while j < len(input_ids) and input_ids[j] != im_end_id:
                    mask[j] = True
                    j += 1
                i = j
            else:
                i += 1
        else:
            i += 1
    return mask
