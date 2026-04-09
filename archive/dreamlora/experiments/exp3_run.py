"""Exp 3: Naturalistic conversation memory — infinite context illusion.

Train on 3 realistic chat sessions, test if session 4 flows naturally.
Key difference from exp1/2: multi-turn conversations, not QA pairs.
"""

import json
import logging
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig, LoRAConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml
from exp3_data import TRAIN_SESSIONS, SESSION_4_TESTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3.5-4B"
LR = 2e-5
BATCH_SIZE = 1  # Multi-turn sessions are longer
MAX_SEQ_LEN = 1024
TARGET_STEPS = 300


def load_fresh_model():
    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lc = LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "out_proj", "up_proj", "down_proj",
    ])
    model, tok = load_model_and_tokenizer(mc)
    model = setup_lora(model, lc)
    return model, tok


def train(model, tok, sessions, target_steps, log_dir=None):
    """Train on multi-turn conversation sessions."""
    dataset = DreamDataset(sessions, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"Dataset: {len(dataset)} sessions")

    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR, weight_decay=0.01,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    steps_per_epoch = max(len(dataset) // BATCH_SIZE, 1)
    num_epochs = max((target_steps + steps_per_epoch - 1) // steps_per_epoch, 1)
    logger.info(f"  {steps_per_epoch} steps/epoch, {num_epochs} epochs → ~{steps_per_epoch * num_epochs} steps")

    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
    global_step = 0

    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if writer:
                writer.add_scalar("loss", loss.item(), global_step)
            if global_step % 20 == 0:
                logger.info(f"  step {global_step}: loss={loss.item():.4f}")

    if writer:
        writer.flush()
    logger.info(f"Training done: {global_step} steps")
    return global_step


def evaluate_session4(model, tok, tests, enable_thinking=False):
    """Simulate session 4: send each test message and check memory activation."""
    model.eval()
    device = next(model.parameters()).device
    results = []

    # Build a running conversation for session 4
    # Start fresh — no system prompt, no history, just like a new session
    for test in tests:
        # Each test is independent (like the user starts a new topic in session 4)
        messages = [{"role": "user", "content": test["user_msg"]}]
        prompt = format_chatml(
            messages, add_generation_prompt=True,
            tokenizer=tok, enable_thinking=enable_thinking,
        )
        inputs = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.3,
            )

        raw = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Strip think block
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not response:
            response = raw

        # Check if any expected keywords appear
        keywords = test["should_remember"]
        hits = [kw for kw in keywords if kw.lower() in response.lower()]
        memory_activated = len(hits) > 0

        results.append({
            "description": test["description"],
            "user_msg": test["user_msg"],
            "should_remember": keywords,
            "response": response[:300],
            "hits": hits,
            "memory_activated": memory_activated,
        })

        status = "✓" if memory_activated else "✗"
        logger.info(f"  {status} {test['description']}")
        logger.info(f"    User: {test['user_msg'][:60]}")
        logger.info(f"    Response: {response[:120]}")
        logger.info(f"    Hits: {hits}")
        logger.info("")

    accuracy = sum(r["memory_activated"] for r in results) / len(results)
    return accuracy, results


def main():
    output_dir = Path("experiments/exp3_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_fresh_model()

    # Baseline: before training, how does base model respond?
    logger.info("\n=== Baseline (no training) ===")
    base_acc, base_results = evaluate_session4(model, tok, SESSION_4_TESTS, enable_thinking=False)
    logger.info(f"Baseline memory activation: {base_acc:.1%}")

    # Train on 3 sessions
    logger.info("\n=== Training on 3 sessions ===")
    train(model, tok, TRAIN_SESSIONS, TARGET_STEPS,
          log_dir=str(output_dir / "tb_logs"))

    # After training: no-thinking
    logger.info("\n=== After training (no thinking) ===")
    trained_acc, trained_results = evaluate_session4(model, tok, SESSION_4_TESTS, enable_thinking=False)
    logger.info(f"Trained memory activation: {trained_acc:.1%}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Baseline: {base_acc:.1%}")
    logger.info(f"Trained:  {trained_acc:.1%}")
    logger.info(f"Improvement: +{trained_acc - base_acc:.1%}")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "baseline_accuracy": base_acc,
            "trained_accuracy": trained_acc,
            "baseline_results": base_results,
            "trained_results": trained_results,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
