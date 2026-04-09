"""Exp 1: Memory encoding + generalization test.

Goal 1: Can LoRA memorize facts across multiple users? (context-free recall)
Goal 2: Do memorized facts generalize to new contexts? (indirect application)

Compares uniform SFT at different epoch counts.
"""

import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ExperimentConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from exp1_data import TRAIN_QA, TEST_GENERALIZATION, USERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3.5-0.8B"
LR = 2e-5
BATCH_SIZE = 4
MAX_SEQ_LEN = 512
RANK = 16
EPOCHS_LIST = [30, 100]  # Test at different training durations


def build_train_messages(qa: dict) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": qa["q"]},
        {"role": "assistant", "content": qa["a"]},
    ]


def load_fresh_model():
    """Load model + tokenizer + LoRA from scratch."""
    from dreamlora.config import ModelConfig, LoRAConfig
    model_config = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lora_config = LoRAConfig(rank=RANK, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "up_proj", "down_proj",
    ])
    model, tokenizer = load_model_and_tokenizer(model_config)
    model = setup_lora(model, lora_config)
    return model, tokenizer


def train(model, tokenizer, qa_pairs, num_epochs, log_dir=None):
    """Train on QA pairs with manual loop."""
    messages_list = [build_train_messages(qa) for qa in qa_pairs]
    dataset = DreamDataset(messages_list, tokenizer, max_length=MAX_SEQ_LEN)

    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.01,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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


def evaluate(model, tokenizer, test_items, label, max_new_tokens=64):
    """Evaluate on test questions. Returns list of results."""
    model.eval()
    device = next(model.parameters()).device
    results = []

    for item in test_items:
        messages = [{"role": "user", "content": item["q"]}]
        prompt = format_chatml(messages, add_generation_prompt=True, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, repetition_penalty=1.3,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()

        # Check keywords
        keywords = item.get("keywords", item.get("expected_in_response", []))
        if not keywords:
            # For train QA, check if answer is in response
            keywords = [item["a"].lower()]

        hit = any(kw.lower() in response.lower() for kw in keywords)
        results.append({
            "user": item.get("user", ""),
            "q": item["q"],
            "expected": item.get("a", item.get("fact", "")),
            "keywords": keywords,
            "response": response[:150],
            "hit": hit,
        })

    accuracy = sum(r["hit"] for r in results) / max(len(results), 1)
    return accuracy, results


def run_experiment(epochs, output_dir):
    """Run one experiment condition."""
    logger.info(f"\n{'='*60}\nExp 1: {epochs} epochs\n{'='*60}")

    model, tokenizer = load_fresh_model()
    log_dir = str(output_dir / f"ep{epochs}" / "tb_logs")

    # Train
    train(model, tokenizer, TRAIN_QA, epochs, log_dir=log_dir)

    # Eval: direct recall (train questions)
    recall_acc, recall_results = evaluate(model, tokenizer, TRAIN_QA, "recall")
    logger.info(f"Direct recall: {recall_acc:.1%}")

    # Eval: generalization (new questions)
    gen_acc, gen_results = evaluate(model, tokenizer, TEST_GENERALIZATION, "generalization")
    logger.info(f"Generalization: {gen_acc:.1%}")

    # Per-user breakdown
    for user in USERS:
        user_recall = [r for r in recall_results if r["user"] == user]
        user_gen = [r for r in gen_results if r["user"] == user]
        r_acc = sum(r["hit"] for r in user_recall) / max(len(user_recall), 1)
        g_acc = sum(r["hit"] for r in user_gen) / max(len(user_gen), 1)
        logger.info(f"  {user}: recall={r_acc:.0%}, generalization={g_acc:.0%}")

    # Log generalization details
    logger.info("\nGeneralization details:")
    for r in gen_results:
        status = "✓" if r["hit"] else "✗"
        logger.info(f"  {status} [{r['user']}] {r['q']}")
        logger.info(f"    → {r['response'][:100]}")

    # Save
    out = output_dir / f"ep{epochs}"
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "epochs": epochs,
        "recall_accuracy": recall_acc,
        "generalization_accuracy": gen_acc,
        "recall_details": recall_results,
        "generalization_details": gen_results,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return recall_acc, gen_acc


def main():
    output_dir = Path("experiments/exp1_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for epochs in EPOCHS_LIST:
        recall_acc, gen_acc = run_experiment(epochs, output_dir)
        summary[f"ep{epochs}"] = {"recall": recall_acc, "generalization": gen_acc}

    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    for label, accs in summary.items():
        logger.info(f"  {label}: recall={accs['recall']:.1%}, generalization={accs['generalization']:.1%}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
