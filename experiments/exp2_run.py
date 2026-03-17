"""Exp 2: QA-only vs QA+Context training, generalization comparison.

Condition A: Train on QA only (30 pairs)
Condition B: Train on QA + Context examples (30 + 25 = 55 pairs)
Both tested on same 30 generalization questions.
Step count matched by adjusting epochs.
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

from dreamlora.config import ModelConfig, LoRAConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml
from exp1_data import TRAIN_QA, TEST_GENERALIZATION, USERS
from exp2_data import CONTEXT_EXAMPLES, TRAIN_QA_PLUS_CONTEXT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3.5-4B"
LR = 2e-5
BATCH_SIZE = 4
MAX_SEQ_LEN = 512
TARGET_STEPS = 250  # Match training budget


def load_fresh_model():
    model_config = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lora_config = LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "up_proj", "down_proj",
    ])
    model, tokenizer = load_model_and_tokenizer(model_config)
    model = setup_lora(model, lora_config)
    return model, tokenizer


def build_messages(qa):
    return [
        {"role": "user", "content": qa["q"]},
        {"role": "assistant", "content": qa["a"]},
    ]


def train(model, tokenizer, qa_pairs, target_steps, log_dir=None):
    messages_list = [build_messages(qa) for qa in qa_pairs]
    dataset = DreamDataset(messages_list, tokenizer, max_length=MAX_SEQ_LEN)
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.01,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    steps_per_epoch = max(len(dataset) // BATCH_SIZE, 1)
    num_epochs = max((target_steps + steps_per_epoch - 1) // steps_per_epoch, 1)
    logger.info(f"  {len(dataset)} examples, {steps_per_epoch} steps/ep, {num_epochs} epochs → ~{steps_per_epoch * num_epochs} steps")

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
                logger.info(f"    step {global_step}: loss={loss.item():.4f}")
    if writer:
        writer.flush()
    return global_step


def evaluate(model, tokenizer, test_items, max_new_tokens=100):
    model.eval()
    device = next(model.parameters()).device
    results = []
    for item in test_items:
        messages = [{"role": "user", "content": item["q"]}]
        prompt = format_chatml(messages, add_generation_prompt=True, tokenizer=tokenizer, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, repetition_penalty=1.3,
            )
        raw_response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
        # Strip think block to get actual response
        import re
        response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
        if not response:
            response = raw_response  # fallback if all was think
        keywords = item.get("keywords", [item["a"].lower()] if "a" in item else [])
        hit = any(kw.lower() in response.lower() for kw in keywords)
        results.append({
            "user": item.get("user", ""),
            "q": item["q"],
            "expected": item.get("a", item.get("fact", "")),
            "response": response[:150],
            "hit": hit,
        })
    accuracy = sum(r["hit"] for r in results) / max(len(results), 1)
    return accuracy, results


def run_condition(name, train_data, output_dir):
    logger.info(f"\n=== {name} ===")
    model, tokenizer = load_fresh_model()
    log_dir = str(output_dir / name / "tb_logs")
    steps = train(model, tokenizer, train_data, TARGET_STEPS, log_dir)

    recall_acc, recall_res = evaluate(model, tokenizer, TRAIN_QA)
    gen_acc, gen_res = evaluate(model, tokenizer, TEST_GENERALIZATION)

    logger.info(f"  Recall: {recall_acc:.1%}, Generalization: {gen_acc:.1%}")

    # Per-user
    for user in USERS:
        ur = [r for r in recall_res if r["user"] == user]
        ug = [r for r in gen_res if r["user"] == user]
        logger.info(f"    {user}: recall={sum(r['hit'] for r in ur)/max(len(ur),1):.0%}, gen={sum(r['hit'] for r in ug)/max(len(ug),1):.0%}")

    # Show generalization details
    for r in gen_res:
        s = "✓" if r["hit"] else "✗"
        logger.info(f"    {s} [{r['user']}] {r['q']} → {r['response'][:80]}")

    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({"recall": recall_acc, "generalization": gen_acc,
                    "recall_details": recall_res, "gen_details": gen_res}, f, ensure_ascii=False, indent=2)
    return recall_acc, gen_acc


def main():
    output_dir = Path("experiments/exp2_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["A_qa_only"] = run_condition("A_qa_only", TRAIN_QA, output_dir)
    results["B_qa_plus_context"] = run_condition("B_qa_plus_context", TRAIN_QA_PLUS_CONTEXT, output_dir)

    logger.info(f"\n{'='*60}\nExp 2 Summary\n{'='*60}")
    for name, (recall, gen) in results.items():
        logger.info(f"  {name}: recall={recall:.1%}, generalization={gen:.1%}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump({k: {"recall": v[0], "gen": v[1]} for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
