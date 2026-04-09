"""Exp 4b: Think retrieval — harder tests, multi-user, 4B model."""

import json, logging, re, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig, LoRAConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml
from exp4b_data import TRAIN_THINK, TEST_THINK

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

BATCH_SIZE = 2
MAX_SEQ_LEN = 512
TARGET_STEPS = 400
LR = 2e-5


def run(model_name):
    logger.info(f"\n{'='*60}\n{model_name}\n{'='*60}")

    mc = ModelConfig(name_or_path=model_name, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lc = LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "out_proj", "up_proj", "down_proj",
    ])
    model, tok = load_model_and_tokenizer(mc)
    model = setup_lora(model, lc)
    device = next(model.parameters()).device

    # Build dataset
    msgs_list = [[{"role": "user", "content": d["q"]},
                   {"role": "assistant", "content": d["a"]}] for d in TRAIN_THINK]
    ds = DreamDataset(msgs_list, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"Dataset: {len(ds)} examples")

    # Train
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    spe = max(len(ds) // BATCH_SIZE, 1)
    epochs = max((TARGET_STEPS + spe - 1) // spe, 1)
    step = 0
    for ep in range(epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 50 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")
    logger.info(f"Training done: {step} steps")

    # Evaluate
    model.eval()
    results = []
    correct, total = 0, 0
    confusion = 0  # cross-user contamination

    for test in TEST_THINK:
        msgs = [{"role": "user", "content": test["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else ""
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not response: response = raw
        full_text = (think_text + " " + response).lower()

        hits = [kw for kw in test["should_remember"] if kw.lower() in full_text]
        activated = len(hits) > 0
        if activated: correct += 1
        total += 1

        # Check cross-user confusion
        confused = False
        if "should_NOT_contain" in test:
            bad_hits = [kw for kw in test["should_NOT_contain"] if kw.lower() in full_text]
            if bad_hits:
                confused = True
                confusion += 1

        results.append({
            "q": test["q"], "description": test["description"],
            "think": think_text[:150], "response": response[:200],
            "hits": hits, "activated": activated, "confused": confused,
        })

        s = "✓" if activated else "✗"
        c = " ⚠️CONFUSED" if confused else ""
        logger.info(f"{s}{c} {test['description']}")
        if think_text: logger.info(f"  💭 {think_text[:100]}")
        logger.info(f"  → {response[:120]}")
        logger.info(f"  hits={hits}")
        logger.info("")

    acc = correct / total
    logger.info(f"Memory activation: {acc:.0%} ({correct}/{total}), Confusion: {confusion}")
    return acc, confusion, results


def main():
    out_dir = Path("experiments/exp4b_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for model_name in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]:
        acc, confusion, results = run(model_name)
        label = model_name.split("-")[-1]
        summary[label] = {"accuracy": acc, "confusion": confusion}
        with open(out_dir / f"{label}_results.json", "w") as f:
            json.dump({"accuracy": acc, "confusion": confusion, "results": results},
                      f, ensure_ascii=False, indent=2)
        # Free GPU
        torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    for label, data in summary.items():
        logger.info(f"  {label}: accuracy={data['accuracy']:.0%}, confusion={data['confusion']}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
