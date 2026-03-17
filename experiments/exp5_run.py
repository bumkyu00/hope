"""Exp 5: Dream density scaling — find generalization threshold."""

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
from exp5_data import build_train_data, build_test_data, FACTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

MODEL = "Qwen/Qwen3.5-0.8B"  # Fast iteration
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
LR = 2e-5
STEPS_PER_DREAM = 20  # Scale steps with data size


def run_condition(dreams_per_fact, tests, out_dir):
    label = f"d{dreams_per_fact}"
    logger.info(f"\n{'='*50} {label}: {dreams_per_fact} dreams/fact {'='*50}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lc = LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "out_proj", "up_proj", "down_proj",
    ])
    model, tok = load_model_and_tokenizer(mc)
    model = setup_lora(model, lc)
    device = next(model.parameters()).device

    # Build data
    train_data = build_train_data(dreams_per_fact)
    msgs_list = [[{"role": "user", "content": d["q"]},
                   {"role": "assistant", "content": d["a"]}] for d in train_data]
    ds = DreamDataset(msgs_list, tok, max_length=MAX_SEQ_LEN)
    target_steps = len(train_data) * STEPS_PER_DREAM
    logger.info(f"  Train: {len(ds)} examples, target ~{target_steps} steps")

    # Train
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    spe = max(len(ds) // BATCH_SIZE, 1)
    epochs = max((target_steps + spe - 1) // spe, 1)
    step = 0
    for ep in range(epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
    logger.info(f"  Trained {step} steps, final loss ~{loss.item():.4f}")

    # Evaluate
    model.eval()
    results_by_cat = {cat: {"correct": 0, "total": 0, "details": []} for cat in FACTS}

    for test in tests:
        msgs = [{"role": "user", "content": test["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw

        hits = [kw for kw in test["should_remember"] if kw.lower() in response.lower()]
        activated = len(hits) > 0
        cat = test["category"]
        results_by_cat[cat]["total"] += 1
        if activated:
            results_by_cat[cat]["correct"] += 1

        s = "✓" if activated else "✗"
        results_by_cat[cat]["details"].append({
            "q": test["q"], "response": response[:150], "hits": hits, "activated": activated,
        })
        logger.info(f"  {s} [{cat}] {test['q'][:40]} → hits={hits}")

    # Summary
    total_correct = sum(v["correct"] for v in results_by_cat.values())
    total_all = sum(v["total"] for v in results_by_cat.values())
    overall = total_correct / total_all if total_all > 0 else 0

    cat_accs = {}
    for cat, data in results_by_cat.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        cat_accs[cat] = acc
        logger.info(f"  {cat}: {acc:.0%} ({data['correct']}/{data['total']})")
    logger.info(f"  OVERALL: {overall:.0%} ({total_correct}/{total_all})")

    # Save
    cond_dir = out_dir / label
    cond_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"dreams_per_fact": dreams_per_fact, "overall": overall,
               "by_category": cat_accs, "results": results_by_cat},
              open(cond_dir / "results.json", "w"), ensure_ascii=False, indent=2)

    del model; torch.cuda.empty_cache()
    return overall, cat_accs


def main():
    out_dir = Path("experiments/exp5_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = build_test_data()

    summary = {}
    for n in [3, 5, 10, 20]:
        overall, cat_accs = run_condition(n, tests, out_dir)
        summary[f"d{n}"] = {"overall": overall, **cat_accs}

    logger.info(f"\n{'='*60}\nDream Density Scaling Summary\n{'='*60}")
    logger.info(f"{'Dreams':>8} | {'Overall':>8} | {'Coding':>8} | {'Food':>8} | {'Activity':>8}")
    logger.info("-" * 50)
    for label, data in summary.items():
        logger.info(f"{label:>8} | {data['overall']:>7.0%} | {data.get('coding',0):>7.0%} | {data.get('food',0):>7.0%} | {data.get('activity',0):>7.0%}")

    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
