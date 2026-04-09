"""Exp 4: Think-based memory retrieval.

Train on QA with <think> retrieval chains, test if model learns
to activate memories via spreading activation in new contexts.
"""

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
from exp4_data import TRAIN_THINK, TEST_THINK

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

MODEL = "Qwen/Qwen3.5-0.8B"
LR = 2e-5
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
TARGET_STEPS = 300


def run():
    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
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
            if step % 20 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")
    logger.info(f"Training done: {step} steps")

    # Evaluate — enable_thinking=False since 0.8B can't natively think
    # But the model learned <think> patterns from training data
    model.eval()
    results = []

    for test in TEST_THINK:
        msgs = [{"role": "user", "content": test["q"]}]
        # Use enable_thinking=False (model will generate think from learned pattern)
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok,
                               enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300,
                                 do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()

        # Check for think block in output
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else ""
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not response:
            response = raw

        hits = [kw for kw in test["should_remember"] if kw.lower() in response.lower()]
        # Also check think block for memory activation
        think_hits = [kw for kw in test["should_remember"] if kw.lower() in think_text.lower()]
        activated = len(hits) > 0 or len(think_hits) > 0

        results.append({
            "q": test["q"], "description": test["description"],
            "think": think_text[:200], "response": response[:200],
            "hits": hits, "think_hits": think_hits, "activated": activated,
        })

        s = "✓" if activated else "✗"
        logger.info(f"{s} {test['description']}")
        if think_text:
            logger.info(f"  💭 {think_text[:120]}")
        logger.info(f"  → {response[:120]}")
        logger.info(f"  response hits={hits}, think hits={think_hits}")
        logger.info("")

    acc = sum(r["activated"] for r in results) / len(results)
    logger.info(f"Memory activation: {acc:.0%} ({sum(r['activated'] for r in results)}/{len(results)})")

    # Save
    out_dir = Path("experiments/exp4_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"accuracy": acc, "results": results},
              open(out_dir / "results.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
