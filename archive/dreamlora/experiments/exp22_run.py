"""Exp 22: Scale test — 30+ facts, 3 dream approaches, 20 novel tests.

Compares:
  A: Raw conversation sessions
  B: QA extraction
  C: Think retrieval chain (with natural recall tone)

All with nested adapter on 4B. Full I/O logging.
"""

import json, logging, re, sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from exp9_nested import AdapterMLP, NestedModel
from exp22_data import (
    build_raw_conversation_dreams, build_qa_dreams, build_think_chain_dreams,
    NOVEL_TESTS, PROJECT_KEYWORDS,
)

MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_POSITIONS = [9, 21]
ADAPTER_SIZE = 64
LR = 1e-3
BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
CHECKPOINTS = [10, 20, 30, 50, 75, 100]
OUT_DIR = Path("experiments/exp22_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


def full_eval(model, tok, device):
    """Evaluate on novel tests. Save full I/O including raw think blocks."""
    model.eval()
    results = []

    for t in NOVEL_TESTS:
        msgs = [{"role": "user", "content": t["q"]}]

        # Generate with thinking enabled (raw response includes think)
        prompt_think = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=True)
        inputs = tok(prompt_think, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2000,
                                 do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
        raw_think = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        # Also generate without thinking (for keyword matching)
        prompt_no = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs_no = tok(prompt_no, return_tensors="pt").to(device)
        with torch.no_grad():
            out_no = model.generate(**inputs_no, max_new_tokens=300,
                                    do_sample=False, repetition_penalty=1.3)
        raw_no = tok.decode(out_no[0][inputs_no["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        clean = re.sub(r"<think>.*?</think>", "", raw_no, flags=re.DOTALL).strip() or raw_no

        # Parse think block from raw_think
        think_text = ""
        if "</think>" in raw_think:
            think_text = raw_think.split("</think>")[0].replace("<think>", "").strip()

        # Keyword matching on clean response
        hits = [kw for kw in t["should"] if kw.lower() in clean.lower()]
        activated = len(hits) > 0

        # Check project contamination for sanity questions
        contaminated = False
        if t.get("sanity"):
            proj_hits = [kw for kw in PROJECT_KEYWORDS if kw.lower() in clean.lower()]
            contaminated = len(proj_hits) > 0

        results.append({
            "q": t["q"],
            "desc": t["desc"],
            "is_sanity": t.get("sanity", False),
            "should": t["should"],
            "think_raw": think_text[:500],
            "response_raw": raw_no[:500],
            "response_clean": clean[:500],
            "hits": hits,
            "activated": activated,
            "contaminated": contaminated,
        })

        s = "✓" if activated else "✗"
        c = " ⚠️CONTAM" if contaminated else ""
        logger.info(f"  {s}{c} {t['desc']}")
        logger.info(f"    Q: {t['q']}")
        if think_text:
            logger.info(f"    💭: {think_text[:150]}")
        logger.info(f"    →: {clean[:150]}")
        logger.info(f"    hits={hits}")
        logger.info("")

    mem_tests = [r for r in results if not r["is_sanity"]]
    san_tests = [r for r in results if r["is_sanity"]]
    mem_acc = sum(r["activated"] for r in mem_tests) / max(len(mem_tests), 1)
    san_acc = sum(not r["contaminated"] and r["activated"] for r in san_tests) / max(len(san_tests), 1)
    contam = sum(r["contaminated"] for r in san_tests)

    return mem_acc, san_acc, contam, results


def train_and_eval(name, dreams, out_dir):
    """Train nested adapter on dreams and evaluate."""
    logger.info(f"\n{'='*60}\n{name}\n{'='*60}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    nested = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

    ds = DreamDataset(dreams, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"  Dataset: {len(ds)} examples")

    # Train
    nested.train()
    opt = torch.optim.AdamW(nested.adapters.parameters(), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; ci = 0; all_evals = {}

    for ep in range(100):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = nested(**batch)
            loss = out.loss; loss.backward(); opt.step(); opt.zero_grad(); step += 1

            if step % 10 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                logger.info(f"\n  --- Eval @ step {step} ---")
                mem, san, contam, results = full_eval(nested, tok, device)
                logger.info(f"  ★ Step {step}: Mem={mem:.0%}, San={san:.0%}, Contam={contam}")

                ckpt_dir = out_dir / name / f"step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(nested.adapters.state_dict(), ckpt_dir / "adapters.pt")
                json.dump({
                    "step": step, "loss": loss.item(),
                    "memory": mem, "sanity": san, "contamination": contam,
                    "details": results,
                }, open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)

                all_evals[step] = {"memory": mem, "sanity": san, "contamination": contam}
                ci += 1; nested.train()
        if ci >= len(CHECKPOINTS): break

    del nested, base_model; torch.cuda.empty_cache()
    return all_evals


def main():
    logger.info(f"Exp 22: Scale test, {MODEL}, {len(NOVEL_TESTS)} novel tests")
    logger.info(f"Started: {datetime.now().isoformat()}")

    # Build dream data for each approach
    approaches = {
        "A_raw_conversation": build_raw_conversation_dreams(),
        "B_qa_extraction": build_qa_dreams(),
        "C_think_chain": build_think_chain_dreams(),
    }

    all_results = {}
    for name, dreams in approaches.items():
        all_results[name] = train_and_eval(name, dreams, OUT_DIR)

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    logger.info(f"{'Step':>5} | {'A_raw Mem/San':>14} | {'B_qa Mem/San':>13} | {'C_think Mem/San':>15}")
    logger.info("-" * 60)
    for step in CHECKPOINTS:
        a = all_results.get("A_raw_conversation", {}).get(step, {})
        b = all_results.get("B_qa_extraction", {}).get(step, {})
        c = all_results.get("C_think_chain", {}).get(step, {})
        logger.info(f"{step:>5} | {a.get('memory',0):>5.0%}/{a.get('sanity',0):>4.0%} c={a.get('contamination',0)} | {b.get('memory',0):>5.0%}/{b.get('sanity',0):>4.0%} c={b.get('contamination',0)} | {c.get('memory',0):>5.0%}/{c.get('sanity',0):>4.0%} c={c.get('contamination',0)}")

    json.dump(all_results, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
