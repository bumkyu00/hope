"""Exp 26b: More steps on best CMS config for novel domain.

From Exp 26: C_cms_1_10 was best (Mem 40%, San 80%)
But 75 steps insufficient — loss still 1.9.

Test: 100, 150, 200 steps with C_cms_1_10 and uniform baseline.
Also test higher lr (1e-3) to accelerate learning on novel domain.

Phase 1 → eval at each checkpoint → Phase 2 → eval.
"""

import json, logging, re, sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from exp9_nested import AdapterMLP, NestedModel
from exp26_data import (
    build_think_chain_dreams, NOVEL_TESTS, PROJECT_KEYWORDS, RETRIEVAL_SFT,
)
from exp26_novel_2b import CMSNestedModel, init_cms_training, cms_step, full_eval

MODEL = "Qwen/Qwen3.5-2B"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
ADAPTER_SIZE = 64
OUT_DIR = Path("experiments/exp26b_results")
CHECKPOINTS = [50, 75, 100, 150, 200]

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


def run_with_checkpoints(name, adapter_configs, p1_steps=200, p2_steps=40):
    """Run with periodic evaluation checkpoints."""
    logger.info(f"\n{'='*60}\n{name}\n{'='*60}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    model = CMSNestedModel(base_model, adapter_configs).to(device=device, dtype=torch.bfloat16)
    optimizers, grad_buffers, step_counts = init_cms_training(model, device, torch.bfloat16)

    dreams = build_think_chain_dreams()
    ds = DreamDataset(dreams, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"  Dataset: {len(ds)} examples")

    for cfg in adapter_configs:
        logger.info(f"  {cfg['name']}: pos={cfg['position']}, chunk={cfg['chunk_size']}, lr={cfg['lr']}")

    cond_dir = OUT_DIR / name
    cond_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1 with checkpoints ===
    logger.info(f"\n--- Phase 1: {p1_steps} steps ---")
    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; ci = 0; p1_results = {}

    for ep in range(500):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            update_info = cms_step(model, loss, optimizers, grad_buffers, step_counts)
            step += 1
            if step % 20 == 0:
                logger.info(f"    [P1] step {step}: loss={loss.item():.4f}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                logger.info(f"\n  --- P1 Eval @ step {step} ---")
                mem, san, results = full_eval(model, tok, device, label=f"{name}-P1-s{step}")
                logger.info(f"  ★ {name} P1 step {step}: Mem={mem:.0%}, San={san:.0%}")

                ckpt_dir = cond_dir / f"p1_step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({n: a.state_dict() for n, a in model.adapters.items()}, ckpt_dir / "adapters.pt")
                json.dump({"phase": "P1", "step": step, "loss": loss.item(),
                           "memory": mem, "sanity": san, "details": results},
                          open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                p1_results[step] = {"memory": mem, "sanity": san}
                ci += 1; model.train()
            if step >= p1_steps:
                break
        if step >= p1_steps:
            break
    logger.info(f"  [P1] Done: {step} steps, final loss={loss.item():.4f}")

    # === Phase 2 ===
    logger.info(f"\n--- Phase 2: Retrieval SFT ({p2_steps} steps) ---")
    retrieval_data = [[{"role": "user", "content": d["q"]},
                       {"role": "assistant", "content": d["a"]}] for d in RETRIEVAL_SFT]
    ds2 = DreamDataset(retrieval_data, tok, max_length=MAX_SEQ_LEN)
    model.train()
    dl2 = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=True)
    step2 = 0

    for ep in range(200):
        for batch in dl2:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            for cfg in adapter_configs:
                optimizers[cfg["name"]].step()
            model.zero_grad()
            step2 += 1
            if step2 % 10 == 0:
                logger.info(f"    [P2] step {step2}: loss={loss.item():.4f}")
            if step2 >= p2_steps:
                break
        if step2 >= p2_steps:
            break
    logger.info(f"  [P2] Done: {step2} steps, final loss={loss.item():.4f}")

    logger.info(f"\n  Eval after Phase 1+2:")
    mem2, san2, res2 = full_eval(model, tok, device, label=f"{name}-P1P2")
    logger.info(f"  ★ {name} P1+P2: Mem={mem2:.0%}, San={san2:.0%}")

    torch.save({n: a.state_dict() for n, a in model.adapters.items()}, cond_dir / "p1p2_adapters.pt")
    json.dump({"phase": "P1+P2", "memory": mem2, "sanity": san2, "details": res2},
              open(cond_dir / "p1p2_eval.json", "w"), ensure_ascii=False, indent=2)

    del model, base_model
    torch.cuda.empty_cache()

    return {"p1": p1_results, "p1p2": {"memory": mem2, "sanity": san2}}


def main():
    logger.info(f"Exp 26b: More steps, novel domain, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    conditions = {
        # Best from Exp 26 — CMS 1/10
        "cms_1_10_lr5e4": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 10, "lr": 5e-4},
        ],
        # Higher lr
        "cms_1_10_lr1e3": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 1e-3},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 10, "lr": 1e-3},
        ],
        # Uniform baseline with more steps
        "uniform_lr5e4": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
        ],
        # Uniform higher lr
        "uniform_lr1e3": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 1e-3},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 1e-3},
        ],
    }

    all_results = {}
    for name, configs in conditions.items():
        all_results[name] = run_with_checkpoints(name, configs, p1_steps=200, p2_steps=40)

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, res in all_results.items():
        logger.info(f"\n{name}:")
        for step, r in sorted(res["p1"].items()):
            logger.info(f"  P1 step {step}: Mem={r['memory']:.0%}, San={r['sanity']:.0%}")
        p2 = res["p1p2"]
        logger.info(f"  P1+P2: Mem={p2['memory']:.0%}, San={p2['sanity']:.0%}")

    json.dump(all_results, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
