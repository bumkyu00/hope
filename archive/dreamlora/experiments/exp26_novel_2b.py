"""Exp 26: Novel domain on 2B — push memory performance with CMS variations.

Domain: Fictional novel "달의 정원" — impossible to guess from general knowledge.
Model: Qwen3.5-2B (proven capable in Exp 25)

Conditions:
  A: Uniform (both adapters every step) — baseline
  B: CMS chunk 1/5 (adapter A every step, adapter B every 5 steps)
  C: CMS chunk 1/10
  D: CMS lr-diff (adapter A lr=5e-4, adapter B lr=1e-4)
  E: 3-adapter (L5/L11/L17, uniform)

Each condition: Phase 1 (think chain dreams) → eval → Phase 2 (retrieval SFT) → eval

Saves: checkpoints, log, full I/O JSON for every test.
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

MODEL = "Qwen/Qwen3.5-2B"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
OUT_DIR = Path("experiments/exp26_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


# ========================================
# CMS Nested Model — supports per-adapter chunk_size and lr
# ========================================

class CMSNestedModel(nn.Module):
    """Nested model with CMS-style per-adapter update schedules."""

    def __init__(self, base_model, adapter_configs):
        """
        adapter_configs: list of dicts with keys:
            name, position, adapter_size, chunk_size, lr
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.adapter_configs = adapter_configs

        for param in base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size
        self.adapters = nn.ModuleDict()

        for cfg in adapter_configs:
            self.adapters[cfg["name"]] = AdapterMLP(hidden_size, cfg["adapter_size"])

        trainable = sum(p.numel() for p in self.adapters.parameters())
        total = sum(p.numel() for p in base_model.parameters())
        logger.info(f"CMS Adapters: {len(adapter_configs)} adapters, {trainable:,} params ({100*trainable/total:.3f}%)")

    def _make_hooks(self):
        hooks = []
        for cfg in self.adapter_configs:
            adapter = self.adapters[cfg["name"]]
            layer = self.base_model.model.layers[cfg["position"]]

            def make_hook(a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        return (a(output[0]),) + output[1:]
                    return a(output)
                return hook_fn

            hooks.append(layer.register_forward_hook(make_hook(adapter)))
        return hooks

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hooks = self._make_hooks()
        try:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            for h in hooks:
                h.remove()

    def generate(self, *args, **kwargs):
        hooks = self._make_hooks()
        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()


def init_cms_training(model, device, dtype):
    """Initialize per-adapter optimizers and gradient buffers."""
    optimizers = {}
    grad_buffers = {}
    step_counts = {}

    for cfg in model.adapter_configs:
        name = cfg["name"]
        adapter = model.adapters[name].to(device=device, dtype=dtype)
        optimizers[name] = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"])
        grad_buffers[name] = {
            n: torch.zeros_like(p) for n, p in adapter.named_parameters()
        }
        step_counts[name] = 0

    return optimizers, grad_buffers, step_counts


def cms_step(model, loss, optimizers, grad_buffers, step_counts):
    """CMS update: accumulate gradients, step at chunk boundaries."""
    loss.backward()

    update_info = {}
    for cfg in model.adapter_configs:
        name = cfg["name"]
        chunk = cfg["chunk_size"]
        adapter = model.adapters[name]

        for n, p in adapter.named_parameters():
            if p.grad is not None:
                grad_buffers[name][n].add_(p.grad)

        step_counts[name] += 1

        if step_counts[name] % chunk == 0:
            for n, p in adapter.named_parameters():
                p.grad = grad_buffers[name][n] / chunk
            optimizers[name].step()
            for n in grad_buffers[name]:
                grad_buffers[name][n].zero_()
            update_info[name] = True
        else:
            update_info[name] = False

    model.zero_grad()
    return update_info


# ========================================
# Evaluation
# ========================================

def full_eval(model, tok, device, label=""):
    """Evaluate with thinking enabled, save full I/O."""
    model.eval()
    results = []

    for t in NOVEL_TESTS:
        msgs = [{"role": "user", "content": t["q"]}]

        # Generate with thinking
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=500,
                                 do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        # Parse think block
        think_text = ""
        if "</think>" in raw:
            think_text = raw.split("</think>")[0].replace("<think>", "").strip()
            clean = raw.split("</think>")[1].strip().split("<|im_end|>")[0].strip()
        else:
            clean = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        # Keyword check on clean response
        hits = [kw for kw in t["should"] if kw.lower() in clean.lower()]
        think_hits = [kw for kw in t["should"] if kw.lower() in think_text.lower()]
        activated = len(hits) > 0 or len(think_hits) > 0

        # Contamination check for sanity questions
        contaminated = False
        if t.get("sanity"):
            proj_hits = [kw for kw in PROJECT_KEYWORDS if kw.lower() in clean.lower()]
            contaminated = len(proj_hits) > 0

        results.append({
            "q": t["q"], "desc": t["desc"], "is_sanity": t.get("sanity", False),
            "should": t["should"],
            "think_raw": think_text[:800],
            "response_clean": clean[:800],
            "hits": hits, "think_hits": think_hits,
            "activated": activated, "contaminated": contaminated,
        })

        s = "✓" if activated else "✗"
        c = " ⚠️CONTAM" if contaminated else ""
        logger.info(f"  {s}{c} [{label}] {t['desc']}")
        if think_text:
            logger.info(f"    💭: {think_text[:150]}")
        logger.info(f"    →: {clean[:150]}")

    mem = [r for r in results if not r["is_sanity"]]
    san = [r for r in results if r["is_sanity"]]
    mem_acc = sum(r["activated"] for r in mem) / max(len(mem), 1)
    san_clean = sum(not r["contaminated"] for r in san) / max(len(san), 1)
    return mem_acc, san_clean, results


# ========================================
# Training functions
# ========================================

def train_phase1(model, tok, dreams, device, optimizers, grad_buffers, step_counts, steps=75):
    """Phase 1: Think chain dream training with CMS."""
    ds = DreamDataset(dreams, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"  [P1] Dataset: {len(ds)} examples, {steps} steps")

    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0

    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            update_info = cms_step(model, loss, optimizers, grad_buffers, step_counts)
            step += 1
            if step % 10 == 0:
                updates = [k for k, v in update_info.items() if v]
                logger.info(f"    [P1] step {step}: loss={loss.item():.4f}, updates={updates or 'none'}")
            if step >= steps:
                break
        if step >= steps:
            break
    logger.info(f"  [P1] Done: {step} steps, final loss={loss.item():.4f}")


def train_phase2(model, tok, device, optimizers, grad_buffers, step_counts, steps=30):
    """Phase 2: Retrieval SFT (teach model to recall via thinking)."""
    retrieval_data = [[{"role": "user", "content": d["q"]},
                       {"role": "assistant", "content": d["a"]}] for d in RETRIEVAL_SFT]
    ds = DreamDataset(retrieval_data, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"  [P2] Retrieval SFT: {len(ds)} examples, {steps} steps")

    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0

    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            # Phase 2 uses uniform update (all adapters every step)
            loss.backward()
            for cfg in model.adapter_configs:
                name = cfg["name"]
                optimizers[name].step()
            model.zero_grad()
            step += 1
            if step % 10 == 0:
                logger.info(f"    [P2] step {step}: loss={loss.item():.4f}")
            if step >= steps:
                break
        if step >= steps:
            break
    logger.info(f"  [P2] Done: {step} steps, final loss={loss.item():.4f}")


# ========================================
# Run one condition
# ========================================

def run_condition(name, adapter_configs, p1_steps=75, p2_steps=30):
    """Run one experimental condition: Phase 1 → eval → Phase 2 → eval."""
    logger.info(f"\n{'='*60}\n{name}\n{'='*60}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    model = CMSNestedModel(base_model, adapter_configs).to(device=device, dtype=torch.bfloat16)
    optimizers, grad_buffers, step_counts = init_cms_training(model, device, torch.bfloat16)

    dreams = build_think_chain_dreams()
    cond_dir = OUT_DIR / name
    cond_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1 ===
    logger.info(f"\n--- {name}: Phase 1 (think chain dreams) ---")
    for cfg in adapter_configs:
        logger.info(f"  {cfg['name']}: pos={cfg['position']}, sz={cfg['adapter_size']}, chunk={cfg['chunk_size']}, lr={cfg['lr']}")

    train_phase1(model, tok, dreams, device, optimizers, grad_buffers, step_counts, steps=p1_steps)

    logger.info(f"\n  Eval after Phase 1:")
    mem1, san1, res1 = full_eval(model, tok, device, label=f"{name}-P1")
    logger.info(f"  ★ {name} P1: Mem={mem1:.0%}, San={san1:.0%}")

    torch.save({n: a.state_dict() for n, a in model.adapters.items()}, cond_dir / "p1_adapters.pt")
    json.dump({"phase": "P1", "memory": mem1, "sanity": san1, "details": res1},
              open(cond_dir / "p1_eval.json", "w"), ensure_ascii=False, indent=2)

    # === Phase 2 ===
    logger.info(f"\n--- {name}: Phase 2 (retrieval SFT) ---")
    # Reset step counts for phase 2
    for k in step_counts:
        step_counts[k] = 0

    train_phase2(model, tok, device, optimizers, grad_buffers, step_counts, steps=p2_steps)

    logger.info(f"\n  Eval after Phase 1+2:")
    mem2, san2, res2 = full_eval(model, tok, device, label=f"{name}-P1P2")
    logger.info(f"  ★ {name} P1+P2: Mem={mem2:.0%}, San={san2:.0%}")

    torch.save({n: a.state_dict() for n, a in model.adapters.items()}, cond_dir / "p1p2_adapters.pt")
    json.dump({"phase": "P1+P2", "memory": mem2, "sanity": san2, "details": res2},
              open(cond_dir / "p1p2_eval.json", "w"), ensure_ascii=False, indent=2)

    del model, base_model
    torch.cuda.empty_cache()

    return {"p1": {"memory": mem1, "sanity": san1},
            "p1p2": {"memory": mem2, "sanity": san2}}


# ========================================
# Main
# ========================================

def main():
    logger.info(f"Exp 26: Novel domain (달의 정원) on {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Domain: Fictional novel — impossible to guess from general knowledge")

    # 2B = 24 layers (same as 0.8B)
    # Adapter positions: L7, L15 (proven best)
    ADAPTER_SIZE = 64

    conditions = {
        # A: Uniform baseline
        "A_uniform": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
        ],
        # B: CMS chunk 1/5
        "B_cms_1_5": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 5, "lr": 5e-4},
        ],
        # C: CMS chunk 1/10
        "C_cms_1_10": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 10, "lr": 5e-4},
        ],
        # D: CMS lr-diff (same chunk, different lr)
        "D_lr_diff": [
            {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 1e-4},
        ],
        # E: 3 adapters uniform
        "E_3adapter": [
            {"name": "a_front", "position": 5, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_mid", "position": 11, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
            {"name": "a_back", "position": 17, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
        ],
    }

    all_results = {}
    for name, configs in conditions.items():
        all_results[name] = run_condition(name, configs)

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    logger.info(f"{'Condition':>15} | {'P1 Mem':>7} | {'P1 San':>7} | {'P1+P2 Mem':>9} | {'P1+P2 San':>9}")
    logger.info("-" * 60)
    for name, res in all_results.items():
        p1 = res["p1"]
        p2 = res["p1p2"]
        logger.info(f"{name:>15} | {p1['memory']:>6.0%} | {p1['sanity']:>6.0%} | {p2['memory']:>8.0%} | {p2['sanity']:>8.0%}")

    json.dump(all_results, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
