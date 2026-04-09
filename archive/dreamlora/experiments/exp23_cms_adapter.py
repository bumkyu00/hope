"""Exp 23: CMS-style frequency differentiation on nested adapters.

Tests whether different update frequencies for adapters create
different abstraction levels (high-freq = episodes, low-freq = patterns).

Adapter A (after L9): chunk_size=1 (every step) → high frequency
Adapter B (after L21): chunk_size=5 (every 5 steps) → low frequency

Uses raw conversation data from Exp 22 (FastAPI project).
Compares: uniform (both every step) vs CMS (different chunk sizes).

Saves: checkpoints, log file, full I/O with think blocks.
"""

import json, logging, re, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
from exp22_data import (
    build_raw_conversation_dreams, build_think_chain_dreams,
    NOVEL_TESTS, PROJECT_KEYWORDS,
)

MODEL = "Qwen/Qwen3.5-0.8B"  # Fast exploration first, scale up later
ADAPTER_SIZE = 64
BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
LR_HIGH = 1e-3   # High frequency adapter
LR_LOW = 1e-3    # Low frequency adapter (same lr, different chunk)
CHECKPOINTS = [10, 20, 30, 50, 75, 100]
OUT_DIR = Path("experiments/exp23_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


class CMSNestedModel(nn.Module):
    """Nested model with CMS-style chunk-based updates for adapters."""

    def __init__(self, base_model, adapter_configs):
        """
        adapter_configs: list of dicts with keys:
            position, adapter_size, chunk_size, lr, name
        """
        super().__init__()
        self.base_model = base_model
        self.adapter_configs = adapter_configs

        for param in base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size
        self.adapters = nn.ModuleDict()
        self.optimizers = {}
        self.grad_buffers = {}
        self.step_counts = defaultdict(int)

        for cfg in adapter_configs:
            name = cfg["name"]
            adapter = AdapterMLP(hidden_size, cfg["adapter_size"])
            self.adapters[name] = adapter

        trainable = sum(p.numel() for p in self.adapters.parameters())
        total = sum(p.numel() for p in base_model.parameters())
        logger.info(f"CMS Adapters: {len(adapter_configs)} adapters, {trainable:,} params ({100*trainable/total:.3f}%)")
        for cfg in adapter_configs:
            logger.info(f"  {cfg['name']}: position={cfg['position']}, chunk={cfg['chunk_size']}, lr={cfg['lr']}")

    def init_training(self, device, dtype):
        """Initialize optimizers and gradient buffers."""
        for cfg in self.adapter_configs:
            name = cfg["name"]
            adapter = self.adapters[name].to(device=device, dtype=dtype)
            self.optimizers[name] = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"])
            self.grad_buffers[name] = {
                n: torch.zeros_like(p) for n, p in adapter.named_parameters()
            }

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
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

        try:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            for h in hooks:
                h.remove()

    def generate(self, *args, **kwargs):
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

        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

    def cms_step(self, loss):
        """CMS-style update: accumulate gradients, step at chunk boundaries."""
        loss.backward()

        update_info = {}

        for cfg in self.adapter_configs:
            name = cfg["name"]
            chunk = cfg["chunk_size"]
            adapter = self.adapters[name]

            # Accumulate gradients
            for n, p in adapter.named_parameters():
                if p.grad is not None:
                    self.grad_buffers[name][n].add_(p.grad)

            self.step_counts[name] += 1

            # Check chunk boundary
            if self.step_counts[name] % chunk == 0:
                # Apply accumulated gradient (normalized by chunk_size)
                for n, p in adapter.named_parameters():
                    p.grad = self.grad_buffers[name][n] / chunk

                self.optimizers[name].step()

                # Reset buffer
                for n in self.grad_buffers[name]:
                    self.grad_buffers[name][n].zero_()

                update_info[name] = True
            else:
                update_info[name] = False

        # Clear all model gradients
        self.zero_grad()

        return update_info


def full_eval(model, tok, device):
    """Evaluate with batched generation for speed."""
    model.eval()

    # Build all prompts at once
    all_prompts = []
    for t in NOVEL_TESTS:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        all_prompts.append(prompt)

    # Batch tokenize with left padding for generation
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    batch_inputs = tok(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)

    # Batch generate
    with torch.no_grad():
        batch_outputs = model.generate(
            **batch_inputs, max_new_tokens=200,
            do_sample=False, repetition_penalty=1.3,
        )

    # Decode all at once
    results = []
    for i, t in enumerate(NOVEL_TESTS):
        prompt_len = batch_inputs["attention_mask"][i].sum().item()
        raw = tok.decode(batch_outputs[i][prompt_len:], skip_special_tokens=True).strip()
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw

        hits = [kw for kw in t["should"] if kw.lower() in clean.lower()]
        activated = len(hits) > 0

        contaminated = False
        if t.get("sanity"):
            proj_hits = [kw for kw in PROJECT_KEYWORDS if kw.lower() in clean.lower()]
            contaminated = len(proj_hits) > 0

        results.append({
            "q": t["q"], "desc": t["desc"], "is_sanity": t.get("sanity", False),
            "should": t["should"],
            "think_raw": "", "response_raw": raw[:500], "response_clean": clean[:500],
            "hits": hits, "activated": activated, "contaminated": contaminated,
        })

        s = "✓" if activated else "✗"
        c = " ⚠️CONTAM" if contaminated else ""
        logger.info(f"  {s}{c} {t['desc']} → {clean[:80]}")

    tok.padding_side = "right"  # Reset

    mem = [r for r in results if not r["is_sanity"]]
    san = [r for r in results if r["is_sanity"]]
    mem_acc = sum(r["activated"] for r in mem) / max(len(mem), 1)
    san_acc = sum(not r["contaminated"] and r["activated"] for r in san) / max(len(san), 1)
    return mem_acc, san_acc, results


def run_condition(name, adapter_configs, dreams, tok_ref):
    """Run one experimental condition."""
    logger.info(f"\n{'='*60}\n{name}\n{'='*60}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    model = CMSNestedModel(base_model, adapter_configs)
    model.init_training(device, torch.bfloat16)

    ds = DreamDataset(dreams, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"  Dataset: {len(ds)} examples")

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    step = 0; ci = 0; all_evals = {}

    for ep in range(100):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            update_info = model.cms_step(loss)
            step += 1

            if step % 10 == 0:
                updates = {k: v for k, v in update_info.items() if v}
                logger.info(f"  step {step}: loss={loss.item():.4f}, updates={updates or 'none'}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                logger.info(f"\n  --- Eval @ step {step} ---")
                mem, san, results = full_eval(model, tok, device)
                logger.info(f"  ★ Step {step}: Mem={mem:.0%}, San={san:.0%}")

                # Save
                ckpt_dir = OUT_DIR / name / f"step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({n: a.state_dict() for n, a in model.adapters.items()}, ckpt_dir / "adapters.pt")
                json.dump({"step": step, "loss": loss.item(), "memory": mem, "sanity": san,
                           "details": results}, open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                all_evals[step] = {"memory": mem, "sanity": san}
                ci += 1; model.train()
        if ci >= len(CHECKPOINTS): break

    del model, base_model; torch.cuda.empty_cache()
    return all_evals


def main():
    logger.info(f"Exp 23: CMS Nested Adapter, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    # Use think chain dreams (best from Exp 22)
    dreams = build_think_chain_dreams()

    # 0.8B = 24 layers → adapters at L7, L15
    conditions = {
        "uniform": [
            {"name": "adapter_A", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": LR_HIGH},
            {"name": "adapter_B", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": LR_LOW},
        ],
        "cms_1_5": [
            {"name": "adapter_A_high", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": LR_HIGH},
            {"name": "adapter_B_low", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 5, "lr": LR_LOW},
        ],
        "cms_1_10": [
            {"name": "adapter_A_high", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": LR_HIGH},
            {"name": "adapter_B_low", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 10, "lr": LR_LOW},
        ],
    }

    all_results = {}
    for name, configs in conditions.items():
        all_results[name] = run_condition(name, configs, dreams, None)

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    logger.info(f"{'Step':>5} | {'Uniform Mem/San':>15} | {'CMS 1/5 Mem/San':>15} | {'CMS 1/10 Mem/San':>16}")
    logger.info("-" * 60)
    for step in CHECKPOINTS:
        u = all_results.get("uniform", {}).get(step, {})
        c5 = all_results.get("cms_1_5", {}).get(step, {})
        c10 = all_results.get("cms_1_10", {}).get(step, {})
        logger.info(f"{step:>5} | {u.get('memory',0):>5.0%}/{u.get('sanity',0):>4.0%} | {c5.get('memory',0):>5.0%}/{c5.get('sanity',0):>4.0%} | {c10.get('memory',0):>5.0%}/{c10.get('sanity',0):>4.0%}")

    json.dump(all_results, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
