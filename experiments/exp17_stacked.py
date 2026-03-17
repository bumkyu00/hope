"""Exp 17: Stacked adapters for temporal memory.

Phase 1: Train adapter pair A (Rust knowledge)
Phase 2: Freeze A, add adapter pair B on top (Go knowledge)

The latest adapter is applied AFTER the older one,
so newer information naturally overrides older.
This mimics CMS high-freq → low-freq layering.
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
from exp9_nested import AdapterMLP
from exp16_temporal import (PHASE1_DREAMS, PHASE2_DREAMS, CURRENT_TESTS,
                             HISTORY_TESTS, SANITY_QS, evaluate_all)

MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_SIZE = 64
LR = 1e-3
BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
OUT_DIR = Path("experiments/exp17_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


class StackableNestedModel(nn.Module):
    """Model with stackable adapter layers. New adapters can be added without touching old ones."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.adapter_stacks = {}  # position -> list of adapters (ordered old→new)

        for param in base_model.parameters():
            param.requires_grad = False

    def add_adapter_pair(self, positions, adapter_size, name, dtype=torch.bfloat16):
        """Add a new adapter pair. Returns the new adapters for training."""
        hidden_size = self.base_model.config.hidden_size
        new_adapters = nn.ModuleDict()

        for pos in positions:
            adapter = AdapterMLP(hidden_size, adapter_size).to(
                device=next(self.base_model.parameters()).device, dtype=dtype)
            new_adapters[str(pos)] = adapter

            if pos not in self.adapter_stacks:
                self.adapter_stacks[pos] = []
            self.adapter_stacks[pos].append(adapter)

        # Register as submodule
        setattr(self, f"adapters_{name}", new_adapters)

        trainable = sum(p.numel() for p in new_adapters.parameters())
        logger.info(f"Added adapter '{name}': {len(positions)} positions x {adapter_size}d = {trainable:,} params")
        return new_adapters

    def freeze_adapters(self, name):
        """Freeze a named adapter set."""
        adapters = getattr(self, f"adapters_{name}")
        for param in adapters.parameters():
            param.requires_grad = False
        logger.info(f"Frozen adapter '{name}'")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hooks = []
        for pos, adapter_list in self.adapter_stacks.items():
            layer = self.base_model.model.layers[pos]

            def make_hook(adapters):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Apply all adapters in order (old → new)
                    for adapter in adapters:
                        hidden = adapter(hidden)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            hooks.append(layer.register_forward_hook(make_hook(adapter_list)))

        try:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            for h in hooks:
                h.remove()

    def generate(self, *args, **kwargs):
        hooks = []
        for pos, adapter_list in self.adapter_stacks.items():
            layer = self.base_model.model.layers[pos]

            def make_hook(adapters):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    for adapter in adapters:
                        hidden = adapter(hidden)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            hooks.append(layer.register_forward_hook(make_hook(adapter_list)))

        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()


def train_phase(model, adapters, tok, dreams, device, steps, label):
    msgs = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in dreams]
    ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)
    opt = torch.optim.AdamW(adapters.parameters(), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    step = 0
    for ep in range(100):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss; loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                logger.info(f"  [{label}] step {step}: loss={loss.item():.4f}")
            if step >= steps: break
        if step >= steps: break


def main():
    logger.info(f"Exp 17: Stacked adapters, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    model = StackableNestedModel(base_model)

    positions = [9, 21]

    # Phase 1: Add and train Rust adapters
    logger.info("\n=== Phase 1: Rust project ===")
    rust_adapters = model.add_adapter_pair(positions, ADAPTER_SIZE, "rust")
    train_phase(model, rust_adapters, tok, PHASE1_DREAMS, device, steps=30, label="Rust")

    p1_curr, p1_hist, p1_san, p1_res = evaluate_all(model, tok, device)
    logger.info(f"After Phase 1: Sanity={p1_san:.0%}")

    ckpt1 = OUT_DIR / "phase1"
    ckpt1.mkdir(parents=True, exist_ok=True)
    torch.save(rust_adapters.state_dict(), ckpt1 / "rust_adapters.pt")
    json.dump({"sanity": p1_san, "details": p1_res}, open(ckpt1 / "eval.json", "w"), ensure_ascii=False, indent=2)

    # Phase 2: Freeze Rust, add Go adapters on TOP
    logger.info("\n=== Phase 2: Freeze Rust, add Go ===")
    model.freeze_adapters("rust")
    go_adapters = model.add_adapter_pair(positions, ADAPTER_SIZE, "go")
    train_phase(model, go_adapters, tok, PHASE2_DREAMS, device, steps=30, label="Go")

    p2_curr, p2_hist, p2_san, p2_res = evaluate_all(model, tok, device)
    logger.info(f"After Phase 2: Current(Go)={p2_curr:.0%}, History(Rust)={p2_hist:.0%}, Sanity={p2_san:.0%}")

    ckpt2 = OUT_DIR / "phase2"
    ckpt2.mkdir(parents=True, exist_ok=True)
    torch.save(go_adapters.state_dict(), ckpt2 / "go_adapters.pt")
    json.dump({"current_go": p2_curr, "history_rust": p2_hist, "sanity": p2_san, "details": p2_res},
              open(ckpt2 / "eval.json", "w"), ensure_ascii=False, indent=2)

    # Summary
    summary = {
        "phase1": {"sanity": p1_san},
        "phase2": {"current_go": p2_curr, "history_rust": p2_hist, "sanity": p2_san},
        "comparison_with_exp16": "Exp 16 (single adapter): Current=0%, History=100%. Exp 17 (stacked): see above."
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Exp 16 (single): Current Go=0%, History Rust=100%")
    logger.info(f"Exp 17 (stacked): Current Go={p2_curr:.0%}, History Rust={p2_hist:.0%}")
    logger.info(f"Sanity: {p2_san:.0%}")
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
