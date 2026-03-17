"""Exp 9: Nested Learning style — insert adapter MLPs between frozen layers.

Key difference from LoRA:
- ALL existing model parameters are FROZEN
- Small adapter MLPs are INSERTED between layer groups
- Only adapters are trained → zero interference with existing knowledge

Architecture (0.8B = 24 layers):
  [Layers 0-7] → [Adapter_A (trainable)] → [Layers 8-15] → [Adapter_B (trainable)] → [Layers 16-23] → output
"""

import json, logging, re, sys, copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from exp5_data import build_train_data, build_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

MODEL = "Qwen/Qwen3.5-0.8B"
BATCH_SIZE = 2
MAX_SEQ_LEN = 512


class AdapterMLP(nn.Module):
    """Small MLP adapter inserted between layer groups.

    residual + down_proj(activation(up_proj(x)))
    Initialized near-zero so initial output ≈ input (no disruption).
    """
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.up = nn.Linear(hidden_size, adapter_size, bias=False)
        self.act = nn.GELU()
        self.down = nn.Linear(adapter_size, hidden_size, bias=False)
        # Init near-zero so adapter starts as identity
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return x + self.down(self.act(self.up(x)))


class NestedModel(nn.Module):
    """Wraps a pretrained model with adapter MLPs between layer groups."""

    def __init__(self, base_model, adapter_positions, adapter_size=64):
        """
        adapter_positions: list of layer indices AFTER which to insert adapters.
            e.g. [7, 15] means adapters after layer 7 and after layer 15.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size
        self.adapters = nn.ModuleDict()
        self.adapter_positions = adapter_positions

        for pos in adapter_positions:
            self.adapters[str(pos)] = AdapterMLP(hidden_size, adapter_size)

        # Count params
        trainable = sum(p.numel() for p in self.adapters.parameters())
        total = sum(p.numel() for p in base_model.parameters())
        logger.info(f"Adapters: {len(adapter_positions)} x {adapter_size}d = {trainable:,} trainable / {total:,} total ({100*trainable/total:.3f}%)")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Use hooks instead of manual forward — much simpler and compatible
        hooks = []
        for pos in self.adapter_positions:
            adapter = self.adapters[str(pos)]
            layer = self.base_model.model.layers[pos]

            def make_hook(adapter_module):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        adapted = adapter_module(hidden)
                        return (adapted,) + output[1:]
                    else:
                        return adapter_module(output)
                return hook_fn

            h = layer.register_forward_hook(make_hook(adapter))
            hooks.append(h)

        try:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            for h in hooks:
                h.remove()

        return outputs

    def generate(self, *args, **kwargs):
        """For generation, temporarily hook adapters into the forward pass."""
        # Simple approach: use base model's generate with hooks
        hooks = []

        for pos in self.adapter_positions:
            adapter = self.adapters[str(pos)]
            layer = self.base_model.model.layers[pos]

            def make_hook(adapter_module):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        adapted = adapter_module(hidden)
                        return (adapted,) + output[1:]
                    else:
                        return adapter_module(output)
                return hook_fn

            h = layer.register_forward_hook(make_hook(adapter))
            hooks.append(h)

        try:
            output = self.base_model.generate(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        return output


CONTAMINATION = ['밀가루', '글루텐', '알레르기', '공포증', 'haskell', '물 공포', '함수형', '프루테리언', '이구아나', '초록이']
SANITY_QS = [
    {'q': '안녕하세요.', 'expect': ['안녕', '반갑', '반가']},
    {'q': '1+1은?', 'expect': ['2']},
    {'q': '파이썬으로 hello world 짜줘.', 'expect': ['print', 'hello']},
    {'q': '대한민국의 수도는?', 'expect': ['서울']},
    {'q': '오늘 날씨가 좋다. 뭐 하면 좋을까?', 'expect': ['산책', '공원', '야외', '운동']},
]
CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300]


def quick_eval(model, tok, tests, device):
    model.eval()
    mem_ok = 0
    for t in tests:
        msgs = [{'role': 'user', 'content': t['q']}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False, repetition_penalty=1.3)
        resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip() or resp
        if any(kw.lower() in resp.lower() for kw in t['should_remember']):
            mem_ok += 1

    san_ok = 0
    for sq in SANITY_QS:
        msgs = [{'role': 'user', 'content': sq['q']}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, repetition_penalty=1.3)
        resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip() or resp
        has_exp = any(kw.lower() in resp.lower() for kw in sq['expect'])
        contam = any(kw in resp.lower() for kw in CONTAMINATION)
        if has_exp and not contam:
            san_ok += 1

    return mem_ok / len(tests), san_ok / len(SANITY_QS)


def run_nested(adapter_positions, adapter_size, lr, label):
    logger.info(f"\n{'='*50} {label} {'='*50}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)

    # Add special tokens
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))

    device = next(base_model.parameters()).device

    # Create nested model
    nested = NestedModel(base_model, adapter_positions, adapter_size).to(device=device, dtype=torch.bfloat16)

    # Build dataset
    tests = build_test_data()
    train_data = build_train_data(5)
    msgs = [[{'role': 'user', 'content': d['q']}, {'role': 'assistant', 'content': d['a']}] for d in train_data]
    ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)

    # Train only adapters
    optimizer = torch.optim.AdamW(nested.adapters.parameters(), lr=lr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    step = 0; ci = 0; results = []
    nested.train()

    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = nested(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                mem, san = quick_eval(nested, tok, tests, device)
                logger.info(f"  Step {step}: loss={loss.item():.4f} Mem={mem:.0%} San={san:.0%}")
                results.append({'step': step, 'memory': mem, 'sanity': san, 'loss': loss.item()})
                ci += 1
                nested.train()
        if ci >= len(CHECKPOINTS):
            break

    del nested, base_model
    torch.cuda.empty_cache()
    return results


def main():
    out_dir = Path("experiments/exp9_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        # Adapters after layer 7 and 15 (between groups)
        "2 adapters (L7,L15) sz64": {
            "positions": [7, 15], "size": 64, "lr": 1e-3,
        },
        # More adapters
        "4 adapters (L5,L11,L17,L23) sz64": {
            "positions": [5, 11, 17, 23], "size": 64, "lr": 1e-3,
        },
        # Bigger adapters
        "2 adapters (L7,L15) sz256": {
            "positions": [7, 15], "size": 256, "lr": 5e-4,
        },
    }

    all_results = {}
    for label, cfg in configs.items():
        all_results[label] = run_nested(cfg["positions"], cfg["size"], cfg["lr"], label)

    logger.info(f"\n{'='*70}\nNested Learning Results\n{'='*70}")
    for label, results in all_results.items():
        logger.info(f"\n{label}:")
        for r in results:
            logger.info(f"  Step {r['step']}: Mem={r['memory']:.0%} San={r['sanity']:.0%}")

    json.dump(all_results, open(out_dir / "results.json", "w"), indent=2)
    logger.info("Done!")


if __name__ == "__main__":
    main()
