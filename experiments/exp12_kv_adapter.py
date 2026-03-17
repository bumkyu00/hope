"""Exp 12: Key-Value Memory Adapter — selective activation via learned keys.

Instead of MLP that transforms ALL inputs, use a key-value memory
that only activates when input matches stored keys.

This is the backprop equivalent of delta rule's associative memory.
"""

import json, logging, re, sys, math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from exp5_data import build_train_data, build_test_data
from exp9_nested import NestedModel, CONTAMINATION

MODEL = "Qwen/Qwen3.5-0.8B"
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
OUT_DIR = Path("experiments/exp12_results")
CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300]

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


class KVMemoryAdapter(nn.Module):
    """Key-Value associative memory adapter.

    Stores N memory slots, each with a key and value.
    On forward pass, input is projected to key space, matched against stored keys,
    and matching values are retrieved and added to the hidden state.

    This provides SELECTIVE activation — only relevant memories fire.
    """
    def __init__(self, hidden_size, num_slots=32, key_size=64):
        super().__init__()
        self.num_slots = num_slots
        self.key_size = key_size
        self.hidden_size = hidden_size

        # Learned memory slots
        self.keys = nn.Parameter(torch.randn(num_slots, key_size) * 0.01)
        self.values = nn.Parameter(torch.zeros(num_slots, hidden_size))

        # Project hidden state to key space
        self.key_proj = nn.Linear(hidden_size, key_size, bias=False)

        # Scale factor for value contribution
        self.scale = nn.Parameter(torch.tensor(0.01))  # Start small

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        q = self.key_proj(x)  # (batch, seq, key_size)

        # Similarity with stored keys
        # keys: (num_slots, key_size)
        sim = torch.matmul(q, self.keys.T) / math.sqrt(self.key_size)  # (batch, seq, num_slots)

        # Sparse attention — use top-k to keep it selective
        topk = min(4, self.num_slots)
        topk_vals, topk_idx = sim.topk(topk, dim=-1)  # (batch, seq, topk)
        weights = F.softmax(topk_vals, dim=-1)  # (batch, seq, topk)

        # Gather corresponding values
        # values: (num_slots, hidden_size)
        selected_values = self.values[topk_idx]  # (batch, seq, topk, hidden_size)
        retrieved = (weights.unsqueeze(-1) * selected_values).sum(dim=-2)  # (batch, seq, hidden)

        return x + self.scale * retrieved


class KVNestedModel(nn.Module):
    """Wraps a pretrained model with KV Memory Adapters."""

    def __init__(self, base_model, adapter_positions, num_slots=32, key_size=64):
        super().__init__()
        self.base_model = base_model
        self.adapter_positions = adapter_positions

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size
        self.adapters = nn.ModuleDict()
        for pos in adapter_positions:
            self.adapters[str(pos)] = KVMemoryAdapter(hidden_size, num_slots, key_size)

        trainable = sum(p.numel() for p in self.adapters.parameters())
        total = sum(p.numel() for p in base_model.parameters())
        logger.info(f"KV Adapters: {len(adapter_positions)} x {num_slots} slots = {trainable:,} trainable ({100*trainable/total:.3f}%)")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hooks = []
        for pos in self.adapter_positions:
            adapter = self.adapters[str(pos)]
            layer = self.base_model.model.layers[pos]
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
            for h in hooks: h.remove()

    def generate(self, *args, **kwargs):
        hooks = []
        for pos in self.adapter_positions:
            adapter = self.adapters[str(pos)]
            layer = self.base_model.model.layers[pos]
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
            for h in hooks: h.remove()


PASSTHROUGH_DATA = [
    {"q": "안녕하세요.", "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n인사. 무관.\n</think>\n안녕하세요! 뭘 도와드릴까요?"},
    {"q": "1+1은?", "a": "<think>\n사용자 기억 확인. 수학 질문. 무관.\n</think>\n2입니다."},
    {"q": "고양이에 대한 시 써줘.", "a": "<think>\n창작 요청. 기억 무관.\n</think>\n창가에 앉은 고양이\n햇살을 베개 삼아\n하루를 보낸다."},
    {"q": "대한민국의 수도는?", "a": "<think>\n일반 지식. 기억 무관.\n</think>\n서울입니다."},
    {"q": "Git이 뭐야?", "a": "<think>\n기술 설명. 기억 무관.\n</think>\n분산 버전 관리 시스템입니다."},
    {"q": "잠이 안 올 때 어떻게 해?", "a": "<think>\n일상 조언. 기억 무관.\n</think>\n따뜻한 차를 마시거나 스트레칭 해보세요."},
    {"q": "Thank you를 한국어로?", "a": "<think>\n번역. 기억 무관.\n</think>\n감사합니다."},
    {"q": "오늘 기분이 좋아!", "a": "<think>\n일상 대화. 기억 무관.\n</think>\n좋은 하루 보내세요!"},
]

SANITY_QS = [
    {"q": "안녕하세요.", "expect": ["안녕", "반갑", "반가", "도와"]},
    {"q": "1+1은?", "expect": ["2"]},
    {"q": "파이썬으로 hello world 짜줘.", "expect": ["print", "hello"]},
    {"q": "대한민국의 수도는?", "expect": ["서울"]},
    {"q": "오늘 날씨가 좋다. 뭐 하면 좋을까?", "expect": ["산책", "공원", "야외", "운동"]},
]


def quick_eval(model, tok, tests, device):
    model.eval()
    mem_results = []
    for t in tests:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        hits = [kw for kw in t["should_remember"] if kw.lower() in resp.lower()]
        mem_results.append({"q": t["q"], "category": t["category"],
            "should_remember": t["should_remember"], "response": resp[:300],
            "hits": hits, "activated": len(hits) > 0})

    san_results = []
    for sq in SANITY_QS:
        msgs = [{"role": "user", "content": sq["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        has_exp = any(kw.lower() in resp.lower() for kw in sq["expect"])
        contam = any(kw in resp.lower() for kw in CONTAMINATION)
        ok = has_exp and not contam and len(resp) > 3
        san_results.append({"q": sq["q"], "response": resp[:300], "ok": ok, "contaminated": contam})

    mem_acc = sum(r["activated"] for r in mem_results) / len(mem_results)
    san_acc = sum(r["ok"] for r in san_results) / len(san_results)
    return mem_acc, san_acc, mem_results, san_results


def main():
    logger.info(f"Exp 12: KV Memory Adapter, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    # Compare: MLP adapter vs KV adapter
    configs = {
        "MLP_adapter": {"type": "mlp", "positions": [7, 15], "size": 64, "lr": 1e-3},
        "KV_adapter_32slots": {"type": "kv", "positions": [7, 15], "slots": 32, "key_size": 64, "lr": 1e-3},
        "KV_adapter_64slots": {"type": "kv", "positions": [7, 15], "slots": 64, "key_size": 64, "lr": 1e-3},
    }

    memory_data = build_train_data(5)
    all_data = memory_data + PASSTHROUGH_DATA
    tests = build_test_data()

    all_results = {}

    for name, cfg in configs.items():
        logger.info(f"\n{'='*50} {name} {'='*50}")

        # Reload model fresh each time
        base_model2, tok2 = load_model_and_tokenizer(mc)
        tok2.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        base_model2.resize_token_embeddings(len(tok2))
        device = next(base_model2.parameters()).device

        if cfg["type"] == "mlp":
            from exp9_nested import AdapterMLP, NestedModel
            model = NestedModel(base_model2, cfg["positions"], cfg["size"]).to(device=device, dtype=torch.bfloat16)
            opt_params = model.adapters.parameters()
        else:
            model = KVNestedModel(base_model2, cfg["positions"], cfg["slots"], cfg["key_size"]).to(device=device, dtype=torch.bfloat16)
            opt_params = model.adapters.parameters()

        msgs = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in all_data]
        ds = DreamDataset(msgs, tok2, max_length=MAX_SEQ_LEN)

        model.train()
        optimizer = torch.optim.AdamW(opt_params, lr=cfg["lr"])
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        step = 0; ci = 0; results = []

        for ep in range(200):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss; loss.backward(); optimizer.step(); optimizer.zero_grad(); step += 1

                if step % 20 == 0:
                    logger.info(f"  step {step}: loss={loss.item():.4f}")

                if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                    mem, san, mem_res, san_res = quick_eval(model, tok2, tests, device)
                    logger.info(f"  ★ Step {step}: Mem={mem:.0%}, San={san:.0%}")

                    # Save checkpoint
                    ckpt_dir = OUT_DIR / name / f"step{step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.adapters.state_dict(), ckpt_dir / "adapters.pt")
                    json.dump({"step": step, "loss": loss.item(), "memory": mem, "sanity": san,
                               "memory_details": mem_res, "sanity_details": san_res},
                              open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)

                    results.append({"step": step, "memory": mem, "sanity": san, "loss": loss.item()})
                    ci += 1; model.train()
            if ci >= len(CHECKPOINTS): break

        all_results[name] = results
        del model, base_model2; torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*60}\nComparison\n{'='*60}")
    for name, results in all_results.items():
        logger.info(f"\n{name}:")
        for r in results:
            logger.info(f"  Step {r['step']}: Mem={r['memory']:.0%} San={r['sanity']:.0%}")

    json.dump(all_results, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
