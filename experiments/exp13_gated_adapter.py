"""Exp 13: Gated Adapter — learned gate decides when to activate memory.

Combines MLP's learning ability with selective activation.
gate = sigmoid(W_gate @ hidden) → 0~1 per token
output = hidden + gate * adapter(hidden)

When gate ≈ 0: memory inactive, clean passthrough
When gate ≈ 1: memory active, applies learned transformation
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
OUT_DIR = Path("experiments/exp13_results")
CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300]

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


class GatedAdapterMLP(nn.Module):
    """MLP adapter with a learned gate for selective activation.

    gate(x) → 0~1: how relevant is this input to stored memories?
    adapter(x) → transformation: what memory to apply?
    output = x + gate(x) * adapter(x)
    """
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        # Gate network — determines relevance
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)  # Start mostly closed (sigmoid(-2)≈0.12)

        # Adapter MLP — stores and retrieves memories
        self.up = nn.Linear(hidden_size, adapter_size, bias=False)
        self.act = nn.GELU()
        self.down = nn.Linear(adapter_size, hidden_size, bias=False)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))  # (batch, seq, 1)
        adapter_out = self.down(self.act(self.up(x)))  # (batch, seq, hidden)
        return x + gate * adapter_out


class GatedNestedModel(nn.Module):
    """Pretrained model with Gated Adapter MLPs between layers."""

    def __init__(self, base_model, adapter_positions, adapter_size=64):
        super().__init__()
        self.base_model = base_model
        self.adapter_positions = adapter_positions

        for param in base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size
        self.adapters = nn.ModuleDict()
        for pos in adapter_positions:
            self.adapters[str(pos)] = GatedAdapterMLP(hidden_size, adapter_size)

        trainable = sum(p.numel() for p in self.adapters.parameters())
        total = sum(p.numel() for p in base_model.parameters())
        logger.info(f"Gated Adapters: {len(adapter_positions)} x {adapter_size}d = {trainable:,} trainable ({100*trainable/total:.3f}%)")

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

    def get_gate_stats(self, input_ids, device):
        """Get average gate activation for analysis."""
        self.eval()
        gate_values = {}
        hooks = []
        for pos in self.adapter_positions:
            adapter = self.adapters[str(pos)]
            layer = self.base_model.model.layers[pos]
            def make_hook(a, p):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    gate = torch.sigmoid(a.gate_proj(hidden))
                    gate_values[p] = gate.mean().item()
                    return output  # Don't modify, just observe
                return hook_fn
            hooks.append(layer.register_forward_hook(make_hook(adapter, pos)))
        with torch.no_grad():
            self.base_model(input_ids=input_ids)
        for h in hooks: h.remove()
        return gate_values


PASSTHROUGH_DATA = [
    {"q": "안녕하세요.", "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n인사. 무관.\n</think>\n안녕하세요! 뭘 도와드릴까요?"},
    {"q": "1+1은?", "a": "<think>\n수학 질문. 기억 무관.\n</think>\n2입니다."},
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
    logger.info(f"Exp 13: Gated Adapter, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)

    memory_data = build_train_data(5)
    all_data = memory_data + PASSTHROUGH_DATA
    tests = build_test_data()

    configs = {
        "MLP_adapter": {"type": "mlp", "size": 64, "lr": 1e-3},
        "Gated_adapter_sz64": {"type": "gated", "size": 64, "lr": 1e-3},
        "Gated_adapter_sz128": {"type": "gated", "size": 128, "lr": 5e-4},
    }

    all_results = {}

    for name, cfg in configs.items():
        logger.info(f"\n{'='*50} {name} {'='*50}")

        base_model, tok = load_model_and_tokenizer(mc)
        tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        base_model.resize_token_embeddings(len(tok))
        device = next(base_model.parameters()).device

        if cfg["type"] == "mlp":
            from exp9_nested import AdapterMLP, NestedModel
            model = NestedModel(base_model, [7, 15], cfg["size"]).to(device=device, dtype=torch.bfloat16)
        else:
            model = GatedNestedModel(base_model, [7, 15], cfg["size"]).to(device=device, dtype=torch.bfloat16)

        msgs = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in all_data]
        ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)

        model.train()
        optimizer = torch.optim.AdamW(model.adapters.parameters(), lr=cfg["lr"])
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
                    mem, san, mem_res, san_res = quick_eval(model, tok, tests, device)
                    logger.info(f"  ★ Step {step}: Mem={mem:.0%}, San={san:.0%}")

                    # Gate analysis for gated models
                    gate_info = ""
                    if cfg["type"] == "gated" and hasattr(model, "get_gate_stats"):
                        # Test gate on memory vs sanity inputs
                        mem_q = "지현한테 피자 사줘도 될까?"
                        san_q = "안녕하세요."
                        mem_inputs = tok(format_chatml([{"role":"user","content":mem_q}], tokenizer=tok, enable_thinking=False, add_generation_prompt=True), return_tensors="pt").to(device)
                        san_inputs = tok(format_chatml([{"role":"user","content":san_q}], tokenizer=tok, enable_thinking=False, add_generation_prompt=True), return_tensors="pt").to(device)
                        mem_gates = model.get_gate_stats(mem_inputs["input_ids"], device)
                        san_gates = model.get_gate_stats(san_inputs["input_ids"], device)
                        gate_info = f" | Gates: mem={list(mem_gates.values())}, san={list(san_gates.values())}"
                        logger.info(f"    Gate analysis: memory_q={mem_gates}, sanity_q={san_gates}")

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
        del model, base_model; torch.cuda.empty_cache()

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
