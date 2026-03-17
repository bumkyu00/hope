"""Exp 6: CMS vs Uniform SFT — memory/sanity trade-off.

Same data (5 dreams/fact), same total steps.
Compare: does CMS preserve sanity while encoding memory?
"""

import json, logging, re, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig, LoRAConfig, LayerGroupConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora, get_layer_group_params
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml
from dreamlora.training.optimizer_groups import create_group_optimizers
from exp5_data import build_train_data, build_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

MODEL = "Qwen/Qwen3.5-0.8B"  # 24 layers
MAX_SEQ_LEN = 512
BATCH_SIZE = 2

SANITY_QS = [
    "안녕하세요. 자기소개 해줘.",
    "파이썬으로 hello world 짜줘.",
    "오늘 날씨가 좋다. 뭐 하면 좋을까?",
    "1+1은?",
]

CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300]


def evaluate(model, tok, tests, device):
    model.eval()
    # Memory test
    correct = 0
    for test in tests:
        msgs = [{"role": "user", "content": test["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        hits = [kw for kw in test["should_remember"] if kw.lower() in response.lower()]
        if hits: correct += 1
    mem_acc = correct / len(tests)

    # Sanity test
    sanity_ok = 0
    for q in SANITY_QS:
        msgs = [{"role": "user", "content": q}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        contaminated = any(w in response.lower() for w in ["밀가루", "글루텐", "알레르기", "공포증", "haskell", "물 공포"])
        if len(response) > 10 and not contaminated:
            sanity_ok += 1
    san_acc = sanity_ok / len(SANITY_QS)
    return mem_acc, san_acc


def run_uniform(model, tok, ds, tests, device):
    """Standard uniform SFT — all layers same lr."""
    logger.info("\n=== UNIFORM SFT (lr=2e-5 all layers) ===")
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    step = 0
    check_idx = 0
    results = []
    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if check_idx < len(CHECKPOINTS) and step == CHECKPOINTS[check_idx]:
                mem, san = evaluate(model, tok, tests, device)
                logger.info(f"  Step {step}: loss={loss.item():.4f} Memory={mem:.0%} Sanity={san:.0%}")
                results.append({"step": step, "loss": loss.item(), "memory": mem, "sanity": san})
                check_idx += 1
                model.train()
        if check_idx >= len(CHECKPOINTS): break
    return results


def run_cms(model, tok, ds, tests, device):
    """CMS — high-freq layers fast, low-freq layers slow."""
    logger.info("\n=== CMS (high=4e-5, mid=1e-5, low=2e-6) ===")

    # 0.8B has 24 layers
    groups = [
        LayerGroupConfig(name="high", layer_start=0, layer_end=7, learning_rate=4e-5, chunk_size=1),
        LayerGroupConfig(name="mid", layer_start=8, layer_end=15, learning_rate=1e-5, chunk_size=1),
        LayerGroupConfig(name="low", layer_start=16, layer_end=23, learning_rate=2e-6, chunk_size=1),
    ]
    group_params = get_layer_group_params(model, groups)
    optimizers = create_group_optimizers(group_params, groups, weight_decay=0.01)

    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    step = 0
    check_idx = 0
    results = []
    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            for g_name, opt in optimizers.items():
                opt.step()
            model.zero_grad()
            step += 1
            if check_idx < len(CHECKPOINTS) and step == CHECKPOINTS[check_idx]:
                mem, san = evaluate(model, tok, tests, device)
                logger.info(f"  Step {step}: loss={loss.item():.4f} Memory={mem:.0%} Sanity={san:.0%}")
                results.append({"step": step, "loss": loss.item(), "memory": mem, "sanity": san})
                check_idx += 1
                model.train()
        if check_idx >= len(CHECKPOINTS): break
    return results


def main():
    tests = build_test_data()
    train_data = build_train_data(5)
    out_dir = Path("experiments/exp6_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run Uniform
    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lc = LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "up_proj", "down_proj"])
    model, tok = load_model_and_tokenizer(mc)
    model = setup_lora(model, lc)
    device = next(model.parameters()).device

    msgs_list = [[{"role": "user", "content": d["q"]},
                   {"role": "assistant", "content": d["a"]}] for d in train_data]
    ds = DreamDataset(msgs_list, tok, max_length=MAX_SEQ_LEN)

    uniform_results = run_uniform(model, tok, ds, tests, device)
    del model; torch.cuda.empty_cache()

    # Run CMS
    model2, tok2 = load_model_and_tokenizer(mc)
    model2 = setup_lora(model2, lc)
    ds2 = DreamDataset(msgs_list, tok2, max_length=MAX_SEQ_LEN)
    cms_results = run_cms(model2, tok2, ds2, tests, device)
    del model2; torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*60}\nComparison: Uniform vs CMS\n{'='*60}")
    logger.info(f"{'Step':>5} | {'U-Mem':>6} {'U-San':>6} | {'C-Mem':>6} {'C-San':>6}")
    logger.info("-" * 45)
    for u, c in zip(uniform_results, cms_results):
        logger.info(f"{u['step']:>5} | {u['memory']:>5.0%} {u['sanity']:>5.0%} | {c['memory']:>5.0%} {c['sanity']:>5.0%}")

    json.dump({"uniform": uniform_results, "cms": cms_results},
              open(out_dir / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
